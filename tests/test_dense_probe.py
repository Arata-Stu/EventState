"""Server-only numerical/geometry tests (NumPy, Torch and official-target readers)."""
import numpy as np
import pytest
import torch

from event_state.dense.evaluation import DenseEvaluator, accumulate
from event_state.dense.head import DenseRegressionHead
from event_state.dense.metrics import RegressionSums
from event_state.dense.targets import disparity_to_depth, read_target


def test_depth_metrics_and_invalid_prediction_rejected():
    m = RegressionSums("depth")
    accumulate(m, np.array([[2., 4.]]), np.array([[1., 2.]]), np.ones((1, 2), bool))
    result = m.compute()
    assert result["abs_rel"] == 1
    assert result["rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["delta1"] == 0
    with pytest.raises(ValueError, match="predictions"):
        accumulate(m, np.array([[np.nan]]), np.array([[1.]]), np.ones((1, 1), bool))


def test_disparity_thresholds_are_strict():
    m = RegressionSums("disparity")
    accumulate(m, np.array([[2., 3., 4.]]), np.ones((1, 3)), np.ones((1, 3), bool))
    assert m.compute()["1pe_percent"] == pytest.approx(200 / 3)
    assert m.compute()["2pe_percent"] == pytest.approx(100 / 3)


def test_duration_groups_and_empty_support():
    e = DenseEvaluator("depth")
    e.update(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2), bool),
             low_activity=True, low_duration_us=600_000)
    assert e.compute()["low_500ms_plus"]["valid_pixels"] == 4
    assert e.compute()["other_frames"]["mae"] is None


def test_calibrated_depth_supports_disparity_offset():
    q = np.array([[1, 0, 0, -3], [0, 1, 0, -2], [0, 0, 0, 100], [0, 0, 2, 1.]])
    z = disparity_to_depth(np.array([[0, 2.]]), q)
    assert np.isnan(z[0, 0])
    assert z[0, 1] == pytest.approx(20)


def test_png_encoding_and_mask(tmp_path):
    from PIL import Image
    path = tmp_path / "disp.png"
    Image.fromarray(np.array([[0, 256, 512]], np.uint16)).save(path)
    value, valid = read_target(dict(target_path=str(path), encoding="dsec_disparity_uint16_x256"), "disparity")
    assert value.tolist() == [[0, 1, 2]]
    assert valid.tolist() == [[False, True, True]]


def test_head_positive_full_geometry_and_gradient():
    model = DenseRegressionHead(4, width=8)
    x = torch.randn(1, 4, 2, 3)
    y = model(x, (32, 48))
    assert y.shape == (1, 32, 48)
    assert bool((y > 0).all())
    y.mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_full_dsec_and_existing_m3ed_geometry():
    from event_state.dense.source import input_geometry
    from event_state.data.activity import scale_event_activation
    dsec = input_geometry(torch.ones(3, 480, 640), "dsec")
    m3ed = input_geometry(torch.ones(3, 360, 640), "m3ed")
    assert dsec.shape == (3, 480, 640)
    assert m3ed.shape == (3, 352, 640)
    for value in (dsec, m3ed):
        image = (value * 255).byte().permute(1, 2, 0).numpy()
        mask = scale_event_activation(image, height=value.shape[1], width=value.shape[2])
        assert mask.shape == (value.shape[1] // 16, value.shape[2] // 16)
        assert not mask.any()


def test_legacy_m3ed_source_without_success_marker(tmp_path):
    import json
    from omegaconf import OmegaConf
    from event_state.dense.source import prepared_frames
    sequence = "car_urban_day_city_hall"
    base = tmp_path / sequence
    (base / "events").mkdir(parents=True)
    metadata = dict(dataset="M3ED", sequence_name=sequence, event_size=[360, 640],
                    downsampling="dagr_stateful_signed_event_downsampling", coordinate_space="rectified_left_event",
                    representation=dict(type="gep_rgb", percentile=90), frame_count=3, timestamps=[0, 50_000, 100_000])
    (base / "metadata.json").write_text(json.dumps(metadata))
    for i in (1, 2):
        torch.save(dict(sequence_name=sequence, frame_index=i, timestamp=metadata["timestamps"][i],
                        previous_timestamp=metadata["timestamps"][i - 1], representation="gep_rgb",
                        event_count=0, events=torch.ones(3, 360, 640)),
                   base / "events" / f"{metadata['timestamps'][i]}.pt")
    cfg = OmegaConf.create(dict(dataset=dict(sensor_height=360, sensor_width=640,
                                             representation=dict(type="gep_rgb", percentile=90))))
    assert len(list(prepared_frames(tmp_path, sequence, cfg, "m3ed"))) == 2
    (base / "events" / "50000.pt").unlink()
    with pytest.raises(ValueError, match="count mismatch"):
        list(prepared_frames(tmp_path, sequence, cfg, "m3ed"))


def test_published_dense_protocol_factory_guard():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from omegaconf import OmegaConf
    from event_state.dense.protocol import load_protocol
    from event_state.training.factory import validate_config
    conf = str(Path(__file__).resolve().parents[1] / "configs")
    p = load_protocol("m3ed_f3")
    with initialize_config_dir(config_dir=conf, version_base=None):
        cfg = compose(config_name="config", overrides=["dataset=m3ed_half_dagr", "experiment=activity_z_h_all"])
    cfg.teacher.cache_dir = "/unused/preflight/teacher"
    OmegaConf.update(cfg, "dataset.dense_evaluation_protocol", "m3ed_f3", force_add=True)
    cfg.dataset.train_sequences = p["splits"]["train"]
    cfg.dataset.val_sequences = None
    cfg.training.validation_enabled = False
    OmegaConf.update(cfg, "dataset.normalization_provenance", dict(sequences=p["splits"]["train"],
                    normalize_mean=list(cfg.dataset.representation.normalize_mean),
                    normalize_std=list(cfg.dataset.representation.normalize_std)), force_add=True)
    validate_config(cfg)
    cfg.dataset.train_sequences = p["splits"]["train"] + p["splits"]["val"][:1]
    with pytest.raises(ValueError, match="published training"):
        validate_config(cfg)


def test_synthetic_frozen_probe_train_evaluate_and_state_pair(tmp_path):
    """Exercise all public CLI stages after extraction without a real GPU/backbone."""
    import json
    from pathlib import Path
    import subprocess
    import sys
    import h5py
    from event_state.dense.protocol import audit_pretraining, digest, load_protocol, sha256_file
    from event_state.dense.data import DenseFeatureDataset

    protocol = load_protocol("m3ed_f3")
    cfg = dict(dataset=dict(name="m3ed", train_sequences=protocol["splits"]["train"], val_sequences=[]),
               training=dict(validation_enabled=False))
    audit = audit_pretraining(cfg, protocol, statistics_sequences=protocol["splits"]["train"])

    def cache(role, state_policy):
        root = tmp_path / f"{role}_{state_policy}"
        root.mkdir()
        records, sources = [], {}
        for sequence in protocol["splits"][role]:
            target_path = tmp_path / f"{sequence}.h5"
            if not target_path.exists():
                with h5py.File(target_path, "w") as stream:
                    stream["timestamps"] = [100_000]
                    target = np.full((1, 32, 32), 2., dtype=np.float32)
                    target[0, 0, 0] = np.nan
                    stream["depth_m"] = target
                    stream["depth_valid"] = np.isfinite(target)
            sources[str(target_path)] = sha256_file(target_path)
            path = root / f"{sequence}.pt"
            torch.save(dict(sequence=sequence, timestamp=100_000, checkpoint_sha256="backbone",
                            protocol_sha256=digest(protocol), state_policy=state_policy,
                            features=dict(z=torch.ones(4, 2, 2), h=torch.ones(4, 2, 2)),
                            size=[32, 32], low_activity=True, low_duration_us=100_000,
                            event_rate_hz=1., activity_fraction=.01), path)
            records.append(dict(sequence=sequence, timestamp=100_000, output_index=0,
                                encoding="m3ed_prepared_depth_m", target_path=str(target_path),
                                target_row=0, feature_path=str(path)))
        meta = dict(protocol=protocol, protocol_sha256=digest(protocol), checkpoint_sha256="backbone",
                    role=role, dataset="m3ed", task="depth", state_policy=state_policy,
                    normalization_audit=audit, records=records, target_source_identity=sources,
                    labeled=True, geometry="synthetic_native", low_activity_threshold=2.,
                    low_activity_metric="event_rate_hz")
        (root / "metadata.json").write_text(json.dumps(meta))
        (root / "_SUCCESS").write_text("complete")
        return root

    train, val, reset = cache("train", "continuous"), cache("val", "continuous"), cache("val", "frame")
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "head"

    def run(script, *args):
        subprocess.run([sys.executable, str(repository / "tools" / script), *map(str, args)],
                       cwd=repository, check=True, capture_output=True, text=True)

    run("train_dense_probe.py", "--train-cache", train, "--val-cache", val, "--feature", "z",
        "--epochs", 1, "--batch-size", 2, "--width", 8, "--device", "cpu", "--output-dir", output)
    assert (output / "best.pt").is_file()
    for name, data in (("continuous", val), ("frame", reset)):
        run("evaluate_dense_probe.py", "--checkpoint", output / "best.pt", "--cache", data,
            "--device", "cpu", "--output", output / f"{name}.json")
    run("compare_dense_probes.py", "--baseline", output / "frame.json", "--candidate", output / "continuous.json",
        "--output", output / "comparison.json")
    result = json.loads((output / "comparison.json").read_text())
    assert result["differences"]["all"]["mae"] == 0
    # A mutable target file must invalidate the frozen evaluation.
    with (tmp_path / f"{protocol['splits']['val'][0]}.h5").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="provenance changed"):
        DenseFeatureDataset(val, "z")


@pytest.mark.parametrize("name", ["dsec_f3", "m3ed_f3"])
@pytest.mark.parametrize("condition", ["baseline", "alpha1"])
def test_launcher_hydra_overrides_compose_and_validate(tmp_path, name, condition):
    import argparse
    import json
    from pathlib import Path
    import sys
    from hydra import compose, initialize_config_dir
    from event_state.dense.protocol import digest, load_protocol
    from event_state.training.factory import validate_config

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from run_dense_pretrain import command
    p = load_protocol(name)
    stats = tmp_path / "statistics.json"
    stats.write_text(json.dumps(dict(protocol_sha256=digest(p), sequences=p["splits"]["train"],
                                    normalize_mean=[.9, .8, .9], normalize_std=[.2, .3, .2])))
    a = argparse.Namespace(protocol=name, statistics=stats, data_root=tmp_path, event_cache_dir=tmp_path,
        teacher_cache_dir=tmp_path, teacher_checkpoint=tmp_path / "teacher.pt", output_dir=tmp_path / "new",
        condition=condition, seed=0, stage="smoke", window_us=None)
    overrides = command(a)[2:]
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    validate_config(cfg)
    assert list(cfg.dataset.train_sequences) == p["splits"]["train"]
