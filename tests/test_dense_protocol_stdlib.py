"""Portable leakage/protocol/paired-result checks; no ML imports."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from event_state.dense.metrics import RegressionSums
from event_state.dense.temporal import LowActivityClock
from event_state.dense.protocol import (audit_pretraining, digest, load_protocol,
                                        load_statistics_provenance, pretraining_audit_config,
                                        sequences_for, validate_protocol)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_dense_probes import compare
from prepare_dense_targets import parse_test_csv
from run_dense_pretrain import command


def config(protocol):
    return dict(dataset=dict(name=protocol["dataset"], train_sequences=protocol["splits"]["train"],
                             val_sequences=[]), training=dict(validation_enabled=False))


class PublishedProtocols(unittest.TestCase):
    def test_audit_projection_does_not_resolve_root_or_output(self):
        p = load_protocol("m3ed_f3")
        cfg = config(p)
        cfg["dataset"]["representation"] = dict(normalize_mean=[0, 0, 0], normalize_std=[1, 1, 1])
        cfg["dataset"]["normalization_provenance"] = dict(sequences=p["splits"]["train"])
        cfg["output"] = dict(directory="${hydra:runtime.output_dir}")
        converted = []

        def convert(value):
            self.assertIsNot(value, cfg)
            self.assertIsNot(value, cfg["output"])
            converted.append(value)
            return value

        plain = pretraining_audit_config(cfg, to_plain=convert)
        self.assertNotIn("output", plain)
        self.assertTrue(converted)
        self.assertEqual(audit_pretraining(plain, p, statistics_sequences=p["splits"]["train"])["status"],
                         "declared_inputs_disjoint")

    def test_audit_projection_keeps_validation_and_history_leakage_checks(self):
        p = load_protocol("m3ed_f3")
        cfg = config(p)
        cfg["dataset"]["representation"] = dict(normalize_mean=[0, 0, 0], normalize_std=[1, 1, 1])
        cfg["pretraining_history"] = [{"sequences": p["splits"]["val"]}]
        with self.assertRaisesRegex(ValueError, "history"):
            audit_pretraining(pretraining_audit_config(cfg), p, statistics_sequences=p["splits"]["train"])

    def test_published_f3_membership(self):
        dsec, m3ed = load_protocol("dsec_f3"), load_protocol("m3ed_f3")
        self.assertEqual([len(dsec["splits"][k]) for k in ("train", "val", "test")], [37, 4, 12])
        self.assertEqual(set(dsec["splits"]["val"]),
                         {"interlaken_00_g", "zurich_city_02_e", "zurich_city_09_e", "zurich_city_11_c"})
        self.assertEqual(set(m3ed["splits"]["val"]),
                         {"car_urban_day_rittenhouse", "car_urban_day_ucity_small_loop"})
        self.assertEqual(len(m3ed["splits"]["train"]), 3)
        for p in (dsec, m3ed):
            self.assertTrue(all(s["revision"] == "833e2c65233f9787449ce0675ae41e3f6e9c193a" for s in p["sources"]))

    def test_no_invented_official_validation(self):
        for name, count in (("dsec_official", 41), ("m3ed_official_car_day", 6)):
            p = load_protocol(name)
            self.assertEqual(len(sequences_for(p, "train")), count)
            with self.assertRaisesRegex(ValueError, "no val"):
                sequences_for(p, "val")
            with self.assertRaisesRegex(ValueError, "not public"):
                sequences_for(p, "test", labels=True)

    def test_no_relabeling_or_duplicate_split(self):
        p = load_protocol("dsec_f3")
        p["splits"]["train"] += p["splits"]["val"][:1]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_protocol(p)

    def test_legacy_train41_rejected_for_f3(self):
        p = load_protocol("dsec_f3")
        cfg = config(p)
        cfg["dataset"]["train_sequences"] = load_protocol("dsec_official")["splits"]["train"]
        with self.assertRaisesRegex(ValueError, "overlaps downstream holdout"):
            audit_pretraining(cfg, p, statistics_sequences=p["splits"]["train"])

    def test_pretrain_selection_on_downstream_val_rejected(self):
        p = load_protocol("m3ed_f3")
        cfg = config(p)
        cfg["dataset"]["val_sequences"] = ["car_urban_day_ucity_small_loop"]
        cfg["training"]["validation_enabled"] = True
        with self.assertRaisesRegex(ValueError, "model selection"):
            audit_pretraining(cfg, p, statistics_sequences=p["splits"]["train"])

    def test_pretrain_test_and_earlier_stages_rejected(self):
        p = load_protocol("dsec_official")
        cfg = config(p)
        cfg["pretraining_history"] = [{"sequences": p["splits"]["test"][:1]}]
        with self.assertRaisesRegex(ValueError, "history"):
            audit_pretraining(cfg, p, statistics_sequences=p["splits"]["train"])

    def test_missing_sequences_and_stats_fail_closed(self):
        p = load_protocol("m3ed_f3")
        with self.assertRaisesRegex(ValueError, "statistics need"):
            audit_pretraining(config(p), p)
        cfg = config(p)
        cfg["dataset"]["train_sequences"] = None
        with self.assertRaisesRegex(ValueError, "actual pretraining"):
            audit_pretraining(cfg, p, statistics_sequences=p["splits"]["train"])

    def test_stats_cannot_see_val_or_unused_data(self):
        p = load_protocol("m3ed_f3")
        for extra in p["splits"]["val"] + ["car_urban_day_horse"]:
            with self.assertRaisesRegex(ValueError, "outside downstream train"):
                audit_pretraining(config(p), p, statistics_sequences=p["splits"]["train"] + [extra])

    def test_clean_passes(self):
        for name in ("dsec_f3", "m3ed_f3", "dsec_official", "m3ed_official_car_day"):
            p = load_protocol(name)
            self.assertEqual(audit_pretraining(config(p), p, statistics_sequences=p["splits"]["train"])["status"],
                             "declared_inputs_disjoint")

    def test_stats_must_match_actual_coefficients(self):
        cfg = dict(dataset=dict(representation=dict(normalize_mean=[0, 0, 0], normalize_std=[1, 1, 1])))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "statistics.json"
            path.write_text(json.dumps(dict(source="train", normalize_mean=[1, 0, 0], normalize_std=[1, 1, 1])))
            with self.assertRaisesRegex(ValueError, "coefficients"):
                load_statistics_provenance(path, cfg)

    def test_clean_launcher_fixed_step_train_only(self):
        import argparse
        with tempfile.TemporaryDirectory() as tmp:
            p = load_protocol("m3ed_f3")
            stats = Path(tmp) / "stats.json"
            stats.write_text(json.dumps(dict(protocol_sha256=digest(p), sequences=p["splits"]["train"],
                                            normalize_mean=[0, 0, 0], normalize_std=[1, 1, 1])))
            a = argparse.Namespace(protocol="m3ed_f3", statistics=stats, data_root=Path(tmp), event_cache_dir=Path(tmp),
                teacher_cache_dir=Path(tmp), teacher_checkpoint=Path(tmp) / "teacher.pt", output_dir=Path(tmp) / "new",
                condition="alpha1", seed=0, stage="smoke", window_us=None)
            cmd = command(a)
            self.assertIn("training.validation_enabled=false", cmd)
            self.assertIn("dataset.val_sequences=null", cmd)
            self.assertIn("++dataset.dense_evaluation_protocol=m3ed_f3", cmd)
            self.assertNotIn("horse", next(x for x in cmd if x.startswith("dataset.train_sequences=")))


class PairedMetrics(unittest.TestCase):
    def test_low_event_clock_includes_intermediate_frames_and_rejects_gaps(self):
        clock = LowActivityClock()
        self.assertEqual(clock.update(0, 50_000, True), 50_000)
        self.assertEqual(clock.update(50_000, 100_000, True), 100_000)
        self.assertEqual(clock.update(100_000, 150_000, False), 0)
        self.assertEqual(clock.update(150_000, 200_000, True), 50_000)
        with self.assertRaisesRegex(ValueError, "gap"):
            clock.update(250_000, 300_000, True)

    def result(self):
        return dict(protocol_sha256="p", role="val", task="depth", feature="h", minimum=.1, maximum=80,
                    scale_alignment="none", samples_targets_sha256="samples", validity_sha256="mask", frames=[],
                    geometry="native", low_activity_threshold=.1, low_activity_metric="activity_fraction",
                    head_sha256="head", backbone_sha256="backbone",
                    state_policy="frame", metrics=dict(all=dict(valid_pixels=4, frames=1, mae=2.0)))

    def test_same_head_and_samples_required(self):
        baseline = self.result()
        candidate = copy.deepcopy(baseline)
        candidate["state_policy"] = "continuous"
        candidate["metrics"]["all"]["mae"] = 1.5
        self.assertEqual(compare(baseline, candidate)["differences"]["all"]["mae"], -.5)
        for key in ("head_sha256", "samples_targets_sha256", "validity_sha256", "geometry"):
            bad = {**candidate, key: "changed"}
            with self.assertRaisesRegex(ValueError, key):
                compare(baseline, bad)

    def test_z_invariance_required(self):
        baseline = {**self.result(), "feature": "z", "z_features_sha256": "one"}
        candidate = {**baseline, "state_policy": "continuous", "z_features_sha256": "two"}
        with self.assertRaisesRegex(ValueError, "z changed"):
            compare(baseline, candidate)

    def test_empty_strata_null_and_pooled_rmse(self):
        meter = RegressionSums("disparity")
        self.assertIsNone(meter.compute()["mae"])
        meter.add(2, mae=3, rmse=5, **{"1pe_percent": 100, "2pe_percent": 0})
        meter.add(1, mae=3, rmse=9, **{"1pe_percent": 100, "2pe_percent": 100})
        self.assertEqual(meter.compute()["mae"], 2)
        self.assertAlmostEqual(meter.compute()["rmse"], (14 / 3) ** .5)

    def test_official_submission_timestamp_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seq.csv"
            path.write_text("timestamp_us,index\n1000000,20\n2000000,40\n")
            self.assertEqual(parse_test_csv(path), [(1000000, 20), (2000000, 40)])
            path.write_text("timestamp_us,index\n1000000,20\n2000000,20\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                parse_test_csv(path)


if __name__ == "__main__":
    unittest.main()
