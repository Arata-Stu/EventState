"""Placement Semantic routing, identity, cleanup and resumption without Torch or data."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_placement_semantic as suite


class PlacementSemanticTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.baselines = suite.baseline_results()

    def test_defaults_plan_all_six_for_both_datasets_without_writes(self):
        output = self.root / "output"
        with redirect_stdout(io.StringIO()) as stream:
            suite.main(["--output", str(output)])
        self.assertFalse(output.exists())
        text = stream.getvalue()
        self.assertEqual(text.count("| Semantic |"), 12)
        for key in suite.PLACEMENTS:
            self.assertIn(f"[m3ed_{key}]", text)
            self.assertIn(f"[dsec_{key}]", text)
        self.assertNotIn("depth", text.lower())

    def metadata(self, ds="dsec", cond="E1_b9"):
        return SimpleNamespace(global_step=99999, config=dict(dataset=dict(name=ds), seed=0,
            training=dict(max_steps=99999), model=dict(temporal=dict(type="four_stage_lstm",
                after_blocks=list(suite.PLACEMENTS[cond]), input_dim=384, output_dim=384, hidden_dim=384)),
            loss=dict(h_distill=dict(enabled=True, cosine_weight=1, mse_weight=1),
                      z_objective=dict(type="none", weight=0))))

    def test_wrong_dataset_seed_position_or_encoder_budget_rejected(self):
        config = self.metadata().config
        with patch.object(suite, "audit_common") as audit:
            suite.validate_config(config, "dsec", "E1_b9")
            audit.assert_called_once()
            for dataset, condition, seed in (("m3ed", "E1_b9", 0), ("dsec", "E1_b3", 0),
                                              ("dsec", "E1_b9_12", 0), ("dsec", "E1_b9", 1)):
                with self.subTest(dataset=dataset, condition=condition, seed=seed), self.assertRaises(ValueError):
                    suite.validate_config(dict(config, seed=seed), dataset, condition)
            with self.assertRaises(ValueError):
                suite.validate_config(dict(config, training=dict(max_steps=99)), "dsec", "E1_b9")

    def make_report(self, ds, cond, output):
        ref = self.baselines[ds]
        output.mkdir(parents=True)
        head = output / ("h/best.pt" if ds == "m3ed" else "final/last.pt")
        head.parent.mkdir()
        head.write_bytes(b"fixture")
        if ds == "dsec":
            suite.save(output/"final/config.json", dict(selected_development_epoch=1, epochs=2))
            development = output/"development"
            development.mkdir()
            records = [{"epoch": i, "learning_rate": 1e-3, "train/loss": 1.1, "val/mIoU": miou}
                       for i, miou in enumerate((0.32, 0.45, 0.40))]
            (development/"metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (output/"_SUCCESS").touch()
        activity = dict(groups=deepcopy(ref["groups"]), activity_manifest=dict(manifest_sha256=ref["manifest_sha256"]),
                        condition_label=suite.placement_label(cond), head_sha256=suite.sha256_file(head),
                        state_policy_by_sequence=ref["state_policy_by_sequence"],
                        **{k: ref[k] for k in suite.COMPARISON_HASHES})
        path = output / ("h/activity_groups.json" if ds == "m3ed" else "activity_test.json")
        data = dict(role="validation" if ds == "m3ed" else "test", feature="h", activity_evaluation=activity)
        suite.save(path, data)
        return path, data

    def test_head_gt_support_and_all_five_quintiles_match_reference(self):
        for ds in suite.DATASETS:
            output, cond = self.root/ds, "E1_b9_12"
            path, data = self.make_report(ds, cond, output)
            values = suite.verified_metrics(ds, cond, output, self.baselines[ds])
            self.assertEqual(set(values["mIoU"]), set(self.baselines[ds]["groups"]))
            self.assertEqual(set(values["delta_from_final_block_only_pp"].values()), {0})
            if ds == "dsec":
                selection = values["development_selection"]
                self.assertEqual((selection["epoch_zero_based"], selection["val_mIoU"]), (1, 0.45))
                suite.save(output/"final/config.json", dict(selected_development_epoch=2, epochs=3))
                with self.assertRaisesRegex(ValueError, "development selection"):
                    suite.verified_metrics(ds, cond, output, self.baselines[ds])
            else:
                self.assertNotIn("development_selection", values)
            data["activity_evaluation"]["samples_targets_sha256"] = "changed"
            suite.save(path, data)
            with self.assertRaisesRegex(ValueError, "matched Semantic protocol"):
                suite.verified_metrics(ds, cond, output, self.baselines[ds])

    def test_m3ed_fits_once_and_keeps_cached_features_on_resume(self):
        ds, condition = "m3ed", "E1_b6_9_12"
        a = SimpleNamespace(stage1_root=self.root/"stage1", output=self.root/"outputs",
            scratch=self.root/"scratch", data=self.root/"data", statistics=self.root/"stats.json",
            m3ed_target_root=self.root/"targets", activity_root=self.root/"activity",
            execute=True)
        key = f"{ds}_{condition}"
        checkpoint = a.stage1_root/f"{key}_w1/checkpoints/step_00099999.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"backbone")
        training = SimpleNamespace(load_checkpoint_config_metadata=lambda p: self.metadata(ds, condition))
        def run(command, **kwargs):
            self.assertEqual(command[command.index("--features")+1], "h")
            self.assertEqual(command[command.index("--checkpoint")+1], str(checkpoint))
            self.assertEqual(command[command.index("--condition-label")+1], suite.placement_label(condition))
            self.assertEqual(kwargs["pass_fds"], (18,))
            cache = Path(command[command.index("--cache")+1])
            (cache/"seq").mkdir(parents=True)
            (cache/"seq/metadata.json").write_text("{}")
            (cache/"seq/000099.pt").write_bytes(b"large sparse tensor")
            (cache/"_SUCCESS").touch()
            self.make_report(ds, condition, Path(command[command.index("--output")+1]))
        with patch.dict("sys.modules", {"event_state.training": training}), patch.object(suite, "audit_common"), \
                patch.object(suite.subprocess, "run", side_effect=run) as child:
            for _ in range(2):
                suite.run_condition(a, ds, condition, self.baselines[ds], 18)
            self.assertEqual(child.call_count, 1)
            result = suite.read(a.output/key/"verified.json")
            cache = Path(result["cache"])
            self.assertTrue((cache/"seq/metadata.json").is_file())
            self.assertTrue((cache/"seq/000099.pt").is_file())
            self.assertTrue((cache/"_SUCCESS").is_file())
            training.load_checkpoint_config_metadata = lambda p: SimpleNamespace(global_step=99)
            with self.assertRaisesRegex(ValueError, "99,999-update"):
                suite.run_condition(a, ds, condition, self.baselines[ds], 18)
        suite.write_summary(a, self.baselines)
        self.assertEqual(list(suite.read(a.output/"summary.json")["semantic"]), [key])
        self.assertFalse((a.output/"_SUCCESS").exists())


if __name__ == "__main__":
    unittest.main()
