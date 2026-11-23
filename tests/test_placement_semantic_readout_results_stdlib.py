import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tools import placement_semantic_readout_results as results


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class MatchedReadoutTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.placement = self.root / "placements"

    def control(self, condition="E1", feature="h", metric=0.5):
        return dict(dataset="m3ed", condition=condition, feature=feature, seed=0, global_step=99999,
                    role="validation", backbone_sha256="backbone-" + condition, head_sha256=condition + feature,
                    head_recipe={"classes": ["a", "b"]}, mIoU=dict.fromkeys(results.GROUPS, metric))

    def cells(self):
        old = {key: self.control(key, "h", score) for key, score in zip(results.CONDITIONS, (.6, .5))}
        new = {key: self.control(key, "stages", score) for key, score in zip(results.CONDITIONS, (.62, .58))}
        return old, new

    def test_interaction_and_remaining_gap_in_percentage_points(self):
        old, new = self.cells()
        new[results.FOUR_STAGE]["mIoU"]["p00_20"] = .47
        table = results.compare_readouts("m3ed", old, new)
        self.assertAlmostEqual(table["gain_pp"]["E1"]["all"], 2)
        self.assertAlmostEqual(table["gain_pp"][results.FOUR_STAGE]["all"], 8)
        self.assertAlmostEqual(table["placement_gap_pp"]["h"]["all"], -10)
        self.assertAlmostEqual(table["placement_gap_pp"]["stages"]["all"], -4)
        self.assertAlmostEqual(table["interaction_pp"]["all"], 6)
        self.assertAlmostEqual(table["interaction_pp"]["p00_20"], -5)

    def test_corrupted_or_mismatched_cell_cannot_enter_summary(self):
        old, new = self.cells()
        for name, value in (("backbone_sha256", "other"), ("feature", "h"), ("dataset", "dsec"),
                            ("head_recipe", {"seed": 7}), ("role", "test"), ("global_step", 4980),
                            ("mIoU", dict.fromkeys(results.GROUPS, float("nan")))):
            with self.subTest(field=name), self.assertRaises(ValueError):
                changed = copy.deepcopy(new)
                changed["E1"][name] = value
                results.compare_readouts("m3ed", old, changed)

    def make_control(self, dataset):
        dsec = dataset == "dsec"
        head_name = "final/last.pt" if dsec else "h/best.pt"
        run = results.semantic_run(dataset, "E1").relative_to(results.ROOT)
        head = self.root / run / head_name
        head.parent.mkdir(parents=True)
        head.write_bytes(b"frozen linear probe")
        manifest_name = "dsec_semantic_split.yaml" if dsec else "m3ed_semantic_common_v1.yaml"
        manifest = Path("tools/manifests") / manifest_name
        (self.root / manifest.parent).mkdir(parents=True, exist_ok=True)
        shutil.copy(results.ROOT / manifest, self.root / manifest)
        split = results._split(self.root / manifest) if dsec else results.load_common("m3ed")["splits"]
        train = split["train"] + split["val"] if dsec else split["train"]
        evaluation = split["test" if dsec else "val"]
        cfg = dict(feature="h", in_channels=384, split_manifest=str(manifest), feature_cache_dir="cache",
                   training_sequences=train, num_classes=11, seed=0, precision="fp16", loss="ce",
                   batch_size=8, head_type="linear", weight_decay=0.05, learning_rate=2e-4,
                   horizontal_flip_probability=0.5, ignore_index=255, classes=list(range(11)),
                   train_frames=100, validation_frames=0 if dsec else 40,
                   label_size=[440 if dsec else 352, 640])
        if dsec:
            cfg.update(protocol="official-final", epochs=12, selected_development_epoch=11,
                       activity_active_weight=1, activity_inactive_weight=1)
        else:
            cfg.update(evaluation_role="validation", validation_sequences=evaluation, epochs=50, validate_every=5)
        write(head.parent / "config.json", cfg)
        for sequence in (*train, *evaluation):
            write(self.root / "cache" / sequence / "metadata.json", dict(checkpoint_sha256="frozen",
                sequence_name=sequence, state_policy="continuous", features=["h"]))
        reference = results.baseline_results()[dataset].copy()
        reference["head_sha256"] = results.sha256_file(head)
        return head, reference

    def test_controls_reuse_archived_scores_with_checkpoint_lineage(self):
        for dataset in ("dsec", "m3ed"):
            head, reference = self.make_control(dataset)
            def load():
                return results.load_final_control(dataset, "E1", project_root=self.root,
                    placement_root=self.placement, backbone_sha256="frozen")
            with self.subTest(dataset=dataset), patch.object(results, "baseline_results",
                                                             return_value={dataset: reference}):
                row = load()
                self.assertEqual(row["head"], str(head))
                self.assertEqual(row["mIoU"]["p60_80"], reference["groups"]["p60_80"]["metrics"]["mIoU"])
                seq = next(iter(reference["state_policy_by_sequence"]))
                path = self.root / "cache" / seq / "metadata.json"
                stale = json.loads(path.read_text())
                stale["checkpoint_sha256"] = "unrelated"
                write(path, stale)
                with self.assertRaisesRegex(ValueError, "Backbone"):
                    load()
                head.write_bytes(b"different head")
                with self.assertRaisesRegex(ValueError, "Head differs"):
                    load()

    def test_four_stage_rejects_invalid_attempt_or_checkpoint_before_opening_report(self):
        key = results.FOUR_STAGE
        path = self.placement / ("dsec_" + key) / "verified.json"
        saved = dict(condition=key, dataset="dsec", seed=0, global_step=99999,
                     backbone_sha256="old", after_blocks=list(results.BLOCKS))
        for attempt, error in (("run-1/../another", "Invalid Semantic attempt"), ("run-1", "Frozen backbone")):
            write(path, dict(saved, attempt=attempt))
            with self.assertRaisesRegex(ValueError, error):
                results.load_final_control("dsec", key, project_root=self.root,
                    placement_root=self.placement, backbone_sha256="new")


if __name__ == "__main__":
    unittest.main()
