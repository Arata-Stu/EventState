"""Temporal indexing/configuration regression tests without ML dependencies."""
from copy import deepcopy
import unittest

from event_state.causal_rgb import history_positions, validate_history, sampled_history, validate_context
from event_state.vjepa_plan import make_plan


class CausalHistoryTests(unittest.TestCase):
    def test_no_future_and_endpoint(self):
        for k in (2, 4, 8, 16):
            for start in (0, 1, 5, 100):
                indices = history_positions(start, 7, k)
                for j in range(7):
                    self.assertEqual(indices[j + k - 1], start + j)
                    self.assertTrue(all(i <= start + j for i in indices[j:j + k]))

    def test_clip_boundary_invariance(self):
        whole = history_positions(1, 16, 8)
        for offset, length in ((0, 4), (4, 8), (12, 4)):
            part = history_positions(1 + offset, length, 8)
            self.assertEqual(part, whole[offset:offset + length + 7])

    def test_only_sequence_prefix_is_padded(self):
        self.assertEqual(history_positions(1, 2, 4), [0, 0, 0, 1, 2])
        self.assertEqual(history_positions(20, 2, 4), [17, 18, 19, 20, 21])

    def test_timestamp_alignment_and_corruption(self):
        indices = history_positions(1, 2, 4)
        times = [1000, 1000, 1000, 3000, 4500]
        validate_history(indices, times, [3000, 4500], 4)
        with self.assertRaises(ValueError):
            validate_history(indices, times, [3001, 4500], 4)
        with self.assertRaises(ValueError):
            validate_history(indices, [1000, 1000, 1000, 3000, 2000], [3000, 2000], 4)

    def test_invalid_context_rejected(self):
        for k in (0, 1, 3, -2):
            with self.assertRaises(ValueError):
                history_positions(0, 1, k)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(
            seed=0,
            teacher=dict(type="dinov3_vits16", checkpoint="dino.pt", backend="torch_hub",
                         repository="dino:commit", source="github"),
            model=dict(event_encoder=dict(architecture="dinov3_vits16"),
                       temporal=dict(type="lstm", dropout=0), projector=dict(output_dim=384)),
            dataset=dict(name="dsec", train_sequences=["train_a"], val_sequences=None,
                         augmentation=dict(enabled=False), representation=dict(normalize_mean=[1, 2, 3]),
                         event_cache_dir="gep", dense_evaluation_protocol="dsec_f3"),
            training=dict(validation_enabled=False, max_steps=100000, resume=None,
                          event_dropout=dict(enabled=False), batch_size=8,
                          sampling=dict(mode="mixed", random_batch_size=4, stream_batch_size=4)),
            scheduler=dict(warmup_steps=1000), loss={},
        )
        self.teacher = dict(type="vjepa2_1_vit_base_384", context_frames=8, checkpoint="vjepa.pt")

    def test_preserve_split_normalization_student_and_budget(self):
        original = deepcopy(self.base)
        plan = make_plan(self.base, self.teacher, output="new", stage="full")
        self.assertEqual(self.base, original)
        for key in ("train_sequences", "val_sequences", "representation", "event_cache_dir", "dense_evaluation_protocol"):
            self.assertEqual(plan["dataset"][key], original["dataset"][key])
        self.assertEqual(plan["model"]["event_encoder"]["checkpoint"], "dino.pt")
        self.assertEqual(plan["teacher"]["checkpoint"], "vjepa.pt")
        self.assertEqual(plan["model"]["projector"]["output_dim"], 768)
        self.assertEqual(plan["loss"]["z_objective"]["type"], "none")
        self.assertEqual(plan["training"]["max_steps"], 100000)
        self.assertEqual(plan["training"]["batch_size"], 8)

    def test_smoke_and_m3ed(self):
        self.base["dataset"]["name"] = "m3ed"
        plan = make_plan(self.base, self.teacher, output="new", stage="smoke", batch_size=2)
        self.assertEqual(plan["training"]["max_steps"], 10)
        self.assertEqual(plan["training"]["sampling"]["stream_batch_size"], 1)
        self.assertEqual(plan["training"]["sampling"]["random_batch_size"], 1)

    def test_no_hidden_test_selection(self):
        self.base["training"]["validation_enabled"] = True
        with self.assertRaises(ValueError):
            make_plan(self.base, self.teacher, output="new", stage="full")

    def test_no_implicit_student_weights(self):
        del self.base["teacher"]["checkpoint"]
        with self.assertRaises(ValueError):
            make_plan(self.base, self.teacher, output="new", stage="full")

    def test_image_and_video_hold_student_and_training_fixed(self):
        image_teacher = dict(self.teacher, input_mode="image", context_frames=1, sample_fps=None)
        video_teacher = dict(self.teacher, input_mode="video", context_frames=16, sample_fps=4)
        image = make_plan(self.base, image_teacher, output="image", stage="full")
        video = make_plan(self.base, video_teacher, output="video", stage="full")
        for key in ("dataset", "model", "loss", "training", "optimizer", "scheduler", "seed"):
            self.assertEqual(image.get(key), video.get(key))
        self.assertNotEqual(image["experiment"], video["experiment"])


class SampledHistoryTests(unittest.TestCase):
    def sample(self, times, start, length=1, k=4, fps=4):
        return sampled_history(times, start, length, mode="video", context_frames=k, sample_fps=fps)

    def test_four_fps_regular_and_context_duration(self):
        times = list(range(0, 10_000_001, 50_000))
        rows, wanted, padded = self.sample(times, 100, k=16)
        self.assertEqual(rows[0], list(range(25, 101, 5)))
        self.assertEqual(wanted[0][-1] - wanted[0][0], 3_750_000)
        self.assertFalse(any(padded[0]))

    def test_irregular_timeline_uses_floor_not_nearest(self):
        times = [0, 240000, 510000, 740000, 1000000]
        rows, wanted, padded = self.sample(times, 4)
        self.assertEqual(wanted[0], [250000, 500000, 750000, 1000000])
        self.assertEqual(rows[0], [1, 1, 3, 4])  # 510000 is future at requested 500000.
        self.assertFalse(any(padded[0]))

    def test_prefix_only_repeats_first_frame(self):
        rows, wanted, padded = self.sample([1000000, 1050000], 1)
        self.assertEqual(rows[0], [0, 0, 0, 1])
        self.assertEqual(padded[0], [True, True, True, False])
        self.assertEqual(wanted[0][-1], 1050000)

    def test_future_and_clip_partition_invariance(self):
        times = list(range(0, 6_000_001, 50000))
        whole = self.sample(times, 30, length=16)
        for offset, length in ((0, 4), (4, 8), (12, 4)):
            part = self.sample(times, 30 + offset, length=length)
            self.assertEqual(part, tuple(x[offset:offset + length] for x in whole))
        self.assertEqual(whole, self.sample(times[:46], 30, length=16))

    def test_single_image_has_no_history(self):
        self.assertEqual(sampled_history([0, 100, 200], 1, 2, mode="image", context_frames=1, sample_fps=None),
                         ([[1], [2]], [[100], [200]], [[False], [False]]))

    def test_fractional_fps_large_timestamp(self):
        times = [10**16 + i * 100000 for i in range(50)]
        rows, wanted, _ = self.sample(times, 40, fps=2.5)
        self.assertEqual(rows[0], [28, 32, 36, 40])
        self.assertEqual(wanted[0], [times[i] for i in rows[0]])

    def test_invalid_modes_and_rates(self):
        for mode, k, fps in [("image", 2, None), ("image", 1, 4), ("video", 1, 4),
                             ("video", 4, 0), ("video", 4, float("nan")), ("other", 4, 4)]:
            with self.assertRaises(ValueError):
                validate_context(mode, k, fps)


if __name__ == "__main__":
    unittest.main()
