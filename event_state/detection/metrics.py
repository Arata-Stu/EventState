"""COCO-style mAP evaluation for DSEC-Detection."""

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np
from torch import Tensor

from .data import DAGR_CLASSES


class COCODetectionEvaluator:
    """Accumulate frame detections and compute COCO mAP@[.50:.95]."""

    def __init__(self, class_names: tuple[str, ...] = DAGR_CLASSES) -> None:
        self.class_names = class_names
        self.images: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self.detections: list[dict[str, Any]] = []
        self._next_image_id = 1
        self._next_annotation_id = 1

    def update(
        self,
        predictions: list[dict[str, Tensor]],
        targets: list[dict[str, Tensor]],
        *,
        sequence_names: list[str],
        timestamps: list[int],
        image_size: tuple[int, int] = (480, 640),
    ) -> None:
        if not (
            len(predictions) == len(targets) == len(sequence_names) == len(timestamps)
        ):
            raise ValueError("Detection evaluation batch fields have different lengths")
        height, width = image_size
        for prediction, target, sequence, timestamp in zip(
            predictions, targets, sequence_names, timestamps
        ):
            image_id = self._next_image_id
            self._next_image_id += 1
            self.images.append(
                {
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": f"{sequence}/{timestamp}",
                }
            )
            boxes = target["boxes"].detach().cpu().float()
            labels = target["labels"].detach().cpu().long()
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box.tolist()
                box_width = max(0.0, x2 - x1)
                box_height = max(0.0, y2 - y1)
                self.annotations.append(
                    {
                        "id": self._next_annotation_id,
                        "image_id": image_id,
                        "category_id": int(label) + 1,
                        "bbox": [x1, y1, box_width, box_height],
                        "area": box_width * box_height,
                        "iscrowd": 0,
                    }
                )
                self._next_annotation_id += 1
            pred_boxes = prediction["boxes"].detach().cpu().float()
            pred_scores = prediction["scores"].detach().cpu().float()
            pred_labels = prediction["labels"].detach().cpu().long()
            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                x1, y1, x2, y2 = box.tolist()
                self.detections.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label) + 1,
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    }
                )

    def compute(self) -> dict[str, float]:
        if not self.images or not self.annotations:
            raise RuntimeError("Cannot evaluate an empty DSEC-Detection set")
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError as error:
            raise RuntimeError(
                "pycocotools is required for detection mAP; install the detection extra"
            ) from error
        ground_truth = COCO()
        ground_truth.dataset = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": [
                {"id": index + 1, "name": name}
                for index, name in enumerate(self.class_names)
            ],
            "info": {"description": "EventState DSEC-Detection evaluation"},
            "licenses": [],
        }
        ground_truth.createIndex()
        if not self.detections:
            return {
                "mAP": 0.0,
                "AP50": 0.0,
                "AP75": 0.0,
                "AP_small": 0.0,
                "AP_medium": 0.0,
                "AP_large": 0.0,
            }
        detections = ground_truth.loadRes(self.detections)
        evaluator = COCOeval(ground_truth, detections, "bbox")
        evaluator.params.imgIds = [image["id"] for image in self.images]
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        stats = np.asarray(evaluator.stats, dtype=np.float64)
        result = {
            "mAP": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "AP_small": float(stats[3]),
            "AP_medium": float(stats[4]),
            "AP_large": float(stats[5]),
        }
        precision = evaluator.eval.get("precision")
        if isinstance(precision, np.ndarray) and precision.ndim == 5:
            for class_index, class_name in enumerate(self.class_names):
                values = precision[:, :, class_index, 0, -1]
                valid = values[values > -1]
                result[f"AP_{class_name}"] = float(valid.mean()) if len(valid) else float("nan")
        return result

    def reset(self) -> None:
        self.images.clear()
        self.annotations.clear()
        self.detections.clear()
        self._next_image_id = 1
        self._next_annotation_id = 1


__all__ = ["COCODetectionEvaluator"]
