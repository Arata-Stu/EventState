"""A compact dense YOLOX-style detector for frozen EventState maps.

The implementation is native to this project.  It follows the public YOLOX
design (decoupled classification/regression branches and dynamic-k matching)
without importing DAGR or RVT code.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import batched_nms


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int = 1):
        padding = kernel // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(16, channels // 2)
        self.block = nn.Sequential(
            ConvBNAct(channels, hidden, 1),
            ConvBNAct(hidden, channels, 3),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.block(value)


class EventStatePyramid(nn.Module):
    """Turn the stride-16 token map into stride 8/16/32 YOLO features."""

    def __init__(self, in_channels: int, width: int = 192) -> None:
        super().__init__()
        self.p4 = nn.Sequential(ConvBNAct(in_channels, width, 1), ResidualBlock(width))
        self.p3 = nn.Sequential(ConvBNAct(width, width, 3), ResidualBlock(width))
        self.p5 = nn.Sequential(ConvBNAct(width, width, 3, stride=2), ResidualBlock(width))
        self.pan4 = nn.Sequential(ConvBNAct(width * 2, width, 3), ResidualBlock(width))
        self.pan5 = nn.Sequential(ConvBNAct(width * 2, width, 3), ResidualBlock(width))

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        p4_base = self.p4(value)
        p3 = self.p3(F.interpolate(p4_base, scale_factor=2.0, mode="nearest"))
        p5_base = self.p5(p4_base)
        p4 = self.pan4(
            torch.cat((p4_base, F.max_pool2d(p3, kernel_size=2, stride=2)), dim=1)
        )
        p5 = self.pan5(
            torch.cat((p5_base, F.max_pool2d(p4, kernel_size=2, stride=2)), dim=1)
        )
        return p3, p4, p5


class DecoupledHeadLevel(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.stem = ConvBNAct(channels, channels, 1)
        self.cls_branch = nn.Sequential(
            ConvBNAct(channels, channels, 3), ConvBNAct(channels, channels, 3)
        )
        self.reg_branch = nn.Sequential(
            ConvBNAct(channels, channels, 3), ConvBNAct(channels, channels, 3)
        )
        self.cls_pred = nn.Conv2d(channels, num_classes, 1)
        self.box_pred = nn.Conv2d(channels, 4, 1)
        self.obj_pred = nn.Conv2d(channels, 1, 1)

    def forward(self, value: Tensor) -> Tensor:
        value = self.stem(value)
        cls_feature = self.cls_branch(value)
        reg_feature = self.reg_branch(value)
        return torch.cat(
            (self.box_pred(reg_feature), self.obj_pred(reg_feature), self.cls_pred(cls_feature)),
            dim=1,
        )


def _xywh_to_xyxy(boxes: Tensor) -> Tensor:
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def _pairwise_iou(first: Tensor, second: Tensor) -> Tensor:
    top_left = torch.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area_first = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(dim=-1)
    area_second = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(dim=-1)
    union = area_first[:, None] + area_second[None, :] - intersection
    return intersection / union.clamp_min(1e-8)


def _aligned_iou(first: Tensor, second: Tensor) -> Tensor:
    top_left = torch.maximum(first[:, :2], second[:, :2])
    bottom_right = torch.minimum(first[:, 2:], second[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area_first = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(dim=-1)
    area_second = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(dim=-1)
    return intersection / (area_first + area_second - intersection).clamp_min(1e-8)


class EventStateYOLOX(nn.Module):
    """YOLOX-style PAFPN/head operating on one frozen EventState feature map."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int = 2,
        width: int = 192,
        input_stride: int = 16,
        confidence_threshold: float = 0.001,
        nms_threshold: float = 0.65,
        max_detections: int = 100,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or num_classes <= 0 or width <= 0:
            raise ValueError("in_channels, num_classes, and width must be positive")
        if input_stride <= 0 or input_stride % 2:
            raise ValueError("input_stride must be a positive even integer")
        self.num_classes = int(num_classes)
        self.input_stride = int(input_stride)
        self.strides = (
            self.input_stride // 2,
            self.input_stride,
            self.input_stride * 2,
        )
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.max_detections = int(max_detections)
        if self.max_detections <= 0:
            raise ValueError("max_detections must be positive")
        self.neck = EventStatePyramid(in_channels, width)
        self.heads = nn.ModuleList(
            DecoupledHeadLevel(width, self.num_classes) for _ in self.strides
        )
        prior = 0.01
        bias = -math.log((1 - prior) / prior)
        for head in self.heads:
            nn.init.constant_(head.cls_pred.bias, bias)
            nn.init.constant_(head.obj_pred.bias, bias)

    def _flatten_predictions(
        self, features: tuple[Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        flattened: list[Tensor] = []
        grids: list[Tensor] = []
        stride_values: list[Tensor] = []
        for feature, head, stride in zip(features, self.heads, self.strides):
            prediction = head(feature)
            batch, _, height, width = prediction.shape
            flattened.append(prediction.flatten(2).permute(0, 2, 1))
            y, x = torch.meshgrid(
                torch.arange(height, device=prediction.device),
                torch.arange(width, device=prediction.device),
                indexing="ij",
            )
            grid = torch.stack((x, y), dim=-1).reshape(1, height * width, 2)
            grids.append(grid.to(prediction.dtype))
            stride_values.append(
                prediction.new_full((1, height * width, 1), float(stride))
            )
            if batch <= 0:
                raise ValueError("Empty detector batch")
        return torch.cat(flattened, dim=1), torch.cat(grids, dim=1), torch.cat(stride_values, dim=1)

    @staticmethod
    def _decode_boxes(raw_boxes: Tensor, grid: Tensor, strides: Tensor) -> Tensor:
        centers = (raw_boxes[..., :2] + grid) * strides
        sizes = raw_boxes[..., 2:].clamp(max=math.log(1e4)).exp() * strides
        return _xywh_to_xyxy(torch.cat((centers, sizes), dim=-1))

    def forward(
        self,
        features: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> Any:
        pyramid = self.neck(features)
        predictions, grid, strides = self._flatten_predictions(pyramid)
        decoded_boxes = self._decode_boxes(predictions[..., :4], grid, strides)
        if targets is not None:
            return self.loss(predictions, decoded_boxes, grid, strides, targets)
        return self.postprocess(
            predictions,
            decoded_boxes,
            image_size=(
                int(features.shape[-2]) * self.input_stride,
                int(features.shape[-1]) * self.input_stride,
            ),
        )

    @torch.no_grad()
    def _assign(
        self,
        boxes: Tensor,
        objectness_logits: Tensor,
        class_logits: Tensor,
        grid: Tensor,
        strides: Tensor,
        target: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        output_dtype = boxes.dtype
        boxes = boxes.float()
        objectness_logits = objectness_logits.float()
        class_logits = class_logits.float()
        grid = grid.float()
        strides = strides.float()
        gt_boxes = target["boxes"].to(boxes.device).float()
        gt_labels = target["labels"].to(boxes.device).long()
        anchor_count = boxes.shape[0]
        if len(gt_boxes) == 0:
            return (
                torch.zeros(anchor_count, dtype=torch.bool, device=boxes.device),
                torch.empty(0, dtype=torch.long, device=boxes.device),
                boxes.new_empty(0),
            )
        pair_iou = _pairwise_iou(gt_boxes, boxes)
        gt_one_hot = F.one_hot(gt_labels, self.num_classes).float()
        joint_probability = (
            class_logits.sigmoid()[None] * objectness_logits.sigmoid()[None]
        ).sqrt()
        cls_cost = F.binary_cross_entropy(
            joint_probability.expand(len(gt_boxes), -1, -1),
            gt_one_hot[:, None, :].expand(-1, anchor_count, -1),
            reduction="none",
        ).sum(dim=-1)
        centers = (grid[0] + 0.5) * strides[0]
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        center_radius = 2.5 * strides[0, :, 0]
        in_center = (
            (centers[None, :, 0] - gt_centers[:, None, 0]).abs() < center_radius[None]
        ) & (
            (centers[None, :, 1] - gt_centers[:, None, 1]).abs() < center_radius[None]
        )
        cost = cls_cost - 3.0 * torch.log(pair_iou.clamp_min(1e-8))
        cost = cost + (~in_center).float() * 1e6
        matching = torch.zeros_like(cost, dtype=torch.bool)
        candidate_k = min(10, anchor_count)
        top_iou = torch.topk(pair_iou, candidate_k, dim=1).values
        dynamic_k = top_iou.sum(dim=1).int().clamp(min=1)
        for gt_index, count in enumerate(dynamic_k.tolist()):
            selected = torch.topk(cost[gt_index], k=count, largest=False).indices
            matching[gt_index, selected] = True
        multiple = matching.sum(dim=0) > 1
        if multiple.any():
            best_gt = cost[:, multiple].argmin(dim=0)
            matching[:, multiple] = False
            matching[best_gt, multiple] = True
        foreground = matching.any(dim=0)
        matched_gt = matching[:, foreground].float().argmax(dim=0).long()
        matched_iou = (matching[:, foreground].float() * pair_iou[:, foreground]).sum(dim=0)
        return foreground, matched_gt, matched_iou.to(output_dtype)

    def loss(
        self,
        predictions: Tensor,
        boxes: Tensor,
        grid: Tensor,
        strides: Tensor,
        targets: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        if len(targets) != predictions.shape[0]:
            raise ValueError("targets length must match detector batch size")
        box_loss = predictions.new_zeros(())
        obj_loss = predictions.new_zeros(())
        cls_loss = predictions.new_zeros(())
        positive_count = 0
        for batch_index, target in enumerate(targets):
            objectness = predictions[batch_index, :, 4:5]
            classes = predictions[batch_index, :, 5:]
            foreground, matched_gt, matched_iou = self._assign(
                boxes[batch_index], objectness, classes, grid, strides, target
            )
            object_target = foreground[:, None].to(predictions.dtype)
            obj_loss = obj_loss + F.binary_cross_entropy_with_logits(
                objectness, object_target, reduction="sum"
            )
            count = int(foreground.sum())
            positive_count += count
            if count == 0:
                continue
            gt_boxes = target["boxes"].to(boxes.device)[matched_gt]
            gt_labels = target["labels"].to(boxes.device).long()[matched_gt]
            iou = _aligned_iou(boxes[batch_index, foreground], gt_boxes)
            box_loss = box_loss + (1.0 - iou.square()).sum()
            class_target = F.one_hot(gt_labels, self.num_classes).to(predictions.dtype)
            class_target = class_target * matched_iou[:, None]
            cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
                classes[foreground], class_target, reduction="sum"
            )
        normalizer = max(1, positive_count)
        box_loss = 5.0 * box_loss / normalizer
        obj_loss = obj_loss / normalizer
        cls_loss = cls_loss / normalizer
        return {
            "loss": box_loss + obj_loss + cls_loss,
            "loss_box": box_loss.detach(),
            "loss_objectness": obj_loss.detach(),
            "loss_classification": cls_loss.detach(),
            "positive_anchors": predictions.new_tensor(float(positive_count)),
        }

    def postprocess(
        self,
        predictions: Tensor,
        boxes: Tensor,
        *,
        image_size: tuple[int, int],
    ) -> list[dict[str, Tensor]]:
        results: list[dict[str, Tensor]] = []
        height, width = image_size
        for prediction, image_boxes in zip(predictions, boxes):
            image_boxes = image_boxes.clone()
            image_boxes[:, 0::2].clamp_(0, width - 1)
            image_boxes[:, 1::2].clamp_(0, height - 1)
            objectness = prediction[:, 4].sigmoid()
            class_probability, labels = prediction[:, 5:].sigmoid().max(dim=1)
            scores = objectness * class_probability
            keep = scores >= self.confidence_threshold
            selected_boxes = image_boxes[keep]
            selected_scores = scores[keep]
            selected_labels = labels[keep]
            if len(selected_boxes):
                keep_nms = batched_nms(
                    selected_boxes, selected_scores, selected_labels, self.nms_threshold
                )[: self.max_detections]
                selected_boxes = selected_boxes[keep_nms]
                selected_scores = selected_scores[keep_nms]
                selected_labels = selected_labels[keep_nms]
            results.append(
                {
                    "boxes": selected_boxes,
                    "scores": selected_scores,
                    "labels": selected_labels,
                }
            )
        return results


__all__ = ["EventStateYOLOX"]
