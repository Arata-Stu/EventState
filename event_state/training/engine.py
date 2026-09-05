"""Step-based Phase 0/1 training and dense-alignment evaluation."""

from __future__ import annotations

import contextlib
import math
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .checkpoint import load_checkpoint, save_checkpoint
from .data import CyclingDataIterator
from .logging import TrainingLogger
from .metrics import EventCountAnalysis, WeightedMean, frame_cosine, global_gradient_norm
from .optim import WarmupCosineScheduler


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


class EventStateTrainer:
    """Train E0/E1/E2 without placing the frozen teacher inside the student model."""

    def __init__(
        self,
        *,
        model: nn.Module,
        teacher: nn.Module | None,
        train_loader: Any | None,
        validation_loader: Any,
        optimizer: torch.optim.Optimizer | None,
        scheduler: WarmupCosineScheduler | None,
        h_distillation_loss: nn.Module,
        z_distillation_loss: nn.Module,
        config: Any,
        device: torch.device,
        logger: TrainingLogger,
        checkpoint_config: Any,
    ) -> None:
        self.model = model
        self.teacher = teacher
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.h_distillation_loss = h_distillation_loss
        self.z_distillation_loss = z_distillation_loss
        self.config = config
        self.training_config = _value(config, "training")
        self.loss_config = _value(config, "loss")
        self.device = device
        self.logger = logger
        self.checkpoint_config = checkpoint_config
        self.output_directory = Path(str(_value(_value(config, "output"), "directory")))
        self.checkpoint_directory = self.output_directory / "checkpoints"
        self.global_step = 0
        self.best_validation_loss = float("inf")
        self._restored_data_state: dict[str, Any] = {}

        self.precision = str(_value(self.training_config, "precision", "bf16")).lower()
        if self.precision not in {"bf16", "fp16", "float16", "fp32", "float32"}:
            raise ValueError(f"Unsupported training precision: {self.precision}")
        self.amp_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        self.amp_enabled = self.precision in {"bf16", "fp16", "float16"} and (
            self.device.type == "cuda" or (self.device.type == "cpu" and self.precision == "bf16")
        )
        self.actual_precision = self.precision if self.amp_enabled else "fp32"
        self.effective_precision = self.actual_precision
        if self.precision in {"bf16", "fp16", "float16"} and not self.amp_enabled:
            warnings.warn(
                f"{self.precision} autocast is unavailable for device {self.device.type}; "
                "falling back to fp32",
                RuntimeWarning,
                stacklevel=2,
            )
        log_text = getattr(self.logger, "text", None)
        if callable(log_text):
            log_text(
                "run/precision",
                f"requested={self.precision}, actual={self.actual_precision}, "
                f"device={self.device.type}",
                0,
            )
        self.scaler = self._make_scaler()

        if self.teacher is None and not bool(
            _value(_value(config, "teacher"), "cache_features", False)
        ):
            raise ValueError("Online teacher mode requires a FrozenDinoV3Teacher instance")
        if self.teacher is not None:
            self.teacher.eval()
            trainable_teacher = [
                name
                for name, parameter in self.teacher.named_parameters()
                if parameter.requires_grad
            ]
            if trainable_teacher:
                raise ValueError(
                    "The teacher must be frozen; trainable parameters include "
                    + ", ".join(trainable_teacher[:5])
                )

    def _make_scaler(self) -> Any:
        enabled = (
            self.amp_enabled
            and self.amp_dtype == torch.float16
            and self.device.type == "cuda"
        )
        if not enabled:
            return None
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):  # Compatible with the oldest supported torch.
            return torch.cuda.amp.GradScaler(enabled=True)

    def _autocast(self):
        if not self.amp_enabled:
            return contextlib.nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=True,
        )

    def resume(self, path: str | Path) -> None:
        if self.optimizer is None or self.scheduler is None:
            raise RuntimeError("Resume requires training optimizer and scheduler components")
        state = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
            restore_rng=True,
            expected_config=self.checkpoint_config,
            compatibility_mode="resume",
        )
        self.global_step = state.global_step
        self.best_validation_loss = state.best_validation_loss
        self._restored_data_state = state.data_state
        self.logger.text("run/resume", str(Path(path).expanduser()), self.global_step)

    def _move_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        non_blocking = self.device.type == "cuda"
        moved = dict(batch)
        for key in ("events", "images", "teacher_features", "event_counts", "timestamps"):
            if key in moved and isinstance(moved[key], Tensor):
                moved[key] = moved[key].to(self.device, non_blocking=non_blocking)
        return moved

    def _teacher_features(self, batch: Mapping[str, Any]) -> Tensor:
        if "teacher_features" in batch:
            features = batch["teacher_features"]
            if features.ndim != 4:
                raise ValueError(
                    "Cached teacher_features must have shape [B, T, N, D], "
                    f"got {features.shape}"
                )
            return features.detach()
        if self.teacher is None:
            raise RuntimeError("Neither cached features nor an online teacher are available")
        images = batch.get("images")
        if not isinstance(images, Tensor) or images.ndim != 5:
            raise ValueError("Online teacher mode requires images with shape [B, T, 3, H, W]")
        batch_size, sequence_length = images.shape[:2]
        flat_images = images.flatten(0, 1)
        self.teacher.eval()
        # Keep online targets bitwise-comparable in dtype policy with the
        # offline cache generator, which runs the frozen teacher in FP32.
        with torch.no_grad():
            features = self.teacher(flat_images.float())
        if not isinstance(features, Tensor) or features.ndim != 3:
            raise ValueError("FrozenDinoV3Teacher must return patch tokens [B*T, N, D]")
        return features.reshape(batch_size, sequence_length, *features.shape[1:]).detach()

    @staticmethod
    def _loss_components(
        loss_module: nn.Module, prediction: Tensor, target: Tensor
    ) -> dict[str, Tensor]:
        prediction = prediction.float()
        target = target.detach().float()
        if hasattr(loss_module, "components"):
            components = loss_module.components(prediction, target)
            if not isinstance(components, Mapping) or "loss" not in components:
                raise TypeError(
                    "DistillationLoss.components() must return a mapping containing loss"
                )
            return dict(components)
        return {"loss": loss_module(prediction, target)}

    @staticmethod
    def _detach_state(state: Any) -> Any:
        if isinstance(state, Tensor):
            return state.detach()
        if isinstance(state, tuple):
            return tuple(EventStateTrainer._detach_state(item) for item in state)
        if isinstance(state, list):
            return [EventStateTrainer._detach_state(item) for item in state]
        if isinstance(state, Mapping):
            return {key: EventStateTrainer._detach_state(value) for key, value in state.items()}
        return state

    def _forward_student(
        self,
        events: Tensor,
        state: Any = None,
    ) -> Mapping[str, Any]:
        temporal_config = _value(_value(self.config, "model"), "temporal")
        detach_every_value = _value(temporal_config, "detach_state_every", None)
        if detach_every_value is None:
            return self.model.forward_sequence(events, state=state)
        detach_every = int(detach_every_value)
        if detach_every <= 0:
            raise ValueError("model.temporal.detach_state_every must be positive or null")

        current_state = state
        z_chunks: list[Tensor] = []
        h_chunks: list[Tensor] = []
        for start in range(0, events.shape[1], detach_every):
            chunk = self.model.forward_sequence(
                events[:, start : start + detach_every], state=current_state
            )
            z_chunks.append(chunk["z"])
            h_chunks.append(chunk["h"])
            current_state = self._detach_state(chunk.get("state"))
        return {
            "z": torch.cat(z_chunks, dim=1),
            "h": torch.cat(h_chunks, dim=1),
            "state": current_state,
        }

    def _forward_objectives(
        self,
        batch: Mapping[str, Any],
        *,
        state: Any = None,
    ) -> dict[str, Any]:
        events = batch.get("events")
        if not isinstance(events, Tensor) or events.ndim != 5:
            raise ValueError("events must have shape [B, T, C, H, W]")
        teacher_features = self._teacher_features(batch)
        h_config = _value(self.loss_config, "h_distill")
        z_config = _value(self.loss_config, "z_objective")
        h_enabled = bool(_value(h_config, "enabled", False))
        z_type = str(_value(z_config, "type", "none")).lower()
        z_enabled = z_type == "direct_dino"
        if z_type not in {"none", "direct_dino"}:
            raise NotImplementedError(
                f"z objective {z_type!r} is reserved for a later phase; use none or direct_dino"
            )
        if not h_enabled and not z_enabled:
            raise ValueError("At least one distillation branch must be enabled")

        with self._autocast():
            # Every randomly sampled clip starts from an empty state. Since every sample in
            # DSECSequenceDataset belongs to exactly one sequence, state cannot cross a boundary.
            outputs = self._forward_student(events, state=state)
            if not isinstance(outputs, Mapping) or "z" not in outputs or "h" not in outputs:
                raise TypeError("EventStateModel.forward_sequence() must return z and h tensors")
            z = outputs["z"]
            h = outputs["h"]
            z_prediction = self.model.project_z(z) if z_enabled else None
            h_prediction = self.model.project_h(h) if h_enabled else None

        # Disabled objectives remain visible as diagnostics, but their
        # projectors must not receive zero gradients (AdamW would otherwise
        # decay parameters belonging to an intentionally inactive branch).
        with torch.no_grad(), self._autocast():
            if z_prediction is None:
                z_prediction = self.model.project_z(z.detach())
            if h_prediction is None:
                h_prediction = self.model.project_h(h.detach())
        for name, tensor in (
            ("z", z),
            ("h", h),
            ("z_prediction", z_prediction),
            ("h_prediction", h_prediction),
        ):
            if not isinstance(tensor, Tensor) or tensor.ndim != 4:
                raise ValueError(f"{name} must have shape [B, T, N, D]")
        if z_prediction.shape != teacher_features.shape:
            raise ValueError(
                f"z projection {z_prediction.shape} does not match teacher {teacher_features.shape}"
            )
        if h_prediction.shape != teacher_features.shape:
            raise ValueError(
                f"h projection {h_prediction.shape} does not match teacher {teacher_features.shape}"
            )

        if h_enabled:
            h_components = self._loss_components(
                self.h_distillation_loss, h_prediction, teacher_features
            )
        else:
            with torch.no_grad():
                h_components = self._loss_components(
                    self.h_distillation_loss, h_prediction, teacher_features
                )
        if z_enabled:
            z_components = self._loss_components(
                self.z_distillation_loss, z_prediction, teacher_features
            )
        else:
            with torch.no_grad():
                z_components = self._loss_components(
                    self.z_distillation_loss, z_prediction, teacher_features
                )

        active_losses: list[Tensor] = []
        if h_enabled:
            active_losses.append(h_components["loss"])
        if z_enabled:
            active_losses.append(
                float(_value(z_config, "weight", 1.0)) * z_components["loss"]
            )
        total_loss = torch.stack(active_losses).sum()

        z_diagnostic = z if z.shape == teacher_features.shape else z_prediction
        h_diagnostic = h if h.shape == teacher_features.shape else h_prediction
        with torch.no_grad():
            z_frame_cosine = frame_cosine(z_diagnostic.detach(), teacher_features)
            h_frame_cosine = frame_cosine(h_diagnostic.detach(), teacher_features)
            z_projected_frame_cosine = frame_cosine(
                z_prediction.detach(), teacher_features
            )
            h_projected_frame_cosine = frame_cosine(
                h_prediction.detach(), teacher_features
            )
        return {
            "loss": total_loss,
            "z": z,
            "h": h,
            "z_prediction": z_prediction,
            "h_prediction": h_prediction,
            "teacher": teacher_features,
            "state": outputs.get("state"),
            "z_components": z_components,
            "h_components": h_components,
            "z_frame_cosine": z_frame_cosine,
            "h_frame_cosine": h_frame_cosine,
            "z_projected_frame_cosine": z_projected_frame_cosine,
            "h_projected_frame_cosine": h_projected_frame_cosine,
        }

    def _component_gradient_norms(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for attribute, label in (
            ("event_encoder", "event_encoder"),
            ("temporal_model", "temporal"),
            ("h_projector", "h_projector"),
            ("z_projector", "z_projector"),
        ):
            module = getattr(self.model, attribute, None)
            if isinstance(module, nn.Module):
                values[label] = global_gradient_norm(module.parameters())
        return values

    def train_optimizer_step(self, iterator: CyclingDataIterator) -> dict[str, float]:
        if self.optimizer is None or self.scheduler is None:
            raise RuntimeError("Training requires optimizer and scheduler components")
        self.model.train()
        if self.teacher is not None:
            self.teacher.eval()
        accumulation = int(_value(self.training_config, "gradient_accumulation", 1))
        if accumulation <= 0:
            raise ValueError("training.gradient_accumulation must be positive")
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        start_time = time.perf_counter()
        meters: dict[str, WeightedMean] = {}
        samples = 0
        frames = 0

        for _ in range(accumulation):
            batch = self._move_batch(next(iterator))
            result = self._forward_objectives(batch)
            scaled_loss = result["loss"] / accumulation
            if self.scaler is None:
                scaled_loss.backward()
            else:
                self.scaler.scale(scaled_loss).backward()

            batch_size, sequence_length = batch["events"].shape[:2]
            samples += batch_size
            frames += batch_size * sequence_length
            batch_metrics = {
                "loss": result["loss"],
                "h_distill_loss": result["h_components"]["loss"],
                "z_distill_loss": result["z_components"]["loss"],
                "h_cosine": result["h_frame_cosine"].mean(),
                "z_cosine": result["z_frame_cosine"].mean(),
                "h_projected_cosine": result["h_projected_frame_cosine"].mean(),
                "z_projected_cosine": result["z_projected_frame_cosine"].mean(),
            }
            for branch in ("h", "z"):
                for name, value in result[f"{branch}_components"].items():
                    if name != "loss":
                        batch_metrics[f"{branch}_{name}_loss"] = value
            if "event_counts" in batch:
                valid_counts = batch["event_counts"][batch["event_counts"] >= 0]
                if valid_counts.numel():
                    batch_metrics["event_count"] = valid_counts.float().mean()
            for name, value in batch_metrics.items():
                meters.setdefault(name, WeightedMean()).update(value, batch_size)

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        gradient_norms = self._component_gradient_norms()
        total_gradient_norm = global_gradient_norm(self.model.parameters())
        clip_value = float(_value(self.training_config, "gradient_clip", 0.0))
        if clip_value > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
        learning_rates = [float(group["lr"]) for group in self.optimizer.param_groups]
        optimizer_updated = True
        if self.scaler is None:
            self.optimizer.step()
        else:
            scale_before = float(self.scaler.get_scale())
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_updated = float(self.scaler.get_scale()) >= scale_before
        if optimizer_updated:
            self.scheduler.step()
            self.global_step += 1

        elapsed = max(time.perf_counter() - start_time, 1e-9)
        metrics = {name: meter.mean for name, meter in meters.items()}
        metrics.update(
            {
                "grad_norm": total_gradient_norm,
                "optimizer_step_skipped": float(not optimizer_updated),
                "samples_per_second": samples / elapsed,
                "frames_per_second": frames / elapsed,
            }
        )
        for name, norm in gradient_norms.items():
            metrics[f"grad_norm/{name}"] = norm
        for index, (group, learning_rate) in enumerate(
            zip(self.optimizer.param_groups, learning_rates)
        ):
            group_name = str(group.get("group_name", index))
            metrics[f"learning_rate/{group_name}"] = learning_rate
        if self.device.type == "cuda":
            metrics["gpu_memory_mb"] = torch.cuda.max_memory_allocated(self.device) / 2**20
        return metrics

    @torch.inference_mode()
    def validate(self, *, max_batches: int | None = None) -> dict[str, float]:
        self.model.eval()
        if self.teacher is not None:
            self.teacher.eval()
        meters: dict[str, WeightedMean] = {}
        event_analysis = EventCountAnalysis()
        seen_batches = 0
        temporal_state = None
        active_sequence: str | None = None
        for batch in self.validation_loader:
            if max_batches is not None and seen_batches >= max_batches:
                break
            batch = self._move_batch(batch)
            batch_size = int(batch["events"].shape[0])
            frame_weight = batch_size * int(batch["events"].shape[1])
            if batch_size != 1:
                raise ValueError(
                    "Streaming validation requires validation batch_size=1"
                )
            sequence_field = batch.get("sequence_name")
            if not isinstance(sequence_field, (list, tuple)) or len(sequence_field) != 1:
                raise ValueError("Validation batches must contain one sequence_name")
            sequence_name = str(sequence_field[0])
            start_field = batch.get("is_sequence_start")
            end_field = batch.get("is_sequence_end")
            if not isinstance(start_field, Tensor) or not isinstance(end_field, Tensor):
                raise ValueError("Validation batches must contain sequence boundary flags")
            is_sequence_start = bool(start_field.item())
            is_sequence_end = bool(end_field.item())
            if active_sequence != sequence_name:
                if not is_sequence_start:
                    raise ValueError(
                        f"Sequence {sequence_name!r} did not begin at its first validation clip"
                    )
                temporal_state = None
                active_sequence = sequence_name
            elif is_sequence_start:
                raise ValueError(f"Sequence {sequence_name!r} restarted without a boundary")

            result = self._forward_objectives(batch, state=temporal_state)
            temporal_state = result["state"]
            values = {
                "loss": result["loss"],
                "h_distill_loss": result["h_components"]["loss"],
                "z_distill_loss": result["z_components"]["loss"],
                "h_cosine": result["h_frame_cosine"].mean(),
                "z_cosine": result["z_frame_cosine"].mean(),
                "h_projected_cosine": result["h_projected_frame_cosine"].mean(),
                "z_projected_cosine": result["z_projected_frame_cosine"].mean(),
            }
            for branch in ("h", "z"):
                for name, value in result[f"{branch}_components"].items():
                    if name != "loss":
                        values[f"{branch}_{name}_loss"] = value
            for name, value in values.items():
                meters.setdefault(name, WeightedMean()).update(value, frame_weight)
            if "event_counts" in batch:
                event_analysis.update(
                    batch["event_counts"],
                    result["z_frame_cosine"],
                    result["h_frame_cosine"],
                    result["z_projected_frame_cosine"],
                    result["h_projected_frame_cosine"],
                )
            if is_sequence_end:
                temporal_state = None
                active_sequence = None
            seen_batches += 1
        if seen_batches == 0:
            raise RuntimeError("Validation dataloader produced no batches")

        metrics = {f"val/{name}": meter.mean for name, meter in meters.items()}
        threshold_config = _value(_value(self.config, "evaluation", {}), "event_count_thresholds")
        thresholds = None
        if threshold_config is not None:
            if len(threshold_config) != 2:
                raise ValueError("evaluation.event_count_thresholds must contain two values")
            thresholds = (float(threshold_config[0]), float(threshold_config[1]))
        metrics.update(
            {f"val/{name}": value for name, value in event_analysis.compute(thresholds).items()}
        )
        return metrics

    def _save(self, path: Path, data_state: Mapping[str, Any]) -> Path:
        if self.optimizer is None or self.scheduler is None:
            raise RuntimeError("Checkpoint saving requires optimizer and scheduler components")
        return save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            global_step=self.global_step,
            best_validation_loss=self.best_validation_loss,
            data_state=data_state,
            config=self.checkpoint_config,
        )

    def fit(self) -> None:
        if self.train_loader is None:
            raise RuntimeError("Training requires a train dataloader")
        if self.optimizer is None or self.scheduler is None:
            raise RuntimeError("Training requires optimizer and scheduler components")
        max_steps = int(_value(self.training_config, "max_steps"))
        if self.global_step >= max_steps:
            print(
                f"Checkpoint is already at step {self.global_step}, "
                f"max_steps={max_steps}; nothing to do.",
                flush=True,
            )
            return
        iterator = CyclingDataIterator(
            self.train_loader,
            seed=int(_value(self.config, "seed", 0)),
            state=self._restored_data_state,
        )
        log_every = int(_value(self.training_config, "log_every", 20))
        validate_every = int(_value(self.training_config, "validate_every", 1000))
        checkpoint_every = int(_value(self.training_config, "checkpoint_every", 1000))
        validation_batches_value = _value(self.training_config, "validation_batches", None)
        validation_batches = (
            None if validation_batches_value is None else int(validation_batches_value)
        )
        last_validation_step = -1
        last_checkpoint_step = -1

        while self.global_step < max_steps:
            metrics = self.train_optimizer_step(iterator)
            if metrics.get("optimizer_step_skipped", 0.0):
                overflow = {
                    "train/optimizer_step_skipped": metrics["optimizer_step_skipped"],
                }
                self.logger.scalars(overflow, self.global_step)
                self.logger.console("train_overflow", self.global_step, overflow)
                self.logger.flush()
                continue
            if self.global_step % log_every == 0 or self.global_step == 1:
                logged = {f"train/{name}": value for name, value in metrics.items()}
                self.logger.scalars(logged, self.global_step)
                self.logger.console("train", self.global_step, metrics)
            if self.global_step % validate_every == 0 or self.global_step == max_steps:
                validation = self.validate(max_batches=validation_batches)
                self.logger.scalars(validation, self.global_step)
                self.logger.console("validation", self.global_step, validation)
                validation_loss = validation["val/loss"]
                if math.isfinite(validation_loss) and validation_loss < self.best_validation_loss:
                    self.best_validation_loss = validation_loss
                    self._save(self.checkpoint_directory / "best.pt", iterator.state_dict())
                last_validation_step = self.global_step
            if self.global_step % checkpoint_every == 0 or self.global_step == max_steps:
                self._save(
                    self.checkpoint_directory / f"step_{self.global_step:08d}.pt",
                    iterator.state_dict(),
                )
                last_checkpoint_step = self.global_step
            self.logger.flush()

        if last_validation_step != self.global_step:
            validation = self.validate(max_batches=validation_batches)
            self.logger.scalars(validation, self.global_step)
        if last_checkpoint_step != self.global_step:
            self._save(
                self.checkpoint_directory / f"step_{self.global_step:08d}.pt",
                iterator.state_dict(),
            )
