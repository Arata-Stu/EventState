"""Paired, sequence-fixed PCA of Semantic inference readouts and their RGB video panels."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import random
import subprocess

from .placement_features import BLOCKS, CONDITIONS


def calibration_indices(frame_count, limit=32):
    """Index the complete input timeline uniformly, including both endpoints."""
    if frame_count < 1 or limit < 2:
        raise ValueError("Expected a positive sequence length and at least two calibration slots")
    count = min(frame_count, limit)
    if count == 1:
        return (0,)
    return tuple(i * (frame_count - 1) // (count - 1) for i in range(count))


def _check_maps(maps):
    if set(maps) != set(CONDITIONS) or any(set(stages) != set(BLOCKS) for stages in maps.values()):
        raise ValueError("Expected both placement models at blocks 3, 6, 9 and 12")
    shapes = {tuple(value.shape) for stages in maps.values() for value in stages.values()}
    if len(shapes) != 1:
        raise ValueError("Both models and all stages must share the patch grid and channel count")
    shape = shapes.pop()
    if len(shape) != 3 or min(shape) < 1 or shape[2] < 3:
        raise ValueError(f"Invalid [patch rows, patch columns, channels] geometry: {shape}")
    return shape


class FeatureCalibration:
    """Hold only the same sparse patch/time samples from each model on CPU."""

    def __init__(self, tokens_per_frame=20):
        if tokens_per_frame < 1:
            raise ValueError("Expected at least one calibration patch per frame")
        self.tokens_per_frame = tokens_per_frame
        self.shape = None
        self.samples = {block: [] for block in BLOCKS}
        self.sampling = []

    def add(self, maps, frame_index):
        import torch

        shape = _check_maps(maps)
        if self.shape is not None and self.shape != shape:
            raise ValueError("Patch geometry changed during calibration")
        if frame_index < 0 or (self.sampling and frame_index <= self.sampling[-1]["frame_index"]):
            raise ValueError("Expected calibration frames once each, in source order")
        self.shape = shape
        rows, columns, channels = shape
        patches = rows * columns
        indices = sorted(random.Random(frame_index).sample(range(patches), min(self.tokens_per_frame, patches)))
        self.sampling.append(dict(frame_index=frame_index, patch_indices=indices))
        reference = maps[next(iter(CONDITIONS))][BLOCKS[0]]
        index = torch.tensor(indices, device=reference.device)
        for block in BLOCKS:
            sampled = [maps[key][block].reshape(patches, channels).index_select(0, index) for key in CONDITIONS]
            self.samples[block].append(torch.cat(sampled).detach().to(device="cpu", dtype=torch.float32))

    def fit(self):
        import torch
        import torch.nn.functional as F

        if not self.sampling:
            raise ValueError("No calibration frame supplied")
        projections = {}
        for block, samples in self.samples.items():
            values = F.normalize(torch.cat(samples).double(), dim=-1)
            if len(values) < 3 or not torch.isfinite(values).all():
                raise ValueError(f"Insufficient or nonfinite PCA samples at block {block}")
            mean = values.mean(0)
            centered = values - mean
            # Full eigendecomposition of the small channel covariance avoids a random PCA initialization.
            variance, vectors = torch.linalg.eigh(centered.T @ centered)
            basis = vectors[:, -3:].flip(1)
            largest_loadings = basis[basis.abs().argmax(0), torch.arange(3)]
            basis *= largest_loadings.sign()
            bounds = torch.quantile(centered @ basis, values.new_tensor([.01, .99]), dim=0)
            nonnegative = variance.clamp_min(0)
            fractions = nonnegative[-3:].flip(0) / nonnegative.sum().clamp_min(1e-30)
            projections[str(block)] = dict(mean=mean.tolist(), basis=basis.tolist(), low=bounds[0].tolist(),
                                           high=bounds[1].tolist(), explained_variance_ratio=fractions.tolist())
        return FeatureProjection(dict(version=1, input="L2-normalized per patch", clip_percentiles=[1, 99],
                                      conditions=list(CONDITIONS), feature_shape=list(self.shape),
                                      sample_frames=self.sampling, blocks=projections))


class FeatureProjection:
    """Apply a fixed, joint RGB transform per block. Only projected RGB is transferred from GPU."""

    def __init__(self, metadata):
        self.metadata = metadata
        self._devices = {}

    def to_dict(self):
        return self.metadata

    @classmethod
    def from_dict(cls, metadata):
        if (metadata["version"] != 1 or metadata["conditions"] != list(CONDITIONS)
                or set(metadata["blocks"]) != {str(block) for block in BLOCKS}):
            raise ValueError("Incompatible placement projection")
        return cls(metadata)

    def _constants(self, device):
        import torch

        if device not in self._devices:
            rows = [self.metadata["blocks"][str(block)] for _ in CONDITIONS for block in BLOCKS]
            def array(key):
                return torch.tensor([row[key] for row in rows], dtype=torch.float32, device=device)
            basis = array("basis")
            offset = torch.bmm(array("mean").unsqueeze(1), basis)
            low, high = array("low").unsqueeze(1), array("high").unsqueeze(1)
            self._devices[device] = basis, offset + low, high - low
        return self._devices[device]

    def render(self, maps):
        import torch
        import torch.nn.functional as F

        shape = _check_maps(maps)
        if list(shape) != self.metadata["feature_shape"]:
            raise ValueError("Feature geometry differs from the calibrated sequence")
        values = torch.stack([maps[key][block] for key in CONDITIONS for block in BLOCKS]).flatten(1, 2)
        basis, offset, span = self._constants(values.device)
        projected = torch.bmm(F.normalize(values.float(), dim=-1), basis)
        if not torch.isfinite(projected).all():
            raise ValueError("Nonfinite feature projection")
        scaled = (projected - offset) / span.clamp_min(1e-8)
        scaled = torch.where(span > 1e-8, scaled, 0.5)  # Constant components have neutral intensity.
        rgb = (255 * scaled.clamp(0, 1)).round().to(torch.uint8).reshape(
            len(CONDITIONS), len(BLOCKS), shape[0], shape[1], 3).cpu().numpy()
        return {key: {block: rgb[i, j] for j, block in enumerate(BLOCKS)} for i, key in enumerate(CONDITIONS)}


_GEOMETRY = {"dsec": (640, 440), "m3ed": (640, 352)}
_TILE_WIDTH = 320
_TITLE_HEIGHT = 42
_ROW_HEADER = 54
_FOOTER = 40
_MASK_WIDTH = 480
_MASK_HEADER = 36


def panel_size(dataset, *, semantic=False):
    height = _GEOMETRY[dataset][1]
    mask_row = _MASK_HEADER + height * 3 // 4 if semantic else 0
    return _TILE_WIDTH * 5, _TITLE_HEIGHT + mask_row + 2 * (_ROW_HEADER + height // 2) + _FOOTER


def select_panel_font(requested=None):
    """Return a readable Japanese font, or select English panels by returning None."""
    from tools.render_activity_comparison import cjk_font

    if requested is not None:
        path = Path(requested)
        if not path.is_file():
            raise FileNotFoundError(f"Requested panel font does not exist: {path}")
        font = cjk_font(path)
        if font.resolve() != path.resolve():
            raise ValueError(f"Requested panel font lacks Japanese glyphs: {path}")
        return font
    try:
        return cjk_font()
    except (OSError, subprocess.CalledProcessError):
        return None


@lru_cache(maxsize=4)
def _fonts(font_path):
    from PIL import ImageFont

    if font_path:
        return ImageFont.truetype(font_path, 21), ImageFont.truetype(font_path, 16)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 21), ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        try:
            return ImageFont.load_default(size=21), ImageFont.load_default(size=16)
        except TypeError:  # Pillow 10.0 has only the built-in fixed-size ASCII font.
            return ImageFont.load_default(), ImageFont.load_default()


@lru_cache(maxsize=1)
def _class_palette():
    import numpy as np
    from tools.render_activity_comparison import semantic_color
    from tools.visualize_dsec_semantic_sequences import DSEC_SEMANTIC_11_CLASSES

    # Keep predictions, pseudo-labels, ignored pixels and legend on the evaluation palette.
    rgb = semantic_color(np.arange(256, dtype=np.uint8))
    return rgb.tobytes(), DSEC_SEMANTIC_11_CLASSES


@lru_cache(maxsize=16)
def _backdrop(dataset, sequence, font_path, has_rgb, semantic):
    from PIL import Image, ImageDraw

    main, small = _fonts(font_path)
    canvas = Image.new("RGB", panel_size(dataset, semantic=semantic), "#101b20")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 5), f"{dataset.upper()}   {sequence}", font=main, fill="#f3f8fa")
    tile_height = _GEOMETRY[dataset][1] // 2
    feature_top = _TITLE_HEIGHT
    if semantic:
        mask_height = _GEOMETRY[dataset][1] * 3 // 4
        if font_path:
            target_name = "DSEC 正解ラベル" if dataset == "dsec" else "M3ED 疑似ラベル（InternImage）"
            headings = [target_name, "E1  12層後 ｜ Semantic 予測", "E1_b3_6_9_12  4段 ｜ Semantic 予測",
                        "クラス色"]
        else:
            target_name = "DSEC ground truth" if dataset == "dsec" else "M3ED pseudo-label (InternImage)"
            headings = [target_name, "E1  Block 12 | Semantic prediction", "E1_b3_6_9_12  Four stages | Semantic prediction",
                        "Mask colors"]
        for col, heading in enumerate(headings):
            draw.text((col * _MASK_WIDTH + 8, _TITLE_HEIGHT + 5), heading, font=small, fill="#f1f8fa")
        palette, names = _class_palette()
        legend = [*enumerate(names), (255, "除外画素" if font_path else "ignore")]
        x = 3 * _MASK_WIDTH + 8
        pitch = min(25, mask_height // len(legend))
        for i, (class_id, name) in enumerate(legend):
            y = _TITLE_HEIGHT + _MASK_HEADER + i * pitch
            color = tuple(palette[3 * class_id:3 * class_id + 3])
            draw.rectangle((x, y + 4, x + 15, y + 17), fill=color, outline="#98a5a8")
            draw.text((x + 22, y), name.replace("_", " "), font=small, fill="#f1f8fa")
        feature_top += _MASK_HEADER + mask_height
    for i, (model, label) in enumerate(CONDITIONS.items()):
        y = feature_top + i * (_ROW_HEADER + tile_height)
        draw.rectangle((0, y, canvas.width, y + _ROW_HEADER - 1), fill="#1a3039")
        if not font_path:
            layers = "12" if i == 0 else "3, 6, 9, 12"
            label = f"{model}   Encoder + Patchwise LSTM (h distillation)   |   after blocks {layers}"
        draw.text((_TILE_WIDTH + 8, y), label, font=main, fill="#f1f8fa")
        reference = ("参照 RGB" if has_rgb else "参照 RGB  未取得") if i == 0 else "入力イベント  GEP"
        if not font_path:
            reference = ("RGB reference" if has_rgb else "RGB unavailable") if i == 0 else "GEP event input"
        draw.text((9, y + 25), reference, font=small, fill="#d7e4e8")
        for j, block in enumerate(BLOCKS, 1):
            vit = model == "E1" and block < 12
            if font_path:
                title = f"{block}層後   " + ("ViT 特徴" if vit else "LSTM 出力 h")
            else:
                title = f"Block {block}   " + ("ViT features" if vit else "LSTM output h")
            draw.text((j * _TILE_WIDTH + 8, y + 26), title, font=small, fill="#d7e4e8")
    legend = ("特徴パネルの色は主成分（RGB＝PC1・2・3）。同じ系列の同じ列で両モデル共通。"
              "各マスは入力16画素間隔のパッチ。" if font_path else
              "Feature panels: RGB = PCs 1, 2, 3. Both rows share one color projection per block for this recording. "
              "Each patch spans 16 input pixels.")
    draw.text((12, canvas.height - 32), legend, font=small, fill="#d7e4e8")
    return canvas


def draw_feature_frame(colors, *, dataset, sequence, timestamp_us, elapsed_s, event_rate_hz,
                       events, rgb=None, font_path=None, predictions=None, target=None):
    """Draw aligned inputs, feature grids and optional evaluated Semantic outputs."""
    from PIL import Image, ImageDraw

    width, height = _GEOMETRY[dataset]
    tile_size = (_TILE_WIDTH, height // 2)
    canvas = _backdrop(dataset, sequence, str(font_path) if font_path else None,
                       rgb is not None, predictions is not None).copy()
    _, small = _fonts(str(font_path) if font_path else None)
    stamp = f"元時刻 {timestamp_us} µs" if font_path else f"source {timestamp_us} us"
    timing = f"t = {elapsed_s:.3f} s     {stamp}     {event_rate_hz/1e6:.3f} M events/s"
    draw = ImageDraw.Draw(canvas)
    draw.text((canvas.width - 12 - draw.textlength(timing, font=small), 10), timing, font=small, fill="#c3e2e6")
    feature_top = _TITLE_HEIGHT
    if predictions is not None:
        import numpy as np

        if predictions.keys() != CONDITIONS.keys():
            raise ValueError("Expected a Semantic prediction from both placement models")
        mask_height = height * 3 // 4
        top = _TITLE_HEIGHT + _MASK_HEADER
        palette, names = _class_palette()
        for col, labels in enumerate([target, *(predictions[key] for key in CONDITIONS)]):
            x = col * _MASK_WIDTH
            if col == 0 and labels is None:
                missing = "この時刻のラベルなし" if font_path else "No label at this timestamp"
                draw.text((_MASK_WIDTH // 2, top + mask_height // 2), missing, anchor="mm", font=small, fill="#b9c7ca")
                continue
            labels = np.asarray(labels)
            if labels.shape != (height, width) or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError(f"Invalid Semantic mask geometry or dtype: {labels.shape}, {labels.dtype}")
            invalid = (labels < 0) | (labels >= len(names))
            if col == 0:
                invalid &= labels != 255
            if invalid.any():
                raise ValueError("Unexpected Semantic class ID in class mask")
            tile = Image.fromarray(labels.astype(np.uint8, copy=False)).resize(
                (_MASK_WIDTH, mask_height), Image.Resampling.NEAREST)
            tile.putpalette(palette)
            canvas.paste(tile.convert("RGB"), (x, top))
        feature_top = top + mask_height
    patch_height = (height + 15) // 16
    for i, key in enumerate(CONDITIONS):
        y = feature_top + (i + 1) * _ROW_HEADER + i * tile_size[1]
        reference = rgb if i == 0 else events
        if reference is not None:
            source = Image.fromarray(reference)
            if source.mode != "RGB" or source.size not in ((width, height), (width, patch_height * 16)):
                raise ValueError(f"Reference image and Semantic feature geometry differ: {source.size}")
            source = source.crop((0, 0, width, height)).resize(tile_size, Image.Resampling.BILINEAR)
            canvas.paste(source, (0, y))
        for j, block in enumerate(BLOCKS, 1):
            grid = Image.fromarray(colors[key][block])
            if grid.mode != "RGB" or grid.size != (width // 16, patch_height):
                raise ValueError(f"Invalid stride-16 grid for {dataset}, {key}, block {block}: {grid.size}")
            tile = grid.resize((_TILE_WIDTH, patch_height * 8), Image.Resampling.NEAREST)
            canvas.paste(tile.crop((0, 0, *tile_size)), (j * _TILE_WIDTH, y))
    return canvas
