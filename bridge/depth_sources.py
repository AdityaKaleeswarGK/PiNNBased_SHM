"""Pluggable mm-per-pixel sources (the scale plugin).

Contract: a DepthSource's `mm_per_pixel(image_bgr, crack_mask)` returns the
object-plane scale at the crack surface, in mm/px. `crack_mask` (union of
detected crack masks, may be None) lets depth-based sources sample depth
where it matters instead of over the whole frame.

Backends, cheapest first:
- manual      : a number you already know
- reference   : known-size object measured in the frame
- standoff    : known camera distance + focal length (pinhole)
- depth-file  : aligned depth map from a depth camera (RealSense-style
                16-bit PNG in mm, or .npy in mm), median over crack region
- monocular   : Depth Anything V2 metric model, same pinhole conversion

The two depth-based sources share the same maths: scale = Z_median / f_px.
Swapping a real depth camera for the monocular model (or back) changes
nothing downstream.
"""

from pathlib import Path

import cv2
import numpy as np

_REGISTRY = {}


def register(name):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def get_depth_source(name, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"unknown depth source '{name}' — available: "
                       f"{sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_depth_sources():
    return sorted(_REGISTRY)


def focal_px_from_specs(focal_length_mm, sensor_width_mm, image_width_px):
    """Convert a datasheet focal length to pixels for the pinhole sources."""
    return focal_length_mm * image_width_px / sensor_width_mm


def _median_depth_mm(depth_mm, crack_mask):
    """Median depth over the crack region (dilated), else central patch."""
    if crack_mask is not None and (crack_mask > 0).any():
        region = cv2.dilate((crack_mask > 0).astype(np.uint8),
                            np.ones((15, 15), np.uint8)) > 0
    else:
        h, w = depth_mm.shape[:2]
        region = np.zeros(depth_mm.shape[:2], dtype=bool)
        region[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    vals = depth_mm[region]
    vals = vals[vals > 0]                      # 0 = no return (depth cameras)
    if len(vals) == 0:
        raise ValueError("no valid depth values in the sampled region")
    return float(np.median(vals))


@register("manual")
class ManualScale:
    def __init__(self, mm_per_pixel):
        self.value = float(mm_per_pixel)

    def mm_per_pixel(self, image_bgr=None, crack_mask=None):
        return self.value


@register("reference")
class ReferenceObject:
    def __init__(self, ref_mm, ref_px):
        self.value = float(ref_mm) / float(ref_px)

    def mm_per_pixel(self, image_bgr=None, crack_mask=None):
        return self.value


@register("standoff")
class StandoffPinhole:
    def __init__(self, standoff_mm, focal_px):
        self.value = float(standoff_mm) / float(focal_px)

    def mm_per_pixel(self, image_bgr=None, crack_mask=None):
        return self.value


@register("depth-file")
class DepthMapFile:
    """Aligned depth map from a depth camera, loaded from a sidecar file.

    Accepts 16-bit PNG with depth in mm (RealSense convention) or a .npy
    array in mm, same resolution as (or resized to) the RGB frame. A live
    camera backend (pyrealsense2 etc.) plugs in by subclassing and
    overriding `_load` — the conversion maths stays here.
    """

    def __init__(self, depth_path, focal_px, depth_scale=1.0):
        self.depth_path = Path(depth_path)
        self.focal_px = float(focal_px)
        self.depth_scale = float(depth_scale)   # multiplier to get mm

    def _load(self, shape_hw):
        if not self.depth_path.exists():
            raise FileNotFoundError(f"depth map not found: {self.depth_path}")
        if self.depth_path.suffix == ".npy":
            depth = np.load(self.depth_path).astype(np.float64)
        else:
            raw = cv2.imread(str(self.depth_path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError(f"cannot read depth map: {self.depth_path}")
            depth = raw.astype(np.float64)
        depth *= self.depth_scale
        if depth.shape[:2] != shape_hw:
            depth = cv2.resize(depth, (shape_hw[1], shape_hw[0]),
                               interpolation=cv2.INTER_NEAREST)
        return depth

    def mm_per_pixel(self, image_bgr, crack_mask=None):
        depth = self._load(image_bgr.shape[:2])
        z = _median_depth_mm(depth, crack_mask)
        return z / self.focal_px


@register("monocular")
class MonocularDepthAnything:
    """Depth Anything V2 metric depth -> pinhole scale. Needs focal_px.

    Model downloads once from HuggingFace (~100 MB for -Small). Indoor
    checkpoint tops out ~20 m, outdoor ~80 m — pick per deployment.
    Accuracy caveat: metric monocular depth carries 5-15% scale error in
    the wild; treat the result as an estimate, not a calibration.
    """

    def __init__(self, focal_px,
                 model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                 device=None):
        self.focal_px = float(focal_px)
        self.model_name = model
        self.device = device
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline
            dev = self.device or (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu")
            self._pipe = pipeline("depth-estimation", model=self.model_name,
                                  device=dev)
        return self._pipe

    def mm_per_pixel(self, image_bgr, crack_mask=None):
        from PIL import Image
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        out = self._pipeline()(Image.fromarray(rgb))
        depth_m = np.array(out["predicted_depth"])
        if depth_m.shape[:2] != image_bgr.shape[:2]:
            depth_m = cv2.resize(depth_m,
                                 (image_bgr.shape[1], image_bgr.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        z_mm = _median_depth_mm(depth_m * 1000.0, crack_mask)
        return z_mm / self.focal_px
