"""Pluggable crack detectors.

Contract: a Detector's `detect(image_bgr, preprocessed_bgr)` returns a list
of Detection(mask, box, score). The geometry stage consumes binary masks
only, so detectors fall into two families:

- mask detectors (instance segmentation): masks pass straight through
- box detectors (SSD, detection-only YOLO): masks are completed by
  thresholding inside each box (`complete_masks`)

Register new backends with @register("name"); select with
get_detector("name", **kwargs). The rest of the pipeline never changes.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_WEIGHTS = str(Path(__file__).resolve().parent.parent / "best-2.pt")

_REGISTRY = {}


def register(name):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def get_detector(name, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector '{name}' — available: "
                       f"{sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_detectors():
    return sorted(_REGISTRY)


@dataclass
class Detection:
    mask: np.ndarray | None      # HxW uint8 (0/255), or None for box-only
    box: tuple | None            # (x1, y1, x2, y2) px
    score: float


# ── Mask completion for box-only detectors ────────────────────────────────

def complete_masks(image_bgr, detections, pad=4):
    """Fill missing masks by segmenting inside each box (CLAHE + Otsu).

    Cracks are dark against concrete, so Otsu with THRESH_BINARY_INV
    inside the (padded) box recovers a usable mask. Quality is below a
    trained seg model — that difference is exactly what the detector
    benchmark harness measures.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Threshold on whole-frame statistics (an in-box histogram skews the
    # Otsu split and turns thin cracks into texture blobs), then crop.
    _, th_full = cv2.threshold(clahe, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    th_full = cv2.morphologyEx(th_full, cv2.MORPH_OPEN, kernel)
    th_full = cv2.morphologyEx(th_full, cv2.MORPH_CLOSE, kernel)

    out = []
    for det in detections:
        if det.mask is not None or det.box is None:
            out.append(det)
            continue
        x1, y1, x2, y2 = det.box
        x1, y1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
        x2, y2 = min(w, int(x2) + pad), min(h, int(y2) + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        th = th_full[y1:y2, x1:x2]

        # One detection = one crack: keep only the dominant component and
        # any comparably sized ones, dropping surface-texture speckle.
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            th, connectivity=8)
        if num <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = {i + 1 for i, a in enumerate(areas) if a >= 50}
        cleaned = np.isin(labels, list(keep)).astype(np.uint8) * 255

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = cleaned
        out.append(Detection(mask=mask, box=det.box, score=det.score))
    return out


# ── Backends ──────────────────────────────────────────────────────────────

@register("ultralytics-seg")
class UltralyticsSegDetector:
    """Any ultralytics instance-segmentation checkpoint (YOLOv8/11/26-seg)."""

    def __init__(self, weights=DEFAULT_WEIGHTS, conf=0.20, iou=0.45,
                 class_id=0):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.iou, self.class_id = conf, iou, class_id

    def detect(self, image_bgr, preprocessed_bgr):
        h, w = image_bgr.shape[:2]
        result = self.model.predict(source=preprocessed_bgr, conf=self.conf,
                                    iou=self.iou, verbose=False)[0]
        dets = []
        if result.masks is not None:
            for i, seg in enumerate(result.masks.data):
                if (self.class_id is not None
                        and int(result.boxes.cls[i].item()) != self.class_id):
                    continue
                seg_np = seg.cpu().numpy().astype(np.uint8) * 255
                mask = cv2.resize(seg_np, (w, h),
                                  interpolation=cv2.INTER_NEAREST)
                box = tuple(result.boxes.xyxy[i].cpu().numpy().tolist())
                dets.append(Detection(mask=mask, box=box,
                                      score=float(result.boxes.conf[i])))
        return dets


@register("ultralytics-box")
class UltralyticsBoxDetector:
    """Detection-only YOLO checkpoint; masks completed by Otsu-in-box."""

    def __init__(self, weights=DEFAULT_WEIGHTS, conf=0.20, iou=0.45,
                 class_id=0):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.iou, self.class_id = conf, iou, class_id

    def detect(self, image_bgr, preprocessed_bgr):
        result = self.model.predict(source=preprocessed_bgr, conf=self.conf,
                                    iou=self.iou, verbose=False)[0]
        dets = []
        if result.boxes is not None:
            for i in range(len(result.boxes)):
                if (self.class_id is not None
                        and int(result.boxes.cls[i].item()) != self.class_id):
                    continue
                box = tuple(result.boxes.xyxy[i].cpu().numpy().tolist())
                dets.append(Detection(mask=None, box=box,
                                      score=float(result.boxes.conf[i])))
        return complete_masks(image_bgr, dets)


@register("torchvision")
class TorchvisionDetector:
    """torchvision detection models: SSD, Faster-RCNN, Mask-RCNN.

    `arch` is any torchvision.models.detection factory name, e.g.
    'ssd300_vgg16', 'fasterrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn'.
    Pass a crack-trained `checkpoint` (state_dict .pth) — COCO weights do
    not know what a crack is and will return nothing useful.
    """

    def __init__(self, arch="ssd300_vgg16", checkpoint=None, conf=0.3,
                 class_id=None, num_classes=None):
        import torch
        import torchvision
        factory = getattr(torchvision.models.detection, arch)
        if checkpoint:
            kwargs = {"num_classes": num_classes} if num_classes else {}
            self.model = factory(weights=None, **kwargs)
            state = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(state)
        else:
            self.model = factory(weights="DEFAULT")
            print(f"[detector] WARNING: {arch} loaded with COCO weights — "
                  "supply checkpoint= for crack detection")
        self.model.eval()
        self.conf, self.class_id = conf, class_id
        self._torch = torch

    def detect(self, image_bgr, preprocessed_bgr):
        torch = self._torch
        rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = self.model([t])[0]
        dets = []
        for i in range(len(out["scores"])):
            score = float(out["scores"][i])
            if score < self.conf:
                continue
            if (self.class_id is not None
                    and int(out["labels"][i]) != self.class_id):
                continue
            box = tuple(out["boxes"][i].tolist())
            mask = None
            if "masks" in out:                      # Mask-RCNN
                m = (out["masks"][i, 0].numpy() > 0.5).astype(np.uint8) * 255
                mask = m
            dets.append(Detection(mask=mask, box=box, score=score))
        return complete_masks(image_bgr, dets)


@register("classical")
class ClassicalDetector:
    """No-model fallback: CLAHE + Otsu + morphology on the whole frame.

    The original pre-YOLO notebook pipeline. Noisy on textured surfaces;
    useful as the zero-cost baseline rung in the detector benchmark.
    """

    def __init__(self, min_area_px=200):
        self.min_area_px = min_area_px

    def detect(self, image_bgr, preprocessed_bgr):
        gray = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            th, connectivity=8)
        dets = []
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area_px:
                continue
            mask = ((labels == i) * 255).astype(np.uint8)
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            dets.append(Detection(mask=mask, box=(x, y, x + bw, y + bh),
                                  score=1.0))
        return dets
