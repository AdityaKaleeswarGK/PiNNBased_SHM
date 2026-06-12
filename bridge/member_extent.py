"""Member extent (W) measurement via MobileSAM, prompted by the crack.

The crack is by definition ON the member, so its skeleton pixels are free,
always-correct prompts: MobileSAM returns the member-surface mask, and for
each crack we cast a chord through it ALONG the crack's growth direction —
the chord length is W (the distance the crack could grow before exiting
the member).

The same mask answers two more questions for free:
- crack type: an endpoint touching the mask boundary => EDGE crack
  (full length drives K); no endpoint touching => CENTER crack
  (half length + secant geometry factor — ~1.6x less severe).
- a chord that exits through the IMAGE border instead of the mask boundary
  means the member extends beyond the frame: W is a lower bound
  (=> a/W upper bound => conservative K; flagged, not hidden).

Error tolerance is friendly: ~10% W error moves K only 2-4% at small a/W,
so a coarse MobileSAM mask is plenty. Weights (~40 MB) auto-download once
via ultralytics.
"""

import math

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

_sam = None

EDGE_TOUCH_TOL_PX = 6     # endpoint within this of the mask boundary = edge


def get_sam(weights="mobile_sam.pt"):
    global _sam
    if _sam is None:
        from ultralytics import SAM
        _sam = SAM(weights)
    return _sam


def member_mask(image_bgr, prompt_points_xy):
    """MobileSAM mask of the surface containing the prompt points."""
    model = get_sam()
    res = model(image_bgr, points=prompt_points_xy,
                labels=[1] * len(prompt_points_xy), verbose=False)[0]
    if res.masks is None or len(res.masks.data) == 0:
        return None
    m = res.masks.data[0].cpu().numpy().astype(bool)
    if m.shape != image_bgr.shape[:2]:
        m = cv2.resize(m.astype(np.uint8),
                       (image_bgr.shape[1], image_bgr.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


def chord_through(mask, start_xy, angle_deg):
    """Walk from start along +-angle until the mask (or image) ends.

    Returns dict with W_px, the two end points, and whether either end was
    clipped by the image border (=> W is a lower bound).
    """
    h, w = mask.shape
    x0, y0 = float(start_xy[0]), float(start_xy[1])
    if not (0 <= int(y0) < h and 0 <= int(x0) < w) or not mask[int(y0), int(x0)]:
        return None
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), -math.sin(rad)   # image coords: y down

    ends, border = [], False
    for sx, sy in ((dx, dy), (-dx, -dy)):
        x, y = x0, y0
        while True:
            nx, ny = x + sx, y + sy
            if not (0 <= nx < w and 0 <= ny < h):
                border = True
                break
            if not mask[int(ny), int(nx)]:
                break
            x, y = nx, ny
        ends.append((round(x, 1), round(y, 1)))

    return {"W_px": round(math.dist(ends[0], ends[1]), 1),
            "ends_xy": ends, "hit_border": border}


def measure_member(image_bgr, cracks, mm_per_pixel=None, mask=None,
                   prompts_per_crack=3):
    """Measure W (and crack type) for every crack against one SAM mask.

    `cracks` are internal crack dicts from analyse_frame (need `_pixels_rc`,
    `endpoints_xy`, `angle_deg`, `id`). Pass a cached `mask` to skip the
    SAM call (live mode re-uses the mask between refreshes).

    Returns {"mask": bool HxW, "per_crack": {crack_id: {...}}} or None.
    """
    if not cracks:
        return None
    if mask is None:
        h, w = image_bgr.shape[:2]
        # Prompt BESIDE the crack, not on it: a point on the dark crack
        # line makes SAM segment the line itself. Offsetting perpendicular
        # to the crack keeps prompts on the member surface around it.
        # Prompts come from the few largest cracks only — they all sit on
        # the same surface, and hundreds of prompts stall SAM when a noisy
        # detector reports dozens of "cracks".
        prompt_cracks = sorted((c for c in cracks if c["angle_deg"] is not None),
                               key=lambda c: -c["arc_length_px"])[:3]
        for offset in (12, 25):
            prompts = []
            for crack in prompt_cracks:
                if crack["angle_deg"] is None:
                    continue
                rad = math.radians(crack["angle_deg"])
                px_, py_ = math.sin(rad), math.cos(rad)   # perpendicular
                pts = crack["_pixels_rc"]
                idx = np.linspace(0, len(pts) - 1,
                                  min(prompts_per_crack, len(pts))).astype(int)
                for i in idx:
                    r, c = pts[i]
                    for sign in (1, -1):
                        x = int(c + sign * offset * px_)
                        y = int(r + sign * offset * py_)
                        if 0 <= x < w and 0 <= y < h:
                            prompts.append([x, y])
            if not prompts:
                return None
            mask = member_mask(image_bgr, prompts)
            # A sane member mask dwarfs the cracks; a tiny mask means SAM
            # latched onto the crack line — retry with a larger offset.
            crack_px = sum(len(c["_pixels_rc"]) for c in cracks)
            if mask is not None and mask.sum() > 10 * crack_px:
                break
        if mask is None:
            return None

    # SAM often excludes the crack itself as a thin hole in the member
    # mask, which would truncate chords at the crack walls. Cracks are
    # thin; closing fills them without moving the member outline.
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((15, 15), np.uint8)).astype(bool)

    dt = distance_transform_edt(mask)
    per_crack = {}
    for crack in cracks:
        if crack["angle_deg"] is None:
            continue
        pts = crack["_pixels_rc"]
        mid = pts[len(pts) // 2]
        ch = chord_through(mask, (mid[1], mid[0]), crack["angle_deg"])
        if ch is None:
            continue

        # Edge vs center: does any endpoint touch the member boundary?
        touch = False
        for (ex, ey) in crack["endpoints_xy"]:
            ey_c = min(max(ey, 0), mask.shape[0] - 1)
            ex_c = min(max(ex, 0), mask.shape[1] - 1)
            if not mask[ey_c, ex_c] or dt[ey_c, ex_c] <= EDGE_TOUCH_TOL_PX:
                touch = True
                break

        rec = {"W_px": ch["W_px"],
               "W_lower_bound": ch["hit_border"],
               "crack_type": "edge" if touch else "center",
               "chord_ends_xy": ch["ends_xy"]}
        if mm_per_pixel is not None:
            rec["W_mm"] = round(ch["W_px"] * mm_per_pixel, 1)
        per_crack[crack["id"]] = rec

    return {"mask": mask, "per_crack": per_crack}
