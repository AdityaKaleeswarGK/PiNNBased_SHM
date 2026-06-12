"""Headless multi-crack analysis core (CV side of the bridge).

Chain: CLAHE -> YOLO-seg (per-instance masks kept separate) -> connected
components -> skeleton branch decomposition at junctions -> collinear
fragment linking -> per-crack geometry (arc length, orientation, width) ->
ACI structural filter -> crack-interaction screening.

Multi-crack design notes
------------------------
- YOLO instance identity is preserved: two crossing cracks detected as two
  instances stay two cracks even where their pixels overlap.
- A single mask containing a junction (X or Y) is cut at skeleton
  branchpoints; the resulting branches are re-joined only when collinear,
  so an X-crossing resolves into two through-going cracks.
- Fragmented detections (mask gaps) are re-linked when endpoints are close,
  orientations agree, and the gap direction matches — the linked gap counts
  toward crack length (conservative, since K_I grows with length).
"""

import math
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt
from scipy.ndimage import label as nd_label
from skimage.morphology import skeletonize

# ── Defaults (mirror the notebook) ────────────────────────────────────────
YOLO_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "best-2.pt")
YOLO_CONF = 0.20
YOLO_IOU = 0.45
CRACK_CLASS_ID = 0

ACI_MIN_WIDTH_MM = 0.3      # ACI 224R cosmetic crack limit
ACI_MIN_WIDTH_PX = 3        # fallback when no calibration
MIN_COMPONENT_AREA_PX = 200

MIN_BRANCH_PX = 8           # skeleton spurs shorter than this are noise
MIN_CRACK_LENGTH_PX = 20    # assembled cracks shorter than this are stubs
LINK_MAX_GAP_PX = 25        # fragment linking: max endpoint distance
LINK_MAX_ANGLE_DEG = 25.0   # fragment linking: max orientation difference

_yolo_model = None

_NEIGH_KERNEL = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
_STRUCT8 = np.ones((3, 3), dtype=int)


_default_detector = None


def get_default_detector():
    """Lazily build the default detector (YOLO-seg with the repo weights)."""
    global _default_detector
    if _default_detector is None:
        from .detectors import get_detector
        _default_detector = get_detector("ultralytics-seg",
                                         weights=YOLO_MODEL_PATH,
                                         conf=YOLO_CONF, iou=YOLO_IOU,
                                         class_id=CRACK_CLASS_ID)
    return _default_detector


def preprocess_for_yolo(image_bgr, clip_limit=2.0, tile_grid=(8, 8)):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)


# ── Geometry helpers ──────────────────────────────────────────────────────

def _orientation_deg(points_rc):
    """PCA orientation of (row, col) points; image-coord angle in (-90, 90]."""
    if len(points_rc) < 2:
        return None
    pts = np.column_stack([points_rc[:, 1], points_rc[:, 0]]).astype(np.float64)
    pts -= pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    vx, vy = vt[0]
    angle = np.degrees(np.arctan2(-vy, vx))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return float(angle)


def _angle_diff(a, b):
    """Difference between two undirected orientations, in [0, 90]."""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _order_path(pix_set):
    """Order a simple 8-connected skeleton segment into a pixel path."""
    def nbrs(p):
        r, c = p
        return [(r + dr, c + dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0) and (r + dr, c + dc) in pix_set]

    start = next((p for p in pix_set if len(nbrs(p)) <= 1), None)
    if start is None:                      # closed loop — rare; start anywhere
        start = next(iter(pix_set))
    path, visited = [start], {start}
    cur = start
    while True:
        nxt = [q for q in nbrs(cur) if q not in visited]
        if not nxt:
            break
        nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
        cur = nxt[0]
        path.append(cur)
        visited.add(cur)
    return path


def _arc_length(path):
    return float(sum(math.dist(path[i], path[i + 1])
                     for i in range(len(path) - 1)))


def split_branches(skel, min_branch_px=MIN_BRANCH_PX):
    """Cut a skeleton at branchpoints; return ordered pixel paths."""
    filt = convolve(skel.astype(np.uint8), _NEIGH_KERNEL,
                    mode="constant", cval=0)
    branch_pts = skel & (filt >= 13)
    if branch_pts.any():
        cut = cv2.dilate(branch_pts.astype(np.uint8),
                         np.ones((3, 3), np.uint8)) > 0
        segments_mask = skel & ~cut
    else:
        segments_mask = skel

    lab, n = nd_label(segments_mask, structure=_STRUCT8)
    paths = []
    for i in range(1, n + 1):
        pix = np.argwhere(lab == i)
        if len(pix) < min_branch_px:
            continue
        paths.append(_order_path(set(map(tuple, pix))))
    return paths


# ── Component extraction (speckle filter + cross-instance dedup) ──────────

def extract_components(instance_masks, min_area_px=MIN_COMPONENT_AREA_PX,
                       dedup_iou=0.5):
    """Connected components per YOLO instance, deduplicated across instances.

    Returns (components, rejected). Each component: {id, mask, dt, area_px}.
    """
    raw = []
    rejected = []
    for inst_idx, mask in enumerate(instance_masks):
        num, labels = cv2.connectedComponents((mask > 0).astype(np.uint8),
                                              connectivity=8)
        for i in range(1, num):
            comp = (labels == i)
            area = int(comp.sum())
            if area < min_area_px:
                rejected.append({"instance": inst_idx, "area_px": area,
                                 "reason": f"area {area} px2 < {min_area_px} px2 minimum"})
                continue
            raw.append({"mask": comp, "area_px": area})

    # Dedup: same crack returned by two instances (high pixel overlap)
    raw.sort(key=lambda c: -c["area_px"])
    kept = []
    for cand in raw:
        dup = False
        for k in kept:
            inter = np.logical_and(cand["mask"], k["mask"]).sum()
            union = cand["area_px"] + k["area_px"] - inter
            if union > 0 and inter / union > dedup_iou:
                dup = True
                break
        if not dup:
            kept.append(cand)

    for cid, comp in enumerate(kept):
        comp["id"] = cid
        comp["dt"] = distance_transform_edt(comp["mask"])
    return kept, rejected


# ── Segment building + collinear linking ──────────────────────────────────

def build_segments(components):
    """Skeletonize each component and cut into branch segments."""
    segments = []
    for comp in components:
        skel = skeletonize(comp["mask"])
        for path in split_branches(skel):
            pts = np.array(path)
            segments.append({
                "path": path,
                "component_id": comp["id"],
                "arc_px": _arc_length(path),
                "angle_deg": _orientation_deg(pts),
            })
    return segments


def link_segments(segments, max_gap_px=LINK_MAX_GAP_PX,
                  max_angle_deg=LINK_MAX_ANGLE_DEG):
    """Union-find merge of collinear, nearby segments into cracks.

    Returns (groups, link_gaps) where groups maps root -> [segment indices]
    and link_gaps maps root -> total linked gap length in px.
    """
    n = len(segments)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    gaps = []
    for i in range(n):
        si = segments[i]
        if si["angle_deg"] is None:
            continue
        for j in range(i + 1, n):
            sj = segments[j]
            if sj["angle_deg"] is None:
                continue
            best = min(((math.dist(ei, ej), ei, ej)
                        for ei in (si["path"][0], si["path"][-1])
                        for ej in (sj["path"][0], sj["path"][-1])),
                       key=lambda t: t[0])
            gap, ei, ej = best
            if gap > max_gap_px:
                continue
            if _angle_diff(si["angle_deg"], sj["angle_deg"]) > max_angle_deg:
                continue
            if gap > 3:
                gap_angle = math.degrees(
                    math.atan2(-(ej[0] - ei[0]), ej[1] - ei[1]))
                if (_angle_diff(gap_angle, si["angle_deg"]) > max_angle_deg + 10
                        or _angle_diff(gap_angle, sj["angle_deg"]) > max_angle_deg + 10):
                    continue
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                gaps.append((rj, gap))

    groups, link_gaps = {}, {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    for r, gap in gaps:
        link_gaps[find(r)] = link_gaps.get(find(r), 0.0) + gap
    return groups, link_gaps


# ── Crack assembly ────────────────────────────────────────────────────────

def assemble_cracks(components, segments, groups, link_gaps,
                    mm_per_pixel=None,
                    aci_min_width_mm=ACI_MIN_WIDTH_MM,
                    aci_min_width_px=ACI_MIN_WIDTH_PX,
                    min_length_px=MIN_CRACK_LENGTH_PX):
    """Merge segment groups into crack records with geometry + width stats."""
    min_width_px = (aci_min_width_mm / mm_per_pixel
                    if mm_per_pixel is not None else aci_min_width_px)
    dt_by_comp = {c["id"]: c["dt"] for c in components}

    cracks, rejected = [], []
    for root, idxs in groups.items():
        segs = [segments[i] for i in idxs]
        all_pts = np.vstack([np.array(s["path"]) for s in segs])
        arc_px = sum(s["arc_px"] for s in segs) + link_gaps.get(root, 0.0)

        if arc_px < min_length_px:
            rejected.append({
                "n_segments": len(segs), "arc_length_px": round(arc_px, 1),
                "reason": f"arc length {arc_px:.0f} px < {min_length_px} px "
                          "minimum — junction stub / noise"})
            continue

        widths = []
        for s in segs:
            dt = dt_by_comp[s["component_id"]]
            for (r, c) in s["path"][::3]:
                widths.append(2.0 * dt[r, c])
        widths = np.array(widths)
        mean_w = float(widths.mean()) if len(widths) else 0.0

        if mean_w < min_width_px:
            rejected.append({
                "n_segments": len(segs), "arc_length_px": round(arc_px, 1),
                "mean_width_px": round(mean_w, 2),
                "reason": f"mean width {mean_w:.1f} px < threshold "
                          f"{min_width_px:.1f} px — non-structural (ACI)"})
            continue

        # Tip-to-tip: farthest pair among segment endpoints
        ends = [s["path"][0] for s in segs] + [s["path"][-1] for s in segs]
        tip_a, tip_b = max(((p, q) for p in ends for q in ends),
                           key=lambda t: math.dist(t[0], t[1]))

        cracks.append({
            "n_segments": len(segs),
            "component_ids": sorted({s["component_id"] for s in segs}),
            "arc_length_px": round(arc_px, 1),
            "tip_to_tip_px": round(math.dist(tip_a, tip_b), 1),
            "angle_deg": _orientation_deg(all_pts),
            "endpoints_xy": [(int(tip_a[1]), int(tip_a[0])),
                             (int(tip_b[1]), int(tip_b[0]))],
            "width_mean_px": round(mean_w, 2),
            "width_max_px": round(float(widths.max()), 2) if len(widths) else None,
            "width_median_px": round(float(np.median(widths)), 2) if len(widths) else None,
            "centroid_xy": (round(float(all_pts[:, 1].mean()), 1),
                            round(float(all_pts[:, 0].mean()), 1)),
            "_pixels_rc": all_pts,
        })

    cracks.sort(key=lambda c: -c["arc_length_px"])
    for k, crack in enumerate(cracks, start=1):
        crack["id"] = k
    return cracks, rejected


def screen_interactions(cracks, angle_tol_deg=30.0, sample_every=5):
    """Flag crack pairs close enough to interact (single-crack K_I suspect).

    Rule of thumb: two similarly oriented cracks closer than the shorter
    crack's length amplify each other's stress fields.
    """
    flags = []
    for i in range(len(cracks)):
        for j in range(i + 1, len(cracks)):
            ci, cj = cracks[i], cracks[j]
            if ci["angle_deg"] is None or cj["angle_deg"] is None:
                continue
            if _angle_diff(ci["angle_deg"], cj["angle_deg"]) > angle_tol_deg:
                continue
            pi = ci["_pixels_rc"][::sample_every].astype(np.float64)
            pj = cj["_pixels_rc"][::sample_every].astype(np.float64)
            d2 = ((pi[:, None, :] - pj[None, :, :]) ** 2).sum(axis=2)
            dmin = float(np.sqrt(d2.min()))
            limit = min(ci["arc_length_px"], cj["arc_length_px"])
            if dmin < limit:
                flags.append({
                    "pair": [ci["id"], cj["id"]],
                    "min_dist_px": round(dmin, 1),
                    "limit_px": round(limit, 1),
                    "note": "closer than the shorter crack's length — "
                            "isolated-crack K_I underestimates here"})
    return flags


# ── Structural context ────────────────────────────────────────────────────

def compute_structural_context(cracks, image_shape, load_point_norm,
                               load_direction_deg):
    h, w = image_shape[:2]
    load_px = (int(load_point_norm[0] * w), int(load_point_norm[1] * h))
    ps_deg = (load_direction_deg + 90) % 180   # expected Mode-I crack line

    for crack in cracks:
        ang = crack["angle_deg"]
        crack["angular_dev_deg"] = (round(_angle_diff(ang, ps_deg), 2)
                                    if ang is not None else None)
        pts = crack["_pixels_rc"]
        dists = np.sqrt((pts[:, 1] - load_px[0]) ** 2
                        + (pts[:, 0] - load_px[1]) ** 2)
        crack["closest_pt_dist_px"] = round(float(dists.min()), 1)
    return load_px, ps_deg


# ── Entry point ───────────────────────────────────────────────────────────

def analyse_frame(img, mm_per_pixel=None, detector=None, depth_source=None,
                  load_point_norm=(0.5, 0.0), load_direction_deg=90):
    """Run the multi-crack pipeline on an in-memory BGR frame.

    Returns (report, cracks): `report` is the JSON-serialisable dict;
    `cracks` is the internal list whose entries keep `_pixels_rc` so live
    viewers can draw skeleton overlays.
    """
    pre = preprocess_for_yolo(img)
    detector = detector or get_default_detector()
    detections = detector.detect(img, pre)
    instance_masks = [d.mask for d in detections if d.mask is not None]

    if depth_source is not None:
        union = np.zeros(img.shape[:2], dtype=np.uint8)
        for m in instance_masks:
            union = cv2.bitwise_or(union, m)
        mm_per_pixel = depth_source.mm_per_pixel(img, union)

    components, rejected_comps = extract_components(instance_masks)
    segments = build_segments(components)
    groups, link_gaps = link_segments(segments)
    cracks, rejected_cracks = assemble_cracks(components, segments, groups,
                                              link_gaps,
                                              mm_per_pixel=mm_per_pixel)
    load_px, ps_deg = compute_structural_context(
        cracks, img.shape, load_point_norm, load_direction_deg)
    interactions = screen_interactions(cracks)

    def to_report(crack):
        rep = {k: v for k, v in crack.items() if not k.startswith("_")}
        if mm_per_pixel is not None:
            rep["length_mm"] = round(crack["arc_length_px"] * mm_per_pixel, 2)
            rep["tip_to_tip_mm"] = round(crack["tip_to_tip_px"] * mm_per_pixel, 2)
            for key in ("width_mean", "width_max", "width_median"):
                px = crack.get(f"{key}_px")
                rep[f"{key}_mm"] = (round(px * mm_per_pixel, 3)
                                    if px is not None else None)
        return rep

    report = {
        "image_size_px": [img.shape[1], img.shape[0]],
        "detector": getattr(detector, "name", type(detector).__name__),
        "calibration": {
            "mm_per_pixel": (round(mm_per_pixel, 6)
                             if mm_per_pixel is not None else None),
            "source": (getattr(depth_source, "name",
                               type(depth_source).__name__)
                       if depth_source is not None else
                       ("manual" if mm_per_pixel is not None else None)),
        },
        "load": {
            "point_norm": list(load_point_norm),
            "point_px": list(load_px),
            "direction_deg": load_direction_deg,
            "principal_stress_deg": ps_deg,
        },
        "n_detections": len(instance_masks),
        "n_components": len(components),
        "n_segments": len(segments),
        "cracks": [to_report(c) for c in cracks],
        "interactions": interactions,
        "rejected": rejected_comps + rejected_cracks,
    }
    return report, cracks


def analyse_image(image_path, mm_per_pixel=None, detector=None,
                  depth_source=None,
                  load_point_norm=(0.5, 0.0), load_direction_deg=90):
    """Run the full headless multi-crack pipeline on one image file.

    `detector` is any bridge.detectors backend (default: YOLO-seg with the
    repo weights). `depth_source` is any bridge.depth_sources backend; when
    given, it supersedes `mm_per_pixel` and is evaluated AFTER detection so
    depth-based sources can sample depth over the crack region.

    Returns a JSON-serialisable dict (the `crack_report`).
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    report, _ = analyse_frame(img, mm_per_pixel=mm_per_pixel,
                              detector=detector, depth_source=depth_source,
                              load_point_norm=load_point_norm,
                              load_direction_deg=load_direction_deg)
    report = {"image": str(image_path), **report}
    return report
