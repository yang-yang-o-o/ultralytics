#!/usr/bin/env python3
"""Compose YOLO26n official-val qualitative video from existing plots + fresh predicts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

# Repo root: .../ultralytics/docs/official_example/scripts/this_file.py → parents[3]
ROOT = Path(__file__).resolve().parents[3]
DATASETS = Path(SETTINGS.get("datasets_dir", ROOT.parent / "datasets"))
OUT_DIR = ROOT / "runs/official_val/results_video"
FRAMES = OUT_DIR / "frames"
PRED = OUT_DIR / "preds"
VIDEO = ROOT / "docs/official_example/yolo26n_official_val_results.mp4"
W, H = 1600, 1000
FPS = 2  # slow enough to read metrics / inspect images
HOLD = 3  # seconds per static frame (written as HOLD*FPS duplicates)

FONT = cv2.FONT_HERSHEY_SIMPLEX
rng = random.Random(26)


def ensure_dirs() -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)


def blank(color=(18, 18, 22)) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = color
    return img


def put(img, text, org, scale=1.0, color=(235, 235, 235), thick=2):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def fit(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    canvas[:] = (28, 28, 32)
    x, y = (tw - nw) // 2, (th - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def reduce_mosaic(img: np.ndarray, keep: int = 4) -> np.ndarray:
    """Crop an Ultralytics val_batch mosaic to a top-left 2x2 (4 tiles) for readability."""
    h, w = img.shape[:2]
    # YOLO val plots are typically 4x4 (bs=16) or 2x4 (bs=8) at ~1920 width
    if h >= 900:
        grid_rows, grid_cols = 4, 4
    elif h >= 600 and w >= 1400:
        # ambiguous: prefer 4x4 when cells would stay >= ~150px tall
        grid_rows, grid_cols = (4, 4) if h / 4 >= 150 else (2, 4)
    elif h * 2 < w:
        grid_rows, grid_cols = 2, 4
    else:
        grid_rows, grid_cols = 4, 4
    th, tw = h // grid_rows, w // grid_cols
    # always take top-left 2x2 block (4 tiles)
    return img[0 : 2 * th, 0 : 2 * tw].copy()


def banner(title: str, lines: list[str], accent=(70, 160, 255)) -> np.ndarray:
    img = blank()
    put(img, title, (60, 120), 1.6, accent, 3)
    y = 220
    for line in lines:
        put(img, line, (60, y), 1.05, (220, 220, 220), 2)
        y += 58
    put(img, "Source: local full-val runs  |  models: yolo26n*", (60, H - 50), 0.7, (140, 140, 150), 1)
    return img


def summary_table() -> np.ndarray:
    """Draw the official-val conclusion as an actual grid table."""
    headers = ["Task", "Protocol", "Local", "Official", "Align"]
    rows = [
        ["Detect", "COCO e2e", "AP 40.0", "40.1", "OK"],
        ["Detect", "COCO o2m+NMS", "AP 40.8", "40.9", "OK"],
        ["Segment", "COCO e2e", "39.8 / 33.9", "39.6 / 33.9", "OK"],
        ["Pose", "COCO-pose e2e", "AP 57.2", "57.2", "OK"],
        ["Semantic", "Cityscapes 2048", "mIoU 78.3", "78.3", "OK"],
        ["Classify", "ImageNet 224", "71.4 / 90.1", "71.4 / 90.1", "OK"],
        ["Depth", "NYU single+median", "d1 0.783", "0.783", "OK"],
        ["Depth", "TTA+log-LS approx", "~0.839", "0.882", "GAP"],
        ["OBB", "DOTA val single", "~43.9", "(no val table)", "-"],
        ["OBB", "DOTA test multi-scale", "n/a (no GT)", "52.4", "SERVER"],
    ]
    img = blank()
    put(img, "Summary Table (local vs official)", (48, 58), 1.25, (70, 160, 255), 2)
    put(img, "Models: yolo26n*  |  same protocol required for alignment", (48, 96), 0.65, (160, 160, 170), 1)

    # Column geometry
    x0, y0 = 48, 130
    col_w = [220, 340, 280, 280, 140]
    row_h = 68
    table_w = sum(col_w)
    table_h = row_h * (1 + len(rows))

    # Table background
    cv2.rectangle(img, (x0, y0), (x0 + table_w, y0 + table_h), (32, 32, 38), -1)
    # Header fill
    cv2.rectangle(img, (x0, y0), (x0 + table_w, y0 + row_h), (48, 52, 62), -1)

    def cell_x(c: int) -> int:
        return x0 + sum(col_w[:c])

    # Grid lines
    for r in range(len(rows) + 2):
        y = y0 + r * row_h
        cv2.line(img, (x0, y), (x0 + table_w, y), (80, 84, 96), 1, cv2.LINE_AA)
    for c in range(len(col_w) + 1):
        x = cell_x(c) if c < len(col_w) else x0 + table_w
        cv2.line(img, (x, y0), (x, y0 + table_h), (80, 84, 96), 1, cv2.LINE_AA)

    # Header text
    for c, h in enumerate(headers):
        put(img, h, (cell_x(c) + 14, y0 + 44), 0.75, (230, 230, 235), 2)

    # Body
    align_color = {
        "OK": (90, 220, 140),
        "GAP": (70, 160, 255),
        "-": (180, 180, 190),
        "SERVER": (90, 180, 255),
    }
    for r, row in enumerate(rows):
        cy = y0 + (r + 1) * row_h
        # zebra striping
        if r % 2 == 1:
            cv2.rectangle(img, (x0 + 1, cy + 1), (x0 + table_w - 1, cy + row_h - 1), (38, 40, 48), -1)
            # redraw vertical lines over zebra
            for c in range(len(col_w) + 1):
                x = cell_x(c) if c < len(col_w) else x0 + table_w
                cv2.line(img, (x, cy), (x, cy + row_h), (80, 84, 96), 1, cv2.LINE_AA)
            cv2.line(img, (x0, cy + row_h), (x0 + table_w, cy + row_h), (80, 84, 96), 1, cv2.LINE_AA)
        for c, text in enumerate(row):
            color = align_color.get(text, (220, 220, 225)) if c == 4 else (220, 220, 225)
            put(img, text, (cell_x(c) + 14, cy + 44), 0.68, color, 2)

    put(img, "OK = aligned  |  GAP = protocol incomplete  |  SERVER = needs DOTA eval server", (48, H - 42), 0.6, (140, 140, 150), 1)
    return img


def panel_pair(top: np.ndarray, bottom: np.ndarray, title: str, subtitle: str, shrink_mosaic: bool = True) -> np.ndarray:
    """GT (top) / Pred (bottom) stacked vertically for readability."""
    if shrink_mosaic:
        top = reduce_mosaic(top, keep=4)
        bottom = reduce_mosaic(bottom, keep=4)
    img = blank()
    put(img, title, (40, 42), 1.0, (70, 160, 255), 2)
    put(img, subtitle, (40, 78), 0.7, (180, 180, 190), 1)
    mid_y, gap = 95, 12
    label_h = 28
    usable = H - mid_y - 20
    ph = (usable - gap - 2 * label_h) // 2
    pw = W - 80
    T = fit(top, pw, ph)
    B = fit(bottom, pw, ph)
    y0 = mid_y
    put(img, "GT / Labels", (40, y0 + 18), 0.65, (90, 220, 140), 2)
    y0 += label_h
    img[y0 : y0 + ph, 40 : 40 + pw] = T
    y0 += ph + gap
    put(img, "Prediction", (40, y0 + 18), 0.65, (90, 180, 255), 2)
    y0 += label_h
    img[y0 : y0 + ph, 40 : 40 + pw] = B
    return img


def panel_pair_horizontal(
    left: np.ndarray, right: np.ndarray, title: str, subtitle: str, shrink_mosaic: bool = False
) -> np.ndarray:
    """Side-by-side GT / Pred layout for square, low-resolution classification mosaics."""
    if shrink_mosaic:
        left = reduce_mosaic(left, keep=4)
        right = reduce_mosaic(right, keep=4)
    img = blank()
    put(img, title, (40, 48), 1.1, (70, 160, 255), 2)
    put(img, subtitle, (40, 88), 0.75, (180, 180, 190), 1)
    mid_y, gap = 115, 20
    label_h = 30
    pw = (W - 80 - gap) // 2
    ph = H - mid_y - label_h - 25
    L = fit(left, pw, ph)
    R = fit(right, pw, ph)
    put(img, "GT / Labels", (40, mid_y + 20), 0.65, (90, 220, 140), 2)
    put(img, "Prediction", (40 + pw + gap, mid_y + 20), 0.65, (90, 180, 255), 2)
    y0 = mid_y + label_h
    img[y0 : y0 + ph, 40 : 40 + pw] = L
    img[y0 : y0 + ph, 40 + pw + gap : 40 + pw + gap + pw] = R
    return img


def panel_pair_auto(left: np.ndarray, right: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    """Choose layout from source geometry: square/portrait → horizontal, wide → vertical."""
    aspect = ((left.shape[1] / left.shape[0]) + (right.shape[1] / right.shape[0])) / 2
    high_res_mosaic = min(left.shape[:2]) >= 1200 and min(right.shape[:2]) >= 1200
    if aspect <= 1.25:
        # ImageNet 896² stays intact; Depth 1920² is cropped to a readable 2x2.
        return panel_pair_horizontal(left, right, title, subtitle, shrink_mosaic=high_res_mosaic)
    # Wide Detect/Semantic/OBB mosaics retain width by stacking vertically.
    return panel_pair(left, right, title, subtitle, shrink_mosaic=True)


def load_class_legend_items(*json_paths: Path) -> list[tuple[int, str]]:
    """Union class ids/names from GT and Pred sidecar JSON files."""
    items: dict[int, str] = {}
    for p in json_paths:
        if not p.exists():
            continue
        meta = json.loads(p.read_text())
        names = {int(k): v for k, v in meta.get("names", {}).items()}
        for c in meta.get("classes", []):
            cid = int(c)
            items[cid] = names.get(cid, str(cid))
    return sorted(items.items(), key=lambda x: x[0])


def render_legend(items: list[tuple[int, str]], height: int, width: int = 240) -> np.ndarray:
    """Small blank legend panel: color swatch matches Ultralytics box/mask colors."""
    from ultralytics.utils.plotting import colors

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (28, 28, 32)
    put(img, "Legend", (12, 28), 0.7, (200, 200, 210), 2)
    if not items:
        put(img, "(no classes)", (12, 60), 0.55, (140, 140, 150), 1)
        return img

    # Fit entries into available height
    top = 48
    bottom_pad = 12
    avail = height - top - bottom_pad
    row_h = max(18, min(34, avail // max(1, len(items))))
    max_rows = max(1, avail // row_h)
    shown = items[:max_rows]
    for i, (cid, name) in enumerate(shown):
        y = top + i * row_h
        bgr = colors(cid, True)
        cv2.rectangle(img, (12, y + 4), (34, y + min(22, row_h - 4)), bgr, -1, cv2.LINE_AA)
        cv2.rectangle(img, (12, y + 4), (34, y + min(22, row_h - 4)), (220, 220, 220), 1, cv2.LINE_AA)
        label = name if len(name) <= 18 else name[:17] + "…"
        put(img, label, (42, y + min(20, row_h - 2)), 0.5, (220, 220, 225), 1)
    if len(items) > max_rows:
        put(img, f"+{len(items) - max_rows} more", (12, height - 16), 0.45, (150, 150, 160), 1)
    return img


def panel_pair_with_legend(
    gt: np.ndarray,
    pred: np.ndarray,
    title: str,
    subtitle: str,
    legend_items: list[tuple[int, str]],
    shrink_mosaic: bool = True,
) -> np.ndarray:
    """GT/Pred stacked vertically; class color legend on the left."""
    if shrink_mosaic:
        gt = reduce_mosaic(gt, keep=4)
        pred = reduce_mosaic(pred, keep=4)

    img = blank()
    put(img, title, (40, 42), 1.0, (70, 160, 255), 2)
    put(img, subtitle, (40, 78), 0.7, (180, 180, 190), 1)

    mid_y, gap = 95, 12
    label_h = 28
    legend_w = 240
    usable = H - mid_y - 20
    content_h = usable
    content_w = W - 80 - legend_w - 16
    ph = (content_h - gap - 2 * label_h) // 2

    legend = render_legend(legend_items, height=content_h, width=legend_w)
    img[mid_y : mid_y + content_h, 40 : 40 + legend_w] = legend

    x0 = 40 + legend_w + 16
    T = fit(gt, content_w, ph)
    B = fit(pred, content_w, ph)
    y0 = mid_y
    put(img, "GT / Labels", (x0, y0 + 18), 0.65, (90, 220, 140), 2)
    y0 += label_h
    img[y0 : y0 + ph, x0 : x0 + content_w] = T
    y0 += ph + gap
    put(img, "Prediction", (x0, y0 + 18), 0.65, (90, 180, 255), 2)
    y0 += label_h
    img[y0 : y0 + ph, x0 : x0 + content_w] = B
    return img


def panel_pair_auto_detection(gt: np.ndarray, pred: np.ndarray, title: str, subtitle: str, legend_items: list[tuple[int, str]]) -> np.ndarray:
    """Detect / Segment / OBB: always use left legend + vertical GT/Pred (wide mosaics)."""
    return panel_pair_with_legend(gt, pred, title, subtitle, legend_items, shrink_mosaic=True)


def panel_single(vis: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    img = blank()
    put(img, title, (40, 48), 1.1, (70, 160, 255), 2)
    put(img, subtitle, (40, 88), 0.75, (180, 180, 190), 1)
    mid_y = 110
    body = fit(vis, W - 80, H - mid_y - 40)
    img[mid_y : mid_y + body.shape[0], 40 : 40 + body.shape[1]] = body
    return img


def load(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    im = cv2.imread(str(path))
    return im


def find_named_run(name: str) -> Path:
    """Locate a val run directory by name under runs/, ignoring extra YOLO task nesting."""
    hits = [p for p in ROOT.glob(f"runs/**/{name}") if p.is_dir() and any(p.glob("val_batch*_pred.jpg"))]
    if hits:
        # Prefer the shortest path (least accidental nesting).
        return sorted(hits, key=lambda p: (len(p.parts), str(p)))[0]
    return ROOT / "runs" / name


def render_pose_gt(image_path: Path, output_path: Path) -> Path:
    """Render COCO person boxes/keypoints on the original image for GT comparison."""
    if output_path.exists():
        return output_path
    data = json.loads((DATASETS / "coco-pose/annotations/person_keypoints_val2017.json").read_text())
    image_id = int(image_path.stem)
    annotations = [a for a in data["annotations"] if a["image_id"] == image_id and a.get("num_keypoints", 0) > 0]
    image = cv2.imread(str(image_path))
    # Standard COCO-17 skeleton, converted from 1-based to 0-based.
    skeleton = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
        (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
        (3, 5), (4, 6),
    ]
    for ann in annotations:
        x, y, w, h = map(int, ann["bbox"])
        cv2.rectangle(image, (x, y), (x + w, y + h), (80, 220, 120), 2)
        put(image, "person GT", (x, max(18, y - 5)), 0.55, (80, 220, 120), 2)
        keypoints = np.asarray(ann["keypoints"], dtype=float).reshape(17, 3)
        for a, b in skeleton:
            if keypoints[a, 2] > 0 and keypoints[b, 2] > 0:
                pa = tuple(keypoints[a, :2].astype(int))
                pb = tuple(keypoints[b, :2].astype(int))
                cv2.line(image, pa, pb, (255, 120, 80), 2, cv2.LINE_AA)
        for px, py, visible in keypoints:
            if visible > 0:
                cv2.circle(image, (int(px), int(py)), 4, (70, 255, 255), -1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return output_path


def pick(files: list[Path], n: int) -> list[Path]:
    files = [p for p in files if p.is_file()]
    if len(files) <= n:
        return files
    return rng.sample(files, n)


def run_predict(task: str, model: str, sources: list[Path], out_name: str, **kwargs) -> list[Path]:
    """Run predict and return saved image paths."""
    dest = PRED / out_name
    if dest.exists() and any(dest.glob("*.jpg")):
        return sorted(dest.glob("*.jpg")) + sorted(dest.glob("*.png"))
    dest.mkdir(parents=True, exist_ok=True)
    m = YOLO(str(ROOT / model) if (ROOT / model).exists() else model)
    # write into dest via project/name relative to cwd
    project = str(PRED)
    name = out_name
    for i, src in enumerate(sources):
        m.predict(
            source=str(src),
            project=project,
            name=name,
            exist_ok=True,
            save=True,
            verbose=False,
            **kwargs,
        )
    return sorted(Path(project, name).glob("*.jpg")) + sorted(Path(project, name).glob("*.png"))


def emit(frames: list[np.ndarray], frame: np.ndarray, seconds: float = HOLD) -> None:
    n = max(1, int(round(seconds * FPS)))
    for _ in range(n):
        frames.append(frame)


def main() -> None:
    ensure_dirs()
    frames: list[np.ndarray] = []

    # --- opening ---
    emit(
        frames,
        banner(
            "YOLO26n Official Val — Qualitative Tour",
            [
                "Local full-val metrics vs Ultralytics docs (same protocol).",
                "Detect / Seg / Pose / Semantic / Classify / Depth / OBB",
                "Aligned where open protocol exists; Depth TTA & OBB test noted.",
            ],
        ),
        4,
    )
    emit(frames, summary_table(), 6)

    # Clean no-label plots live under results_video/clean (see render_clean_batches.py)
    clean = ROOT / "runs/official_val/results_video/clean"

    # --- Detect (clean val batches: colored boxes + left legend, no text labels) ---
    det_e2e = next(clean.glob("**/det_coco_e2e"), clean / "det_coco_e2e")
    det_o2m = next(clean.glob("**/det_coco_o2m"), clean / "det_coco_o2m")
    emit(frames, banner("Detect — COCO val", ["e2e AP 40.0 ~ official 40.1", "end2end=False AP 40.8 ~ 40.9", "Boxes colored by class; legend on the left"]), 3)
    for i in range(3):
        lab, pred = load(det_e2e / f"val_batch{i}_labels.jpg"), load(det_e2e / f"val_batch{i}_pred.jpg")
        if lab is not None and pred is not None:
            items = load_class_legend_items(det_e2e / f"val_batch{i}_labels.classes.json", det_e2e / f"val_batch{i}_pred.classes.json")
            emit(frames, panel_pair_auto_detection(lab, pred, "Detect (e2e)", f"AP 40.0 vs 40.1  |  val_batch{i}", items), 3)
    for i in range(3):
        lab, pred = load(det_o2m / f"val_batch{i}_labels.jpg"), load(det_o2m / f"val_batch{i}_pred.jpg")
        if lab is not None and pred is not None:
            items = load_class_legend_items(det_o2m / f"val_batch{i}_labels.classes.json", det_o2m / f"val_batch{i}_pred.classes.json")
            emit(frames, panel_pair_auto_detection(lab, pred, "Detect (end2end=False / o2m+NMS)", f"AP 40.8 vs 40.9  |  val_batch{i}", items), 3)

    # --- Segment (clean val batches) ---
    seg_dir = next(clean.glob("**/seg_coco_e2e"), clean / "seg_coco_e2e")
    emit(frames, banner("Segment — COCO val e2e", ["box 39.8 / mask 33.9  ~  official 39.6 / 33.9", "Masks/boxes colored by class; legend on the left"]), 3)
    for i in range(3):
        lab, pred = load(seg_dir / f"val_batch{i}_labels.jpg"), load(seg_dir / f"val_batch{i}_pred.jpg")
        if lab is not None and pred is not None:
            items = load_class_legend_items(seg_dir / f"val_batch{i}_labels.classes.json", seg_dir / f"val_batch{i}_pred.classes.json")
            emit(frames, panel_pair_auto_detection(lab, pred, "Instance Segment", f"box/mask 39.8/33.9  |  val_batch{i}", items), 3)

    # --- Pose: curated val images with visible, successfully predicted people ---
    pose_out = PRED / "pose_e2e"
    pose_ids = [151629, 169996, 301061, 524456, 570471, 6763]
    pose_root = DATASETS / "coco-pose/images/val2017"
    pose_imgs = [pose_root / f"{iid:012d}.jpg" for iid in pose_ids]
    expected = {p.name for p in pose_imgs}
    existing = {p.name for p in pose_out.glob("*.jpg")} if pose_out.exists() else set()
    if existing != expected:
        if pose_out.exists():
            for old in pose_out.glob("*"):
                old.unlink()
        run_predict("pose", "yolo26n-pose.pt", pose_imgs, "pose_e2e", imgsz=640, device=0)
    pose_preds = [pose_out / p.name for p in pose_imgs]
    pose_gt_dir = PRED / "pose_gt"
    pose_gts = [render_pose_gt(src, pose_gt_dir / src.name) for src in pose_imgs]
    emit(frames, banner("Pose — COCO-pose val e2e", ["pose AP 57.2 = official 57.2"]), 3)
    for gt_path, pred_path in zip(pose_gts, pose_preds):
        gt, pred = load(gt_path), load(pred_path)
        if gt is not None and pred is not None:
            emit(frames, panel_pair_auto(gt, pred, "Pose", f"AP 57.2 = 57.2  |  {pred_path.name}"), 2.5)

    # --- Semantic (existing) ---
    sem = find_named_run("sem_cityscapes")
    emit(frames, banner("Semantic — Cityscapes val", ["imgsz=2048", "mIoU 78.3 = official 78.3"]), 3)
    for i in range(3):
        lab, pred = load(sem / f"val_batch{i}_labels.jpg"), load(sem / f"val_batch{i}_pred.jpg")
        if lab is not None and pred is not None:
            emit(frames, panel_pair_auto(lab, pred, "Semantic Segmentation", f"mIoU 78.3 = 78.3  |  val_batch{i}"), 3)

    # --- Classify ---
    cls = find_named_run("cls_imagenet")
    emit(frames, banner("Classify — ImageNet val", ["top1 71.4 / top5 90.1 = official"]), 3)
    for i in range(3):
        lab, pred = load(cls / f"val_batch{i}_labels.jpg"), load(cls / f"val_batch{i}_pred.jpg")
        if lab is not None and pred is not None:
            emit(frames, panel_pair_auto(lab, pred, "ImageNet Classification", f"71.4 / 90.1  |  val_batch{i}"), 3)

    # --- Depth (val batches: GT + Prediction) ---
    depth_single = find_named_run("depth_single")
    depth_tta = find_named_run("depth_tta_logls")
    emit(
        frames,
        banner(
            "Depth — NYU Eigen",
            [
                "Single-scale + median: d1 0.783 = official 0.783",
                "TTA + log-LS (local approx): ~0.839 vs official 0.882",
            ],
        ),
        4,
    )
    for i in range(3):
        lab, pred = load(depth_single / f"val_batch{i}_labels.jpg"), load(depth_single / f"val_batch{i}_pred.jpg")
        if lab is not None and pred is not None:
            emit(frames, panel_pair_auto(lab, pred, "Depth (single-scale)", f"d1 0.783 = 0.783  |  val_batch{i}"), 3)
    for i in range(3):
        lab, pred = load(depth_tta / f"val_batch{i}_labels.jpg"), load(depth_tta / f"val_batch{i}_pred.jpg")
        if lab is not None and pred is not None:
            emit(frames, panel_pair_auto(lab, pred, "Depth (TTA + log-LS approx)", f"~0.839 vs 0.882  |  val_batch{i}"), 3)

    # --- OBB (clean val batches) ---
    obb = next(clean.glob("**/obb_val"), clean / "obb_val")
    emit(
        frames,
        banner(
            "OBB — DOTAv1",
            [
                "val single-scale local mAP50-95 ~43.9 (no public val table)",
                "official 52.4 is test multi-scale → eval server only",
                "Oriented boxes colored by class; legend on the left",
            ],
        ),
        4,
    )
    for i in range(3):
        lab, pred = load(obb / f"val_batch{i}_labels.jpg"), load(obb / f"val_batch{i}_pred.jpg")
        if lab is not None and pred is not None:
            items = load_class_legend_items(obb / f"val_batch{i}_labels.classes.json", obb / f"val_batch{i}_pred.classes.json")
            emit(frames, panel_pair_auto_detection(lab, pred, "OBB val (qualitative)", f"local ~43.9  |  val_batch{i}", items), 3)

    # --- close ---
    emit(
        frames,
        banner(
            "Takeaway",
            [
                "Open reproducible protocols: metrics align; visuals above.",
                "Remaining gaps: Depth headline TTA script; OBB test GT/server.",
                "Docs: docs/official_example/  |  this video for qualitative check",
            ],
        ),
        5,
    )

    # write frame sequence + encode
    for i, fr in enumerate(frames):
        cv2.imwrite(str(FRAMES / f"{i:05d}.jpg"), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(VIDEO), fourcc, FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {VIDEO}")
    for fr in frames:
        writer.write(fr)
    writer.release()

    # also try ffmpeg re-encode to H.264 if available
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        h264 = VIDEO.with_name(VIDEO.stem + "_h264.mp4")
        subprocess.run(
            [ffmpeg, "-y", "-i", str(VIDEO), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(h264)],
            check=False,
        )
        if h264.exists() and h264.stat().st_size > 0:
            h264.replace(VIDEO)

    print(f"frames={len(frames)} duration_s≈{len(frames)/FPS:.1f}")
    print(f"video={VIDEO} size={VIDEO.stat().st_size}")


if __name__ == "__main__":
    main()
