#!/usr/bin/env python3
"""Visualize IC-BIN seg6d GT: RGB + mask polygon + 9 control points + AABB edges.

Samples a few train/val images and writes side-by-side check images.
Also overlays BOP-projected corners recomputed from scene_gt (green) vs
label file keypoints (cyan) to catch conversion bugs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from icbin_seg6d_prepare import (  # noqa: E402
    EDGES_CORNERS,
    corners_3d_from_info,
    load_models_info,
    mask_to_polygon,
    project_corners,
)


def draw_pose(img: np.ndarray, pts: np.ndarray, color=(0, 255, 0), center_color=(0, 165, 255)) -> None:
    """pts: (9,2) pixels. Draw edges among corners 1..8 and center."""
    pts_i = np.round(pts).astype(int)
    for a, b in EDGES_CORNERS:
        cv2.line(img, tuple(pts_i[a]), tuple(pts_i[b]), color, 1, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts_i):
        c = center_color if i == 0 else color
        cv2.circle(img, (int(x), int(y)), 3 if i == 0 else 2, c, -1, cv2.LINE_AA)


def draw_poly(img: np.ndarray, poly: np.ndarray, color=(255, 128, 0)) -> None:
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)


def parse_pose_label(line: str, w: int, h: int) -> tuple[int, np.ndarray, tuple[float, float, float, float]]:
    vals = list(map(float, line.split()))
    cls = int(vals[0])
    cx, cy, bw, bh = vals[1:5]
    pts = np.array(vals[5:23], dtype=np.float64).reshape(9, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    box = (cx * w, cy * h, bw * w, bh * h)
    return cls, pts, box


def parse_seg_label(line: str, w: int, h: int) -> tuple[int, np.ndarray]:
    vals = list(map(float, line.split()))
    cls = int(vals[0])
    poly = np.array(vals[1:], dtype=np.float64).reshape(-1, 2)
    poly[:, 0] *= w
    poly[:, 1] *= h
    return cls, poly


def bop_reproject_frame(
    bop_root: Path,
    stem: str,
    corners_by_obj: dict[int, np.ndarray],
    mask_subdir: str = "mask",
    min_visib: float = 0.1,
    prefer_split: str | None = None,
) -> list[tuple[int, np.ndarray, np.ndarray | None]]:
    """Recompute projections from original BOP GT for stem=scene_im.

    Applies the same visibility / mask filters as prepare, so instance order
    matches converted label lines.
    """
    scene_s, im_s = stem.split("_", 1)
    scene_id, im_id = int(scene_s), int(im_s)
    scene_dir = None
    splits = [prefer_split] if prefer_split else []
    splits += [s for s in ("train", "test", "train_pbr") if s not in splits]
    for split in splits:
        if split is None:
            continue
        cand = bop_root / split / f"{scene_id:06d}"
        if cand.is_dir():
            scene_dir = cand
            break
    if scene_dir is None:
        return []
    gt_all = json.loads((scene_dir / "scene_gt.json").read_text())
    info_all = json.loads((scene_dir / "scene_gt_info.json").read_text())
    cam_all = json.loads((scene_dir / "scene_camera.json").read_text())
    gts = gt_all[str(im_id)]
    infos = info_all[str(im_id)]
    cam = cam_all[str(im_id)]
    K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)
    out = []
    for inst_idx, (gt, info) in enumerate(zip(gts, infos)):
        if float(info.get("visib_fract", 1.0)) < min_visib:
            continue
        if int(info.get("px_count_visib", 0)) < 8:
            continue
        obj_id = int(gt["obj_id"])
        if obj_id not in corners_by_obj:
            continue
        mask_path = scene_dir / mask_subdir / f"{im_id:06d}_{inst_idx:06d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        poly = mask_to_polygon(mask) if mask is not None else None
        if poly is None:
            continue
        R = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        t = np.array(gt["cam_t_m2c"], dtype=np.float64).reshape(3, 1)
        pts = project_corners(corners_by_obj[obj_id], R, t, K)
        out.append((obj_id - 1, pts, poly))
    return out


def viz_one(
    img_path: Path,
    pose_path: Path,
    seg_path: Path,
    bop_root: Path,
    corners_by_obj: dict[int, np.ndarray],
    out_path: Path,
    prefer_split: str | None = None,
) -> list[float]:
    img = cv2.imread(str(img_path))
    assert img is not None, img_path
    h, w = img.shape[:2]
    left = img.copy()
    right = img.copy()

    # Left: from converted labels
    pose_lines = pose_path.read_text().strip().splitlines() if pose_path.exists() else []
    seg_lines = seg_path.read_text().strip().splitlines() if seg_path.exists() else []
    for pl, sl in zip(pose_lines, seg_lines):
        cls, pts, (cx, cy, bw, bh) = parse_pose_label(pl, w, h)
        _, poly = parse_seg_label(sl, w, h)
        draw_poly(left, poly, color=(255, 128, 0))
        draw_pose(left, pts, color=(0, 255, 255), center_color=(0, 140, 255))
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(left, f"cls{cls}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Right: recomputed from BOP GT (same filters as prepare)
    bop_inst = bop_reproject_frame(bop_root, img_path.stem, corners_by_obj, prefer_split=prefer_split)
    for cls, pts, poly in bop_inst:
        if poly is not None:
            draw_poly(right, poly, color=(255, 128, 0))
        draw_pose(right, pts, color=(0, 255, 0), center_color=(0, 165, 255))
        cv2.putText(right, f"cls{cls}", tuple(np.round(pts[0]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    diffs: list[float] = []
    for pl, (_, pts_b, _) in zip(pose_lines, bop_inst):
        _, pts_l, _ = parse_pose_label(pl, w, h)
        diffs.append(float(np.linalg.norm(pts_l - pts_b, axis=1).mean()))
    diff_txt = f"mean_kpt_err={np.mean(diffs):.3f}px n={len(diffs)}" if diffs else "no_pair"

    canvas = np.concatenate([left, right], axis=1)
    cv2.putText(canvas, f"LABELS ({img_path.stem}) {diff_txt}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "BOP reproject", (w + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bop-root", type=Path, default=Path("/root/YOLO6D/datasets/bop/icbin"))
    parser.add_argument("--yolo-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-train", type=int, default=4)
    parser.add_argument("--n-val", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bop = args.bop_root
    yolo = args.yolo_root or (bop / "yolo_seg6d")
    out_dir = args.out_dir or (yolo / "viz_gt")
    models_info = load_models_info(bop / "models" / "models_info.json")
    corners_by_obj = {oid: corners_3d_from_info(info) for oid, info in models_info.items()}

    rng = random.Random(args.seed)
    all_errs: list[float] = []

    # yolo val comes from BOP test/; train from BOP train/
    prefer = {"train": "train", "val": "test"}
    for split, n in [("train", args.n_train), ("val", args.n_val)]:
        imgs = sorted((yolo / "images" / split).glob("*.png"))
        if not imgs:
            print(f"no images in {split}")
            continue
        # prefer multi-instance for val: pick by label line count
        scored = []
        for p in imgs:
            n_lines = len((yolo / "labels" / split / f"{p.stem}.txt").read_text().strip().splitlines())
            scored.append((n_lines, p))
        scored.sort(key=lambda x: -x[0])
        picks = [p for _, p in scored[: max(1, n // 2)]]
        rest = [p for _, p in scored[max(1, n // 2) :]]
        rng.shuffle(rest)
        picks += rest[: max(0, n - len(picks))]
        for p in picks:
            pose_p = yolo / "labels" / split / f"{p.stem}.txt"
            seg_p = yolo / "labels_seg" / split / f"{p.stem}.txt"
            out_p = out_dir / split / f"{p.stem}.jpg"
            diffs = viz_one(
                p, pose_p, seg_p, bop, corners_by_obj, out_p, prefer_split=prefer.get(split)
            )
            all_errs.extend(diffs)
            print(f"wrote {out_p}")

    if all_errs:
        print(f"label vs BOP reproject: mean={np.mean(all_errs):.4f}px  max={np.max(all_errs):.4f}px  n={len(all_errs)}")
    print(f"viz dir: {out_dir}")


if __name__ == "__main__":
    main()
