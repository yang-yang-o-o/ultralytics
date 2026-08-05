#!/usr/bin/env python3
"""Convert BOP IC-BIN to Ultralytics seg6d layout (pose 9-pts + seg polygons).

Input (BOP):
  <bop_root>/{train,test}/<scene_id>/{rgb,mask,mask_visib,scene_gt.json,...}
  <bop_root>/models/models_info.json

Output (same convention as beer yolo_seg6d):
  <out>/images/{train,val}/<scene>_<im>.png   (symlink)
  <out>/labels/{train,val}/...                # cls cx cy w h + 9*(x,y)  [norm]
  <out>/labels_seg/{train,val}/...            # cls + polygon [norm]

Detection box = seg polygon AABB (not 9-pt AABB).
9 control points = object AABB center + 8 corners (YOLO6D order), projected
with per-frame K and GT cam_R_m2c / cam_t_m2c (BOP units: mm).

Class ids: BOP obj_id (1..N) → YOLO cls (0..N-1).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

# YOLO6D edge list for optional debug (center=0, corners=1..8)
EDGES_CORNERS = [
    (1, 2),
    (1, 3),
    (1, 5),
    (2, 4),
    (2, 6),
    (3, 4),
    (3, 7),
    (4, 8),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 8),
]


def load_models_info(path: Path) -> dict[int, dict]:
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def corners_3d_from_info(info: dict) -> np.ndarray:
    """Return (4, 9) homogeneous AABB points in object frame (YOLO6D order)."""
    min_x = float(info["min_x"])
    min_y = float(info["min_y"])
    min_z = float(info["min_z"])
    max_x = min_x + float(info["size_x"])
    max_y = min_y + float(info["size_y"])
    max_z = min_z + float(info["size_z"])
    corners = np.array(
        [
            [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0],
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ],
        dtype=np.float64,
    )
    return np.concatenate((corners.T, np.ones((1, 9), dtype=np.float64)), axis=0)


def project_corners(
    corners_h: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Project (4,9) → (9,2) pixel coords."""
    rt = np.concatenate((R.reshape(3, 3), t.reshape(3, 1)), axis=1)
    cam = (K @ rt) @ corners_h  # (3, 9)
    return np.vstack((cam[0] / cam[2], cam[1] / cam[2])).T  # (9, 2)


def mask_to_polygon(mask: np.ndarray, epsilon_ratio: float = 0.0005) -> np.ndarray | None:
    """Largest contour → (N,2) pixel polygon, or None if empty."""
    if mask is None or mask.size == 0 or int((mask > 0).sum()) < 8:
        return None
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 8:
        return None
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon_ratio * peri, True)
    if len(approx) < 3:
        return None
    return approx.reshape(-1, 2).astype(np.float64)


def poly_to_xywh_norm(poly_xy: np.ndarray, w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1 = poly_xy.min(axis=0)
    x2, y2 = poly_xy.max(axis=0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    return cx / w, cy / h, bw / w, bh / h


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def process_scene(
    scene_dir: Path,
    split_out: str,
    out: Path,
    corners_by_obj: dict[int, np.ndarray],
    mask_subdir: str,
    min_visib: float,
    epsilon_ratio: float,
    max_images: int = 0,
) -> tuple[int, int]:
    """Convert one BOP scene. Returns (n_images_written, n_instances_written)."""
    scene_id = int(scene_dir.name)
    gt_all = json.loads((scene_dir / "scene_gt.json").read_text())
    info_all = json.loads((scene_dir / "scene_gt_info.json").read_text())
    cam_all = json.loads((scene_dir / "scene_camera.json").read_text())

    n_img = 0
    n_inst = 0
    for im_key, gts in gt_all.items():
        if max_images > 0 and n_img >= max_images:
            break
        im_id = int(im_key)
        infos = info_all[im_key]
        cam = cam_all[im_key]
        K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)

        rgb_path = scene_dir / "rgb" / f"{im_id:06d}.png"
        if not rgb_path.exists():
            # some packs use .jpg
            rgb_path = scene_dir / "rgb" / f"{im_id:06d}.jpg"
        if not rgb_path.exists():
            continue

        img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = f"{scene_id:06d}_{im_id:06d}"

        pose_lines: list[str] = []
        seg_lines: list[str] = []

        for inst_idx, (gt, info) in enumerate(zip(gts, infos)):
            visib = float(info.get("visib_fract", 1.0))
            if visib < min_visib:
                continue
            if int(info.get("px_count_visib", 0)) < 8:
                continue

            obj_id = int(gt["obj_id"])
            if obj_id not in corners_by_obj:
                continue
            cls = obj_id - 1  # YOLO 0-based

            mask_path = scene_dir / mask_subdir / f"{im_id:06d}_{inst_idx:06d}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            poly = mask_to_polygon(mask, epsilon_ratio=epsilon_ratio)
            if poly is None:
                continue

            R = np.array(gt["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            t = np.array(gt["cam_t_m2c"], dtype=np.float64).reshape(3, 1)
            pts2d = project_corners(corners_by_obj[obj_id], R, t, K)  # (9,2)

            # keep points that are mostly in-frame (center at least)
            cx_px, cy_px = pts2d[0]
            if not ( -0.5 * w < cx_px < 1.5 * w and -0.5 * h < cy_px < 1.5 * h):
                continue

            cx, cy, bw, bh = poly_to_xywh_norm(poly, w, h)
            pts_n = pts2d.copy()
            pts_n[:, 0] /= w
            pts_n[:, 1] /= h
            # Ultralytics label cache rejects any coord outside [0,1]; AABB corners
            # often sit slightly OOB — clip for loader compatibility (viz/PnP still OK).
            pts_n = np.clip(pts_n, 0.0, 1.0)
            kpt_str = " ".join(f"{v:.6f}" for v in pts_n.reshape(-1))
            pose_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kpt_str}")

            poly_n = poly.copy()
            poly_n[:, 0] /= w
            poly_n[:, 1] /= h
            poly_n = np.clip(poly_n, 0.0, 1.0)
            seg_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly_n)
            seg_lines.append(f"{cls} {seg_str}")

        if not pose_lines:
            continue

        link(rgb_path, out / "images" / split_out / f"{stem}{rgb_path.suffix}")
        (out / "labels" / split_out).mkdir(parents=True, exist_ok=True)
        (out / "labels_seg" / split_out).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split_out / f"{stem}.txt").write_text("\n".join(pose_lines) + "\n")
        (out / "labels_seg" / split_out / f"{stem}.txt").write_text("\n".join(seg_lines) + "\n")
        n_img += 1
        n_inst += len(pose_lines)

    return n_img, n_inst


def clear_split(out: Path, split: str) -> None:
    """Remove previous images/labels for a split so rebuild is clean."""
    for sub in ("images", "labels", "labels_seg"):
        d = out / sub / split
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.is_dir():
                # unexpected
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--bop-root",
        type=Path,
        default=Path("/root/ultralytics/YOLO6D/datasets/bop/icbin"),
        help="Extracted IC-BIN root (models/, train/, test/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output yolo_seg6d dir (default: <bop-root>/yolo_seg6d)",
    )
    parser.add_argument(
        "--mask",
        choices=("full", "visib"),
        default="full",
        help="Use BOP mask/ (amodal) or mask_visib/ (visible). Default: full",
    )
    parser.add_argument("--min-visib", type=float, default=0.1, help="Skip instances with visib_fract < this")
    parser.add_argument("--epsilon-ratio", type=float, default=0.0005, help="approxPolyDP epsilon / perimeter")
    parser.add_argument(
        "--train-split",
        default="train",
        help="BOP folder name used as YOLO train (default: train). Use train_pbr later if present.",
    )
    parser.add_argument("--val-split", default="test", help="BOP folder name used as YOLO val (default: test)")
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="If >0, only process first N scenes of each split (sorted by id).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If >0, stop after writing this many train images (val unaffected).",
    )
    parser.add_argument("--fresh-train", action="store_true", help="Clear existing train split before writing")
    args = parser.parse_args()

    bop: Path = args.bop_root
    out: Path = args.out or (bop / "yolo_seg6d")
    models_info = load_models_info(bop / "models" / "models_info.json")
    corners_by_obj = {oid: corners_3d_from_info(info) for oid, info in models_info.items()}
    mask_subdir = "mask" if args.mask == "full" else "mask_visib"

    print(f"bop_root={bop}")
    print(f"out={out}")
    print(f"objects={sorted(models_info)} mask={mask_subdir} min_visib={args.min_visib}")

    if args.fresh_train:
        clear_split(out, "train")
        print("cleared previous train split")

    totals: dict[str, tuple[int, int]] = {}
    for yolo_split, bop_split in [("train", args.train_split), ("val", args.val_split)]:
        split_dir = bop / bop_split
        if not split_dir.is_dir():
            raise FileNotFoundError(split_dir)
        n_img = n_inst = 0
        scenes = sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit())
        if args.max_scenes > 0:
            scenes = scenes[: args.max_scenes]
        for scene in scenes:
            if yolo_split == "train" and args.max_images > 0 and n_img >= args.max_images:
                break
            a, b = process_scene(
                scene,
                yolo_split,
                out,
                corners_by_obj,
                mask_subdir,
                args.min_visib,
                args.epsilon_ratio,
                max_images=(args.max_images - n_img) if (yolo_split == "train" and args.max_images > 0) else 0,
            )
            n_img += a
            n_inst += b
            print(f"  {bop_split}/{scene.name}: images={a} instances={b}")
        totals[yolo_split] = (n_img, n_inst)
        print(f"{yolo_split}: images={n_img} instances={n_inst}")

    for cache in out.rglob("*.cache"):
        cache.unlink()
        print(f"removed cache {cache}")

    # list files for convenience
    for split, (n_img, _) in totals.items():
        img_dir = out / "images" / split
        imgs = sorted([*img_dir.glob("*.png"), *img_dir.glob("*.jpg")])
        list_path = bop / f"{split}_seg6d.txt"
        list_path.write_text("\n".join(str(p.resolve()) for p in imgs) + ("\n" if imgs else ""))
        print(f"wrote {list_path} ({len(imgs)} lines)")

    print("done.")
    sample = next((out / "labels" / "train").glob("*.txt"), None)
    if sample is not None:
        print(f"sample pose: {sample.read_text().strip()[:160]}")


if __name__ == "__main__":
    main()
