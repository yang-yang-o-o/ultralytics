#!/usr/bin/env python3
"""Native-res val visualization for yolo26s-seg-6dpose.

Draws instance masks + YOLO6D 9 control points + AABB wireframe (and optional
PnP-reprojected corners) at original beer resolution (964×1292).

Outputs under the training run:
  runs/segment/beer-seg/yolo26s-seg-6dpose/predict-val-native-retina/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from ultralytics import YOLO
from ultralytics.models.yolo.seg6d.utils import camera_matrix, get_3d_corners, load_mesh_corners, pnp, project_points

# YOLO6D AABB edges among the 8 corners (indices 1..8; 0 is center)
EDGES = (
    np.array(
        [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]],
        dtype=np.int32,
    )
    + 1
)

RUN = Path("/root/ultralytics/runs/segment/beer-seg/yolo26s-seg-6dpose-v2")
SOURCE = Path("/root/ultralytics/YOLO6D/data/beer/yolo_seg6d/images/val")
DATA_YAML = Path("/root/ultralytics/ultralytics/cfg/datasets/beer-seg6d.yaml")
IMGSZ = 1312


def draw_pose6d(
    bgr: np.ndarray,
    kpts_xy: np.ndarray,
    color_pts=(0, 255, 0),
    color_edge=(0, 200, 255),
    color_center=(0, 0, 255),
) -> np.ndarray:
    """Draw 9 points (center+8 corners) and AABB edges on BGR image."""
    out = bgr
    pts = np.asarray(kpts_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 9:
        return out
    for i, j in EDGES:
        p1 = tuple(np.round(pts[i]).astype(int))
        p2 = tuple(np.round(pts[j]).astype(int))
        cv2.line(out, p1, p2, color_edge, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts[:9]):
        c = color_center if i == 0 else color_pts
        cv2.circle(out, (int(round(x)), int(round(y))), 5 if i == 0 else 4, c, -1, cv2.LINE_AA)
        cv2.circle(out, (int(round(x)), int(round(y))), 5 if i == 0 else 4, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_pnp_reproj(bgr: np.ndarray, kpts_xy: np.ndarray, corners3d: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Solve PnP from predicted 9 pts and overlay reprojected AABB (magenta)."""
    try:
        r, t = pnp(corners3d[:3].T, kpts_xy.reshape(9, 2), k)
        rt = np.concatenate((r, t), axis=1)
        proj = project_points(corners3d, rt, k).T  # (9,2)
    except Exception:
        return bgr
    return draw_pose6d(bgr, proj, color_pts=(255, 0, 255), color_edge=(255, 0, 180), color_center=(255, 128, 0))


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, color=(0, 180, 255), alpha=0.35) -> np.ndarray:
    """Alpha-blend a binary mask onto BGR."""
    out = bgr.copy()
    m = mask.astype(bool)
    if m.ndim == 3:
        m = m.any(axis=0) if m.shape[0] in {1, 3} else m[..., 0]
    if m.shape[:2] != out.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    out[m] = (out[m].astype(np.float32) * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=Path,
        default=RUN,
        help="Training run directory containing weights/",
    )
    parser.add_argument("--weights", default="best", choices=["last", "best"], help="Which checkpoint to visualize")
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--no-pnp", action="store_true", help="Skip PnP reprojected overlay")
    args = parser.parse_args()

    run = args.run
    weights = run / "weights" / f"{args.weights}.pt"
    out_name = f"predict-val-native-retina-{args.weights}"
    out_dir = run / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(DATA_YAML) as f:
        data = yaml.safe_load(f)
    k = camera_matrix(float(data["fx"]), float(data["fy"]), float(data["u0"]), float(data["v0"]))
    _, corners3d = load_mesh_corners(data["mesh"])

    print(f"weights: {weights}")
    print(f"source:  {SOURCE}")
    print(f"out:     {out_dir}")
    model = YOLO(str(weights))

    results = model.predict(
        source=str(SOURCE),
        imgsz=args.imgsz,
        device=0,
        retina_masks=True,
        conf=args.conf,
        iou=0.7,
        save=False,
        verbose=True,
    )

    n_ok = 0
    for r in results:
        # Ultralytics may give RGB or BGR depending on path; Results.orig_img is typically BGR from cv2
        img = r.orig_img.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if r.masks is not None and len(r.masks.data):
            for mi in range(len(r.masks.data)):
                m = r.masks.data[mi].cpu().numpy()
                img = overlay_mask(img, m)

        if r.boxes is not None and len(r.boxes):
            for bi in range(len(r.boxes)):
                xyxy = r.boxes.xyxy[bi].cpu().numpy().astype(int)
                conf = float(r.boxes.conf[bi])
                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (50, 220, 50), 2)
                cv2.putText(
                    img,
                    f"beer {conf:.2f}",
                    (xyxy[0], max(20, xyxy[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (50, 220, 50),
                    2,
                    cv2.LINE_AA,
                )

        if r.keypoints is not None and len(r.keypoints.data):
            for ki in range(len(r.keypoints.data)):
                kpts = r.keypoints.data[ki].cpu().numpy()
                xy = kpts[:, :2]
                img = draw_pose6d(img, xy)  # predicted 2D points (green/cyan)
                if not args.no_pnp:
                    img = draw_pnp_reproj(img, xy, corners3d, k)  # PnP reproj (magenta)

        stem = Path(r.path).stem
        # write as jpg to match prior convention; keep native HxW
        save_path = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(save_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        n_ok += 1
        if n_ok == 1:
            print(f"sample0: {stem} shape={img.shape[:2]} (H,W)")

    # legend strip note
    (out_dir / "README.txt").write_text(
        "Native-res seg6d visualization\n"
        f"weights={weights}\n"
        f"imgsz={args.imgsz} retina_masks=True\n"
        "green/cyan points+edges = predicted 9 control points\n"
        "magenta points+edges = PnP reprojection from beer.ply\n"
        "orange center (PnP) / red center (pred)\n"
        f"n_images={n_ok}\n"
    )
    print(f"Saved {n_ok} overlays -> {out_dir}")


if __name__ == "__main__":
    main()
