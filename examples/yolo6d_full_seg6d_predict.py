#!/usr/bin/env python3
"""Native-res val visualization for yolo6d_full seg6d: box + mask + 9pts + optional PnP."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from ultralytics import YOLO
from ultralytics.models.yolo.seg6d.utils import load_mesh_corners, pnp, project_points

EDGES = (
    np.array(
        [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]],
        dtype=np.int32,
    )
    + 1
)

RUN = Path("/root/ultralytics/runs/segment/yolo6d-full-seg/yolo26s-seg6d-bestcfg")
SOURCE = Path("/root/ultralytics/projects/object_6d_pose_annotation/outputs/run1/yolo6d_full/yolo_seg6d/images/val")
DATA_YAML = Path("/root/ultralytics/ultralytics/cfg/datasets/yolo6d-full-seg6d.yaml")


def draw_pose6d(bgr, kpts_xy, color_pts=(0, 255, 0), color_edge=(0, 200, 255), color_center=(0, 0, 255)):
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


def draw_pnp_reproj(bgr, kpts_xy, corners3d, k):
    try:
        r, t = pnp(corners3d[:3].T, kpts_xy.reshape(9, 2), k)
        rt = np.concatenate((r, t), axis=1)
        proj = project_points(corners3d, rt, k).T
    except Exception:
        return bgr
    return draw_pose6d(bgr, proj, color_pts=(255, 0, 255), color_edge=(255, 0, 180), color_center=(255, 128, 0))


def overlay_mask(bgr, mask, color=(0, 180, 255), alpha=0.35):
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
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--weights", default="best", choices=["last", "best"])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--imgsz", type=int, default=960, help="Match training imgsz for best recall")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all val images")
    parser.add_argument("--no-pnp", action="store_true")
    parser.add_argument("--name", default=None, help="output folder name under run/")
    args = parser.parse_args()

    run = args.run
    weights = run / "weights" / f"{args.weights}.pt"
    assert weights.exists(), weights
    data = yaml.safe_load(args.data.read_text())
    scale = float(data.get("mesh_scale", 0.001))
    _, corners3d = load_mesh_corners(data["mesh"], scale=scale)
    k = np.array(
        [[float(data["fx"]), 0, float(data["u0"])], [0, float(data["fy"]), float(data["v0"])], [0, 0, 1]],
        dtype=np.float64,
    )

    out_name = args.name or f"predict-val-native-retina-{args.weights}"
    out_dir = run / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    imgs = sorted([*args.source.glob("*.jpg"), *args.source.glob("*.png")])
    if args.max_images > 0:
        imgs = imgs[: args.max_images]
    print(f"weights={weights} n={len(imgs)} out={out_dir}")

    for img_path in imgs:
        bgr = cv2.imread(str(img_path))
        assert bgr is not None, img_path
        preds = model.predict(
            source=str(img_path),
            conf=args.conf,
            imgsz=args.imgsz,
            retina_masks=True,
            verbose=False,
        )
        r = preds[0]
        vis = bgr.copy()
        if r.masks is not None and len(r.masks):
            for mi in range(len(r.masks)):
                mask = r.masks.data[mi].cpu().numpy()
                vis = overlay_mask(vis, mask)
        if r.boxes is not None and len(r.boxes):
            for bi in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[bi].cpu().numpy()
                conf = float(r.boxes.conf[bi].cpu().numpy())
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"{conf:.2f}",
                    (int(x1), max(20, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
        if r.keypoints is not None and len(r.keypoints):
            for ki in range(len(r.keypoints)):
                xy = r.keypoints.data[ki].cpu().numpy()[:, :2]
                vis = draw_pose6d(vis, xy)
                if not args.no_pnp:
                    vis = draw_pnp_reproj(vis, xy, corners3d, k)
        cv2.imwrite(str(out_dir / f"{img_path.stem}.jpg"), vis)

    (out_dir / "README.txt").write_text(
        "green box = 2D det\n"
        "orange overlay = instance mask\n"
        "green/cyan points+edges = predicted 9 control points\n"
        "magenta = PnP reprojection from object.ply\n"
    )
    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()
