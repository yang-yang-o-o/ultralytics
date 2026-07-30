#!/usr/bin/env python3
"""Run IC-BIN seg6d inference and export BOP Challenge CSV results.

Output format (BOP19):
  scene_id,im_id,obj_id,score,R,t,time
  R = 9 floats row-major; t = 3 floats in mm.

Example:
  python examples/icbin_seg6d_export_bop.py \\
    --weights runs/segment/icbin-seg/icbin-seg6d-smoke/weights/best.pt \\
    --out runs/segment/icbin-seg/icbin-seg6d-smoke/bop_results/seg6d_icbin-test.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.seg6d.utils import load_mesh_corners, pnp
from ultralytics.utils import YAML


def load_data_cfg(path: Path) -> dict:
    return YAML.load(path)


def corners3d_for_cls(data: dict, cls_id: int) -> np.ndarray:
    scale = float(data.get("mesh_scale", 1.0))
    meshes = data.get("meshes") or {}
    mesh = meshes.get(cls_id, meshes.get(str(cls_id))) or data["mesh"]
    _, corners = load_mesh_corners(mesh, scale=scale)
    return corners  # (4, 9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("/root/ultralytics/runs/segment/icbin-seg/icbin-seg6d-smoke/weights/best.pt"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/root/ultralytics/ultralytics/cfg/datasets/icbin-seg6d.yaml"),
    )
    parser.add_argument(
        "--bop-root",
        type=Path,
        default=Path("/root/YOLO6D/datasets/bop/icbin"),
        help="IC-BIN root with test/ scenes",
    )
    parser.add_argument("--split", default="test", help="BOP split folder name")
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="test_targets_bop19.json (default: <bop-root>/test_targets_bop19.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (BOP naming: METHOD_DATASET-test.csv)",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=0)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()

    data = load_data_cfg(args.data)
    targets_path = args.targets or (args.bop_root / "test_targets_bop19.json")
    targets = json.loads(targets_path.read_text())
    # unique images from targets
    im_keys = sorted({(int(t["scene_id"]), int(t["im_id"])) for t in targets})

    out = args.out or (
        Path("/root/ultralytics/runs/segment/icbin-seg/icbin-seg6d-smoke/bop_results")
        / "seg6d-smoke_icbin-test.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    corners_cache: dict[int, np.ndarray] = {}

    rows: list[str] = ["scene_id,im_id,obj_id,score,R,t,time"]
    n_pose_ok = 0
    n_pose_fail = 0

    for scene_id, im_id in im_keys:
        rgb_path = args.bop_root / args.split / f"{scene_id:06d}" / "rgb" / f"{im_id:06d}.png"
        if not rgb_path.exists():
            rgb_path = rgb_path.with_suffix(".jpg")
        if not rgb_path.exists():
            print(f"missing {rgb_path}")
            continue

        cam_path = args.bop_root / args.split / f"{scene_id:06d}" / "scene_camera.json"
        cam = json.loads(cam_path.read_text())[str(im_id)]
        K = np.array(cam["cam_K"], dtype=np.float64).reshape(3, 3)

        t0 = time.perf_counter()
        preds = model.predict(
            source=str(rgb_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
            verbose=False,
            retina_masks=False,
        )
        elapsed = time.perf_counter() - t0
        r = preds[0]

        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            # still need consistent timing? BOP allows images with no estimates
            continue

        kpts = r.keypoints.data.cpu().numpy()  # (N, 9, 2|3)
        boxes = r.boxes
        clss = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            cls_id = int(clss[i])
            obj_id = cls_id + 1  # BOP 1-based
            score = float(scores[i])
            pts2d = kpts[i, :, :2].astype(np.float64)
            if cls_id not in corners_cache:
                corners_cache[cls_id] = corners3d_for_cls(data, cls_id)
            corners3d = corners_cache[cls_id]
            try:
                R, t = pnp(corners3d[:3].T, pts2d, K)
            except Exception:
                n_pose_fail += 1
                continue
            n_pose_ok += 1
            R_str = " ".join(f"{v:.8f}" for v in R.reshape(-1))
            t_str = " ".join(f"{v:.8f}" for v in t.reshape(-1))
            rows.append(
                f"{scene_id},{im_id},{obj_id},{score:.6f},{R_str},{t_str},{elapsed:.6f}"
            )

    out.write_text("\n".join(rows) + "\n")
    print(f"wrote {out}  lines={len(rows)-1}  pnp_ok={n_pose_ok} pnp_fail={n_pose_fail} images={len(im_keys)}")


if __name__ == "__main__":
    main()
