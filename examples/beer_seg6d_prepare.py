#!/usr/bin/env python3
"""Prepare beer joint labels for yolo26s-seg-6dpose.

Creates layout under YOLO6D/data/beer/yolo_seg6d/:
  images/{train,val}     -> symlink to JPEGImages
  labels/{train,val}     -> YOLO pose format: cls cx cy w h + 9*(x,y)
  labels_seg/{train,val} -> YOLO seg polygons (from labels_seg)

Point order matches YOLO6D: center, then 8 AABB corners.

Detection box uses the segmentation polygon AABB (tight object box), NOT the
9-keypoint AABB (which is larger and was making val_batch*_labels look oversized).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def seg_poly_xywh(seg_line: str) -> tuple[float, float, float, float]:
    """Return normalized cx,cy,w,h from a YOLO polygon label line."""
    vals = list(map(float, seg_line.split()))
    assert len(vals) >= 7 and (len(vals) - 1) % 2 == 0, f"bad seg line: {seg_line[:80]}"
    poly = np.array(vals[1:], dtype=np.float64).reshape(-1, 2)
    x1, y1 = poly.min(axis=0)
    x2, y2 = poly.max(axis=0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    return float(cx), float(cy), float(bw), float(bh)


def yolo6d_to_pose_line(pose_line: str, seg_line: str | None = None) -> str:
    """Convert YOLO6D label line to Ultralytics pose [9,2] line.

    Box comes from seg polygon AABB when available; otherwise falls back to 9-pt AABB.
    """
    vals = list(map(float, pose_line.split()))
    assert len(vals) >= 19, f"expected >=19 fields, got {len(vals)}"
    cls = int(vals[0])
    pts = np.array(vals[1:19], dtype=np.float64).reshape(9, 2)
    if seg_line is not None:
        cx, cy, bw, bh = seg_poly_xywh(seg_line)
    else:
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    coords = " ".join(f"{p:.6f}" for p in pts.reshape(-1))
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {coords}"


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/ultralytics/YOLO6D/data/beer"))
    args = parser.parse_args()
    root: Path = args.root
    out = root / "yolo_seg6d"

    test_ids = {Path(x.strip()).stem for x in open(root / "test.txt") if x.strip()}
    all_ids = sorted(p.stem for p in (root / "JPEGImages").glob("*.png"))
    train_ids = [i for i in all_ids if i not in test_ids]
    val_ids = sorted(test_ids)
    print(f"train={len(train_ids)} val={len(val_ids)}")

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for i in ids:
            link(root / "JPEGImages" / f"{i}.png", out / "images" / split / f"{i}.png")
            src_pose = root / "labels" / f"{i}.txt"
            src_seg = root / "labels_seg" / f"{i}.txt"
            seg_text = src_seg.read_text().strip().splitlines()[0] if src_seg.exists() else None
            pose_line = yolo6d_to_pose_line(src_pose.read_text().strip().splitlines()[0], seg_text)
            pose_out = out / "labels" / split / f"{i}.txt"
            pose_out.parent.mkdir(parents=True, exist_ok=True)
            pose_out.write_text(pose_line + "\n")
            link(src_seg, out / "labels_seg" / split / f"{i}.txt")

    (root / "train_seg6d.txt").write_text(
        "\n".join(str((out / "images" / "train" / f"{i}.png").resolve()) for i in train_ids) + "\n"
    )
    (root / "val_seg6d.txt").write_text(
        "\n".join(str((out / "images" / "val" / f"{i}.png").resolve()) for i in val_ids) + "\n"
    )
    # Invalidate label caches so training reloads new boxes
    for cache in out.rglob("*.cache"):
        cache.unlink()
        print(f"removed cache {cache}")
    print(f"Wrote layout at {out}")
    sample = (out / "labels" / "train" / f"{train_ids[0]}.txt").read_text().strip()
    print("sample pose:", sample[:140])


if __name__ == "__main__":
    main()
