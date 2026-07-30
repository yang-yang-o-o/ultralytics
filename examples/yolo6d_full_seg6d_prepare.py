#!/usr/bin/env python3
"""Prepare projects/object_6d_pose_annotation yolo6d_full for Ultralytics seg6d.

Input (YOLO6D-style):
  <root>/{rgb,mask,labels}/full_XXXXXX.*
  <root>/{train,test}.txt

Output:
  <root>/yolo_seg6d/
    images/{train,val}     symlink → rgb
    labels/{train,val}     cls cx cy w h + 9*(x,y)  (box from seg AABB)
    labels_seg/{train,val} cls + polygon from mask
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def mask_to_seg_line(mask_path: Path, class_id: int = 0, epsilon_ratio: float = 0.0005) -> str | None:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    h, w = m.shape
    binary = (m > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 20:
        return None
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon_ratio * peri, True)
    if len(approx) < 3:
        return None
    pts = approx.reshape(-1, 2).astype(np.float64)
    pts[:, 0] /= w
    pts[:, 1] /= h
    pts = np.clip(pts, 0.0, 1.0)
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
    return f"{class_id} {coords}"


def seg_poly_xywh(seg_line: str) -> tuple[float, float, float, float]:
    vals = list(map(float, seg_line.split()))
    poly = np.array(vals[1:], dtype=np.float64).reshape(-1, 2)
    x1, y1 = poly.min(axis=0)
    x2, y2 = poly.max(axis=0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    return float(cx), float(cy), float(bw), float(bh)


def yolo6d_to_pose_line(pose_line: str, seg_line: str | None = None) -> str:
    vals = list(map(float, pose_line.split()))
    assert len(vals) >= 19, f"expected >=19 fields, got {len(vals)}"
    cls = int(vals[0])
    pts = np.array(vals[1:19], dtype=np.float64).reshape(9, 2)
    pts = np.clip(pts, 0.0, 1.0)
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


def stems_from_list(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(Path(line).stem)
    return out


def find_rgb(root: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = root / "rgb" / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/root/ultralytics/projects/object_6d_pose_annotation/outputs/run1/yolo6d_full"),
    )
    parser.add_argument("--epsilon-ratio", type=float, default=0.0005)
    args = parser.parse_args()
    root: Path = args.root
    out = root / "yolo_seg6d"

    test_ids = set(stems_from_list(root / "test.txt"))
    all_ids = sorted(p.stem for p in (root / "rgb").glob("*.*"))
    train_ids = [i for i in all_ids if i not in test_ids]
    val_ids = sorted(test_ids)
    print(f"root={root} train={len(train_ids)} val={len(val_ids)}")

    n_ok = 0
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for stem in ids:
            rgb = find_rgb(root, stem)
            if rgb is None:
                print(f"missing rgb {stem}")
                continue
            mask_path = root / "mask" / f"{stem}.png"
            pose_path = root / "labels" / f"{stem}.txt"
            if not pose_path.exists():
                print(f"missing pose {stem}")
                continue
            seg_line = mask_to_seg_line(mask_path, epsilon_ratio=args.epsilon_ratio)
            if seg_line is None:
                print(f"bad mask {stem}")
                continue
            pose_line = yolo6d_to_pose_line(pose_path.read_text().strip().splitlines()[0], seg_line)

            link(rgb, out / "images" / split / f"{stem}{rgb.suffix}")
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels_seg" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split / f"{stem}.txt").write_text(pose_line + "\n")
            (out / "labels_seg" / split / f"{stem}.txt").write_text(seg_line + "\n")
            n_ok += 1

    for cache in out.rglob("*.cache"):
        cache.unlink()
        print(f"removed cache {cache}")

    print(f"Wrote {out}  images={n_ok}")
    sample = next((out / "labels" / "train").glob("*.txt"), None)
    if sample:
        print("sample pose:", sample.read_text().strip()[:160])


if __name__ == "__main__":
    main()
