#!/usr/bin/env python3
"""Convert YOLO6D beer binary masks to YOLO segmentation polygon labels."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def mask_to_yolo_seg(mask_path: Path, class_id: int = 0, epsilon_ratio: float = 0.0005) -> list[str]:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(mask_path)
    h, w = m.shape
    _, binary = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_ratio * peri, True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2).astype(np.float64)
        pts[:, 0] /= w
        pts[:, 1] /= h
        pts = np.clip(pts, 0.0, 1.0)
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        lines.append(f"{class_id} {coords}")
    return lines


def build_yolo_layout(root: Path, labels_dir: Path) -> tuple[int, int]:
    img_dir = root / "JPEGImages"
    yolo_root = root / "yolo_seg"
    train_img = yolo_root / "images" / "train"
    val_img = yolo_root / "images" / "val"
    train_lbl = yolo_root / "labels" / "train"
    val_lbl = yolo_root / "labels" / "val"
    for d in (train_img, val_img, train_lbl, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    test_ids = {Path(line.strip()).stem for line in open(root / "test.txt") if line.strip()}
    all_ids = sorted(p.stem for p in img_dir.glob("*.png"))
    train_ids = [i for i in all_ids if i not in test_ids]
    val_ids = sorted(test_ids)

    def link(src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst)

    for i in train_ids:
        link(img_dir / f"{i}.png", train_img / f"{i}.png")
        link(labels_dir / f"{i}.txt", train_lbl / f"{i}.txt")
    for i in val_ids:
        link(img_dir / f"{i}.png", val_img / f"{i}.png")
        link(labels_dir / f"{i}.txt", val_lbl / f"{i}.txt")

    (root / "train_seg.txt").write_text("\n".join(str((img_dir / f"{i}.png").resolve()) for i in train_ids) + "\n")
    (root / "val_seg.txt").write_text("\n".join(str((img_dir / f"{i}.png").resolve()) for i in val_ids) + "\n")
    return len(train_ids), len(val_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/YOLO6D/data/beer"))
    parser.add_argument("--epsilon-ratio", type=float, default=0.0005,
                        help="approxPolyDP epsilon / perimeter; smaller = denser polygons")
    args = parser.parse_args()

    mask_dir = args.root / "mask"
    out_dir = args.root / "labels_seg"
    out_dir.mkdir(exist_ok=True)

    n_ok = 0
    for mf in sorted(mask_dir.glob("*.png")):
        lines = mask_to_yolo_seg(mf, epsilon_ratio=args.epsilon_ratio)
        (out_dir / f"{mf.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_ok += 1

    n_train, n_val = build_yolo_layout(args.root, out_dir)
    print(f"Converted {n_ok} masks -> {out_dir}")
    print(f"YOLO layout: train={n_train} val={n_val} at {args.root / 'yolo_seg'}")


if __name__ == "__main__":
    main()
