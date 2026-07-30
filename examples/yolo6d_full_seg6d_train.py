#!/usr/bin/env python3
"""Train seg6d on yolo6d_full with beer BEST_CONFIG hyperparams.

Defaults mirror docs/.../BEST_CONFIG.md:
  official yolo26s-seg.pt, 100ep, imgsz=960, batch=8, workers=8,
  pose=24, pose_warmup=15, VOC bg_replace=1.0, close_mosaic=20, patience=60.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path("/root/ultralytics")
DEFAULT_WEIGHTS = ROOT / "yolo26s-seg.pt"
DEFAULT_PROJECT = ROOT / "runs" / "segment" / "yolo6d-full-seg"
DEFAULT_BG_DIR = Path("/root/YOLO6D/VOCdevkit/VOC2012/JPEGImages")
DEFAULT_BG_MASK_DIR = Path("/root/ultralytics/projects/object_6d_pose_annotation/outputs/run1/yolo6d_full/mask")
MODEL_YAML = "yolo26s-seg6d.yaml"
DATA = "yolo6d-full-seg6d.yaml"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=0)
    parser.add_argument("--name", default="yolo26s-seg6d-bestcfg")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--pose", type=float, default=24.0)
    parser.add_argument("--pose-warmup-epochs", type=int, default=15)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--bg-replace", type=float, default=1.0)
    parser.add_argument("--bg-dir", type=str, default=str(DEFAULT_BG_DIR))
    parser.add_argument("--bg-mask-dir", type=str, default=str(DEFAULT_BG_MASK_DIR))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    run_dir = project / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    log_f = open(log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)
    print(f"Logging to {log_path}")
    print(
        f"data={DATA} weights={args.weights} epochs={args.epochs} imgsz={args.imgsz} "
        f"batch={args.batch} pose={args.pose} warmup={args.pose_warmup_epochs} "
        f"bg_replace={args.bg_replace} run={run_dir}"
    )

    train_kw = dict(
        data=DATA,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        mask_ratio=2,
        close_mosaic=args.close_mosaic,
        mosaic=0.5,
        scale=0.3,
        degrees=0.0,
        fliplr=0.0,
        flipud=0.0,
        pose=args.pose,
        pose_warmup_epochs=args.pose_warmup_epochs,
        device=args.device,
        project=str(project),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        workers=args.workers,
        amp=True,
        resume=args.resume,
        pretrained=args.weights,
        bg_replace=args.bg_replace,
    )
    if args.cache:
        train_kw["cache"] = args.cache
    if args.bg_replace > 0:
        if not Path(args.bg_dir).is_dir():
            raise SystemExit(f"--bg-dir not found: {args.bg_dir}")
        train_kw["bg_dir"] = args.bg_dir
        if args.bg_mask_dir:
            train_kw["bg_mask_dir"] = args.bg_mask_dir

    model = YOLO(MODEL_YAML)
    model.train(**train_kw)
    print(f"Training finished. Artifacts: {run_dir}")
    print(f"best: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
