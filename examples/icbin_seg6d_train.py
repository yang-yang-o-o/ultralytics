#!/usr/bin/env python3
"""Short / full train for IC-BIN seg6d (seg + 9 control points).

Smoke defaults: 5 epochs, imgsz=640, short pose warmup — enough to produce
weights/best.pt then hook bop_toolkit eval.

    python examples/icbin_seg6d_train.py
    python examples/icbin_seg6d_train.py --epochs 40 --pose-warmup-epochs 10 --name icbin-seg6d-ep40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path("/root/ultralytics")
DEFAULT_WEIGHTS = ROOT / "yolo26s-seg.pt"
DEFAULT_PROJECT = ROOT / "runs" / "segment" / "icbin-seg"
MODEL_YAML = "yolo26s-seg6d.yaml"
DATA = "icbin-seg6d.yaml"


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
    parser.add_argument("--epochs", type=int, default=5, help="Smoke default 5; raise for real runs")
    parser.add_argument("--imgsz", type=int, default=640, help="IC-BIN native 640×480")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=0)
    parser.add_argument("--name", default="icbin-seg6d-smoke")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--pose", type=float, default=24.0)
    parser.add_argument("--pose-warmup-epochs", type=int, default=2)
    parser.add_argument("--close-mosaic", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
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
        f"batch={args.batch} pose={args.pose} pose_warmup={args.pose_warmup_epochs} run={run_dir}"
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
    )
    if args.cache:
        train_kw["cache"] = args.cache

    model = YOLO(MODEL_YAML)
    model.train(**train_kw)
    print(f"Training finished. Artifacts: {run_dir}")
    print(f"best: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
