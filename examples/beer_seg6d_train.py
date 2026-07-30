#!/usr/bin/env python3
"""Train beer seg6d (seg + 9 control points).

Default: official yolo26s-seg.pt, 40 epochs, YOLO6D-style pose curriculum
(det+seg only for first 15 epochs, then joint with elevated pose gain).

Optional VOC random-background replace (YOLO6D change_background), enabled via
--bg-dir / --bg-replace.

Training logs are written under the run directory (not docs/).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path("/root/ultralytics")
DEFAULT_WEIGHTS = ROOT / "yolo26s-seg.pt"
DEFAULT_PROJECT = ROOT / "runs" / "segment" / "beer-seg"
DEFAULT_BG_DIR = Path("/root/YOLO6D/VOCdevkit/VOC2012/JPEGImages")
DEFAULT_BG_MASK_DIR = Path("/root/YOLO6D/data/beer/mask")
MODEL_YAML = "yolo26s-seg6d.yaml"
DATA = "beer-seg6d.yaml"


class _Tee:
    """Tee stdout/stderr to a log file under the run directory."""

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
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Pretrained checkpoint (.pt). Default: official yolo26s-seg.pt",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=0)
    parser.add_argument("--name", default="yolo26s-seg-6dpose-official")
    parser.add_argument(
        "--project",
        default=str(DEFAULT_PROJECT),
        help="Absolute path under runs/segment/beer-seg (avoids runs/seg6d).",
    )
    parser.add_argument(
        "--pose",
        type=float,
        default=24.0,
        help="Pose/kpt MSE gain after warmup (v2 used 12; raised to push past 1–2 plateau).",
    )
    parser.add_argument(
        "--pose-warmup-epochs",
        type=int,
        default=15,
        help="First N epochs: det+seg only (pose gain=0); then joint training.",
    )
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument(
        "--bg-replace",
        type=float,
        default=0.0,
        help="Probability of YOLO6D-style random background replace (1.0 = always).",
    )
    parser.add_argument(
        "--bg-dir",
        type=str,
        default="",
        help="Background image directory (VOC2012/JPEGImages). Required if --bg-replace>0.",
    )
    parser.add_argument(
        "--bg-mask-dir",
        type=str,
        default=str(DEFAULT_BG_MASK_DIR),
        help="FG masks by stem (beer/mask). Empty string = use instance polygons only.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers (capped by CPU/batch).")
    parser.add_argument(
        "--cache",
        type=str,
        default="",
        help="Image cache: ram|disk|empty. VOC bg still random-loads; mainly speeds beer images.",
    )
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
        f"weights={args.weights} epochs={args.epochs} pose={args.pose} "
        f"pose_warmup_epochs={args.pose_warmup_epochs} bg_replace={args.bg_replace} "
        f"bg_dir={args.bg_dir or None} workers={args.workers} cache={args.cache or False} run={run_dir}"
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
        fliplr=0.0,  # 9-point order is not left-right symmetric
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
        bg_dir = args.bg_dir or str(DEFAULT_BG_DIR)
        if not Path(bg_dir).is_dir():
            raise SystemExit(f"--bg-dir not found: {bg_dir} (extract VOCdevkit first)")
        train_kw["bg_dir"] = bg_dir
        if args.bg_mask_dir:
            train_kw["bg_mask_dir"] = args.bg_mask_dir

    model = YOLO(MODEL_YAML)
    model.train(**train_kw)
    print(f"Training finished. Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
