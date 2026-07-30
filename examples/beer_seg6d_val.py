#!/usr/bin/env python3
"""Validate a beer seg6d run (mAP + Pose6D) and write metrics under the run dir."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

DEFAULT_RUN = Path("/root/ultralytics/runs/segment/beer-seg/yolo26s-seg-6dpose-official")
DATA = "beer-seg6d.yaml"


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
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--weights", default="best", choices=["best", "last"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default=0)
    args = parser.parse_args()

    run = args.run.resolve()
    weights = run / "weights" / f"{args.weights}.pt"
    out_name = f"val-{args.weights}"
    log_path = run / f"val-{args.weights}.log"
    log_f = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)

    print(f"weights: {weights}")
    model = YOLO(str(weights))
    metrics = model.val(
        data=DATA,
        split="val",
        imgsz=args.imgsz,
        device=args.device,
        project=str(run),
        name=out_name,
        exist_ok=True,
        plots=True,
    )
    # Pose6D metrics live on validator / results csv; also dump summary if present
    lines = [
        f"run={run}",
        f"weights={weights}",
        f"box mAP50-95={float(metrics.box.map):.4f}",
        f"seg mAP50-95={float(metrics.seg.map):.4f}",
        f"seg mAP50={float(metrics.seg.map50):.4f}",
    ]
    pose = getattr(metrics, "pose6d", None) or getattr(getattr(metrics, "seg", None), "pose6d", None)
    if isinstance(pose, dict):
        lines.append(
            f"Pose6D mean_proj={pose.get('mean_proj2d')} Acc@5px={pose.get('acc_proj_5px')} "
            f"Acc@0.1d={pose.get('acc_add_0.1d')}"
        )
    summary = run / f"val-{args.weights}-metrics.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {summary} and {log_path}")


if __name__ == "__main__":
    main()
