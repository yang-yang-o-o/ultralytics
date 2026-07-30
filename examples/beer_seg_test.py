#!/usr/bin/env python3
"""Validate and predict with quality-tuned beer seg model (retina_masks)."""

from pathlib import Path

from ultralytics import YOLO

RUN = Path("runs/segment/beer-seg/yolo26s-seg-imgsz960-mr2")
IMGSZ = 960


def main() -> None:
    weights = RUN / "weights" / "best.pt"
    if not weights.exists():
        weights = RUN / "weights" / "last.pt"
    model = YOLO(str(weights))

    metrics = model.val(
        data="beer-seg.yaml",
        split="val",
        imgsz=IMGSZ,
        device=0,
        retina_masks=True,
    )
    print(f"box mAP50-95: {metrics.box.map:.4f}")
    print(f"seg mAP50-95: {metrics.seg.map:.4f}")
    print(f"seg mAP50:    {metrics.seg.map50:.4f}")

    pred_dir = Path("/root/YOLO6D/data/beer/yolo_seg/images/val")
    model.predict(
        source=str(pred_dir),
        imgsz=IMGSZ,
        device=0,
        retina_masks=True,
        project=str(RUN),
        name="predict-s960-retina",
        exist_ok=True,
        save=True,
    )
    print(f"Predictions saved to {RUN / 'predict-s960-retina'}")


if __name__ == "__main__":
    main()
