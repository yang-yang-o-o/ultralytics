#!/usr/bin/env python3
"""Train YOLO26s-seg on beer with quality-oriented settings.

Pipeline (see docs/experiment-logs/.../QUALITY_IMPROVEMENT.md):
  denser polygons → yolo26s-seg → imgsz=960 → mask_ratio=2
  → longer train + weaker mosaic → predict with retina_masks
"""

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolo26s-seg.pt")
    model.train(
        data="beer-seg.yaml",
        epochs=100,
        imgsz=960,
        batch=8,
        mask_ratio=2,
        close_mosaic=20,
        mosaic=0.5,
        scale=0.3,
        degrees=0.0,
        device=0,
        project="beer-seg",
        name="yolo26s-seg-imgsz960-mr2",
        exist_ok=True,
        patience=40,
        workers=4,
        amp=True,
    )


if __name__ == "__main__":
    main()
