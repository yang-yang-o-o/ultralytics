#!/usr/bin/env python3
"""Predict beer val/test set at near-native resolution with retina_masks.

Outputs are stored under the training run dir for experiment isolation:
  runs/segment/beer-seg/yolo26s-seg-imgsz960-mr2/predict-val-native-retina/
"""

from pathlib import Path

from ultralytics import YOLO

RUN = Path("runs/segment/beer-seg/yolo26s-seg-imgsz960-mr2")
# Native beer images are 1292x964; use stride-aligned size >= long side
IMGSZ = 1312
SOURCE = Path("/root/ultralytics/YOLO6D/data/beer/yolo_seg/images/val")
OUT_NAME = "predict-val-native-retina"


def main() -> None:
    weights = RUN / "weights" / "best.pt"
    if not weights.exists():
        weights = RUN / "weights" / "last.pt"
    print(f"weights: {weights.resolve()}")
    model = YOLO(str(weights))

    results = model.predict(
        source=str(SOURCE),
        imgsz=IMGSZ,
        device=0,
        retina_masks=True,  # high-res mask rendering
        project=str(RUN),  # nest under this training run
        name=OUT_NAME,
        exist_ok=True,
        save=True,
        save_txt=False,
        conf=0.25,
        iou=0.7,
        verbose=True,
    )

    out_dir = RUN / OUT_NAME
    if results:
        r0 = results[0]
        print(f"orig0 shape: {r0.orig_shape}")  # (h, w)
        print(f"retina_masks enabled; n_images={len(results)}")
    print(f"Saved overlays -> {out_dir.resolve()}")
    print(f"n_files: {len(list(out_dir.glob('*.jpg'))) + len(list(out_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
