#!/usr/bin/env python3
"""Re-render Detect / Segment / OBB val_batch plots without class text; save class lists for legends."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import ultralytics.models.yolo.detect.val as detect_val
import ultralytics.models.yolo.obb.val as obb_val
import ultralytics.utils.plotting as plotting
from ultralytics import YOLO

# Repo root = parents[2] from docs/official_example/scripts/this_file.py
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/official_val/results_video/clean"
OUT.mkdir(parents=True, exist_ok=True)

_orig_plot_images = plotting.plot_images


def plot_images_nolabel(labels=None, images=None, paths=None, fname="images.jpg", names=None, **kwargs):
    """Force box/mask color only — no class names, conf, or filenames."""
    kwargs["show_labels"] = False
    kwargs["show_conf"] = False
    paths = None

    cls_ids: set[int] = set()
    if labels is not None and "cls" in labels:
        cls = labels["cls"]
        if hasattr(cls, "detach"):
            cls = cls.detach().cpu().numpy()
        cls = np.asarray(cls).reshape(-1)
        confs = labels.get("conf")
        if confs is not None:
            if hasattr(confs, "detach"):
                confs = confs.detach().cpu().numpy()
            confs = np.asarray(confs).reshape(-1)
            thr = float(kwargs.get("conf_thres", 0.25))
            if confs.shape[0] == cls.shape[0]:
                cls = cls[confs > thr]
        cls_ids = {int(c) for c in cls.tolist()}

    fname = Path(fname)
    if names is not None:
        name_map = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
        meta = {
            "classes": sorted(cls_ids),
            "names": {int(c): str(name_map.get(c, c)) for c in sorted(cls_ids)},
        }
        fname.with_suffix(".classes.json").write_text(json.dumps(meta, indent=2))

    return _orig_plot_images(labels=labels, images=images, paths=paths, fname=str(fname), names=names, **kwargs)


def install_patch() -> None:
    plotting.plot_images = plot_images_nolabel
    detect_val.plot_images = plot_images_nolabel
    obb_val.plot_images = plot_images_nolabel


def restore_patch() -> None:
    plotting.plot_images = _orig_plot_images
    detect_val.plot_images = _orig_plot_images
    obb_val.plot_images = _orig_plot_images


def run_one(model: str, data: str, name: str, **val_kwargs) -> Path:
    save_dir = OUT / name
    if save_dir.exists():
        for p in save_dir.glob("*"):
            if p.is_file():
                p.unlink()
    save_dir.mkdir(parents=True, exist_ok=True)

    m = YOLO(str(ROOT / model) if (ROOT / model).exists() else model)
    m.val(
        data=data,
        project=str(OUT),
        name=name,
        exist_ok=True,
        plots=True,
        save_json=False,
        device=0,
        **val_kwargs,
    )
    candidates = [save_dir, *OUT.glob(f"**/{name}")]
    for c in candidates:
        if c.is_dir() and any(c.glob("val_batch*_pred.jpg")):
            return c
    raise FileNotFoundError(f"clean plots not found for {name}")


def main() -> None:
    jobs = [
        ("yolo26n.pt", "coco.yaml", "det_coco_e2e", {"imgsz": 640}),
        ("yolo26n.pt", "coco.yaml", "det_coco_o2m", {"imgsz": 640, "end2end": False}),
        ("yolo26n-seg.pt", "coco.yaml", "seg_coco_e2e", {"imgsz": 640}),
        ("yolo26n-obb.pt", "DOTAv1.yaml", "obb_val", {"imgsz": 1024, "split": "val"}),
    ]
    index = {}
    install_patch()
    try:
        for model, data, name, kw in jobs:
            print(f"=== {name} ===", flush=True)
            d = run_one(model, data, name, **kw)
            n_json = len(list(d.glob("*.classes.json")))
            index[name] = {"dir": str(d), "class_jsons": n_json}
            print("saved", d, "class_jsons", n_json, flush=True)
    finally:
        restore_patch()
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print("ALL_CLEAN_DONE", index, flush=True)


if __name__ == "__main__":
    main()
