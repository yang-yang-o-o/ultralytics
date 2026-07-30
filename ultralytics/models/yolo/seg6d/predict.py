# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class Seg6DPredictor(SegmentationPredictor):
    """Predictor for seg6d: boxes + masks + 9 control points (+ optional PnP later)."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize with task=seg6d (bypass SegmentationPredictor task=segment)."""
        DetectionPredictor.__init__(self, cfg, overrides, _callbacks)
        self.args.task = "seg6d"

    def construct_result(self, pred, img, orig_img, img_path, proto):
        """Build Results with masks and keypoints."""
        nm = 32
        if hasattr(self.model, "model") and isinstance(self.model.model, (torch.nn.Sequential, torch.nn.ModuleList)):
            nm = getattr(self.model.model[-1], "nm", 32)
        elif hasattr(self.model, "nm"):
            nm = self.model.nm

        # pred: xyxy, conf, cls, mask_coeffs(nm), kpts(nk)
        if pred.shape[0] == 0:
            masks = None
            kpts = None
            boxes = pred[:, :6]
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6 : 6 + nm], pred[:, :4], orig_img.shape[:2])
            kpts = pred[:, 6 + nm :].view(pred.shape[0], *self.model.kpt_shape)
            kpts = ops.scale_coords(img.shape[2:], kpts, orig_img.shape)
            boxes = pred[:, :6]
        else:
            masks = ops.process_mask(proto, pred[:, 6 : 6 + nm], pred[:, :4], img.shape[2:], upsample=True)
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            kpts = pred[:, 6 + nm :].view(pred.shape[0], *self.model.kpt_shape)
            kpts = ops.scale_coords(img.shape[2:], kpts, orig_img.shape)
            boxes = pred[:, :6]
        if masks is not None:
            keep = masks.amax((-2, -1)) > 0
            if not (all(keep) or getattr(self, "_feats", None) is not None):
                pred, masks = pred[keep], masks[keep]
                if kpts is not None:
                    kpts = kpts[keep]
                boxes = pred[:, :6]
        from ultralytics.engine.results import Results

        return Results(orig_img, path=img_path, names=self.model.names, boxes=boxes, masks=masks, keypoints=kpts)
