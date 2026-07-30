# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.models.yolo.seg6d.utils import Pose6DEval
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import SegmentMetrics, box_iou


class Seg6DValidator(SegmentationValidator):
    """Validate seg6d: mask mAP + YOLO6D proj/ADD via PnP (beer.ply + intrinsics)."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize Seg6DValidator without forcing task=segment."""
        DetectionValidator.__init__(self, dataloader, save_dir, args, _callbacks)
        self.process = None
        self.args.task = "seg6d"
        self.metrics = SegmentMetrics()
        self.kpt_shape = None
        self.pose6d = None
        self.nm = 32

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess masks and keypoints."""
        batch = super().preprocess(batch)
        if "keypoints" in batch:
            batch["keypoints"] = batch["keypoints"].float()
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize segment metrics + pose6d evaluator."""
        super().init_metrics(model)
        head = model
        if hasattr(model, "model") and isinstance(model.model, (torch.nn.Sequential, torch.nn.ModuleList)):
            head = model.model[-1]
        self.nm = getattr(head, "nm", 32)
        self.kpt_shape = self.data["kpt_shape"]
        try:
            self.pose6d = Pose6DEval(self.data)
        except Exception as e:
            LOGGER.warning(f"PnP metrics disabled: {e}")
            self.pose6d = None

    def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Split extra into mask coeffs + keypoints, then build masks."""
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        preds = DetectionValidator.postprocess(self, preds[0])
        imgsz = [4 * x for x in proto.shape[2:]]
        nk = self.kpt_shape[0] * self.kpt_shape[1]
        for i, pred in enumerate(preds):
            extra = pred.pop("extra")
            coeff = extra[:, : self.nm]
            kpts = extra[:, self.nm : self.nm + nk]
            pred["masks"] = self.process(proto[i], coeff, pred["bboxes"], shape=imgsz)
            pred["keypoints"] = kpts.view(-1, *self.kpt_shape)
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare masks and keypoints for one sample."""
        prepared = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = prepared["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        prepared["keypoints"] = kpts
        return prepared

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update segment metrics and accumulate PnP / ADD errors."""
        super().update_metrics(preds, batch)
        if self.pose6d is None:
            return
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            gt = pbatch["keypoints"]
            if gt.shape[0] == 0 or pred["cls"].shape[0] == 0 or "keypoints" not in pred:
                continue
            gt_boxes = pbatch["bboxes"]
            pr_boxes = pred["bboxes"]
            if gt_boxes.numel() == 0 or pr_boxes.numel() == 0:
                continue
            iou = box_iou(gt_boxes, pr_boxes)
            used: set[int] = set()
            for gi in range(gt_boxes.shape[0]):
                row = iou[gi].cpu().numpy()
                order = np.argsort(-row)
                for pj in order:
                    pj = int(pj)
                    if pj in used or row[pj] < 0.1:
                        continue
                    used.add(pj)
                    # Prefer same-class match when available
                    gt_cls = int(pbatch["cls"][gi].item()) if "cls" in pbatch else 0
                    pr_cls = int(pred["cls"][pj].item()) if pred["cls"].numel() else gt_cls
                    if pr_cls != gt_cls and row[pj] < 0.5:
                        continue
                    gt_k = gt[gi, :, :2].detach().cpu().numpy()
                    pr_k = pred["keypoints"][pj, :, :2].detach().cpu().numpy()
                    gt_px, pr_px = gt_k.copy(), pr_k.copy()
                    if "ratio_pad" in pbatch:
                        gain = pbatch["ratio_pad"][0][0]
                        pad = pbatch["ratio_pad"][1]
                        for arr in (gt_px, pr_px):
                            arr[:, 0] = (arr[:, 0] - pad[0]) / gain
                            arr[:, 1] = (arr[:, 1] - pad[1]) / gain
                    else:
                        oh, ow = pbatch["ori_shape"]
                        ih, iw = pbatch["imgsz"]
                        for arr in (gt_px, pr_px):
                            arr[:, 0] *= ow / iw
                            arr[:, 1] *= oh / ih
                    self.pose6d.update(gt_px, pr_px, cls_id=gt_cls)
                    break

    def finalize_metrics(self) -> None:
        """Log YOLO6D-style summary after standard metrics."""
        super().finalize_metrics()
        if self.pose6d is not None:
            s = getattr(self.metrics, "pose6d", None) or self.pose6d.summary()
            self.metrics.pose6d = s
            LOGGER.info(
                f"Pose6D: n={int(s['n'])} mean_corner={s['mean_corner2d']:.2f}px "
                f"mean_proj={s['mean_proj2d']:.2f}px Acc@5px={s['acc_proj_5px']:.1f}% "
                f"mean_ADD={s['mean_add']:.4f} Acc@0.1d={s['acc_add_0.1d']:.1f}%"
            )

    def get_stats(self) -> dict[str, Any]:
        """Return metrics with fitness that includes Pose6D (not only box/mask mAP).

        EarlyStopping previously selected ep15 because mask/box mAP saturated while
        proj/ADD kept improving through later epochs. Pose terms are added so best.pt
        tracks 6D quality as well.
        """
        stats = super().get_stats()
        if self.pose6d is None:
            return stats
        s = self.pose6d.summary()
        self.metrics.pose6d = s
        stats["metrics/pose6d_acc5px"] = float(s["acc_proj_5px"])
        stats["metrics/pose6d_add_acc"] = float(s["acc_add_0.1d"])
        stats["metrics/pose6d_mean_proj"] = float(s["mean_proj2d"])
        stats["metrics/pose6d_mean_corner"] = float(s["mean_corner2d"])
        stats["metrics/pose6d_mean_add"] = float(s["mean_add"])
        # During pose warmup, fitness is mAP-only (pose head not trained yet)
        warm = int(getattr(self.args, "pose_warmup_epochs", 0) or 0)
        epoch = int(getattr(self.args, "epoch", 10**9))
        if warm and epoch < warm:
            return stats
        # Accuracies in percent → [0, 1]; each contributes up to 0.5 so pose adds ≤ 1.0
        pose_fit = 0.5 * (s["acc_proj_5px"] / 100.0) + 0.5 * (s["acc_add_0.1d"] / 100.0)
        # Soft bonus for lower mean projection error (reference: 5px → ~0.5, 2.5px → ~0.67)
        mean_proj = max(float(s["mean_proj2d"]), 1e-3)
        pose_fit += 0.5 * (5.0 / (5.0 + mean_proj))
        map_fit = float(stats.get("fitness", 0.0))
        stats["fitness"] = map_fit + pose_fit
        return stats
