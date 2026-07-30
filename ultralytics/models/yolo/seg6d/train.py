# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import Seg6DModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK


class Seg6DTrainer(yolo.detect.DetectionTrainer):
    """Trainer for segment + YOLO6D-style 9×2D control points (task=seg6d)."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        """Initialize Seg6DTrainer with task=seg6d."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "seg6d"
        super().__init__(cfg, overrides, _callbacks)
        self.add_callback("on_train_epoch_start", self._update_pose_warmup)

    def _update_pose_warmup(self, trainer=None):
        """Zero pose loss for pose_warmup_epochs, then restore configured pose gain (YOLO6D-style curriculum)."""
        warm = int(getattr(self.args, "pose_warmup_epochs", 0) or 0)
        self.args.epoch = int(self.epoch)  # 0-based; read by Seg6DLoss / validator
        if warm and self.epoch == 0:
            LOGGER.info(
                f"Seg6D pose curriculum: epochs 0..{warm - 1} det+seg only (pose=0); "
                f"from epoch {warm} enable pose={float(self.args.pose)}"
            )
        self.args.pose_active = 0.0 if (warm and self.epoch < warm) else float(self.args.pose)
        if getattr(self, "validator", None) is not None:
            self.validator.args.epoch = self.args.epoch
            self.validator.args.pose_warmup_epochs = warm
            self.validator.args.pose_active = self.args.pose_active

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        """Build Seg6DModel and optionally load pretrained weights (seg or seg6d)."""
        model = self.set_model_names_for_load(
            Seg6DModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                data_kpt_shape=self.data["kpt_shape"],
                verbose=verbose and RANK == -1,
            )
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """Set names and kpt_shape on the model."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Return Seg6DValidator; loss names include kpt MSE."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss", "kpt_loss"
        return yolo.seg6d.Seg6DValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        """Require kpt_shape in data yaml."""
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in {self.args.data}")
        return data
