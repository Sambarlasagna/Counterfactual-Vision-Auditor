import sys
import os
import torch
import logging
import gc
from typing import Dict, Any, Optional

# Ensure root src is in path so we can import cva
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from cva.models import create_model
from cva.interventions import SegmentationModel, InterventionRunner
from cva.utils import load_checkpoint, get_device

logger = logging.getLogger("cva_api")

DEMO_ID = "waterbirds-resnet18"

# Static config — no heavy objects
DEMO_CONFIG = {
    "id": DEMO_ID,
    "name": "Waterbirds ResNet-18 (ERM Baseline)",
    "architecture": "resnet18",
    "num_classes": 2,
    "labels": {0: "Landbird", 1: "Waterbird"},
    "input_size": [224, 224],
    "description": "Trained on Waterbirds benchmark using Empirical Risk Minimization. Demonstrates significant background correlation bias.",
    "is_ready": True,  # will be loaded lazily on first request
    "provenance": {
        "dataset": "Waterbirds (WILDS v1.0 / CUB-200-2011 + Places)",
        "epochs": 30,
        "batch_size": 16,
        "optimizer": "Adam, lr=1e-3, cosine decay",
        "worst_group_accuracy": "24.45% on test set",
        "majority_accuracy": ">92% on matching backgrounds"
    }
}

MODEL_CONFIG = {
    "model": {
        "name": "resnet18",
        "pretrained": False,
        "num_classes": 2,
        "freeze_backbone": False
    },
    "dataset": {
        "image_size": 224
    }
}


class ModelRegistry:
    """Lazy-loading model registry to stay within Render 512MB RAM."""

    def __init__(self):
        self.device = get_device()
        self.model_configs: Dict[str, Dict[str, Any]] = {DEMO_ID: DEMO_CONFIG}
        self._classifier: Optional[torch.nn.Module] = None
        self._seg_model: Optional[SegmentationModel] = None
        logger.info(f"ModelRegistry initialized (device={self.device}). Models will be loaded on first request.")

    def _find_checkpoint(self):
        """Find the best available checkpoint path."""
        packaged = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "resnet18_weights.pt")
        local = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "resnet18_best.pt")
        if os.path.exists(packaged):
            return os.path.abspath(packaged), True  # (path, is_state_dict_only)
        if os.path.exists(local):
            return os.path.abspath(local), False
        return None, False

    def _load_classifier(self) -> torch.nn.Module:
        """Load classifier on first use."""
        path, is_state_dict = self._find_checkpoint()
        if not path:
            raise RuntimeError("No checkpoint found for demo model.")

        logger.info(f"Lazy-loading classifier from {path}...")
        model = create_model(MODEL_CONFIG)
        if is_state_dict:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
        else:
            ckpt = load_checkpoint(path, self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            del ckpt
            gc.collect()
        model = model.to(self.device)
        model.eval()
        logger.info("Classifier loaded successfully.")
        return model

    def get_model(self, model_id: str) -> Optional[torch.nn.Module]:
        if model_id != DEMO_ID:
            return None
        if self._classifier is None:
            self._classifier = self._load_classifier()
        return self._classifier

    def get_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.model_configs.get(model_id)

    def list_models(self):
        return list(self.model_configs.values())

    def get_segmentation_model(self) -> SegmentationModel:
        if self._seg_model is None:
            cache_dir = os.path.join(PROJECT_ROOT, "outputs", "masks")
            os.makedirs(cache_dir, exist_ok=True)
            logger.info("Lazy-loading DeepLabV3 segmentation model...")
            self._seg_model = SegmentationModel(device=self.device, cache_dir=cache_dir)
            logger.info("Segmentation model loaded.")
        return self._seg_model


registry = ModelRegistry()
