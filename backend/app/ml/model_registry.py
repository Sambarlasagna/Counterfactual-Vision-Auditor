import sys
import os
import torch
import logging
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

class ModelRegistry:
    def __init__(self):
        self.device = get_device()
        self.models: Dict[str, torch.nn.Module] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self.seg_model: Optional[SegmentationModel] = None
        self._init_demo_model()

    def _init_demo_model(self):
        demo_id = "waterbirds-resnet18"
        ckpt_path = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "resnet18_best.pt")
        
        config = {
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
        
        self.model_configs[demo_id] = {
            "id": demo_id,
            "name": "Waterbirds ResNet-18 (ERM Baseline)",
            "architecture": "resnet18",
            "num_classes": 2,
            "labels": {0: "Landbird", 1: "Waterbird"},
            "input_size": [224, 224],
            "description": "Trained on Waterbirds benchmark using Empirical Risk Minimization. Demonstrates significant background correlation bias.",
            "is_ready": False,
            "provenance": {
                "dataset": "Waterbirds (WILDS v1.0 / CUB-200-2011 + Places)",
                "epochs": 30,
                "batch_size": 16,
                "optimizer": "Adam, lr=1e-3, cosine decay",
                "worst_group_accuracy": "24.45% on test set",
                "majority_accuracy": ">92% on matching backgrounds"
            }
        }
        
        if os.path.exists(ckpt_path):
            try:
                model = create_model(config)
                ckpt = load_checkpoint(ckpt_path, self.device)
                model.load_state_dict(ckpt["model_state_dict"])
                model = model.to(self.device)
                model.eval()
                self.models[demo_id] = model
                self.model_configs[demo_id]["is_ready"] = True
                logger.info(f"Loaded demo model from {ckpt_path} onto {self.device}")
            except Exception as e:
                logger.error(f"Failed to load demo checkpoint {ckpt_path}: {e}")
        else:
            logger.warning(f"Checkpoint not found at {ckpt_path}")

    def get_model(self, model_id: str) -> Optional[torch.nn.Module]:
        return self.models.get(model_id)

    def get_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.model_configs.get(model_id)

    def list_models(self):
        return list(self.model_configs.values())

    def get_segmentation_model(self) -> SegmentationModel:
        if self.seg_model is None:
            cache_dir = os.path.join(PROJECT_ROOT, "outputs", "masks")
            os.makedirs(cache_dir, exist_ok=True)
            self.seg_model = SegmentationModel(device=self.device, cache_dir=cache_dir)
        return self.seg_model

registry = ModelRegistry()
