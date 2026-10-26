"""
Model factory for CVA experiments.

Provides pretrained models with configurable classification heads:
  - ResNet-18 (primary baseline)
  - ViT-Small (optional, via timm)
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("cva")


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create a model based on the configuration.

    Args:
        config: Full experiment configuration dict. Uses config["model"].

    Returns:
        A PyTorch model ready for training or evaluation.

    Raises:
        ValueError: If the model name is not recognized.
    """
    model_config = config["model"]
    name = model_config["name"].lower()
    num_classes = model_config["num_classes"]
    pretrained = model_config.get("pretrained", True)
    freeze_backbone = model_config.get("freeze_backbone", False)

    if name == "resnet18":
        model = _create_resnet18(num_classes, pretrained, freeze_backbone)
    elif name in ("vit_small", "vit_s", "vit"):
        model = _create_vit_small(num_classes, pretrained, freeze_backbone)
    else:
        raise ValueError(
            f"Unknown model name: '{name}'. "
            f"Supported: 'resnet18', 'vit_small'"
        )

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Created model '{name}': "
        f"{total_params:,} total params, "
        f"{trainable_params:,} trainable params"
    )

    return model


def _create_resnet18(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Create a ResNet-18 with a new classification head.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
        freeze_backbone: Whether to freeze all layers except the FC head.

    Returns:
        ResNet-18 model.
    """
    from torchvision import models
    from torchvision.models import ResNet18_Weights

    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        logger.info("Loaded ResNet-18 with ImageNet pretrained weights")
    else:
        model = models.resnet18(weights=None)
        logger.info("Created ResNet-18 from scratch (no pretrained weights)")

    # Replace the final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    logger.info(f"Replaced FC layer: {in_features} -> {num_classes}")

    if freeze_backbone:
        _freeze_backbone(model, exclude=["fc"])
        logger.info("Froze backbone; only FC layer is trainable")

    return model


def _create_vit_small(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Create a ViT-Small via timm with a new classification head.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.
        freeze_backbone: Whether to freeze the backbone.

    Returns:
        ViT-Small model.
    """
    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm is required for ViT models. Install with: pip install timm"
        )

    model_name = "vit_small_patch16_224"
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    logger.info(
        f"Created {model_name} via timm "
        f"(pretrained={pretrained}, num_classes={num_classes})"
    )

    if freeze_backbone:
        # Freeze everything except the classification head
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        logger.info("Froze ViT backbone; only head is trainable")

    return model


def _freeze_backbone(model: nn.Module, exclude: list = None) -> None:
    """
    Freeze all model parameters except those in excluded layers.

    Args:
        model: The model to freeze.
        exclude: List of layer name prefixes to keep trainable.
    """
    exclude = exclude or []
    for name, param in model.named_parameters():
        if not any(name.startswith(ex) for ex in exclude):
            param.requires_grad = False


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get model information for logging.

    Args:
        model: A PyTorch model.

    Returns:
        Dict with parameter counts and layer info.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "total_params_mb": round(total_params * 4 / (1024 ** 2), 2),  # ~4 bytes per float32
    }
