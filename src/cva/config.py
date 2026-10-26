"""
Configuration loader for CVA experiments.

Loads YAML config files with sensible defaults and provides
a clean dict-like interface for experiment parameters.
"""

import os
import yaml
import copy
from pathlib import Path
from typing import Any, Dict, Optional


# ── Default configuration ──────────────────────────────────────────────

DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "name": "resnet18",
        "pretrained": True,
        "num_classes": 2,
        "freeze_backbone": False,
    },
    "dataset": {
        "name": "waterbirds",
        "root": "data/waterbirds",
        "image_size": 224,
        "num_workers": 0,
        "pin_memory": True,
    },
    "training": {
        "batch_size": 16,
        "epochs": 30,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "scheduler": "cosine",
        "warmup_epochs": 2,
        "mixed_precision": True,
        "gradient_clip": 1.0,
    },
    "evaluation": {
        "batch_size": 32,
        "metrics": [
            "accuracy",
            "group_accuracy",
            "worst_group_accuracy",
            "confusion_matrix",
        ],
    },
    "checkpointing": {
        "save_dir": "outputs/checkpoints",
        "save_best": True,
        "save_every": 5,
        "metric": "worst_group_accuracy",
    },
    "logging": {
        "log_dir": "outputs/logs",
        "log_every": 10,
        "save_metrics": True,
    },
    "seed": 42,
    "deterministic": True,
    "smoke_test": {
        "enabled": False,
        "num_samples": 100,
        "epochs": 2,
    },
}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load experiment configuration from a YAML file.

    Merges the YAML contents on top of DEFAULT_CONFIG so that
    any missing keys fall back to sensible defaults.

    Args:
        config_path: Path to a YAML config file. If None, returns defaults.

    Returns:
        A fully-populated configuration dictionary.

    Raises:
        FileNotFoundError: If config_path does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    if config_path is None:
        return copy.deepcopy(DEFAULT_CONFIG)

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    merged = _deep_merge(DEFAULT_CONFIG, user_config)
    # Store the source path for logging / reproducibility
    merged["_config_path"] = str(config_path.resolve())
    return merged


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save a configuration dictionary to a YAML file.

    Useful for logging the exact config used in an experiment.

    Args:
        config: Configuration dictionary to save.
        save_path: Destination file path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove internal keys before saving
    to_save = {k: v for k, v in config.items() if not k.startswith("_")}

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(to_save, f, default_flow_style=False, sort_keys=False)


def config_to_flat_dict(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten a nested config dict for logging (e.g., to CSV or TensorBoard).

    Example:
        {"model": {"name": "resnet18"}} → {"model.name": "resnet18"}
    """
    flat = {}
    for key, value in config.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(config_to_flat_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat
