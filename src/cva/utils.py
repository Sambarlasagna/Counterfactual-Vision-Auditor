"""
Utility functions for CVA experiments.

Provides: seed setting, device detection, logging setup,
timing helpers, and directory management.
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from contextlib import contextmanager

import numpy as np
import torch


# ── Seed & Reproducibility ─────────────────────────────────────────────

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value.
        deterministic: If True, enables deterministic cuDNN operations
                       (may reduce performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# ── Device Detection ───────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Detect and return the best available device.

    Returns:
        torch.device: CUDA device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_device_info() -> Dict[str, Any]:
    """
    Gather detailed device information for logging.

    Returns:
        Dictionary with Python, PyTorch, CUDA, and GPU details.
    """
    info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "cudnn_version": None,
        "gpu_name": None,
        "gpu_count": 0,
        "gpu_memory_total_mb": None,
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = str(torch.backends.cudnn.version())
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1
        )

    return info


# ── Logging ────────────────────────────────────────────────────────────

def setup_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    name: str = "cva",
) -> logging.Logger:
    """
    Set up a logger that writes to console and optionally to a file.

    Args:
        log_dir: Directory for log files. If None, logs to console only.
        level: Logging level.
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"cva_{timestamp}.log", encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ── Directory Management ───────────────────────────────────────────────

def ensure_dirs(*paths: str) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ── Timing ─────────────────────────────────────────────────────────────

@contextmanager
def timer(description: str = "Operation", logger: Optional[logging.Logger] = None):
    """
    Context manager for timing code blocks.

    Usage:
        with timer("Training"):
            train(...)
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"{description} completed in {elapsed:.2f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


# ── Metrics I/O ────────────────────────────────────────────────────────

def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """Save metrics dictionary to a JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)


def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load metrics dictionary from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Checkpoint Helpers ─────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    filepath: str,
    **extra,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state.
        epoch: Current epoch number.
        metrics: Metrics at this checkpoint.
        filepath: Destination path.
        **extra: Any additional items to include.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    checkpoint.update(extra)
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, device: Optional[torch.device] = None) -> Dict:
    """
    Load a training checkpoint.

    Args:
        filepath: Path to the checkpoint file.
        device: Device to map tensors to.

    Returns:
        Checkpoint dictionary.
    """
    map_location = device if device else "cpu"
    return torch.load(filepath, map_location=map_location, weights_only=False)
