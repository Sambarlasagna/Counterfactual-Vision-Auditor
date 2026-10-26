"""
Evaluation utilities for CVA experiments.

Provides high-level evaluation functions that load a checkpoint,
run evaluation on a dataset, and produce complete metric reports.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_all_classification_metrics
from .train import evaluate as _evaluate_loop
from .utils import load_checkpoint, save_metrics

logger = logging.getLogger("cva")


def evaluate_checkpoint(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    num_classes: int = 2,
    desc: str = "Test",
) -> Dict[str, Any]:
    """
    Evaluate a model (optionally loaded from checkpoint) on a dataset.

    Args:
        model: Model architecture (must match checkpoint).
        dataloader: Evaluation data loader.
        device: Device.
        checkpoint_path: Path to checkpoint. If None, uses model as-is.
        num_classes: Number of classes.
        desc: Description for progress bar.

    Returns:
        Dictionary with all evaluation metrics.
    """
    if checkpoint_path is not None:
        ckpt = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"Loaded checkpoint from {checkpoint_path} "
            f"(epoch {ckpt.get('epoch', '?')})"
        )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    metrics = _evaluate_loop(
        model, dataloader, criterion, device,
        num_classes=num_classes, desc=desc,
    )

    return metrics


def print_evaluation_report(metrics: Dict[str, Any], title: str = "Evaluation Report"):
    """
    Print a formatted evaluation report.

    Args:
        metrics: Evaluation metrics dictionary.
        title: Report title.
    """
    from .datasets import WATERBIRDS_GROUP_NAMES

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    print(f"\n  Overall Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy:      {metrics.get('balanced_accuracy', 'N/A')}")
    print(f"  Macro F1:               {metrics.get('macro_f1', 'N/A')}")
    print(f"  Worst-Group Accuracy:   {metrics['worst_group_accuracy']:.4f}")
    print(f"  Worst Group:            {metrics.get('worst_group_id', 'N/A')}")

    print(f"\n--- Per-Group Accuracy {'---' * 10}")
    group_accs = metrics.get("group_accuracy", {})
    group_counts = metrics.get("group_counts", {})
    for g in sorted(group_accs.keys()):
        name = WATERBIRDS_GROUP_NAMES.get(g, f"Group {g}")
        count = group_counts.get(g, "?")
        print(f"  [{g}] {name:25s}  Acc: {group_accs[g]:.4f}  (n={count})")

    print(f"\n--- Confusion Matrix {'---' * 10}")
    cm = metrics.get("confusion_matrix", [])
    if cm:
        cm = np.array(cm)
        print(f"  Predicted ->  {'  '.join(str(i) for i in range(cm.shape[1]))}")
        for i, row in enumerate(cm):
            print(f"  Actual {i}:    {'  '.join(str(int(v)) for v in row)}")

    print(f"\n  Loss:                   {metrics.get('loss', 'N/A')}")
    print(f"  Total samples:          {metrics.get('total_samples', 'N/A')}")
    print(f"{'=' * 60}\n")
