"""
Training loop for CVA experiments.

Supports:
  - Mixed precision training (AMP)
  - Gradient clipping
  - Cosine annealing with warmup
  - Checkpoint saving (best + periodic)
  - Per-epoch group-wise evaluation on validation set
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_all_classification_metrics
from .utils import save_checkpoint, save_metrics

logger = logging.getLogger("cva")


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create an optimizer from config."""
    train_cfg = config["training"]
    name = train_cfg.get("optimizer", "adam").lower()
    lr = train_cfg["learning_rate"]
    wd = train_cfg.get("weight_decay", 0.0)

    params = filter(lambda p: p.requires_grad, model.parameters())

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: '{name}'")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create a learning rate scheduler from config."""
    train_cfg = config["training"]
    name = train_cfg.get("scheduler", "cosine").lower()
    epochs = train_cfg["epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 0)

    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1)
        )
        return scheduler
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: '{name}'")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: The model to train.
        dataloader: Training data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to train on.
        epoch: Current epoch number.
        config: Experiment configuration.
        scaler: Optional GradScaler for mixed precision.

    Returns:
        Dict with epoch training metrics.
    """
    model.train()
    train_cfg = config["training"]
    log_every = config["logging"].get("log_every", 10)
    use_amp = train_cfg.get("mixed_precision", False) and device.type == "cuda"
    grad_clip = train_cfg.get("gradient_clip", 0.0)

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_groups = []
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for batch_idx, (images, labels, groups, _meta) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_groups.extend(groups.numpy())

        if (batch_idx + 1) % log_every == 0:
            avg_loss = running_loss / num_batches
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

    # Compute epoch metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_groups = np.array(all_groups)

    epoch_metrics = {
        "loss": running_loss / max(num_batches, 1),
        "accuracy": float(np.mean(all_preds == all_labels)),
    }

    return epoch_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 2,
    desc: str = "Eval",
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset.

    Args:
        model: The model to evaluate.
        dataloader: Evaluation data loader.
        criterion: Loss function.
        device: Device.
        num_classes: Number of classes.
        desc: Description for progress bar.

    Returns:
        Dict with evaluation metrics including group-wise accuracy.
    """
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_groups = []
    all_probs = []
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"[{desc}]", leave=False)
    for images, labels, groups, _meta in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        num_batches += 1

        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_groups.extend(groups.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_groups = np.array(all_groups)
    all_probs = np.array(all_probs)

    # Compute all metrics
    metrics = compute_all_classification_metrics(
        all_labels, all_preds, all_groups, num_classes
    )
    metrics["loss"] = running_loss / max(num_batches, 1)
    metrics["probabilities"] = all_probs  # Keep for intervention analysis

    return metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Full training loop.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Experiment configuration.
        device: Training device.

    Returns:
        Dict with training history and best metrics.
    """
    train_cfg = config["training"]
    ckpt_cfg = config["checkpointing"]
    log_cfg = config["logging"]

    epochs = train_cfg["epochs"]
    use_amp = train_cfg.get("mixed_precision", False) and device.type == "cuda"
    num_classes = config["model"]["num_classes"]

    # Smoke test override
    if config.get("smoke_test", {}).get("enabled", False):
        epochs = config["smoke_test"]["epochs"]
        logger.info(f"SMOKE TEST MODE: training for {epochs} epochs only")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    scaler = GradScaler() if use_amp else None

    # Setup directories
    save_dir = Path(ckpt_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_cfg.get("log_dir", "outputs/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_worst_group_acc": [],
        "lr": [],
    }
    best_metric = 0.0
    best_epoch = 0

    logger.info(f"Starting training: {epochs} epochs, AMP={use_amp}")

    for epoch in range(epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, config, scaler,
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion, device,
            num_classes=num_classes, desc=f"Val E{epoch+1}",
        )

        # Remove large arrays from val_metrics before logging
        val_probs = val_metrics.pop("probabilities", None)

        # Step scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_worst_group_acc"].append(val_metrics["worst_group_accuracy"])
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Log
        logger.info(
            f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"WG-Acc: {val_metrics['worst_group_accuracy']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Log per-group accuracy
        group_accs = val_metrics.get("group_accuracy", {})
        group_str = " | ".join(
            f"G{g}: {a:.4f}" for g, a in sorted(group_accs.items())
        )
        logger.info(f"  Group Accuracies: {group_str}")

        # Save best checkpoint
        metric_name = ckpt_cfg.get("metric", "worst_group_accuracy")
        current_metric = val_metrics.get(metric_name, val_metrics["accuracy"])
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            if ckpt_cfg.get("save_best", True):
                model_name = config["model"]["name"]
                best_path = save_dir / f"{model_name}_best.pt"
                save_checkpoint(
                    model, optimizer, epoch, val_metrics, str(best_path),
                    config=config,
                )
                logger.info(
                    f"  Saved best checkpoint ({metric_name}={current_metric:.4f})"
                )

        # Save periodic checkpoint
        save_every = ckpt_cfg.get("save_every", 5)
        if save_every > 0 and (epoch + 1) % save_every == 0:
            model_name = config["model"]["name"]
            periodic_path = save_dir / f"{model_name}_epoch{epoch+1}.pt"
            save_checkpoint(
                model, optimizer, epoch, val_metrics, str(periodic_path),
                config=config,
            )

        # Save epoch metrics
        if log_cfg.get("save_metrics", True):
            epoch_results = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": {k: v for k, v in val_metrics.items() if k != "probabilities"},
                "lr": current_lr,
                "time_seconds": epoch_time,
            }
            metrics_path = log_dir / f"epoch_{epoch+1:03d}.json"
            save_metrics(epoch_results, str(metrics_path))

    # Final summary
    logger.info(
        f"Training complete. Best {metric_name}: {best_metric:.4f} at epoch {best_epoch}"
    )

    return {
        "history": history,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "total_epochs": epochs,
    }
