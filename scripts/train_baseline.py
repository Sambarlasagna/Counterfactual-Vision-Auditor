#!/usr/bin/env python3
"""
Train a baseline classifier on the Waterbirds dataset.

Usage:
    python scripts/train_baseline.py --config configs/resnet18.yaml
    python scripts/train_baseline.py --config configs/resnet18.yaml --smoke-test
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cva.config import load_config, save_config
from cva.datasets import (
    WaterbirdsDataset,
    create_dataloader,
    create_smoke_subset,
    get_dataset_stats,
)
from cva.models import create_model, get_model_info
from cva.train import train
from cva.utils import (
    set_seed,
    get_device,
    get_device_info,
    setup_logging,
    ensure_dirs,
    save_metrics,
    timer,
)


def main():
    parser = argparse.ArgumentParser(description="Train CVA baseline model")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g., configs/resnet18.yaml)",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run in smoke test mode (tiny subset, few epochs)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.smoke_test:
        config["smoke_test"]["enabled"] = True

    # Setup
    seed = config.get("seed", 42)
    set_seed(seed, config.get("deterministic", True))
    device = get_device()
    logger = setup_logging(
        log_dir=config["logging"].get("log_dir", "outputs/logs"),
        name="cva",
    )

    # Ensure output directories
    ensure_dirs(
        config["checkpointing"]["save_dir"],
        config["logging"]["log_dir"],
    )

    # Log environment
    device_info = get_device_info()
    logger.info(f"Device: {device} ({device_info.get('gpu_name', 'CPU')})")
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {seed}")

    # Save the config used for this run
    model_name = config["model"]["name"]
    save_config(config, f"outputs/logs/{model_name}_config.yaml")

    # Load dataset
    logger.info("Loading dataset...")
    dataset_cfg = config["dataset"]
    image_size = dataset_cfg.get("image_size", 224)

    try:
        train_dataset = WaterbirdsDataset(
            root=dataset_cfg["root"],
            split="train",
            image_size=image_size,
        )
        val_dataset = WaterbirdsDataset(
            root=dataset_cfg["root"],
            split="val",
            image_size=image_size,
        )
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.error("Please download the dataset first. See data/README.md")
        sys.exit(1)

    # Smoke test subset
    if config["smoke_test"]["enabled"]:
        num_samples = config["smoke_test"]["num_samples"]
        train_dataset = create_smoke_subset(train_dataset, num_samples)
        val_dataset = create_smoke_subset(val_dataset, num_samples // 2)
        logger.info(f"Smoke test: using {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Log dataset stats
    logger.info(f"Train: {len(train_dataset)} samples")
    logger.info(f"Val:   {len(val_dataset)} samples")

    # Create data loaders
    train_cfg = config["training"]
    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=dataset_cfg.get("num_workers", 0),
        pin_memory=dataset_cfg.get("pin_memory", True),
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 0),
        pin_memory=dataset_cfg.get("pin_memory", True),
    )

    # Create model
    model = create_model(config)
    model_info = get_model_info(model)
    logger.info(f"Model: {model_name} ({model_info['total_params']:,} params)")

    # Train
    with timer("Training", logger):
        results = train(model, train_loader, val_loader, config, device)

    # Log final results
    logger.info(f"Best {config['checkpointing']['metric']}: {results['best_metric']:.4f} "
                f"at epoch {results['best_epoch']}")

    # Save training history
    history_path = f"outputs/metrics/{model_name}_training_history.json"
    save_metrics(results["history"], history_path)
    logger.info(f"Training history saved to {history_path}")

    print("\nTraining complete!")
    print(f"  Best checkpoint: outputs/checkpoints/{model_name}_best.pt")
    print(f"  Training history: {history_path}")


if __name__ == "__main__":
    main()
