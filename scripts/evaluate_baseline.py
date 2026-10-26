#!/usr/bin/env python3
"""
Evaluate a trained baseline model on the test set.

Usage:
    python scripts/evaluate_baseline.py --checkpoint outputs/checkpoints/resnet18_best.pt --config configs/resnet18.yaml
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cva.config import load_config
from cva.datasets import WaterbirdsDataset, create_dataloader
from cva.models import create_model
from cva.evaluate import evaluate_checkpoint, print_evaluation_report
from cva.utils import (
    set_seed,
    get_device,
    setup_logging,
    save_metrics,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CVA baseline model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/resnet18.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()
    logger = setup_logging(name="cva")

    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")

    # Load dataset
    dataset_cfg = config["dataset"]
    try:
        eval_dataset = WaterbirdsDataset(
            root=dataset_cfg["root"],
            split=args.split,
            image_size=dataset_cfg.get("image_size", 224),
        )
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        sys.exit(1)

    eval_loader = create_dataloader(
        eval_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 0),
    )

    # Create model and evaluate
    model = create_model(config)
    metrics = evaluate_checkpoint(
        model, eval_loader, device,
        checkpoint_path=args.checkpoint,
        num_classes=config["model"]["num_classes"],
        desc=args.split.capitalize(),
    )

    # Remove probabilities before saving (large array)
    probs = metrics.pop("probabilities", None)

    # Print report
    print_evaluation_report(metrics, title=f"{args.split.capitalize()} Evaluation Report")

    # Save metrics
    model_name = config["model"]["name"]
    metrics_path = f"outputs/metrics/{model_name}_{args.split}_metrics.json"
    save_metrics(metrics, metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
