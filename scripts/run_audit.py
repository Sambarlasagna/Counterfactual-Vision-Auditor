#!/usr/bin/env python3
"""
Run the counterfactual audit pipeline on a trained model.

Usage:
    python scripts/run_audit.py --checkpoint outputs/checkpoints/resnet18_best.pt
    python scripts/run_audit.py --checkpoint outputs/checkpoints/resnet18_best.pt --max-samples 200
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cva.config import load_config
from cva.datasets import WaterbirdsDataset
from cva.models import create_model
from cva.audit import run_audit, print_audit_summary
from cva.utils import (
    set_seed,
    get_device,
    setup_logging,
    load_checkpoint,
    timer,
)


def main():
    parser = argparse.ArgumentParser(description="Run CVA counterfactual audit")
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
        help="Dataset split to audit",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples to process (for debugging)",
    )
    parser.add_argument(
        "--methods", type=str, nargs="+",
        default=["gray", "blur"],
        help="Intervention methods to run",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="outputs/masks",
        help="Directory for caching segmentation masks",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()
    logger = setup_logging(
        log_dir=config["logging"].get("log_dir", "outputs/logs"),
        name="cva",
    )

    logger.info(f"Running audit with checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}, Methods: {args.methods}")

    # Load dataset (raw, without training transforms)
    dataset_cfg = config["dataset"]
    try:
        dataset = WaterbirdsDataset(
            root=dataset_cfg["root"],
            split=args.split,
            image_size=dataset_cfg.get("image_size", 224),
        )
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        sys.exit(1)

    # Create model and load checkpoint
    model = create_model(config)
    ckpt = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Run audit
    with timer("Audit pipeline", logger):
        results = run_audit(
            model=model,
            dataset=dataset,
            device=device,
            methods=args.methods,
            cache_dir=args.cache_dir,
            output_dir="outputs/metrics",
            max_samples=args.max_samples,
            image_size=dataset_cfg.get("image_size", 224),
        )

    # Print summary
    print_audit_summary(results)

    print("\nAudit complete!")
    print(f"  Results saved to outputs/metrics/")
    print(f"  Masks cached at {args.cache_dir}/")


if __name__ == "__main__":
    main()
