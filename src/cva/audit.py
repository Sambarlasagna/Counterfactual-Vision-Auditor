"""
Full audit pipeline for CVA.

Runs interventions across a dataset, collects per-example scores,
aggregates by group, and compares with baseline group metrics.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .datasets import WaterbirdsDataset, WATERBIRDS_GROUP_NAMES
from .interventions import InterventionRunner, SegmentationModel
from .metrics import compute_all_intervention_metrics
from .utils import save_metrics

logger = logging.getLogger("cva")


def run_audit(
    model: torch.nn.Module,
    dataset: WaterbirdsDataset,
    device: torch.device,
    intervention_types: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    cache_dir: str = "outputs/masks",
    output_dir: str = "outputs/metrics",
    max_samples: Optional[int] = None,
    image_size: int = 224,
) -> Dict[str, Any]:
    """
    Run the full audit pipeline.

    For each intervention type and method:
      1. Generate/load foreground masks
      2. Apply intervention to each test image
      3. Collect original and modified predictions
      4. Compute intervention metrics (flip rate, confidence change, etc.)
      5. Aggregate by group

    Args:
        model: Trained classifier to audit.
        dataset: Test dataset.
        device: Device for inference.
        intervention_types: List of intervention types to run.
        methods: List of methods per intervention type.
        cache_dir: Directory for mask caching.
        output_dir: Directory for saving results.
        max_samples: Limit the number of samples processed (for debugging).
        image_size: Image size for model input.

    Returns:
        Dictionary with all audit results.
    """
    if intervention_types is None:
        intervention_types = ["background_modification", "foreground_suppression"]

    if methods is None:
        methods = ["gray", "blur"]

    # Setup
    seg_model = SegmentationModel(device=device, cache_dir=cache_dir)
    runner = InterventionRunner(model, seg_model, device, image_size=image_size)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

    all_results = {}

    for intervention_type in intervention_types:
        for method in methods:
            key = f"{intervention_type}__{method}"
            logger.info(f"Running audit: {key} ({num_samples} samples)")

            start_time = time.time()

            # Collect predictions
            all_probs_orig = []
            all_probs_mod = []
            all_labels = []
            all_groups = []
            all_preds_orig = []
            all_preds_mod = []
            all_flipped = []

            # Sample examples for visualization
            viz_examples = []
            viz_count = 0
            max_viz = 20  # Save up to 20 examples for visualization

            for idx in tqdm(
                range(num_samples),
                desc=f"Audit [{intervention_type}/{method}]",
                leave=False,
            ):
                # Load raw image (without transforms)
                sample = dataset.samples[idx]
                try:
                    image = Image.open(sample["img_path"]).convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to load {sample['img_path']}: {e}")
                    continue

                label = sample["label"]
                group_id = sample["group_id"]
                image_path = sample["img_path"]

                # Run intervention
                result = runner.run_intervention(
                    image=image,
                    image_path=image_path,
                    intervention_type=intervention_type,
                    method=method,
                )

                all_probs_orig.append(result["original_probs"])
                all_probs_mod.append(result["modified_probs"])
                all_labels.append(label)
                all_groups.append(group_id)
                all_preds_orig.append(result["original_pred"])
                all_preds_mod.append(result["modified_pred"])
                all_flipped.append(result["prediction_flipped"])

                # Collect visualization examples (balanced across groups)
                if viz_count < max_viz:
                    viz_examples.append({
                        "idx": idx,
                        "image_path": image_path,
                        "label": label,
                        "group_id": group_id,
                        "original_pred": result["original_pred"],
                        "modified_pred": result["modified_pred"],
                        "original_probs": result["original_probs"].tolist(),
                        "modified_probs": result["modified_probs"].tolist(),
                        "flipped": result["prediction_flipped"],
                    })
                    viz_count += 1

            # Convert to arrays
            all_probs_orig = np.array(all_probs_orig)
            all_probs_mod = np.array(all_probs_mod)
            all_labels = np.array(all_labels)
            all_groups = np.array(all_groups)

            # Compute intervention metrics
            intervention_metrics = compute_all_intervention_metrics(
                all_probs_orig, all_probs_mod, all_labels, all_groups
            )

            elapsed = time.time() - start_time

            # Compile results
            audit_result = {
                "intervention_type": intervention_type,
                "method": method,
                "num_samples": num_samples,
                "elapsed_seconds": round(elapsed, 2),
                "metrics": intervention_metrics,
                "examples": viz_examples,
            }

            all_results[key] = audit_result

            # Log summary
            logger.info(
                f"  {key}: flip_rate={intervention_metrics['flip_rate']:.4f}, "
                f"mean_prob_change={intervention_metrics['mean_probability_change']:.4f}, "
                f"mean_jsd={intervention_metrics['mean_jsd']:.4f} "
                f"({elapsed:.1f}s)"
            )

            # Log per-group
            per_group = intervention_metrics.get("per_group", {})
            for g in sorted(per_group.keys()):
                name = WATERBIRDS_GROUP_NAMES.get(g, f"Group {g}")
                gm = per_group[g]
                logger.info(
                    f"    [{g}] {name:25s} "
                    f"flip={gm['flip_rate']:.4f}  "
                    f"prob_change={gm['mean_probability_change']:.4f}  "
                    f"jsd={gm['mean_jsd']:.4f}  "
                    f"(n={gm['count']})"
                )

            # Save per-intervention results
            save_path = output_path / f"audit_{key}.json"
            save_metrics(audit_result, str(save_path))

    # Save combined results
    combined_path = output_path / "audit_combined.json"
    # Remove large arrays for JSON serialization
    combined_save = {}
    for k, v in all_results.items():
        combined_save[k] = {
            "intervention_type": v["intervention_type"],
            "method": v["method"],
            "num_samples": v["num_samples"],
            "elapsed_seconds": v["elapsed_seconds"],
            "metrics": v["metrics"],
        }
    save_metrics(combined_save, str(combined_path))
    logger.info(f"All audit results saved to {output_path}")

    return all_results


def print_audit_summary(results: Dict[str, Any]) -> None:
    """Print a formatted summary of audit results."""
    print(f"\n{'=' * 70}")
    print(f"  Counterfactual Vision Auditor -- Audit Summary")
    print(f"{'=' * 70}")

    for key, result in results.items():
        metrics = result["metrics"]
        print(f"\n--- {key} {'---' * 15}")
        print(f"  Samples:              {result['num_samples']}")
        print(f"  Flip Rate:            {metrics['flip_rate']:.4f}")
        print(f"  Mean Prob Change:     {metrics['mean_probability_change']:.4f}")
        print(f"  Mean Confidence Drop: {metrics['mean_confidence_change']:.4f}")
        print(f"  Mean JSD:             {metrics['mean_jsd']:.4f}")

        per_group = metrics.get("per_group", {})
        if per_group:
            print(f"\n  Per-Group Breakdown:")
            print(f"  {'Group':30s} {'Flip':>8s} {'ProbChg':>10s} {'JSD':>8s} {'N':>6s}")
            print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*6}")
            for g in sorted(per_group.keys()):
                name = WATERBIRDS_GROUP_NAMES.get(g, f"Group {g}")
                gm = per_group[g]
                print(
                    f"  {name:30s} {gm['flip_rate']:8.4f} "
                    f"{gm['mean_probability_change']:10.4f} "
                    f"{gm['mean_jsd']:8.4f} "
                    f"{gm['count']:6d}"
                )

    print(f"\n{'=' * 70}\n")
