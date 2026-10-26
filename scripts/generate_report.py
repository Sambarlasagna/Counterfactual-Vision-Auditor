#!/usr/bin/env python3
"""
Generate figures and a summary report from saved experiment results.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --eval-metrics outputs/metrics/resnet18_test_metrics.json
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cva.utils import setup_logging, load_metrics
from cva.visualization import generate_all_figures


def main():
    parser = argparse.ArgumentParser(description="Generate CVA report and figures")
    parser.add_argument(
        "--eval-metrics", type=str, default=None,
        help="Path to evaluation metrics JSON (auto-detected if not provided)",
    )
    parser.add_argument(
        "--audit-results", type=str, default=None,
        help="Path to combined audit results JSON",
    )
    parser.add_argument(
        "--training-history", type=str, default=None,
        help="Path to training history JSON",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/figures",
        help="Directory to save generated figures",
    )
    args = parser.parse_args()

    logger = setup_logging(name="cva")

    metrics_dir = Path("outputs/metrics")
    figures_dir = Path(args.output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect files
    eval_metrics = {}
    if args.eval_metrics:
        eval_metrics = load_metrics(args.eval_metrics)
    else:
        # Look for test metrics
        candidates = list(metrics_dir.glob("*_test_metrics.json"))
        if candidates:
            eval_metrics = load_metrics(str(candidates[0]))
            logger.info(f"Auto-detected eval metrics: {candidates[0]}")

    audit_results = {}
    if args.audit_results:
        audit_results = load_metrics(args.audit_results)
    else:
        combined = metrics_dir / "audit_combined.json"
        if combined.exists():
            audit_results = load_metrics(str(combined))
            logger.info(f"Auto-detected audit results: {combined}")

    training_history = None
    if args.training_history:
        training_history = load_metrics(args.training_history)
    else:
        candidates = list(metrics_dir.glob("*_training_history.json"))
        if candidates:
            training_history = load_metrics(str(candidates[0]))
            logger.info(f"Auto-detected training history: {candidates[0]}")

    if not eval_metrics and not audit_results:
        logger.warning(
            "No metrics files found. Run training and evaluation first.\n"
            "  python scripts/train_baseline.py --config configs/resnet18.yaml\n"
            "  python scripts/evaluate_baseline.py --checkpoint outputs/checkpoints/resnet18_best.pt\n"
            "  python scripts/run_audit.py --checkpoint outputs/checkpoints/resnet18_best.pt"
        )
        sys.exit(1)

    # Generate figures
    saved = generate_all_figures(
        eval_metrics=eval_metrics,
        audit_results=audit_results,
        training_history=training_history,
        output_dir=args.output_dir,
    )

    # Generate text summary
    report_path = Path("outputs/reports/summary.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Counterfactual Vision Auditor -- Experiment Summary\n\n")
        f.write(f"Generated automatically from saved experiment results.\n\n")

        if eval_metrics:
            f.write("## Classification Metrics\n\n")
            f.write(f"- **Overall Accuracy:** {eval_metrics.get('accuracy', 'N/A')}\n")
            f.write(f"- **Balanced Accuracy:** {eval_metrics.get('balanced_accuracy', 'N/A')}\n")
            f.write(f"- **Worst-Group Accuracy:** {eval_metrics.get('worst_group_accuracy', 'N/A')}\n")
            f.write(f"- **Worst Group ID:** {eval_metrics.get('worst_group_id', 'N/A')}\n\n")

            group_accs = eval_metrics.get("group_accuracy", {})
            if group_accs:
                f.write("### Per-Group Accuracy\n\n")
                f.write("| Group | Accuracy | Count |\n")
                f.write("|-------|----------|-------|\n")
                group_counts = eval_metrics.get("group_counts", {})
                for g in sorted(group_accs.keys(), key=int):
                    count = group_counts.get(str(g), group_counts.get(int(g), "?"))
                    f.write(f"| {g} | {group_accs[g]:.4f} | {count} |\n")
                f.write("\n")

        if audit_results:
            f.write("## Audit Results\n\n")
            for key, result in audit_results.items():
                metrics = result.get("metrics", result)
                f.write(f"### {key}\n\n")
                f.write(f"- **Flip Rate:** {metrics.get('flip_rate', 'N/A')}\n")
                f.write(f"- **Mean Prob Change:** {metrics.get('mean_probability_change', 'N/A')}\n")
                f.write(f"- **Mean JSD:** {metrics.get('mean_jsd', 'N/A')}\n\n")

        if saved:
            f.write("## Generated Figures\n\n")
            for s in saved:
                f.write(f"- `{s}`\n")

    logger.info(f"Summary report saved to {report_path}")
    print(f"\nReport generated!")
    print(f"  Figures:  {args.output_dir}/ ({len(saved)} files)")
    print(f"  Summary:  {report_path}")


if __name__ == "__main__":
    main()
