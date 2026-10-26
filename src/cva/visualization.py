"""
Visualization utilities for CVA experiments.

Generates:
  - Group accuracy bar plots
  - Prediction change distributions
  - Before/after counterfactual example grids
  - Intervention score vs. error scatter plots
  - Ablation comparison tables
  - Training history curves
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from PIL import Image

from .datasets import WATERBIRDS_GROUP_NAMES

logger = logging.getLogger("cva")

# Style settings
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = sns.color_palette("Set2", 8)
GROUP_COLORS = {0: COLORS[0], 1: COLORS[1], 2: COLORS[2], 3: COLORS[3]}


def save_figure(fig, filepath: str, dpi: int = 150) -> None:
    """Save a matplotlib figure and close it."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved figure: {filepath}")


# ── Classification Visualizations ──────────────────────────────────────

def plot_group_accuracy(
    group_accs: Dict[int, float],
    worst_group_id: int,
    overall_acc: float,
    save_path: str = "outputs/figures/group_accuracy.png",
    title: str = "Per-Group Accuracy",
) -> None:
    """
    Bar plot of per-group accuracy with overall accuracy line.

    Args:
        group_accs: Dict of group_id -> accuracy.
        worst_group_id: Group with worst accuracy (highlighted).
        overall_acc: Overall accuracy (horizontal line).
        save_path: Path to save the figure.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = sorted(group_accs.keys())
    names = [WATERBIRDS_GROUP_NAMES.get(g, f"Group {g}") for g in groups]
    values = [group_accs[g] for g in groups]
    colors = [
        "#e74c3c" if g == worst_group_id else GROUP_COLORS.get(g, COLORS[0])
        for g in groups
    ]

    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=1.5)

    # Overall accuracy line
    ax.axhline(y=overall_acc, color="gray", linestyle="--", linewidth=1.5,
               label=f"Overall: {overall_acc:.3f}")

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10,
        )

    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    plt.xticks(rotation=15, ha="right")

    save_figure(fig, save_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: str = "outputs/figures/confusion_matrix.png",
    title: str = "Confusion Matrix",
) -> None:
    """Plot a heatmap confusion matrix."""
    if class_names is None:
        class_names = ["Landbird", "Waterbird"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=1, linecolor="white",
    )
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_title(title, fontweight="bold")

    save_figure(fig, save_path)


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = "outputs/figures/training_history.png",
) -> None:
    """Plot training loss, accuracy, and worst-group accuracy over epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train", color=COLORS[0])
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val", color=COLORS[1])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train", color=COLORS[0])
    axes[1].plot(epochs, history["val_acc"], "s-", label="Val", color=COLORS[1])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy", fontweight="bold")
    axes[1].legend()

    # Worst-group accuracy
    axes[2].plot(epochs, history["val_worst_group_acc"], "D-",
                 label="Val Worst-Group", color=COLORS[2])
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Worst-Group Accuracy", fontweight="bold")
    axes[2].legend()

    plt.tight_layout()
    save_figure(fig, save_path)


# ── Intervention Visualizations ────────────────────────────────────────

def plot_prediction_change_distribution(
    prob_changes: np.ndarray,
    groups: np.ndarray,
    save_path: str = "outputs/figures/prediction_changes.png",
    title: str = "True-Class Probability Change Distribution",
) -> None:
    """
    Histogram/KDE of probability changes, colored by group.

    Args:
        prob_changes: (N,) array of probability changes.
        groups: (N,) group IDs.
        save_path: Path to save.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for g in sorted(np.unique(groups)):
        mask = groups == g
        name = WATERBIRDS_GROUP_NAMES.get(int(g), f"Group {g}")
        sns.kdeplot(
            prob_changes[mask], ax=ax, label=name,
            color=GROUP_COLORS.get(int(g), None),
            linewidth=2, fill=True, alpha=0.3,
        )

    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("True-Class Probability Change (original - intervened)")
    ax.set_ylabel("Density")
    ax.set_title(title, fontweight="bold")
    ax.legend()

    save_figure(fig, save_path)


def plot_flip_rate_by_group(
    per_group_metrics: Dict[int, Dict[str, Any]],
    save_path: str = "outputs/figures/flip_rate_by_group.png",
    title: str = "Prediction Flip Rate by Group",
) -> None:
    """Bar plot of flip rate per group."""
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = sorted(per_group_metrics.keys())
    names = [WATERBIRDS_GROUP_NAMES.get(g, f"Group {g}") for g in groups]
    flip_rates = [per_group_metrics[g]["flip_rate"] for g in groups]
    colors = [GROUP_COLORS.get(g, COLORS[0]) for g in groups]

    bars = ax.bar(names, flip_rates, color=colors, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, flip_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontweight="bold",
        )

    ax.set_ylabel("Flip Rate")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, max(flip_rates) * 1.3 if flip_rates else 1.0)
    plt.xticks(rotation=15, ha="right")

    save_figure(fig, save_path)


def plot_intervention_comparison(
    audit_results: Dict[str, Any],
    save_path: str = "outputs/figures/intervention_comparison.png",
    title: str = "Intervention Method Comparison",
) -> None:
    """
    Grouped bar plot comparing metrics across intervention methods.

    Args:
        audit_results: Dict with keys like 'background_modification__gray'.
        save_path: Path to save.
        title: Plot title.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    methods = list(audit_results.keys())
    short_names = [k.split("__")[-1] for k in methods]

    # Flip rate
    flip_rates = [audit_results[m]["metrics"]["flip_rate"] for m in methods]
    axes[0].bar(short_names, flip_rates, color=COLORS[:len(methods)],
                edgecolor="white", linewidth=1.5)
    axes[0].set_ylabel("Flip Rate")
    axes[0].set_title("Prediction Flip Rate", fontweight="bold")
    for i, v in enumerate(flip_rates):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

    # Mean probability change
    prob_changes = [
        audit_results[m]["metrics"]["mean_probability_change"] for m in methods
    ]
    axes[1].bar(short_names, prob_changes, color=COLORS[:len(methods)],
                edgecolor="white", linewidth=1.5)
    axes[1].set_ylabel("Mean Prob Change")
    axes[1].set_title("True-Class Probability Change", fontweight="bold")
    for i, v in enumerate(prob_changes):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")

    # Mean JSD
    jsds = [audit_results[m]["metrics"]["mean_jsd"] for m in methods]
    axes[2].bar(short_names, jsds, color=COLORS[:len(methods)],
                edgecolor="white", linewidth=1.5)
    axes[2].set_ylabel("Mean JSD")
    axes[2].set_title("Jensen-Shannon Divergence", fontweight="bold")
    for i, v in enumerate(jsds):
        axes[2].text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")

    plt.suptitle(title, fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_counterfactual_examples(
    examples: List[Dict],
    save_path: str = "outputs/figures/counterfactual_examples.png",
    max_examples: int = 8,
    title: str = "Counterfactual Examples",
) -> None:
    """
    Grid showing original, mask, and modified images with predictions.

    Args:
        examples: List of dicts from audit results with image_path, mask, etc.
        save_path: Path to save.
        max_examples: Maximum number of examples to show.
        title: Plot title.
    """
    n = min(len(examples), max_examples)
    if n == 0:
        logger.warning("No examples to plot.")
        return

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    class_names = ["Landbird", "Waterbird"]

    for i in range(n):
        ex = examples[i]

        # Load original image
        try:
            orig_img = Image.open(ex["image_path"]).convert("RGB")
        except Exception:
            continue

        # Original
        axes[i, 0].imshow(orig_img)
        orig_pred = ex["original_pred"]
        orig_prob = ex["original_probs"][orig_pred]
        axes[i, 0].set_title(
            f"Original: {class_names[orig_pred]} ({orig_prob:.2f})",
            fontsize=10,
        )
        axes[i, 0].axis("off")

        # Group info
        group_name = WATERBIRDS_GROUP_NAMES.get(
            ex["group_id"], f"Group {ex['group_id']}"
        )
        axes[i, 0].set_ylabel(group_name, fontsize=9, rotation=0, labelpad=80)

        # Placeholder for mask/modified (would need actual images)
        axes[i, 1].text(
            0.5, 0.5, "Mask",
            ha="center", va="center", fontsize=12, color="gray",
        )
        axes[i, 1].set_title("Segmentation Mask", fontsize=10)
        axes[i, 1].axis("off")

        # Modified prediction info
        mod_pred = ex["modified_pred"]
        mod_prob = ex["modified_probs"][mod_pred]
        flipped = "FLIPPED" if ex["flipped"] else "same"
        color = "red" if ex["flipped"] else "green"
        axes[i, 2].text(
            0.5, 0.5,
            f"{class_names[mod_pred]} ({mod_prob:.2f})\n[{flipped}]",
            ha="center", va="center", fontsize=12, color=color, fontweight="bold",
        )
        axes[i, 2].set_title("After Intervention", fontsize=10)
        axes[i, 2].axis("off")

    plt.suptitle(title, fontweight="bold", fontsize=14)
    plt.tight_layout()
    save_figure(fig, save_path)


def plot_audit_score_vs_error(
    group_audit_scores: Dict[int, float],
    group_error_rates: Dict[int, float],
    save_path: str = "outputs/figures/audit_vs_error.png",
    score_name: str = "Flip Rate",
    title: str = "Audit Score vs. Group Error Rate",
) -> None:
    """
    Scatter plot of audit score vs. group error rate.

    Args:
        group_audit_scores: Dict of group_id -> audit metric value.
        group_error_rates: Dict of group_id -> error rate (1 - accuracy).
        save_path: Path to save.
        score_name: Name of the audit metric.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    groups = sorted(set(group_audit_scores.keys()) & set(group_error_rates.keys()))

    xs = [group_audit_scores[g] for g in groups]
    ys = [group_error_rates[g] for g in groups]
    colors_list = [GROUP_COLORS.get(g, COLORS[0]) for g in groups]
    names = [WATERBIRDS_GROUP_NAMES.get(g, f"G{g}") for g in groups]

    ax.scatter(xs, ys, c=colors_list, s=150, edgecolors="black", zorder=5)

    for x, y, name in zip(xs, ys, names):
        ax.annotate(
            name, (x, y), textcoords="offset points",
            xytext=(8, 8), fontsize=9,
        )

    # Trend line if enough points
    if len(groups) >= 3:
        from scipy import stats
        slope, intercept, r_val, p_val, _ = stats.linregress(xs, ys)
        x_line = np.linspace(min(xs), max(xs), 50)
        ax.plot(x_line, slope * x_line + intercept, "--", color="gray", alpha=0.7,
                label=f"r={r_val:.3f}, p={p_val:.3f}")
        ax.legend()

    ax.set_xlabel(score_name, fontweight="bold")
    ax.set_ylabel("Error Rate (1 - Accuracy)", fontweight="bold")
    ax.set_title(title, fontweight="bold")

    save_figure(fig, save_path)


def generate_all_figures(
    eval_metrics: Dict[str, Any],
    audit_results: Dict[str, Any],
    training_history: Optional[Dict[str, List[float]]] = None,
    output_dir: str = "outputs/figures",
) -> List[str]:
    """
    Generate all standard figures for the project.

    Args:
        eval_metrics: Classification evaluation metrics.
        audit_results: Audit pipeline results.
        training_history: Optional training history.
        output_dir: Directory to save figures.

    Returns:
        List of saved figure paths.
    """
    saved = []
    out = Path(output_dir)

    # 1. Group accuracy
    if "group_accuracy" in eval_metrics:
        path = str(out / "group_accuracy.png")
        plot_group_accuracy(
            eval_metrics["group_accuracy"],
            eval_metrics.get("worst_group_id", -1),
            eval_metrics["accuracy"],
            save_path=path,
        )
        saved.append(path)

    # 2. Confusion matrix
    if "confusion_matrix" in eval_metrics:
        path = str(out / "confusion_matrix.png")
        cm = np.array(eval_metrics["confusion_matrix"])
        plot_confusion_matrix(cm, save_path=path)
        saved.append(path)

    # 3. Training history
    if training_history:
        path = str(out / "training_history.png")
        plot_training_history(training_history, save_path=path)
        saved.append(path)

    # 4. Intervention comparison
    if audit_results:
        path = str(out / "intervention_comparison.png")
        plot_intervention_comparison(audit_results, save_path=path)
        saved.append(path)

    # 5. Per-intervention figures
    for key, result in audit_results.items():
        metrics = result["metrics"]
        safe_key = key.replace("/", "_")

        # Flip rate by group
        if "per_group" in metrics:
            path = str(out / f"flip_rate_{safe_key}.png")
            plot_flip_rate_by_group(
                metrics["per_group"], save_path=path,
                title=f"Flip Rate by Group ({key})",
            )
            saved.append(path)

        # Counterfactual examples
        if "examples" in result:
            path = str(out / f"examples_{safe_key}.png")
            plot_counterfactual_examples(
                result["examples"], save_path=path,
                title=f"Counterfactual Examples ({key})",
            )
            saved.append(path)

    # 6. Audit score vs error (if we have both)
    if audit_results and "group_accuracy" in eval_metrics:
        group_accs = eval_metrics["group_accuracy"]
        group_errors = {g: 1.0 - a for g, a in group_accs.items()}

        for key, result in audit_results.items():
            per_group = result["metrics"].get("per_group", {})
            if per_group:
                flip_scores = {g: m["flip_rate"] for g, m in per_group.items()}
                safe_key = key.replace("/", "_")
                path = str(out / f"audit_vs_error_{safe_key}.png")
                plot_audit_score_vs_error(
                    flip_scores, group_errors, save_path=path,
                    score_name=f"Flip Rate ({key})",
                )
                saved.append(path)

    logger.info(f"Generated {len(saved)} figures in {output_dir}")
    return saved
