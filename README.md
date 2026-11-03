# Counterfactual Vision Auditor (CVA)

A reproducible tool for auditing whether image classifiers rely on background
context instead of foreground objects when making predictions.

## Overview

This project investigates **shortcut learning** in vision models by:
1. Training baseline classifiers on the [Waterbirds benchmark](https://github.com/kohpangwei/group_DRO)
2. Applying **counterfactual interventions** (background modification, foreground suppression)
3. Measuring how predictions change when foreground or background is altered
4. Testing whether intervention-derived audit scores predict failures under distribution shift

## Research Question

> How reliably do different foreground/background interventions reveal background
> dependence in image classifiers, and do the resulting audit scores predict
> failures under background distribution shifts?

---

## Results

### Baseline Evaluation (ResNet-18, Test Set)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 60.56% |
| Balanced Accuracy | 59.77% |
| Worst-Group Accuracy | 24.45% |

### Per-Group Accuracy

| Group | Description | Accuracy | Count |
|-------|-------------|----------|-------|
| 0 | Landbird on Land (majority) | **94.99%** | 2,255 |
| 1 | Landbird on Water (minority) | 27.41% | 2,255 |
| 2 | Waterbird on Land (minority) | **24.45%** | 642 |
| 3 | Waterbird on Water (majority) | **92.21%** | 642 |

> The model achieves >92% accuracy when bird type matches the typical background
> but drops below 28% on mismatched groups — clear evidence of background-dependent
> shortcut learning.

### Counterfactual Audit Results (500 test samples)

| Intervention | Method | Flip Rate | Mean JSD |
|-------------|--------|-----------|----------|
| Background Modification | Gray | **47.2%** | 0.157 |
| Background Modification | Blur | **46.4%** | 0.143 |
| Foreground Suppression | Gray | 10.2% | 0.027 |
| Foreground Suppression | Blur | 8.6% | 0.020 |

**Key finding:** Changing the background flips the model's prediction **~47% of the time**,
while masking the foreground (the actual bird) only flips it **~9% of the time**.
This confirms the model relies far more on background context than on the foreground object.

### Per-Group Flip Rates (Background Modification, Gray)

| Group | Flip Rate | JSD |
|-------|-----------|-----|
| Landbird on Land | 71.8% | 0.272 |
| Landbird on Water | 25.0% | 0.051 |
| Waterbird on Land | 79.5% | 0.265 |
| Waterbird on Water | 10.5% | 0.032 |

> Groups where the model achieves high accuracy (matching background) show the
> **highest sensitivity to background changes** (>70% flip rate), confirming that
> correct predictions in these groups are driven by background cues.

### Generated Figures

| Figure | Description |
|--------|-------------|
| `outputs/figures/training_history.png` | Training curves (loss, accuracy, worst-group accuracy) |
| `outputs/figures/group_accuracy.png` | Per-group test accuracy bar chart |
| `outputs/figures/confusion_matrix.png` | Test set confusion matrix |
| `outputs/figures/intervention_comparison.png` | Comparison across all interventions |
| `outputs/figures/flip_rate_*.png` | Per-group flip rates for each intervention |
| `outputs/figures/audit_vs_error_*.png` | Audit score vs group error rate scatter plots |

---

## Quick Start

### 1. Environment Setup

```bash
# Activate existing virtual environment (Windows):
D:\Programming\Projects\ml-venv\Scripts\activate

# Or create a new one:
python -m venv venv
pip install -r requirements.txt
```

### 2. Verify Environment

```bash
python scripts/check_environment.py
```

### 3. Download Dataset

See [`data/README.md`](data/README.md) for dataset download instructions.

### 4. Train Baseline

```bash
python scripts/train_baseline.py --config configs/resnet18.yaml
```

### 5. Evaluate

```bash
python scripts/evaluate_baseline.py --checkpoint outputs/checkpoints/resnet18_best.pt
```

### 6. Run Audit

```bash
python scripts/run_audit.py --checkpoint outputs/checkpoints/resnet18_best.pt --max-samples 500
```

### 7. Generate Report

```bash
python scripts/generate_report.py
```

### Smoke Test

To verify the full pipeline quickly without training:

```bash
python scripts/train_baseline.py --config configs/resnet18.yaml --smoke-test
```

---

## Project Structure

```
├── configs/              # Experiment configuration files (YAML)
│   ├── resnet18.yaml     # ResNet-18 baseline config
│   └── vit.yaml          # ViT config (stretch goal)
├── data/                 # Dataset directory (not committed; see data/README.md)
├── src/cva/              # Core library
│   ├── config.py         # Configuration loader + validation
│   ├── datasets.py       # Waterbirds data loading with group metadata
│   ├── models.py         # Model factory (ResNet-18, ViT)
│   ├── train.py          # Training loop with AMP, checkpointing
│   ├── evaluate.py       # Evaluation with per-group metrics
│   ├── interventions.py  # Background modification, foreground suppression, background swap
│   ├── audit.py          # Full counterfactual audit pipeline
│   ├── metrics.py        # Classification + intervention metrics (flip rate, JSD, etc.)
│   ├── visualization.py  # All plots and figures
│   └── utils.py          # Seeds, device detection, logging, I/O
├── scripts/              # CLI entry points
│   ├── check_environment.py
│   ├── download_dataset.py
│   ├── extract_dataset.py
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   ├── run_audit.py
│   └── generate_report.py
├── tests/                # Unit and smoke tests (46 tests)
│   ├── test_smoke.py
│   ├── test_interventions.py
│   └── test_metrics.py
├── outputs/              # Generated results
│   ├── checkpoints/      # Model checkpoints (~134 MB each)
│   ├── figures/          # Generated plots (12 figures)
│   ├── metrics/          # JSON metrics and audit results
│   ├── logs/             # Training logs and per-epoch metrics
│   ├── masks/            # Cached segmentation masks
│   └── reports/          # Generated summary report
└── docs/                 # Documentation
    ├── related_work.md   # Literature review table
    ├── experiment_plan.md # Hypotheses and experimental protocol
    └── limitations.md    # Known limitations and caveats
```

## Hardware Requirements

- **GPU:** NVIDIA RTX 3050 Laptop (6 GB VRAM) or equivalent
- **Training time:** ~27 minutes for 30 epochs (ResNet-18)
- **Audit time:** ~86 seconds for 500 samples (4 interventions)
- **CPU fallback:** Supported for inference, significantly slower for training
- **Disk:** ~2 GB for dataset, ~940 MB for all checkpoints, ~160 MB for segmentation model

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | ResNet-18 (pretrained ImageNet) |
| Optimizer | Adam |
| Learning Rate | 0.001 (cosine annealing) |
| Batch Size | 16 |
| Epochs | 30 |
| Mixed Precision | Enabled (AMP) |
| Seed | 42 |

## Key Findings

1. **H1 confirmed:** The ERM-trained model shows extreme group disparity — 95% accuracy
   on majority groups vs. 24-27% on minority groups.

2. **H2 confirmed:** Background modification causes ~47% prediction flips overall,
   with majority-background groups showing >70% flip rates.

3. **H3 confirmed:** Foreground suppression causes only ~10% flips — many "correct"
   predictions survive even when the bird is removed, proving they are background-driven.

4. **H4 partially confirmed:** Groups with higher audit scores (flip rate, JSD) also
   have the highest error rates under distribution shift. The pattern is consistent
   but statistical significance is limited with only 4 groups and 1 model.

## Limitations

- Results are from a single seed and single model architecture (ResNet-18)
- Segmentation masks from DeepLabV3 may introduce compositing artifacts
- Background replacement creates unnatural images; model response to unnaturalness
  is a potential confound
- Intervention effects suggest sensitivity, not causal reliance
- Findings are specific to the Waterbirds benchmark and may not generalize

See [`docs/limitations.md`](docs/limitations.md) for a full discussion.

## Tests

Run the test suite (46 tests covering config, model, interventions, metrics, and end-to-end):

```bash
python -m pytest tests/ -v
```

## References

- Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts* (ICLR 2020)
- Geirhos et al., *Shortcut Learning in Deep Neural Networks*, Nature Machine Intelligence (2020)
- Koh et al., *WILDS: A Benchmark of in-the-Wild Distribution Shifts* (ICML 2021)

See [`docs/related_work.md`](docs/related_work.md) for the full literature review.

## License

This project is for academic/research purposes.
