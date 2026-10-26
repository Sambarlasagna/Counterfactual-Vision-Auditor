# Counterfactual Vision Auditor (CVA)

A reproducible tool for auditing whether image classifiers rely on background
context instead of foreground objects when making predictions.

## Overview

This project investigates **shortcut learning** in vision models by:
1. Training baseline classifiers on the Waterbirds benchmark
2. Applying **counterfactual interventions** (background modification, foreground suppression)
3. Measuring how predictions change when foreground or background is altered
4. Testing whether intervention-derived audit scores predict failures under distribution shift

## Quick Start

### 1. Environment Setup

```bash
# Use existing virtual environment
# Windows:
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
python scripts/run_audit.py --checkpoint outputs/checkpoints/resnet18_best.pt
```

### 7. Generate Report

```bash
python scripts/generate_report.py
```

## Project Structure

```
├── configs/           # Experiment configuration files (YAML)
├── data/              # Dataset directory (not committed; see data/README.md)
├── src/cva/           # Core library
│   ├── config.py      # Configuration loader
│   ├── datasets.py    # Data loading with group metadata
│   ├── models.py      # Model factory (ResNet-18, ViT)
│   ├── train.py       # Training loop
│   ├── evaluate.py    # Evaluation with group-wise metrics
│   ├── interventions.py  # Counterfactual interventions
│   ├── audit.py       # Full audit pipeline
│   ├── metrics.py     # Classification + intervention metrics
│   ├── visualization.py  # Plots and figures
│   └── utils.py       # Seeds, device, logging, checkpoints
├── scripts/           # CLI entry points
├── tests/             # Unit and smoke tests
├── outputs/           # Generated results (figures, metrics, reports)
└── docs/              # Documentation and related work
```

## Hardware Requirements

- **GPU:** NVIDIA RTX 3050 Laptop (6 GB VRAM) or equivalent
- **CPU fallback:** Supported but significantly slower
- **Disk:** ~2 GB for dataset, ~200 MB for checkpoints

## Research Question

> How reliably do different foreground/background interventions reveal background
> dependence in image classifiers, and do the resulting audit scores predict
> failures under background distribution shifts?

## License

This project is for academic/research purposes.

## Status

🚧 **Under active development** — See commits for progress.
