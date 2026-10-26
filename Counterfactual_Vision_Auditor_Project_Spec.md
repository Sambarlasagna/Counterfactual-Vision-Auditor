# Counterfactual Vision Auditor

## Project Specification, Implementation Plan, and Acceptance Criteria

**Purpose:** Build a technically credible Computer Vision + Deep
Learning project that can be demonstrated to a professor for
research-internship selection and presented on a resume.

**Target duration:** 2--4 weeks\
**Primary framework:** PyTorch\
**Target hardware:** Windows laptop with NVIDIA RTX 3050 Laptop GPU (6
GB VRAM), Ryzen 5 7640HS\
**Project category:** Visual model auditing, shortcut learning,
counterfactual evaluation, robustness\
**Working title:** Counterfactual Vision Auditor (CVA)

------------------------------------------------------------------------

## 1. Executive Summary

Build a reproducible tool that investigates whether image classifiers
rely on background context instead of the foreground object when making
predictions.

For example, a bird classifier may learn that certain bird classes tend
to appear near water or in forests. It may perform well on an ordinary
test set but fail when the same bird appears against an unfamiliar
background. The project will measure this failure using controlled
evaluation groups and counterfactual image interventions.

The deliverable must be more than a classifier, Grad-CAM visualization,
or polished interface. It must include:

1.  Reproducible baseline models.
2.  A well-defined evaluation protocol for distribution shift.
3.  At least two intervention methods that test sensitivity to
    foreground/background information.
4.  Quantitative comparisons and ablation studies.
5.  A concise technical report explaining what the experiments do and do
    not establish.
6.  A clean, usable demo and a professional Git repository.

**Important scientific constraint:** This project does not begin with a
claim that the method is novel. Shortcut learning, explainability,
counterfactual evaluation, and distribution robustness are established
research areas. The project must review related work before defining its
final contribution. Any claim of novelty must be narrow, supported by a
literature search, and validated by experiments.

------------------------------------------------------------------------

## 2. Motivation and Problem Statement

Deep neural networks can exploit statistical correlations that are
predictive in the training set but unreliable in new environments. In
image classification, these correlations can include backgrounds,
textures, watermarks, camera artifacts, or other contextual signals.

A conventional test score may conceal this behavior if the test
distribution preserves the same correlations as the training
distribution.

### Core problem

Given a trained image classifier and an image containing a foreground
object and its background:

-   Measure how the prediction changes when the foreground or background
    is altered.
-   Compare intervention-derived scores with actual performance under
    distribution shift.
-   Determine whether those scores are useful indicators of background
    dependence and failure.

### Main research question

**How reliably do different foreground/background interventions reveal
background dependence in image classifiers, and do the resulting audit
scores predict failures under background distribution shifts?**

This is a research-oriented engineering question, not a guaranteed novel
research contribution. The outcome may be positive, negative, or mixed.
All outcomes are acceptable if the experiments are rigorous and reported
honestly.

### Secondary questions

1.  Do different architectures show different levels of background
    dependence?
2.  Do masking and background replacement produce consistent audit
    rankings?
3.  Do intervention-derived scores correlate with worst-group accuracy
    or errors on shifted examples?
4.  How sensitive are results to segmentation quality and intervention
    artifacts?

------------------------------------------------------------------------

## 3. Project Scope

### Must-have deliverables

-   A training/evaluation pipeline using PyTorch.
-   A simple, documented dataset loader.
-   At least one trained baseline image classifier.
-   Group-wise evaluation under background-related distribution shift.
-   Two counterfactual intervention methods.
-   Metrics, plots, example visualizations, and saved results.
-   Reproducible configuration and random seeds.
-   A command-line workflow that works without a web interface.
-   A README with setup, commands, results, limitations, and
    screenshots.
-   A technical report of approximately 4--6 pages.

### Stretch goals, only after the core is reliable

-   Compare a CNN and a pretrained Vision Transformer.
-   Add a lightweight Streamlit dashboard for inspecting images and
    interventions.
-   Test a simple robustness intervention during training.
-   Add bootstrap confidence intervals or a paired statistical
    comparison.
-   Export an experiment summary to CSV/JSON.

### Explicitly out of scope

-   Training a large vision model from scratch.
-   Building a custom foundation model.
-   Claiming state-of-the-art results.
-   Claiming that a saliency map proves causal reasoning.
-   Building a complicated frontend before the experiments work.
-   Depending on a large diffusion model for image editing.
-   Spending the whole project integrating tools instead of evaluating a
    clear hypothesis.

------------------------------------------------------------------------

## 4. Dataset Strategy

### Preferred benchmark: Waterbirds

Use the Waterbirds benchmark, commonly accessed through the WILDS
ecosystem, if download and licensing/access conditions are workable. It
was designed to study correlations between bird class and background
environment. Its group structure enables evaluation beyond aggregate
accuracy.

**First task:** verify the current dataset access method, license/terms,
split definitions, metadata schema, and storage requirements. Do not
silently substitute a different dataset while retaining Waterbirds
claims.

Use the official train/validation/test split and preserve group
metadata. Record the exact dataset version, source, preprocessing, and
number of samples per split/group.

### Dataset fallback

If Waterbirds cannot be downloaded or used promptly, use CUB-200-2011
with its available annotations to build a smaller controlled experiment,
or create a synthetic background-correlation benchmark from
appropriately licensed foreground images and backgrounds. Document all
transformations and do not present a synthetic benchmark as Waterbirds.

Do not spend more than half a day blocked on dataset access. Implement
the dataset interface so the data source can be swapped without
rewriting the training/evaluation pipeline.

### Dataset integrity rules

-   Never use test examples for model selection or hyperparameter
    tuning.
-   Do not rebalance or relabel groups without documenting the change.
-   Keep train/validation/test boundaries intact.
-   Report per-group counts and metrics.
-   Store only dataset paths and metadata in the repository; do not
    commit datasets or model weights.
-   Follow the dataset's license and terms.

------------------------------------------------------------------------

## 5. Models and Baselines

### Baseline A: ResNet-18

Use a pretrained torchvision ResNet-18, replace the final classification
layer, and fine-tune it for the target classes. This is the primary
baseline because it is relatively lightweight and practical on a 6 GB
GPU.

### Baseline B: pretrained Vision Transformer (optional but desirable)

If time and memory allow, use a small or base pretrained ViT from
torchvision or timm. Begin by freezing most of the backbone and training
the classification head. Fine-tune more layers only if memory and time
permit.

Do not make a ViT comparison a blocker. A rigorous single-model
experiment is better than two incomplete model runs.

### Required baseline evaluation

Report: - Overall accuracy. - Accuracy for each predefined group. -
Worst-group accuracy. - Confusion matrix. - Per-group sample counts. -
Training duration and inference throughput, where practical.

Keep preprocessing, split usage, and evaluation code consistent across
models.

------------------------------------------------------------------------

## 6. Counterfactual Intervention Design

The project must implement at least two interventions and document their
assumptions. An intervention is an input perturbation; it is not
automatically a causal proof.

### Intervention 1: foreground preservation, background modification

Preserve the target object as well as possible while modifying the
background.

Possible implementation: - Obtain a foreground mask using an available
dataset mask or a pretrained segmentation model. - Cache masks so they
are not recomputed for every experiment. - Replace the background with a
background from another image or use a simple neutral/blurred
background. - Keep the original image, mask, transformed image, and
model predictions for inspection.

For the first version, prefer a reproducible method over photorealistic
synthesis. Avoid making the segmentation model itself the main project.

### Intervention 2: foreground suppression or masking

Mask, blur, or replace the foreground while leaving the background
largely intact. Measure whether the model's prediction remains confident
or changes.

This intervention is especially sensitive to artifacts. Use more than
one masking style if time allows, such as a neutral fill and a blurred
fill.

### Optional Intervention 3: matched background swap

Swap backgrounds between images while preserving foreground masks. When
possible, use swaps that create meaningful foreground/background
combinations without changing the target class.

This is more informative than arbitrary image corruption, but mask
errors and unnatural composites must be acknowledged.

### Segmentation strategy

Prefer existing masks or annotations if they are available and
compatible with the chosen dataset. Otherwise, use a pretrained
segmentation model, cache its output, and manually inspect a small
random sample.

Do not train a segmentation model from scratch. Do not assume masks are
accurate simply because a pretrained model generated them.

### Intervention quality checks

For a small curated set of examples: - Visualize original image, mask,
modified image, and predictions side by side. - Record obvious
segmentation failures. - Check that transformations do not accidentally
remove the object in foreground-preserving experiments. - Keep the same
transformation policy across compared models. - Report sensitivity to at
least one intervention parameter or masking style if possible.

------------------------------------------------------------------------

## 7. Metrics and Analysis

### Primary classification metrics

1.  **Overall accuracy:** useful but insufficient by itself.
2.  **Group accuracy:** accuracy for every predefined class/background
    group.
3.  **Worst-group accuracy:** minimum accuracy over the defined groups.
4.  **Balanced accuracy or macro-F1:** optional, depending on class
    balance.
5.  **Negative log-likelihood and calibration error:** optional if
    probability calibration is part of the analysis.

### Intervention metrics

For an image `x`, original prediction distribution `p(y|x)`, and
intervention `T(x)`, measure:

-   **True-class probability change:** `p(y_true|x) - p(y_true|T(x))`.
-   **Prediction flip rate:** fraction of examples whose predicted class
    changes after intervention.
-   **Confidence change:** difference in probability assigned to the
    original predicted class.
-   **Distribution divergence:** optionally use Jensen--Shannon
    divergence between original and intervened prediction distributions.

Define score directions consistently and document them. Do not combine
metrics into one opaque score unless the formula is justified and
ablated.

### Central analysis

Test whether intervention-derived scores predict failures on held-out
background-shifted groups. Use per-example analysis or model-level
comparisons where statistically appropriate. Include scatter plots and
correlation estimates when enough independent observations exist.

Avoid claiming a strong correlation from a tiny number of models. If
only two models are evaluated, emphasize per-example analysis,
group-level patterns, and qualitative inspection rather than
overinterpreting model-level correlations.

### Required plots/tables

-   Overall and worst-group accuracy comparison.
-   Accuracy by group.
-   Prediction change distribution for each intervention.
-   Example counterfactuals with before/after predictions.
-   Intervention score versus errors on shifted examples.
-   Ablation table for intervention method or mask style.
-   Runtime/memory notes if available.

### Statistical discipline

-   Fix random seeds where feasible.
-   Save the exact configuration for each run.
-   Report repeated runs when time allows; otherwise state that results
    are from a single seed.
-   Do not cherry-pick examples.
-   Separate exploratory analysis from final evaluation.
-   If a result is negative, report it honestly.

------------------------------------------------------------------------

## 8. Potential Differentiating Contribution

The initial candidate contribution is:

> A reproducible comparison of foreground/background counterfactual
> interventions, testing whether intervention-derived audit scores
> predict background-shift failures more reliably than a simple
> saliency-based diagnostic.

This is a hypothesis, not a guaranteed novelty claim.

Before implementation is finalized, conduct a focused literature review
covering: - Shortcut learning in deep vision models. - Waterbirds and
worst-group evaluation. - Counterfactual image explanations and feature
perturbation. - Saliency-based model auditing. - Background debiasing
and distribution-shift robustness. - Recent work from the last 2--3
years.

Create a related-work table with columns: paper/year, problem, method,
dataset, metric, limitation relevant to this project, and what this
project will test differently.

The contribution must be narrowed after reading the literature. If a
closely matching method already exists, do not hide it. Change the
question, focus on a reproducibility/benchmarking contribution, or
choose a different hypothesis.

Do not write "first ever," "novel architecture," or "state of the art"
unless a defensible literature review and evidence justify it.

------------------------------------------------------------------------

## 9. Hardware and Engineering Constraints

Target system: - Windows. - NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM. -
Ryzen 5 7640HS. - PyTorch with CUDA, where compatible with the installed
driver.

### Engineering rules

-   Check `torch.cuda.is_available()` and report the detected GPU at
    startup.
-   Use small batches (start at 16; reduce to 8 or 4 if needed).
-   Use mixed precision only when supported and validated.
-   Resize images consistently, starting around 224×224.
-   Cache expensive masks and transformed images where storage allows.
-   Use `num_workers` conservatively on Windows; keep a `0` fallback.
-   Keep model checkpoints and dataset files out of Git.
-   Make every long-running task resumable where practical.
-   Save training logs and machine-readable metrics.
-   Avoid downloading large models without checking disk and VRAM
    requirements.
-   Provide a CPU fallback for evaluation and smoke tests, even if
    slower.

### Dependency strategy

Use a clean virtual environment. Pin important dependency versions in
`requirements.txt` or a lock file. Avoid unnecessary dependencies. Use
torchvision/timm only where needed. Do not install multiple competing
packages for the same functionality without a reason.

------------------------------------------------------------------------

## 10. Suggested Repository Structure

``` text
counterfactual-vision-auditor/
├── README.md
├── requirements.txt
├── pyproject.toml                 # optional
├── configs/
│   ├── resnet18.yaml
│   └── vit.yaml                   # optional
├── data/
│   └── README.md                  # dataset instructions only; no data committed
├── src/
│   ├── cva/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── datasets.py
│   │   ├── models.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── interventions.py
│   │   ├── audit.py
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   └── utils.py
├── scripts/
│   ├── check_environment.py
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   ├── run_audit.py
│   └── generate_report.py
├── tests/
│   ├── test_dataset.py
│   ├── test_interventions.py
│   ├── test_metrics.py
│   └── test_smoke.py
├── notebooks/
│   └── exploratory_analysis.ipynb  # optional; core logic stays in src/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── reports/
└── docs/
    ├── related_work.md
    ├── experiment_plan.md
    └── limitations.md
```

Adapt the structure if needed, but keep data loading, model training,
interventions, metrics, and visualization separate. Do not put the
entire project in one notebook or one giant Python file.

------------------------------------------------------------------------

## 11. Required CLI Workflow

The exact commands can vary, but the README must provide a complete
end-to-end workflow similar to:

``` bash
python scripts/check_environment.py
python scripts/train_baseline.py --config configs/resnet18.yaml
python scripts/evaluate_baseline.py --checkpoint outputs/checkpoints/resnet18.pt
python scripts/run_audit.py --checkpoint outputs/checkpoints/resnet18.pt
python scripts/generate_report.py
```

Every command must either work as documented or the README must clearly
explain prerequisites and expected paths. Use explicit argument parsing
and helpful error messages.

A smoke-test mode should run on a tiny subset with minimal epochs so
that a new user can verify the pipeline without downloading or training
everything.

------------------------------------------------------------------------

## 12. Interface Expectations

The core project must work without a frontend.

If the experiment pipeline is complete early, build a lightweight
Streamlit interface with: - Upload/select an image. - Display original
and transformed images. - Display the model's original and intervened
predictions. - Show confidence/probability changes. - Explain which
intervention was applied. - Include a clear warning that intervention
effects are not proof of causality. - Show a few benchmark results from
saved outputs.

Do not allow the UI to consume the project timeline. The scientific
evaluation and reproducibility are more important than styling.

------------------------------------------------------------------------

## 13. Testing and Acceptance Criteria

The project is considered complete only when the following are
satisfied.

### Functional correctness

-   [ ] Environment check reports PyTorch, CUDA status, and GPU.
-   [ ] Dataset loader returns images, labels, and group metadata
    correctly.
-   [ ] A small smoke training run completes.
-   [ ] Checkpoints can be loaded for evaluation.
-   [ ] Metrics are computed correctly on a hand-checkable toy example.
-   [ ] Both intervention methods return valid images of the expected
    shape and range.
-   [ ] Audit results can be saved and reloaded.
-   [ ] A clean run can reproduce the documented workflow.

### Experimental quality

-   [ ] At least one baseline is trained and evaluated.
-   [ ] Per-group metrics and worst-group accuracy are reported.
-   [ ] At least two intervention methods are evaluated.
-   [ ] Results include quantitative tables and visual examples.
-   [ ] At least one ablation or sensitivity analysis is included.
-   [ ] Related work is documented with verifiable sources.
-   [ ] Limitations and possible confounders are explicitly discussed.

### Repository quality

-   [ ] No dataset, credentials, or large checkpoint files are
    committed.
-   [ ] Dependencies are documented.
-   [ ] Random seeds and configuration are saved.
-   [ ] Code has type/shape checks and useful errors where appropriate.
-   [ ] Tests cover key data and metric functions.
-   [ ] README includes setup, data access, training, evaluation,
    results, and limitations.
-   [ ] Results shown in the README are generated by the code and not
    fabricated.

### Scientific integrity

-   [ ] No invented metrics, fabricated results, or unsupported novelty
    claims.
-   [ ] A prediction change is not described as proof of causal
    reliance.
-   [ ] Masking/compositing artifacts are considered.
-   [ ] Failed or inconclusive experiments are reported honestly.
-   [ ] Dataset license and usage requirements are followed.

------------------------------------------------------------------------

## 14. Timeline

### Minimum viable version: 2 weeks

**Days 1--2** - Review related work. - Verify dataset access and GPU
environment. - Finalize the exact research question and experiment
protocol. - Implement data loading and a tiny smoke test.

**Days 3--5** - Train ResNet-18 baseline. - Compute overall, group-wise,
and worst-group metrics. - Save checkpoints and run configurations.

**Days 6--9** - Implement two interventions. - Visualize and inspect
mask/intervention quality. - Compute prediction sensitivity and
flip-rate metrics.

**Days 10--12** - Run the core comparison and one ablation. - Generate
tables and plots. - Investigate unexpected results.

**Days 13--14** - Write README and concise report. - Add tests and
reproduce the full workflow. - Capture demo screenshots.

### Stronger version: 3--4 weeks

Use the additional time for: - A second model architecture. - More
robust mask quality checks. - Repeated runs or confidence intervals. - A
better baseline comparison. - A lightweight demo interface. - A more
careful literature review and final report.

Prioritize in this order: **correctness → baseline → interventions →
evaluation → interpretation → presentation/UI**.

------------------------------------------------------------------------

## 15. Literature Review and References

The implementation agent must verify current bibliographic details, code
availability, dataset access, and licensing rather than inventing
citations. Start by investigating these established resources and
research directions:

1.  **WILDS / Waterbirds** --- benchmark for distribution shifts and
    group robustness. Paper: *WILDS: A Benchmark of in-the-Wild
    Distribution Shifts* (ICML 2021).\
    https://wilds.stanford.edu/

2.  **Shortcut learning in deep neural networks** --- Geirhos et al.,
    *Shortcut Learning in Deep Neural Networks*, Nature Machine
    Intelligence (2020).\
    https://doi.org/10.1038/s42256-020-00257-z

3.  **CUB-200-2011** --- fine-grained bird classification dataset with
    annotations.\
    https://www.vision.caltech.edu/datasets/cub_200_2011/

4.  **Captum** --- PyTorch model interpretability toolkit; useful for
    baseline attribution methods, not a substitute for the core
    experiments.\
    https://captum.ai/

5.  **Torchvision models** --- pretrained ResNet and ViT
    implementations.\
    https://pytorch.org/vision/stable/models.html

6.  **Recent literature search** --- search papers from CVPR, ICCV,
    ECCV, NeurIPS, ICLR, ICML, and IJCAI from the last 2--3 years for
    terms including:

    -   shortcut learning and background bias
    -   counterfactual explanations for image classifiers
    -   visual feature perturbation and model auditing
    -   worst-group robustness
    -   background debiasing
    -   spurious correlations in vision

These are starting points, not a claim that they are exhaustive or that
the proposed comparison is novel.

------------------------------------------------------------------------

## 16. Instructions for the Coding Agent (Antigravity)

Act as a senior ML engineer who values reproducibility and scientific
honesty. Build the project in the current workspace.

### Before coding

1.  Inspect the current workspace and avoid overwriting unrelated files.
2.  Verify current dataset access and relevant documentation.
3.  Conduct a concise literature scan and create `docs/related_work.md`.
4.  Write `docs/experiment_plan.md` specifying hypotheses, datasets,
    splits, metrics, baselines, interventions, and expected outputs.
5.  Check Python, PyTorch, CUDA, and GPU availability.
6.  Present a short implementation plan, then proceed without waiting
    for approval unless a genuine blocker requires a decision.

### During implementation

1.  Implement the smallest end-to-end working version first.
2.  Keep the code modular and testable.
3.  Use the available GPU efficiently but respect the 6 GB VRAM limit.
4.  Prefer simple, interpretable interventions over unnecessary model
    complexity.
5.  Cache expensive operations and make experiments resumable.
6.  Add logging, saved configurations, deterministic seeds where
    feasible, and machine-readable results.
7.  Run tests and smoke tests as each component is added.
8.  If the preferred dataset is unavailable, document the blocker and
    use the defined fallback without misrepresenting the dataset.
9.  Do not fabricate results, citations, benchmarks, or completion
    status.
10. Do not claim a research contribution is novel without checking
    relevant prior work.

### At completion

Provide: - A concise summary of the implementation. - Exact
environment/setup commands. - Exact commands to reproduce training,
evaluation, and audit runs. - Dataset download/access instructions. - A
list of tests actually run and their outcomes. - Generated figures and
metric files. - A table of actual experimental results. - Known bugs,
limitations, and unfinished items. - A short explanation of the
strongest defensible contribution. - A resume bullet based only on
verified completed work and actual results.

If a full experiment cannot fit within the deadline or hardware budget,
deliver a tested, reproducible smaller experiment rather than a large
incomplete system.

------------------------------------------------------------------------

## 17. Final Definition of Success

Success is not defined by achieving a predetermined accuracy or proving
the hypothesis.

Success means delivering a project that: - Runs end to end on the target
laptop. - Has a valid baseline and a defensible evaluation protocol. -
Implements and compares meaningful interventions. - Produces
reproducible, inspectable results. - Demonstrates awareness of related
literature and methodological limitations. - Can be explained clearly in
a 5-minute demonstration and defended in a technical discussion.

**Final principle:** Build a smaller project with rigorous experiments
and honest conclusions. Do not sacrifice scientific credibility for a
flashy interface or an exaggerated novelty claim.
