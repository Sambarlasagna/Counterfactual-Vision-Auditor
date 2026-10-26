# Experiment Plan

*To be finalized after dataset verification and literature review.*

## Research Question

> How reliably do different foreground/background interventions reveal background
> dependence in image classifiers, and do the resulting audit scores predict
> failures under background distribution shifts?

## Hypotheses

1. **H1:** Models trained with ERM on Waterbirds will show significantly lower accuracy
   on minority groups (mismatched bird/background) compared to majority groups.

2. **H2:** Background modification interventions will cause larger prediction changes
   for images where the model relies on background cues.

3. **H3:** Foreground suppression will reveal that some correct predictions are driven
   primarily by background, not the bird itself.

4. **H4:** Intervention-derived audit scores will correlate with per-group error rates,
   making them useful indicators of background dependence.

## Dataset

- **Primary:** Waterbirds (WILDS)
- **Fallback:** CUB-200-2011 with synthetic groups
- **Splits:** Standard train / val / test
- **Groups:** 4 groups (2 bird types × 2 background types)

## Models

- **Baseline A:** ResNet-18 (pretrained, fine-tuned)
- **Baseline B (stretch):** ViT-Small (pretrained, head-only then fine-tuned)

## Interventions

1. Background modification (gray fill, blur, cross-group swap)
2. Foreground suppression (gray mask, blur mask)
3. Background swap between groups (optional)

## Metrics

- Overall accuracy, group accuracy, worst-group accuracy
- True-class probability change, prediction flip rate, confidence change
- JSD between original and intervened prediction distributions (optional)

## Expected Outputs

- Trained model checkpoint(s)
- Per-group accuracy table
- Intervention metrics table
- Ablation: mask style comparison
- Scatter plot: audit score vs. group error rate
- Counterfactual visualization gallery
- Technical report (4–6 pages)
