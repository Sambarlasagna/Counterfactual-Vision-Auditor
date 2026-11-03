# Counterfactual Vision Auditor -- Experiment Summary

Generated automatically from saved experiment results.

## Classification Metrics

- **Overall Accuracy:** 0.6056265101829479
- **Balanced Accuracy:** 0.5976533628972653
- **Worst-Group Accuracy:** 0.24454828660436137
- **Worst Group ID:** 2

### Per-Group Accuracy

| Group | Accuracy | Count |
|-------|----------|-------|
| 0 | 0.9499 | 2255 |
| 1 | 0.2741 | 2255 |
| 2 | 0.2445 | 642 |
| 3 | 0.9221 | 642 |

## Audit Results

### background_modification__gray

- **Flip Rate:** 0.472
- **Mean Prob Change:** 0.1282818764448166
- **Mean JSD:** 0.1565941721200943

### background_modification__blur

- **Flip Rate:** 0.464
- **Mean Prob Change:** 0.09889461100101471
- **Mean JSD:** 0.14266164600849152

### foreground_suppression__gray

- **Flip Rate:** 0.102
- **Mean Prob Change:** 0.0496382899582386
- **Mean JSD:** 0.026657287031412125

### foreground_suppression__blur

- **Flip Rate:** 0.086
- **Mean Prob Change:** 0.030964476987719536
- **Mean JSD:** 0.01994476094841957

## Generated Figures

- `outputs\figures\group_accuracy.png`
- `outputs\figures\confusion_matrix.png`
- `outputs\figures\training_history.png`
- `outputs\figures\intervention_comparison.png`
- `outputs\figures\flip_rate_background_modification__gray.png`
- `outputs\figures\flip_rate_background_modification__blur.png`
- `outputs\figures\flip_rate_foreground_suppression__gray.png`
- `outputs\figures\flip_rate_foreground_suppression__blur.png`
- `outputs\figures\audit_vs_error_background_modification__gray.png`
- `outputs\figures\audit_vs_error_background_modification__blur.png`
- `outputs\figures\audit_vs_error_foreground_suppression__gray.png`
- `outputs\figures\audit_vs_error_foreground_suppression__blur.png`
