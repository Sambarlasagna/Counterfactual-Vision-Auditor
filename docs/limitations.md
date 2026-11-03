# Known Limitations

## Confirmed Limitations (Empirically Observed)

### 1. Segmentation Artifacts

The pretrained DeepLabV3-ResNet50 was used for foreground/background segmentation. While it performs reasonably on many bird images, it is not specifically trained for birds and produces imperfect masks. Segmentation errors can:
- Include background elements in the "foreground" mask (e.g., branches near the bird)
- Miss parts of the bird (e.g., thin legs, tail feathers)
- Create visible compositing artifacts at mask boundaries

These artifacts are a confound: the model may respond to the artifact itself rather than to the background change. We did not systematically quantify segmentation quality, though a manual spot-check of cached masks showed generally reasonable results for the Waterbirds images (which tend to have relatively distinct foreground/background separation).

### 2. Single Seed, Single Model

All results are from a single random seed (42) and a single model architecture (ResNet-18). Observations include:
- Worst-group accuracy was volatile during training (ranging from 3% to 29% across epochs)
- Different seeds could produce somewhat different final group accuracies
- ResNet-18's behavior may not generalize to other architectures (e.g., ViT, larger ResNets)

Without repeated runs, we cannot estimate the variance of our results.

### 3. Unnatural Composites

Background replacement (gray fill, blur fill) creates images that are visually unnatural. The model's response may partly reflect sensitivity to unnaturalness rather than to the specific background content. This is a fundamental limitation of mask-based interventions:
- Gray-filled backgrounds are never seen during training
- Blurred backgrounds create an out-of-distribution artifact

The consistency between gray and blur results (47.2% vs. 46.4% flip rate) provides some evidence that the specific fill method matters less than the removal of background information, but this confound cannot be fully eliminated without more sophisticated synthesis.

### 4. Intervention ≠ Causation

A prediction change after intervention demonstrates **sensitivity** to the modified region but does not prove **causal reliance** in the strong sense. Specifically:
- Modifying the background also changes global image statistics (color distribution, spatial frequency content)
- The model may respond to these statistical changes rather than to semantic background content
- Compositing artifacts introduce additional confounds

We describe our results in terms of "sensitivity" and "prediction changes" rather than "causal reliance."

### 5. Dataset-Specific Findings

Results are specific to the Waterbirds benchmark, which has strong, deliberate spurious correlations (95% of landbirds appear on land backgrounds). This extreme correlation makes background dependence easy to detect. In real-world settings:
- Spurious correlations may be subtler
- Groups may not be well-defined or labeled
- Multiple spurious features may interact

Our findings demonstrate that the audit pipeline *works on a benchmark designed for this purpose* but should not be assumed to transfer to all deployment scenarios.

### 6. Audit Sample Size

The counterfactual audit was run on 500 of the 5,794 test samples (8.6%) due to time constraints. While results appear stable and per-group patterns are clear, a full audit on all test samples would provide:
- More precise per-group metrics
- Better representation of edge cases
- Stronger statistical power for per-group comparisons

### 7. Hardware Constraints

The 6 GB VRAM limit of the RTX 3050 Laptop GPU constrained:
- Batch sizes (16 for training, 32 for evaluation)
- Model choices (ResNet-18 is small enough; larger models may not fit alongside the segmentation model)
- The feasibility of running the segmentation model + classifier simultaneously during audit

Training required ~27 minutes; the audit required ~86 seconds for 500 samples.

### 8. Foreground Suppression Ceiling Effect

Foreground suppression flip rates were low (8-10%) across all groups. This could mean:
- The model genuinely does not rely on the foreground (likely for majority groups)
- The DeepLabV3 masks do not fully suppress the bird (some foreground texture may leak through)
- The model has learned to use background information so strongly that even perfect foreground removal would show low flip rates

The first and third explanations are consistent; the second would underestimate foreground suppression's true effect.

## Limitations NOT Observed

- **GPU out-of-memory:** Training with batch size 16 and AMP was stable throughout 30 epochs.
- **Dataset access issues:** Waterbirds was successfully downloaded and extracted without fallback to CUB-200-2011.
- **Reproducibility failures:** The pipeline ran end-to-end without manual intervention.
