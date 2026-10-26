# Known Limitations

*To be updated as experiments progress.*

## Anticipated Limitations

1. **Segmentation artifacts:** Pretrained DeepLabV3 masks may be imperfect,
   creating compositing artifacts that could affect model predictions
   independently of the intended intervention.

2. **Single seed:** Unless time allows repeated runs, results are from a single
   random seed and may not reflect the full distribution of outcomes.

3. **Small model count:** Comparing only 1-2 models limits the generality of
   any model-level correlation claims.

4. **Unnatural composites:** Background replacement creates images that look
   unnatural. The model's response to unnaturalness itself (vs. background
   content) is a confound.

5. **Intervention ≠ Causation:** A prediction change after intervention suggests
   sensitivity to the modified region but does not prove causal reliance.

6. **Dataset-specific findings:** Results on Waterbirds may not generalize to
   other datasets or real-world deployment settings.

7. **Hardware constraints:** 6 GB VRAM limits batch sizes and model sizes,
   potentially affecting training dynamics.
