# Related Work

## Literature Review Table

| Paper | Year | Problem | Method | Dataset | Metric | Limitation | What CVA Tests Differently |
|-------|------|---------|--------|---------|--------|------------|---------------------------|
| Geirhos et al., *Shortcut Learning in Deep Neural Networks*, Nature Machine Intelligence | 2020 | Survey of shortcut learning across deep learning | Taxonomy of shortcut types; analysis of texture bias, background bias, and other spurious cues | Multiple (ImageNet, stylized variants) | Classification accuracy under distribution shift | Broad survey; does not provide a standardized audit tool or per-example intervention scores | CVA implements concrete per-example interventions and measures prediction sensitivity, rather than surveying the phenomenon |
| Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization*, ICLR | 2020 | Worst-group accuracy under spurious correlations | Group DRO — reweights training loss to minimize worst-group error | Waterbirds, CelebA, MultiNLI | Worst-group accuracy | Requires group labels at training time; does not audit a fixed model post-hoc | CVA audits a fixed ERM-trained model without requiring group labels during training; focuses on post-hoc detection rather than training-time mitigation |
| Koh et al., *WILDS: A Benchmark of in-the-Wild Distribution Shifts*, ICML | 2021 | Standardized benchmark for distribution shift evaluation | Unified data loaders, evaluation protocols, and baselines across 10 datasets | Waterbirds, Camelyon17, FMoW, iWildCam, etc. | Overall accuracy, worst-group accuracy, OOD accuracy | Benchmark focus; does not provide intervention-based auditing tools | CVA uses WILDS/Waterbirds as a testbed but adds counterfactual interventions to explain *why* models fail on certain groups |
| Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV | 2017 | Visual explanations for CNN predictions | Gradient-weighted class activation maps highlighting discriminative regions | ImageNet, VQA, image captioning | Localization accuracy, human trust studies | Saliency maps show where the model looks but do not test what happens when that region is changed; can be noisy/misleading | CVA directly modifies image regions and measures prediction changes, providing stronger evidence than saliency alone |
| Goyal et al., *Counterfactual Visual Explanations*, ICML | 2019 | Generating counterfactual image explanations | Finds minimal image edits that change the model's prediction using a generative model | CUB-200, ImageNet | Human evaluation, edit fidelity | Requires a generative model; edits are not controlled foreground/background interventions | CVA uses explicit foreground/background decomposition with segmentation masks for controlled, reproducible interventions |
| Singla et al., *Salient ImageNet: How to Discover Spurious Features in Deep Learning?*, ICLR | 2022 | Discovering spurious features in trained classifiers | Core and spurious feature identification via human annotation + saliency disagreement | ImageNet | Feature-level accuracy, spurious feature prevalence | Requires large-scale human annotation; limited to ImageNet classes | CVA uses automated segmentation and group structure rather than manual annotation; focuses on background vs. foreground at the image level |
| Wah et al., *The Caltech-UCSD Birds-200-2011 Dataset*, Tech Report | 2011 | Fine-grained bird species classification | 200-class dataset with part annotations, bounding boxes, and attributes | CUB-200-2011 | Classification accuracy, part localization | No explicit background/foreground group structure for bias analysis | CVA uses CUB images (via Waterbirds) with added group structure to study background dependence specifically |
| Kirichenko et al., *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations*, ICLR | 2023 | Mitigating spurious correlations via last-layer retraining | Deep Feature Reweighting (DFR) — retrain only the last layer with balanced data | Waterbirds, CelebA, MultiNLI | Worst-group accuracy | Assumes access to a balanced validation set with group labels; mitigation focus, not auditing | CVA focuses on diagnosing the problem (measuring background dependence) rather than fixing it; results could inform whether retraining is needed |

## Key Research Directions

1. **Shortcut learning in deep vision models** — Models exploit spurious training-set correlations (textures, backgrounds, watermarks) that are unreliable under distribution shift (Geirhos et al., 2020).

2. **Waterbirds and worst-group evaluation** — The Waterbirds benchmark (Sagawa et al., 2020) is specifically designed to study background-bird correlations, with 4 groups defined by bird type × background type. Worst-group accuracy reveals failures hidden by aggregate metrics.

3. **Counterfactual image explanations and feature perturbation** — Directly modifying image regions (Goyal et al., 2019) provides stronger evidence of model reliance than passive saliency maps, though compositing artifacts are a confound.

4. **Saliency-based model auditing** — Methods like Grad-CAM (Selvaraju et al., 2017) highlight discriminative regions but do not causally test whether removing or changing those regions alters predictions. CVA bridges this gap.

5. **Background debiasing and distribution-shift robustness** — Training-time methods like Group DRO (Sagawa et al., 2020) and DFR (Kirichenko et al., 2023) mitigate spurious correlations but require group labels. CVA provides a complementary post-hoc auditing approach.

## CVA's Position in the Literature

CVA does **not** claim novelty in any single component — shortcut learning, worst-group evaluation, counterfactual explanations, and segmentation-based interventions are all established techniques. The contribution is a **reproducible, integrated audit pipeline** that:

1. Combines foreground/background interventions with group-structured evaluation
2. Measures multiple prediction sensitivity metrics (flip rate, probability change, JSD)
3. Tests whether intervention scores predict distribution-shift failures
4. Provides both quantitative metrics and visual examples for inspection

This positions CVA as a practical auditing tool rather than a novel algorithm.
