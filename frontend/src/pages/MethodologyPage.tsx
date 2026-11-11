import React from 'react';
import { BookOpen, ShieldAlert, Cpu, CheckCircle } from 'lucide-react';

export const MethodologyPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#0A0D14] text-[#F3F6FC] py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto space-y-10">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#111827] border border-indigo-500/30 text-indigo-400 text-xs font-mono mb-3">
            <BookOpen className="w-3.5 h-3.5" />
            Auditing Methodology & Constraints
          </div>
          <h1 className="text-3xl font-bold tracking-tight">Theoretical Framework & Scientific Limits</h1>
          <p className="text-sm text-[#A7B1C2] mt-1">
            How Counterfactual Vision Auditor formulates visual shortcut auditing, how interventions are generated, and what these tests establish.
          </p>
        </div>

        {/* Section 1: The Core Problem */}
        <section className="p-6 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)] space-y-4">
          <h2 className="text-lg font-semibold text-[#F3F6FC]">1. Shortcut Learning in Deep Vision</h2>
          <p className="text-xs text-[#A7B1C2] leading-relaxed">
            Standard empirical risk minimization (ERM) optimizes a deep neural network to minimize average loss over a training distribution:
          </p>
          <div className="p-3 rounded-lg bg-[#0A0D14] font-mono text-xs text-indigo-300 border border-[rgba(148,163,184,0.1)]">
            min_θ  (1/N) ∑ L( f_θ(x_i), y_i )
          </div>
          <p className="text-xs text-[#A7B1C2] leading-relaxed">
            When training images possess spurious statistical correlations (such as 95% of waterbirds appearing against water bodies), the network exploits these background features as low-complexity shortcuts rather than learning invariant anatomical structures. Standard test accuracy conceals this failure because test sets preserve the same correlations.
          </p>
        </section>

        {/* Section 2: Intervention Pipeline */}
        <section className="p-6 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)] space-y-4">
          <h2 className="text-lg font-semibold text-[#F3F6FC]">2. Counterfactual Interventions via DeepLabV3</h2>
          <p className="text-xs text-[#A7B1C2] leading-relaxed">
            Unlike gradient-based saliency methods (e.g., Grad-CAM) which merely localize discriminative activations without testing causal dependency, CVA performs concrete input-level perturbations:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-mono">
            <div className="p-4 rounded-lg bg-[#0A0D14] border border-[rgba(148,163,184,0.1)] space-y-2">
              <span className="text-emerald-400 font-semibold block">Background Modification:</span>
              <p className="text-[11px] text-[#A7B1C2]">
                I_new = I ⊙ M + Background_fill ⊙ (1 - M)
              </p>
              <p className="text-[10px] text-[#718096]">
                Preserves foreground object pixels while neutralizing or blurring context.
              </p>
            </div>

            <div className="p-4 rounded-lg bg-[#0A0D14] border border-[rgba(148,163,184,0.1)] space-y-2">
              <span className="text-indigo-400 font-semibold block">Foreground Suppression:</span>
              <p className="text-[11px] text-[#A7B1C2]">
                I_new = Foreground_fill ⊙ M + I ⊙ (1 - M)
              </p>
              <p className="text-[10px] text-[#718096]">
                Suppresses the foreground bird while preserving background texture intact.
              </p>
            </div>
          </div>
        </section>

        {/* Section 3: Limitations & Honesty */}
        <section className="p-6 rounded-xl bg-amber-950/20 border border-amber-500/30 space-y-4">
          <h2 className="text-lg font-semibold text-amber-300 flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-amber-400" />
            3. Scientific Constraints & Known Limitations
          </h2>
          <ul className="text-xs text-[#A7B1C2] space-y-2.5 list-disc list-inside leading-relaxed">
            <li>
              <strong className="text-[#F3F6FC]">Sensitivity ≠ Causality:</strong> A prediction change under perturbation indicates prediction sensitivity to the altered region, but does not constitute formal mathematical proof of causal feature reliance.
            </li>
            <li>
              <strong className="text-[#F3F6FC]">Compositing Artifacts:</strong> Off-the-shelf segmentation masks (DeepLabV3) may imperfectly crop boundary feathers, introducing high-frequency visual edges that classifiers can react to as out-of-distribution noise.
            </li>
            <li>
              <strong className="text-[#F3F6FC]">Unnatural Visual Distributions:</strong> Replacing organic backgrounds with flat gray or Gaussian blur synthesizes images outside the natural training manifold.
            </li>
            <li>
              <strong className="text-[#F3F6FC]">Single-Seed Evaluation:</strong> Current baseline results derive from random seed 42 on an RTX 3050 Laptop GPU. Repeated seeds and architectural comparisons (ViT) are documented as future expansions.
            </li>
          </ul>
        </section>
      </div>
    </div>
  );
};

