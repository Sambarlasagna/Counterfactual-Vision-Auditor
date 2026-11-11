import React from 'react';
import { Link } from 'react-router-dom';
import { Eye, ArrowRight, ShieldAlert, Cpu, Sparkles, Sliders, CheckCircle2 } from 'lucide-react';

export const LandingPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#0A0D14] flex flex-col">
      {/* Hero Section */}
      <section className="relative pt-20 pb-16 md:pt-28 md:pb-24 overflow-hidden border-b border-[rgba(148,163,184,0.16)]">
        {/* Subtle grid background glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[400px] bg-indigo-500/10 blur-[130px] pointer-events-none rounded-full" />

        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#111827] border border-indigo-500/30 text-indigo-400 text-xs font-mono mb-6">
            <Sparkles className="w-3.5 h-3.5" />
            Empirical Visual Shortcut Auditing
          </div>

          <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight text-[#F3F6FC] max-w-4xl mx-auto leading-tight">
            See what your vision model <br className="hidden sm:inline" />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-indigo-300 to-emerald-400">
              actually relies on.
            </span>
          </h1>

          <p className="mt-6 text-base sm:text-lg text-[#A7B1C2] max-w-2xl mx-auto leading-relaxed">
            Standard test metrics conceal shortcut learning. CVA uses semantic counterfactual interventions
            to stress-test image classifiers against spurious background context and measure true distribution-shift resilience.
          </p>

          {/* Action CTAs */}
          <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
            <Link
              to="/studio"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-[#818CF8] hover:bg-[#6366F1] text-[#0A0D14] font-semibold text-sm transition-all shadow-lg shadow-indigo-500/20"
            >
              Launch Studio
              <ArrowRight className="w-4 h-4" />
            </Link>

            <Link
              to="/benchmarks"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-[#111827] hover:bg-[#151D2B] border border-[rgba(148,163,184,0.16)] text-[#F3F6FC] font-medium text-sm transition-all"
            >
              View Real Experiment Data
            </Link>
          </div>

          {/* Visual Showcase Card */}
          <div className="mt-14 max-w-5xl mx-auto bg-[#111827] border border-[rgba(148,163,184,0.16)] rounded-xl p-4 sm:p-6 shadow-2xl">
            <div className="flex items-center justify-between pb-4 border-b border-[rgba(148,163,184,0.1)] text-xs font-mono text-[#718096]">
              <span className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400" />
                Intervention Pipeline Preview (Waterbirds ResNet-18)
              </span>
              <span>DeepLabV3 + ResNet-18</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4">
              {/* Card 1 */}
              <div className="bg-[#0A0D14] p-3 rounded-lg border border-[rgba(148,163,184,0.1)]">
                <span className="text-[11px] font-mono text-[#A7B1C2] block mb-2">1. Original Image</span>
                <div className="aspect-square rounded overflow-hidden bg-black/40 relative">
                  <img src="/samples/sample_group_1.jpg" alt="Original" className="w-full h-full object-cover" />
                  <div className="absolute bottom-2 left-2 right-2 bg-[#111827]/90 backdrop-blur p-2 rounded text-[11px] border border-[rgba(148,163,184,0.16)]">
                    <p className="font-semibold text-[#F3F6FC]">Predicted: Waterbird (78.4%)</p>
                    <p className="text-[#FB7185] text-[10px]">Ground Truth: Landbird</p>
                  </div>
                </div>
              </div>

              {/* Card 2 */}
              <div className="bg-[#0A0D14] p-3 rounded-lg border border-[rgba(148,163,184,0.1)]">
                <span className="text-[11px] font-mono text-[#A7B1C2] block mb-2">2. DeepLabV3 Masking</span>
                <div className="aspect-square rounded overflow-hidden bg-[#151D2B] flex items-center justify-center p-4">
                  <div className="w-3/4 h-3/4 rounded-full border-2 border-indigo-400/40 bg-indigo-500/10 flex items-center justify-center text-center p-4">
                    <span className="text-xs font-mono text-indigo-300">
                      Foreground Bird Isolated<br/>
                      <span className="text-[10px] text-[#718096]">Context Separated</span>
                    </span>
                  </div>
                </div>
              </div>

              {/* Card 3 */}
              <div className="bg-[#0A0D14] p-3 rounded-lg border border-indigo-500/30 bg-gradient-to-b from-indigo-950/20 to-transparent">
                <span className="text-[11px] font-mono text-indigo-400 block mb-2">3. Counterfactual Intervention</span>
                <div className="aspect-square rounded overflow-hidden bg-black/40 relative flex flex-col justify-between p-4 border border-[rgba(148,163,184,0.1)]">
                  <div className="space-y-1">
                    <p className="text-xs font-mono text-emerald-400">Background Modified (Gray/Blur)</p>
                    <p className="text-[11px] text-[#A7B1C2]">Water context suppressed</p>
                  </div>
                  <div className="bg-[#111827]/90 backdrop-blur p-2.5 rounded text-[11px] border border-emerald-500/30">
                    <p className="font-semibold text-emerald-400">Prediction Flipped: Landbird (92.1%)</p>
                    <p className="text-[#A7B1C2] text-[10px]">Confirms model relied on water background</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3 Core Pillars Section */}
      <section className="py-16 md:py-24 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-2xl sm:text-3xl font-bold text-[#F3F6FC]">Rigorous Model Verification</h2>
          <p className="text-sm text-[#A7B1C2] mt-2 max-w-xl mx-auto">
            Moving beyond passive saliency heatmaps to active, mathematical perturbation testing.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)]">
            <div className="w-10 h-10 rounded-lg bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400 mb-4">
              <Sliders className="w-5 h-5" />
            </div>
            <h3 className="text-base font-semibold text-[#F3F6FC]">Pixel-Level Interventions</h3>
            <p className="text-xs text-[#A7B1C2] mt-2 leading-relaxed">
              We apply controlled background replacement, blur, and foreground suppression to determine whether the model is classifying the object or merely memorizing the scenery.
            </p>
          </div>

          <div className="p-6 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)]">
            <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 mb-4">
              <CheckCircle2 className="w-5 h-5" />
            </div>
            <h3 className="text-base font-semibold text-[#F3F6FC]">Worst-Group Accuracy (WGA)</h3>
            <p className="text-xs text-[#A7B1C2] mt-2 leading-relaxed">
              Aggregate test accuracy conceals catastrophic failure. CVA decomposes performance into subpopulation groups to compute true minimum worst-group resilience.
            </p>
          </div>

          <div className="p-6 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)]">
            <div className="w-10 h-10 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400 mb-4">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <h3 className="text-base font-semibold text-[#F3F6FC]">Explicit Scientific Limits</h3>
            <p className="text-xs text-[#A7B1C2] mt-2 leading-relaxed">
              Interventions reveal prediction sensitivity, not absolute causal proof. Our reports transparently communicate segmentation artifacts and out-of-distribution confounds.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-auto border-t border-[rgba(148,163,184,0.16)] bg-[#0A0D14] py-8">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-[#718096]">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-[#F3F6FC]">CVA Visual Reliability Lab</span>
            <span>&bull;</span>
            <span>Academic Computer Vision Project</span>
          </div>
          <div className="flex items-center gap-6">
            <Link to="/studio" className="hover:text-[#F3F6FC]">Studio</Link>
            <Link to="/benchmarks" className="hover:text-[#F3F6FC]">Benchmarks</Link>
            <Link to="/methodology" className="hover:text-[#F3F6FC]">Methodology</Link>
            <Link to="/about" className="hover:text-[#F3F6FC]">About</Link>
          </div>
        </div>
      </footer>
    </div>
  );
};

