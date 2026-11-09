import React from 'react';
import { Cpu, Github, ExternalLink, Code2, Database } from 'lucide-react';

export const AboutPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-[#0A0D14] text-[#F3F6FC] py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto space-y-10">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#111827] border border-indigo-500/30 text-indigo-400 text-xs font-mono mb-3">
            <Cpu className="w-3.5 h-3.5" />
            Project Architecture
          </div>
          <h1 className="text-3xl font-bold tracking-tight">About Counterfactual Vision Auditor</h1>
          <p className="text-sm text-[#A7B1C2] mt-1">
            An open-source visual model reliability and shortcut auditing system.
          </p>
        </div>

        {/* Tech Stack Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="p-5 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)] space-y-2">
            <div className="flex items-center gap-2 text-indigo-400">
              <Code2 className="w-4 h-4" />
              <h3 className="font-semibold text-xs text-[#F3F6FC] uppercase tracking-wider">Frontend Client</h3>
            </div>
            <p className="text-xs text-[#A7B1C2] leading-relaxed">
              Built with React 19, TypeScript, Vite, and Tailwind CSS. Provides an interactive side-by-side comparison workspace with responsive dark-workstation styling.
            </p>
          </div>

          <div className="p-5 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)] space-y-2">
            <div className="flex items-center gap-2 text-emerald-400">
              <Cpu className="w-4 h-4" />
              <h3 className="font-semibold text-xs text-[#F3F6FC] uppercase tracking-wider">Inference Backend</h3>
            </div>
            <p className="text-xs text-[#A7B1C2] leading-relaxed">
              FastAPI service hosting PyTorch & torchvision models. Manages model registries, executes DeepLabV3 semantic segmentation with disk caching, and performs on-the-fly counterfactual interventions.
            </p>
          </div>
        </div>

        {/* Provenance */}
        <div className="p-6 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)] space-y-3">
          <h2 className="text-base font-semibold text-[#F3F6FC] flex items-center gap-2">
            <Database className="w-4 h-4 text-indigo-400" />
            Dataset & Model Provenance
          </h2>
          <div className="text-xs text-[#A7B1C2] space-y-2 leading-relaxed">
            <p>
              <strong>Dataset:</strong> Waterbirds benchmark (CUB-200-2011 bird species superimposed on Places backgrounds by Sagawa et al., 2020 / WILDS benchmark, Koh et al., 2021).
            </p>
            <p>
              <strong>Classifier:</strong> ResNet-18 initialized with ImageNet-1K pretrained weights and fine-tuned end-to-end on Waterbirds for 30 epochs using Adam, cosine annealing, and mixed precision.
            </p>
            <p>
              <strong>Source Repository:</strong>{' '}
              <a 
                href="https://github.com/Sambarlasagna/Counterfactual-Vision-Auditor" 
                target="_blank" 
                rel="noreferrer"
                className="text-indigo-400 hover:underline inline-flex items-center gap-1"
              >
                github.com/Sambarlasagna/Counterfactual-Vision-Auditor
                <ExternalLink className="w-3 h-3" />
              </a>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
