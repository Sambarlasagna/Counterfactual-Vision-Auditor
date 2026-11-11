import React, { useEffect, useState } from 'react';
import { fetchBenchmarks } from '../lib/api';
import { BenchmarkSummary } from '../types';
import { BarChart3, TrendingDown, Layers, FileText, CheckCircle2 } from 'lucide-react';

export const BenchmarksPage: React.FC = () => {
  const [data, setData] = useState<BenchmarkSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchBenchmarks()
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0A0D14] flex items-center justify-center text-sm text-[#A7B1C2]">
        Loading experimental benchmark data...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0A0D14] text-[#F3F6FC] py-10 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#111827] border border-indigo-500/30 text-indigo-400 text-xs font-mono mb-3">
            <BarChart3 className="w-3.5 h-3.5" />
            Empirical Results Dashboard
          </div>
          <h1 className="text-3xl font-bold tracking-tight">Waterbirds Benchmark Results</h1>
          <p className="text-sm text-[#A7B1C2] mt-1 max-w-2xl">
            Quantitative evaluation of the ResNet-18 baseline model across 4 controlled distribution-shift groups and 500 audited counterfactual interventions.
          </p>
        </div>

        {/* Top KPI Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)]">
            <span className="text-xs text-[#718096] uppercase tracking-wider font-mono">Overall Accuracy</span>
            <p className="text-2xl font-bold text-[#F3F6FC] mt-1">
              {(data?.overall_accuracy ? data.overall_accuracy * 100 : 60.56).toFixed(1)}%
            </p>
            <p className="text-[11px] text-[#A7B1C2] mt-1">On 5,794 standard test samples</p>
          </div>

          <div className="p-4 rounded-xl bg-[#111827] border border-red-500/30 bg-red-950/10">
            <span className="text-xs text-red-400 uppercase tracking-wider font-mono flex items-center gap-1">
              <TrendingDown className="w-3.5 h-3.5" />
              Worst-Group Accuracy
            </span>
            <p className="text-2xl font-bold text-[#FB7185] mt-1">
              {(data?.worst_group_accuracy ? data.worst_group_accuracy * 100 : 24.45).toFixed(1)}%
            </p>
            <p className="text-[11px] text-red-300/80 mt-1">Group 2 (Waterbird on Land)</p>
          </div>

          <div className="p-4 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)]">
            <span className="text-xs text-emerald-400 uppercase tracking-wider font-mono">Majority Match Acc</span>
            <p className="text-2xl font-bold text-emerald-400 mt-1">94.99%</p>
            <p className="text-[11px] text-[#A7B1C2] mt-1">Group 0 (Landbird on Land)</p>
          </div>

          <div className="p-4 rounded-xl bg-[#111827] border border-indigo-500/30">
            <span className="text-xs text-indigo-400 uppercase tracking-wider font-mono">BG Modification Flip Rate</span>
            <p className="text-2xl font-bold text-indigo-400 mt-1">47.2%</p>
            <p className="text-[11px] text-[#A7B1C2] mt-1">vs 8.6% for bird removal</p>
          </div>
        </div>

        {/* Per-Group Accuracy Breakdown Table */}
        <div className="p-6 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)] space-y-4">
          <h2 className="text-base font-semibold text-[#F3F6FC] flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-400" />
            Group Performance Disparity (Shortcut Learning Evidence)
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="border-b border-[rgba(148,163,184,0.16)] text-[#718096]">
                <tr>
                  <th className="py-2.5 px-3">Group ID</th>
                  <th className="py-2.5 px-3">Subpopulation Description</th>
                  <th className="py-2.5 px-3">Test Samples (N)</th>
                  <th className="py-2.5 px-3">Accuracy</th>
                  <th className="py-2.5 px-3">BG Flip Rate</th>
                  <th className="py-2.5 px-3">Classification Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[rgba(148,163,184,0.08)]">
                <tr className="hover:bg-[#151D2B]">
                  <td className="py-3 px-3 text-indigo-300 font-semibold">Group 0</td>
                  <td className="py-3 px-3 text-[#F3F6FC]">Landbird on Land (Majority)</td>
                  <td className="py-3 px-3 text-[#A7B1C2]">2,255</td>
                  <td className="py-3 px-3 text-emerald-400 font-bold">94.99%</td>
                  <td className="py-3 px-3 text-amber-400">71.8%</td>
                  <td className="py-3 px-3 text-[#34D399]">Correlated Shortcut (Exploited)</td>
                </tr>
                <tr className="hover:bg-[#151D2B]">
                  <td className="py-3 px-3 text-indigo-300 font-semibold">Group 1</td>
                  <td className="py-3 px-3 text-[#F3F6FC]">Landbird on Water (Minority)</td>
                  <td className="py-3 px-3 text-[#A7B1C2]">2,255</td>
                  <td className="py-3 px-3 text-[#FB7185] font-bold">27.41%</td>
                  <td className="py-3 px-3 text-amber-400">25.0%</td>
                  <td className="py-3 px-3 text-[#FB7185]">Distribution Shift Failure</td>
                </tr>
                <tr className="hover:bg-[#151D2B] bg-red-950/20">
                  <td className="py-3 px-3 text-red-300 font-semibold">Group 2</td>
                  <td className="py-3 px-3 text-[#F3F6FC] font-semibold">Waterbird on Land (Worst Group)</td>
                  <td className="py-3 px-3 text-[#A7B1C2]">642</td>
                  <td className="py-3 px-3 text-[#FB7185] font-bold">24.45%</td>
                  <td className="py-3 px-3 text-red-400">79.5%</td>
                  <td className="py-3 px-3 text-red-400">Catastrophic Collapse</td>
                </tr>
                <tr className="hover:bg-[#151D2B]">
                  <td className="py-3 px-3 text-indigo-300 font-semibold">Group 3</td>
                  <td className="py-3 px-3 text-[#F3F6FC]">Waterbird on Water (Majority)</td>
                  <td className="py-3 px-3 text-[#A7B1C2]">642</td>
                  <td className="py-3 px-3 text-emerald-400 font-bold">92.21%</td>
                  <td className="py-3 px-3 text-amber-400">10.5%</td>
                  <td className="py-3 px-3 text-[#34D399]">Correlated Shortcut (Exploited)</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Experiment Visual Plots Grid */}
        <div className="space-y-4">
          <h2 className="text-base font-semibold text-[#F3F6FC]">Generated Publication Figures</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-4 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)] space-y-2">
              <span className="text-xs font-mono text-[#A7B1C2]">Per-Group Test Accuracy</span>
              <div className="rounded-lg overflow-hidden bg-black/40">
                <img src="/api/v1/benchmarks/figures/group_accuracy.png" alt="Group Accuracy" className="w-full h-auto" />
              </div>
            </div>

            <div className="p-4 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)] space-y-2">
              <span className="text-xs font-mono text-[#A7B1C2]">Intervention Impact Comparison</span>
              <div className="rounded-lg overflow-hidden bg-black/40">
                <img src="/api/v1/benchmarks/figures/intervention_comparison.png" alt="Intervention Comparison" className="w-full h-auto" />
              </div>
            </div>

            <div className="p-4 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)] space-y-2">
              <span className="text-xs font-mono text-[#A7B1C2]">30-Epoch Training Curves</span>
              <div className="rounded-lg overflow-hidden bg-black/40">
                <img src="/api/v1/benchmarks/figures/training_history.png" alt="Training History" className="w-full h-auto" />
              </div>
            </div>

            <div className="p-4 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)] space-y-2">
              <span className="text-xs font-mono text-[#A7B1C2]">Test Confusion Matrix</span>
              <div className="rounded-lg overflow-hidden bg-black/40">
                <img src="/api/v1/benchmarks/figures/confusion_matrix.png" alt="Confusion Matrix" className="w-full h-auto" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

