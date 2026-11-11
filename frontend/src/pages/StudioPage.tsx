import React, { useState, useEffect } from 'react';
import { 
  Upload, Sparkles, RefreshCw, AlertTriangle, Download, 
  Layers, CheckCircle, ChevronRight, Eye, Sliders, Info
} from 'lucide-react';
import { runAudit, fetchModels } from '../lib/api';
import { AuditResponse, ModelInfo, InterventionResult } from '../types';

interface SampleItem {
  name: string;
  url: string;
  description: string;
  groundTruth: string;
  expectedClassIndex: number;
}

const SAMPLES: SampleItem[] = [
  {
    name: 'Landbird on Land',
    url: '/samples/sample_group_0.jpg',
    description: 'Majority training group (Match). Model is typically confident.',
    groundTruth: 'Landbird',
    expectedClassIndex: 0,
  },
  {
    name: 'Landbird on Water',
    url: '/samples/sample_group_1.jpg',
    description: 'Minority group (Mismatch). Background often tricks model into predicting Waterbird.',
    groundTruth: 'Landbird',
    expectedClassIndex: 0,
  },
  {
    name: 'Waterbird on Land',
    url: '/samples/sample_group_2.jpg',
    description: 'Severe minority group (Worst-group 24.5% accuracy). Often misclassified as Landbird.',
    groundTruth: 'Waterbird',
    expectedClassIndex: 1,
  },
  {
    name: 'Waterbird on Water',
    url: '/samples/sample_group_3.jpg',
    description: 'Majority training group (Match). Natural habitat.',
    groundTruth: 'Waterbird',
    expectedClassIndex: 1,
  },
];

export const StudioPage: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('waterbirds-resnet18');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedSample, setSelectedSample] = useState<SampleItem | null>(null);
  
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [auditResult, setAuditResult] = useState<AuditResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Intervention viewer state
  const [activeTab, setActiveTab] = useState<'comparison' | 'mask' | 'metrics'>('comparison');
  const [activeInterventionIdx, setActiveInterventionIdx] = useState<number>(0);

  useEffect(() => {
    fetchModels().then((res) => {
      setModels(res.models);
      if (res.models.length > 0) {
        setSelectedModel(res.models[0].id);
      }
    }).catch(() => {});
  }, []);

  const handleSelectSample = async (sample: SampleItem) => {
    setSelectedSample(sample);
    setErrorMsg(null);
    setAuditResult(null);
    try {
      const response = await fetch(sample.url);
      const blob = await response.blob();
      const file = new File([blob], sample.url.split('/').pop() || 'sample.jpg', { type: blob.type });
      setSelectedFile(file);
      setPreviewUrl(sample.url);
    } catch (e) {
      setErrorMsg('Failed to load sample image.');
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedSample(null);
      setErrorMsg(null);
      setAuditResult(null);
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleRunAudit = async () => {
    if (!selectedFile) {
      setErrorMsg('Please select or upload an image first.');
      return;
    }

    setIsLoading(true);
    setErrorMsg(null);

    try {
      const result = await runAudit(
        selectedFile,
        selectedModel,
        [
          { type: 'background_modification', method: 'gray' },
          { type: 'background_modification', method: 'blur' },
          { type: 'foreground_suppression', method: 'gray' },
          { type: 'foreground_suppression', method: 'blur' },
        ],
        selectedSample?.expectedClassIndex
      );
      setAuditResult(result);
      setActiveInterventionIdx(0);
    } catch (err: any) {
      setErrorMsg(err.message || 'Audit execution failed. Please ensure the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const downloadReportJson = () => {
    if (!auditResult) return;
    const blob = new Blob([JSON.stringify(auditResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${auditResult.audit_id}_report.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const currentIntervention: InterventionResult | undefined = auditResult?.interventions[activeInterventionIdx];

  return (
    <div className="min-h-screen bg-[#0A0D14] flex flex-col">
      {/* Studio Toolbar Header */}
      <div className="border-b border-[rgba(148,163,184,0.16)] bg-[#111827]/60 px-4 sm:px-6 py-3">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <h1 className="text-base font-semibold text-[#F3F6FC] flex items-center gap-2">
              <Sliders className="w-4 h-4 text-indigo-400" />
              Auditing Workspace
            </h1>
            <span className="text-xs text-[#718096]">|</span>
            <span className="text-xs text-[#A7B1C2] font-mono">
              Model: <span className="text-indigo-300">Waterbirds ResNet-18 (ImageNet Pretrained)</span>
            </span>
          </div>

          <div className="flex items-center gap-2">
            {auditResult && (
              <button
                onClick={downloadReportJson}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-[#151D2B] hover:bg-[#1B2638] text-xs text-[#A7B1C2] hover:text-[#F3F6FC] border border-[rgba(148,163,184,0.16)] transition-all"
              >
                <Download className="w-3.5 h-3.5" />
                Export JSON Report
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main Studio Grid */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 w-full flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Image / Sample Selection (4 cols) */}
        <div className="lg:col-span-4 space-y-6">
          {/* Sample selector */}
          <div className="p-4 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)]">
            <h2 className="text-xs font-semibold text-[#F3F6FC] uppercase tracking-wider mb-3 flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
              1. Choose a Test Sample
            </h2>
            <div className="grid grid-cols-2 gap-2">
              {SAMPLES.map((s) => (
                <button
                  key={s.name}
                  onClick={() => handleSelectSample(s)}
                  className={`p-2 rounded-lg text-left text-xs border transition-all ${
                    selectedSample?.name === s.name
                      ? 'bg-indigo-950/40 border-indigo-500/60 text-[#F3F6FC]'
                      : 'bg-[#151D2B] border-[rgba(148,163,184,0.1)] text-[#A7B1C2] hover:bg-[#1B2638]'
                  }`}
                >
                  <div className="aspect-square rounded overflow-hidden bg-black/40 mb-2">
                    <img src={s.url} alt={s.name} className="w-full h-full object-cover" />
                  </div>
                  <p className="font-semibold text-[11px] truncate">{s.name}</p>
                  <p className="text-[10px] text-[#718096] truncate">{s.groundTruth}</p>
                </button>
              ))}
            </div>

            {/* Custom File Upload */}
            <div className="mt-4 pt-4 border-t border-[rgba(148,163,184,0.1)]">
              <label className="block text-[11px] font-medium text-[#A7B1C2] mb-2">Or Upload Custom Image</label>
              <label className="flex flex-col items-center justify-center p-4 rounded-lg border-2 border-dashed border-[rgba(148,163,184,0.2)] hover:border-indigo-500/50 bg-[#0A0D14]/60 cursor-pointer transition-all">
                <Upload className="w-5 h-5 text-[#718096] mb-1" />
                <span className="text-[11px] text-[#A7B1C2]">Click or drag & drop</span>
                <span className="text-[9px] text-[#718096]">JPEG, PNG, WEBP (Max 10MB)</span>
                <input type="file" accept="image/*" onChange={handleFileUpload} className="hidden" />
              </label>
            </div>

            {/* Run Button */}
            <button
              onClick={handleRunAudit}
              disabled={!selectedFile || isLoading}
              className={`w-full mt-4 py-2.5 px-4 rounded-lg font-medium text-xs flex items-center justify-center gap-2 transition-all shadow-md ${
                !selectedFile || isLoading
                  ? 'bg-[#151D2B] text-[#718096] cursor-not-allowed border border-[rgba(148,163,184,0.1)]'
                  : 'bg-[#818CF8] hover:bg-[#6366F1] text-[#0A0D14]'
              }`}
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  Running Interventions...
                </>
              ) : (
                <>
                  <Eye className="w-3.5 h-3.5" />
                  Execute Counterfactual Audit
                </>
              )}
            </button>
          </div>

          {/* Model Card & Metadata */}
          <div className="p-4 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)] text-xs space-y-2">
            <h3 className="font-semibold text-[#F3F6FC] text-xs flex items-center gap-2">
              <Info className="w-3.5 h-3.5 text-indigo-400" />
              Target Model Spec
            </h3>
            <div className="space-y-1.5 text-[11px] text-[#A7B1C2] font-mono">
              <div className="flex justify-between py-1 border-b border-[rgba(148,163,184,0.08)]">
                <span>Architecture</span>
                <span className="text-[#F3F6FC]">ResNet-18</span>
              </div>
              <div className="flex justify-between py-1 border-b border-[rgba(148,163,184,0.08)]">
                <span>Classes</span>
                <span className="text-[#F3F6FC]">2 (Landbird / Waterbird)</span>
              </div>
              <div className="flex justify-between py-1 border-b border-[rgba(148,163,184,0.08)]">
                <span>Worst-Group Acc</span>
                <span className="text-[#FB7185] font-semibold">24.45% (G2)</span>
              </div>
              <div className="flex justify-between py-1">
                <span>Segmentation</span>
                <span className="text-emerald-400">DeepLabV3-ResNet50</span>
              </div>
            </div>
          </div>
        </div>

        {/* Center / Right Column: Comparison & Results (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          {errorMsg && (
            <div className="p-4 rounded-xl bg-red-950/30 border border-red-500/30 text-red-200 text-xs flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">Audit Execution Error</p>
                <p className="text-[11px] text-red-300/80 mt-0.5">{errorMsg}</p>
              </div>
            </div>
          )}

          {!auditResult && !isLoading && (
            <div className="h-96 rounded-xl bg-[#111827] border border-dashed border-[rgba(148,163,184,0.16)] flex flex-col items-center justify-center p-8 text-center">
              <div className="w-12 h-12 rounded-xl bg-[#151D2B] border border-[rgba(148,163,184,0.16)] flex items-center justify-center text-indigo-400 mb-3">
                <Sliders className="w-6 h-6" />
              </div>
              <h3 className="text-sm font-semibold text-[#F3F6FC]">No Active Audit</h3>
              <p className="text-xs text-[#718096] max-w-sm mt-1">
                Select one of the 4 benchmark samples on the left or upload your own image, then click "Execute Counterfactual Audit".
              </p>
            </div>
          )}

          {isLoading && (
            <div className="h-96 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)] flex flex-col items-center justify-center p-8 text-center">
              <RefreshCw className="w-8 h-8 text-indigo-400 animate-spin mb-3" />
              <h3 className="text-sm font-semibold text-[#F3F6FC]">Synthesizing Counterfactuals...</h3>
              <p className="text-xs text-[#718096] max-w-sm mt-1">
                1. Generating DeepLabV3 foreground mask<br/>
                2. Applying background/foreground interventions<br/>
                3. Running PyTorch model re-inference
              </p>
            </div>
          )}

          {auditResult && !isLoading && (
            <div className="space-y-6">
              {/* Baseline vs Intervention Comparison Card */}
              <div className="p-5 rounded-xl bg-[#111827] border border-[rgba(148,163,184,0.16)]">
                {/* Intervention Tabs */}
                <div className="flex items-center justify-between pb-4 border-b border-[rgba(148,163,184,0.1)] mb-4">
                  <div className="flex items-center gap-1.5 overflow-x-auto">
                    {auditResult.interventions.map((item, idx) => (
                      <button
                        key={`${item.intervention_type}_${item.method}`}
                        onClick={() => setActiveInterventionIdx(idx)}
                        className={`px-3 py-1.5 rounded-md text-xs font-mono transition-all flex items-center gap-1.5 ${
                          activeInterventionIdx === idx
                            ? 'bg-[#818CF8] text-[#0A0D14] font-semibold'
                            : 'bg-[#151D2B] text-[#A7B1C2] hover:bg-[#1B2638] border border-[rgba(148,163,184,0.1)]'
                        }`}
                      >
                        <span>{item.intervention_type === 'background_modification' ? 'BG Mod' : 'FG Supp'}</span>
                        <span className="opacity-75">({item.method})</span>
                        {item.prediction_flipped && (
                          <span className="w-1.5 h-1.5 rounded-full bg-red-400" title="Prediction Flipped" />
                        )}
                      </button>
                    ))}
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setActiveTab('comparison')}
                      className={`text-xs px-2.5 py-1 rounded ${activeTab === 'comparison' ? 'text-indigo-400 bg-indigo-950/50' : 'text-[#718096]'}`}
                    >
                      Side-by-Side
                    </button>
                    <button
                      onClick={() => setActiveTab('mask')}
                      className={`text-xs px-2.5 py-1 rounded ${activeTab === 'mask' ? 'text-indigo-400 bg-indigo-950/50' : 'text-[#718096]'}`}
                    >
                      Mask View
                    </button>
                  </div>
                </div>

                {/* Images View */}
                {activeTab === 'comparison' && currentIntervention && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Before Image */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-[#A7B1C2] font-mono">Original Input</span>
                        <span className="font-semibold text-indigo-400">
                          {auditResult.original_prediction.label} ({(auditResult.original_prediction.score * 100).toFixed(1)}%)
                        </span>
                      </div>
                      <div className="aspect-square rounded-lg overflow-hidden bg-black/50 border border-[rgba(148,163,184,0.1)]">
                        <img src={auditResult.original_image_base64} alt="Original" className="w-full h-full object-cover" />
                      </div>
                    </div>

                    {/* After Image */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-[#A7B1C2] font-mono">
                          Intervention: {currentIntervention.intervention_type} ({currentIntervention.method})
                        </span>
                        <span className={`font-semibold ${currentIntervention.prediction_flipped ? 'text-amber-400' : 'text-emerald-400'}`}>
                          {currentIntervention.prediction.label} ({(currentIntervention.prediction.score * 100).toFixed(1)}%)
                        </span>
                      </div>
                      <div className="aspect-square rounded-lg overflow-hidden bg-black/50 border border-[rgba(148,163,184,0.1)]">
                        <img src={currentIntervention.modified_image_base64} alt="Intervention" className="w-full h-full object-cover" />
                      </div>
                    </div>
                  </div>
                )}

                {activeTab === 'mask' && auditResult.mask_base64 && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <span className="text-xs text-[#A7B1C2] font-mono block">Original Image</span>
                      <div className="aspect-square rounded-lg overflow-hidden bg-black/50 border border-[rgba(148,163,184,0.1)]">
                        <img src={auditResult.original_image_base64} alt="Original" className="w-full h-full object-cover" />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <span className="text-xs text-[#A7B1C2] font-mono block">DeepLabV3 Binary Foreground Mask</span>
                      <div className="aspect-square rounded-lg overflow-hidden bg-black/50 border border-[rgba(148,163,184,0.1)]">
                        <img src={auditResult.mask_base64} alt="Mask" className="w-full h-full object-cover" />
                      </div>
                    </div>
                  </div>
                )}

                {/* Quantitative Impact Card */}
                {currentIntervention && (
                  <div className="mt-5 p-3.5 rounded-lg bg-[#0A0D14] border border-[rgba(148,163,184,0.1)] grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
                    <div>
                      <span className="text-[#718096] text-[10px] block">Prediction Status</span>
                      <span className={`font-semibold ${currentIntervention.prediction_flipped ? 'text-[#FB7185]' : 'text-emerald-400'}`}>
                        {currentIntervention.prediction_flipped ? 'Flipped Label' : 'Label Maintained'}
                      </span>
                    </div>

                    <div>
                      <span className="text-[#718096] text-[10px] block">Orig Class Prob Drop</span>
                      <span className="text-[#F3F6FC] font-semibold">
                        {(currentIntervention.original_predicted_class_score_change * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div>
                      <span className="text-[#718096] text-[10px] block">Distribution JSD</span>
                      <span className="text-indigo-300 font-semibold">
                        {currentIntervention.jsd?.toFixed(4) || 'N/A'}
                      </span>
                    </div>

                    <div>
                      <span className="text-[#718096] text-[10px] block">Intervention Fill</span>
                      <span className="text-[#A7B1C2] capitalize">{currentIntervention.method}</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Automated Audit Diagnostic Summary */}
              <div className="p-4 rounded-xl bg-indigo-950/20 border border-indigo-500/30 text-xs space-y-2">
                <h4 className="font-semibold text-indigo-300 flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-indigo-400" />
                  Deterministic Audit Summary
                </h4>
                <p className="text-[#F3F6FC] leading-relaxed">
                  {auditResult.summary_text}
                </p>
                <div className="pt-2 border-t border-indigo-500/20 text-[11px] text-[#A7B1C2] space-y-1">
                  <span className="font-semibold text-amber-300 block">Scientific Caveat:</span>
                  <p>
                    {auditResult.limitations[0]}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

