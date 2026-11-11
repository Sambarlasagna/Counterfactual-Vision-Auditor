import { HealthStatus, ModelInfo, AuditResponse, BenchmarkSummary } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/api/v1/health`);
  if (!res.ok) throw new Error('API server unavailable');
  return res.json();
}

export async function fetchModels(): Promise<{ models: ModelInfo[] }> {
  const res = await fetch(`${API_BASE}/api/v1/models`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

export async function fetchBenchmarks(): Promise<BenchmarkSummary> {
  const res = await fetch(`${API_BASE}/api/v1/benchmarks`);
  if (!res.ok) throw new Error('Failed to fetch benchmarks');
  return res.json();
}

export async function runAudit(
  imageFile: File,
  modelId: string = 'waterbirds-resnet18',
  interventions?: Array<{ type: string; method: string }>,
  trueLabel?: number
): Promise<AuditResponse> {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('model_id', modelId);
  if (interventions) {
    formData.append('interventions_json', JSON.stringify(interventions));
  }
  if (trueLabel !== undefined && trueLabel !== null) {
    formData.append('true_label', trueLabel.toString());
  }

  const res = await fetch(`${API_BASE}/api/v1/audits`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Audit execution failed');
  }

  return res.json();
}

