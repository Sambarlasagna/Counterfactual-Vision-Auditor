export interface PredictionItem {
  class_index: number;
  label: string;
  score: number;
}

export interface InterventionResult {
  intervention_type: string;
  method: string;
  prediction: PredictionItem;
  top_k: PredictionItem[];
  original_predicted_class_score_change: number;
  true_class_score_change?: number | null;
  prediction_flipped: boolean;
  modified_image_base64: string;
  jsd?: number | null;
  notes?: string | null;
}

export interface AuditResponse {
  audit_id: string;
  created_at: string;
  model: {
    id: string;
    name: string;
    architecture: string;
    num_classes: number;
  };
  input_metadata: {
    filename: string;
    width: number;
    height: number;
    channels: number;
  };
  original_image_base64: string;
  mask_base64?: string | null;
  original_prediction: PredictionItem;
  top_k: PredictionItem[];
  interventions: InterventionResult[];
  summary_text: string;
  limitations: string[];
}

export interface ModelInfo {
  id: string;
  name: string;
  architecture: string;
  num_classes: number;
  labels: Record<string, string>;
  input_size: number[];
  description: string;
  is_ready: boolean;
  provenance: Record<string, any>;
}

export interface BenchmarkSummary {
  dataset: string;
  split: string;
  overall_accuracy: number;
  balanced_accuracy: number;
  worst_group_accuracy: number;
  worst_group_id: number;
  per_group_accuracy: Record<string, number>;
  group_counts: Record<string, number>;
  group_names: Record<string, string>;
  audit_summary: Record<string, any>;
  figures: string[];
}

export interface HealthStatus {
  status: string;
  version: string;
  device: string;
  cuda_available: boolean;
  gpu_name?: string | null;
  demo_model_ready: boolean;
}

