from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    device: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    demo_model_ready: bool

class ModelInfo(BaseModel):
    id: str
    name: str
    architecture: str
    num_classes: int
    labels: Dict[int, str]
    input_size: List[int]
    description: str
    is_ready: bool
    provenance: Dict[str, Any]

class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    supported_custom_architectures: List[str]

class PredictionItem(BaseModel):
    class_index: int
    label: str
    score: float

class InterventionResult(BaseModel):
    intervention_type: str
    method: str
    prediction: PredictionItem
    top_k: List[PredictionItem]
    original_predicted_class_score_change: float
    true_class_score_change: Optional[float] = None
    prediction_flipped: bool
    modified_image_base64: str
    jsd: Optional[float] = None
    notes: Optional[str] = None

class AuditResponse(BaseModel):
    audit_id: str
    created_at: str
    model: Dict[str, Any]
    input_metadata: Dict[str, Any]
    original_image_base64: str
    mask_base64: Optional[str] = None
    original_prediction: PredictionItem
    top_k: List[PredictionItem]
    interventions: List[InterventionResult]
    summary_text: str
    limitations: List[str]

class BenchmarkSummary(BaseModel):
    dataset: str
    split: str
    overall_accuracy: float
    balanced_accuracy: float
    worst_group_accuracy: float
    worst_group_id: int
    per_group_accuracy: Dict[str, float]
    group_counts: Dict[str, int]
    group_names: Dict[str, str]
    audit_summary: Dict[str, Any]
    figures: List[str]
