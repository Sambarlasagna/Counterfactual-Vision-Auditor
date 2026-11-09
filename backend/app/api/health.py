from fastapi import APIRouter
from app.schemas.audit import HealthResponse
from app.ml.model_registry import registry
from cva.utils import get_device_info

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
def get_health():
    dev_info = get_device_info()
    demo_ready = "waterbirds-resnet18" in registry.models
    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment="development",
        device=str(registry.device),
        cuda_available=dev_info.get("cuda_available", False),
        gpu_name=dev_info.get("gpu_name"),
        demo_model_ready=demo_ready
    )
