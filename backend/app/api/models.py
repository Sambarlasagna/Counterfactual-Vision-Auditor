from fastapi import APIRouter
from app.schemas.audit import ModelsResponse, ModelInfo
from app.ml.model_registry import registry

router = APIRouter(tags=["Models"])

@router.get("/models", response_model=ModelsResponse)
def get_models():
    models_list = []
    for m in registry.list_models():
        models_list.append(ModelInfo(**m))

    return ModelsResponse(
        models=models_list,
        supported_custom_architectures=["resnet18", "resnet50", "mobilenet_v3_large"]
    )
