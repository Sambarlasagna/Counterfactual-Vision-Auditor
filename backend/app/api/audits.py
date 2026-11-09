from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import json

from app.schemas.audit import AuditResponse
from app.services.audit_service import audit_service
from app.core.config import settings

router = APIRouter(tags=["Audits"])

@router.post("/audits", response_model=AuditResponse)
async def create_audit(
    file: UploadFile = File(...),
    model_id: str = Form("waterbirds-resnet18"),
    interventions_json: Optional[str] = Form(None),
    true_label: Optional[int] = Form(None)
):
    # Validate content type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload JPEG, PNG, or WEBP.")

    image_bytes = await file.read()
    max_bytes = settings.MAX_IMAGE_UPLOAD_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(status_code=400, detail=f"Image exceeds max size of {settings.MAX_IMAGE_UPLOAD_MB}MB.")

    interventions_config = None
    if interventions_json:
        try:
            interventions_config = json.loads(interventions_json)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid interventions_json format.")

    try:
        response = audit_service.run_single_audit(
            image_bytes=image_bytes,
            filename=file.filename or "upload.jpg",
            model_id=model_id,
            intervention_configs=interventions_config,
            true_label=true_label
        )
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@router.get("/reports/{audit_id}")
def get_report(audit_id: str):
    report = audit_service.get_report(audit_id)
    if not report:
        raise HTTPException(status_code=404, detail="Audit report not found or expired.")
    return report
