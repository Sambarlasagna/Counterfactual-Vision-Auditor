import base64
import io
import uuid
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, List, Optional

from app.ml.model_registry import registry
from app.schemas.audit import AuditResponse, PredictionItem, InterventionResult
from cva.interventions import InterventionRunner
from cva.metrics import compute_jsd

def pil_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def mask_to_base64(mask_np: np.ndarray) -> str:
    # mask is binary 0/1 -> convert to 0/255 visual grayscale
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
    return pil_to_base64(mask_img, format="PNG")

class AuditService:
    def __init__(self):
        self.cached_reports: Dict[str, Dict[str, Any]] = {}

    def run_single_audit(
        self,
        image_bytes: bytes,
        filename: str,
        model_id: str = "waterbirds-resnet18",
        intervention_configs: Optional[List[Dict[str, str]]] = None,
        true_label: Optional[int] = None
    ) -> AuditResponse:
        model = registry.get_model(model_id)
        model_meta = registry.get_config(model_id)
        if not model or not model_meta:
            raise ValueError(f"Model '{model_id}' is not ready or does not exist.")

        # Default interventions if not provided
        if not intervention_configs:
            intervention_configs = [
                {"type": "background_modification", "method": "gray"},
                {"type": "background_modification", "method": "blur"},
                {"type": "foreground_suppression", "method": "gray"},
                {"type": "foreground_suppression", "method": "blur"}
            ]

        # Decode image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image file: {e}")

        orig_w, orig_h = image.size
        seg_model = registry.get_segmentation_model()
        runner = InterventionRunner(
            model=model,
            segmentation=seg_model,
            device=registry.device,
            image_size=model_meta["input_size"][0]
        )

        # Baseline prediction
        orig_probs, orig_pred_idx = runner.get_prediction(image)
        labels_map = model_meta["labels"]

        top_k = []
        # Sort indices by probability desc
        sorted_indices = np.argsort(orig_probs)[::-1]
        for idx in sorted_indices:
            top_k.append(PredictionItem(
                class_index=int(idx),
                label=labels_map.get(idx, f"Class {idx}"),
                score=float(orig_probs[idx])
            ))

        orig_prediction = PredictionItem(
            class_index=int(orig_pred_idx),
            label=labels_map.get(orig_pred_idx, f"Class {orig_pred_idx}"),
            score=float(orig_probs[orig_pred_idx])
        )

        # Get mask for visualization
        mask_np = seg_model.get_mask(image)
        mask_b64 = mask_to_base64(mask_np)

        # Run configured interventions
        intervention_results: List[InterventionResult] = []
        for conf in intervention_configs:
            itype = conf["type"]
            imethod = conf.get("method", "gray")
            
            res = runner.run_intervention(
                image=image,
                intervention_type=itype,
                method=imethod
            )
            
            mod_probs = res["modified_probs"]
            mod_pred_idx = res["modified_pred"]
            
            # Prediction score change for original predicted class
            score_change = float(mod_probs[orig_pred_idx] - orig_probs[orig_pred_idx])
            true_score_change = None
            if true_label is not None and 0 <= true_label < len(orig_probs):
                true_score_change = float(mod_probs[true_label] - orig_probs[true_label])

            # Compute JSD between distributions
            orig_p_2d = np.atleast_2d(orig_probs)
            mod_p_2d = np.atleast_2d(mod_probs)
            jsd_val = float(compute_jsd(orig_p_2d, mod_p_2d)[0])

            mod_top_k = []
            for m_idx in np.argsort(mod_probs)[::-1]:
                mod_top_k.append(PredictionItem(
                    class_index=int(m_idx),
                    label=labels_map.get(m_idx, f"Class {m_idx}"),
                    score=float(mod_probs[m_idx])
                ))

            # Base64 encoded transformed image
            mod_img_b64 = pil_to_base64(res["modified_image"], format="JPEG")

            intervention_results.append(InterventionResult(
                intervention_type=itype,
                method=imethod,
                prediction=PredictionItem(
                    class_index=int(mod_pred_idx),
                    label=labels_map.get(mod_pred_idx, f"Class {mod_pred_idx}"),
                    score=float(mod_probs[mod_pred_idx])
                ),
                top_k=mod_top_k,
                original_predicted_class_score_change=score_change,
                true_class_score_change=true_score_change,
                prediction_flipped=bool(res["prediction_flipped"]),
                modified_image_base64=f"data:image/jpeg;base64,{mod_img_b64}",
                jsd=jsd_val,
                notes=f"{itype} using {imethod} fill"
            ))

        # Generate deterministic scientific summary text
        flips = [ir for ir in intervention_results if ir.prediction_flipped]
        bg_flips = [ir for ir in flips if "background" in ir.intervention_type]
        fg_flips = [ir for ir in flips if "foreground" in ir.intervention_type]

        if bg_flips and not fg_flips:
            summary = (
                f"The model predicted '{orig_prediction.label}' with {orig_prediction.score*100:.1f}% confidence. "
                f"Modifying the background flipped the model's prediction, while suppressing the foreground object maintained consistent behavior. "
                "This demonstrates severe background-context dependency."
            )
        elif bg_flips and fg_flips:
            summary = (
                f"The model predicted '{orig_prediction.label}' ({orig_prediction.score*100:.1f}%). "
                "Both background modification and foreground suppression triggered prediction flips, "
                "indicating sensitivity to both contextual cues and foreground features."
            )
        else:
            avg_bg_drop = np.mean([abs(ir.original_predicted_class_score_change) for ir in intervention_results if "background" in ir.intervention_type]) if intervention_results else 0
            summary = (
                f"The model predicted '{orig_prediction.label}' ({orig_prediction.score*100:.1f}%). "
                f"Prediction remained stable across interventions, with an average confidence fluctuation of {avg_bg_drop*100:.1f}%. "
                "The classification decision is resilient to background perturbations for this sample."
            )

        audit_id = f"audit_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.utcnow().isoformat() + "Z"

        response = AuditResponse(
            audit_id=audit_id,
            created_at=now_iso,
            model={
                "id": model_meta["id"],
                "name": model_meta["name"],
                "architecture": model_meta["architecture"],
                "num_classes": model_meta["num_classes"]
            },
            input_metadata={
                "filename": filename,
                "width": orig_w,
                "height": orig_h,
                "channels": 3
            },
            original_image_base64=f"data:image/jpeg;base64,{pil_to_base64(image)}",
            mask_base64=f"data:image/png;base64,{mask_b64}",
            original_prediction=orig_prediction,
            top_k=top_k,
            interventions=intervention_results,
            summary_text=summary,
            limitations=[
                "Prediction sensitivity under intervention indicates feature dependence, but does not provide definitive causal proof.",
                "Semantic segmentation masks generated via DeepLabV3 may introduce minor boundary or compositing artifacts.",
                "Replacing natural backgrounds with synthetic gray or blur introduces out-of-distribution visual properties."
            ]
        )

        self.cached_reports[audit_id] = response.model_dump()
        return response

    def get_report(self, audit_id: str) -> Optional[Dict[str, Any]]:
        return self.cached_reports.get(audit_id)

audit_service = AuditService()
