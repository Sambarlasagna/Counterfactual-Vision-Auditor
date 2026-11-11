import os
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.schemas.audit import BenchmarkSummary
from app.core.config import settings

router = APIRouter(tags=["Benchmarks"])

@router.get("/benchmarks", response_model=BenchmarkSummary)
def get_benchmarks():
    test_metrics_path = os.path.join(settings.METRICS_DIR, "resnet18_test_metrics.json")
    audit_combined_path = os.path.join(settings.METRICS_DIR, "audit_combined.json")

    if not os.path.exists(test_metrics_path):
        raise HTTPException(status_code=404, detail="Benchmark metrics not yet computed.")

    with open(test_metrics_path, "r", encoding="utf-8") as f:
        test_metrics = json.load(f)

    audit_summary = {}
    if os.path.exists(audit_combined_path):
        with open(audit_combined_path, "r", encoding="utf-8") as f:
            audit_summary = json.load(f)

    # Figure filenames
    figures_list = []
    if os.path.exists(settings.FIGURES_DIR):
        for fname in os.listdir(settings.FIGURES_DIR):
            if fname.endswith(".png"):
                figures_list.append(fname)

    return BenchmarkSummary(
        dataset="Waterbirds (WILDS Benchmark / CUB-200-2011 + Places)",
        split=test_metrics.get("split", "test"),
        overall_accuracy=test_metrics.get("accuracy", 0.6056),
        balanced_accuracy=test_metrics.get("balanced_accuracy", 0.5976),
        worst_group_accuracy=test_metrics.get("worst_group_accuracy", 0.2445),
        worst_group_id=test_metrics.get("worst_group_id", 2),
        per_group_accuracy={str(k): float(v) for k, v in test_metrics.get("group_accuracy", {}).items()},
        group_counts={str(k): int(v) for k, v in test_metrics.get("group_counts", {}).items()},
        group_names={
            "0": "Landbird on Land (Majority)",
            "1": "Landbird on Water (Minority)",
            "2": "Waterbird on Land (Minority - Worst)",
            "3": "Waterbird on Water (Majority)"
        },
        audit_summary=audit_summary,
        figures=sorted(figures_list)
    )

@router.get("/benchmarks/figures/{figure_name}")
def get_benchmark_figure(figure_name: str):
    # Prevent directory traversal
    clean_name = os.path.basename(figure_name)
    fig_path = os.path.join(settings.FIGURES_DIR, clean_name)
    if not os.path.exists(fig_path):
        raise HTTPException(status_code=404, detail="Figure not found.")
    return FileResponse(fig_path, media_type="image/png")

