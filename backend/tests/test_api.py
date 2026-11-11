from fastapi.testclient import TestClient
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "demo_model_ready" in data

def test_models():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data["models"]) >= 1
    assert data["models"][0]["id"] == "waterbirds-resnet18"

def test_benchmarks():
    response = client.get("/api/v1/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "overall_accuracy" in data
    assert "worst_group_accuracy" in data
    assert len(data["figures"]) > 0

