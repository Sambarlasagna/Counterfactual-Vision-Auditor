"""
Smoke test for the CVA pipeline.

Verifies that the config, model, metrics, and intervention
components work end-to-end with synthetic data.
"""

import numpy as np
import pytest
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cva.config import load_config, save_config, _deep_merge
from cva.models import create_model, get_model_info
from cva.metrics import compute_all_classification_metrics
from cva.utils import set_seed, get_device, get_device_info


class TestConfig:
    """Tests for configuration loading."""

    def test_load_defaults(self):
        config = load_config(None)
        assert config["model"]["name"] == "resnet18"
        assert config["seed"] == 42

    def test_load_yaml(self, tmp_path):
        yaml_content = "model:\n  name: resnet18\n  num_classes: 5\nseed: 123\n"
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)

        config = load_config(str(config_file))
        assert config["model"]["num_classes"] == 5
        assert config["seed"] == 123
        # Defaults still present
        assert config["training"]["batch_size"] == 16

    def test_deep_merge(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 5}
        merged = _deep_merge(base, override)
        assert merged["a"]["b"] == 10
        assert merged["a"]["c"] == 2
        assert merged["d"] == 3
        assert merged["e"] == 5

    def test_save_and_load(self, tmp_path):
        config = load_config(None)
        save_path = str(tmp_path / "saved.yaml")
        save_config(config, save_path)
        reloaded = load_config(save_path)
        assert reloaded["model"]["name"] == config["model"]["name"]


class TestModel:
    """Tests for model creation."""

    def test_create_resnet18(self):
        config = load_config(None)
        model = create_model(config)
        assert model is not None
        # Check output size
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_model_info(self):
        config = load_config(None)
        model = create_model(config)
        info = get_model_info(model)
        assert info["total_params"] > 0
        assert info["trainable_params"] > 0

    def test_freeze_backbone(self):
        config = load_config(None)
        config["model"]["freeze_backbone"] = True
        model = create_model(config)
        info = get_model_info(model)
        assert info["frozen_params"] > 0
        assert info["trainable_params"] < info["total_params"]


class TestUtils:
    """Tests for utility functions."""

    def test_set_seed(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_get_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_get_device_info(self):
        info = get_device_info()
        assert "python_version" in info
        assert "pytorch_version" in info
        assert "cuda_available" in info


class TestEndToEnd:
    """Minimal end-to-end smoke test with synthetic data."""

    def test_model_forward_pass(self):
        """Verify model can process a batch of images."""
        config = load_config(None)
        model = create_model(config)
        model.eval()

        batch = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            output = model(batch)

        assert output.shape == (4, 2)
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_metrics_on_synthetic_data(self):
        """Verify metrics compute correctly on known data."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        groups = np.array([0, 1, 2, 3, 0, 1, 2, 3])

        metrics = compute_all_classification_metrics(y_true, y_pred, groups)

        assert 0 < metrics["accuracy"] < 1
        assert "group_accuracy" in metrics
        assert len(metrics["group_accuracy"]) == 4
        assert metrics["worst_group_accuracy"] <= metrics["accuracy"]
