"""
Tests for CVA intervention functions.
"""

import numpy as np
import pytest
from PIL import Image

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cva.interventions import (
    apply_background_modification,
    apply_foreground_suppression,
    apply_background_swap,
)


def _create_test_image(w=100, h=100) -> Image.Image:
    """Create a simple test image."""
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _create_test_mask(w=100, h=100) -> np.ndarray:
    """Create a test mask with center foreground."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[25:75, 25:75] = 1  # Center square is foreground
    return mask


class TestBackgroundModification:
    """Tests for background modification intervention."""

    def test_output_shape_matches_input(self):
        image = _create_test_image()
        mask = _create_test_mask()
        result = apply_background_modification(image, mask, method="gray")
        assert result.size == image.size

    def test_output_is_pil_image(self):
        image = _create_test_image()
        mask = _create_test_mask()
        result = apply_background_modification(image, mask, method="gray")
        assert isinstance(result, Image.Image)

    @pytest.mark.parametrize("method", ["gray", "blur", "noise", "black", "white"])
    def test_all_methods_run(self, method):
        image = _create_test_image()
        mask = _create_test_mask()
        result = apply_background_modification(image, mask, method=method)
        assert isinstance(result, Image.Image)
        assert result.size == image.size

    def test_foreground_preserved_with_gray_bg(self):
        image = _create_test_image()
        mask = _create_test_mask()
        result = apply_background_modification(image, mask, method="gray")

        orig_arr = np.array(image)
        result_arr = np.array(result)

        # Foreground pixels should be unchanged
        fg_orig = orig_arr[25:75, 25:75]
        fg_result = result_arr[25:75, 25:75]
        assert np.array_equal(fg_orig, fg_result)

    def test_background_changed(self):
        image = _create_test_image()
        mask = _create_test_mask()
        result = apply_background_modification(image, mask, method="gray")

        orig_arr = np.array(image)
        result_arr = np.array(result)

        # Background should be different (gray = 128)
        bg_result = result_arr[0:25, 0:25]
        assert np.all(bg_result == 128)

    def test_invalid_method_raises(self):
        image = _create_test_image()
        mask = _create_test_mask()
        with pytest.raises(ValueError):
            apply_background_modification(image, mask, method="invalid")


class TestForegroundSuppression:
    """Tests for foreground suppression intervention."""

    def test_output_shape_matches_input(self):
        image = _create_test_image()
        mask = _create_test_mask()
        result = apply_foreground_suppression(image, mask, method="gray")
        assert result.size == image.size

    @pytest.mark.parametrize("method", ["gray", "blur", "black", "noise"])
    def test_all_methods_run(self, method):
        image = _create_test_image()
        mask = _create_test_mask()
        result = apply_foreground_suppression(image, mask, method=method)
        assert isinstance(result, Image.Image)

    def test_background_preserved_with_gray_fg(self):
        image = _create_test_image()
        mask = _create_test_mask()
        result = apply_foreground_suppression(image, mask, method="gray")

        orig_arr = np.array(image)
        result_arr = np.array(result)

        # Background pixels should be unchanged
        bg_orig = orig_arr[0:25, 0:25]
        bg_result = result_arr[0:25, 0:25]
        assert np.array_equal(bg_orig, bg_result)

    def test_foreground_suppressed(self):
        image = _create_test_image()
        mask = _create_test_mask()
        result = apply_foreground_suppression(image, mask, method="gray")

        result_arr = np.array(result)
        # Foreground should be gray (128)
        fg_result = result_arr[25:75, 25:75]
        assert np.all(fg_result == 128)


class TestBackgroundSwap:
    """Tests for background swap intervention."""

    def test_output_types(self):
        img1 = _create_test_image()
        img2 = _create_test_image()
        mask1 = _create_test_mask()
        mask2 = _create_test_mask()

        result1, result2 = apply_background_swap(img1, mask1, img2, mask2)
        assert isinstance(result1, Image.Image)
        assert isinstance(result2, Image.Image)

    def test_output_sizes(self):
        img1 = _create_test_image(100, 100)
        img2 = _create_test_image(150, 150)
        mask1 = _create_test_mask(100, 100)
        mask2 = _create_test_mask(150, 150)

        result1, result2 = apply_background_swap(img1, mask1, img2, mask2)
        # Both should match img1's size
        assert result1.size == img1.size
        assert result2.size == img1.size

    def test_mask_resize_handling(self):
        """Test that different mask and image sizes are handled."""
        img = _create_test_image(100, 100)
        mask = _create_test_mask(50, 50)  # Different size
        result = apply_background_modification(img, mask, method="gray")
        assert result.size == img.size
