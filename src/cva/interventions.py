"""
Counterfactual intervention methods for CVA.

Implements:
  1. Background Modification — preserve foreground, replace background
  2. Foreground Suppression — mask/blur foreground, keep background
  3. Background Swap — swap backgrounds between groups (optional)

All interventions operate on PIL images and return PIL images.
Segmentation masks are obtained from a pretrained DeepLabV3 or
from dataset-provided masks, and are cached for efficiency.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms

logger = logging.getLogger("cva")


# ── Segmentation ───────────────────────────────────────────────────────

class SegmentationModel:
    """
    Pretrained segmentation model for generating foreground masks.

    Uses DeepLabV3 with ResNet-50 backbone from torchvision.
    Masks are cached to disk to avoid recomputation.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        cache_dir: Optional[str] = "outputs/masks",
        threshold: float = 0.5,
    ):
        """
        Args:
            device: Device to run the segmentation model on.
            cache_dir: Directory to cache computed masks. None to disable caching.
            threshold: Probability threshold for binary mask.
        """
        self.device = device
        self.threshold = threshold
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        """Lazy-load the segmentation model."""
        if self._model is not None:
            return

        from torchvision.models.segmentation import (
            deeplabv3_resnet50,
            DeepLabV3_ResNet50_Weights,
        )

        logger.info("Loading DeepLabV3-ResNet50 segmentation model...")
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self._model = deeplabv3_resnet50(weights=weights)
        self._model = self._model.to(self.device)
        self._model.eval()
        logger.info("Segmentation model loaded successfully")

    def _get_cache_key(self, image_path: str) -> str:
        """Generate a cache key from the image path."""
        return hashlib.md5(image_path.encode()).hexdigest()

    def _load_cached_mask(self, image_path: str) -> Optional[np.ndarray]:
        """Try to load a cached mask."""
        if self.cache_dir is None:
            return None
        cache_key = self._get_cache_key(image_path)
        cache_path = self.cache_dir / f"{cache_key}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        return None

    def _save_cached_mask(self, image_path: str, mask: np.ndarray) -> None:
        """Save a mask to cache."""
        if self.cache_dir is None:
            return
        cache_key = self._get_cache_key(image_path)
        cache_path = self.cache_dir / f"{cache_key}.npy"
        np.save(cache_path, mask)

    @torch.no_grad()
    def get_mask(
        self,
        image: Image.Image,
        image_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate a binary foreground mask for an image.

        The mask uses the "bird" class (class 3 in COCO/VOC) from DeepLabV3.
        Falls back to a saliency-based approach if no bird is detected.

        Args:
            image: PIL Image (RGB).
            image_path: Optional path for caching.

        Returns:
            Binary mask as numpy array (H, W), 1=foreground, 0=background.
        """
        # Check cache
        if image_path:
            cached = self._load_cached_mask(image_path)
            if cached is not None:
                return cached

        self._load_model()

        # Prepare input
        input_tensor = self._transform(image).unsqueeze(0).to(self.device)

        # Run segmentation
        output = self._model(input_tensor)["out"]
        probs = torch.softmax(output, dim=1)

        # Bird class index in COCO/VOC is 3
        # Also consider class 0 as background
        # Take the max of all non-background classes as foreground
        bg_prob = probs[0, 0]  # background probability
        fg_prob = 1.0 - bg_prob

        # Resize to original image size
        original_size = (image.height, image.width)
        fg_prob = F.interpolate(
            fg_prob.unsqueeze(0).unsqueeze(0),
            size=original_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Create binary mask
        mask = (fg_prob > self.threshold).astype(np.uint8)

        # If mask is too small (< 5% of image), try a lower threshold
        mask_ratio = mask.sum() / mask.size
        if mask_ratio < 0.05:
            mask = (fg_prob > 0.3).astype(np.uint8)

        # If still too small, use center crop heuristic
        if mask.sum() / mask.size < 0.02:
            logger.warning(
                f"Very small mask detected ({mask.sum()/mask.size:.3f}). "
                f"Using center region fallback."
            )
            h, w = mask.shape
            ch, cw = h // 4, w // 4
            mask[ch:h-ch, cw:w-cw] = 1

        # Cache
        if image_path:
            self._save_cached_mask(image_path, mask)

        return mask

    def get_masks_batch(
        self,
        images: List[Image.Image],
        image_paths: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """
        Generate masks for a batch of images.

        Args:
            images: List of PIL Images.
            image_paths: Optional list of paths for caching.

        Returns:
            List of binary masks.
        """
        if image_paths is None:
            image_paths = [None] * len(images)

        masks = []
        for img, path in zip(images, image_paths):
            masks.append(self.get_mask(img, path))
        return masks


# ── Intervention Functions ─────────────────────────────────────────────

def apply_background_modification(
    image: Image.Image,
    mask: np.ndarray,
    method: Literal["gray", "blur", "noise", "black", "white"] = "gray",
    blur_radius: int = 20,
) -> Image.Image:
    """
    Intervention 1: Preserve foreground, modify background.

    Args:
        image: Original PIL image (RGB).
        mask: Binary mask (H, W), 1=foreground.
        method: Background replacement method.
        blur_radius: Blur radius for 'blur' method.

    Returns:
        Modified PIL image with altered background.
    """
    img_array = np.array(image).copy()
    h, w = mask.shape[:2]

    # Resize mask if needed
    if mask.shape[:2] != img_array.shape[:2]:
        mask_img = Image.fromarray(mask * 255)
        mask_img = mask_img.resize((img_array.shape[1], img_array.shape[0]), Image.NEAREST)
        mask = (np.array(mask_img) > 127).astype(np.uint8)

    bg_mask = 1 - mask  # Invert: 1=background

    if method == "gray":
        # Replace background with mean gray
        bg_color = [128, 128, 128]
        for c in range(3):
            img_array[:, :, c] = (
                img_array[:, :, c] * mask + bg_color[c] * bg_mask
            )

    elif method == "blur":
        # Apply heavy Gaussian blur to background
        blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blurred_array = np.array(blurred)
        for c in range(3):
            img_array[:, :, c] = (
                img_array[:, :, c] * mask + blurred_array[:, :, c] * bg_mask
            )

    elif method == "noise":
        # Replace background with random noise
        noise = np.random.randint(0, 256, img_array.shape, dtype=np.uint8)
        for c in range(3):
            img_array[:, :, c] = (
                img_array[:, :, c] * mask + noise[:, :, c] * bg_mask
            )

    elif method == "black":
        for c in range(3):
            img_array[:, :, c] = img_array[:, :, c] * mask

    elif method == "white":
        for c in range(3):
            img_array[:, :, c] = (
                img_array[:, :, c] * mask + 255 * bg_mask
            )

    else:
        raise ValueError(f"Unknown background modification method: '{method}'")

    return Image.fromarray(img_array.astype(np.uint8))


def apply_foreground_suppression(
    image: Image.Image,
    mask: np.ndarray,
    method: Literal["gray", "blur", "black", "noise"] = "gray",
    blur_radius: int = 20,
) -> Image.Image:
    """
    Intervention 2: Mask/blur foreground, keep background.

    If the model still predicts correctly with the foreground suppressed,
    it suggests the model relies on background cues.

    Args:
        image: Original PIL image (RGB).
        mask: Binary mask (H, W), 1=foreground.
        method: Foreground replacement method.
        blur_radius: Blur radius for 'blur' method.

    Returns:
        Modified PIL image with suppressed foreground.
    """
    img_array = np.array(image).copy()

    # Resize mask if needed
    if mask.shape[:2] != img_array.shape[:2]:
        mask_img = Image.fromarray(mask * 255)
        mask_img = mask_img.resize((img_array.shape[1], img_array.shape[0]), Image.NEAREST)
        mask = (np.array(mask_img) > 127).astype(np.uint8)

    fg_mask = mask  # 1=foreground (to be suppressed)
    bg_mask = 1 - mask

    if method == "gray":
        bg_color = [128, 128, 128]
        for c in range(3):
            img_array[:, :, c] = (
                bg_color[c] * fg_mask + img_array[:, :, c] * bg_mask
            )

    elif method == "blur":
        blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blurred_array = np.array(blurred)
        for c in range(3):
            img_array[:, :, c] = (
                blurred_array[:, :, c] * fg_mask + img_array[:, :, c] * bg_mask
            )

    elif method == "black":
        for c in range(3):
            img_array[:, :, c] = img_array[:, :, c] * bg_mask

    elif method == "noise":
        noise = np.random.randint(0, 256, img_array.shape, dtype=np.uint8)
        for c in range(3):
            img_array[:, :, c] = (
                noise[:, :, c] * fg_mask + img_array[:, :, c] * bg_mask
            )

    else:
        raise ValueError(f"Unknown foreground suppression method: '{method}'")

    return Image.fromarray(img_array.astype(np.uint8))


def apply_background_swap(
    image1: Image.Image,
    mask1: np.ndarray,
    image2: Image.Image,
    mask2: np.ndarray,
) -> Tuple[Image.Image, Image.Image]:
    """
    Intervention 3: Swap backgrounds between two images.

    Puts the foreground of image1 on the background of image2, and vice versa.

    Args:
        image1: First PIL image.
        mask1: Foreground mask for image1.
        image2: Second PIL image.
        mask2: Foreground mask for image2.

    Returns:
        Tuple of (image1_fg_on_image2_bg, image2_fg_on_image1_bg).
    """
    # Ensure same size
    target_size = image1.size  # (W, H)
    image2_resized = image2.resize(target_size, Image.BILINEAR)

    img1_array = np.array(image1).copy()
    img2_array = np.array(image2_resized).copy()

    # Resize masks
    def _resize_mask(mask, target_hw):
        if mask.shape[:2] != target_hw:
            mask_img = Image.fromarray(mask * 255)
            mask_img = mask_img.resize((target_hw[1], target_hw[0]), Image.NEAREST)
            return (np.array(mask_img) > 127).astype(np.uint8)
        return mask

    h, w = img1_array.shape[:2]
    mask1 = _resize_mask(mask1, (h, w))
    mask2 = _resize_mask(mask2, (h, w))

    # Image 1 foreground on Image 2 background
    composite1 = img1_array.copy()
    for c in range(3):
        composite1[:, :, c] = (
            img1_array[:, :, c] * mask1 + img2_array[:, :, c] * (1 - mask1)
        )

    # Image 2 foreground on Image 1 background
    composite2 = img2_array.copy()
    for c in range(3):
        composite2[:, :, c] = (
            img2_array[:, :, c] * mask2 + img1_array[:, :, c] * (1 - mask2)
        )

    return (
        Image.fromarray(composite1.astype(np.uint8)),
        Image.fromarray(composite2.astype(np.uint8)),
    )


# ── Intervention Runner ────────────────────────────────────────────────

class InterventionRunner:
    """
    High-level runner that applies interventions to a dataset
    and collects model predictions before and after.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        segmentation: SegmentationModel,
        device: torch.device,
        image_size: int = 224,
    ):
        """
        Args:
            model: Trained classifier to audit.
            segmentation: Segmentation model for mask generation.
            device: Device for inference.
            image_size: Size for model input.
        """
        self.model = model
        self.segmentation = segmentation
        self.device = device
        self.image_size = image_size

        self.eval_transform = transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.model.eval()

    @torch.no_grad()
    def get_prediction(self, image: Image.Image) -> Tuple[np.ndarray, int]:
        """
        Get model prediction for a single image.

        Args:
            image: PIL Image (RGB).

        Returns:
            Tuple of (probability_distribution, predicted_class).
        """
        tensor = self.eval_transform(image).unsqueeze(0).to(self.device)
        output = self.model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
        return probs, pred

    def run_intervention(
        self,
        image: Image.Image,
        image_path: Optional[str] = None,
        intervention_type: str = "background_modification",
        method: str = "gray",
        **kwargs,
    ) -> Dict:
        """
        Apply an intervention and compare predictions.

        Args:
            image: Original PIL image.
            image_path: Path for mask caching.
            intervention_type: One of 'background_modification', 'foreground_suppression'.
            method: Specific method (e.g., 'gray', 'blur').

        Returns:
            Dict with original/intervened images, predictions, and metrics.
        """
        # Get mask
        mask = self.segmentation.get_mask(image, image_path)

        # Get original prediction
        probs_orig, pred_orig = self.get_prediction(image)

        # Apply intervention
        if intervention_type == "background_modification":
            modified = apply_background_modification(image, mask, method=method, **kwargs)
        elif intervention_type == "foreground_suppression":
            modified = apply_foreground_suppression(image, mask, method=method, **kwargs)
        else:
            raise ValueError(f"Unknown intervention type: '{intervention_type}'")

        # Get intervened prediction
        probs_mod, pred_mod = self.get_prediction(modified)

        return {
            "original_image": image,
            "modified_image": modified,
            "mask": mask,
            "original_probs": probs_orig,
            "modified_probs": probs_mod,
            "original_pred": pred_orig,
            "modified_pred": pred_mod,
            "prediction_flipped": pred_orig != pred_mod,
            "intervention_type": intervention_type,
            "method": method,
        }
