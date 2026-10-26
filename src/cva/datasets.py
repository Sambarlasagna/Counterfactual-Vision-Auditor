"""
Dataset loaders for CVA experiments.

Supports:
  - Waterbirds benchmark (primary)
  - CUB-200-2011 with synthetic groups (fallback)

All loaders return (image, label, group_id, metadata_dict) tuples
and provide group definitions for evaluation.
"""

import os
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image


logger = logging.getLogger("cva")


# ── Group Definitions ──────────────────────────────────────────────────

WATERBIRDS_GROUPS = {
    0: {"bird": "landbird", "background": "land", "label": 0, "place": 0},
    1: {"bird": "landbird", "background": "water", "label": 0, "place": 1},
    2: {"bird": "waterbird", "background": "land", "label": 1, "place": 0},
    3: {"bird": "waterbird", "background": "water", "label": 1, "place": 1},
}

WATERBIRDS_GROUP_NAMES = {
    0: "Landbird on Land",
    1: "Landbird on Water",
    2: "Waterbird on Land",
    3: "Waterbird on Water",
}


# ── Transforms ─────────────────────────────────────────────────────────

def get_transforms(image_size: int = 224, split: str = "train") -> transforms.Compose:
    """
    Get standard image transforms for training or evaluation.

    Args:
        image_size: Target image size (square).
        split: One of 'train', 'val', 'test'.

    Returns:
        A torchvision Compose transform.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


# ── Waterbirds Dataset ─────────────────────────────────────────────────

class WaterbirdsDataset(Dataset):
    """
    Waterbirds benchmark dataset.

    Expects a directory structure with:
      - metadata.csv containing columns: img_id, img_filename, y, place, split
      - Image files accessible via the paths in metadata.csv

    Groups are defined by (bird_type, background_type):
      0: landbird on land     (majority)
      1: landbird on water    (minority)
      2: waterbird on land    (minority)
      3: waterbird on water   (majority)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224,
    ):
        """
        Args:
            root: Root directory containing the Waterbirds dataset.
            split: One of 'train' (0), 'val' (1), 'test' (2).
            transform: Optional image transform. If None, uses defaults.
            image_size: Image size for default transforms.
        """
        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        # Map split names to split indices
        split_map = {"train": 0, "val": 1, "test": 2}
        if split not in split_map:
            raise ValueError(f"split must be one of {list(split_map.keys())}, got '{split}'")
        split_idx = split_map[split]

        # Set transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transforms(image_size, split)

        # Load metadata
        metadata_path = self._find_metadata()
        self.samples = self._load_metadata(metadata_path, split_idx)

        # Compute group counts
        self.num_groups = 4
        self.group_counts = self._count_groups()

        logger.info(
            f"WaterbirdsDataset [{split}]: {len(self.samples)} samples, "
            f"group counts: {self.group_counts}"
        )

    def _find_metadata(self) -> Path:
        """Find the metadata CSV file in the dataset root."""
        candidates = [
            self.root / "metadata.csv",
            self.root / "waterbird_complete95_forest2water2" / "metadata.csv",
        ]
        for path in candidates:
            if path.exists():
                return path

        # Search recursively
        found = list(self.root.rglob("metadata.csv"))
        if found:
            return found[0]

        raise FileNotFoundError(
            f"metadata.csv not found in {self.root}. "
            f"Please download the Waterbirds dataset. See data/README.md."
        )

    def _load_metadata(self, metadata_path: Path, split_idx: int) -> List[Dict[str, Any]]:
        """Load and filter metadata for the requested split."""
        samples = []
        metadata_dir = metadata_path.parent

        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["split"]) != split_idx:
                    continue

                label = int(row["y"])
                place = int(row["place"])
                group_id = label * 2 + place  # 0, 1, 2, 3

                # Resolve image path
                img_filename = row["img_filename"]
                img_path = metadata_dir / img_filename
                if not img_path.exists():
                    img_path = self.root / img_filename

                samples.append({
                    "img_path": str(img_path),
                    "label": label,
                    "place": place,
                    "group_id": group_id,
                    "img_filename": img_filename,
                })

        if not samples:
            raise RuntimeError(
                f"No samples found for split index {split_idx} in {metadata_path}. "
                f"Check that the metadata file has the expected format."
            )

        return samples

    def _count_groups(self) -> Dict[int, int]:
        """Count samples per group."""
        counts = {g: 0 for g in range(self.num_groups)}
        for s in self.samples:
            counts[s["group_id"]] += 1
        return counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, Dict[str, Any]]:
        """
        Returns:
            image: Transformed image tensor.
            label: Class label (0=landbird, 1=waterbird).
            group_id: Group index (0-3).
            metadata: Dict with additional info.
        """
        sample = self.samples[idx]

        # Load image
        try:
            image = Image.open(sample["img_path"]).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image: {sample['img_path']}: {e}"
            )

        if self.transform is not None:
            image = self.transform(image)

        metadata = {
            "img_path": sample["img_path"],
            "img_filename": sample["img_filename"],
            "place": sample["place"],
        }

        return image, sample["label"], sample["group_id"], metadata


# ── Data Loader Factory ────────────────────────────────────────────────

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader with custom collate for metadata dicts.

    Args:
        dataset: A CVA dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for CUDA.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        A DataLoader instance.
    """
    def collate_fn(batch):
        images, labels, groups, metas = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        groups = torch.tensor(groups, dtype=torch.long)
        # Collate metadata dicts
        meta_batch = {key: [m[key] for m in metas] for key in metas[0]}
        return images, labels, groups, meta_batch

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def create_smoke_subset(dataset: Dataset, num_samples: int = 100) -> Subset:
    """
    Create a small subset of a dataset for smoke testing.

    Tries to sample equally from all groups.

    Args:
        dataset: The full dataset.
        num_samples: Target number of samples.

    Returns:
        A Subset with approximately num_samples items.
    """
    if len(dataset) <= num_samples:
        return dataset

    # Try to get balanced group representation
    group_indices = {}
    for i in range(len(dataset)):
        _, _, group_id, _ = dataset[i]
        if group_id not in group_indices:
            group_indices[group_id] = []
        group_indices[group_id].append(i)

    per_group = max(1, num_samples // max(len(group_indices), 1))
    selected = []
    for group_id, indices in group_indices.items():
        np.random.shuffle(indices)
        selected.extend(indices[:per_group])

    # If we still need more, add random remaining
    if len(selected) < num_samples:
        remaining = set(range(len(dataset))) - set(selected)
        remaining = list(remaining)
        np.random.shuffle(remaining)
        selected.extend(remaining[:num_samples - len(selected)])

    return Subset(dataset, selected[:num_samples])


def get_dataset_stats(dataset: Dataset) -> Dict[str, Any]:
    """
    Compute basic statistics about a dataset.

    Args:
        dataset: A CVA dataset instance.

    Returns:
        Dict with total count, per-group counts, per-label counts.
    """
    labels = []
    groups = []
    for i in range(len(dataset)):
        _, label, group_id, _ = dataset[i]
        labels.append(label)
        groups.append(group_id)

    labels = np.array(labels)
    groups = np.array(groups)

    stats = {
        "total": len(dataset),
        "num_classes": len(np.unique(labels)),
        "num_groups": len(np.unique(groups)),
        "per_label": {int(l): int((labels == l).sum()) for l in np.unique(labels)},
        "per_group": {int(g): int((groups == g).sum()) for g in np.unique(groups)},
        "group_names": WATERBIRDS_GROUP_NAMES,
    }
    return stats
