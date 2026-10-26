#!/usr/bin/env python3
"""
Download the Waterbirds dataset for CVA experiments.

Tries multiple sources:
  1. Direct download from Stanford/group_DRO
  2. WILDS package
  3. Google Drive mirror

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --root data/waterbirds
"""

import argparse
import os
import sys
import tarfile
import urllib.request
import shutil
from pathlib import Path


URLS = [
    "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz",
    "https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4c88a0c55fcecb1ca0/contents/blob/",
]

DATASET_DIR_NAME = "waterbird_complete95_forest2water2"


def download_file(url, dest_path, desc="Downloading"):
    """Download a file with progress."""
    print(f"  {desc}: {url}")
    print(f"  Destination: {dest_path}")

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                mb_done = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {pct:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)
            else:
                mb_done = downloaded / (1024 * 1024)
                print(f"\r  Downloaded: {mb_done:.1f} MB", end="", flush=True)

        urllib.request.urlretrieve(url, dest_path, _progress)
        print()  # newline after progress
        return True
    except Exception as e:
        print(f"\n  Failed: {e}")
        return False


def extract_tar(tar_path, extract_to):
    """Extract a tar.gz file."""
    print(f"  Extracting {tar_path} to {extract_to}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"  Extraction complete.")


def try_direct_download(root):
    """Try downloading from direct URLs."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tar_path = root / "waterbirds.tar.gz"

    for i, url in enumerate(URLS):
        print(f"\n  Trying source {i+1}/{len(URLS)}...")
        if download_file(url, str(tar_path)):
            # Check file is valid
            if tar_path.stat().st_size > 1_000_000:  # At least 1MB
                try:
                    extract_tar(str(tar_path), str(root))
                    # Clean up tar
                    tar_path.unlink()
                    return True
                except Exception as e:
                    print(f"  Extraction failed: {e}")
                    if tar_path.exists():
                        tar_path.unlink()
            else:
                print(f"  File too small ({tar_path.stat().st_size} bytes), skipping.")
                tar_path.unlink()

    return False


def try_wilds_download(root):
    """Try downloading via WILDS package."""
    print("\n  Trying WILDS package...")
    try:
        from wilds import get_dataset
        dataset = get_dataset("waterbirds", download=True, root_dir=str(root))
        print("  WILDS download successful.")
        return True
    except ImportError:
        print("  WILDS not installed. Installing...")
        os.system(f'"{sys.executable}" -m pip install wilds -q')
        try:
            from wilds import get_dataset
            dataset = get_dataset("waterbirds", download=True, root_dir=str(root))
            print("  WILDS download successful.")
            return True
        except Exception as e:
            print(f"  WILDS download failed: {e}")
            return False
    except Exception as e:
        print(f"  WILDS download failed: {e}")
        return False


def verify_dataset(root):
    """Verify the downloaded dataset has the expected structure."""
    root = Path(root)

    # Look for metadata.csv
    metadata_paths = list(root.rglob("metadata.csv"))
    if not metadata_paths:
        print("  [!] metadata.csv not found!")
        return False

    metadata_path = metadata_paths[0]
    print(f"  Found metadata: {metadata_path}")

    # Count lines
    with open(metadata_path, "r") as f:
        lines = f.readlines()

    num_samples = len(lines) - 1  # minus header
    print(f"  Total samples: {num_samples}")

    # Check for images
    img_dir = metadata_path.parent
    img_count = sum(1 for _ in img_dir.rglob("*.jpg"))
    img_count += sum(1 for _ in img_dir.rglob("*.png"))
    print(f"  Image files found: {img_count}")

    if img_count > 0 and num_samples > 0:
        print("  [OK] Dataset verification passed!")
        return True
    else:
        print("  [!] Dataset seems incomplete.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Waterbirds dataset")
    parser.add_argument(
        "--root", type=str, default="data/waterbirds",
        help="Root directory for the dataset",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Waterbirds Dataset Downloader")
    print("=" * 60)

    root = Path(args.root)

    # Check if already downloaded
    if list(root.rglob("metadata.csv")):
        print(f"\n  Dataset may already exist at {root}")
        if verify_dataset(str(root)):
            print("  Skipping download.")
            return

    # Try direct download first
    print("\n  Attempting direct download...")
    if try_direct_download(str(root)):
        if verify_dataset(str(root)):
            print(f"\n  Dataset ready at: {root.resolve()}")
            return

    # Try WILDS fallback
    if try_wilds_download(str(root)):
        if verify_dataset(str(root)):
            print(f"\n  Dataset ready at: {root.resolve()}")
            return

    # If all else fails
    print("\n" + "=" * 60)
    print("  [!] Automatic download failed.")
    print("  Please download manually:")
    print("  1. Go to: https://github.com/kohpangwei/group_DRO")
    print("  2. Download waterbird_complete95_forest2water2.tar.gz")
    print(f"  3. Extract to: {root.resolve()}")
    print("  4. Verify metadata.csv exists in the extracted folder")
    print("=" * 60)


if __name__ == "__main__":
    main()
