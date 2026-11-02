#!/usr/bin/env python3
"""
Extract Waterbirds dataset from HuggingFace parquet files
into the standard file structure expected by CVA.

Creates:
  data/waterbirds/waterbird_complete95_forest2water2/
    metadata.csv
    images/
      train/  (jpg files)
      val/    (jpg files)  
      test/   (jpg files)
"""

import io
import os
import csv
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def extract_split(parquet_path, split_name, split_idx, output_dir, metadata_rows):
    """Extract images from a parquet file for one split."""
    print(f"  Reading {parquet_path}...")
    table = pq.read_table(parquet_path)
    n = table.num_rows
    print(f"  {split_name}: {n} samples")

    img_dir = output_dir / "images" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}

        # Extract image
        img_data = row["image"]["bytes"]
        img_filename = row["image"]["path"] or f"{split_name}_{i:05d}.jpg"

        # Save image
        img_path = img_dir / img_filename
        if not img_path.exists():
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            img.save(img_path, "JPEG", quality=95)

        # Build metadata row
        label = row.get("label", row.get("bird", 0))
        place = row.get("place", 0)
        rel_path = f"images/{split_name}/{img_filename}"

        metadata_rows.append({
            "img_id": len(metadata_rows),
            "img_filename": rel_path,
            "y": label,
            "place": place,
            "split": split_idx,
        })

        if (i + 1) % 500 == 0:
            print(f"    Extracted {i+1}/{n} images...")

    print(f"  {split_name} done: {n} images extracted.")


def main():
    base_dir = Path("data/waterbirds/waterbird_complete95_forest2water2")
    data_dir = base_dir / "data"
    output_dir = base_dir

    # Map split files
    split_files = {
        "train": ("train-00000-of-00001.parquet", 0),
        "val": ("validation-00000-of-00001.parquet", 1),
        "test": ("test-00000-of-00001.parquet", 2),
    }

    print("=" * 60)
    print("  Extracting Waterbirds dataset from parquet files")
    print("=" * 60)

    metadata_rows = []

    for split_name, (filename, split_idx) in split_files.items():
        pq_path = data_dir / filename
        if not pq_path.exists():
            print(f"  [!] Missing: {pq_path}")
            continue
        extract_split(pq_path, split_name, split_idx, output_dir, metadata_rows)

    # Write metadata.csv
    metadata_path = output_dir / "metadata.csv"
    print(f"\n  Writing metadata.csv ({len(metadata_rows)} rows)...")
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["img_id", "img_filename", "y", "place", "split"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"  Metadata saved to: {metadata_path}")

    # Summary
    from collections import Counter
    splits = Counter(r["split"] for r in metadata_rows)
    groups = Counter((r["y"], r["place"]) for r in metadata_rows)
    print(f"\n  Split counts: train={splits[0]}, val={splits[1]}, test={splits[2]}")
    print(f"  Group counts:")
    group_names = {
        (0, 0): "Landbird on Land",
        (0, 1): "Landbird on Water",
        (1, 0): "Waterbird on Land",
        (1, 1): "Waterbird on Water",
    }
    for (y, p), count in sorted(groups.items()):
        name = group_names.get((y, p), f"y={y}, place={p}")
        print(f"    {name}: {count}")

    print(f"\n  Dataset ready at: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
