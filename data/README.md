# Dataset Instructions

## Waterbirds (Primary)

The Waterbirds benchmark is used to study spurious correlations between bird type
(landbird / waterbird) and background (land / water).

### Download

The dataset can be obtained via the WILDS package:

```bash
pip install wilds
python -c "from wilds import get_dataset; get_dataset('waterbirds', download=True, root_dir='data/waterbirds')"
```

Or manually from the original sources. See:
- https://wilds.stanford.edu/
- https://github.com/kohpangwei/group_DRO

### Expected Structure

After download, the `data/waterbirds/` directory should contain:
- Image files organized by split
- Metadata CSV with columns: `img_filename`, `y` (label), `place` (background), `split`

### Groups

| Group | Bird Type  | Background |
|-------|-----------|------------|
| 0     | Landbird  | Land       |
| 1     | Landbird  | Water      |
| 2     | Waterbird | Land       |
| 3     | Waterbird | Water      |

Groups 1 and 2 are the minority groups (mismatched bird/background).

## CUB-200-2011 (Fallback)

If Waterbirds is unavailable, CUB-200-2011 can be used with synthetic group assignments.

Download: https://www.vision.caltech.edu/datasets/cub_200_2011/

---

> **Note:** Raw dataset files are NOT committed to this repository.
> See `.gitignore` for excluded patterns.
