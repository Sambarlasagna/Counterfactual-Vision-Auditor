#!/usr/bin/env python3
"""
CVA Environment Checker

Reports Python version, PyTorch installation, CUDA availability,
GPU details, and key package versions.

Usage:
    python scripts/check_environment.py
"""

import sys
import platform


def _header(title: str) -> str:
    """Create an ASCII section header."""
    return f"\n--- {title} " + "-" * (55 - len(title))


def check_environment():
    """Run all environment checks and print a report."""

    print("=" * 60)
    print("  Counterfactual Vision Auditor -- Environment Check")
    print("=" * 60)

    # -- System --
    print(_header("System"))
    print(f"  OS:            {platform.system()} {platform.release()}")
    print(f"  Architecture:  {platform.machine()}")
    print(f"  Python:        {sys.version}")
    print(f"  Executable:    {sys.executable}")

    # -- PyTorch --
    print(_header("PyTorch"))
    try:
        import torch
        print(f"  PyTorch:       {torch.__version__}")
        print(f"  CUDA available:{torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version:  {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  GPU count:     {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / (1024 ** 3)
                print(f"  GPU {i}:         {props.name} ({mem_gb:.1f} GB)")
        else:
            print("  [!] No CUDA GPU detected. Training will use CPU (slow).")
    except ImportError:
        print("  [X] PyTorch is NOT installed!")
        return False

    # -- torchvision --
    print(_header("torchvision"))
    try:
        import torchvision
        print(f"  torchvision:   {torchvision.__version__}")
    except ImportError:
        print("  [X] torchvision is NOT installed!")

    # -- timm --
    try:
        import timm
        print(f"  timm:          {timm.__version__}")
    except ImportError:
        print("  timm:          not installed (optional, for ViT)")

    # -- Key packages --
    print(_header("Key Packages"))
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "Pillow": "PIL",
        "PyYAML": "yaml",
        "tqdm": "tqdm",
    }
    for display_name, import_name in packages.items():
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  {display_name:15s} {ver}")
        except ImportError:
            print(f"  {display_name:15s} [X] NOT installed")

    # -- Quick CUDA test --
    if torch.cuda.is_available():
        print(_header("CUDA Smoke Test"))
        try:
            x = torch.randn(2, 3, 224, 224, device="cuda")
            y = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            print(f"  Tensor on CUDA: OK ({x.device})")
            print(f"  Pooling result: OK shape={y.shape}")
            del x, y
            torch.cuda.empty_cache()
            print("  CUDA memory freed: OK")
        except Exception as e:
            print(f"  [X] CUDA test failed: {e}")

    print("\n" + "=" * 60)
    print("  Environment check complete.")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
