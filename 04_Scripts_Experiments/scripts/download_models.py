#!/usr/bin/env python3
"""Compatibility entry point for the model registry's downloader."""
from pathlib import Path
import runpy
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "03_Models/download_models.py").is_file())
if __name__ == "__main__":
    runpy.run_path(str(ROOT / "03_Models/download_models.py"), run_name="__main__")
