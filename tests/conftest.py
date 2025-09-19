"""Pytest configuration to ensure src package is importable during tests."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
SRC_STR = str(SRC_PATH)
if SRC_PATH.exists() and SRC_STR not in sys.path:
    sys.path.insert(0, SRC_STR)
