"""Path configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data"
MODELS_DIR = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"
