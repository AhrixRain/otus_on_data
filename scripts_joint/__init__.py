"""Joint-resonance CMS OTUS training series."""

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
for _directory in (_REPO_ROOT / "scripts", _REPO_ROOT / "scripts_sota"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from .joint_data import load_joint_regions, resolve_joint_config
from .joint_metrics import score_joint_metrics
from .joint_model import build_joint_autoencoder
from .joint_trainer import run_joint_training

__all__ = [
    "build_joint_autoencoder",
    "load_joint_regions",
    "resolve_joint_config",
    "run_joint_training",
    "score_joint_metrics",
]
