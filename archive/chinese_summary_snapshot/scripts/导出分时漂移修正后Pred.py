from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pipelines import run_zero_centered_pred_export


if __name__ == "__main__":
    run_zero_centered_pred_export()
