from __future__ import annotations

from pathlib import Path
import runpy


UTILS_DIR = Path(__file__).resolve().parent


def _run_implementation(module_filename: str) -> None:
    """Execute one self-contained workflow implementation."""
    runpy.run_path(str(UTILS_DIR / module_filename), run_name="__main__")


def run_segmented_score_drift() -> None:
    """Select segmented drift parameters and render monthly drift figures."""
    _run_implementation("score_drift_workflow.py")


def run_zero_centered_pred_export() -> None:
    """Rebuild causal state and export the final zero-centered prediction."""
    _run_implementation("zero_centered_export.py")


def run_adjusted_pred_analysis() -> None:
    """Render grouping, cumulative-return, and profit/loss diagnostics."""
    _run_implementation("adjusted_pred_analysis.py")
