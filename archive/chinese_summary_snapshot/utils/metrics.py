from __future__ import annotations

import numpy as np


def score_from_pred(pred, offset: float = 0.5):
    """Convert the centered model output to a clipped 0-1 score."""
    return np.clip(np.asarray(pred, dtype=float) + offset, 0.0, 1.0)


def direction_from_threshold(score, threshold):
    """Return -1 for short and +1 for long; equality belongs to long."""
    score_array = np.asarray(score, dtype=float)
    threshold_array = np.asarray(threshold, dtype=float)
    return np.where(score_array < threshold_array, -1, 1).astype(np.int8)


def strategy_return(direction, tag):
    """Long earns Tag and short earns -Tag."""
    return np.asarray(direction, dtype=float) * np.asarray(tag, dtype=float)


def directional_metrics(direction, tag):
    """Compute the project's canonical direction, precision and recall metrics."""
    direction_array = np.asarray(direction)
    tag_array = np.asarray(tag, dtype=float)
    short = direction_array == -1
    long = direction_array == 1
    negative = tag_array < 0
    positive = tag_array > 0
    short_wins = int(np.sum(short & negative))
    long_wins = int(np.sum(long & positive))

    def rate(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    n = len(tag_array)
    return {
        "sample_n": n,
        "short_share": rate(int(np.sum(short)), n),
        "long_share": rate(int(np.sum(long)), n),
        "short_precision": rate(short_wins, int(np.sum(short))),
        "long_precision": rate(long_wins, int(np.sum(long))),
        "short_recall": rate(short_wins, int(np.sum(negative))),
        "long_recall": rate(long_wins, int(np.sum(positive))),
        "direction_accuracy": rate(short_wins + long_wins, n),
        "mean_strategy_return": float(np.mean(strategy_return(direction_array, tag_array))),
    }
