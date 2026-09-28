import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import (  # noqa: E402
    direction_from_threshold,
    directional_metrics,
    score_from_pred,
    strategy_return,
)


class MetricsTests(unittest.TestCase):
    def test_score_offset_and_clip(self):
        actual = score_from_pred([-1.0, 0.0, 0.7])
        np.testing.assert_allclose(actual, [0.0, 0.5, 1.0])

    def test_threshold_equality_belongs_to_long(self):
        actual = direction_from_threshold([0.48, 0.50, 0.52], 0.50)
        np.testing.assert_array_equal(actual, [-1, 1, 1])

    def test_strategy_return_reverses_short_tag(self):
        actual = strategy_return([-1, 1], [-0.02, 0.03])
        np.testing.assert_allclose(actual, [0.02, 0.03])

    def test_directional_metrics(self):
        metrics = directional_metrics([-1, -1, 1, 1], [-0.02, 0.01, 0.03, -0.04])
        self.assertAlmostEqual(metrics["short_precision"], 0.5)
        self.assertAlmostEqual(metrics["long_precision"], 0.5)
        self.assertAlmostEqual(metrics["direction_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["mean_strategy_return"], 0.0)


if __name__ == "__main__":
    unittest.main()
