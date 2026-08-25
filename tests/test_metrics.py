import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import (  # noqa: E402
    direction_from_threshold,
    directional_metrics,
    score_from_pred,
    strategy_return,
)
from utils.adjusted_pred_analysis import (  # noqa: E402
    _build_zero_threshold_cumulative,
    _direction_color,
    build_paired_decile_cumulative_returns,
    build_zero_threshold_metric_rows,
    equal_count_bins,
    zero_threshold_strategy_returns,
)


class MetricsTests(unittest.TestCase):
    def test_mixed_bin_color_follows_majority_direction(self):
        self.assertEqual(_direction_color(0.98), "#3498f0")
        self.assertEqual(_direction_color(0.50), "#3498f0")
        self.assertEqual(_direction_color(0.072059), "#f18436")

    def test_zero_threshold_strategy_is_sample_level(self):
        actual = zero_threshold_strategy_returns(
            np.array([-0.10, 0.00, 0.20]),
            np.array([-0.02, 0.03, -0.04]),
        )
        np.testing.assert_allclose(actual, [0.02, 0.03, -0.04])

    def test_equal_count_bins_do_not_define_direction(self):
        values = np.array([-0.03, -0.02, -0.01, 0.00, 0.01, 0.02])
        bins = equal_count_bins(values, 3)
        np.testing.assert_array_equal(bins, [0, 0, 1, 1, 2, 2])
        # 中间档同时包含负值与零值，方向仍必须逐条由pred符号决定。
        returns = zero_threshold_strategy_returns(values, np.ones(6))
        np.testing.assert_array_equal(returns, [-1, -1, -1, 1, 1, 1])

    def test_paired_return_equal_weights_two_legs_then_adds(self):
        pred_levels = np.linspace(-0.09, 0.09, 10)
        tags = np.zeros(20)
        tags[[0, 10]] = -0.02
        tags[[9, 19]] = 0.04
        data = pl.DataFrame(
            {
                "Date": ["2026-01-01"] * 10 + ["2026-01-02"] * 10,
                "pred_value": np.tile(pred_levels, 2),
                "tag": tags,
            }
        ).with_columns(pl.col("Date").str.to_date())
        curves, summary = build_paired_decile_cumulative_returns(
            data,
            variant="测试",
            scope="总体",
        )
        expected_final = 0.06
        self.assertAlmostEqual(summary["期末累计收益率"][0], expected_final)
        self.assertEqual(summary.height, 5)
        self.assertEqual(curves["配对"].n_unique(), 5)
        first_pair = curves.filter(pl.col("配对") == "Q01空 + Q10多").sort("Date")
        np.testing.assert_allclose(first_pair["配对日收益"], [0.03, 0.03])
        np.testing.assert_allclose(
            first_pair["累计收益率"],
            [0.03, expected_final],
        )

    def test_paired_return_keeps_cash_when_one_leg_is_missing(self):
        data = pl.DataFrame(
            {
                "Date": ["2026-01-01"] * 5 + ["2026-01-02"] * 5,
                "pred_value": np.arange(10, dtype=float),
                "tag": [-0.02] * 5 + [0.04] * 5,
            }
        ).with_columns(pl.col("Date").str.to_date())
        curves, _ = build_paired_decile_cumulative_returns(
            data,
            variant="测试",
            scope="总体",
        )
        first_pair = curves.filter(pl.col("配对") == "Q01空 + Q10多").sort("Date")
        np.testing.assert_allclose(first_pair["配对日收益"], [0.01, 0.02])
        np.testing.assert_allclose(first_pair["累计收益率"], [0.01, 0.03])

    def test_zero_threshold_cumulative_is_simple_and_fills_missing_days(self):
        data = pl.DataFrame(
            {
                "Date": ["2026-01-01"] * 10 + ["2026-01-02"] * 10,
                "pred_value": np.arange(20, dtype=float),
                "tag": [0.02, 0.04] + [0.0] * 18,
            }
        ).with_columns(pl.col("Date").str.to_date())
        curves, summary = _build_zero_threshold_cumulative(data, scope="总体")
        first_bin = curves.filter(pl.col("档位") == "Q01").sort("Date")
        np.testing.assert_allclose(first_bin["当日等权策略收益"], [0.03, 0.0])
        np.testing.assert_allclose(first_bin["累计收益率"], [0.03, 0.03])
        self.assertAlmostEqual(summary["期末累计收益率"][0], 0.03)

    def test_daily_metrics_use_median_counts_and_expectancy_excludes_zero_tags(self):
        pred_levels = np.linspace(-0.10, 0.10, 20)
        tags = np.zeros(60)
        tags[[0, 20, 40]] = [-0.02, 0.01, 0.0]
        dates = (
            ["2026-01-01"] * 20
            + ["2026-01-02"] * 20
            + ["2026-01-03"] * 20
        )
        # Q01三笔交易分布为[2, 1, 0]，验证无交易日按0计入中位数。
        dates[20] = "2026-01-01"
        dates[40] = "2026-01-02"
        data = pl.DataFrame(
            {
                "Date": dates,
                "pred_value": np.tile(pred_levels, 3),
                "tag": tags,
            }
        ).with_columns(pl.col("Date").str.to_date())
        metrics, _ = build_zero_threshold_metric_rows(
            data,
            variant="测试",
            scope="总体",
        )
        first_bin = metrics.row(0, named=True)
        self.assertEqual(first_bin["样本量"], 3)
        self.assertEqual(first_bin["交易日数"], 3)
        self.assertAlmostEqual(first_bin["每日开仓次数中位数"], 1.0)
        self.assertAlmostEqual(first_bin["平均策略收益"], (0.02 - 0.01) / 3.0)
        self.assertAlmostEqual(
            first_bin["平均每日收益"],
            first_bin["平均策略收益"] * first_bin["每日开仓次数中位数"],
        )
        self.assertAlmostEqual(first_bin["胜率_排除Tag零值"], 0.5)
        self.assertAlmostEqual(first_bin["平均盈亏比"], 2.0)
        self.assertAlmostEqual(first_bin["期望R倍数"], 0.5)

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
