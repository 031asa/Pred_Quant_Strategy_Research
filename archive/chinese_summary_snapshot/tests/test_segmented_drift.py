import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.segmented_drift import (  # noqa: E402
    SEGMENT_AFTER,
    SEGMENT_BEFORE,
    build_causal_state,
    build_independent_segment_states,
    build_selected_daily_states,
    directional_hit_rate,
    direction_from_slow_baseline,
    segment_from_clock,
)


class SegmentedDriftTests(unittest.TestCase):
    def test_active_scripts_have_no_csv_or_excel_interfaces(self):
        scripts = [
            PROJECT_ROOT / "scripts" / "总体Score时序漂移.py",
            PROJECT_ROOT / "scripts" / "导出分时漂移修正后Pred.py",
            PROJECT_ROOT / "scripts" / "修正后Pred分组收益.py",
        ]
        forbidden = ("read_csv", "write_csv", ".csv", ".xlsx")
        for script in scripts:
            source = script.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{script.name} contains {token}")

    def test_active_scripts_are_thin_entrypoints(self):
        scripts = [
            PROJECT_ROOT / "scripts" / "总体Score时序漂移.py",
            PROJECT_ROOT / "scripts" / "导出分时漂移修正后Pred.py",
            PROJECT_ROOT / "scripts" / "修正后Pred分组收益.py",
        ]
        for script in scripts:
            source = script.read_text(encoding="utf-8")
            self.assertLessEqual(len(source.splitlines()), 30, script.name)
            self.assertNotIn("\ndef ", source, script.name)
            self.assertIn("from utils.pipelines import", source, script.name)

    def test_cutoff_equality_belongs_to_later_segment(self):
        self.assertEqual(segment_from_clock(14, 29), SEGMENT_BEFORE)
        self.assertEqual(segment_from_clock(14, 30), SEGMENT_AFTER)
        self.assertEqual(segment_from_clock(15, 0), SEGMENT_AFTER)

    def test_current_day_does_not_change_its_own_state(self):
        original = np.array([0.48, 0.49, 0.47, 0.50, 0.46])
        changed = original.copy()
        changed[3] = 0.99
        original_state = build_causal_state(original, cold_start_days=2, half_life_days=1)
        changed_state = build_causal_state(changed, cold_start_days=2, half_life_days=1)
        self.assertEqual(original_state["slow_baseline"][3], changed_state["slow_baseline"][3])
        self.assertEqual(original_state["recent_center"][3], changed_state["recent_center"][3])

    def test_segment_states_are_independent(self):
        before = np.array([0.48, 0.49, 0.47, 0.50, 0.46])
        after = np.array([0.54, 0.55, 0.53, 0.56, 0.52])
        initial = build_independent_segment_states(
            {SEGMENT_BEFORE: before, SEGMENT_AFTER: after},
            cold_start_days=2,
            half_life_days_by_segment={SEGMENT_BEFORE: 1, SEGMENT_AFTER: 2},
        )
        changed_before = before.copy()
        changed_before[:3] += 0.20
        changed = build_independent_segment_states(
            {SEGMENT_BEFORE: changed_before, SEGMENT_AFTER: after},
            cold_start_days=2,
            half_life_days_by_segment={SEGMENT_BEFORE: 1, SEGMENT_AFTER: 2},
        )
        np.testing.assert_allclose(
            initial[SEGMENT_AFTER]["slow_baseline"],
            changed[SEGMENT_AFTER]["slow_baseline"],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            initial[SEGMENT_AFTER]["recent_center"],
            changed[SEGMENT_AFTER]["recent_center"],
            equal_nan=True,
        )
        self.assertFalse(
            np.allclose(
                initial[SEGMENT_BEFORE]["slow_baseline"][2:],
                changed[SEGMENT_BEFORE]["slow_baseline"][2:],
            )
        )

    def test_slow_baseline_equality_is_long(self):
        actual = direction_from_slow_baseline(
            [0.49, 0.50, 0.51],
            [0.50, 0.50, 0.50],
        )
        np.testing.assert_array_equal(actual, [-1, 1, 1])

    def test_zero_tag_is_excluded_from_hit_rate(self):
        actual = directional_hit_rate([-1, 1, -1], [-0.1, 0.0, 0.2])
        self.assertEqual(actual, 0.5)

    def test_all_zero_tags_have_no_hit_rate(self):
        self.assertIsNone(directional_hit_rate([-1, 1], [0.0, 0.0]))

    def test_selected_state_rebuild_needs_no_csv(self):
        dates = [f"2024-01-{day:02d}" for day in range(1, 6)]
        samples = pl.DataFrame(
            {
                "Date": dates + dates,
                "时段": [SEGMENT_BEFORE] * 5 + [SEGMENT_AFTER] * 5,
                "score": [0.48, 0.49, 0.47, 0.50, 0.46, 0.54, 0.55, 0.53, 0.56, 0.52],
            }
        ).with_columns(pl.col("Date").str.to_date())
        config = {
            "cold_start_effective_trading_days": 2,
            "segments": {
                SEGMENT_BEFORE: {
                    "selected_half_life_trading_days": 1,
                    "selected_development_balance_offset": 0.001,
                },
                SEGMENT_AFTER: {
                    "selected_half_life_trading_days": 2,
                    "selected_development_balance_offset": -0.002,
                },
            },
        }
        state = build_selected_daily_states(samples, config)
        self.assertEqual(state.height, 6)
        effective = state
        self.assertEqual(
            effective.select(pl.struct(["Date", "时段"]).is_duplicated().sum()).item(),
            0,
        )


if __name__ == "__main__":
    unittest.main()
