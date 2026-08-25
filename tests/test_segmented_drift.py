import ast
import json
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
    def test_return_script_declares_all_split_outputs(self):
        script = PROJECT_ROOT / "scripts" / "analyze_adjusted_pred_returns.py"
        tree = ast.parse(script.read_text(encoding="utf-8"))
        assigned_names = {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        paired_paths = {
            name
            for name in assigned_names
            if name.endswith("_PAIRED_CUMULATIVE_FIGURE_PATH")
        }
        distribution_paths = {
            name
            for name in assigned_names
            if name.endswith("_DISTRIBUTION_FIGURE_PATH")
        }
        cumulative_paths = {
            name
            for name in assigned_names
            if name.endswith("_CUMULATIVE_FIGURE_PATH")
            and "PAIRED" not in name
        }
        self.assertEqual(len(paired_paths), 6)
        self.assertEqual(len(distribution_paths), 6)
        self.assertEqual(len(cumulative_paths), 6)

    def test_data_configuration_has_single_source(self):
        config = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))
        expected_keys = {
            "source_path",
            "datetime_column",
            "pred_column",
            "return_column",
            "time_cutoff",
            "score_offset",
            "score_min",
            "score_max",
        }
        self.assertTrue(expected_keys <= config.keys())
        self.assertNotIn("rolling_days", config)

        duplicated_constants = {
            "SOURCE_PARQUET_PATH",
            "DATETIME_COLUMN",
            "PRED_COLUMN",
            "TAG_COLUMN",
            "TIME_CUTOFF",
            "SCORE_OFFSET",
            "SCORE_MIN",
            "SCORE_MAX",
        }
        for script in (PROJECT_ROOT / "scripts").glob("*.py"):
            tree = ast.parse(script.read_text(encoding="utf-8"))
            assigned_names = {
                target.id
                for node in tree.body
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            self.assertFalse(assigned_names & duplicated_constants, script.name)

    def test_active_scripts_have_no_csv_or_excel_interfaces(self):
        scripts = [
            PROJECT_ROOT / "scripts" / "analyze_segmented_score_drift.py",
            PROJECT_ROOT / "scripts" / "export_zero_centered_pred.py",
            PROJECT_ROOT / "scripts" / "analyze_adjusted_pred_returns.py",
        ]
        forbidden = ("read_csv", "write_csv", ".csv", ".xlsx")
        for script in scripts:
            source = script.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{script.name} contains {token}")

    def test_active_scripts_expose_readable_interfaces(self):
        scripts = [
            PROJECT_ROOT / "scripts" / "analyze_segmented_score_drift.py",
            PROJECT_ROOT / "scripts" / "export_zero_centered_pred.py",
            PROJECT_ROOT / "scripts" / "analyze_adjusted_pred_returns.py",
        ]
        for script in scripts:
            source = script.read_text(encoding="utf-8")
            tree = ast.parse(source)
            functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
            self.assertEqual(functions, ["main"], script.name)

            assigned_names = {
                target.id
                for node in tree.body
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            self.assertTrue(
                any(name.endswith("_PATH") for name in assigned_names),
                f"{script.name}没有显式输入路径",
            )
            self.assertTrue(
                any("OUTPUT" in name or "FIGURE" in name for name in assigned_names),
                f"{script.name}没有显式输出路径",
            )

            main = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
            workflow_calls = [
                node
                for node in ast.walk(main)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {
                    "analyze_segmented_score_drift",
                    "export_zero_centered_pred",
                    "analyze_adjusted_pred_returns",
                }
            ]
            self.assertEqual(len(workflow_calls), 1, script.name)
            self.assertFalse(workflow_calls[0].args, f"{script.name}存在位置参数")
            self.assertGreater(len(workflow_calls[0].keywords), 5, script.name)

    def test_workflow_modules_have_no_import_time_io(self):
        modules = [
            PROJECT_ROOT / "utils" / "score_drift_workflow.py",
            PROJECT_ROOT / "utils" / "zero_centered_export.py",
            PROJECT_ROOT / "utils" / "adjusted_pred_analysis.py",
        ]
        forbidden_names = {"read_parquet", "write_parquet", "savefig", "mkdir", "open"}
        for module in modules:
            source = module.read_text(encoding="utf-8")
            self.assertNotIn("runpy", source, module.name)
            tree = ast.parse(source)
            top_level_calls = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    continue
                top_level_calls.extend(
                    child for child in ast.walk(node) if isinstance(child, ast.Call)
                )
            called_attributes = {
                call.func.attr
                for call in top_level_calls
                if isinstance(call.func, ast.Attribute)
            }
            self.assertFalse(called_attributes & forbidden_names, module.name)

    def test_active_code_has_no_csv_or_excel_interfaces(self):
        files = list((PROJECT_ROOT / "scripts").glob("*.py")) + list(
            (PROJECT_ROOT / "utils").glob("*.py")
        )
        forbidden = ("read_csv", "write_csv", ".csv", ".xlsx")
        for path in files:
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{path.name} contains {token}")

    def test_return_analysis_has_only_zero_threshold_direction(self):
        module = PROJECT_ROOT / "utils" / "adjusted_pred_analysis.py"
        source = module.read_text(encoding="utf-8")
        self.assertNotIn("慢速基准预测", source)
        self.assertNotIn("group_direction", source)
        self.assertNotIn("#8b5fbf", source)
        self.assertIn("pred<0做空、pred>=0做多", source)

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
