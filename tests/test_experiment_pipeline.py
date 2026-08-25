import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_pipeline import (  # noqa: E402
    INTERNAL_PRED,
    INTERNAL_ROW_ID,
    INTERNAL_TAG,
    MODULE_GROUP_CONTRACT,
    MODULE_GROUP_TIME,
    MODULE_SMOOTH,
    PipelineValidationError,
    SchemaMapping,
    TIME_SELECTION_AFTER,
    TIME_SELECTION_BEFORE,
    TIME_SELECTION_BOTH,
    delete_pipeline_module,
    execute_pipeline,
    infer_schema_mapping,
    normalize_pipeline_modules,
    pipeline_cache_key,
    prepare_uploaded_frame,
    move_pipeline_module,
)
from utils.web_charts import (  # noqa: E402
    CHART_TYPES,
    SCOPE_AFTER,
    SCOPE_BEFORE,
    SCOPE_OVERALL,
    prepare_chart_data,
    render_experiment_chart,
)
from utils.plot_fonts import PLOTLY_FONT_STACK  # noqa: E402
from utils.plotly_charts import build_experiment_plotly_chart  # noqa: E402


def sample_source(days: int = 90) -> pl.DataFrame:
    start = datetime(2024, 1, 1)
    rows = []
    for day in range(days):
        for contract_index, contract in enumerate(("A", "B")):
            for hour in (10, 15):
                trend = 0.0007 * day
                cycle = 0.025 * np.sin(day / 6.0 + contract_index)
                rows.append(
                    {
                        "Datetime": start + timedelta(days=day, hours=hour),
                        "Contract": contract,
                        "pred": trend + cycle + (hour - 12) * 0.002,
                        "Tag": ((day + contract_index + hour) % 9 - 4) * 0.001,
                    }
                )
    return pl.DataFrame(rows)


def fixed_smooth() -> dict[str, object]:
    return {
        "type": MODULE_SMOOTH,
        "enabled": True,
        "selection_mode": "fixed",
        "fixed_half_life": 2,
        "cold_start_days": 5,
        "test_fraction": 0.10,
    }


class ExperimentPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = SchemaMapping("Datetime", "Contract", "pred", "Tag")
        cls.prepared = prepare_uploaded_frame(sample_source(), cls.mapping)

    def test_schema_is_inferred_case_insensitively(self):
        actual = infer_schema_mapping(["DATETIME", "contract", "Pred", "tag"])
        self.assertEqual(
            actual,
            {
                "datetime": "DATETIME",
                "contract": "contract",
                "pred": "Pred",
                "tag": "tag",
            },
        )

    def test_duplicate_group_module_is_rejected(self):
        with self.assertRaises(PipelineValidationError):
            normalize_pipeline_modules(
                [
                    {"type": MODULE_GROUP_CONTRACT},
                    {"type": MODULE_GROUP_CONTRACT, "enabled": False},
                ]
            )

    def test_pipeline_key_depends_on_order_but_not_ui_ids(self):
        first = [
            {"id": "one", "type": MODULE_GROUP_CONTRACT},
            {"id": "two", **fixed_smooth()},
        ]
        same = [
            {"id": "different", "type": MODULE_GROUP_CONTRACT},
            {"id": "another", **fixed_smooth()},
        ]
        reversed_modules = list(reversed(first))
        self.assertEqual(
            pipeline_cache_key("hash", self.mapping, first),
            pipeline_cache_key("hash", self.mapping, same),
        )
        self.assertNotEqual(
            pipeline_cache_key("hash", self.mapping, first),
            pipeline_cache_key("hash", self.mapping, reversed_modules),
        )

    def test_modules_can_move_and_delete_without_mutating_source(self):
        modules = [
            {"type": MODULE_GROUP_CONTRACT},
            {"type": MODULE_GROUP_TIME, "cutoff": "14:30"},
            fixed_smooth(),
        ]
        moved = move_pipeline_module(modules, 2, -1)
        self.assertEqual(
            [module["type"] for module in moved],
            [MODULE_GROUP_CONTRACT, MODULE_SMOOTH, MODULE_GROUP_TIME],
        )
        self.assertEqual(modules[1]["type"], MODULE_GROUP_TIME)
        deleted = delete_pipeline_module(moved, 0)
        self.assertEqual(
            [module["type"] for module in deleted],
            [MODULE_SMOOTH, MODULE_GROUP_TIME],
        )

    def test_group_dimensions_accumulate_for_smoothing(self):
        result = execute_pipeline(
            self.prepared,
            [
                {"type": MODULE_GROUP_CONTRACT},
                {"type": MODULE_GROUP_TIME, "cutoff": "14:30"},
                fixed_smooth(),
            ],
        )
        final_step = result.step_summaries[-1]
        self.assertEqual(len(final_step.groups), 4)
        self.assertEqual(len(final_step.active_group_dimensions), 2)
        self.assertTrue(all(group.selected_half_life == 2 for group in final_step.groups))
        self.assertEqual(result.output_scope_label, "总体（14:30前后均保留）")

    def test_time_preprocessing_filters_before_smoothing_and_labels_chart(self):
        before = execute_pipeline(
            self.prepared,
            [
                {
                    "type": MODULE_GROUP_TIME,
                    "cutoff": "14:30",
                    "time_selection": TIME_SELECTION_BEFORE,
                }
            ],
        )
        after = execute_pipeline(
            self.prepared,
            [
                {
                    "type": MODULE_GROUP_TIME,
                    "cutoff": "14:30",
                    "time_selection": TIME_SELECTION_AFTER,
                }
            ],
        )
        self.assertEqual(before.output_rows, self.prepared.height // 2)
        self.assertEqual(after.output_rows, self.prepared.height // 2)
        self.assertEqual(before.output_scope_label, "14:30前")
        self.assertEqual(after.output_scope_label, "14:30及以后")
        chart_data, chart_label = prepare_chart_data(
            before,
            scope=SCOPE_OVERALL,
            cutoff="09:00",
        )
        self.assertEqual(chart_data.height, before.output_rows)
        self.assertEqual(chart_label, "14:30前")

    def test_cutoff_equality_is_kept_by_after_selection(self):
        source = pl.DataFrame(
            {
                "Datetime": [
                    datetime(2026, 1, 2, 14, 29),
                    datetime(2026, 1, 2, 14, 30),
                ],
                "Contract": ["A", "A"],
                "pred": [-0.01, 0.01],
                "Tag": [0.001, 0.002],
            }
        )
        prepared = prepare_uploaded_frame(source, self.mapping)
        result = execute_pipeline(
            prepared,
            [
                {
                    "type": MODULE_GROUP_TIME,
                    "cutoff": "14:30",
                    "time_selection": TIME_SELECTION_AFTER,
                }
            ],
        )
        self.assertEqual(result.output_rows, 1)
        self.assertEqual(result.data["__web_datetime"][0].strftime("%H:%M"), "14:30")

    def test_invalid_time_selection_is_rejected(self):
        with self.assertRaises(PipelineValidationError):
            normalize_pipeline_modules(
                [
                    {
                        "type": MODULE_GROUP_TIME,
                        "cutoff": "14:30",
                        "time_selection": "middle",
                    }
                ]
            )

    def test_smoothing_before_preprocessing_changes_result_and_range(self):
        after_preprocessing = execute_pipeline(
            self.prepared,
            [{"type": MODULE_GROUP_CONTRACT}, fixed_smooth()],
        )
        before_and_after = execute_pipeline(
            self.prepared,
            [fixed_smooth(), {"type": MODULE_GROUP_CONTRACT}, fixed_smooth()],
        )
        self.assertLess(before_and_after.output_rows, after_preprocessing.output_rows)
        self.assertNotEqual(
            before_and_after.output_start_date,
            after_preprocessing.output_start_date,
        )

    def test_test_day_does_not_change_its_dynamic_center(self):
        original = execute_pipeline(self.prepared, [fixed_smooth()])
        last_date = self.prepared["Date"].max()
        changed_prepared = self.prepared.with_columns(
            pl.when(pl.col("Date") == last_date)
            .then(pl.col(INTERNAL_PRED) + 0.005)
            .otherwise(pl.col(INTERNAL_PRED))
            .alias(INTERNAL_PRED)
        )
        changed = execute_pipeline(changed_prepared, [fixed_smooth()])
        original_last = original.data.filter(pl.col("Date") == last_date).sort(INTERNAL_ROW_ID)
        changed_last = changed.data.filter(pl.col("Date") == last_date).sort(INTERNAL_ROW_ID)
        np.testing.assert_allclose(
            changed_last[INTERNAL_PRED].to_numpy()
            - original_last[INTERNAL_PRED].to_numpy(),
            0.005,
            atol=1e-12,
        )

    def test_row_identity_and_tag_alignment_are_preserved(self):
        result = execute_pipeline(
            self.prepared,
            [
                {"type": MODULE_GROUP_TIME, "cutoff": "14:30"},
                fixed_smooth(),
            ],
        )
        self.assertEqual(result.data[INTERNAL_ROW_ID].n_unique(), result.output_rows)
        expected = self.prepared.select([INTERNAL_ROW_ID, INTERNAL_TAG])
        aligned = result.data.select([INTERNAL_ROW_ID, INTERNAL_TAG]).join(
            expected.rename({INTERNAL_TAG: "expected_tag"}),
            on=INTERNAL_ROW_ID,
            how="left",
            validate="1:1",
        )
        np.testing.assert_allclose(aligned[INTERNAL_TAG], aligned["expected_tag"])

    def test_independent_experiments_do_not_mutate_shared_input(self):
        original_pred = self.prepared[INTERNAL_PRED].to_numpy().copy()
        raw_result = execute_pipeline(self.prepared, [])
        smoothed_result = execute_pipeline(self.prepared, [fixed_smooth()])
        np.testing.assert_allclose(self.prepared[INTERNAL_PRED], original_pred)
        np.testing.assert_allclose(raw_result.data[INTERNAL_PRED], original_pred)
        self.assertLess(smoothed_result.output_rows, raw_result.output_rows)

    def test_all_chart_types_and_scopes_return_png(self):
        result = execute_pipeline(self.prepared, [])
        for chart_type in CHART_TYPES:
            for scope in (SCOPE_OVERALL, SCOPE_BEFORE, SCOPE_AFTER):
                with self.subTest(chart_type=chart_type, scope=scope):
                    chart = render_experiment_chart(
                        result,
                        experiment_name="测试",
                        chart_type=chart_type,
                        scope=scope,
                        cutoff="14:30",
                    )
                    self.assertTrue(chart.png.startswith(b"\x89PNG\r\n\x1a\n"))
                    self.assertGreater(chart.rows, 0)

    def test_all_plotly_chart_types_use_bundled_font_stack(self):
        result = execute_pipeline(self.prepared, [])
        for chart_type in CHART_TYPES:
            with self.subTest(chart_type=chart_type):
                chart = build_experiment_plotly_chart(
                    result,
                    experiment_name="测试",
                    chart_type=chart_type,
                    scope=SCOPE_OVERALL,
                    cutoff="14:30",
                )
                self.assertEqual(chart.rows, result.output_rows)
                self.assertEqual(chart.figure.layout.font.family, PLOTLY_FONT_STACK)
                self.assertGreater(len(chart.figure.data), 0)
                if chart_type in ("cumulative", "paired"):
                    self.assertEqual(chart.figure.layout.legend.orientation, "h")
                    self.assertLessEqual(chart.figure.layout.margin.r, 35)
                    self.assertGreaterEqual(chart.figure.layout.margin.b, 165)


if __name__ == "__main__":
    unittest.main()
