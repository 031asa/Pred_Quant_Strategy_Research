import io
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class StreamlitAppTests(unittest.TestCase):
    def test_upload_time_selection_and_ab_experiment_smoke(self):
        start = datetime(2026, 1, 1, 10, 0)
        rows = [
            {
                "Datetime": start + timedelta(days=day, hours=hour),
                "Contract": contract,
                "pred": (day - 20) * 0.002 + (0.01 if contract == "B" else 0.0),
                "Tag": ((day + hour) % 7 - 3) * 0.001,
            }
            for day in range(40)
            for contract in ("A", "B")
            for hour in (0, 5)
        ]
        buffer = io.BytesIO()
        pl.DataFrame(rows).write_parquet(buffer)

        app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"))
        app.run(timeout=30)
        self.assertFalse(app.exception)
        app.session_state["pipelines"] = {
            "A": [
                {
                    "id": "time_a",
                    "type": "group_time",
                    "enabled": True,
                    "cutoff": "14:30",
                    "time_selection": "before",
                }
            ],
            "B": [
                {
                    "id": "time_b",
                    "type": "group_time",
                    "enabled": True,
                    "cutoff": "14:30",
                    "time_selection": "after",
                }
            ],
        }
        app.get("file_uploader")[0].upload(
            "sample.parquet",
            buffer.getvalue(),
            "application/octet-stream",
        ).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 5)
        self.assertEqual(
            [item.label for item in app.selectbox].count("数据选择"),
            2,
        )
        self.assertNotIn("图片范围", [item.label for item in app.selectbox])
        self.assertNotIn("出图切点", [item.label for item in app.get("time_input")])

        run_button = next(
            button for button in app.button if button.label == "执行 A/B 实验"
        )
        run_button.click().run(timeout=120)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("download_button")), 1)
        self.assertEqual(len(app.get("plotly_chart")), 2)
        self.assertEqual(len(app.dataframe), 1)
        layout_control = next(item for item in app.radio if item.label == "结果布局")
        self.assertEqual(layout_control.value, "上下完整")
        comparison_values = app.dataframe[0].value.astype(str).to_numpy().ravel()
        self.assertIn("80", comparison_values)
        self.assertIn("2026-01-01", comparison_values)
        result_keys = dict(app.session_state["pipeline_result_keys"])
        layout_control.set_value("左右并排").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("plotly_chart")), 2)
        self.assertEqual(app.session_state["pipeline_result_keys"], result_keys)
        self.assertEqual(app.session_state["pipeline_results"]["A"].output_rows, 80)
        self.assertEqual(app.session_state["pipeline_results"]["B"].output_rows, 80)
        self.assertEqual(
            app.session_state["pipeline_results"]["A"].output_scope_label,
            "14:30前",
        )
        self.assertEqual(
            app.session_state["pipeline_results"]["B"].output_scope_label,
            "14:30及以后",
        )


if __name__ == "__main__":
    unittest.main()
