from __future__ import annotations

"""步骤一：按 14:30 前后独立分析 Score 漂移并选择参数。

输入、输出和业务参数集中列在本文件顶部；计算与绘图实现位于 utils。
"""

import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "pred_drift_matplotlib"),
)

from utils.score_drift_workflow import analyze_segmented_score_drift


# 输入配置
CONFIG_PATH = PROJECT_ROOT / "config.json"
# 原始数据路径、字段名、14:30切点和Score换算参数统一从CONFIG_PATH读取。
COLD_START_DAYS = 20
TEST_FRACTION = 0.10
SHORT_RATE_TARGET = 0.50
SHORT_RATE_TOLERANCE = 0.01
HALF_LIFE_CANDIDATES = (1, 2, 5, 10, 20, 40, 60, 90, 120)
EXPECTED_SOURCE_SAMPLE_COUNT = 1_728_000

# 输出配置
PARAMETERS_OUTPUT_PATH = (
    PROJECT_ROOT / "result" / "parameters"
    / "segmented_score_drift_selected_parameters.json"
)
BEFORE_1430_FIGURE_PATH = (
    PROJECT_ROOT / "result" / "visualizations"
    / "drift_comparison"
    / "before_1430_monthly_score_drift.png"
)
AFTER_1430_FIGURE_PATH = (
    PROJECT_ROOT / "result" / "visualizations"
    / "drift_comparison"
    / "after_1430_monthly_score_drift.png"
)


def main() -> None:
    """核对配置并通过命名参数调用漂移分析接口。"""

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    source_parquet_path = PROJECT_ROOT / config["source_path"]

    result = analyze_segmented_score_drift(
        config_path=CONFIG_PATH,
        source_path=source_parquet_path,
        parameters_output_path=PARAMETERS_OUTPUT_PATH,
        before_1430_figure_path=BEFORE_1430_FIGURE_PATH,
        after_1430_figure_path=AFTER_1430_FIGURE_PATH,
        datetime_column=config["datetime_column"],
        pred_column=config["pred_column"],
        tag_column=config["return_column"],
        time_cutoff=config["time_cutoff"],
        score_offset=float(config["score_offset"]),
        score_min=float(config["score_min"]),
        score_max=float(config["score_max"]),
        cold_start_days=COLD_START_DAYS,
        test_fraction=TEST_FRACTION,
        short_rate_target=SHORT_RATE_TARGET,
        short_rate_tolerance=SHORT_RATE_TOLERANCE,
        half_life_candidates=HALF_LIFE_CANDIDATES,
        expected_source_sample_count=EXPECTED_SOURCE_SAMPLE_COUNT,
    )

    print("\n步骤一完成：分时 Score 漂移选参")
    print(f"输入 Parquet：{source_parquet_path}")
    print(f"源样本量：{result.source_sample_count:,}")
    print(f"有效样本量：{result.effective_sample_count:,}")
    for segment, half_life in result.selected_half_life_by_segment.items():
        short_share = result.development_short_share_by_segment[segment]
        print(f"{segment}：半衰期={half_life}日，开发段做空占比={short_share:.4%}")
    print(f"参数 JSON：{result.parameters_path}")
    for path in result.figure_paths:
        print(f"漂移图：{path}")


if __name__ == "__main__":
    main()
