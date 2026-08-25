from __future__ import annotations

"""步骤三：比较调整前后Pred，并生成零阈值收益分析图。

输入为原始Parquet和最终零中心Parquet。两者使用完全相同的1,667,520条
有效样本。零阈值指标逐样本由pred符号定方向；配对累计收益固定低档做空、
高档做多，并将两腿日均收益各占50%后按交易日单利累计。
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
    str(Path(tempfile.gettempdir()) / "pred_returns_matplotlib"),
)

from utils.adjusted_pred_analysis import analyze_adjusted_pred_returns


# 输入配置
CONFIG_PATH = PROJECT_ROOT / "config.json"
ADJUSTED_INPUT_PARQUET_PATH = (
    PROJECT_ROOT / "result" / "data" / "pred_eval_segmented_drift_adjusted.parquet"
)
# 原始Parquet路径、字段名和14:30切点统一从CONFIG_PATH读取。
EXPECTED_SAMPLE_COUNT = 1_667_520
EXPECTED_BEFORE_1430_COUNT = 1_459_080
EXPECTED_AFTER_1430_COUNT = 208_440

# 输出配置：调整前后分别输出三张六联图、三张零阈值单利累计收益图、
# 三张配对单利累计收益图及三张独立Pred分布图。
FIGURE_DIR = PROJECT_ROOT / "result" / "visualizations"
BEFORE_OVERALL_METRIC_FIGURE_PATH = (
    FIGURE_DIR / "before_adjustment" / "overall" / "overall_ventile_zero_threshold_metrics.png"
)
BEFORE_BEFORE_1430_METRIC_FIGURE_PATH = (
    FIGURE_DIR / "before_adjustment" / "before_1430" / "before_1430_ventile_zero_threshold_metrics.png"
)
BEFORE_AFTER_1430_METRIC_FIGURE_PATH = (
    FIGURE_DIR / "before_adjustment" / "after_1430" / "after_1430_ventile_zero_threshold_metrics.png"
)
ADJUSTED_OVERALL_METRIC_FIGURE_PATH = (
    FIGURE_DIR / "after_adjustment" / "overall" / "overall_ventile_zero_threshold_metrics.png"
)
ADJUSTED_BEFORE_1430_METRIC_FIGURE_PATH = (
    FIGURE_DIR / "after_adjustment" / "before_1430" / "before_1430_ventile_zero_threshold_metrics.png"
)
ADJUSTED_AFTER_1430_METRIC_FIGURE_PATH = (
    FIGURE_DIR / "after_adjustment" / "after_1430" / "after_1430_ventile_zero_threshold_metrics.png"
)
BEFORE_OVERALL_CUMULATIVE_FIGURE_PATH = (
    FIGURE_DIR / "before_adjustment" / "overall" / "overall_decile_zero_threshold_cumulative_return.png"
)
BEFORE_BEFORE_1430_CUMULATIVE_FIGURE_PATH = (
    FIGURE_DIR / "before_adjustment" / "before_1430" / "before_1430_decile_zero_threshold_cumulative_return.png"
)
BEFORE_AFTER_1430_CUMULATIVE_FIGURE_PATH = (
    FIGURE_DIR / "before_adjustment" / "after_1430" / "after_1430_decile_zero_threshold_cumulative_return.png"
)
ADJUSTED_OVERALL_CUMULATIVE_FIGURE_PATH = (
    FIGURE_DIR / "after_adjustment" / "overall" / "overall_decile_zero_threshold_cumulative_return.png"
)
ADJUSTED_BEFORE_1430_CUMULATIVE_FIGURE_PATH = (
    FIGURE_DIR / "after_adjustment" / "before_1430" / "before_1430_decile_zero_threshold_cumulative_return.png"
)
ADJUSTED_AFTER_1430_CUMULATIVE_FIGURE_PATH = (
    FIGURE_DIR / "after_adjustment" / "after_1430" / "after_1430_decile_zero_threshold_cumulative_return.png"
)
BEFORE_OVERALL_PAIRED_CUMULATIVE_FIGURE_PATH = (
    FIGURE_DIR / "before_adjustment" / "overall" / "paired_decile_cumulative_return.png"
)
BEFORE_BEFORE_1430_PAIRED_CUMULATIVE_FIGURE_PATH = (
    FIGURE_DIR / "before_adjustment" / "before_1430" / "paired_decile_cumulative_return.png"
)
BEFORE_AFTER_1430_PAIRED_CUMULATIVE_FIGURE_PATH = (
    FIGURE_DIR / "before_adjustment" / "after_1430" / "paired_decile_cumulative_return.png"
)
ADJUSTED_OVERALL_PAIRED_CUMULATIVE_FIGURE_PATH = (
    FIGURE_DIR / "after_adjustment" / "overall" / "paired_decile_cumulative_return.png"
)
ADJUSTED_BEFORE_1430_PAIRED_CUMULATIVE_FIGURE_PATH = (
    FIGURE_DIR / "after_adjustment" / "before_1430" / "paired_decile_cumulative_return.png"
)
ADJUSTED_AFTER_1430_PAIRED_CUMULATIVE_FIGURE_PATH = (
    FIGURE_DIR / "after_adjustment" / "after_1430" / "paired_decile_cumulative_return.png"
)
BEFORE_OVERALL_DISTRIBUTION_FIGURE_PATH = (
    FIGURE_DIR / "before_adjustment" / "overall" / "pred_distribution.png"
)
BEFORE_BEFORE_1430_DISTRIBUTION_FIGURE_PATH = (
    FIGURE_DIR / "before_adjustment" / "before_1430" / "pred_distribution.png"
)
BEFORE_AFTER_1430_DISTRIBUTION_FIGURE_PATH = (
    FIGURE_DIR / "before_adjustment" / "after_1430" / "pred_distribution.png"
)
ADJUSTED_OVERALL_DISTRIBUTION_FIGURE_PATH = (
    FIGURE_DIR / "after_adjustment" / "overall" / "pred_distribution.png"
)
ADJUSTED_BEFORE_1430_DISTRIBUTION_FIGURE_PATH = (
    FIGURE_DIR / "after_adjustment" / "before_1430" / "pred_distribution.png"
)
ADJUSTED_AFTER_1430_DISTRIBUTION_FIGURE_PATH = (
    FIGURE_DIR / "after_adjustment" / "after_1430" / "pred_distribution.png"
)


def main() -> None:
    """读取公共配置，以完整命名参数执行调整前后收益分析。"""

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    raw_input_parquet_path = PROJECT_ROOT / config["source_path"]

    result = analyze_adjusted_pred_returns(
        raw_input_path=raw_input_parquet_path,
        adjusted_input_path=ADJUSTED_INPUT_PARQUET_PATH,
        before_overall_metric_figure_path=BEFORE_OVERALL_METRIC_FIGURE_PATH,
        before_before_1430_metric_figure_path=BEFORE_BEFORE_1430_METRIC_FIGURE_PATH,
        before_after_1430_metric_figure_path=BEFORE_AFTER_1430_METRIC_FIGURE_PATH,
        adjusted_overall_metric_figure_path=ADJUSTED_OVERALL_METRIC_FIGURE_PATH,
        adjusted_before_1430_metric_figure_path=ADJUSTED_BEFORE_1430_METRIC_FIGURE_PATH,
        adjusted_after_1430_metric_figure_path=ADJUSTED_AFTER_1430_METRIC_FIGURE_PATH,
        before_overall_cumulative_figure_path=BEFORE_OVERALL_CUMULATIVE_FIGURE_PATH,
        before_before_1430_cumulative_figure_path=BEFORE_BEFORE_1430_CUMULATIVE_FIGURE_PATH,
        before_after_1430_cumulative_figure_path=BEFORE_AFTER_1430_CUMULATIVE_FIGURE_PATH,
        adjusted_overall_cumulative_figure_path=ADJUSTED_OVERALL_CUMULATIVE_FIGURE_PATH,
        adjusted_before_1430_cumulative_figure_path=ADJUSTED_BEFORE_1430_CUMULATIVE_FIGURE_PATH,
        adjusted_after_1430_cumulative_figure_path=ADJUSTED_AFTER_1430_CUMULATIVE_FIGURE_PATH,
        before_overall_paired_cumulative_figure_path=BEFORE_OVERALL_PAIRED_CUMULATIVE_FIGURE_PATH,
        before_before_1430_paired_cumulative_figure_path=BEFORE_BEFORE_1430_PAIRED_CUMULATIVE_FIGURE_PATH,
        before_after_1430_paired_cumulative_figure_path=BEFORE_AFTER_1430_PAIRED_CUMULATIVE_FIGURE_PATH,
        adjusted_overall_paired_cumulative_figure_path=ADJUSTED_OVERALL_PAIRED_CUMULATIVE_FIGURE_PATH,
        adjusted_before_1430_paired_cumulative_figure_path=ADJUSTED_BEFORE_1430_PAIRED_CUMULATIVE_FIGURE_PATH,
        adjusted_after_1430_paired_cumulative_figure_path=ADJUSTED_AFTER_1430_PAIRED_CUMULATIVE_FIGURE_PATH,
        before_overall_distribution_figure_path=BEFORE_OVERALL_DISTRIBUTION_FIGURE_PATH,
        before_before_1430_distribution_figure_path=BEFORE_BEFORE_1430_DISTRIBUTION_FIGURE_PATH,
        before_after_1430_distribution_figure_path=BEFORE_AFTER_1430_DISTRIBUTION_FIGURE_PATH,
        adjusted_overall_distribution_figure_path=ADJUSTED_OVERALL_DISTRIBUTION_FIGURE_PATH,
        adjusted_before_1430_distribution_figure_path=ADJUSTED_BEFORE_1430_DISTRIBUTION_FIGURE_PATH,
        adjusted_after_1430_distribution_figure_path=ADJUSTED_AFTER_1430_DISTRIBUTION_FIGURE_PATH,
        datetime_column=config["datetime_column"],
        pred_column=config["pred_column"],
        tag_column=config["return_column"],
        time_cutoff=config["time_cutoff"],
        expected_sample_count=EXPECTED_SAMPLE_COUNT,
        expected_before_1430_count=EXPECTED_BEFORE_1430_COUNT,
        expected_after_1430_count=EXPECTED_AFTER_1430_COUNT,
    )

    print("\n步骤三完成：调整前后六指标、单利累计收益及Pred分布分析")
    print(f"调整前输入：{raw_input_parquet_path}")
    print(f"调整后输入：{ADJUSTED_INPUT_PARQUET_PATH}")
    print(f"每个状态的同期间样本量：{result.sample_count_per_variant:,}")
    for key in result.short_rates:
        print(
            f"{key}：做空占比={result.short_rates[key]:.2%}，"
            f"零阈值胜率={result.hit_rates[key]:.2%}"
        )
    for path in result.figure_paths:
        print(f"输出图片：{path}")


if __name__ == "__main__":
    main()
