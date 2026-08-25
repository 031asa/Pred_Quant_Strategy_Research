from __future__ import annotations

"""零中心 Pred 的导出实现。

本模块只提供纯入口函数和返回类型；导入模块时不会读取数据或写入文件。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from utils.segmented_drift import (
    SEGMENT_AFTER,
    SEGMENT_BEFORE,
    build_selected_daily_states,
)


@dataclass(frozen=True)
class PredExportResult:
    """最终 Parquet 的输出位置和验收摘要。"""

    output_path: Path
    sample_count: int
    pred_min: float
    pred_max: float
    short_share: float


def export_zero_centered_pred(
    *,
    source_path: Path,
    selected_parameters_path: Path,
    output_path: Path,
    date_column: str,
    datetime_column: str,
    pred_column: str,
    time_cutoff: str,
    score_offset: float,
    score_min: float,
    score_max: float,
    expected_sample_count: int,
) -> PredExportResult:
    """重建分时因果状态并导出以 0 为方向阈值的最终 Pred。"""

    source_path = Path(source_path)
    selected_parameters_path = Path(selected_parameters_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.parquet")

    with selected_parameters_path.open(encoding="utf-8") as file:
        selected_config = json.load(file)

    source = pl.read_parquet(source_path).with_row_index("__row_id")
    output_columns = [column for column in source.columns if column != "__row_id"]

    prepared = source.with_columns(
        pl.when(pl.col(datetime_column).dt.strftime("%H:%M") < time_cutoff)
        .then(pl.lit(SEGMENT_BEFORE))
        .otherwise(pl.lit(SEGMENT_AFTER))
        .alias("时段"),
        (
            (pl.col(pred_column).cast(pl.Float64) + score_offset)
            .clip(score_min, score_max)
        ).alias("score"),
    )
    state = build_selected_daily_states(
        prepared.select([date_column, "时段", "score"]).rename({date_column: "Date"}),
        selected_config,
    ).select(["Date", "时段", "slow_baseline", "effective_time_offset"])

    join_source = prepared.rename({date_column: "Date"}) if date_column != "Date" else prepared
    result = (
        join_source.join(state, on=["Date", "时段"], how="inner", validate="m:1")
        .with_columns(
            (
                pl.col("score") - pl.col("effective_time_offset")
                < pl.col("slow_baseline")
            ).alias("__expected_short"),
            (
                pl.col("score")
                - pl.col("effective_time_offset")
                - pl.col("slow_baseline")
            ).alias(pred_column),
        )
        .sort("__row_id")
    )

    if result.height != expected_sample_count:
        raise AssertionError(
            f"有效样本量为 {result.height:,}，预期 {expected_sample_count:,}。"
        )
    if result.select(pl.col(pred_column).is_null().sum()).item() != 0:
        raise AssertionError("最终 pred 中存在空值。")

    direction_mismatch = result.filter(
        (pl.col(pred_column) < 0) != pl.col("__expected_short")
    ).height
    if direction_mismatch:
        raise AssertionError(
            f"零阈值方向验收失败，共 {direction_mismatch:,} 条不一致。"
        )

    final_output_columns = [
        "Date" if column == date_column else column for column in output_columns
    ]
    result.select(final_output_columns).write_parquet(temporary_path)
    temporary_path.replace(output_path)

    return PredExportResult(
        output_path=output_path,
        sample_count=result.height,
        pred_min=float(result[pred_column].min()),
        pred_max=float(result[pred_column].max()),
        short_share=float(result.select((pl.col(pred_column) < 0).mean()).item()),
    )
