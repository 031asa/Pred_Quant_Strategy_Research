from __future__ import annotations

# Implementation module. Public execution goes through utils.pipelines.

import json
from pathlib import Path
import sys

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.segmented_drift import (
    SEGMENT_AFTER,
    SEGMENT_BEFORE,
    build_selected_daily_states,
)


OUTPUT_DIR = PROJECT_ROOT / "result" / "数据"
PARAMETER_PATH = PROJECT_ROOT / "result" / "参数" / "分时Score时序漂移_选定参数.json"
OUTPUT_PATH = OUTPUT_DIR / "pred_eval_分时漂移修正后.parquet"
TEMP_PATH = OUTPUT_PATH.with_suffix(".tmp.parquet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


with (PROJECT_ROOT / "config.json").open(encoding="utf-8") as file:
    cfg = json.load(file)

source_path = PROJECT_ROOT / cfg["source_path"]
pred_column = cfg["pred_column"]
datetime_column = cfg["datetime_column"]
cutoff = cfg["time_cutoff"]
score_offset = float(cfg["score_offset"])
score_min = float(cfg["score_min"])
score_max = float(cfg["score_max"])

with PARAMETER_PATH.open(encoding="utf-8") as file:
    selected_config = json.load(file)

source = pl.read_parquet(source_path).with_row_index("__row_id")
output_columns = [column for column in source.columns if column != "__row_id"]

prepared = source.with_columns(
    pl.when(pl.col(datetime_column).dt.strftime("%H:%M") < cutoff)
    .then(pl.lit(SEGMENT_BEFORE))
    .otherwise(pl.lit(SEGMENT_AFTER))
    .alias("时段"),
    (
        (pl.col(pred_column).cast(pl.Float64) + score_offset)
        .clip(score_min, score_max)
    ).alias("score"),
)
state = build_selected_daily_states(
    prepared.select(["Date", "时段", "score"]), selected_config
).select(["Date", "时段", "slow_baseline", "effective_time_offset"])

result = (
    prepared
    .join(state, on=["Date", "时段"], how="inner", validate="m:1")
    .with_columns(
        (
            pl.col("score")
            - pl.col("effective_time_offset")
            < pl.col("slow_baseline")
        ).alias("__expected_short"),
        (
            pl.col("score")
            - pl.col("effective_time_offset")
            - pl.col("slow_baseline")
        ).alias(pred_column)
    )
    .sort("__row_id")
)

if result.height != 1_667_520:
    raise AssertionError(f"有效样本量为 {result.height:,}，预期 1,667,520。")
if result.select(pl.col(pred_column).is_null().sum()).item() != 0:
    raise AssertionError("最终 pred 中存在空值。")

# 最终 pred 已归零：其符号必须与修正后 Score 相对慢速基准的方向逐条一致。
direction_mismatch = result.filter(
    (pl.col(pred_column) < 0) != pl.col("__expected_short")
).height
if direction_mismatch:
    raise AssertionError(f"零阈值方向验收失败，共 {direction_mismatch:,} 条不一致。")

result.select(output_columns).write_parquet(TEMP_PATH)
TEMP_PATH.replace(OUTPUT_PATH)

print(f"已写入：{OUTPUT_PATH}")
print(f"有效样本量：{result.height:,}")
print(f"pred范围：[{result[pred_column].min():.12f}, {result[pred_column].max():.12f}]")
print(f"做空占比(pred < 0)：{result.select((pl.col(pred_column) < 0).mean()).item():.12%}")
