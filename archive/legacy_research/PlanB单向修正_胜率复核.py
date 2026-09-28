from pathlib import Path
import json

import numpy as np
import polars as pl


BASE_DIR = Path(__file__).resolve().parent
with (BASE_DIR / "中位数阈值_20与50等分_参数.json").open(encoding="utf-8") as f:
    config = json.load(f)

SOURCE = Path(config["source_path"])
DATETIME = config["datetime_column"]
PRED = config["pred_column"]
TAG = config["return_column"]
CENTER = 0.5
ROLLING_DAYS = 20
STRENGTHS = [0.0, 0.25, 0.50, 0.75, 1.00]


df = (
    pl.read_parquet(SOURCE)
    .select([DATETIME, PRED, TAG])
    .drop_nulls()
    .with_columns(
        (pl.col(PRED).cast(pl.Float64) + float(config["score_offset"]))
        .clip(float(config["score_min"]), float(config["score_max"]))
        .alias("score"),
        pl.col(TAG).cast(pl.Float64).alias("tag"),
        pl.col(DATETIME).dt.date().alias("Date"),
        (
            pl.col(DATETIME).dt.hour().cast(pl.Int32) * 60
            + pl.col(DATETIME).dt.minute().cast(pl.Int32)
        ).alias("minute"),
    )
    .with_columns(
        pl.when(pl.col("minute") < 14 * 60 + 30)
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .cast(pl.Int8)
        .alias("segment")
    )
    .select([DATETIME, "Date", "segment", "score", "tag"])
)


center_rows = []
for segment in (0, 1):
    part = df.filter(pl.col("segment") == segment)
    dates = part["Date"].unique().sort().to_list()
    daily = {
        d: part.filter(pl.col("Date") == d)["score"].to_numpy()
        for d in dates
    }
    for i, current_date in enumerate(dates):
        rolling_center = None
        if i >= ROLLING_DAYS:
            history = np.concatenate([daily[d] for d in dates[i - ROLLING_DAYS:i]])
            rolling_center = float(np.median(history))
        center_rows.append(
            {
                "Date": current_date,
                "segment": segment,
                "rolling_center": rolling_center,
            }
        )

centers = pl.DataFrame(center_rows).with_columns(pl.col("Date").cast(pl.Date))
evaluation = (
    df.join(centers, on=["Date", "segment"], how="left")
    .filter(pl.col("rolling_center").is_not_null())
    .sort(["Date", DATETIME])
)

score = evaluation["score"].to_numpy()
tag = evaluation["tag"].to_numpy()
rolling_center = evaluation["rolling_center"].to_numpy()
segment = evaluation["segment"].to_numpy()
raw_direction = np.where(score > CENTER, 1, np.where(score < CENTER, -1, 0))


def rate(numerator, denominator):
    return float(numerator / denominator) if denominator else None


rows = []
transition_rows = []
segments = [
    ("全体样本", np.ones(len(score), dtype=bool)),
    ("14:30以前", segment == 0),
    ("14:30及以后", segment == 1),
]

for strength in STRENGTHS:
    # 单向修正：只有历史中位数高于0.5时才向下调；偏空时绝不上调。
    positive_long_bias = np.maximum(rolling_center - CENTER, 0.0)
    adjusted_score = np.clip(score - strength * positive_long_bias, 0.0, 1.0)
    adjusted_direction = np.where(
        adjusted_score > CENTER, 1, np.where(adjusted_score < CENTER, -1, 0)
    )

    for label, mask in segments:
        n = int(mask.sum())
        long_mask = mask & (adjusted_direction == 1)
        short_mask = mask & (adjusted_direction == -1)
        long_n = int(long_mask.sum())
        short_n = int(short_mask.sum())
        rows.append(
            {
                "时段": label,
                "修正强度": strength,
                "样本量": n,
                "做多占比": rate(long_n, n),
                "做多胜率": rate(int(np.sum(long_mask & (tag > 0))), long_n),
                "做空占比": rate(short_n, n),
                "做空胜率": rate(int(np.sum(short_mask & (tag < 0))), short_n),
                "方向变化数": int(np.sum(mask & (raw_direction != adjusted_direction))),
            }
        )

        changed = mask & (raw_direction == 1) & (adjusted_direction == -1)
        changed_n = int(changed.sum())
        transition_rows.append(
            {
                "时段": label,
                "修正强度": strength,
                "多转空样本量": changed_n,
                "多转空占时段比例": rate(changed_n, n),
                "多转空后的做空胜率": rate(int(np.sum(changed & (tag < 0))), changed_n),
            }
        )

result = pl.DataFrame(rows)
transitions = pl.DataFrame(transition_rows)
result.write_csv(BASE_DIR / "PlanB单向修正_多空胜率.csv")
transitions.write_csv(BASE_DIR / "PlanB单向修正_多转空样本.csv")

center_summary = (
    centers.filter(pl.col("rolling_center").is_not_null())
    .group_by("segment")
    .agg(
        pl.len().alias("有效交易日数"),
        pl.col("rolling_center").median().alias("滚动中位数的中位数"),
        (pl.col("rolling_center") > CENTER).sum().alias("存在多头偏置的交易日数"),
    )
    .with_columns(
        (pl.col("存在多头偏置的交易日数") / pl.col("有效交易日数"))
        .alias("存在多头偏置的交易日占比")
    )
    .with_columns(
        pl.when(pl.col("segment") == 0)
        .then(pl.lit("14:30以前"))
        .otherwise(pl.lit("14:30及以后"))
        .alias("时段")
    )
    .select(["时段", "有效交易日数", "滚动中位数的中位数", "存在多头偏置的交易日数", "存在多头偏置的交易日占比"])
)
center_summary.write_csv(BASE_DIR / "PlanB单向修正_滚动中心摘要.csv")

print("\n=== 单向修正：多空占比与胜率 ===")
print(result)
print("\n=== 原做多 -> 修正后做空 ===")
print(transitions)
print("\n=== 滚动中心摘要 ===")
print(center_summary)
