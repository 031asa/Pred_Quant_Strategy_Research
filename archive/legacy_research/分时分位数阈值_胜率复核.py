from pathlib import Path
import json

import numpy as np
import polars as pl


BASE_DIR = Path(__file__).resolve().parent
with (BASE_DIR / "中位数阈值_20与50等分_参数.json").open(encoding="utf-8") as f:
    cfg = json.load(f)

SOURCE = Path(cfg["source_path"])
DT = cfg["datetime_column"]
PRED = cfg["pred_column"]
TAG = cfg["return_column"]
ROLLING_DAYS = 20


df = (
    pl.read_parquet(SOURCE)
    .select([DT, PRED, TAG])
    .drop_nulls()
    .with_columns(
        (pl.col(PRED).cast(pl.Float64) + float(cfg["score_offset"]))
        .clip(float(cfg["score_min"]), float(cfg["score_max"]))
        .alias("score"),
        pl.col(TAG).cast(pl.Float64).alias("tag"),
        pl.col(DT).dt.date().alias("Date"),
        (
            pl.col(DT).dt.hour().cast(pl.Int32) * 60
            + pl.col(DT).dt.minute().cast(pl.Int32)
        ).alias("minute"),
    )
    .with_columns(
        pl.when(pl.col("minute") < 14 * 60 + 30)
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .cast(pl.Int8)
        .alias("segment")
    )
    .select([DT, "Date", "segment", "score", "tag"])
)


threshold_rows = []
for segment in (0, 1):
    part = df.filter(pl.col("segment") == segment)
    dates = part["Date"].unique().sort().to_list()
    daily_score = {d: part.filter(pl.col("Date") == d)["score"].to_numpy() for d in dates}
    daily_tag = {d: part.filter(pl.col("Date") == d)["tag"].to_numpy() for d in dates}

    for i, current_date in enumerate(dates):
        threshold = None
        short_base_rate = None
        if i >= ROLLING_DAYS:
            history_dates = dates[i - ROLLING_DAYS:i]
            history_score = np.concatenate([daily_score[d] for d in history_dates])
            history_tag = np.concatenate([daily_tag[d] for d in history_dates])
            # 空单占全体订单的比例：Tag<0的样本数 / 全部样本数。
            short_base_rate = float(np.mean(history_tag < 0))
            threshold = float(np.quantile(history_score, short_base_rate))
        threshold_rows.append(
            {
                "Date": current_date,
                "segment": segment,
                "historical_short_rate": short_base_rate,
                "quantile_threshold": threshold,
            }
        )

thresholds = pl.DataFrame(threshold_rows).with_columns(pl.col("Date").cast(pl.Date))
evaluation = (
    df.join(thresholds, on=["Date", "segment"], how="left")
    .filter(pl.col("quantile_threshold").is_not_null())
    .sort(["Date", DT])
)

score = evaluation["score"].to_numpy()
tag = evaluation["tag"].to_numpy()
segment = evaluation["segment"].to_numpy()
q_threshold = evaluation["quantile_threshold"].to_numpy()


def safe_rate(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def summarize(label, direction, mask):
    selected_n = int(mask.sum())
    short = mask & (direction == -1)
    long = mask & (direction == 1)
    short_n = int(short.sum())
    long_n = int(long.sum())
    actual_short_n = int(np.sum(mask & (tag < 0)))
    actual_long_n = int(np.sum(mask & (tag > 0)))
    short_wins = int(np.sum(short & (tag < 0)))
    long_wins = int(np.sum(long & (tag > 0)))
    return {
        "方案": label,
        "样本量": selected_n,
        "实际负类占比": safe_rate(actual_short_n, selected_n),
        "预测做空占比": safe_rate(short_n, selected_n),
        "做空胜率_precision": safe_rate(short_wins, short_n),
        "空头召回率_recall": safe_rate(short_wins, actual_short_n),
        "预测做多占比": safe_rate(long_n, selected_n),
        "做多胜率_precision": safe_rate(long_wins, long_n),
        "多头召回率_recall": safe_rate(long_wins, actual_long_n),
        "总体方向准确率": safe_rate(short_wins + long_wins, selected_n),
    }


fixed_direction = np.where(score < 0.5, -1, 1)
quantile_direction = np.where(score < q_threshold, -1, 1)

summary_rows = []
change_rows = []
segments = [
    ("全体样本", np.ones(len(score), dtype=bool)),
    ("14:30以前", segment == 0),
    ("14:30及以后", segment == 1),
]

for segment_label, mask in segments:
    for method_label, direction in (("固定阈值0.5", fixed_direction), ("滚动分位数阈值", quantile_direction)):
        row = summarize(method_label, direction, mask)
        row = {"时段": segment_label, **row}
        summary_rows.append(row)

    transitions = [
        (-1, 1, "固定空→分位数多"),
        (1, -1, "固定多→分位数空"),
        (-1, -1, "两方案均为空"),
        (1, 1, "两方案均为多"),
    ]
    for old_direction, new_direction, transition_name in transitions:
        changed = mask & (fixed_direction == old_direction) & (quantile_direction == new_direction)
        n = int(changed.sum())
        change_rows.append(
            {
                "时段": segment_label,
                "方向变化": transition_name,
                "样本量": n,
                "占时段样本比例": safe_rate(n, int(mask.sum())),
                "Tag负类比例": safe_rate(int(np.sum(changed & (tag < 0))), n),
                "Tag正类比例": safe_rate(int(np.sum(changed & (tag > 0))), n),
            }
        )

threshold_summary = (
    thresholds.filter(pl.col("quantile_threshold").is_not_null())
    .group_by("segment")
    .agg(
        pl.len().alias("有效交易日数"),
        pl.col("historical_short_rate").mean().alias("历史负类占比均值"),
        pl.col("historical_short_rate").median().alias("历史负类占比中位数"),
        pl.col("quantile_threshold").mean().alias("分位数阈值均值"),
        pl.col("quantile_threshold").median().alias("分位数阈值中位数"),
        pl.col("quantile_threshold").min().alias("分位数阈值最小值"),
        pl.col("quantile_threshold").max().alias("分位数阈值最大值"),
    )
    .with_columns(
        pl.when(pl.col("segment") == 0)
        .then(pl.lit("14:30以前"))
        .otherwise(pl.lit("14:30及以后"))
        .alias("时段")
    )
    .select(pl.exclude("segment"))
)

summary = pl.DataFrame(summary_rows)
changes = pl.DataFrame(change_rows)
daily_thresholds = thresholds.with_columns(
    pl.when(pl.col("segment") == 0)
    .then(pl.lit("14:30以前"))
    .otherwise(pl.lit("14:30及以后"))
    .alias("时段")
).select(["Date", "时段", "historical_short_rate", "quantile_threshold"])

summary.write_csv(BASE_DIR / "分时分位数阈值_方案对比.csv")
changes.write_csv(BASE_DIR / "分时分位数阈值_方向变化样本.csv")
threshold_summary.write_csv(BASE_DIR / "分时分位数阈值_阈值摘要.csv")
daily_thresholds.write_csv(BASE_DIR / "分时分位数阈值_每日阈值.csv")

print("=== 阈值摘要 ===")
print(threshold_summary)
print("\n=== 固定0.5 vs 滚动分位数 ===")
print(summary)
print("\n=== 方向变化样本 ===")
print(changes)
