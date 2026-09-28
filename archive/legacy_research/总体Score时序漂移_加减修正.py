from pathlib import Path
from datetime import datetime
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


BASE_DIR = Path(__file__).resolve().parent
PARAM_PATH = BASE_DIR / "中位数阈值_20与50等分_参数.json"

COLD_START_DAYS = 20
TEST_FRACTION = 0.10
SHORT_RATE_TARGET = 0.50
SHORT_RATE_TOLERANCE = 0.01
HALF_LIFE_CANDIDATES = [1, 2, 5, 10, 20, 40, 60, 90, 120]


with PARAM_PATH.open(encoding="utf-8") as f:
    cfg = json.load(f)

SOURCE = Path(cfg["source_path"])
DT = cfg["datetime_column"]
PRED = cfg["pred_column"]
TAG = cfg["return_column"]


def safe_rate(numerator: int, denominator: int):
    return float(numerator / denominator) if denominator else None


def alpha_from_half_life(half_life_days: int) -> float:
    return float(1.0 - 0.5 ** (1.0 / half_life_days))


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
        pl.col(DT).dt.strftime("%Y-%m").alias("Month"),
    )
    .select([DT, "Date", "Month", "score", "tag"])
)

daily = (
    df.group_by("Date")
    .agg(
        pl.col("score").median().alias("daily_score_median"),
        pl.len().alias("daily_samples"),
    )
    .sort("Date")
    .with_row_index("date_index")
)

dates = daily["Date"].to_list()
daily_medians = daily["daily_score_median"].to_numpy()
daily_samples = daily["daily_samples"].to_numpy()
n_dates = len(dates)
test_days = int(math.ceil(n_dates * TEST_FRACTION))
test_start_index = n_dates - test_days
test_start_date = dates[test_start_index]

# "One month" is implemented as the complete first calendar month. The minimum
# remains 20 trading days, but a leftover one-day partial month is not evaluated
# as if it were a full monthly quantile observation.
first_month_key = (dates[0].year, dates[0].month)
first_next_month_index = next(
    (
        i
        for i, value in enumerate(dates)
        if (value.year, value.month) != first_month_key
    ),
    n_dates,
)
cold_start_days_effective = max(COLD_START_DAYS, first_next_month_index)

if n_dates <= cold_start_days_effective + test_days:
    raise ValueError("交易日数量不足以同时满足冷启动期和测试集要求。")

df = df.join(daily.select(["Date", "date_index"]), on="Date", how="left")


def build_state(half_life_days: int) -> pl.DataFrame:
    """All states used on date t are based only on dates strictly before t."""
    alpha = alpha_from_half_life(half_life_days)
    baselines = np.full(n_dates, np.nan, dtype=float)
    recent_centers = np.full(n_dates, np.nan, dtype=float)
    offsets = np.full(n_dates, np.nan, dtype=float)
    current_center = None

    for i in range(cold_start_days_effective, n_dates):
        # Slow baseline: expanding median of historical daily medians.
        # It is allowed to correct the initial one-month estimate as evidence grows.
        baseline = float(np.median(daily_medians[:i]))
        if i == cold_start_days_effective:
            current_center = baseline
        else:
            # Update with yesterday's median only; today's data never enters today's state.
            current_center = (1.0 - alpha) * current_center + alpha * float(daily_medians[i - 1])

        baselines[i] = baseline
        recent_centers[i] = current_center
        offsets[i] = current_center - baseline

    return pl.DataFrame(
        {
            "Date": dates,
            "date_index": np.arange(n_dates, dtype=np.int64),
            "slow_baseline": baselines,
            "recent_center": recent_centers,
            "time_offset": offsets,
        }
    ).with_columns(pl.col("Date").cast(pl.Date))


def attach_state(half_life_days: int, balance_offset: float = 0.0) -> pl.DataFrame:
    state = build_state(half_life_days)
    return (
        df.join(
            state.select(["Date", "slow_baseline", "recent_center"]),
            on="Date",
            how="left",
        )
        .filter(
            pl.col("slow_baseline").is_finite()
            & pl.col("recent_center").is_finite()
        )
        .with_columns(
            (pl.col("recent_center") + pl.lit(balance_offset)).alias(
                "effective_center"
            )
        )
        .with_columns(
            (pl.col("effective_center") - pl.col("slow_baseline")).alias(
                "time_offset"
            ),
            (pl.col("score") - pl.col("slow_baseline")).alias("raw_residual"),
            (pl.col("score") < pl.col("slow_baseline")).alias("raw_short"),
            (pl.col("date_index") >= test_start_index).alias("is_test"),
        )
        .with_columns(
            (pl.col("score") - pl.col("time_offset")).alias("adjusted_score"),
            (pl.col("score") - pl.col("effective_center")).alias(
                "adjusted_residual"
            ),
            (pl.col("score") < pl.col("effective_center")).alias(
                "adjusted_short"
            ),
        )
    )


def monthly_curve_metrics(data: pl.DataFrame, residual_column: str):
    monthly = (
        data.group_by("Month")
        .agg(pl.col(residual_column).median().alias("monthly_median_residual"))
        .sort("Month")
    )
    residual = monthly["monthly_median_residual"].to_numpy()
    area = float(np.mean(np.abs(residual))) if len(residual) else np.nan
    slope = float(np.mean(np.diff(residual) ** 2)) if len(residual) > 1 else np.nan
    max_abs = float(np.max(np.abs(residual))) if len(residual) else np.nan
    return area, slope, max_abs, monthly


search_rows = []
for half_life in HALF_LIFE_CANDIDATES:
    uncalibrated = attach_state(half_life).filter(
        pl.col("date_index") < test_start_index
    )
    # A fixed additive offset, estimated on the development segment only, makes
    # the overall development short rate 50% (up to score ties). It is not tuned
    # on the final 10% test segment and remains a simple additive transform.
    balance_offset = float(
        (uncalibrated["score"] - uncalibrated["recent_center"]).median()
    )
    candidate = attach_state(half_life, balance_offset).filter(
        pl.col("date_index") < test_start_index
    )
    short_rate = float(candidate["adjusted_short"].mean())
    area, slope, max_abs, _ = monthly_curve_metrics(candidate, "adjusted_residual")
    state = build_state(half_life).filter(
        (pl.col("date_index") >= cold_start_days_effective)
        & (pl.col("date_index") < test_start_index)
    )
    daily_offset = state["time_offset"].drop_nulls().to_numpy()
    offset_change = np.diff(daily_offset)
    offset_change_std = float(np.std(offset_change, ddof=0)) if len(offset_change) else np.nan
    offset_abs_change_p95 = (
        float(np.quantile(np.abs(offset_change), 0.95)) if len(offset_change) else np.nan
    )
    offset_abs_change_max = (
        float(np.max(np.abs(offset_change))) if len(offset_change) else np.nan
    )
    search_rows.append(
        {
            "半衰期_交易日": half_life,
            "alpha": alpha_from_half_life(half_life),
            "开发段固定平衡偏移量": balance_offset,
            "开发段总体做空占比": short_rate,
            "偏离50%": abs(short_rate - SHORT_RATE_TARGET),
            "满足49%至51%硬约束": abs(short_rate - SHORT_RATE_TARGET)
            <= SHORT_RATE_TOLERANCE,
            "开发段月度绝对面积均值": area,
            "开发段月度斜率均方": slope,
            "开发段最大月度中位偏移": max_abs,
            "开发段日偏移变化标准差": offset_change_std,
            "开发段日偏移绝对变化P95": offset_abs_change_p95,
            "开发段日偏移绝对变化最大值": offset_abs_change_max,
        }
    )

search = pl.DataFrame(search_rows).sort(
    ["满足49%至51%硬约束", "开发段月度绝对面积均值", "开发段月度斜率均方"],
    descending=[True, False, False],
)
feasible = search.filter(pl.col("满足49%至51%硬约束"))
if feasible.height:
    selected = feasible.sort(
        ["开发段月度绝对面积均值", "开发段月度斜率均方"]
    ).row(0, named=True)
    selection_note = "在满足总体做空占比49%至51%的候选中，优先选择月度绝对面积最小者，斜率作为次级排序。"
else:
    selected = search.sort(
        ["偏离50%", "开发段月度绝对面积均值", "开发段月度斜率均方"]
    ).row(0, named=True)
    selection_note = "没有候选满足49%至51%硬约束，退化为先最小化做空比例偏差，再比较面积与斜率。"

selected_half_life = int(selected["半衰期_交易日"])
selected_alpha = float(selected["alpha"])
selected_balance_offset = float(selected["开发段固定平衡偏移量"])
selected_state = build_state(selected_half_life)
evaluation = attach_state(selected_half_life, selected_balance_offset)


def summarize_period(period_label: str, data: pl.DataFrame, method: str):
    if method == "未修正_慢基准":
        short = data["raw_short"].to_numpy()
        residual_column = "raw_residual"
    else:
        short = data["adjusted_short"].to_numpy()
        residual_column = "adjusted_residual"

    tag = data["tag"].to_numpy()
    direction = np.where(short, -1, 1)
    short_n = int(short.sum())
    long_n = int((~short).sum())
    short_wins = int(np.sum(short & (tag < 0)))
    long_wins = int(np.sum((~short) & (tag > 0)))
    actual_short_n = int(np.sum(tag < 0))
    strategy_return = direction * tag
    area, slope, max_abs, _ = monthly_curve_metrics(data, residual_column)

    return {
        "数据段": period_label,
        "方案": method,
        "样本量": data.height,
        "交易日数": data["Date"].n_unique(),
        "实际负类占比": safe_rate(actual_short_n, data.height),
        "预测做空占比": safe_rate(short_n, data.height),
        "预测做多占比": safe_rate(long_n, data.height),
        "偏离50%": abs(safe_rate(short_n, data.height) - 0.5),
        "做空胜率_precision": safe_rate(short_wins, short_n),
        "做多胜率_precision": safe_rate(long_wins, long_n),
        "总体方向准确率": safe_rate(short_wins + long_wins, data.height),
        "平均策略收益率": float(np.mean(strategy_return)),
        "月度绝对面积均值": area,
        "月度斜率均方": slope,
        "最大月度中位偏移": max_abs,
    }


periods = [
    ("开发段", evaluation.filter(pl.col("date_index") < test_start_index)),
    ("最终10%测试段", evaluation.filter(pl.col("date_index") >= test_start_index)),
    ("全部有效段", evaluation),
]
summary_rows = []
for period_label, part in periods:
    summary_rows.append(summarize_period(period_label, part, "未修正_慢基准"))
    summary_rows.append(summarize_period(period_label, part, "加减修正_近期中心"))
summary = pl.DataFrame(summary_rows)


monthly = (
    evaluation.group_by("Month")
    .agg(
        pl.len().alias("样本量"),
        pl.col("Date").n_unique().alias("交易日数"),
        pl.col("is_test").mean().alias("测试样本占比"),
        pl.col("slow_baseline").median().alias("慢速基准中位数"),
        pl.col("recent_center").median().alias("近期中心中位数"),
        pl.col("time_offset").median().alias("加减修正值中位数"),
        pl.col("score").quantile(0.10).alias("原始P10"),
        pl.col("score").quantile(0.25).alias("原始P25"),
        pl.col("score").median().alias("原始P50"),
        pl.col("score").quantile(0.75).alias("原始P75"),
        pl.col("score").quantile(0.90).alias("原始P90"),
        pl.col("adjusted_score").quantile(0.10).alias("修正P10"),
        pl.col("adjusted_score").quantile(0.25).alias("修正P25"),
        pl.col("adjusted_score").median().alias("修正P50"),
        pl.col("adjusted_score").quantile(0.75).alias("修正P75"),
        pl.col("adjusted_score").quantile(0.90).alias("修正P90"),
        pl.col("raw_residual").median().alias("原始相对基准偏移"),
        pl.col("adjusted_residual").median().alias("修正后相对基准偏移"),
        pl.col("raw_short").mean().alias("原始做空占比"),
        pl.col("adjusted_short").mean().alias("修正后做空占比"),
        (pl.col("tag") < 0).mean().alias("实际负类占比"),
    )
    .sort("Month")
)

daily_state = (
    selected_state.filter(
        pl.col("slow_baseline").is_finite()
        & pl.col("recent_center").is_finite()
    )
    .join(
        daily.select(["Date", "daily_score_median", "daily_samples"]),
        on="Date",
        how="left",
    )
    .with_columns(
        (pl.col("recent_center") + pl.lit(selected_balance_offset)).alias(
            "effective_center"
        ),
        (pl.col("date_index") >= test_start_index).alias("is_test"),
    )
    .with_columns(
        (pl.col("effective_center") - pl.col("slow_baseline")).alias(
            "effective_time_offset"
        ),
        (pl.col("daily_score_median") - pl.col("effective_center")).alias(
            "daily_residual_after_adjustment"
        ),
    )
)


search.write_csv(BASE_DIR / "总体Score时序漂移_参数搜索.csv")
summary.write_csv(BASE_DIR / "总体Score时序漂移_方案对比.csv")
monthly.write_csv(BASE_DIR / "总体Score时序漂移_月度分位数.csv")
daily_state.write_csv(BASE_DIR / "总体Score时序漂移_每日状态.csv")

selected_config = {
    "source_path": str(SOURCE),
    "cold_start_minimum_trading_days": COLD_START_DAYS,
    "cold_start_effective_trading_days": cold_start_days_effective,
    "cold_start_calendar_month": f"{dates[0].year:04d}-{dates[0].month:02d}",
    "total_trading_days": n_dates,
    "test_fraction": TEST_FRACTION,
    "test_trading_days": test_days,
    "test_start_date": str(test_start_date),
    "test_end_date": str(dates[-1]),
    "selected_half_life_trading_days": selected_half_life,
    "selected_alpha": selected_alpha,
    "selected_development_balance_offset": selected_balance_offset,
    "selected_development_daily_offset_abs_change_p95": float(
        selected["开发段日偏移绝对变化P95"]
    ),
    "selected_development_daily_offset_abs_change_max": float(
        selected["开发段日偏移绝对变化最大值"]
    ),
    "short_rate_target": SHORT_RATE_TARGET,
    "short_rate_tolerance": SHORT_RATE_TOLERANCE,
    "selection_note": selection_note,
    "causal_rule": "Date t uses only daily medians from dates strictly earlier than t.",
    "tag_used_for_selection": False,
    "baseline_definition": "Expanding median of historical daily score medians.",
    "transform": "adjusted_score = score - ((recent_center + balance_offset) - slow_baseline)",
}
with (BASE_DIR / "总体Score时序漂移_选定参数.json").open("w", encoding="utf-8") as f:
    json.dump(selected_config, f, ensure_ascii=False, indent=2)


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

month_dates = [datetime.strptime(x, "%Y-%m") for x in monthly["Month"].to_list()]
baseline_m = monthly["慢速基准中位数"].to_numpy()
test_mark = datetime(test_start_date.year, test_start_date.month, 1)

fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

ax = axes[0]
ax.fill_between(month_dates, monthly["原始P10"], monthly["原始P90"], alpha=0.16, color="#7aa6d8", label="P10～P90")
ax.fill_between(month_dates, monthly["原始P25"], monthly["原始P75"], alpha=0.28, color="#3f83c5", label="P25～P75")
ax.plot(month_dates, monthly["原始P50"], color="#1558a0", marker="o", linewidth=2, label="原始P50")
ax.plot(month_dates, baseline_m, color="#4f5964", linestyle="--", linewidth=1.6, label="时序慢基准")
ax.set_title("A. 总体 score 月度分位数（修正前）")
ax.set_ylabel("score")
ax.legend(ncol=4, loc="best")
ax.grid(alpha=0.2)

ax = axes[1]
ax.fill_between(month_dates, monthly["修正P10"], monthly["修正P90"], alpha=0.16, color="#7bbf9e", label="P10～P90")
ax.fill_between(month_dates, monthly["修正P25"], monthly["修正P75"], alpha=0.28, color="#2f9c67", label="P25～P75")
ax.plot(month_dates, monthly["修正P50"], color="#147a4b", marker="o", linewidth=2, label="修正P50")
ax.plot(month_dates, baseline_m, color="#4f5964", linestyle="--", linewidth=1.6, label="时序慢基准")
ax.set_title(f"B. 总体 score 月度分位数（纯加减修正，半衰期={selected_half_life}日）")
ax.set_ylabel("adjusted score")
ax.legend(ncol=4, loc="best")
ax.grid(alpha=0.2)

ax = axes[2]
ax.axhline(0.0, color="#4f5964", linestyle="--", linewidth=1.4, label="基准=0")
ax.plot(month_dates, monthly["原始相对基准偏移"], color="#b44b4b", marker="o", linewidth=1.8, label="修正前")
ax.plot(month_dates, monthly["修正后相对基准偏移"], color="#147a4b", marker="o", linewidth=2, label="修正后")
ax.set_title("C. 月度中位数相对时序慢基准的偏移")
ax.set_ylabel("中位数偏移")
ax.legend(loc="best")
ax.grid(alpha=0.2)

for ax in axes:
    ax.axvline(test_mark, color="#8a5a9f", linestyle=":", linewidth=2)
    ax.text(test_mark, ax.get_ylim()[1], " 最终10%测试段", color="#6f3f83", va="top", ha="left")

axes[-1].set_xlabel("月份")
fig.suptitle("总体 Score 时序漂移：扩展基准 + EWMA近期中心 + 加减修正", fontsize=16, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(BASE_DIR / "总体Score时序漂移_月度分位数对比.png", dpi=180, bbox_inches="tight")
plt.close(fig)


def equal_count_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign exactly equal-count rank bins; ties are stable by input order."""
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(len(values), dtype=np.int16)
    bins[order] = np.minimum(
        (np.arange(len(values), dtype=np.int64) * n_bins) // len(values),
        n_bins - 1,
    )
    return bins


def quantile_diagnostics(score_column: str, variant_label: str, file_label: str):
    values = evaluation[score_column].to_numpy()
    tag = evaluation["tag"].to_numpy()
    date_values = evaluation["Date"]
    bins20 = equal_count_bins(values, 20)
    bins10 = bins20 // 2

    metric_rows = []
    for bin_index in range(20):
        mask = bins20 == bin_index
        direction = -1 if bin_index < 10 else 1
        strategy_return = direction * tag[mask]
        wins = strategy_return[strategy_return > 0]
        losses = strategy_return[strategy_return < 0]
        win_loss_ratio = (
            float(np.mean(wins) / abs(np.mean(losses)))
            if len(wins) and len(losses)
            else None
        )
        metric_rows.append(
            {
                "版本": variant_label,
                "档位": f"Q{bin_index + 1:02d}",
                "方向": "做空" if direction == -1 else "做多",
                "分数下界": float(np.min(values[mask])),
                "分数上界": float(np.max(values[mask])),
                "样本量": int(mask.sum()),
                "平均策略收益": float(np.mean(strategy_return)),
                "胜率": float(np.mean(strategy_return > 0)),
                "平均盈亏比": win_loss_ratio,
            }
        )

    metric_df = pl.DataFrame(metric_rows)
    x = np.arange(1, 21)
    avg_return = metric_df["平均策略收益"].to_numpy()
    win_rate = metric_df["胜率"].to_numpy()
    win_loss = metric_df["平均盈亏比"].to_numpy()
    colors = ["#2f80d0"] * 10 + ["#f08031"] * 10

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10.5), sharex=True)
    fig.suptitle(f"全体样本20等分三指标（{variant_label}）", fontsize=17, y=0.985)
    fig.text(
        0.5,
        0.945,
        "Q01-Q10做空（-Tag）｜Q11-Q20做多（Tag）｜各档严格等样本｜虚线为中位数分界",
        ha="center",
        color="#526071",
        fontsize=11,
    )

    axes[0].bar(x, avg_return, color=colors, width=0.82)
    axes[0].axhline(0, color="#6f7780", linewidth=1, linestyle="--")
    axes[0].set_ylabel("平均策略收益")
    axes[0].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_title("平均策略收益", loc="left", fontsize=11)

    axes[1].plot(x[:10], win_rate[:10], color="#2f80d0", marker="o", linewidth=2)
    axes[1].plot(x[10:], win_rate[10:], color="#f08031", marker="o", linewidth=2)
    axes[1].axhline(0.5, color="#6f7780", linewidth=1, linestyle="--")
    axes[1].set_ylabel("胜率")
    axes[1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    axes[1].set_title("胜率", loc="left", fontsize=11)

    axes[2].plot(x[:10], win_loss[:10], color="#2f80d0", marker="o", linewidth=2)
    axes[2].plot(x[10:], win_loss[10:], color="#f08031", marker="o", linewidth=2)
    axes[2].axhline(1.0, color="#6f7780", linewidth=1, linestyle="--")
    axes[2].set_ylabel("平均盈亏比")
    axes[2].set_title("平均盈亏比", loc="left", fontsize=11)

    for ax in axes:
        ax.axvline(10.5, color="#6f7780", linewidth=1.2, linestyle=(0, (3, 3)))
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)

    axes[2].set_xticks(x, [f"Q{i:02d}" for i in x], rotation=0)
    axes[2].set_xlabel("档位（左侧做空，右侧做多）")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    fig.savefig(
        BASE_DIR / f"总体Score时序漂移_{file_label}_20等分三指标.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)

    strategy_direction = np.where(bins10 < 5, -1, 1)
    strategy_return = strategy_direction * tag
    daily_bin = (
        pl.DataFrame(
            {
                "Date": date_values,
                "bin_index": bins10,
                "strategy_return": strategy_return,
            }
        )
        .group_by(["Date", "bin_index"])
        .agg(
            pl.col("strategy_return").mean().alias("daily_equal_weight_return"),
            pl.len().alias("daily_samples"),
        )
        .sort(["bin_index", "Date"])
    )

    cumulative_rows = []
    fig, ax = plt.subplots(figsize=(16, 9.5))
    short_colors = plt.cm.Greens(np.linspace(0.88, 0.45, 5))
    long_colors = plt.cm.Reds(np.linspace(0.45, 0.88, 5))
    legend_lines = []
    legend_labels = []

    for bin_index in range(10):
        part = daily_bin.filter(pl.col("bin_index") == bin_index).sort("Date")
        part_dates = part["Date"].to_list()
        daily_returns = part["daily_equal_weight_return"].to_numpy()
        cumulative = np.cumprod(1.0 + daily_returns) - 1.0
        mask = bins10 == bin_index
        direction_label = "做空" if bin_index < 5 else "做多"
        color = short_colors[bin_index] if bin_index < 5 else long_colors[bin_index - 5]
        line, = ax.plot(part_dates, cumulative, color=color, linewidth=2.2)
        legend_lines.append(line)
        legend_labels.append(
            f"Q{bin_index + 1:02d} {direction_label}｜"
            f"[{np.min(values[mask]):.6f}, {np.max(values[mask]):.6f}]｜"
            f"N={int(mask.sum()):,}｜期末={cumulative[-1]:.2%}"
        )
        for date_value, daily_return, cumulative_return, daily_n in zip(
            part_dates,
            daily_returns,
            cumulative,
            part["daily_samples"].to_numpy(),
        ):
            cumulative_rows.append(
                {
                    "版本": variant_label,
                    "Date": date_value,
                    "档位": f"Q{bin_index + 1:02d}",
                    "方向": direction_label,
                    "当日样本量": int(daily_n),
                    "当日等权策略收益": float(daily_return),
                    "累计收益率": float(cumulative_return),
                }
            )

    ax.axhline(0.0, color="#66717f", linewidth=1.2)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"全体样本10等分累计收益率（{variant_label}）", fontsize=18, pad=26)
    ax.text(
        0.5,
        1.015,
        "全体样本独立10等分｜Q01-Q05做空（-Tag）｜Q06-Q10做多（Tag）｜日内等权后按日复利",
        transform=ax.transAxes,
        ha="center",
        color="#526071",
        fontsize=11,
    )
    ax.set_xlabel("交易日")
    ax.set_ylabel("累计收益率")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(
        legend_lines,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9.5,
        title="档位｜方向｜分数区间｜样本量｜期末累计收益",
    )
    fig.text(
        0.08,
        0.015,
        f"有效样本：{len(values):,}｜交易日：{evaluation['Date'].n_unique()}｜"
        f"区间：{evaluation['Date'].min()} 至 {evaluation['Date'].max()}",
        color="#526071",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 0.79, 0.96])
    fig.savefig(
        BASE_DIR / f"总体Score时序漂移_{file_label}_10等分累计收益.png",
        dpi=190,
        bbox_inches="tight",
    )
    plt.close(fig)

    return metric_df, pl.DataFrame(cumulative_rows)


raw_quantile_metrics, raw_cumulative = quantile_diagnostics(
    "score", "修正前", "修正前"
)
adjusted_quantile_metrics, adjusted_cumulative = quantile_diagnostics(
    "adjusted_score", "时序加减修正后", "修正后"
)
pl.concat([raw_quantile_metrics, adjusted_quantile_metrics]).write_csv(
    BASE_DIR / "总体Score时序漂移_20等分指标.csv"
)
pl.concat([raw_cumulative, adjusted_cumulative]).write_csv(
    BASE_DIR / "总体Score时序漂移_10等分累计收益.csv"
)


def distribution_diagnostics(score_column: str, variant_label: str, file_label: str):
    values = evaluation[score_column].to_numpy()
    tag = evaluation["tag"].to_numpy()
    bins20 = equal_count_bins(values, 20)
    rows = []

    for bin_index in range(20):
        mask = bins20 == bin_index
        direction = -1 if bin_index < 10 else 1
        samples = {
            "Tag": tag[mask],
            "策略收益": direction * tag[mask],
        }
        for object_label, object_values in samples.items():
            quantiles = np.quantile(
                object_values,
                [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
            )
            q05 = quantiles[1]
            q95 = quantiles[7]
            rows.append(
                {
                    "版本": variant_label,
                    "档位": f"Q{bin_index + 1:02d}",
                    "方向": "做空" if direction == -1 else "做多",
                    "对象": object_label,
                    "分数下界": float(np.min(values[mask])),
                    "分数上界": float(np.max(values[mask])),
                    "样本量": int(mask.sum()),
                    "P01": float(quantiles[0]),
                    "P05": float(q05),
                    "P10": float(quantiles[2]),
                    "P25": float(quantiles[3]),
                    "P50": float(quantiles[4]),
                    "P75": float(quantiles[5]),
                    "P90": float(quantiles[6]),
                    "P95": float(q95),
                    "P99": float(quantiles[8]),
                    "均值": float(np.mean(object_values)),
                    "标准差": float(np.std(object_values, ddof=0)),
                    "正值比例": float(np.mean(object_values > 0)),
                    "负值比例": float(np.mean(object_values < 0)),
                    "零值比例": float(np.mean(object_values == 0)),
                    "下尾5%均值": float(np.mean(object_values[object_values <= q05])),
                    "上尾5%均值": float(np.mean(object_values[object_values >= q95])),
                }
            )

    distribution = pl.DataFrame(rows)

    def draw_distribution(object_label: str, output_suffix: str):
        part = distribution.filter(pl.col("对象") == object_label).sort("档位")
        x = np.arange(1, 21)
        colors = ["#2f80d0"] * 10 + ["#f08031"] * 10
        stats = []
        for row in part.iter_rows(named=True):
            stats.append(
                {
                    "label": row["档位"],
                    "whislo": row["P10"],
                    "q1": row["P25"],
                    "med": row["P50"],
                    "q3": row["P75"],
                    "whishi": row["P90"],
                    "fliers": [],
                }
            )

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(14.5, 9.5),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1]},
        )
        fig.suptitle(
            f"全体样本20等分{object_label}分布（{variant_label}）",
            fontsize=17,
            y=0.985,
        )
        fig.text(
            0.5,
            0.945,
            "箱体=P25～P75｜中线=P50｜须线=P10～P90｜Q01-Q10按做空观察，Q11-Q20按做多观察",
            ha="center",
            color="#526071",
            fontsize=10.5,
        )

        box = axes[0].bxp(
            stats,
            positions=x,
            widths=0.65,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#26313d", "linewidth": 1.4},
            whiskerprops={"color": "#6f7780", "linewidth": 1.0},
            capprops={"color": "#6f7780", "linewidth": 1.0},
        )
        for patch_item, color in zip(box["boxes"], colors):
            patch_item.set_facecolor(color)
            patch_item.set_alpha(0.55)
            patch_item.set_edgecolor(color)
        axes[0].axhline(0, color="#596573", linewidth=1.1, linestyle="--")
        axes[0].axvline(10.5, color="#6f7780", linewidth=1.2, linestyle=(0, (3, 3)))
        axes[0].set_ylabel(f"{object_label}（分位区间）")
        axes[0].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        axes[0].grid(axis="y", alpha=0.22)
        axes[0].spines[["top", "right"]].set_visible(False)

        positive_rate = part["正值比例"].to_numpy()
        negative_rate = part["负值比例"].to_numpy()
        zero_rate = part["零值比例"].to_numpy()
        axes[1].bar(x, negative_rate, color="#d86359", width=0.78, label="负值")
        axes[1].bar(
            x,
            zero_rate,
            bottom=negative_rate,
            color="#aeb6bf",
            width=0.78,
            label="零值",
        )
        axes[1].bar(
            x,
            positive_rate,
            bottom=negative_rate + zero_rate,
            color="#3a9d68",
            width=0.78,
            label="正值",
        )
        axes[1].axhline(0.5, color="#596573", linewidth=1.0, linestyle="--")
        axes[1].axvline(10.5, color="#6f7780", linewidth=1.2, linestyle=(0, (3, 3)))
        axes[1].set_ylabel("正负占比")
        axes[1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        axes[1].set_ylim(0, 1)
        axes[1].legend(ncol=3, frameon=False, loc="upper center")
        axes[1].grid(axis="y", alpha=0.18)
        axes[1].spines[["top", "right"]].set_visible(False)
        axes[1].set_xticks(x, [f"Q{i:02d}" for i in x])
        axes[1].set_xlabel("档位（左侧低分，右侧高分）")
        fig.tight_layout(rect=[0, 0.02, 1, 0.90])
        fig.savefig(
            BASE_DIR
            / f"总体Score时序漂移_{file_label}_20等分{output_suffix}分布.png",
            dpi=190,
            bbox_inches="tight",
        )
        plt.close(fig)

    draw_distribution("策略收益", "策略收益")
    draw_distribution("Tag", "Tag")
    return distribution


raw_distribution = distribution_diagnostics("score", "修正前", "修正前")
adjusted_distribution = distribution_diagnostics(
    "adjusted_score", "时序加减修正后", "修正后"
)
pl.concat([raw_distribution, adjusted_distribution]).write_csv(
    BASE_DIR / "总体Score时序漂移_20等分分组分布.csv"
)


dev_summary = summary.filter(pl.col("数据段") == "开发段")
test_summary = summary.filter(pl.col("数据段") == "最终10%测试段")
dev_raw = dev_summary.filter(pl.col("方案") == "未修正_慢基准").row(0, named=True)
dev_adjusted = dev_summary.filter(pl.col("方案") == "加减修正_近期中心").row(0, named=True)
test_raw = test_summary.filter(pl.col("方案") == "未修正_慢基准").row(0, named=True)
test_adjusted = test_summary.filter(pl.col("方案") == "加减修正_近期中心").row(0, named=True)
dev_area_reduction = 1.0 - dev_adjusted["月度绝对面积均值"] / dev_raw["月度绝对面积均值"]
test_area_reduction = 1.0 - test_adjusted["月度绝对面积均值"] / test_raw["月度绝对面积均值"]
test_slope_reduction = 1.0 - test_adjusted["月度斜率均方"] / test_raw["月度斜率均方"]
report_lines = [
    "# 总体 Score 时序漂移加减修正：首次实测",
    "",
    "## 口径",
    "",
    f"- 冷启动：完整首个自然月，共 {cold_start_days_effective} 个交易日。",
    "- 慢速基准：截至前一日的历史日中位数之扩展中位数，会随样本增加而修正。",
    "- 近期中心：历史日中位数的因果 EWMA。",
    "- 加减修正：`adjusted_score = score - ((recent_center + balance_offset) - slow_baseline)`。",
    f"- 最终测试：最后 {test_days} 个交易日（{test_start_date} 至 {dates[-1]}），不参与选参。",
    "- Tag 不进入参数选择，仅用于事后方向与收益报告。",
    "",
    "## 选参结果",
    "",
    f"- 选定半衰期：{selected_half_life} 个交易日。",
    f"- 对应 alpha：{selected_alpha:.8f}。",
    f"- 开发段固定平衡偏移量：{selected_balance_offset:.8f}。",
    f"- 规则：{selection_note}",
    f"- 开发段月度面积下降：{dev_area_reduction:.2%}。",
    f"- 最终测试段月度面积下降：{test_area_reduction:.2%}。",
    f"- 最终测试段月度斜率均方下降：{test_slope_reduction:.2%}。",
    f"- 最终测试段做空占比：{test_adjusted['预测做空占比']:.2%}。",
    f"- 最终测试段平均策略收益率：修正前 {test_raw['平均策略收益率']:.8f}，修正后 {test_adjusted['平均策略收益率']:.8f}。",
    f"- 注意：选定1日半衰期的开发段日偏移绝对变化P95为 {selected['开发段日偏移绝对变化P95']:.6f}，月度平滑改善不代表日级修正同样平滑。",
    "",
    "## 结果文件",
    "",
    "- `总体Score时序漂移_参数搜索.csv`",
    "- `总体Score时序漂移_方案对比.csv`",
    "- `总体Score时序漂移_月度分位数.csv`",
    "- `总体Score时序漂移_每日状态.csv`",
    "- `总体Score时序漂移_月度分位数对比.png`",
    "- `总体Score时序漂移_修正后_20等分三指标.png`",
    "- `总体Score时序漂移_修正后_10等分累计收益.png`",
    "- `总体Score时序漂移_修正前_20等分三指标.png`",
    "- `总体Score时序漂移_修正前_10等分累计收益.png`",
    "- `总体Score时序漂移_20等分指标.csv`",
    "- `总体Score时序漂移_10等分累计收益.csv`",
    "- `总体Score时序漂移_修正后_20等分策略收益分布.png`",
    "- `总体Score时序漂移_修正后_20等分Tag分布.png`",
    "- `总体Score时序漂移_修正前_20等分策略收益分布.png`",
    "- `总体Score时序漂移_修正前_20等分Tag分布.png`",
    "- `总体Score时序漂移_20等分分组分布.csv`",
    "- `总体Score时序漂移_选定参数.json`",
    "",
    "## 开发段摘要",
    "",
    "```text",
    str(dev_summary),
    "```",
    "",
    "## 最终10%测试段摘要",
    "",
    "```text",
    str(test_summary),
    "```",
]
(BASE_DIR / "README_总体Score时序漂移_加减修正.md").write_text(
    "\n".join(report_lines), encoding="utf-8"
)


print("=== 数据切分 ===")
print(
    {
        "交易日总数": n_dates,
        "冷启动交易日": cold_start_days_effective,
        "开发段最后日期": str(dates[test_start_index - 1]),
        "测试段起止": f"{test_start_date} 至 {dates[-1]}",
        "测试交易日": test_days,
    }
)
print("\n=== 参数搜索 ===")
print(search)
print("\n=== 选定参数 ===")
print(selected_config)
print("\n=== 方案对比 ===")
print(summary)
