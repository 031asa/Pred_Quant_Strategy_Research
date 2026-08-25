from __future__ import annotations

"""14:30 前后独立的 Score 漂移选参和绘图实现。"""

from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", ".")) / "pred_drift_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from utils.segmented_drift import (
    SEGMENT_AFTER,
    SEGMENT_BEFORE,
    SEGMENT_ORDER,
    alpha_from_half_life,
    build_causal_state,
)
from utils.plot_fonts import register_matplotlib_chinese_font


@dataclass(frozen=True)
class DriftAnalysisResult:
    """分时漂移选参的输出位置和验收摘要。"""

    parameters_path: Path
    figure_paths: tuple[Path, Path]
    source_sample_count: int
    effective_sample_count: int
    selected_half_life_by_segment: dict[str, int]
    development_short_share_by_segment: dict[str, float]


def safe_rate(numerator: int, denominator: int):
    return float(numerator / denominator) if denominator else None


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
    return area, slope, max_abs


def state_frame(
    segment: str,
    daily_medians: np.ndarray,
    dates,
    half_life: int,
    cold_start_days_effective: int,
):
    arrays = build_causal_state(daily_medians, cold_start_days_effective, half_life)
    return pl.DataFrame(
        {
            "Date": dates,
            "date_index": np.arange(len(dates), dtype=np.int64),
            "时段": [segment] * len(dates),
            **arrays,
        }
    ).with_columns(pl.col("Date").cast(pl.Date))


def attach_state(
    part: pl.DataFrame,
    state: pl.DataFrame,
    test_start_index: int,
    balance_offset: float = 0.0,
):
    return (
        part.join(
            state.select(["Date", "slow_baseline", "recent_center"]),
            on="Date",
            how="left",
        )
        .filter(pl.col("slow_baseline").is_finite() & pl.col("recent_center").is_finite())
        .with_columns(
            (pl.col("recent_center") + pl.lit(balance_offset)).alias("effective_center")
        )
        .with_columns(
            (pl.col("effective_center") - pl.col("slow_baseline")).alias("time_offset"),
            (pl.col("score") - pl.col("slow_baseline")).alias("raw_residual"),
            (pl.col("score") < pl.col("slow_baseline")).alias("raw_short"),
            (pl.col("date_index") >= test_start_index).alias("is_test"),
        )
        .with_columns(
            (pl.col("score") - pl.col("time_offset")).alias("adjusted_score"),
            (pl.col("score") - pl.col("effective_center")).alias("adjusted_residual"),
            (pl.col("score") < pl.col("effective_center")).alias("adjusted_short"),
        )
    )


def summarize_period(
    segment: str,
    period_label: str,
    data: pl.DataFrame,
    method: str,
    short_rate_target: float,
):
    if method == "修正前_分段慢基准":
        short = data["raw_short"].to_numpy()
        residual_column = "raw_residual"
    else:
        short = data["adjusted_short"].to_numpy()
        residual_column = "adjusted_residual"

    tag = data["tag"].to_numpy()
    direction = np.where(short, -1, 1)
    strategy_return = direction * tag
    short_n = int(short.sum())
    long_n = int((~short).sum())
    short_wins = int(np.sum(short & (tag < 0)))
    long_wins = int(np.sum((~short) & (tag > 0)))
    area, slope, max_abs = monthly_curve_metrics(data, residual_column)
    return {
        "时段": segment,
        "数据段": period_label,
        "方案": method,
        "样本量": data.height,
        "交易日数": data["Date"].n_unique(),
        "实际负类占比": safe_rate(int(np.sum(tag < 0)), data.height),
        "预测做空占比": safe_rate(short_n, data.height),
        "预测做多占比": safe_rate(long_n, data.height),
        "偏离50%": abs(safe_rate(short_n, data.height) - short_rate_target),
        "做空胜率_precision": safe_rate(short_wins, short_n),
        "做多胜率_precision": safe_rate(long_wins, long_n),
        "总体方向准确率": safe_rate(short_wins + long_wins, data.height),
        "平均策略收益率": float(np.mean(strategy_return)),
        "月度绝对面积均值": area,
        "月度斜率均方": slope,
        "最大月度中位偏移": max_abs,
    }


def direction_change_rows(segment: str, period_label: str, data: pl.DataFrame):
    raw_short = data["raw_short"].to_numpy()
    adjusted_short = data["adjusted_short"].to_numpy()
    tag = data["tag"].to_numpy()
    rows = []
    transitions = [
        (raw_short & ~adjusted_short, "慢基准空→修正后多", 1),
        (~raw_short & adjusted_short, "慢基准多→修正后空", -1),
    ]
    for mask, label, new_direction in transitions:
        sample_n = int(mask.sum())
        sample_tag = tag[mask]
        strategy_return = new_direction * sample_tag
        rows.append(
            {
                "时段": segment,
                "数据段": period_label,
                "方向变化": label,
                "样本量": sample_n,
                "占该数据段比例": safe_rate(sample_n, data.height),
                "新方向胜率": float(np.mean(strategy_return > 0)) if sample_n else None,
                "Tag负类比例": float(np.mean(sample_tag < 0)) if sample_n else None,
                "Tag正类比例": float(np.mean(sample_tag > 0)) if sample_n else None,
                "Tag零值比例": float(np.mean(sample_tag == 0)) if sample_n else None,
                "平均新策略收益率": float(np.mean(strategy_return)) if sample_n else None,
            }
        )
    return rows


def plot_segment(
    segment: str,
    monthly: pl.DataFrame,
    selected_half_life: int,
    test_start_date,
    output_path: Path,
):
    month_dates = [datetime.strptime(value, "%Y-%m") for value in monthly["Month"].to_list()]
    baseline = monthly["慢速基准中位数"].to_numpy()
    test_mark = datetime(test_start_date.year, test_start_date.month, 1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    axes[0].fill_between(
        month_dates, monthly["原始P10"], monthly["原始P90"],
        alpha=0.16, color="#7aa6d8", label="P10～P90",
    )
    axes[0].fill_between(
        month_dates, monthly["原始P25"], monthly["原始P75"],
        alpha=0.28, color="#3f83c5", label="P25～P75",
    )
    axes[0].plot(month_dates, monthly["原始P50"], color="#1558a0", marker="o", linewidth=2, label="原始P50")
    axes[0].plot(month_dates, baseline, color="#4f5964", linestyle="--", linewidth=1.6, label="分段慢基准")
    axes[0].set_title(f"A. {segment} score月度分位数（修正前）")
    axes[0].set_ylabel("score")
    axes[0].legend(ncol=4, loc="best")

    axes[1].fill_between(
        month_dates, monthly["修正P10"], monthly["修正P90"],
        alpha=0.16, color="#7bbf9e", label="P10～P90",
    )
    axes[1].fill_between(
        month_dates, monthly["修正P25"], monthly["修正P75"],
        alpha=0.28, color="#2f9c67", label="P25～P75",
    )
    axes[1].plot(month_dates, monthly["修正P50"], color="#147a4b", marker="o", linewidth=2, label="修正P50")
    axes[1].plot(month_dates, baseline, color="#4f5964", linestyle="--", linewidth=1.6, label="分段慢基准")
    axes[1].set_title(f"B. {segment}纯加减修正（半衰期={selected_half_life}日）")
    axes[1].set_ylabel("adjusted score")
    axes[1].legend(ncol=4, loc="best")

    axes[2].axhline(0.0, color="#4f5964", linestyle="--", linewidth=1.4, label="基准=0")
    axes[2].plot(month_dates, monthly["原始相对基准偏移"], color="#b44b4b", marker="o", linewidth=1.8, label="修正前")
    axes[2].plot(month_dates, monthly["修正后相对基准偏移"], color="#147a4b", marker="o", linewidth=2, label="修正后")
    axes[2].set_title(f"C. {segment}月度中位数相对分段慢基准的偏移")
    axes[2].set_ylabel("中位数偏移")
    axes[2].legend(loc="best")

    for axis in axes:
        axis.axvline(test_mark, color="#8a5a9f", linestyle=":", linewidth=2)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("月份")
    fig.suptitle(
        f"{segment} Score时序漂移：独立扩展基准 + 独立EWMA近期中心",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def analyze_segmented_score_drift(
    *,
    config_path: Path,
    source_path: Path,
    parameters_output_path: Path,
    before_1430_figure_path: Path,
    after_1430_figure_path: Path,
    datetime_column: str,
    pred_column: str,
    tag_column: str,
    time_cutoff: str,
    score_offset: float,
    score_min: float,
    score_max: float,
    cold_start_days: int,
    test_fraction: float,
    short_rate_target: float,
    short_rate_tolerance: float,
    half_life_candidates: tuple[int, ...],
    expected_source_sample_count: int,
) -> DriftAnalysisResult:
    """按 14:30 前后独立选参，只输出参数 JSON 和两张月度漂移图。"""

    config_path = Path(config_path)
    source_path = Path(source_path)
    parameters_output_path = Path(parameters_output_path)
    figure_paths = {
        SEGMENT_BEFORE: Path(before_1430_figure_path),
        SEGMENT_AFTER: Path(after_1430_figure_path),
    }
    parameters_output_path.parent.mkdir(parents=True, exist_ok=True)
    for path in figure_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    SOURCE = source_path
    source_path_for_config = source_path.relative_to(config_path.parent).as_posix()
    DT = datetime_column
    PRED = pred_column
    TAG = tag_column
    CUTOFF = time_cutoff
    COLD_START_DAYS = cold_start_days
    TEST_FRACTION = test_fraction
    SHORT_RATE_TARGET = short_rate_target
    SHORT_RATE_TOLERANCE = short_rate_tolerance
    HALF_LIFE_CANDIDATES = half_life_candidates
    
    df = (
        pl.read_parquet(SOURCE)
        .select([DT, PRED, TAG])
        .drop_nulls()
        .with_columns(
            (pl.col(PRED).cast(pl.Float64) + score_offset)
            .clip(score_min, score_max)
            .alias("score"),
            pl.col(TAG).cast(pl.Float64).alias("tag"),
            pl.col(DT).dt.date().alias("Date"),
            pl.col(DT).dt.strftime("%Y-%m").alias("Month"),
            pl.when(pl.col(DT).dt.strftime("%H:%M") < CUTOFF)
            .then(pl.lit(SEGMENT_BEFORE))
            .otherwise(pl.lit(SEGMENT_AFTER))
            .alias("时段"),
        )
        .select([DT, "Date", "Month", "时段", "score", "tag"])
    )
    
    daily = (
        df.group_by(["时段", "Date"])
        .agg(
            pl.col("score").median().alias("daily_score_median"),
            pl.len().alias("daily_samples"),
        )
        .sort(["时段", "Date"])
    )
    dates = sorted(df["Date"].unique().to_list())
    n_dates = len(dates)
    date_index = pl.DataFrame({"Date": dates, "date_index": np.arange(n_dates, dtype=np.int64)}).with_columns(pl.col("Date").cast(pl.Date))
    test_days = int(math.ceil(n_dates * TEST_FRACTION))
    test_start_index = n_dates - test_days
    test_start_date = dates[test_start_index]
    first_month_key = (dates[0].year, dates[0].month)
    first_next_month_index = next(
        (index for index, value in enumerate(dates) if (value.year, value.month) != first_month_key),
        n_dates,
    )
    cold_start_days_effective = max(COLD_START_DAYS, first_next_month_index)
    if n_dates <= cold_start_days_effective + test_days:
        raise ValueError("交易日数量不足以同时满足冷启动期和测试集要求。")
    
    for segment in SEGMENT_ORDER:
        if daily.filter(pl.col("时段") == segment)["Date"].n_unique() != n_dates:
            raise ValueError(f"{segment}并非每个交易日都有样本，无法使用共同日期切分。")
    
    df = df.join(date_index, on="Date", how="left")
    source_sample_n = df.height
    search_frames = []
    summary_rows = []
    change_rows = []
    monthly_frames = []
    daily_state_frames = []
    selected_configs = {}
    evaluation_sample_n = 0
    
    register_matplotlib_chinese_font()
    
    for segment in SEGMENT_ORDER:
        part = df.filter(pl.col("时段") == segment)
        segment_daily = daily.filter(pl.col("时段") == segment).sort("Date")
        daily_medians = segment_daily["daily_score_median"].to_numpy()
        candidate_rows = []
        states = {}
    
        for half_life in HALF_LIFE_CANDIDATES:
            state = state_frame(
                segment,
                daily_medians,
                dates,
                half_life,
                cold_start_days_effective,
            )
            states[half_life] = state
            uncalibrated = attach_state(
                part,
                state,
                test_start_index,
            ).filter(pl.col("date_index") < test_start_index)
            balance_offset = float((uncalibrated["score"] - uncalibrated["recent_center"]).median())
            candidate = attach_state(
                part,
                state,
                test_start_index,
                balance_offset,
            ).filter(pl.col("date_index") < test_start_index)
            short_rate = float(candidate["adjusted_short"].mean())
            area, slope, max_abs = monthly_curve_metrics(candidate, "adjusted_residual")
            state_dev = state.filter(
                (pl.col("date_index") >= cold_start_days_effective)
                & (pl.col("date_index") < test_start_index)
            )
            daily_offset = state_dev["time_offset"].drop_nulls().to_numpy()
            offset_change = np.diff(daily_offset)
            candidate_rows.append(
                {
                    "时段": segment,
                    "半衰期_交易日": half_life,
                    "alpha": alpha_from_half_life(half_life),
                    "开发段固定平衡偏移量": balance_offset,
                    "开发段做空占比": short_rate,
                    "偏离50%": abs(short_rate - SHORT_RATE_TARGET),
                    "满足49%至51%硬约束": abs(short_rate - SHORT_RATE_TARGET) <= SHORT_RATE_TOLERANCE,
                    "开发段月度绝对面积均值": area,
                    "开发段月度斜率均方": slope,
                    "开发段最大月度中位偏移": max_abs,
                    "开发段日偏移变化标准差": float(np.std(offset_change, ddof=0)),
                    "开发段日偏移绝对变化P95": float(np.quantile(np.abs(offset_change), 0.95)),
                    "开发段日偏移绝对变化最大值": float(np.max(np.abs(offset_change))),
                }
            )
    
        search = pl.DataFrame(candidate_rows).sort(
            ["满足49%至51%硬约束", "开发段月度绝对面积均值", "开发段月度斜率均方"],
            descending=[True, False, False],
        )
        feasible = search.filter(pl.col("满足49%至51%硬约束"))
        if feasible.height:
            selected = feasible.sort(["开发段月度绝对面积均值", "开发段月度斜率均方"]).row(0, named=True)
            selection_note = "满足本时段开发段做空占比49%至51%后，优先最小化月度绝对面积，月度斜率作为次级排序。"
        else:
            selected = search.sort(["偏离50%", "开发段月度绝对面积均值", "开发段月度斜率均方"]).row(0, named=True)
            selection_note = "没有候选满足49%至51%，先最小化做空比例偏差，再比较月度面积与斜率。"
    
        selected_half_life = int(selected["半衰期_交易日"])
        selected_balance_offset = float(selected["开发段固定平衡偏移量"])
        selected_state = states[selected_half_life]
        evaluation = attach_state(
            part,
            selected_state,
            test_start_index,
            selected_balance_offset,
        )
        development_short_rate = float(
            evaluation.filter(pl.col("date_index") < test_start_index)["adjusted_short"].mean()
        )
        if abs(development_short_rate - SHORT_RATE_TARGET) > SHORT_RATE_TOLERANCE:
            raise AssertionError(
                f"{segment}开发段做空占比{development_short_rate:.4%}，超出49%至51%约束。"
            )
        evaluation_sample_n += evaluation.height
    
        periods = [
            ("开发段", evaluation.filter(pl.col("date_index") < test_start_index)),
            ("最终60日测试段", evaluation.filter(pl.col("date_index") >= test_start_index)),
            ("全部有效段", evaluation),
        ]
        for period_label, period_data in periods:
            summary_rows.append(
                summarize_period(
                    segment,
                    period_label,
                    period_data,
                    "修正前_分段慢基准",
                    SHORT_RATE_TARGET,
                )
            )
            summary_rows.append(
                summarize_period(
                    segment,
                    period_label,
                    period_data,
                    "修正后_分段近期中心",
                    SHORT_RATE_TARGET,
                )
            )
            change_rows.extend(direction_change_rows(segment, period_label, period_data))
    
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
            .with_columns(pl.lit(segment).alias("时段"))
            .select(["时段", pl.exclude("时段")])
            .sort("Month")
        )
        monthly_frames.append(monthly)
    
        daily_state = (
            selected_state.filter(pl.col("slow_baseline").is_finite() & pl.col("recent_center").is_finite())
            .join(segment_daily.select(["Date", "daily_score_median", "daily_samples"]), on="Date", how="left")
            .with_columns(
                (pl.col("recent_center") + pl.lit(selected_balance_offset)).alias("effective_center"),
                (pl.col("date_index") >= test_start_index).alias("is_test"),
            )
            .with_columns(
                (pl.col("effective_center") - pl.col("slow_baseline")).alias("effective_time_offset"),
                (pl.col("daily_score_median") - pl.col("effective_center")).alias("daily_residual_after_adjustment"),
            )
        )
        daily_state_frames.append(daily_state)
        search_frames.append(search)
        plot_segment(
            segment,
            monthly,
            selected_half_life,
            test_start_date,
            figure_paths[segment],
        )
    
        selected_configs[segment] = {
            "selected_half_life_trading_days": selected_half_life,
            "selected_alpha": float(selected["alpha"]),
            "selected_development_balance_offset": selected_balance_offset,
            "selected_development_daily_offset_abs_change_p95": float(selected["开发段日偏移绝对变化P95"]),
            "selected_development_daily_offset_abs_change_max": float(selected["开发段日偏移绝对变化最大值"]),
            "selection_note": selection_note,
        }
    
    expected_evaluation_n = df.filter(pl.col("date_index") >= cold_start_days_effective).height
    if evaluation_sample_n != expected_evaluation_n:
        raise AssertionError(
            f"分时有效样本合计{evaluation_sample_n:,}，预期{expected_evaluation_n:,}；存在遗漏或重复。"
        )
    if source_sample_n != expected_source_sample_count:
        raise AssertionError(
            f"源数据样本量变为{source_sample_n:,}，"
            f"与项目基准{expected_source_sample_count:,}不一致。"
        )
    
    selected_config = {
        "source_path": source_path_for_config,
        "time_cutoff": CUTOFF,
        "segment_rule": f"Datetime < {CUTOFF} is {SEGMENT_BEFORE}; equality belongs to {SEGMENT_AFTER}.",
        "cold_start_minimum_trading_days": COLD_START_DAYS,
        "cold_start_effective_trading_days": cold_start_days_effective,
        "cold_start_calendar_month": f"{dates[0].year:04d}-{dates[0].month:02d}",
        "total_trading_days": n_dates,
        "test_fraction": TEST_FRACTION,
        "test_trading_days": test_days,
        "test_start_date": str(test_start_date),
        "test_end_date": str(dates[-1]),
        "short_rate_target_per_segment": SHORT_RATE_TARGET,
        "short_rate_tolerance": SHORT_RATE_TOLERANCE,
        "causal_rule": "Date t uses only the same segment's daily medians from dates strictly earlier than t.",
        "tag_used_for_selection": False,
        "baseline_definition": "Per-segment expanding median of historical daily score medians.",
        "transform": "adjusted_score = score - ((segment_recent_center + segment_balance_offset) - segment_slow_baseline)",
        "segments": selected_configs,
    }
    with parameters_output_path.open("w", encoding="utf-8") as file:
        json.dump(selected_config, file, ensure_ascii=False, indent=2)

    return DriftAnalysisResult(
        parameters_path=parameters_output_path,
        figure_paths=(figure_paths[SEGMENT_BEFORE], figure_paths[SEGMENT_AFTER]),
        source_sample_count=source_sample_n,
        effective_sample_count=evaluation_sample_n,
        selected_half_life_by_segment={
            segment: int(config["selected_half_life_trading_days"])
            for segment, config in selected_configs.items()
        },
        development_short_share_by_segment={
            row["时段"]: float(row["预测做空占比"])
            for row in summary_rows
            if row["数据段"] == "开发段" and row["方案"] == "修正后_分段近期中心"
        },
    )
