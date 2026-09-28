# %% [markdown]
# # Plan B胜率与Tag周期诊断
#
# 本阶段只比较多空方向、方向转换与胜率，不计算累计收益。
# 同时从当前parquet内部检查Tag的连续重复、唯一值、典型日、日内分布与滞后相关。

# %%
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np
import polars as pl


# %%
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

CONFIG_PATH = BASE_DIR / "中位数阈值_20与50等分_参数.json"
OUTPUT_JSON = BASE_DIR / "PlanB胜率与Tag周期诊断_中间数据.json"
OUTPUT_WIN_IMAGE = BASE_DIR / "PlanB_多空胜率与样本占比.png"
OUTPUT_TRANSITION_IMAGE = BASE_DIR / "PlanB_方向转换效果.png"
OUTPUT_TYPICAL_IMAGE = BASE_DIR / "Tag周期诊断_典型日曲线.png"
OUTPUT_DIAG_IMAGE = BASE_DIR / "Tag周期诊断_滞后相关与日内结构.png"
OUTPUT_README = BASE_DIR / "README_PlanB胜率与Tag周期诊断.md"

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = json.load(config_file)

SOURCE_PATH = Path(CONFIG["source_path"])
PRED_COLUMN = CONFIG["pred_column"]
RETURN_COLUMN = CONFIG["return_column"]
DATETIME_COLUMN = CONFIG["datetime_column"]
SCORE_OFFSET = float(CONFIG["score_offset"])
SCORE_MIN = float(CONFIG["score_min"])
SCORE_MAX = float(CONFIG["score_max"])

CLASSIFICATION_CENTER = 0.5
ROLLING_DAYS = 20
CORRECTION_STRENGTHS = [0.0, 0.25, 0.50, 0.75, 1.00]
LAGS = [1, 2, 5, 10, 15, 20, 29, 30, 45, 60]
NEAR_REPEAT_TOLERANCE = 1e-8


def configure_chinese_font() -> None:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def minute_text(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def direction_name(value: int) -> str:
    return "做多" if value == 1 else "做空" if value == -1 else "中性"


def load_source() -> pl.DataFrame:
    return (
        pl.read_parquet(SOURCE_PATH)
        .select([DATETIME_COLUMN, "Contract", PRED_COLUMN, RETURN_COLUMN])
        .drop_nulls()
        .with_columns(
            (pl.col(PRED_COLUMN).cast(pl.Float64) + SCORE_OFFSET)
            .clip(SCORE_MIN, SCORE_MAX)
            .alias("score"),
            pl.col(RETURN_COLUMN).cast(pl.Float64).alias("tag"),
            pl.col(DATETIME_COLUMN).dt.date().alias("Date"),
            pl.col(DATETIME_COLUMN).dt.strftime("%Y-%m").alias("month"),
            (
                pl.col(DATETIME_COLUMN).dt.hour().cast(pl.Int32) * 60
                + pl.col(DATETIME_COLUMN).dt.minute().cast(pl.Int32)
            ).alias("minute"),
        )
        .with_columns(
            pl.when(pl.col("minute") < 14 * 60 + 30)
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .cast(pl.Int8)
            .alias("segment_code")
        )
        .select(
            [
                DATETIME_COLUMN,
                "Date",
                "month",
                "Contract",
                "minute",
                "segment_code",
                "score",
                "tag",
            ]
        )
    )


def build_rolling_centers(source: pl.DataFrame) -> pl.DataFrame:
    """分别在两个时段内，用此前20个交易日的全部score估计中位数。"""
    rows: list[dict] = []
    for segment_code in [0, 1]:
        segment = source.filter(pl.col("segment_code") == segment_code)
        dates = segment["Date"].unique().sort().to_list()
        daily_values = {
            current_date: segment.filter(pl.col("Date") == current_date)["score"].to_numpy()
            for current_date in dates
        }
        for index, current_date in enumerate(dates):
            if index < ROLLING_DAYS:
                center = None
            else:
                history = np.concatenate(
                    [daily_values[value] for value in dates[index - ROLLING_DAYS : index]]
                )
                center = float(np.median(history))
            rows.append(
                {
                    "Date": current_date,
                    "segment_code": segment_code,
                    "rolling_center": center,
                }
            )
    return pl.DataFrame(rows).with_columns(pl.col("Date").cast(pl.Date))


def mask_summary(
    tag: np.ndarray,
    raw_direction: np.ndarray,
    adjusted_direction: np.ndarray,
    mask: np.ndarray,
) -> dict:
    selected_tag = tag[mask]
    raw = raw_direction[mask]
    adjusted = adjusted_direction[mask]
    n = int(mask.sum())
    long_mask = adjusted == 1
    short_mask = adjusted == -1
    neutral_mask = adjusted == 0
    long_n = int(long_mask.sum())
    short_n = int(short_mask.sum())
    neutral_n = int(neutral_mask.sum())
    long_wins = int(np.sum(long_mask & (selected_tag > 0)))
    short_wins = int(np.sum(short_mask & (selected_tag < 0)))
    return {
        "sample_n": n,
        "long_n": long_n,
        "long_share": safe_rate(long_n, n),
        "long_wins": long_wins,
        "long_win_rate": safe_rate(long_wins, long_n),
        "short_n": short_n,
        "short_share": safe_rate(short_n, n),
        "short_wins": short_wins,
        "short_win_rate": safe_rate(short_wins, short_n),
        "neutral_n": neutral_n,
        "neutral_share": safe_rate(neutral_n, n),
        "direction_change_n": int(np.sum(raw != adjusted)),
        "direction_change_rate": safe_rate(int(np.sum(raw != adjusted)), n),
    }


def build_plan_b_results(source: pl.DataFrame) -> dict:
    centers = build_rolling_centers(source)
    evaluation = (
        source.join(centers, on=["Date", "segment_code"], how="left")
        .filter(pl.col("rolling_center").is_not_null())
        .sort(["Date", DATETIME_COLUMN, "Contract"])
    )

    score = evaluation["score"].to_numpy()
    tag = evaluation["tag"].to_numpy()
    rolling_center = evaluation["rolling_center"].to_numpy()
    segment_code = evaluation["segment_code"].to_numpy()
    month_values = np.asarray(evaluation["month"].to_list())
    months = sorted(set(month_values.tolist()))
    raw_direction = np.where(
        score > CLASSIFICATION_CENTER,
        1,
        np.where(score < CLASSIFICATION_CENTER, -1, 0),
    ).astype(np.int8)

    summary_rows: list[dict] = []
    monthly_rows: list[dict] = []
    transition_rows: list[dict] = []

    segment_definitions = [
        ("全体样本", np.ones(len(score), dtype=bool)),
        ("14:30以前", segment_code == 0),
        ("14:30及以后", segment_code == 1),
    ]

    for strength in CORRECTION_STRENGTHS:
        adjusted_score = np.clip(
            score - strength * (rolling_center - CLASSIFICATION_CENTER),
            SCORE_MIN,
            SCORE_MAX,
        )
        adjusted_direction = np.where(
            adjusted_score > CLASSIFICATION_CENTER,
            1,
            np.where(adjusted_score < CLASSIFICATION_CENTER, -1, 0),
        ).astype(np.int8)

        for segment_label, segment_mask in segment_definitions:
            summary_rows.append(
                {
                    "segment": segment_label,
                    "strength": strength,
                    **mask_summary(
                        tag,
                        raw_direction,
                        adjusted_direction,
                        segment_mask,
                    ),
                }
            )

            for month in months:
                mask = segment_mask & (month_values == month)
                if not mask.any():
                    continue
                monthly_rows.append(
                    {
                        "segment": segment_label,
                        "month": month,
                        "strength": strength,
                        **mask_summary(
                            tag,
                            raw_direction,
                            adjusted_direction,
                            mask,
                        ),
                    }
                )

            transitions = [
                (1, 1, "多头→仍做多"),
                (1, -1, "多头→转为空头"),
                (-1, -1, "空头→仍做空"),
                (-1, 1, "空头→转为多头"),
                (0, 1, "中性→做多"),
                (0, -1, "中性→做空"),
            ]
            for raw_value, adjusted_value, transition_label in transitions:
                mask = (
                    segment_mask
                    & (raw_direction == raw_value)
                    & (adjusted_direction == adjusted_value)
                )
                n = int(mask.sum())
                if adjusted_value == 1:
                    wins = int(np.sum(mask & (tag > 0)))
                else:
                    wins = int(np.sum(mask & (tag < 0)))
                transition_rows.append(
                    {
                        "segment": segment_label,
                        "strength": strength,
                        "transition": transition_label,
                        "sample_n": n,
                        "sample_share": safe_rate(n, int(segment_mask.sum())),
                        "wins": wins,
                        "win_rate": safe_rate(wins, n),
                        "tag_positive_rate": safe_rate(int(np.sum(mask & (tag > 0))), n),
                        "tag_negative_rate": safe_rate(int(np.sum(mask & (tag < 0))), n),
                    }
                )

    center_rows = [
        {
            "date": str(row["Date"]),
            "segment": "14:30以前" if row["segment_code"] == 0 else "14:30及以后",
            "rolling_center": row["rolling_center"],
            "estimated_bias": (
                None
                if row["rolling_center"] is None
                else row["rolling_center"] - CLASSIFICATION_CENTER
            ),
        }
        for row in centers.iter_rows(named=True)
    ]
    return {
        "summary": summary_rows,
        "monthly": monthly_rows,
        "transitions": transition_rows,
        "rolling_centers": center_rows,
        "evaluation_n": evaluation.height,
        "warmup_days": ROLLING_DAYS,
    }


def update_corr(accumulator: dict, key: tuple, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) == 0:
        return
    current = accumulator[key]
    current["n"] += len(x)
    current["sum_x"] += float(np.sum(x))
    current["sum_y"] += float(np.sum(y))
    current["sum_xx"] += float(np.dot(x, x))
    current["sum_yy"] += float(np.dot(y, y))
    current["sum_xy"] += float(np.dot(x, y))


def finish_corr(values: dict) -> float | None:
    n = values["n"]
    if n < 2:
        return None
    covariance = values["sum_xy"] - values["sum_x"] * values["sum_y"] / n
    variance_x = values["sum_xx"] - values["sum_x"] ** 2 / n
    variance_y = values["sum_yy"] - values["sum_y"] ** 2 / n
    denominator = np.sqrt(max(variance_x, 0) * max(variance_y, 0))
    return float(covariance / denominator) if denominator > 0 else None


def build_tag_diagnostics(source: pl.DataFrame) -> dict:
    ordered = source.sort(["Date", "Contract", "segment_code", "minute"])
    date_values = np.asarray([value.toordinal() for value in ordered["Date"].to_list()])
    contract_values = np.asarray(ordered["Contract"].to_list())
    segment_values = ordered["segment_code"].to_numpy()
    minute_values = ordered["minute"].to_numpy()
    tag_values = ordered["tag"].to_numpy()

    boundaries = np.flatnonzero(
        np.r_[
            True,
            (date_values[1:] != date_values[:-1])
            | (contract_values[1:] != contract_values[:-1])
            | (segment_values[1:] != segment_values[:-1]),
            True,
        ]
    )

    group_rows: list[dict] = []
    run_rows: list[dict] = []
    correlation_accumulators = defaultdict(
        lambda: {
            "n": 0,
            "sum_x": 0.0,
            "sum_y": 0.0,
            "sum_xx": 0.0,
            "sum_yy": 0.0,
            "sum_xy": 0.0,
        }
    )
    minute_repeat = defaultdict(lambda: {"pairs": 0, "exact": 0, "near": 0})

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tags = tag_values[start:end]
        minutes = minute_values[start:end]
        current_date = date.fromordinal(int(date_values[start]))
        contract = str(contract_values[start])
        segment_code = int(segment_values[start])
        segment = "14:30以前" if segment_code == 0 else "14:30及以后"

        if len(tags) > 1:
            adjacent_diff = np.abs(np.diff(tags))
            exact_repeat = adjacent_diff == 0
            near_repeat = adjacent_diff < NEAR_REPEAT_TOLERANCE
            for position, minute in enumerate(minutes[1:]):
                key = (segment, int(minute))
                minute_repeat[key]["pairs"] += 1
                minute_repeat[key]["exact"] += int(exact_repeat[position])
                minute_repeat[key]["near"] += int(near_repeat[position])
        else:
            exact_repeat = np.array([], dtype=bool)
            near_repeat = np.array([], dtype=bool)

        changes = np.r_[True, tags[1:] != tags[:-1], True]
        run_boundaries = np.flatnonzero(changes)
        run_lengths = np.diff(run_boundaries)
        for run_start, run_end, run_length in zip(
            run_boundaries[:-1], run_boundaries[1:], run_lengths
        ):
            if run_length >= 2:
                run_rows.append(
                    {
                        "date": str(current_date),
                        "contract": contract,
                        "segment": segment,
                        "start_time": minute_text(int(minutes[run_start])),
                        "end_time": minute_text(int(minutes[run_end - 1])),
                        "run_length": int(run_length),
                        "tag_raw": float(tags[run_start]),
                        "tag_bp": float(tags[run_start] * 10000),
                    }
                )

        group_rows.append(
            {
                "date": str(current_date),
                "contract": contract,
                "segment": segment,
                "row_n": int(len(tags)),
                "unique_tag_n": int(len(np.unique(tags))),
                "unique_ratio": safe_rate(int(len(np.unique(tags))), int(len(tags))),
                "adjacent_pairs": int(max(len(tags) - 1, 0)),
                "exact_repeat_n": int(exact_repeat.sum()),
                "exact_repeat_rate": safe_rate(int(exact_repeat.sum()), len(exact_repeat)),
                "near_repeat_n": int(near_repeat.sum()),
                "near_repeat_rate": safe_rate(int(near_repeat.sum()), len(near_repeat)),
                "median_run_length": float(np.median(run_lengths)),
                "max_run_length": int(run_lengths.max()),
            }
        )

        for lag in LAGS:
            if len(tags) <= lag:
                continue
            x = tags[:-lag]
            y = tags[lag:]
            update_corr(correlation_accumulators, (segment, contract, lag), x, y)
            update_corr(correlation_accumulators, (segment, "全部合约", lag), x, y)

    group_frame = pl.DataFrame(group_rows)
    aggregate_rows: list[dict] = []
    for segment in ["14:30以前", "14:30及以后"]:
        segment_frame = group_frame.filter(pl.col("segment") == segment)
        for contract in ["全部合约", *sorted(segment_frame["contract"].unique().to_list())]:
            frame = (
                segment_frame
                if contract == "全部合约"
                else segment_frame.filter(pl.col("contract") == contract)
            )
            adjacent_pairs = int(frame["adjacent_pairs"].sum())
            exact_repeat_n = int(frame["exact_repeat_n"].sum())
            near_repeat_n = int(frame["near_repeat_n"].sum())
            aggregate_rows.append(
                {
                    "segment": segment,
                    "contract": contract,
                    "contract_day_groups": frame.height,
                    "total_rows": int(frame["row_n"].sum()),
                    "weighted_unique_ratio": safe_rate(
                        int(frame["unique_tag_n"].sum()), int(frame["row_n"].sum())
                    ),
                    "exact_repeat_rate": safe_rate(exact_repeat_n, adjacent_pairs),
                    "near_repeat_rate": safe_rate(near_repeat_n, adjacent_pairs),
                    "median_max_run": float(frame["max_run_length"].median()),
                    "p95_max_run": float(frame["max_run_length"].quantile(0.95)),
                    "max_run": int(frame["max_run_length"].max()),
                }
            )

    acf_rows = []
    for (segment, contract, lag), values in sorted(correlation_accumulators.items()):
        acf_rows.append(
            {
                "segment": segment,
                "contract": contract,
                "lag_minutes": lag,
                "pair_n": int(values["n"]),
                "correlation": finish_corr(values),
            }
        )

    intraday = (
        source.group_by(["segment_code", "minute"])
        .agg(
            pl.len().alias("sample_n"),
            pl.col("tag").mean().alias("tag_mean"),
            pl.col("tag").std().alias("tag_std"),
            pl.col("tag").quantile(0.10).alias("tag_p10"),
            pl.col("tag").median().alias("tag_median"),
            pl.col("tag").quantile(0.90).alias("tag_p90"),
            (pl.col("tag") > 0).mean().alias("tag_positive_rate"),
        )
        .sort("minute")
    )
    intraday_rows = []
    for row in intraday.iter_rows(named=True):
        segment = "14:30以前" if row["segment_code"] == 0 else "14:30及以后"
        repeat = minute_repeat.get((segment, int(row["minute"])), {})
        pairs = repeat.get("pairs", 0)
        intraday_rows.append(
            {
                "segment": segment,
                "minute": int(row["minute"]),
                "time": minute_text(int(row["minute"])),
                "sample_n": int(row["sample_n"]),
                "tag_mean": float(row["tag_mean"]),
                "tag_std": float(row["tag_std"]),
                "tag_p10": float(row["tag_p10"]),
                "tag_median": float(row["tag_median"]),
                "tag_p90": float(row["tag_p90"]),
                "tag_positive_rate": float(row["tag_positive_rate"]),
                "adjacent_pair_n": int(pairs),
                "exact_repeat_rate": safe_rate(repeat.get("exact", 0), pairs),
                "near_repeat_rate": safe_rate(repeat.get("near", 0), pairs),
            }
        )

    daily_after = (
        source.filter(pl.col("segment_code") == 1)
        .group_by("Date")
        .agg(pl.col("tag").mean().alias("tag_mean"))
        .sort("Date")
    )
    daily_dates = daily_after["Date"].to_list()
    daily_means = daily_after["tag_mean"].to_numpy()
    max_index = int(np.argmax(daily_means))
    min_index = int(np.argmin(daily_means))
    median_index = int(np.argmin(np.abs(daily_means - np.median(daily_means))))
    selected = [
        (daily_dates[max_index], "尾盘Tag均值最高日"),
        (daily_dates[min_index], "尾盘Tag均值最低日"),
        (daily_dates[median_index], "尾盘Tag均值中位附近日"),
    ]
    selected_dates = {value for value, _ in selected}
    reason_map = {value: reason for value, reason in selected}
    typical = source.filter(pl.col("Date").is_in(list(selected_dates))).sort(
        ["Date", "Contract", "minute"]
    )
    typical_rows = [
        {
            "date": str(row["Date"]),
            "reason": reason_map[row["Date"]],
            "contract": row["Contract"],
            "time": minute_text(int(row["minute"])),
            "minute": int(row["minute"]),
            "segment": "14:30以前" if row["segment_code"] == 0 else "14:30及以后",
            "tag_raw": float(row["tag"]),
            "tag_bp": float(row["tag"] * 10000),
        }
        for row in typical.iter_rows(named=True)
    ]

    quantile_rows = []
    for segment_code, segment_label in [(0, "14:30以前"), (1, "14:30及以后")]:
        values = source.filter(pl.col("segment_code") == segment_code)["tag"].to_numpy()
        quantiles = np.quantile(values, [0, 0.001, 0.005, 0.01, 0.10, 0.50, 0.90, 0.99, 0.995, 0.999, 1])
        quantile_rows.append(
            {
                "segment": segment_label,
                "sample_n": int(len(values)),
                "mean": float(np.mean(values)),
                "min": float(quantiles[0]),
                "p001": float(quantiles[1]),
                "p005": float(quantiles[2]),
                "p01": float(quantiles[3]),
                "p10": float(quantiles[4]),
                "median": float(quantiles[5]),
                "p90": float(quantiles[6]),
                "p99": float(quantiles[7]),
                "p995": float(quantiles[8]),
                "p999": float(quantiles[9]),
                "max": float(quantiles[10]),
            }
        )

    run_rows.sort(key=lambda row: (row["run_length"], abs(row["tag_raw"])), reverse=True)
    return {
        "repeat_summary": aggregate_rows,
        "contract_day": group_rows,
        "long_runs": run_rows[:500],
        "acf": acf_rows,
        "intraday": intraday_rows,
        "typical_days": typical_rows,
        "selected_days": [
            {"date": str(current_date), "reason": reason}
            for current_date, reason in selected
        ],
        "tag_quantiles": quantile_rows,
    }


def plot_plan_b_win_rate(plan_b: dict) -> None:
    configure_chinese_font()
    summary = plan_b["summary"]
    segments = ["全体样本", "14:30以前", "14:30及以后"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), sharex=True)
    for column, segment in enumerate(segments):
        rows = sorted(
            [row for row in summary if row["segment"] == segment],
            key=lambda row: row["strength"],
        )
        x = np.array([row["strength"] for row in rows])
        axes[0, column].plot(x, [row["long_win_rate"] for row in rows], marker="o", color="#C2413B", label="做多胜率")
        axes[0, column].plot(x, [row["short_win_rate"] for row in rows], marker="s", color="#287C51", label="做空胜率")
        axes[0, column].axhline(0.5, color="#94A3B8", linewidth=1, linestyle="--")
        axes[0, column].set_title(segment, fontweight="bold")
        axes[0, column].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0, column].legend(frameon=False)
        axes[0, column].grid(axis="y", alpha=0.25)

        axes[1, column].plot(x, [row["long_share"] for row in rows], marker="o", color="#C2413B", label="做多占比")
        axes[1, column].plot(x, [row["short_share"] for row in rows], marker="s", color="#287C51", label="做空占比")
        axes[1, column].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1, column].set_xlabel("偏置修正强度")
        axes[1, column].legend(frameon=False)
        axes[1, column].grid(axis="y", alpha=0.25)

    axes[0, 0].set_ylabel("胜率")
    axes[1, 0].set_ylabel("样本占比")
    fig.suptitle("Plan B：偏置修正强度与多空胜率、样本占比", fontsize=19, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_WIN_IMAGE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_transitions(plan_b: dict) -> None:
    configure_chinese_font()
    transitions = plan_b["transitions"]
    segments = ["全体样本", "14:30以前", "14:30及以后"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, segment in zip(axes, segments):
        rows = sorted(
            [
                row
                for row in transitions
                if row["segment"] == segment
                and row["transition"] == "多头→转为空头"
                and row["strength"] > 0
            ],
            key=lambda row: row["strength"],
        )
        x = np.array([row["strength"] for row in rows])
        counts = np.array([row["sample_n"] for row in rows])
        win_rates = np.array([row["win_rate"] for row in rows], dtype=float)
        ax.bar(x, counts, width=0.14, color="#A7C7B5", label="转换样本量")
        ax.set_title(segment, fontweight="bold")
        ax.set_xlabel("偏置修正强度")
        ax.set_ylabel("多头转空头样本量")
        ax.grid(axis="y", alpha=0.2)
        second = ax.twinx()
        second.plot(x, win_rates, marker="o", color="#1F6E45", linewidth=2.2, label="转空后胜率")
        second.set_ylabel("转空后胜率")
        second.yaxis.set_major_formatter(PercentFormatter(1))
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = second.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
    fig.suptitle("Plan B：原多头转为空头的样本量与胜率", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_TRANSITION_IMAGE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_typical_days(tag_diagnostics: dict) -> None:
    configure_chinese_font()
    rows = tag_diagnostics["typical_days"]
    selected = tag_diagnostics["selected_days"]
    family_colors = {"IC": "#2E6FBB", "IF": "#D14D41", "IH": "#6B5CA5", "IM": "#2D8B57"}
    suffix_styles = {"01": "-", "02": "--", "03": ":"}
    fig, axes = plt.subplots(3, 1, figsize=(18, 13), sharex=True)
    for ax, selected_day in zip(axes, selected):
        day_rows = [row for row in rows if row["date"] == selected_day["date"]]
        contracts = sorted({row["contract"] for row in day_rows})
        for contract in contracts:
            contract_rows = sorted(
                [row for row in day_rows if row["contract"] == contract],
                key=lambda row: row["minute"],
            )
            family = contract[:2]
            suffix = contract[-2:]
            ax.plot(
                [row["minute"] for row in contract_rows],
                [row["tag_bp"] for row in contract_rows],
                color=family_colors.get(family, "#64748B"),
                linestyle=suffix_styles.get(suffix, "-"),
                linewidth=1.25,
                alpha=0.85,
                label=contract,
            )
        ax.axvline(14 * 60 + 30, color="#111827", linestyle="--", linewidth=1)
        ax.axhline(0, color="#94A3B8", linewidth=0.8)
        ax.set_title(f"{selected_day['date']}｜{selected_day['reason']}", loc="left", fontweight="bold")
        ax.set_ylabel("Tag（bp）")
        ax.grid(axis="y", alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, ncol=6, frameon=False, fontsize=8)
    tick_minutes = [570, 600, 630, 660, 690, 780, 810, 840, 870, 900]
    axes[-1].set_xticks(tick_minutes)
    axes[-1].set_xticklabels([minute_text(value) for value in tick_minutes])
    axes[-1].set_xlabel("日内时间（虚线为14:30）")
    fig.suptitle("Tag周期诊断：三个典型交易日的分钟Tag曲线", fontsize=19, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_TYPICAL_IMAGE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_tag_diagnostics(tag_diagnostics: dict) -> None:
    configure_chinese_font()
    acf_rows = [row for row in tag_diagnostics["acf"] if row["contract"] == "全部合约"]
    intraday = tag_diagnostics["intraday"]
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    for ax, segment in zip(axes[0], ["14:30以前", "14:30及以后"]):
        rows = sorted([row for row in acf_rows if row["segment"] == segment], key=lambda row: row["lag_minutes"])
        ax.plot([row["lag_minutes"] for row in rows], [row["correlation"] for row in rows], marker="o", color="#315D9C")
        ax.set_title(f"{segment}｜Tag滞后相关", fontweight="bold")
        ax.set_xlabel("滞后分钟")
        ax.set_ylabel("相关系数")
        ax.grid(alpha=0.25)

    before = [row for row in intraday if row["segment"] == "14:30以前"]
    after = [row for row in intraday if row["segment"] == "14:30及以后"]
    all_rows = sorted(before + after, key=lambda row: row["minute"])
    axes[1, 0].plot([row["minute"] for row in all_rows], [row["tag_std"] * 10000 for row in all_rows], color="#C65A32")
    axes[1, 0].axvline(14 * 60 + 30, color="#111827", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Tag标准差的日内变化", fontweight="bold")
    axes[1, 0].set_ylabel("标准差（bp）")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot([row["minute"] for row in all_rows], [row["exact_repeat_rate"] for row in all_rows], color="#4B7F52")
    axes[1, 1].axvline(14 * 60 + 30, color="#111827", linestyle="--", linewidth=1)
    axes[1, 1].set_title("相邻分钟Tag完全相同的比例", fontweight="bold")
    axes[1, 1].set_ylabel("重复比例")
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 1].grid(alpha=0.25)

    for ax in axes[1]:
        ax.set_xticks([570, 630, 690, 780, 840, 870, 900])
        ax.set_xticklabels([minute_text(value) for value in [570, 630, 690, 780, 840, 870, 900]])
        ax.set_xlabel("日内时间")
    fig.suptitle("Tag周期诊断：滞后相关、日内波动与连续重复", fontsize=19, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_DIAG_IMAGE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_readme() -> None:
    text = f"""# Plan B胜率与Tag周期诊断

## 本阶段范围

- Plan B只比较多空方向、方向转换和胜率，不计算累计收益。
- `score = pred + 0.5`，分类中心为 `{CLASSIFICATION_CENTER:.2f}`。
- 使用此前 `{ROLLING_DAYS}` 个交易日的score中位数估计公共偏置。
- 14:30前后分别估计偏置，14:30整点属于后一段。
- 测试修正强度：{', '.join(f'{value:.0%}' for value in CORRECTION_STRENGTHS)}。
- 最初 `{ROLLING_DAYS}` 个交易日作为预热期，不进入方案比较。

## 胜率定义

- 做多胜率：`Tag > 0`。
- 做空胜率：`Tag < 0`。
- 最关键的转换组是“原做多、修正后做空”。如果该组做空胜率稳定提高，才说明公共多头偏置可能具有修正价值。

## Tag周期诊断

当前parquet只能用于推断，不能百分之百确认Tag公式。本次检查包括：

- 同一合约、同一交易日、同一时段内的Tag唯一值数量；
- 相邻分钟Tag完全相同和近似相同的比例；
- 连续相同Tag的最长持续分钟数；
- 1至60分钟的Tag滞后相关；
- Tag标准差和连续重复率的日内变化；
- 尾盘Tag均值最高、最低和中位附近三个典型日的分钟曲线。

若要最终确认Tag是未来1、5、10、30分钟收益，还是到收盘/下一交易日收益，仍需标签生成代码或原始分钟价格进行候选收益复算。

## 文件

- `PlanB胜率与Tag周期诊断.xlsx`：统计工作簿。
- `{Path(__file__).name}`：Jupyter/PyCharm可运行源码。
- `{OUTPUT_WIN_IMAGE.name}`：多空胜率与样本占比。
- `{OUTPUT_TRANSITION_IMAGE.name}`：多头转空头的样本量与胜率。
- `{OUTPUT_TYPICAL_IMAGE.name}`：典型日分钟Tag曲线。
- `{OUTPUT_DIAG_IMAGE.name}`：滞后相关与日内结构。
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


# %%
source = load_source()
plan_b_results = build_plan_b_results(source)
tag_diagnostics = build_tag_diagnostics(source)

payload = {
    "metadata": {
        "source_path": str(SOURCE_PATH),
        "source_rows": source.height,
        "classification_center": CLASSIFICATION_CENTER,
        "rolling_days": ROLLING_DAYS,
        "correction_strengths": CORRECTION_STRENGTHS,
        "near_repeat_tolerance": NEAR_REPEAT_TOLERANCE,
        "note": "本阶段只比较方向与胜率；Tag周期尚未由标签代码或原始行情复算确认。",
    },
    "plan_b": plan_b_results,
    "tag_diagnostics": tag_diagnostics,
}
OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

plot_plan_b_win_rate(plan_b_results)
plot_transitions(plan_b_results)
plot_typical_days(tag_diagnostics)
plot_tag_diagnostics(tag_diagnostics)
write_readme()

print(f"已生成中间数据：{OUTPUT_JSON}")
print(f"已生成图片：{OUTPUT_WIN_IMAGE}")
print(f"已生成图片：{OUTPUT_TRANSITION_IMAGE}")
print(f"已生成图片：{OUTPUT_TYPICAL_IMAGE}")
print(f"已生成图片：{OUTPUT_DIAG_IMAGE}")
print(f"已生成说明：{OUTPUT_README}")

