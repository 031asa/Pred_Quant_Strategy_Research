from __future__ import annotations

"""调整前后 Pred 的零阈值收益、六指标和单利累计收益分析。"""

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TEMP", ".")) / "pred_returns_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import polars as pl

from utils.segmented_drift import SEGMENT_AFTER, SEGMENT_BEFORE, SEGMENT_ORDER
from utils.plot_fonts import register_matplotlib_chinese_font


VARIANT_BEFORE = "调整前（未平滑）"
VARIANT_AFTER = "调整后（已平滑）"
SCOPE_OVERALL = "总体"
SCOPE_ORDER = (SCOPE_OVERALL, SEGMENT_BEFORE, SEGMENT_AFTER)


@dataclass(frozen=True)
class ReturnAnalysisResult:
    """生成图片的位置及六个“状态×范围”的零阈值验收指标。"""

    figure_paths: tuple[Path, ...]
    sample_count_per_variant: int
    short_rates: dict[str, float]
    hit_rates: dict[str, float]
    scope_sample_counts: dict[str, int]
    paired_final_cumulative_returns: dict[str, float]
    distribution_moments: dict[str, dict[str, float]]


def equal_count_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """按稳定排序分配严格等样本档位。"""

    if n_bins <= 0 or len(values) < n_bins:
        raise ValueError("档位数必须为正且不得超过样本量。")
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(len(values), dtype=np.int16)
    bins[order] = np.minimum(
        np.arange(len(values), dtype=np.int64) * n_bins // len(values),
        n_bins - 1,
    )
    return bins


def zero_threshold_strategy_returns(
    pred: np.ndarray,
    tag: np.ndarray,
) -> np.ndarray:
    """pred<0做空、pred>=0做多；返回逐样本策略收益。"""

    pred_values = np.asarray(pred, dtype=np.float64)
    tag_values = np.asarray(tag, dtype=np.float64)
    if pred_values.shape != tag_values.shape:
        raise ValueError("pred与Tag形状不一致。")
    return np.where(pred_values < 0, -tag_values, tag_values)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _prepare_data(
    source: pl.DataFrame,
    *,
    datetime_column: str,
    pred_column: str,
    tag_column: str,
    time_cutoff: str,
) -> pl.DataFrame:
    return (
        source.select([datetime_column, pred_column, tag_column])
        .drop_nulls()
        .with_columns(
            pl.col(datetime_column).dt.date().alias("Date"),
            pl.when(pl.col(datetime_column).dt.strftime("%H:%M") < time_cutoff)
            .then(pl.lit(SEGMENT_BEFORE))
            .otherwise(pl.lit(SEGMENT_AFTER))
            .alias("时段"),
            pl.col(pred_column).cast(pl.Float64).alias("pred_value"),
            pl.col(tag_column).cast(pl.Float64).alias("tag"),
        )
        .with_columns(
            (pl.col("pred_value") < 0).alias("预测做空"),
            pl.when(pl.col("pred_value") < 0)
            .then(-pl.col("tag"))
            .otherwise(pl.col("tag"))
            .alias("策略收益"),
        )
    )


def _load_matched_variants(
    *,
    raw_input_path: Path,
    adjusted_input_path: Path,
    datetime_column: str,
    pred_column: str,
    tag_column: str,
    time_cutoff: str,
    expected_sample_count: int,
) -> dict[str, pl.DataFrame]:
    adjusted_source = pl.read_parquet(adjusted_input_path).select(
        [datetime_column, pred_column, tag_column]
    )
    if adjusted_source.height != expected_sample_count:
        raise AssertionError(
            f"调整后样本量为{adjusted_source.height:,}，预期{expected_sample_count:,}。"
        )

    # 使用整数时间戳筛选，避免Windows环境仅为比较边界而依赖tzdata包。
    effective_start_ns = adjusted_source.select(
        pl.col(datetime_column).dt.timestamp("ns").min()
    ).item()
    raw_source = (
        pl.read_parquet(raw_input_path)
        .select([datetime_column, pred_column, tag_column])
        .filter(pl.col(datetime_column).dt.timestamp("ns") >= effective_start_ns)
    )
    if raw_source.height != expected_sample_count:
        raise AssertionError(
            f"调整前同期间样本量为{raw_source.height:,}，预期{expected_sample_count:,}。"
        )
    if not raw_source[datetime_column].equals(adjusted_source[datetime_column]):
        raise AssertionError("调整前后Datetime未逐行对齐。")
    if not np.array_equal(
        raw_source[tag_column].to_numpy(),
        adjusted_source[tag_column].to_numpy(),
        equal_nan=True,
    ):
        raise AssertionError("调整前后Tag未逐行对齐。")

    return {
        VARIANT_BEFORE: _prepare_data(
            raw_source,
            datetime_column=datetime_column,
            pred_column=pred_column,
            tag_column=tag_column,
            time_cutoff=time_cutoff,
        ),
        VARIANT_AFTER: _prepare_data(
            adjusted_source,
            datetime_column=datetime_column,
            pred_column=pred_column,
            tag_column=tag_column,
            time_cutoff=time_cutoff,
        ),
    }


def build_zero_threshold_metric_rows(
    data: pl.DataFrame,
    *,
    variant: str,
    scope: str,
) -> tuple[pl.DataFrame, np.ndarray]:
    """20等分仅负责分组；方向和六项指标均逐样本使用pred符号。"""

    pred = data["pred_value"].to_numpy()
    tag = data["tag"].to_numpy()
    strategy_returns = zero_threshold_strategy_returns(pred, tag)
    bins20 = equal_count_bins(pred, 20)
    trading_days = int(data["Date"].n_unique())
    if trading_days <= 0:
        raise ValueError("指标范围内没有交易日。")
    trading_calendar = data.select("Date").unique().sort("Date")
    daily_bin_counts = (
        pl.DataFrame({"Date": data["Date"], "bin_index": bins20})
        .group_by(["Date", "bin_index"])
        .len(name="当日开仓次数")
    )
    rows: list[dict[str, object]] = []

    for bin_index in range(20):
        mask = bins20 == bin_index
        group_returns = strategy_returns[mask]
        nonzero = tag[mask] != 0
        wins = group_returns[group_returns > 0]
        losses = group_returns[group_returns < 0]
        short_share = float(np.mean(pred[mask] < 0))
        sample_count = int(mask.sum())
        mean_strategy_return = float(group_returns.mean())
        daily_open_counts = (
            trading_calendar.join(
                daily_bin_counts.filter(pl.col("bin_index") == bin_index).select(
                    ["Date", "当日开仓次数"]
                ),
                on="Date",
                how="left",
                validate="1:1",
            )
            .with_columns(pl.col("当日开仓次数").fill_null(0))
            ["当日开仓次数"]
            .to_numpy()
        )
        median_daily_open_count = float(np.median(daily_open_counts))
        average_daily_return = mean_strategy_return * median_daily_open_count
        hit_rate = _safe_rate(
            int((group_returns[nonzero] > 0).sum()),
            int(nonzero.sum()),
        )
        payoff_ratio = (
            float(wins.mean() / abs(losses.mean()))
            if len(wins) and len(losses)
            else None
        )
        expectancy_r = (
            float(hit_rate * payoff_ratio - (1.0 - hit_rate))
            if hit_rate is not None and payoff_ratio is not None
            else None
        )
        rows.append(
            {
                "状态": variant,
                "范围": scope,
                "档位": f"Q{bin_index + 1:02d}",
                "Pred下界": float(pred[mask].min()),
                "Pred上界": float(pred[mask].max()),
                "样本量": sample_count,
                "交易日数": trading_days,
                "做空占比": short_share,
                "平均策略收益": mean_strategy_return,
                "每日开仓次数中位数": median_daily_open_count,
                "平均每日收益": average_daily_return,
                "胜率_排除Tag零值": hit_rate,
                "平均盈亏比": payoff_ratio,
                "期望R倍数": expectancy_r,
            }
        )
    return pl.DataFrame(rows), bins20


def _direction_color(short_share: float) -> str:
    """混合档按档内多数方向着色；恰好各半时归为做空蓝色。"""
    if short_share >= 0.5:
        return "#3498f0"
    return "#f18436"


def draw_zero_threshold_metric_chart(
    metric_df: pl.DataFrame,
    *,
    title: str,
    output_path: Path,
) -> None:
    """绘制均笔收益、日开仓中位数、日均收益、胜率、盈亏比和期望R六联图。"""

    x = np.arange(1, 21)
    short_rate = float(
        np.average(metric_df["做空占比"].to_numpy(), weights=metric_df["样本量"])
    )
    zero_axis_x = 0.5 + 20.0 * short_rate
    colors = [_direction_color(value) for value in metric_df["做空占比"]]

    fig, axes = plt.subplots(6, 1, figsize=(13.5, 18.5), sharex=True)
    fig.suptitle(title, fontsize=17, y=0.992)
    fig.text(
        0.5,
        0.964,
        (
            "20等分仅用于观察Pred强弱｜每条样本pred<0做空、pred>=0做多｜"
            "Tag=0排除胜率、盈亏比和期望R计算\n"
            "平均每日收益=平均策略收益×每日开仓次数中位数，表示典型频次下的逐笔收益日内求和，"
            "不等同于累计曲线的日内资金均分账户收益"
        ),
        ha="center",
        color="#526071",
        fontsize=10.2,
    )

    axes[0].bar(x, metric_df["平均策略收益"], color=colors, width=0.82)
    axes[0].axhline(0, color="#6f7780", linewidth=1, linestyle="--")
    axes[0].set_ylabel("平均策略收益")
    axes[0].set_title("平均策略收益", loc="left", fontsize=11)
    axes[0].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

    axes[1].bar(x, metric_df["每日开仓次数中位数"], color=colors, width=0.82)
    axes[1].set_ylabel("次数/日")
    axes[1].set_title("每日开仓次数中位数", loc="left", fontsize=11)

    axes[2].bar(x, metric_df["平均每日收益"], color=colors, width=0.82)
    axes[2].axhline(0, color="#6f7780", linewidth=1, linestyle="--")
    axes[2].set_ylabel("平均每日收益")
    axes[2].set_title("平均每日收益", loc="left", fontsize=11)
    axes[2].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

    for axis, column, ylabel, baseline in (
        (axes[3], "胜率_排除Tag零值", "胜率", 0.5),
        (axes[4], "平均盈亏比", "平均盈亏比", 1.0),
        (axes[5], "期望R倍数", "期望R倍数", 0.0),
    ):
        values = metric_df[column].to_numpy()
        axis.plot(x, values, color="#8d98a6", linewidth=1.5, zorder=1)
        axis.scatter(x, values, color=colors, s=30, zorder=2)
        axis.axhline(baseline, color="#6f7780", linewidth=1, linestyle="--")
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel, loc="left", fontsize=11)
    axes[3].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

    for axis in axes:
        axis.axvline(
            zero_axis_x,
            color="#59636e",
            linewidth=1.3,
            linestyle=(0, (3, 3)),
        )
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        Patch(color="#3498f0", label="做空档"),
        Patch(color="#f18436", label="做多档"),
        Line2D(
            [0],
            [0],
            color="#59636e",
            linestyle=(0, (3, 3)),
            label=f"pred=0分界｜做空占比{short_rate:.2%}",
        ),
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.38),
        frameon=False,
        ncol=3,
        fontsize=9.5,
    )
    axes[5].set_xticks(x, [f"Q{i:02d}" for i in x])
    axes[5].set_xlabel("Pred由低到高的20个严格等样本档位（交易方向始终由0决定）")
    fig.tight_layout(rect=[0, 0.02, 1, 0.925])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _build_zero_threshold_cumulative(
    data: pl.DataFrame,
    *,
    scope: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    pred = data["pred_value"].to_numpy()
    tag = data["tag"].to_numpy()
    bins10 = equal_count_bins(pred, 10)
    strategy_returns = zero_threshold_strategy_returns(pred, tag)
    daily = (
        pl.DataFrame(
            {
                "Date": data["Date"],
                "bin_index": bins10,
                "strategy_return": strategy_returns,
            }
        )
        .group_by(["Date", "bin_index"])
        .agg(
            pl.col("strategy_return").mean().alias("当日等权策略收益"),
            pl.len().alias("当日样本量"),
        )
        .sort(["bin_index", "Date"])
    )
    trading_calendar = data.select("Date").unique().sort("Date")

    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for bin_index in range(10):
        mask = bins10 == bin_index
        part = (
            trading_calendar.join(
                daily.filter(pl.col("bin_index") == bin_index).select(
                    ["Date", "当日等权策略收益", "当日样本量"]
                ),
                on="Date",
                how="left",
                validate="1:1",
            )
            .with_columns(
                pl.col("当日等权策略收益").fill_null(0.0),
                pl.col("当日样本量").fill_null(0),
            )
            .sort("Date")
        )
        daily_returns = part["当日等权策略收益"].to_numpy()
        cumulative = np.cumsum(daily_returns)
        summary_rows.append(
            {
                "范围": scope,
                "档位": f"Q{bin_index + 1:02d}",
                "Pred下界": float(pred[mask].min()),
                "Pred上界": float(pred[mask].max()),
                "样本量": int(mask.sum()),
                "做空占比": float(np.mean(pred[mask] < 0)),
                "期末累计收益率": float(cumulative[-1]),
            }
        )
        for date_value, daily_return, cumulative_return, daily_n in zip(
            part["Date"].to_list(),
            daily_returns,
            cumulative,
            part["当日样本量"].to_numpy(),
        ):
            curve_rows.append(
                {
                    "范围": scope,
                    "Date": date_value,
                    "档位": f"Q{bin_index + 1:02d}",
                    "当日样本量": int(daily_n),
                    "当日等权策略收益": float(daily_return),
                    "累计收益率": float(cumulative_return),
                }
            )
    return pl.DataFrame(curve_rows), pl.DataFrame(summary_rows)


def _draw_zero_threshold_cumulative(
    cumulative_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    *,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 9.5))
    lines = []
    labels = []
    for bin_index in range(10):
        label = f"Q{bin_index + 1:02d}"
        part = cumulative_df.filter(pl.col("档位") == label).sort("Date")
        summary = summary_df.filter(pl.col("档位") == label).row(0, named=True)
        short_share = float(summary["做空占比"])
        color = _direction_color(short_share)
        line, = ax.plot(part["Date"], part["累计收益率"], color=color, linewidth=2.0)
        lines.append(line)
        labels.append(
            f"{label}｜[{summary['Pred下界']:.6f}, {summary['Pred上界']:.6f}]｜"
            f"空占比={short_share:.1%}｜N={summary['样本量']:,}｜"
            f"期末={summary['期末累计收益率']:.2%}"
        )

    ax.axhline(0, color="#66717f", linewidth=1.2)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=18, pad=26)
    ax.text(
        0.5,
        1.015,
        "每条样本pred<0做空、pred>=0做多｜每档日内等权后按交易日单利累计",
        transform=ax.transAxes,
        ha="center",
        color="#526071",
        fontsize=11,
    )
    ax.set_xlabel("交易日")
    ax.set_ylabel("单利累计收益率")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(
        lines,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9.3,
        title="档位｜Pred区间｜做空占比｜样本量｜期末单利累计收益",
    )
    fig.tight_layout(rect=[0, 0.03, 0.78, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_paired_decile_cumulative_returns(
    data: pl.DataFrame,
    *,
    variant: str,
    scope: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """低档做空、高档做多，两腿日均收益各占50%后按交易日单利累计。"""

    pred = data["pred_value"].to_numpy()
    bins10 = equal_count_bins(pred, 10)
    daily = (
        pl.DataFrame(
            {
                "Date": data["Date"],
                "bin_index": bins10,
                "tag": data["tag"],
            }
        )
        .group_by(["Date", "bin_index"])
        .agg(pl.col("tag").mean().alias("日内Tag均值"))
    )
    trading_calendar = data.select("Date").unique().sort("Date")

    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for low_bin in range(5):
        high_bin = 9 - low_bin
        short_leg = (
            daily.filter(pl.col("bin_index") == low_bin)
            .select(["Date", (-pl.col("日内Tag均值")).alias("空腿日收益")])
        )
        long_leg = (
            daily.filter(pl.col("bin_index") == high_bin)
            .select(["Date", pl.col("日内Tag均值").alias("多腿日收益")])
        )
        paired = (
            trading_calendar.join(short_leg, on="Date", how="left", validate="1:1")
            .join(long_leg, on="Date", how="left", validate="1:1")
            .with_columns(
                pl.col("空腿日收益").fill_null(0.0),
                pl.col("多腿日收益").fill_null(0.0),
            )
            .sort("Date")
        )
        if paired.height != trading_calendar.height:
            raise AssertionError(
                f"{variant}/{scope}/Q{low_bin + 1:02d}+Q{high_bin + 1:02d}交易日历不完整。"
            )
        short_daily_returns = paired["空腿日收益"].to_numpy()
        long_daily_returns = paired["多腿日收益"].to_numpy()
        paired_daily_returns = (short_daily_returns + long_daily_returns) / 2.0
        cumulative_returns = np.cumsum(paired_daily_returns)
        pair_label = f"Q{low_bin + 1:02d}空 + Q{high_bin + 1:02d}多"
        low_sample_count = int((bins10 == low_bin).sum())
        high_sample_count = int((bins10 == high_bin).sum())
        summary_rows.append(
            {
                "状态": variant,
                "范围": scope,
                "配对": pair_label,
                "低档": f"Q{low_bin + 1:02d}",
                "高档": f"Q{high_bin + 1:02d}",
                "交易日数": paired.height,
                "低档样本量": low_sample_count,
                "高档样本量": high_sample_count,
                "期末累计收益率": float(cumulative_returns[-1]),
            }
        )
        for (
            date_value,
            short_return,
            long_return,
            paired_daily_return,
            cumulative_return,
        ) in zip(
            paired["Date"].to_list(),
            short_daily_returns,
            long_daily_returns,
            paired_daily_returns,
            cumulative_returns,
        ):
            curve_rows.append(
                {
                    "状态": variant,
                    "范围": scope,
                    "Date": date_value,
                    "配对": pair_label,
                    "空腿日收益": float(short_return),
                    "多腿日收益": float(long_return),
                    "配对日收益": float(paired_daily_return),
                    "累计收益率": float(cumulative_return),
                }
            )
    return pl.DataFrame(curve_rows), pl.DataFrame(summary_rows)


def draw_paired_decile_cumulative_chart(
    cumulative_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    *,
    variant: str,
    scope: str,
    output_path: Path,
) -> None:
    """将五组对称档位画成五条真实累计收益曲线。"""

    fig, ax = plt.subplots(figsize=(16, 9.5))
    colors = ["#174a7e", "#2f6fa8", "#4d91c5", "#7ab2d4", "#a6cfe3"]
    lines: list[Line2D] = []
    labels: list[str] = []
    for pair_index, row in enumerate(summary_df.iter_rows(named=True)):
        pair_label = str(row["配对"])
        part = cumulative_df.filter(pl.col("配对") == pair_label).sort("Date")
        line, = ax.plot(
            part["Date"],
            part["累计收益率"],
            color=colors[pair_index],
            linewidth=2.1,
        )
        lines.append(line)
        labels.append(
            f"{pair_label}｜两腿N={row['低档样本量']:,}+{row['高档样本量']:,}｜"
            f"期末={row['期末累计收益率']:.2%}"
        )

    ax.axhline(0, color="#66717f", linewidth=1.2)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"{variant} Pred · {scope} · 10等分对称配对单利累计收益",
        fontsize=18,
        pad=26,
    )
    ax.text(
        0.5,
        1.015,
        "低档固定做空、高档固定做多｜两腿每日各占50%｜合并日收益后按交易日单利累计",
        transform=ax.transAxes,
        ha="center",
        color="#526071",
        fontsize=11,
    )
    ax.set_xlabel("交易日")
    ax.set_ylabel("单利累计收益率")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(
        lines,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9.5,
        title="配对组合｜两腿样本量｜期末单利累计收益",
    )
    fig.tight_layout(rect=[0, 0.03, 0.78, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _distribution_statistics(pred: np.ndarray) -> dict[str, float]:
    values = np.asarray(pred, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    centered = values - mean
    skewness = float(np.mean(centered ** 3) / std ** 3) if std else 0.0
    excess_kurtosis = float(np.mean(centered ** 4) / std ** 4 - 3.0) if std else 0.0
    return {
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "short_rate": float(np.mean(values < 0)),
    }


def draw_pred_distribution_chart(
    data: pl.DataFrame,
    *,
    variant: str,
    scope: str,
    output_path: Path,
) -> dict[str, float]:
    """绘制一个状态、一个范围的Pred密度及正态分布对照。"""

    pred = data["pred_value"].to_numpy()
    stats = _distribution_statistics(pred)
    fig, ax = plt.subplots(figsize=(13.5, 7.8))
    _, edges, _ = ax.hist(
        pred,
        bins=100,
        density=True,
        color="#4e91cf",
        alpha=0.72,
        edgecolor="none",
        label="Pred密度直方图",
    )
    x = np.linspace(float(edges[0]), float(edges[-1]), 600)
    if stats["std"] > 0:
        normal_density = (
            np.exp(-0.5 * ((x - stats["mean"]) / stats["std"]) ** 2)
            / (stats["std"] * np.sqrt(2.0 * np.pi))
        )
        ax.plot(
            x,
            normal_density,
            color="#e58032",
            linewidth=2.0,
            linestyle="--",
            label="同均值方差正态曲线",
        )
    ax.axvline(0, color="#303942", linewidth=1.3, label="pred=0")
    ax.set_title(f"{variant} Pred · {scope} · 分布与正态曲线对照", fontsize=17, pad=27)
    ax.text(
        0.5,
        1.015,
        "密度直方图｜橙色虚线为相同均值和标准差的正态分布｜黑色竖线为pred=0",
        transform=ax.transAxes,
        ha="center",
        color="#526071",
        fontsize=10.5,
    )
    ax.set_xlabel("Pred")
    ax.set_ylabel("密度")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.99,
        0.94,
        (
            f"N={len(pred):,}\n均值={stats['mean']:.6f}\n标准差={stats['std']:.6f}\n"
            f"偏度={stats['skewness']:.3f}\n超额峰度={stats['excess_kurtosis']:.3f}\n"
            f"做空占比={stats['short_rate']:.2%}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#34495e",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.88, "edgecolor": "#d5dce3"},
    )
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout(rect=[0, 0.025, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stats


def _scope_data(data: pl.DataFrame, scope: str) -> pl.DataFrame:
    return data if scope == SCOPE_OVERALL else data.filter(pl.col("时段") == scope)


def analyze_adjusted_pred_returns(
    *,
    raw_input_path: Path,
    adjusted_input_path: Path,
    before_overall_metric_figure_path: Path,
    before_before_1430_metric_figure_path: Path,
    before_after_1430_metric_figure_path: Path,
    adjusted_overall_metric_figure_path: Path,
    adjusted_before_1430_metric_figure_path: Path,
    adjusted_after_1430_metric_figure_path: Path,
    before_overall_cumulative_figure_path: Path,
    before_before_1430_cumulative_figure_path: Path,
    before_after_1430_cumulative_figure_path: Path,
    adjusted_overall_cumulative_figure_path: Path,
    adjusted_before_1430_cumulative_figure_path: Path,
    adjusted_after_1430_cumulative_figure_path: Path,
    before_overall_paired_cumulative_figure_path: Path,
    before_before_1430_paired_cumulative_figure_path: Path,
    before_after_1430_paired_cumulative_figure_path: Path,
    adjusted_overall_paired_cumulative_figure_path: Path,
    adjusted_before_1430_paired_cumulative_figure_path: Path,
    adjusted_after_1430_paired_cumulative_figure_path: Path,
    before_overall_distribution_figure_path: Path,
    before_before_1430_distribution_figure_path: Path,
    before_after_1430_distribution_figure_path: Path,
    adjusted_overall_distribution_figure_path: Path,
    adjusted_before_1430_distribution_figure_path: Path,
    adjusted_after_1430_distribution_figure_path: Path,
    datetime_column: str,
    pred_column: str,
    tag_column: str,
    time_cutoff: str,
    expected_sample_count: int,
    expected_before_1430_count: int,
    expected_after_1430_count: int,
) -> ReturnAnalysisResult:
    """同样本比较调整前后，生成零阈值指标、累计收益和分布图。"""

    metric_paths = {
        (VARIANT_BEFORE, SCOPE_OVERALL): Path(before_overall_metric_figure_path),
        (VARIANT_BEFORE, SEGMENT_BEFORE): Path(before_before_1430_metric_figure_path),
        (VARIANT_BEFORE, SEGMENT_AFTER): Path(before_after_1430_metric_figure_path),
        (VARIANT_AFTER, SCOPE_OVERALL): Path(adjusted_overall_metric_figure_path),
        (VARIANT_AFTER, SEGMENT_BEFORE): Path(adjusted_before_1430_metric_figure_path),
        (VARIANT_AFTER, SEGMENT_AFTER): Path(adjusted_after_1430_metric_figure_path),
    }
    cumulative_paths = {
        (VARIANT_BEFORE, SCOPE_OVERALL): Path(before_overall_cumulative_figure_path),
        (VARIANT_BEFORE, SEGMENT_BEFORE): Path(before_before_1430_cumulative_figure_path),
        (VARIANT_BEFORE, SEGMENT_AFTER): Path(before_after_1430_cumulative_figure_path),
        (VARIANT_AFTER, SCOPE_OVERALL): Path(adjusted_overall_cumulative_figure_path),
        (VARIANT_AFTER, SEGMENT_BEFORE): Path(adjusted_before_1430_cumulative_figure_path),
        (VARIANT_AFTER, SEGMENT_AFTER): Path(adjusted_after_1430_cumulative_figure_path),
    }
    paired_paths = {
        (VARIANT_BEFORE, SCOPE_OVERALL): Path(before_overall_paired_cumulative_figure_path),
        (VARIANT_BEFORE, SEGMENT_BEFORE): Path(before_before_1430_paired_cumulative_figure_path),
        (VARIANT_BEFORE, SEGMENT_AFTER): Path(before_after_1430_paired_cumulative_figure_path),
        (VARIANT_AFTER, SCOPE_OVERALL): Path(adjusted_overall_paired_cumulative_figure_path),
        (VARIANT_AFTER, SEGMENT_BEFORE): Path(adjusted_before_1430_paired_cumulative_figure_path),
        (VARIANT_AFTER, SEGMENT_AFTER): Path(adjusted_after_1430_paired_cumulative_figure_path),
    }
    distribution_paths = {
        (VARIANT_BEFORE, SCOPE_OVERALL): Path(before_overall_distribution_figure_path),
        (VARIANT_BEFORE, SEGMENT_BEFORE): Path(before_before_1430_distribution_figure_path),
        (VARIANT_BEFORE, SEGMENT_AFTER): Path(before_after_1430_distribution_figure_path),
        (VARIANT_AFTER, SCOPE_OVERALL): Path(adjusted_overall_distribution_figure_path),
        (VARIANT_AFTER, SEGMENT_BEFORE): Path(adjusted_before_1430_distribution_figure_path),
        (VARIANT_AFTER, SEGMENT_AFTER): Path(adjusted_after_1430_distribution_figure_path),
    }
    all_figure_paths = (
        tuple(metric_paths.values())
        + tuple(cumulative_paths.values())
        + tuple(paired_paths.values())
        + tuple(distribution_paths.values())
    )

    variants = _load_matched_variants(
        raw_input_path=Path(raw_input_path),
        adjusted_input_path=Path(adjusted_input_path),
        datetime_column=datetime_column,
        pred_column=pred_column,
        tag_column=tag_column,
        time_cutoff=time_cutoff,
        expected_sample_count=expected_sample_count,
    )

    register_matplotlib_chinese_font()

    expected_scope_counts = {
        SCOPE_OVERALL: expected_sample_count,
        SEGMENT_BEFORE: expected_before_1430_count,
        SEGMENT_AFTER: expected_after_1430_count,
    }
    short_rates: dict[str, float] = {}
    hit_rates: dict[str, float] = {}
    paired_final_cumulative_returns: dict[str, float] = {}
    distribution_moments: dict[str, dict[str, float]] = {}

    for variant, data in variants.items():
        for scope in SCOPE_ORDER:
            part = _scope_data(data, scope)
            expected_n = expected_scope_counts[scope]
            if part.height != expected_n:
                raise AssertionError(
                    f"{variant}/{scope}样本量为{part.height:,}，预期{expected_n:,}。"
                )
            metric_df, _ = build_zero_threshold_metric_rows(
                part,
                variant=variant,
                scope=scope,
            )
            expected_bin_sizes = [expected_n // 20] * 20
            if metric_df["样本量"].to_list() != expected_bin_sizes:
                raise AssertionError(f"{variant}/{scope}的20等分未严格等样本。")

            key = f"{variant}/{scope}"
            pred = part["pred_value"].to_numpy()
            tag = part["tag"].to_numpy()
            strategy_returns = zero_threshold_strategy_returns(pred, tag)
            nonzero = tag != 0
            short_rates[key] = float(np.mean(pred < 0))
            hit_rates[key] = float(np.mean(strategy_returns[nonzero] > 0))
            draw_zero_threshold_metric_chart(
                metric_df,
                title=f"{variant} Pred · {scope} · 20等分零阈值六指标",
                output_path=metric_paths[(variant, scope)],
            )

            cumulative_df, cumulative_summary_df = _build_zero_threshold_cumulative(
                part,
                scope=scope,
            )
            if cumulative_summary_df["样本量"].to_list() != [part.height // 10] * 10:
                raise AssertionError(f"{variant}/{scope}的10等分未严格等样本。")
            _draw_zero_threshold_cumulative(
                cumulative_df,
                cumulative_summary_df,
                title=f"{variant} Pred · {scope} · 10等分零阈值单利累计收益",
                output_path=cumulative_paths[(variant, scope)],
            )

            paired_curve_df, paired_summary_df = build_paired_decile_cumulative_returns(
                part,
                variant=variant,
                scope=scope,
            )
            expected_decile_size = expected_n // 10
            if (
                paired_summary_df["低档样本量"].to_list()
                != [expected_decile_size] * 5
                or paired_summary_df["高档样本量"].to_list()
                != [expected_decile_size] * 5
            ):
                raise AssertionError(f"{variant}/{scope}的配对10等分未严格等样本。")
            draw_paired_decile_cumulative_chart(
                paired_curve_df,
                paired_summary_df,
                variant=variant,
                scope=scope,
                output_path=paired_paths[(variant, scope)],
            )
            for row in paired_summary_df.iter_rows(named=True):
                result_key = f"{variant}/{scope}/{row['配对']}"
                paired_final_cumulative_returns[result_key] = float(row["期末累计收益率"])

            distribution_moments[f"{variant}/{scope}"] = draw_pred_distribution_chart(
                part,
                variant=variant,
                scope=scope,
                output_path=distribution_paths[(variant, scope)],
            )

    return ReturnAnalysisResult(
        figure_paths=all_figure_paths,
        sample_count_per_variant=expected_sample_count,
        short_rates=short_rates,
        hit_rates=hit_rates,
        scope_sample_counts=expected_scope_counts,
        paired_final_cumulative_returns=paired_final_cumulative_returns,
        distribution_moments=distribution_moments,
    )
