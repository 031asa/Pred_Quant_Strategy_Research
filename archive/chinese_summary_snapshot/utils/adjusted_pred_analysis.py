from __future__ import annotations

# Implementation module. Public execution goes through utils.pipelines.

import json
import os
from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "result" / "可视化"
MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from utils.segmented_drift import SEGMENT_AFTER, SEGMENT_BEFORE, SEGMENT_ORDER


INPUT_PATH = PROJECT_ROOT / "result" / "数据" / "pred_eval_分时漂移修正后.parquet"
OUTPUT_PREFIX = "分时漂移修正后pred"
SEGMENT_FILE_LABEL = {
    SEGMENT_BEFORE: "14点30以前",
    SEGMENT_AFTER: "14点30及以后",
}


def equal_count_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """按稳定排序分配严格等样本档位。"""
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(len(values), dtype=np.int16)
    bins[order] = np.minimum(
        np.arange(len(values), dtype=np.int64) * n_bins // len(values),
        n_bins - 1,
    )
    return bins


def safe_rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def build_metric_rows(data: pl.DataFrame, scope: str) -> tuple[pl.DataFrame, np.ndarray]:
    scores = data["score"].to_numpy()
    tags = data["tag"].to_numpy()
    drift_short = data["慢速基准预测做空"].to_numpy()
    bins20 = equal_count_bins(scores, 20)
    rows = []

    for bin_index in range(20):
        mask = bins20 == bin_index
        group_direction = -1 if bin_index < 10 else 1
        group_return = group_direction * tags[mask]
        wins = group_return[group_return > 0]
        losses = group_return[group_return < 0]
        tag_nonzero = tags[mask] != 0
        drift_hit = (
            (drift_short[mask] & (tags[mask] < 0))
            | (~drift_short[mask] & (tags[mask] > 0))
        )
        rows.append(
            {
                "范围": scope,
                "档位": f"Q{bin_index + 1:02d}",
                "分组策略方向": "做空" if group_direction == -1 else "做多",
                "分数下界": float(np.min(scores[mask])),
                "分数上界": float(np.max(scores[mask])),
                "样本量": int(mask.sum()),
                "平均分组策略收益": float(np.mean(group_return)),
                "分组策略胜率": float(np.mean(group_return > 0)),
                "分组策略平均盈亏比": (
                    float(np.mean(wins) / abs(np.mean(losses)))
                    if len(wins) and len(losses)
                    else None
                ),
                "慢速基准命中有效样本量": int(tag_nonzero.sum()),
                "慢速基准预测命中率": float(np.mean(drift_hit[tag_nonzero]))
                if tag_nonzero.any()
                else None,
                "慢速基准预测做空占比": float(np.mean(drift_short[mask])),
                "慢速基准预测做多占比": float(np.mean(~drift_short[mask])),
            }
        )
    return pl.DataFrame(rows), bins20


def draw_metric_chart(metric_df: pl.DataFrame, title: str, output_path: Path) -> None:
    x = np.arange(1, 21)
    colors = ["#2f80d0"] * 10 + ["#f08031"] * 10
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 13), sharex=True)
    fig.suptitle(title, fontsize=17, y=0.992)
    fig.text(
        0.5,
        0.962,
        "收益按档位方向｜命中按修正后Score与当日分时慢速基准比较｜Tag=0排除命中分母",
        ha="center",
        color="#526071",
        fontsize=10.5,
    )

    axes[0].bar(x, metric_df["平均分组策略收益"], color=colors, width=0.82)
    axes[0].axhline(0, color="#6f7780", linewidth=1, linestyle="--")
    axes[0].set_ylabel("平均分组策略收益")
    axes[0].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

    axes[1].plot(x[:10], metric_df["分组策略胜率"][:10], color="#2f80d0", marker="o")
    axes[1].plot(x[10:], metric_df["分组策略胜率"][10:], color="#f08031", marker="o")
    axes[1].axhline(0.5, color="#6f7780", linewidth=1, linestyle="--")
    axes[1].set_ylabel("分组策略胜率")
    axes[1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

    axes[2].plot(x, metric_df["慢速基准预测命中率"], color="#7b4ba1", marker="o")
    axes[2].axhline(0.5, color="#6f7780", linewidth=1, linestyle="--")
    axes[2].set_ylabel("慢速基准预测命中率")
    axes[2].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

    axes[3].bar(x, metric_df["慢速基准预测做空占比"], color="#3a9d73", width=0.82)
    axes[3].axhline(0.5, color="#6f7780", linewidth=1, linestyle="--")
    axes[3].set_ylabel("慢速基准预测做空占比")
    axes[3].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

    for axis in axes:
        axis.axvline(10.5, color="#6f7780", linewidth=1.2, linestyle=(0, (3, 3)))
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[3].set_xticks(x, [f"Q{i:02d}" for i in x])
    axes[3].set_xlabel("档位（左半档分组做空，右半档分组做多）")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def build_cumulative(
    data: pl.DataFrame,
    bins20: np.ndarray,
    scope: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    scores = data["score"].to_numpy()
    tags = data["tag"].to_numpy()
    bins10 = bins20 // 2
    strategy_direction = np.where(bins10 < 5, -1, 1)
    daily = (
        pl.DataFrame(
            {
                "Date": data["Date"],
                "bin_index": bins10,
                "strategy_return": strategy_direction * tags,
            }
        )
        .group_by(["Date", "bin_index"])
        .agg(
            pl.col("strategy_return").mean().alias("当日等权策略收益"),
            pl.len().alias("当日样本量"),
        )
        .sort(["bin_index", "Date"])
    )

    rows = []
    summaries = []
    for bin_index in range(10):
        part = daily.filter(pl.col("bin_index") == bin_index).sort("Date")
        daily_returns = part["当日等权策略收益"].to_numpy()
        cumulative = np.cumprod(1.0 + daily_returns) - 1.0
        mask = bins10 == bin_index
        direction = "做空" if bin_index < 5 else "做多"
        summaries.append(
            {
                "范围": scope,
                "档位": f"Q{bin_index + 1:02d}",
                "分组策略方向": direction,
                "分数下界": float(np.min(scores[mask])),
                "分数上界": float(np.max(scores[mask])),
                "样本量": int(mask.sum()),
                "期末累计收益率": float(cumulative[-1]),
            }
        )
        for date_value, daily_return, cumulative_return, daily_n in zip(
            part["Date"].to_list(),
            daily_returns,
            cumulative,
            part["当日样本量"].to_numpy(),
        ):
            rows.append(
                {
                    "范围": scope,
                    "Date": date_value,
                    "档位": f"Q{bin_index + 1:02d}",
                    "分组策略方向": direction,
                    "当日样本量": int(daily_n),
                    "当日等权策略收益": float(daily_return),
                    "累计收益率": float(cumulative_return),
                }
            )
    return pl.DataFrame(rows), pl.DataFrame(summaries)


def draw_cumulative_chart(
    cumulative_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 9.5))
    short_colors = plt.cm.Greens(np.linspace(0.88, 0.45, 5))
    long_colors = plt.cm.Reds(np.linspace(0.45, 0.88, 5))
    legend_lines = []
    legend_labels = []
    for bin_index in range(10):
        label = f"Q{bin_index + 1:02d}"
        part = cumulative_df.filter(pl.col("档位") == label).sort("Date")
        summary = summary_df.filter(pl.col("档位") == label).row(0, named=True)
        color = short_colors[bin_index] if bin_index < 5 else long_colors[bin_index - 5]
        line, = ax.plot(part["Date"], part["累计收益率"], color=color, linewidth=2.2)
        legend_lines.append(line)
        legend_labels.append(
            f"{label} {summary['分组策略方向']}｜"
            f"[{summary['分数下界']:.6f}, {summary['分数上界']:.6f}]｜"
            f"N={summary['样本量']:,}｜期末={summary['期末累计收益率']:.2%}"
        )
    ax.axhline(0, color="#66717f", linewidth=1.2)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=18, pad=26)
    ax.text(
        0.5,
        1.015,
        "Q01-Q05做空（-Tag）｜Q06-Q10做多（Tag）｜日内等权后按交易日复利",
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
        title="档位｜分组方向｜分数区间｜样本量｜期末累计收益",
    )
    fig.tight_layout(rect=[0, 0.03, 0.79, 0.96])
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def build_profit_loss_summary(
    data: pl.DataFrame,
    bins20: np.ndarray,
) -> pl.DataFrame:
    """计算与总体10等分累计收益完全同口径的胜率、盈亏比和复利结果。"""
    scores = data["score"].to_numpy()
    tags = data["tag"].to_numpy()
    dates = data["Date"].to_numpy()
    bins10 = bins20 // 2
    rows = []

    for bin_index in range(10):
        mask = bins10 == bin_index
        direction_value = -1 if bin_index < 5 else 1
        strategy_returns = direction_value * tags[mask]
        nonzero = strategy_returns != 0
        wins = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]

        group_dates = dates[mask]
        unique_dates = np.unique(group_dates)
        daily_returns = np.array(
            [strategy_returns[group_dates == date].mean() for date in unique_dates],
            dtype=np.float64,
        )
        daily_nonzero = daily_returns != 0
        winning_days = daily_returns[daily_returns > 0]
        losing_days = daily_returns[daily_returns < 0]
        zero_short = scores[mask] < 0
        zero_hit = (
            (zero_short & (tags[mask] < 0))
            | (~zero_short & (tags[mask] > 0))
        )

        rows.append(
            {
                "范围": "全部有效样本",
                "档位": f"Q{bin_index + 1:02d}",
                "分组策略方向": "做空" if direction_value == -1 else "做多",
                "分数下界": float(scores[mask].min()),
                "分数上界": float(scores[mask].max()),
                "样本量": int(mask.sum()),
                "Tag非零样本量": int(nonzero.sum()),
                "样本胜率_排除Tag零值": safe_rate(
                    int((strategy_returns[nonzero] > 0).sum()), int(nonzero.sum())
                ),
                "零阈值预测命中率_排除Tag零值": safe_rate(
                    int(zero_hit[nonzero].sum()), int(nonzero.sum())
                ),
                "平均盈利": float(wins.mean()),
                "平均亏损绝对值": float(abs(losses.mean())),
                "逐样本盈亏比": float(wins.mean() / abs(losses.mean())),
                "单笔期望收益": float(strategy_returns.mean()),
                "交易日数": int(len(daily_returns)),
                "盈利日胜率_排除零收益日": safe_rate(
                    int((daily_returns[daily_nonzero] > 0).sum()),
                    int(daily_nonzero.sum()),
                ),
                "平均盈利日收益": float(winning_days.mean()),
                "平均亏损日绝对值": float(abs(losing_days.mean())),
                "日收益盈亏比": float(winning_days.mean() / abs(losing_days.mean())),
                "期末累计收益率": float(np.prod(1.0 + daily_returns) - 1.0),
            }
        )
    return pl.DataFrame(rows)


def draw_profit_loss_explanation(summary: pl.DataFrame, output_path: Path) -> None:
    x = np.arange(1, 11)
    labels = summary["档位"].to_list()
    colors = ["#3a9d73"] * 5 + ["#e8873a"] * 4 + ["#b6402e"]
    q10 = summary.filter(pl.col("档位") == "Q10").row(0, named=True)

    fig = plt.figure(figsize=(16, 10.5), facecolor="#fbfcfe")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], hspace=0.34, wspace=0.23)
    ax_win = fig.add_subplot(grid[0, 0])
    ax_ratio = fig.add_subplot(grid[1, 0])
    ax_cumulative = fig.add_subplot(grid[0, 1])
    ax_note = fig.add_subplot(grid[1, 1])

    fig.suptitle("最终 Pred 总体10等分：胜率、盈亏比与累计收益", fontsize=19, y=0.992)
    fig.text(
        0.5,
        0.953,
        "按零中心Pred排序｜Q01-Q05做空，Q06-Q10做多｜胜率排除Tag=0",
        ha="center",
        color="#526071",
        fontsize=11.5,
    )

    sample_win = summary["样本胜率_排除Tag零值"].to_numpy()
    daily_win = summary["盈利日胜率_排除零收益日"].to_numpy()
    ax_win.plot(x, sample_win, color="#276fbf", marker="o", linewidth=2.2, label="逐样本胜率")
    ax_win.plot(x, daily_win, color="#8c5ab5", marker="s", linewidth=2.2, label="盈利日胜率")
    ax_win.axhline(0.5, color="#707b87", linewidth=1.1, linestyle="--")
    ax_win.axvline(5.5, color="#9aa3ad", linewidth=1, linestyle=(0, (3, 3)))
    ax_win.scatter([10], [sample_win[-1]], s=115, color="#b6402e", zorder=5)
    ax_win.scatter([10], [daily_win[-1]], s=115, color="#b6402e", marker="s", zorder=5)
    ax_win.annotate(f"{sample_win[-1]:.2%}", (10, sample_win[-1]), xytext=(-42, -22), textcoords="offset points", color="#8f2f22", fontweight="bold")
    ax_win.annotate(f"{daily_win[-1]:.2%}", (10, daily_win[-1]), xytext=(-42, 10), textcoords="offset points", color="#8f2f22", fontweight="bold")
    ax_win.set_title("胜率：Q10并非特别高，但稳定高于50%", loc="left", fontsize=13.5)
    ax_win.set_ylabel("胜率")
    ax_win.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax_win.legend(frameon=False, ncol=2, loc="upper left")

    sample_ratio = summary["逐样本盈亏比"].to_numpy()
    daily_ratio = summary["日收益盈亏比"].to_numpy()
    ax_ratio.plot(x, sample_ratio, color="#276fbf", marker="o", linewidth=2.2, label="逐样本盈亏比")
    ax_ratio.plot(x, daily_ratio, color="#8c5ab5", marker="s", linewidth=2.2, label="日收益盈亏比")
    ax_ratio.axhline(1.0, color="#707b87", linewidth=1.1, linestyle="--")
    ax_ratio.axvline(5.5, color="#9aa3ad", linewidth=1, linestyle=(0, (3, 3)))
    ax_ratio.scatter([10], [sample_ratio[-1]], s=115, color="#b6402e", zorder=5)
    ax_ratio.scatter([10], [daily_ratio[-1]], s=115, color="#b6402e", marker="s", zorder=5)
    ax_ratio.annotate(f"{sample_ratio[-1]:.3f}", (10, sample_ratio[-1]), xytext=(-40, -24), textcoords="offset points", color="#8f2f22", fontweight="bold")
    ax_ratio.annotate(f"{daily_ratio[-1]:.3f}", (10, daily_ratio[-1]), xytext=(-40, 11), textcoords="offset points", color="#8f2f22", fontweight="bold")
    ax_ratio.set_title("赔率：Q10盈利幅度显著大于亏损幅度", loc="left", fontsize=13.5)
    ax_ratio.set_ylabel("盈亏比")
    ax_ratio.set_xlabel("最终Pred档位（左侧做空｜右侧做多）")
    ax_ratio.legend(frameon=False, ncol=2, loc="upper left")

    cumulative = summary["期末累计收益率"].to_numpy()
    bars = ax_cumulative.bar(x, cumulative, color=colors, width=0.76)
    ax_cumulative.axhline(0, color="#707b87", linewidth=1.1)
    ax_cumulative.axvline(5.5, color="#9aa3ad", linewidth=1, linestyle=(0, (3, 3)))
    for bar, value in zip(bars, cumulative):
        ax_cumulative.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.012 if value >= 0 else -0.012),
            f"{value:.1%}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    ax_cumulative.set_title("日内等权后按交易日复利", loc="left", fontsize=13.5)
    ax_cumulative.set_ylabel("期末累计收益率")
    ax_cumulative.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

    ax_note.axis("off")
    ax_note.set_facecolor("#f3f6fa")
    ax_note.add_patch(
        plt.Rectangle((0.02, 0.04), 0.96, 0.92, transform=ax_note.transAxes, color="#f1f4f8", ec="#d5dce5", lw=1.2)
    )
    ax_note.text(0.08, 0.88, "Q10 为什么累计收益达到56.06%？", transform=ax_note.transAxes, fontsize=15, fontweight="bold", color="#25364a", va="top")
    explanation = (
        f"样本胜率（排零）       {q10['样本胜率_排除Tag零值']:.2%}\n"
        f"逐样本盈亏比           {q10['逐样本盈亏比']:.3f}\n"
        f"盈利日胜率             {q10['盈利日胜率_排除零收益日']:.2%}\n"
        f"日收益盈亏比           {q10['日收益盈亏比']:.3f}\n"
        f"单笔期望收益           {q10['单笔期望收益']:.4%}\n"
        f"期末累计收益率         {q10['期末累计收益率']:.2%}"
    )
    ax_note.text(0.09, 0.74, explanation, transform=ax_note.transAxes, fontsize=12.5, linespacing=1.65, color="#26384b", va="top")
    ax_note.text(
        0.08,
        0.13,
        "结论：Q10不是靠极高胜率，而是“胜率略高于50%”\n叠加“平均盈利大于平均亏损”，经日复利累积形成高收益。",
        transform=ax_note.transAxes,
        fontsize=11.5,
        linespacing=1.55,
        color="#8f2f22",
        fontweight="bold",
    )

    for axis in [ax_win, ax_ratio, ax_cumulative]:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.018,
        "盈亏比 = 平均盈利 ÷ 平均亏损绝对值｜累计收益 = 日内等权策略收益按交易日复利｜未计手续费、滑点和资金占用",
        ha="center",
        color="#526071",
        fontsize=10.5,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


with (PROJECT_ROOT / "config.json").open(encoding="utf-8") as file:
    cfg = json.load(file)

pred_column = cfg["pred_column"]
tag_column = cfg["return_column"]
datetime_column = cfg["datetime_column"]
cutoff = cfg["time_cutoff"]

data = (
    pl.read_parquet(INPUT_PATH)
    .with_columns(
        pl.col(datetime_column).dt.date().alias("Date"),
        pl.when(pl.col(datetime_column).dt.strftime("%H:%M") < cutoff)
        .then(pl.lit(SEGMENT_BEFORE))
        .otherwise(pl.lit(SEGMENT_AFTER))
        .alias("时段"),
        pl.col(pred_column).cast(pl.Float64).alias("score"),
        pl.col(tag_column).cast(pl.Float64).alias("tag"),
    )
    .with_columns(
        (pl.col("score") < 0).alias("慢速基准预测做空"),
    )
)

if data.height != 1_667_520:
    raise AssertionError(f"有效样本量为{data.height:,}，预期1,667,520。")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 总体：按已经归零的最终 pred 统一分档，并直接以 0 判断命中方向。
overall_metric, overall_bins20 = build_metric_rows(data, "全部有效样本")
draw_metric_chart(
    overall_metric,
    "分时漂移修正后 Pred：总体20等分收益与慢速基准命中率",
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_20等分分组收益率.png",
)
overall_cumulative, overall_final = build_cumulative(data, overall_bins20, "全部有效样本")
draw_cumulative_chart(
    overall_cumulative,
    overall_final,
    "分时漂移修正后 Pred：总体10等分累计收益率",
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_10等分累计收益率.png",
)
profit_loss_summary = build_profit_loss_summary(data, overall_bins20)
draw_profit_loss_explanation(
    profit_loss_summary,
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_总体10等分胜率盈亏比与累计收益.png",
)

# 分时：两个时段各自独立重分。
segment_metrics = []
segment_cumulative = []
accuracy_rows = []
for segment in SEGMENT_ORDER:
    part = data.filter(pl.col("时段") == segment)
    metric, bins20 = build_metric_rows(part, segment)
    segment_metrics.append(metric)
    cumulative, final = build_cumulative(part, bins20, segment)
    segment_cumulative.append(cumulative)

    tag = part["tag"].to_numpy()
    short = part["慢速基准预测做空"].to_numpy()
    nonzero = tag != 0
    hit = (short & (tag < 0)) | (~short & (tag > 0))
    accuracy_rows.append(
        {
            "时段": segment,
            "样本量": part.height,
            "慢速基准命中有效样本量": int(nonzero.sum()),
            "Tag零值样本量": int((~nonzero).sum()),
            "慢速基准预测命中率": float(np.mean(hit[nonzero])),
            "慢速基准预测做空占比": float(np.mean(short)),
            "慢速基准预测做多占比": float(np.mean(~short)),
        }
    )

    file_label = SEGMENT_FILE_LABEL[segment]
    draw_metric_chart(
        metric,
        f"分时漂移修正后 Pred：{segment}独立20等分",
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_{file_label}_20等分收益与命中率.png",
    )
    draw_cumulative_chart(
        cumulative,
        final,
        f"分时漂移修正后 Pred：{segment}独立10等分累计收益率",
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_{file_label}_10等分累计收益率.png",
    )

segment_metric_df = pl.concat(segment_metrics)
segment_cumulative_df = pl.concat(segment_cumulative)
accuracy_df = pl.DataFrame(accuracy_rows)

# 强约束验收。
expected_segment_counts = {SEGMENT_BEFORE: 1_459_080, SEGMENT_AFTER: 208_440}
for segment, expected_n in expected_segment_counts.items():
    actual_n = data.filter(pl.col("时段") == segment).height
    if actual_n != expected_n:
        raise AssertionError(f"{segment}样本量{actual_n:,}，预期{expected_n:,}。")
    metric = segment_metric_df.filter(pl.col("范围") == segment)
    if metric["样本量"].to_list() != [expected_n // 20] * 20:
        raise AssertionError(f"{segment}的20等分未严格等样本。")
    cumulative = segment_cumulative_df.filter(pl.col("范围") == segment)
    if cumulative.group_by("档位").agg(pl.col("当日样本量").sum()).sort("档位")["当日样本量"].to_list() != [expected_n // 10] * 10:
        raise AssertionError(f"{segment}的10等分未严格等样本。")

if overall_metric["样本量"].sum() != data.height:
    raise AssertionError("总体20等分样本合计不一致。")
if profit_loss_summary["样本量"].to_list() != [data.height // 10] * 10:
    raise AssertionError("总体10等分盈亏比汇总未严格等样本。")
q10_mask = (overall_bins20 // 2) == 9
if np.any(data["score"].to_numpy()[q10_mask] < 0):
    raise AssertionError("Q10存在pred < 0样本，无法保证固定做多与零阈值方向一致。")
q10_summary = profit_loss_summary.filter(pl.col("档位") == "Q10").row(0, named=True)
if not np.isclose(
    q10_summary["样本胜率_排除Tag零值"],
    q10_summary["零阈值预测命中率_排除Tag零值"],
    rtol=0,
    atol=1e-15,
):
    raise AssertionError("Q10分组胜率与零阈值预测命中率不一致。")
if not np.isclose(
    q10_summary["期末累计收益率"],
    overall_final.filter(pl.col("档位") == "Q10")["期末累计收益率"].item(),
    rtol=0,
    atol=1e-12,
):
    raise AssertionError("Q10盈亏比汇总与原累计收益结果不一致。")
if overall_cumulative.select(pl.struct(["Date", "档位"]).is_duplicated().sum()).item():
    raise AssertionError("总体累计收益存在重复Date + 档位。")
if segment_cumulative_df.select(pl.struct(["范围", "Date", "档位"]).is_duplicated().sum()).item():
    raise AssertionError("分时累计收益存在重复时段 + Date + 档位。")

print(accuracy_df)
print(segment_metric_df.select(["范围", "档位", "平均分组策略收益", "慢速基准预测命中率"]))
print(f"结果已写入：{OUTPUT_DIR}")
