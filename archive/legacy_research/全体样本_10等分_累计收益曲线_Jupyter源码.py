# %% [markdown]
# # 全体样本10等分累计收益率曲线
#
# - score = pred + 0.5，并限制在[0,1]
# - 全体样本按score等样本10等分
# - Q01-Q05做空，策略收益=-Tag；Q06-Q10做多，策略收益=Tag
# - 每档在每个交易日内先对全部有效信号等权平均
# - 累计收益率 = cumprod(1 + 日均策略收益) - 1

# %%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
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
OUTPUT_BEFORE_IMAGE = BASE_DIR / "全体样本_10等分_14点30以前_累计收益率曲线.png"
OUTPUT_AFTER_IMAGE = BASE_DIR / "全体样本_10等分_14点30及以后_累计收益率曲线.png"
OUTPUT_BEFORE_INDEPENDENT_IMAGE = (
    BASE_DIR / "14点30以前独立样本_10等分_累计收益率曲线.png"
)
OUTPUT_AFTER_INDEPENDENT_IMAGE = (
    BASE_DIR / "14点30及以后独立样本_10等分_累计收益率曲线.png"
)

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = json.load(config_file)

SOURCE_PATH = Path(CONFIG["source_path"])
PRED_COLUMN = CONFIG["pred_column"]
RETURN_COLUMN = CONFIG["return_column"]
DATETIME_COLUMN = CONFIG["datetime_column"]
SCORE_OFFSET = float(CONFIG["score_offset"])
SCORE_MIN = float(CONFIG["score_min"])
SCORE_MAX = float(CONFIG["score_max"])
DECILES = 10
TIME_CUTOFF = CONFIG["time_cutoff"]


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


def boundary_text(value: float) -> str:
    return f"{value:.6f}"


def build_decile_curves(
    time_segment: str,
    independent_regroup: bool = False,
) -> tuple[list[dict], dict]:
    """生成指定时段曲线；可选择是否在时段内部重新等分。"""
    if time_segment not in {"before", "after"}:
        raise ValueError("time_segment必须是before或after。")
    source = (
        pl.read_parquet(SOURCE_PATH)
        .select([DATETIME_COLUMN, PRED_COLUMN, RETURN_COLUMN])
        .drop_nulls()
    )
    scores = np.clip(
        source[PRED_COLUMN].to_numpy().astype(float) + SCORE_OFFSET,
        SCORE_MIN,
        SCORE_MAX,
    )
    raw_returns = source[RETURN_COLUMN].to_numpy().astype(float)
    n = len(scores)

    cutoff_hour, cutoff_minute = [int(value) for value in TIME_CUTOFF.split(":")]
    cutoff_total_minutes = cutoff_hour * 60 + cutoff_minute
    minute_of_day = (
        source.select(
            (
                pl.col(DATETIME_COLUMN).dt.hour().cast(pl.Int32) * 60
                + pl.col(DATETIME_COLUMN).dt.minute().cast(pl.Int32)
            ).alias("minute_of_day")
        )["minute_of_day"]
        .to_numpy()
        .astype(int)
    )
    before_mask = minute_of_day < cutoff_total_minutes
    segment_mask = before_mask if time_segment == "before" else ~before_mask
    segment_label = "14:30以前" if time_segment == "before" else "14:30及以后"

    if independent_regroup:
        selected_indices = np.flatnonzero(segment_mask)
        grouping_scores = scores[segment_mask]
        grouping_n = len(grouping_scores)
        local_order = np.argsort(grouping_scores, kind="stable")
        sorted_scores = grouping_scores[local_order]
        sorted_bucket_ids = (np.arange(grouping_n) * DECILES // grouping_n) + 1
        bucket_ids = np.zeros(n, dtype=np.int16)
        bucket_ids[selected_indices[local_order]] = sorted_bucket_ids
        grouping_label = "时段内独立重新10等分"
    else:
        grouping_n = n
        order = np.argsort(scores, kind="stable")
        sorted_scores = scores[order]
        sorted_bucket_ids = (np.arange(n) * DECILES // n) + 1
        bucket_ids = np.empty(n, dtype=np.int16)
        bucket_ids[order] = sorted_bucket_ids
        grouping_label = "使用全体样本10等分边界"

    split_positions = [
        int(grouping_n * index / DECILES) for index in range(1, DECILES)
    ]
    internal_boundaries = [
        float((sorted_scores[position - 1] + sorted_scores[position]) / 2.0)
        for position in split_positions
    ]
    boundaries = [
        float(sorted_scores[0]),
        *internal_boundaries,
        float(sorted_scores[-1]),
    ]

    strategy_returns = np.where(bucket_ids <= 5, -raw_returns, raw_returns)
    dates = source.select(pl.col(DATETIME_COLUMN).dt.date().alias("Date"))["Date"]
    daily = (
        pl.DataFrame(
            {
                "Date": dates,
                "bucket": bucket_ids,
                "strategy_return": strategy_returns,
                "in_segment": segment_mask,
            }
        )
        .filter(pl.col("in_segment"))
        .group_by(["bucket", "Date"])
        .agg(
            pl.col("strategy_return").mean().alias("daily_strategy_return"),
            pl.len().alias("daily_signal_n"),
        )
        .sort(["bucket", "Date"])
    )

    curves: list[dict] = []
    for bucket in range(1, DECILES + 1):
        bucket_daily = daily.filter(pl.col("bucket") == bucket)
        daily_returns = bucket_daily["daily_strategy_return"].to_numpy()
        wealth = np.cumprod(1.0 + daily_returns)
        cumulative_return = wealth - 1.0
        running_peak = np.maximum.accumulate(np.maximum(wealth, 1.0))
        drawdown = wealth / running_peak - 1.0
        is_short = bucket <= 5
        lower = boundaries[bucket - 1]
        upper = boundaries[bucket]
        interval = (
            f"[{boundary_text(lower)}, {boundary_text(upper)})"
            if is_short
            else f"({boundary_text(lower)}, {boundary_text(upper)}]"
        )
        curves.append(
            {
                "bucket": bucket,
                "label": f"Q{bucket:02d}",
                "direction": "做空" if is_short else "做多",
                "interval": interval,
                "sample_n": int(np.sum((bucket_ids == bucket) & segment_mask)),
                "dates": bucket_daily["Date"].to_list(),
                "daily_returns": daily_returns,
                "cumulative_return": cumulative_return,
                "final_return": float(cumulative_return[-1]),
                "max_drawdown": float(drawdown.min()),
            }
        )

    metadata = {
        "sample_n": int(np.sum(segment_mask)),
        "global_sample_n": n,
        "grouping_sample_n": grouping_n,
        "segment": time_segment,
        "segment_label": segment_label,
        "independent_regroup": independent_regroup,
        "grouping_label": grouping_label,
        "direction_threshold": boundaries[5],
        "start_date": source.filter(pl.Series(segment_mask))[DATETIME_COLUMN].min().date(),
        "end_date": source.filter(pl.Series(segment_mask))[DATETIME_COLUMN].max().date(),
        "boundaries": boundaries,
    }
    return curves, metadata


def plot_decile_curves(
    curves: list[dict], metadata: dict, output_image: Path
) -> Path:
    """把十条累计收益曲线放入同一张图。"""
    configure_chinese_font()

    # 每个方向内部按“期末累计收益率”决定颜色深浅：收益越大颜色越深。
    def return_based_colors(direction_curves: list[dict], color_map) -> list:
        final_returns = np.array(
            [curve["final_return"] for curve in direction_curves], dtype=float
        )
        spread = float(np.ptp(final_returns))
        if spread == 0:
            intensity = np.full(len(final_returns), 0.68)
        else:
            normalized = (final_returns - final_returns.min()) / spread
            intensity = 0.42 + normalized * 0.48
        return [color_map(value) for value in intensity]

    short_colors = return_based_colors(curves[:5], plt.cm.Greens)
    long_colors = return_based_colors(curves[5:], plt.cm.Reds)
    colors = [*short_colors, *long_colors]

    fig, ax = plt.subplots(figsize=(18, 9.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FBFCFE")

    for index, curve in enumerate(curves):
        linewidth = 2.5 if curve["bucket"] in {1, 5, 6, 10} else 1.8
        label = (
            f"{curve['label']} {curve['direction']} ｜ {curve['interval']} ｜ "
            f"N={curve['sample_n']:,} ｜ 期末={curve['final_return']:.2%}"
        )
        ax.plot(
            curve["dates"],
            curve["cumulative_return"],
            color=colors[index],
            linewidth=linewidth,
            alpha=0.96,
            label=label,
        )

    ax.axhline(0, color="#64748B", linewidth=1)
    ax.grid(axis="y", color="#D8E0EA", linewidth=0.85)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#94A3B8")
    ax.set_title(
        (
            f"{metadata['segment_label']}独立样本10等分累计收益率对比"
            if metadata["independent_regroup"]
            else f"全体样本10等分 · {metadata['segment_label']}累计收益率对比"
        ),
        fontsize=22,
        fontweight="bold",
        pad=36,
    )
    ax.text(
        0.5,
        1.025,
        (
            f"{metadata['grouping_label']}（中位数={metadata['direction_threshold']:.6f}） ｜ "
            "本时段有效信号日内等权 → 日频复利累计 ｜ "
            "Q01-Q05做空（-Tag） ｜ Q06-Q10做多（Tag） ｜ 各方向内期末累计收益越大，颜色越深"
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#475569",
    )
    ax.set_xlabel("交易日", fontsize=12)
    ax.set_ylabel("累计收益率", fontsize=12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=0)

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.012, 0.5),
        frameon=False,
        fontsize=9,
        handlelength=3.0,
        labelspacing=0.9,
        title="档位 ｜ 方向 ｜ 概率区间 ｜ 样本量 ｜ 期末累计收益",
        title_fontsize=10,
    )
    legend._legend_box.align = "left"

    ax.text(
        0.0,
        -0.13,
        (
            f"当前时段样本数：{metadata['sample_n']:,} ｜ 分档母样本数：{metadata['grouping_sample_n']:,} ｜ "
            f"区间：{metadata['start_date']} 至 {metadata['end_date']} ｜ "
            "累计口径：cumprod(1 + 每档每日等权策略收益) - 1"
        ),
        transform=ax.transAxes,
        fontsize=9.5,
        color="#64748B",
    )

    fig.subplots_adjust(left=0.075, right=0.69, top=0.87, bottom=0.15)
    fig.savefig(output_image, dpi=260, bbox_inches="tight", facecolor="white")
    return output_image


# %%
generated_images = []
for segment, output_path in [
    ("before", OUTPUT_BEFORE_IMAGE),
    ("after", OUTPUT_AFTER_IMAGE),
]:
    decile_curves, curve_metadata = build_decile_curves(segment)
    image_path = plot_decile_curves(decile_curves, curve_metadata, output_path)
    generated_images.append(image_path)
    print(f"图片已生成：{image_path}")

for segment, output_path in [
    ("before", OUTPUT_BEFORE_INDEPENDENT_IMAGE),
    ("after", OUTPUT_AFTER_INDEPENDENT_IMAGE),
]:
    decile_curves, curve_metadata = build_decile_curves(
        segment,
        independent_regroup=True,
    )
    image_path = plot_decile_curves(decile_curves, curve_metadata, output_path)
    generated_images.append(image_path)
    print(f"图片已生成：{image_path}")

try:
    get_ipython  # type: ignore[name-defined]
except NameError:
    plt.close("all")
else:
    plt.show()
