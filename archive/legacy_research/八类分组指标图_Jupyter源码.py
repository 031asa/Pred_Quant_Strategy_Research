# %% [markdown]
# # 八类分组指标图
#
# 运行本文件会生成8张高清PNG：
# 1. 全体样本20/50等分后，分别观察14:30以前和14:30及以后；
# 2. 先筛选14:30以前或14:30及以后，再用各自中位数重新做20/50等分。
#
# 每张图都直接显示平均策略收益、胜率、平均盈亏比；底部显示档位、概率区间和样本量。

# %%
from __future__ import annotations

from pathlib import Path
import runpy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


# %%
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

ANALYSIS_SOURCE = BASE_DIR / "中位数阈值_20与50等分_Jupyter源码.py"
OUTPUT_DIR = BASE_DIR / "八类分组指标图"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CHART_SPECS = [
    {
        "dataset": "global",
        "divisions": 20,
        "metric_scope": "before",
        "title": "全体样本20等分 · 14:30以前",
        "subtitle": "先按全体样本等分，再筛选14:30以前；各档样本量允许不同",
        "filename": "01_全体样本20等分_14点30以前.png",
    },
    {
        "dataset": "global",
        "divisions": 20,
        "metric_scope": "after",
        "title": "全体样本20等分 · 14:30及以后",
        "subtitle": "先按全体样本等分，再筛选14:30及以后；各档样本量允许不同",
        "filename": "02_全体样本20等分_14点30及以后.png",
    },
    {
        "dataset": "global",
        "divisions": 50,
        "metric_scope": "before",
        "title": "全体样本50等分 · 14:30以前",
        "subtitle": "先按全体样本等分，再筛选14:30以前；各档样本量允许不同",
        "filename": "03_全体样本50等分_14点30以前.png",
    },
    {
        "dataset": "global",
        "divisions": 50,
        "metric_scope": "after",
        "title": "全体样本50等分 · 14:30及以后",
        "subtitle": "先按全体样本等分，再筛选14:30及以后；各档样本量允许不同",
        "filename": "04_全体样本50等分_14点30及以后.png",
    },
    {
        "dataset": "before",
        "divisions": 20,
        "metric_scope": "overall",
        "title": "14:30以前独立样本 · 20等分",
        "subtitle": "先筛选14:30以前，再以该时段自己的中位数等样本分档",
        "filename": "05_14点30以前独立样本_20等分.png",
    },
    {
        "dataset": "before",
        "divisions": 50,
        "metric_scope": "overall",
        "title": "14:30以前独立样本 · 50等分",
        "subtitle": "先筛选14:30以前，再以该时段自己的中位数等样本分档",
        "filename": "06_14点30以前独立样本_50等分.png",
    },
    {
        "dataset": "after",
        "divisions": 20,
        "metric_scope": "overall",
        "title": "14:30及以后独立样本 · 20等分",
        "subtitle": "先筛选14:30及以后，再以该时段自己的中位数等样本分档",
        "filename": "07_14点30及以后独立样本_20等分.png",
    },
    {
        "dataset": "after",
        "divisions": 50,
        "metric_scope": "overall",
        "title": "14:30及以后独立样本 · 50等分",
        "subtitle": "先筛选14:30及以后，再以该时段自己的中位数等样本分档",
        "filename": "08_14点30及以后独立样本_50等分.png",
    },
]


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


def load_analysis() -> dict:
    module = runpy.run_path(str(ANALYSIS_SOURCE))
    return module["run_analysis"]()


def metric_key(metric_scope: str, metric_name: str) -> str:
    if metric_scope in {"before", "after"}:
        return f"{metric_scope}_{metric_name}"
    return metric_name


def split_interval(interval: str) -> str:
    return interval.replace(", ", ",\n")


def annotate_points(ax, x, values, formatter, span, colors, fontsize) -> None:
    for index, (x_value, value) in enumerate(zip(x, values)):
        offset = span * (0.034 if index % 2 == 0 else -0.052)
        ax.text(
            x_value,
            value + offset,
            formatter(value),
            ha="center",
            va="bottom" if offset > 0 else "top",
            fontsize=fontsize,
            color=colors[index],
            fontweight="bold",
        )


def build_chart(analysis: dict, spec: dict) -> Path:
    dataset = analysis["datasets"][spec["dataset"]]
    threshold = float(dataset["threshold"])
    rows = dataset["analyses"][str(spec["divisions"])]["rows"]
    divisions = int(spec["divisions"])
    half = divisions // 2
    scope = spec["metric_scope"]

    labels = [row["bucket_label"] for row in rows]
    directions = [row["direction"] for row in rows]
    intervals = [row["interval"] for row in rows]
    counts = [int(row[metric_key(scope, "sample_n")]) for row in rows]
    average_returns = np.array(
        [row[metric_key(scope, "average_strategy_return")] for row in rows],
        dtype=float,
    ) * 100
    win_rates = np.array(
        [row[metric_key(scope, "win_rate")] for row in rows], dtype=float
    ) * 100
    profit_loss_ratios = np.array(
        [row[metric_key(scope, "average_profit_loss_ratio")] for row in rows],
        dtype=float,
    )

    x = np.arange(divisions)
    short_color = "#D9772D"
    long_color = "#3F7F5A"
    colors = [short_color if direction == "做空" else long_color for direction in directions]

    if divisions == 20:
        figure_width = 22
        value_fontsize = 8
        tick_fontsize = 7.4
        dpi = 260
    else:
        figure_width = 36
        value_fontsize = 6.2
        tick_fontsize = 5.8
        dpi = 220

    fig, axes = plt.subplots(3, 1, figsize=(figure_width, 15), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(spec["title"], fontsize=23, fontweight="bold", y=0.978)
    fig.text(
        0.5,
        0.949,
        (
            f"{spec['subtitle']} ｜ 方向阈值={threshold:.6f} ｜ "
            f"Q01-Q{half:02d}做空（收益=-Tag） ｜ "
            f"Q{half + 1:02d}-Q{divisions:02d}做多（收益=Tag）"
        ),
        ha="center",
        fontsize=11.5,
        color="#475569",
    )

    for ax in axes:
        ax.set_facecolor("#FBFCFE")
        ax.grid(axis="y", color="#DCE3EC", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axvspan(-0.5, half - 0.5, color=short_color, alpha=0.055)
        ax.axvspan(half - 0.5, divisions - 0.5, color=long_color, alpha=0.055)
        ax.axvline(half - 0.5, color="#334155", linewidth=1.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#94A3B8")
        ax.tick_params(colors="#475569")

    # 平均策略收益柱状图
    ax = axes[0]
    bars = ax.bar(x, average_returns, width=0.72, color=colors, alpha=0.90)
    ax.axhline(0, color="#64748B", linewidth=1)
    ax.set_ylabel("平均策略收益（%）", fontsize=12)
    ax.set_title("平均策略收益", loc="left", fontsize=15, fontweight="bold")
    return_span = max(float(np.ptp(average_returns)), 0.01)
    ax.margins(y=0.25)
    for bar, value, color in zip(bars, average_returns, colors):
        offset = return_span * 0.035
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (offset if value >= 0 else -offset),
            f"{value:.4f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=value_fontsize,
            color=color,
            fontweight="bold",
        )

    # 胜率折线
    ax = axes[1]
    ax.plot(x[:half], win_rates[:half], color=short_color, marker="o", linewidth=2.1)
    ax.plot(x[half:], win_rates[half:], color=long_color, marker="o", linewidth=2.1)
    ax.axhline(50, color="#64748B", linewidth=1, linestyle=":")
    ax.set_ylabel("胜率（%）", fontsize=12)
    ax.set_title("胜率", loc="left", fontsize=15, fontweight="bold")
    win_pad = max(float(np.ptp(win_rates)) * 0.25, 1.0)
    ax.set_ylim(float(win_rates.min() - win_pad), float(win_rates.max() + win_pad))
    annotate_points(
        ax,
        x,
        win_rates,
        lambda value: f"{value:.2f}%",
        ax.get_ylim()[1] - ax.get_ylim()[0],
        colors,
        value_fontsize,
    )

    # 平均盈亏比折线
    ax = axes[2]
    ax.plot(
        x[:half], profit_loss_ratios[:half], color=short_color, marker="o", linewidth=2.1
    )
    ax.plot(
        x[half:], profit_loss_ratios[half:], color=long_color, marker="o", linewidth=2.1
    )
    ax.axhline(1, color="#64748B", linewidth=1, linestyle=":")
    ax.set_ylabel("平均盈亏比（倍）", fontsize=12)
    ax.set_title("平均盈亏比", loc="left", fontsize=15, fontweight="bold")
    ratio_pad = max(float(np.ptp(profit_loss_ratios)) * 0.25, 0.025)
    ax.set_ylim(
        float(profit_loss_ratios.min() - ratio_pad),
        float(profit_loss_ratios.max() + ratio_pad),
    )
    annotate_points(
        ax,
        x,
        profit_loss_ratios,
        lambda value: f"{value:.2f}x",
        ax.get_ylim()[1] - ax.get_ylim()[0],
        colors,
        value_fontsize,
    )

    tick_labels = [
        f"{label}\n{split_interval(interval)}\n{count:,}条"
        for label, interval, count in zip(labels, intervals, counts)
    ]
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tick_labels, fontsize=tick_fontsize, linespacing=1.24)
    axes[2].tick_params(axis="x", length=0, pad=8)
    axes[2].set_xlabel("档位 / 概率区间 / 当前图表口径下的样本量", fontsize=12, labelpad=12)

    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.text(
            (half - 1) / 2,
            ymax,
            "阈值左侧 · 做空",
            ha="center",
            va="bottom",
            color=short_color,
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            half + (half - 1) / 2,
            ymax,
            "阈值右侧 · 做多",
            ha="center",
            va="bottom",
            color=long_color,
            fontsize=10,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.055, right=0.992, top=0.91, bottom=0.20, hspace=0.38)
    output_path = OUTPUT_DIR / spec["filename"]
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def generate_all_charts() -> list[Path]:
    configure_chinese_font()
    analysis = load_analysis()
    output_paths = []
    for spec in CHART_SPECS:
        output_path = build_chart(analysis, spec)
        output_paths.append(output_path)
        print(f"已生成：{output_path.name}")
    return output_paths


# %%
generated_images = generate_all_charts()

# 在Jupyter中运行时，依次显示生成结果；普通Python运行时只保存文件。
try:
    get_ipython  # type: ignore[name-defined]
except NameError:
    pass
else:
    from IPython.display import Image, display

    for generated_image in generated_images:
        display(Image(filename=str(generated_image), width=1600))

