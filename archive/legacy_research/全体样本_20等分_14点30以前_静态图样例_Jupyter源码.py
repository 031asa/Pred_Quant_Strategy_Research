# %% [markdown]
# # 全体样本 - 20等分 - 14:30以前
#
# 本文件可在 PyCharm 的 Jupyter 单元中直接运行，也可以复制到 Notebook。
# 图中直接标注指标数值、概率区间和样本量，不依赖悬停。

# %%
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
OUTPUT_IMAGE = BASE_DIR / "全体样本_20等分_14点30以前_静态图样例.png"


def configure_chinese_font() -> None:
    """优先使用Windows中文字体，避免图片中的中文消失。"""
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=str(font_path)).get_name()
            ]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_rows() -> tuple[list[dict], float]:
    """复用现有分析源码，保证图片与Excel的计算口径完全一致。"""
    module = runpy.run_path(str(ANALYSIS_SOURCE))
    analysis = module["run_analysis"]()
    dataset = analysis["datasets"]["global"]
    rows = dataset["analyses"]["20"]["rows"]
    return rows, float(dataset["threshold"])


def split_interval(interval: str) -> str:
    """把长区间拆成两行，方便20个档位同时显示。"""
    return interval.replace(", ", ",\n")


def annotate_points(ax, x, values, formatter, span, colors) -> None:
    """上下错开标注，减少相邻点文字重叠。"""
    for index, (x_value, value) in enumerate(zip(x, values)):
        offset = span * (0.035 if index % 2 == 0 else -0.055)
        va = "bottom" if offset > 0 else "top"
        ax.text(
            x_value,
            value + offset,
            formatter(value),
            ha="center",
            va=va,
            fontsize=8,
            color=colors[index],
            fontweight="bold",
        )


def build_chart(rows: list[dict], threshold: float) -> Path:
    configure_chinese_font()

    labels = [row["bucket_label"] for row in rows]
    directions = [row["direction"] for row in rows]
    intervals = [row["interval"] for row in rows]
    counts = [int(row["before_sample_n"]) for row in rows]
    average_returns = np.array(
        [row["before_average_strategy_return"] for row in rows], dtype=float
    ) * 100
    win_rates = np.array([row["before_win_rate"] for row in rows], dtype=float) * 100
    profit_loss_ratios = np.array(
        [row["before_average_profit_loss_ratio"] for row in rows], dtype=float
    )

    x = np.arange(len(rows))
    short_color = "#D9772D"
    long_color = "#3F7F5A"
    colors = [short_color if direction == "做空" else long_color for direction in directions]

    fig, axes = plt.subplots(3, 1, figsize=(22, 15), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "全体样本 · 20等分 · 14:30以前",
        fontsize=23,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.945,
        f"方向阈值（全体样本中位数）={threshold:.6f} ｜ Q01-Q10做空（收益=-Tag） ｜ Q11-Q20做多（收益=Tag）",
        ha="center",
        fontsize=12,
        color="#475569",
    )

    for ax in axes:
        ax.set_facecolor("#FBFCFE")
        ax.grid(axis="y", color="#DCE3EC", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axvspan(-0.5, 9.5, color=short_color, alpha=0.055)
        ax.axvspan(9.5, 19.5, color=long_color, alpha=0.055)
        ax.axvline(9.5, color="#334155", linewidth=1.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#94A3B8")
        ax.tick_params(colors="#475569")

    # 1. 平均策略收益
    ax = axes[0]
    bars = ax.bar(x, average_returns, width=0.72, color=colors, alpha=0.90)
    ax.axhline(0, color="#64748B", linewidth=1)
    ax.set_ylabel("平均策略收益（%）", fontsize=12)
    ax.set_title("平均策略收益", loc="left", fontsize=15, fontweight="bold")
    return_span = max(float(np.ptp(average_returns)), 0.01)
    ax.margins(y=0.24)
    for bar, value, color in zip(bars, average_returns, colors):
        offset = return_span * 0.035
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (offset if value >= 0 else -offset),
            f"{value:.4f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
            color=color,
            fontweight="bold",
        )

    # 2. 胜率
    ax = axes[1]
    ax.plot(x[:10], win_rates[:10], color=short_color, marker="o", linewidth=2.2)
    ax.plot(x[10:], win_rates[10:], color=long_color, marker="o", linewidth=2.2)
    ax.axhline(50, color="#64748B", linewidth=1, linestyle=":")
    ax.set_ylabel("胜率（%）", fontsize=12)
    ax.set_title("胜率", loc="left", fontsize=15, fontweight="bold")
    win_pad = max(float(np.ptp(win_rates)) * 0.22, 1.0)
    ax.set_ylim(float(win_rates.min() - win_pad), float(win_rates.max() + win_pad))
    annotate_points(ax, x, win_rates, lambda value: f"{value:.2f}%", ax.get_ylim()[1] - ax.get_ylim()[0], colors)

    # 3. 平均盈亏比
    ax = axes[2]
    ax.plot(x[:10], profit_loss_ratios[:10], color=short_color, marker="o", linewidth=2.2)
    ax.plot(x[10:], profit_loss_ratios[10:], color=long_color, marker="o", linewidth=2.2)
    ax.axhline(1, color="#64748B", linewidth=1, linestyle=":")
    ax.set_ylabel("平均盈亏比（倍）", fontsize=12)
    ax.set_title("平均盈亏比", loc="left", fontsize=15, fontweight="bold")
    ratio_pad = max(float(np.ptp(profit_loss_ratios)) * 0.24, 0.025)
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
    )

    tick_labels = [
        f"{label}\n{split_interval(interval)}\n{count:,}条"
        for label, interval, count in zip(labels, intervals, counts)
    ]
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tick_labels, fontsize=7.4, linespacing=1.25)
    axes[2].tick_params(axis="x", length=0, pad=8)
    axes[2].set_xlabel("档位 / 概率区间 / 14:30以前样本量", fontsize=12, labelpad=12)

    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.text(4.5, ymax, "阈值左侧 · 做空", ha="center", va="bottom", color=short_color, fontsize=10, fontweight="bold")
        ax.text(14.5, ymax, "阈值右侧 · 做多", ha="center", va="bottom", color=long_color, fontsize=10, fontweight="bold")

    fig.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.20, hspace=0.38)
    fig.savefig(OUTPUT_IMAGE, dpi=260, bbox_inches="tight", facecolor="white")
    plt.show()
    return OUTPUT_IMAGE


# %%
rows, threshold = load_rows()
image_path = build_chart(rows, threshold)
print(f"图片已生成：{image_path}")

