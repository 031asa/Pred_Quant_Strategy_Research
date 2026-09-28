# %% [markdown]
# # 预测值与 Tag 的月度时间漂移分析
#
# 口径：
# - score = pred + 0.5，并限制在 [0, 1]
# - 14:30 整点归入“14:30及以后”
# - 分别观察 14:30 前、14:30 后的 score 与原始 Tag 分布
# - 同一时段内所有月份共用固定分箱，避免逐月重新分箱掩盖漂移
# - 热力图显示范围采用该时段全样本 P0.5～P99.5，超出部分压入两端箱
# - 基准期为各时段最早 6 个自然月份，仅用于画中心变化参考线
# - 不计算 PSI、KS、Wasserstein 等量化漂移指标

# %%
from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import PowerNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import polars as pl


# %%
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

CONFIG_PATH = BASE_DIR / "中位数阈值_20与50等分_参数.json"
OUTPUT_BEFORE = BASE_DIR / "预测值与Tag时间漂移_14点30以前.png"
OUTPUT_AFTER = BASE_DIR / "预测值与Tag时间漂移_14点30及以后.png"
OUTPUT_PDF = BASE_DIR / "预测值与Tag时间漂移分析.pdf"
OUTPUT_CSV = BASE_DIR / "预测值与Tag月度漂移统计.csv"
OUTPUT_README = BASE_DIR / "README_预测值与Tag时间漂移分析.md"
OUTPUT_ZIP = BASE_DIR / "预测值与Tag时间漂移分析_交付包.zip"

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = json.load(config_file)

SOURCE_PATH = Path(CONFIG["source_path"])
PRED_COLUMN = CONFIG["pred_column"]
RETURN_COLUMN = CONFIG["return_column"]
DATETIME_COLUMN = CONFIG["datetime_column"]
SCORE_OFFSET = float(CONFIG["score_offset"])
SCORE_MIN = float(CONFIG["score_min"])
SCORE_MAX = float(CONFIG["score_max"])
TIME_CUTOFF = CONFIG["time_cutoff"]
BASELINE_MONTHS = 6
HIST_BINS = 42


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
    # 嵌入 TrueType 字体，避免 PDF 在其他电脑上打开时中文消失。
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def load_source() -> pl.DataFrame:
    cutoff_hour, cutoff_minute = [int(value) for value in TIME_CUTOFF.split(":")]
    cutoff_minutes = cutoff_hour * 60 + cutoff_minute
    return (
        pl.read_parquet(SOURCE_PATH)
        .select([DATETIME_COLUMN, PRED_COLUMN, RETURN_COLUMN])
        .drop_nulls()
        .with_columns(
            (pl.col(PRED_COLUMN).cast(pl.Float64) + SCORE_OFFSET)
            .clip(SCORE_MIN, SCORE_MAX)
            .alias("score"),
            pl.col(RETURN_COLUMN).cast(pl.Float64).alias("tag"),
            pl.col(DATETIME_COLUMN).dt.strftime("%Y-%m").alias("month"),
            (
                pl.col(DATETIME_COLUMN).dt.hour().cast(pl.Int32) * 60
                + pl.col(DATETIME_COLUMN).dt.minute().cast(pl.Int32)
            ).alias("minute_of_day"),
        )
        .with_columns(
            pl.when(pl.col("minute_of_day") < cutoff_minutes)
            .then(pl.lit("14:30以前"))
            .otherwise(pl.lit("14:30及以后"))
            .alias("segment")
        )
        .select(["month", "segment", "score", "tag"])
    )


def quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "p10": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p90": float(q[4]),
    }


def build_segment_data(source: pl.DataFrame, segment: str) -> dict:
    frame = source.filter(pl.col("segment") == segment).sort("month")
    months = frame["month"].unique(maintain_order=True).to_list()

    score_all = frame["score"].to_numpy()
    tag_all = frame["tag"].to_numpy()
    score_display = np.quantile(score_all, [0.005, 0.995])
    tag_display = np.quantile(tag_all, [0.005, 0.995])
    score_edges = np.linspace(score_display[0], score_display[1], HIST_BINS + 1)
    tag_edges = np.linspace(tag_display[0], tag_display[1], HIST_BINS + 1)

    score_histograms = []
    tag_histograms = []
    rows = []
    for month in months:
        monthly = frame.filter(pl.col("month") == month)
        score = monthly["score"].to_numpy()
        tag = monthly["tag"].to_numpy()
        score_clipped = np.clip(score, score_edges[0] + 1e-15, score_edges[-1] - 1e-15)
        tag_clipped = np.clip(tag, tag_edges[0] + 1e-15, tag_edges[-1] - 1e-15)
        score_hist, _ = np.histogram(score_clipped, bins=score_edges)
        tag_hist, _ = np.histogram(tag_clipped, bins=tag_edges)
        score_histograms.append(score_hist / score_hist.sum())
        tag_histograms.append(tag_hist / tag_hist.sum())

        sq = quantiles(score)
        tq = quantiles(tag)
        rows.append(
            {
                "segment": segment,
                "month": month,
                "sample_n": int(len(score)),
                "score_mean": float(np.mean(score)),
                "score_std": float(np.std(score, ddof=1)),
                **{f"score_{key}": value for key, value in sq.items()},
                "tag_mean": float(np.mean(tag)),
                "tag_std": float(np.std(tag, ddof=1)),
                **{f"tag_{key}": value for key, value in tq.items()},
                "tag_positive_rate": float(np.mean(tag > 0)),
            }
        )

    baseline_months = months[:BASELINE_MONTHS]
    baseline = frame.filter(pl.col("month").is_in(baseline_months))
    baseline_score_median = float(np.median(baseline["score"].to_numpy()))
    baseline_tag_mean = float(np.mean(baseline["tag"].to_numpy()))
    return {
        "segment": segment,
        "months": months,
        "rows": rows,
        "score_edges": score_edges,
        "tag_edges": tag_edges,
        "score_hist": np.asarray(score_histograms).T,
        "tag_hist": np.asarray(tag_histograms).T,
        "baseline_months": baseline_months,
        "baseline_score_median": baseline_score_median,
        "baseline_tag_mean": baseline_tag_mean,
    }


def _month_axis(ax, x: np.ndarray, labels: list[str]) -> None:
    step = 2 if len(labels) > 18 else 1
    ticks = x[::step]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels[::step], rotation=45, ha="right")
    ax.set_xlim(-0.5, len(labels) - 0.5)


def _heatmap(ax, values, edges, months, title, y_label, formatter=None, cmap="Blues"):
    x_edges = np.arange(len(months) + 1) - 0.5
    vmax = max(float(np.quantile(values[values > 0], 0.995)), 1e-8)
    mesh = ax.pcolormesh(
        x_edges,
        edges,
        values,
        cmap=cmap,
        norm=PowerNorm(gamma=0.58, vmin=0, vmax=vmax),
        shading="auto",
    )
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel(y_label)
    _month_axis(ax, np.arange(len(months)), months)
    if formatter is not None:
        ax.yaxis.set_major_formatter(formatter)
    colorbar = ax.figure.colorbar(mesh, ax=ax, pad=0.012, fraction=0.035)
    colorbar.set_label("当月样本占比密度")
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1%}"))


def plot_segment(data: dict, output_path: Path):
    months = data["months"]
    rows = data["rows"]
    x = np.arange(len(months))
    n = np.array([row["sample_n"] for row in rows])

    score_p10 = np.array([row["score_p10"] for row in rows])
    score_p25 = np.array([row["score_p25"] for row in rows])
    score_median = np.array([row["score_median"] for row in rows])
    score_p75 = np.array([row["score_p75"] for row in rows])
    score_p90 = np.array([row["score_p90"] for row in rows])
    tag_p10 = np.array([row["tag_p10"] for row in rows]) * 10000
    tag_p25 = np.array([row["tag_p25"] for row in rows]) * 10000
    tag_median = np.array([row["tag_median"] for row in rows]) * 10000
    tag_p75 = np.array([row["tag_p75"] for row in rows]) * 10000
    tag_p90 = np.array([row["tag_p90"] for row in rows]) * 10000
    tag_mean = np.array([row["tag_mean"] for row in rows]) * 10000

    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.35, 1.35, 0.82])
    ax_score_heat = fig.add_subplot(grid[0, 0])
    ax_score_band = fig.add_subplot(grid[0, 1])
    ax_tag_heat = fig.add_subplot(grid[1, 0])
    ax_tag_band = fig.add_subplot(grid[1, 1])
    ax_compare = fig.add_subplot(grid[2, 0])
    ax_n = fig.add_subplot(grid[2, 1])

    fig.suptitle(
        f"预测值与 Tag 月度时间漂移｜{data['segment']}",
        fontsize=22,
        fontweight="bold",
        y=1.035,
    )
    fig.text(
        0.5,
        1.005,
        (
            "score = pred + 0.5｜固定分箱｜原始 Tag（未加多空方向）｜"
            f"基准期：{data['baseline_months'][0]} 至 {data['baseline_months'][-1]}"
        ),
        ha="center",
        color="#526174",
        fontsize=11,
    )

    _heatmap(
        ax_score_heat,
        data["score_hist"],
        data["score_edges"],
        months,
        "A. score 固定分箱分布",
        "score",
        cmap="Blues",
    )

    ax_score_band.fill_between(x, score_p10, score_p90, color="#B7D2F0", alpha=0.42, label="P10～P90")
    ax_score_band.fill_between(x, score_p25, score_p75, color="#5A96DB", alpha=0.35, label="P25～P75")
    ax_score_band.plot(x, score_median, color="#1756A9", linewidth=2.2, marker="o", markersize=3.5, label="中位数")
    ax_score_band.axhline(data["baseline_score_median"], color="#64748B", linewidth=1.2, linestyle="--", label="基准期中位数")
    ax_score_band.set_title("B. score 月度分位数", loc="left", fontsize=13, fontweight="bold")
    ax_score_band.set_ylabel("score")
    _month_axis(ax_score_band, x, months)
    ax_score_band.legend(loc="best", frameon=False, ncol=2)

    _heatmap(
        ax_tag_heat,
        data["tag_hist"],
        data["tag_edges"] * 10000,
        months,
        "C. Tag 固定分箱分布",
        "Tag（bp）",
        formatter=FuncFormatter(lambda value, _: f"{value:.0f}"),
        cmap="YlOrRd",
    )

    ax_tag_band.fill_between(x, tag_p10, tag_p90, color="#FFD5B5", alpha=0.45, label="P10～P90")
    ax_tag_band.fill_between(x, tag_p25, tag_p75, color="#F59E63", alpha=0.36, label="P25～P75")
    ax_tag_band.plot(x, tag_median, color="#C44B17", linewidth=2.0, marker="o", markersize=3.2, label="中位数")
    ax_tag_band.plot(x, tag_mean, color="#7F1D1D", linewidth=1.8, marker="s", markersize=3.0, label="均值")
    ax_tag_band.axhline(data["baseline_tag_mean"] * 10000, color="#64748B", linewidth=1.2, linestyle="--", label="基准期均值")
    ax_tag_band.axhline(0, color="#94A3B8", linewidth=0.9)
    ax_tag_band.set_title("D. Tag 月度分位数与均值", loc="left", fontsize=13, fontweight="bold")
    ax_tag_band.set_ylabel("Tag（bp）")
    _month_axis(ax_tag_band, x, months)
    ax_tag_band.legend(loc="best", frameon=False, ncol=2)

    score_shift = (score_median - data["baseline_score_median"]) * 100
    tag_shift = tag_mean - data["baseline_tag_mean"] * 10000
    ax_compare.plot(x, score_shift, color="#1756A9", linewidth=2.1, marker="o", markersize=3.2, label="score 中位数变化（百分点）")
    ax_compare.axhline(0, color="#94A3B8", linewidth=0.9)
    ax_compare.set_ylabel("score变化（百分点）", color="#1756A9")
    ax_compare.tick_params(axis="y", labelcolor="#1756A9")
    ax_compare_2 = ax_compare.twinx()
    ax_compare_2.plot(x, tag_shift, color="#C44B17", linewidth=2.1, marker="s", markersize=3.2, label="Tag 均值变化（bp）")
    ax_compare_2.set_ylabel("Tag均值变化（bp）", color="#C44B17")
    ax_compare_2.tick_params(axis="y", labelcolor="#C44B17")
    ax_compare.set_title("E. 相对基准期的漂移方向对照", loc="left", fontsize=13, fontweight="bold")
    _month_axis(ax_compare, x, months)
    lines = ax_compare.get_lines()[:1] + ax_compare_2.get_lines()
    ax_compare.legend(lines, [line.get_label() for line in lines], loc="best", frameon=False)

    ax_n.bar(x, n, width=0.76, color="#8CAED6")
    ax_n.set_title("F. 每月样本量", loc="left", fontsize=13, fontweight="bold")
    ax_n.set_ylabel("条数")
    ax_n.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:.0f}k"))
    _month_axis(ax_n, x, months)

    for ax in [ax_score_band, ax_tag_band, ax_compare, ax_n]:
        ax.grid(axis="y", color="#DDE4EC", linewidth=0.65, alpha=0.75)
        ax.set_axisbelow(True)
    for ax in [ax_score_heat, ax_tag_heat, ax_score_band, ax_tag_band, ax_compare, ax_n]:
        ax.spines[["top", "right"]].set_visible(False)
    ax_compare_2.spines["top"].set_visible(False)

    fig.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    return fig


def write_csv(datasets: list[dict]) -> None:
    rows = [row for data in datasets for row in data["rows"]]
    fieldnames = list(rows[0].keys())
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme() -> None:
    text = f"""# 预测值与 Tag 时间漂移分析

## 1. 这份报告要回答什么

报告只观察两个原始变量随日历时间的分布变化：

1. 模型预测分数 `score` 是否逐渐向高分或低分移动；
2. 原始收益 `Tag` 是否同步向正收益或负收益移动；
3. 两者的漂移方向是否大致一致。

当前阶段不寻找最优阈值、不计算策略收益，也不计算 PSI、KS、Wasserstein 等漂移指标。

## 2. 为什么先这样做

模型最终是否有效，至少受到三件事影响：

1. **预测端变化**：模型输出的分数水平是否随时间改变；
2. **市场端变化**：真实 `Tag` 的收益分布是否随时间改变；
3. **映射关系变化**：相同分数是否仍然对应相近的 `Tag`。

当前报告先处理前两件事。逻辑顺序如下：

1. 先看 `score` 的边际分布，确认固定阈值面对的样本结构是否在变化；
2. 再看原始 `Tag` 的边际分布，确认真实市场收益环境是否同时变化；
3. 对照两者方向，判断预测值漂移是否至少与结果端漂移相容；
4. 如果方向经常相反，固定概率阈值的经济含义可能不稳定，需要继续检查校准和条件收益；
5. 如果方向大体一致，也只能说明现象相容，不能据此证明模型产生了超额收益，因为两者可能共同受到市场状态或时间因素推动。

之所以拆分14:30，是因为此前已观察到两个时段的 `score` 中位数和 `Tag` 水平差异很大。若混合分析，每个月14:30后样本结构或交易日构成的变化，可能被误判为跨月份的模型漂移。

之所以按月，是因为数据约覆盖两年半：按日噪声太大，按季度又容易掩盖变化发生的具体时间，月份是在稳定性与定位能力之间的折中。

之所以同时使用热力图和分位数图，是因为单看均值或中位数只能看到中心，无法区分“整体平移”“分布变宽”和“少量极端值推动”。热力图负责展示完整形状，分位数图负责准确追踪中心与带宽。

之所以使用最初6个月作基准，是为了给后续月份提供固定参照，而不是让基准随着后续数据一起变化。这个基准只用于描述，不是训练集划分，也不是最优参数。

可以用下面的判断框架理解结果：

| score漂移 | Tag漂移 | 初步含义 | 下一步 |
|---|---|---|---|
| 不明显 | 不明显 | 两端边际分布相对稳定 | 检查同月分档排序能力 |
| 明显 | 同方向 | 预测变化与市场变化相容，但可能只是共同市场状态 | 做市场中性化并检查条件收益 |
| 明显 | 反方向 | 分数绝对水平可能失去稳定经济含义 | 检查校准、滚动阈值和特征漂移 |
| 不明显 | 明显 | 模型输出未充分响应市场状态变化 | 检查模型敏感度和训练窗口 |

无论属于哪一种情况，边际分布都不能单独证明预测能力。真正的模型检验还要回答：在同一月份、同一时段内，`score` 越高时 `Tag` 是否仍然系统性越高。

## 3. 数据与计算口径

- 数据：`{SOURCE_PATH}`
- `score = pred + 0.5`，并限制在 `[0, 1]`
- `Tag` 使用 parquet 中原始值，不施加做多/做空符号
- 时间切分：`< {TIME_CUTOFF}` 与 `>= {TIME_CUTOFF}`，14:30 整点属于后一段
- 按自然月观察，从 2024-01 到 2026-06
- 最初 {BASELINE_MONTHS} 个月作为图中漂移方向的参考基准

`Tag` 图中采用基点（bp）显示：`1 bp = 0.01% = 0.0001` 原始收益。

## 4. 热力图的分箱方法

### 4.1 为什么必须固定分箱

每个时段先使用该时段的全部月份共同确定一套固定边界，之后每个月都套用完全相同的边界。
如果每个月重新做等频分箱，每档都会被强行放入相近数量的样本，真实的分布移动反而会被隐藏。

### 4.2 当前具体做法

- 14:30以前和14:30及以后分别建图、分别确定显示范围；
- 对 `score` 和 `Tag` 分别取该时段全样本的 P0.5 与 P99.5；
- 在 P0.5～P99.5 之间等宽切成 {HIST_BINS} 个固定小区间；
- 低于 P0.5 的极端样本并入最下面一箱；
- 高于 P99.5 的极端样本并入最上面一箱；
- 每一个色块表示“该月落在这个固定区间内的样本数 / 该月总样本数”。

两张页面各自使用一套边界，因此适合观察同一时段内部随月份的迁移；不要仅凭颜色深浅直接比较14:30前和14:30后哪个时段样本更多。

### 4.3 热力图怎么读

- 横轴：月份；
- 纵轴：固定的 `score` 区间或 `Tag` 区间；
- 颜色越深：该月集中在该区间的样本占比越高；
- 深色带整体向上移动：分布中心向更高数值漂移；
- 深色带整体向下移动：分布中心向更低数值漂移；
- 深色带变宽：离散程度或波动扩大；
- 上下两端同时变深：尾部变厚，极端值出现得更多；
- 单独某个月出现竖向亮带或宽带：该月分布结构与其他月份明显不同。

颜色表示的是当月占比，不是收益好坏，也不是样本的绝对数量。绝对样本量应查看 F 图。

## 5. 六个面板分别怎么看

### A. score 固定分箱分布

观察预测值整体向左或向右漂移，以及分布是否变宽。重点跟踪深蓝色高密度区域的位置。

### B. score 月度分位数

- 实线：每月中位数；
- 深色带：P25～P75，表示中间 50% 样本；
- 浅色带：P10～P90，表示中间 80% 样本；
- 灰色虚线：最初 6 个月的基准期中位数。

中位数反映分布中心，带宽反映离散程度。中位数上移但带宽不变，属于整体平移；带宽扩大则说明模型输出同时变得更分散。

### C. Tag 固定分箱分布

读法与 A 相同，但纵轴单位为 bp。深色带向上表示真实收益分布偏向正值，向下表示偏向负值，变宽表示收益波动扩大。

### D. Tag 月度分位数与均值

除了中位数和分位带，还画出了月度均值。均值容易受到少量极端收益影响，中位数更代表典型样本：

- 均值和中位数同方向移动：漂移相对广泛；
- 均值明显移动而中位数接近零：更可能由尾部或少量极端样本推动；
- P10～P90 明显变宽：当月波动或尾部风险上升。

### E. 相对基准期的漂移方向对照

- 蓝线：每月 `score` 中位数减去基准期中位数，单位为百分点；
- 橙线：每月 `Tag` 均值减去基准期均值，单位为 bp；
- 两条线都在零线上方或都在零线下方：当月漂移方向一致；
- 一条在线上、一条在线下：当月漂移方向相反。

这是双纵轴图，只能比较方向和发生时间，不能比较两条线的高度大小，也不能据此判断因果关系。

### F. 每月样本量

用于排查春节、交易日数量等导致的月度样本差异。样本较少月份的中心和尾部分位数通常更容易波动。

## 6. 当前图形的初步观察

### 14:30以前

- `score` 中位数围绕基准上下波动，没有看到持续单向上移或下移；
- `Tag` 的主体长期集中在零附近，但不同月份的带宽和尾部厚度有变化；
- `score` 与 `Tag` 的月度方向并不总是一致，暂时不支持“分数整体上移时 Tag 必然同步上移”。

### 14:30及以后

- `score` 的月度中心变化和分布宽度明显大于14:30以前；
- `Tag` 的均值、分位区间和尾部也更不稳定；
- 存在明显错位月份，例如 2024-09 的 `score` 中位数约为 0.517，而 `Tag` 均值约为 +99.42 bp；2024-10 的 `score` 中位数升至约 0.583，但 `Tag` 均值约为 -17.54 bp；
- 2026-03 的 `score` 中位数约为 0.558、`Tag` 均值约为 -7.32 bp，而 2026-04 的 `score` 中位数降至约 0.512、`Tag` 均值却约为 +34.23 bp。

这些现象说明预测分数的绝对水平与 `Tag` 的月度中心并非稳定同步。它不等同于模型失效：模型可能仍具有同月横截面或分档排序能力，后续需要另外验证。

## 7. 阅读时的限制

- 本报告是描述性分析，不构成统计显著性结论；
- `Tag` 均值可能受极端值影响，必须结合中位数和热力图一起看；
- 两个时段的样本量相差较大，14:30后的月度曲线天然更容易波动；
- 当前比较的是未经市场中性化的原始 `Tag`，其中可能包含市场整体方向；
- 当前只看边际分布，尚未回答“同一个月内高分是否对应更高 Tag”。

## 8. 输出文件

- `{OUTPUT_BEFORE.name}`：14:30 以前
- `{OUTPUT_AFTER.name}`：14:30 及以后
- `{OUTPUT_PDF.name}`：两页合并报告
- `{OUTPUT_CSV.name}`：月度统计明细
- `{Path(__file__).name}`：可在 PyCharm/Jupyter 中按单元运行的源码
"""
    OUTPUT_README.write_text(text, encoding="utf-8")


def build_zip() -> None:
    files = [
        OUTPUT_BEFORE,
        OUTPUT_AFTER,
        OUTPUT_PDF,
        OUTPUT_CSV,
        OUTPUT_README,
        Path(__file__).resolve(),
        CONFIG_PATH,
    ]
    with ZipFile(OUTPUT_ZIP, "w", compression=ZIP_DEFLATED) as archive:
        for file_path in files:
            archive.write(file_path, arcname=file_path.name)


# %%
configure_chinese_font()
source = load_source()
before_data = build_segment_data(source, "14:30以前")
after_data = build_segment_data(source, "14:30及以后")

before_figure = plot_segment(before_data, OUTPUT_BEFORE)
after_figure = plot_segment(after_data, OUTPUT_AFTER)

with PdfPages(OUTPUT_PDF) as pdf:
    pdf.savefig(before_figure, bbox_inches="tight", facecolor="white")
    pdf.savefig(after_figure, bbox_inches="tight", facecolor="white")

write_csv([before_data, after_data])
write_readme()
build_zip()

plt.close(before_figure)
plt.close(after_figure)

print(f"已生成：{OUTPUT_BEFORE}")
print(f"已生成：{OUTPUT_AFTER}")
print(f"已生成：{OUTPUT_PDF}")
print(f"已生成：{OUTPUT_CSV}")
print(f"已生成：{OUTPUT_ZIP}")
