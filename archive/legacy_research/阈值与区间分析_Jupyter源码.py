# %%
"""
预测分数方向阈值与区间收益分析

可直接在 PyCharm 中按 # %% 分单元运行，也可以复制到 Jupyter Notebook。

核心定义：
1. score = pred + SCORE_OFFSET。
2. 对于“移动阈值表”，每一列 threshold 都是一个候选方向阈值：
   score > threshold 做多，score < threshold 做空，score == threshold 不交易。
3. 对于“固定区间表”，方向阈值固定为 DIRECTION_THRESHOLD：
   左侧区间做空，策略收益 = -Tag；
   右侧区间做多，策略收益 = Tag；
   score == DIRECTION_THRESHOLD 单列展示，但不计算收益或盈亏比。
4. 左侧区间左闭右开 [a,b)，右侧区间左开右闭 (a,b]。
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl


# %%
# =========================
# 可修改参数：统一放在同目录的“阈值分析参数.json”
# =========================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # 把源码复制进 Jupyter 单元格运行时，默认从当前工作目录找配置文件。
    BASE_DIR = Path.cwd()

CONFIG_PATH = BASE_DIR / "阈值分析参数.json"
with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = json.load(config_file)

SOURCE_PATH = Path(CONFIG["source_path"])

PRED_COLUMN = CONFIG["pred_column"]
RETURN_COLUMN = CONFIG["return_column"]
DATE_COLUMN = CONFIG["date_column"]
CONTRACT_COLUMN = CONFIG["contract_column"]

SCORE_OFFSET = float(CONFIG["score_offset"])
SCORE_MIN = float(CONFIG["score_min"])
SCORE_MAX = float(CONFIG["score_max"])

# 固定的多空方向阈值；本次配置为 0.50。
DIRECTION_THRESHOLD = float(CONFIG["direction_threshold"])

# 在方向阈值左右各 DENSE_HALF_WIDTH 范围内按 DENSE_STEP 加密。
# 本次对应 0.30–0.70 每 0.02 一档；两端分别合并。
DENSE_HALF_WIDTH = float(CONFIG["dense_half_width"])
DENSE_STEP = float(CONFIG["dense_step"])

# 数值比较容差，仅用于检查配置和去除浮点误差。
EPSILON = 1e-12


# %%
def _decimal_grid(start: float, stop: float, step: float) -> list[float]:
    """使用 Decimal 生成包含 start 和 stop 的稳定网格。"""
    current = Decimal(str(start))
    end = Decimal(str(stop))
    increment = Decimal(str(step))
    values: list[float] = []

    while current <= end:
        values.append(float(current))
        current += increment

    if not np.isclose(values[-1], stop, atol=EPSILON):
        values.append(float(end))
    return values


def validate_config() -> None:
    if not SCORE_MIN < DIRECTION_THRESHOLD < SCORE_MAX:
        raise ValueError("DIRECTION_THRESHOLD 必须严格位于 SCORE_MIN 与 SCORE_MAX 之间。")
    if DENSE_HALF_WIDTH <= 0 or DENSE_STEP <= 0:
        raise ValueError("DENSE_HALF_WIDTH 和 DENSE_STEP 必须大于 0。")

    dense_left = DIRECTION_THRESHOLD - DENSE_HALF_WIDTH
    dense_right = DIRECTION_THRESHOLD + DENSE_HALF_WIDTH
    if dense_left <= SCORE_MIN or dense_right >= SCORE_MAX:
        raise ValueError("密集区间必须严格位于 SCORE_MIN 与 SCORE_MAX 内部。")


def build_thresholds() -> list[float]:
    """
    生成移动阈值和固定区间共用的边界。

    默认结果：
    0, 0.30, 0.32, ..., 0.50, ..., 0.68, 0.70, 1.00
    """
    validate_config()
    dense_left = DIRECTION_THRESHOLD - DENSE_HALF_WIDTH
    dense_right = DIRECTION_THRESHOLD + DENSE_HALF_WIDTH
    dense_values = _decimal_grid(dense_left, dense_right, DENSE_STEP)

    values = [SCORE_MIN, *dense_values, DIRECTION_THRESHOLD, SCORE_MAX]
    return sorted({round(value, 12) for value in values})


def _point_label(value: float) -> int | float:
    scaled = value * 100
    rounded = round(scaled)
    return int(rounded) if np.isclose(scaled, rounded, atol=EPSILON) else round(scaled, 6)


def _point_text(value: float) -> str:
    return str(_point_label(value))


def build_interval_specs(thresholds: list[float]) -> list[dict]:
    """
    默认生成：
    [0,30)、[30,32)、...、[48,50)、50、(50,52]、...、(68,70]、(70,100]
    """
    left_specs: list[dict] = []
    right_specs: list[dict] = []

    for lower, upper in zip(thresholds[:-1], thresholds[1:]):
        if upper <= DIRECTION_THRESHOLD + EPSILON:
            left_specs.append(
                {
                    "label": f"[{_point_text(lower)},{_point_text(upper)})",
                    "direction": "short",
                    "lower": lower,
                    "upper": upper,
                }
            )
        elif lower >= DIRECTION_THRESHOLD - EPSILON:
            right_specs.append(
                {
                    "label": f"({_point_text(lower)},{_point_text(upper)}]",
                    "direction": "long",
                    "lower": lower,
                    "upper": upper,
                }
            )
        else:
            raise ValueError("区间跨越了方向阈值，请检查网格参数。")

    neutral = {
        "label": _point_text(DIRECTION_THRESHOLD),
        "direction": "neutral",
        "lower": DIRECTION_THRESHOLD,
        "upper": DIRECTION_THRESHOLD,
    }
    return [*left_specs, neutral, *right_specs]


# %%
def load_source(source_path: Path = SOURCE_PATH) -> pl.DataFrame:
    """读取 Parquet，并构造按月份分组字段。"""
    return (
        pl.read_parquet(source_path)
        .select([DATE_COLUMN, CONTRACT_COLUMN, PRED_COLUMN, RETURN_COLUMN])
        .drop_nulls()
        .with_columns(
            pl.col(DATE_COLUMN).dt.strftime("%Y-%m").alias("Month"),
        )
    )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def calculate_moving_threshold_metrics(
    returns: np.ndarray,
    scores: np.ndarray,
    thresholds: list[float],
) -> dict[str, list[float | None]]:
    """
    对每一个移动阈值分别计算：
    - Accuracy：多空两个方向合计的赚钱单数 / 实际交易单数
    - 做多 Precision：做多赚钱单数 / 做多单数
    - 做空 Precision：做空赚钱单数 / 做空单数
    - 做多/做空 Coverage：对应方向单数 / 全部样本数
    """
    total_n = len(scores)
    accuracy: list[float | None] = []
    long_precision: list[float | None] = []
    short_precision: list[float | None] = []
    long_coverage: list[float] = []
    short_coverage: list[float] = []

    for threshold in thresholds:
        long_mask = scores > threshold
        short_mask = scores < threshold

        long_n = int(np.sum(long_mask))
        short_n = int(np.sum(short_mask))
        traded_n = long_n + short_n

        long_wins = int(np.sum(long_mask & (returns > 0)))
        short_wins = int(np.sum(short_mask & (returns < 0)))

        accuracy.append(_safe_ratio(long_wins + short_wins, traded_n))
        long_precision.append(_safe_ratio(long_wins, long_n))
        short_precision.append(_safe_ratio(short_wins, short_n))
        long_coverage.append(float(long_n / total_n))
        short_coverage.append(float(short_n / total_n))

    return {
        "accuracy": accuracy,
        "long_precision": long_precision,
        "short_precision": short_precision,
        "long_coverage": long_coverage,
        "short_coverage": short_coverage,
    }


def calculate_fixed_interval_metrics(
    returns: np.ndarray,
    scores: np.ndarray,
    interval_specs: list[dict],
) -> dict[str, list[float | int | None]]:
    """
    固定 DIRECTION_THRESHOLD，只查询每个分数区间内的表现。

    平均策略收益率：
    - 做空区间：mean(-Tag)
    - 做多区间：mean(Tag)

    胜率：
    count(策略收益 > 0) / (count(策略收益 > 0) + count(策略收益 < 0))
    策略收益 == 0 不进入分子或分母；没有盈亏样本时返回空值。
    """
    average_strategy_return: list[float | None] = []
    interval_win_rate: list[float | None] = []
    average_profit_loss_ratio: list[float | None] = []
    interval_sample_n: list[int] = []

    for spec in interval_specs:
        direction = spec["direction"]
        lower = spec["lower"]
        upper = spec["upper"]

        if direction == "short":
            # 左侧左闭右开：[lower, upper)
            mask = (scores >= lower) & (scores < upper)
            strategy_returns = -returns[mask]
        elif direction == "long":
            # 右侧左开右闭：(lower, upper]
            mask = (scores > lower) & (scores <= upper)
            strategy_returns = returns[mask]
        else:
            # score == DIRECTION_THRESHOLD 单列，但不参与交易和收益统计。
            mask = scores == DIRECTION_THRESHOLD
            interval_sample_n.append(int(np.sum(mask)))
            average_strategy_return.append(None)
            interval_win_rate.append(None)
            average_profit_loss_ratio.append(None)
            continue

        sample_n = len(strategy_returns)
        wins = int(np.sum(strategy_returns > 0))
        losses = int(np.sum(strategy_returns < 0))
        positive_returns = strategy_returns[strategy_returns > 0]
        negative_returns = strategy_returns[strategy_returns < 0]
        positive_average = (
            float(np.mean(positive_returns)) if len(positive_returns) > 0 else None
        )
        negative_average = (
            float(np.mean(negative_returns)) if len(negative_returns) > 0 else None
        )

        interval_sample_n.append(int(sample_n))
        average_strategy_return.append(
            float(np.mean(strategy_returns)) if sample_n > 0 else None
        )
        interval_win_rate.append(_safe_ratio(wins, wins + losses))
        average_profit_loss_ratio.append(
            float(positive_average / abs(negative_average))
            if positive_average is not None
            and negative_average is not None
            and negative_average != 0
            else None
        )

    return {
        "average_strategy_return": average_strategy_return,
        "interval_win_rate": interval_win_rate,
        "average_profit_loss_ratio": average_profit_loss_ratio,
        "interval_sample_n": interval_sample_n,
    }


# %%
def calculate_one_group(
    group_name: str,
    group_df: pl.DataFrame,
    thresholds: list[float],
    interval_specs: list[dict],
) -> dict:
    returns = group_df[RETURN_COLUMN].to_numpy().astype(float)
    scores = np.clip(
        group_df[PRED_COLUMN].to_numpy().astype(float) + SCORE_OFFSET,
        SCORE_MIN,
        SCORE_MAX,
    )

    moving = calculate_moving_threshold_metrics(returns, scores, thresholds)
    intervals = calculate_fixed_interval_metrics(returns, scores, interval_specs)

    long_returns = returns[scores > DIRECTION_THRESHOLD]
    short_returns = -returns[scores < DIRECTION_THRESHOLD]
    fixed_returns = np.concatenate([long_returns, short_returns])

    def summarize_strategy(strategy_returns: np.ndarray) -> dict:
        wins = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        return {
            "trade_n": int(len(strategy_returns)),
            "average_strategy_return": (
                float(np.mean(strategy_returns))
                if len(strategy_returns) > 0
                else None
            ),
            "win_rate": _safe_ratio(len(wins), len(wins) + len(losses)),
            "average_profit_loss_ratio": (
                float(np.mean(wins) / abs(np.mean(losses)))
                if len(wins) > 0 and len(losses) > 0
                else None
            ),
        }

    return {
        "group": str(group_name),
        "n": int(len(scores)),
        "positive_n": int(np.sum(returns > 0)),
        "negative_n": int(np.sum(returns < 0)),
        "fixed_summary": {
            "overall": summarize_strategy(fixed_returns),
            "long": summarize_strategy(long_returns),
            "short": summarize_strategy(short_returns),
        },
        **moving,
        **intervals,
    }


def calculate_groups(
    df: pl.DataFrame,
    group_column: str,
    thresholds: list[float],
    interval_specs: list[dict],
) -> list[dict]:
    output: list[dict] = []
    group_values = sorted(df[group_column].unique().to_list())

    for group_value in group_values:
        group_df = df.filter(pl.col(group_column) == group_value)
        output.append(
            calculate_one_group(
                str(group_value),
                group_df,
                thresholds,
                interval_specs,
            )
        )
    return output


def run_analysis(source_path: Path = SOURCE_PATH) -> dict:
    df = load_source(source_path)
    thresholds = build_thresholds()
    interval_specs = build_interval_specs(thresholds)

    global_metrics = calculate_one_group(
        "全体",
        df,
        thresholds,
        interval_specs,
    )

    return {
        "source": str(source_path),
        "config": {
            "score_offset": SCORE_OFFSET,
            "score_min": SCORE_MIN,
            "score_max": SCORE_MAX,
            "direction_threshold": DIRECTION_THRESHOLD,
            "dense_half_width": DENSE_HALF_WIDTH,
            "dense_step": DENSE_STEP,
        },
        "total_n": df.height,
        "total_positive_n": global_metrics["positive_n"],
        "total_negative_n": global_metrics["negative_n"],
        "thresholds": thresholds,
        "threshold_labels": [_point_label(value) for value in thresholds],
        "interval_specs": interval_specs,
        "interval_labels": [spec["label"] for spec in interval_specs],
        "global": global_metrics,
        "contracts": calculate_groups(
            df,
            CONTRACT_COLUMN,
            thresholds,
            interval_specs,
        ),
        "months": calculate_groups(
            df,
            "Month",
            thresholds,
            interval_specs,
        ),
    }


# %%
def metric_table(
    groups: list[dict],
    group_header: str,
    labels: list[str | int | float],
    metric_key: str,
    global_metrics: dict,
) -> pl.DataFrame:
    """把某个指标整理成适合在 Jupyter 中查看的宽表。"""
    rows = [
        {
            group_header: group["group"],
            **dict(zip(labels, group[metric_key])),
        }
        for group in groups
    ]
    rows.append(
        {
            group_header: "全体",
            **dict(zip(labels, global_metrics[metric_key])),
        }
    )
    return pl.DataFrame(rows)


def build_jupyter_tables(analysis: dict) -> dict[str, dict[str, pl.DataFrame]]:
    threshold_labels = analysis["threshold_labels"]
    interval_labels = analysis["interval_labels"]

    metric_definitions = {
        "做多Precision_移动阈值": ("long_precision", threshold_labels),
        "做空Precision_移动阈值": ("short_precision", threshold_labels),
        "做多Coverage_移动阈值": ("long_coverage", threshold_labels),
        "做空Coverage_移动阈值": ("short_coverage", threshold_labels),
        "区间平均策略收益率_固定方向阈值": (
            "average_strategy_return",
            interval_labels,
        ),
        "区间胜率_固定方向阈值": (
            "interval_win_rate",
            interval_labels,
        ),
        "区间平均盈亏比_固定方向阈值": (
            "average_profit_loss_ratio",
            interval_labels,
        ),
    }

    output: dict[str, dict[str, pl.DataFrame]] = {}
    for dimension, groups, group_header in [
        ("按合约", analysis["contracts"], "合约"),
        ("按月份", analysis["months"], "月份"),
    ]:
        output[dimension] = {}
        for table_name, (metric_key, labels) in metric_definitions.items():
            output[dimension][table_name] = metric_table(
                groups,
                group_header,
                labels,
                metric_key,
                analysis["global"],
            )
    return output


def display_jupyter_tables(tables: dict[str, dict[str, pl.DataFrame]]) -> None:
    try:
        from IPython.display import display
    except ImportError:
        display = print

    for dimension, dimension_tables in tables.items():
        print(f"\n===== {dimension} =====")
        for table_name, table in dimension_tables.items():
            print(f"\n--- {table_name} ---")
            display(table)


# %%
def build_cumulative_return_curve(
    source_path: Path = SOURCE_PATH,
    threshold: float = DIRECTION_THRESHOLD,
) -> pl.DataFrame:
    """
    按给定阈值生成日频累计收益曲线数据。

    同一交易日内的全部有效信号先做等权平均，再用复利口径累计：
    cumulative_return = cumprod(1 + daily_strategy_return) - 1
    """
    source = load_source(source_path)
    daily = (
        source.with_columns(
            (
                pl.col(PRED_COLUMN).cast(pl.Float64) + SCORE_OFFSET
            )
            .clip(SCORE_MIN, SCORE_MAX)
            .alias("score")
        )
        .with_columns(
            pl.when(pl.col("score") > threshold)
            .then(pl.col(RETURN_COLUMN).cast(pl.Float64))
            .when(pl.col("score") < threshold)
            .then(-pl.col(RETURN_COLUMN).cast(pl.Float64))
            .otherwise(None)
            .alias("strategy_return")
        )
        .group_by(DATE_COLUMN)
        .agg(
            pl.col("strategy_return").mean().alias("daily_strategy_return"),
            pl.col("strategy_return").count().alias("trade_n"),
        )
        .drop_nulls("daily_strategy_return")
        .sort(DATE_COLUMN)
        .with_columns(
            (1.0 + pl.col("daily_strategy_return"))
            .cum_prod()
            .alias("wealth")
        )
        .with_columns(
            (pl.col("wealth") - 1.0).alias("cumulative_return"),
            (
                pl.col("wealth")
                / pl.max_horizontal(
                    pl.lit(1.0),
                    pl.col("wealth").cum_max(),
                )
                - 1.0
            ).alias("drawdown"),
        )
    )
    return daily


def plot_cumulative_return_curve(
    curve: pl.DataFrame,
    threshold: float = DIRECTION_THRESHOLD,
) -> None:
    """在 Jupyter 中绘制给定阈值的累计收益率曲线。"""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    dates = curve[DATE_COLUMN].to_list()
    cumulative_return = curve["cumulative_return"].to_numpy()
    final_return = float(cumulative_return[-1])
    max_drawdown = float(curve["drawdown"].min())

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(
        dates,
        cumulative_return,
        color="#4472C4",
        linewidth=1.8,
        label="累计收益率",
    )
    ax.axhline(0, color="#808080", linewidth=0.8)
    ax.fill_between(
        dates,
        cumulative_return,
        0,
        where=cumulative_return >= 0,
        color="#E2F0D9",
        alpha=0.45,
    )
    ax.fill_between(
        dates,
        cumulative_return,
        0,
        where=cumulative_return < 0,
        color="#F4CCCC",
        alpha=0.45,
    )
    ax.set_title(
        f"固定阈值 {threshold:.2f} 的累计收益率"
        f"｜期末 {final_return:.2%}｜最大回撤 {max_drawdown:.2%}"
    )
    ax.set_xlabel("交易日")
    ax.set_ylabel("累计收益率")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument(
        "--json",
        action="store_true",
        help="供 Excel 构建脚本读取；普通 Jupyter 运行时不要设置。",
    )
    args, _unknown = parser.parse_known_args()

    analysis_result = run_analysis(args.source)

    if args.json:
        print(json.dumps(analysis_result, ensure_ascii=False, allow_nan=False))
    else:
        jupyter_tables = build_jupyter_tables(analysis_result)
        display_jupyter_tables(jupyter_tables)
        cumulative_curve = build_cumulative_return_curve(
            args.source,
            DIRECTION_THRESHOLD,
        )
        plot_cumulative_return_curve(
            cumulative_curve,
            DIRECTION_THRESHOLD,
        )
