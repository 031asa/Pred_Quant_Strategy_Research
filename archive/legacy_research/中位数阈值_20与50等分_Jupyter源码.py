# %%
"""
全体样本：中位数方向阈值 + 等样本20/50等分分析

定义：
1. score = pred + SCORE_OFFSET，并限制在 [SCORE_MIN, SCORE_MAX]。
2. 方向阈值取全体 score 的 50% 分位数（中位数）。
3. 阈值左边全部做空，策略收益 = -Tag；阈值右边全部做多，策略收益 = Tag。
4. 20等分时 Q01-Q10 做空、Q11-Q20 做多；50等分时 Q01-Q25 做空、
   Q26-Q50 做多。
5. 左侧区间采用 [a,b)，右侧区间采用 (a,b]；阈值相等样本不交易。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


# %%
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

CONFIG_PATH = BASE_DIR / "中位数阈值_20与50等分_参数.json"
with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = json.load(config_file)

SOURCE_PATH = Path(CONFIG["source_path"])
PRED_COLUMN = CONFIG["pred_column"]
RETURN_COLUMN = CONFIG["return_column"]
DATETIME_COLUMN = CONFIG["datetime_column"]
SCORE_OFFSET = float(CONFIG["score_offset"])
SCORE_MIN = float(CONFIG["score_min"])
SCORE_MAX = float(CONFIG["score_max"])
THRESHOLD_QUANTILE = float(CONFIG["threshold_quantile"])
TIME_CUTOFF = CONFIG["time_cutoff"]
DIVISION_COUNTS = [int(value) for value in CONFIG["division_counts"]]


# %%
def load_source(source_path: Path = SOURCE_PATH) -> pl.DataFrame:
    """读取必要字段并删除空值。"""
    return (
        pl.read_parquet(source_path)
        .select([DATETIME_COLUMN, PRED_COLUMN, RETURN_COLUMN])
        .drop_nulls()
    )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _summarize_strategy(strategy_returns: np.ndarray) -> dict:
    """计算指定样本集合的样本量、平均收益、胜率和平均盈亏比。"""
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    return {
        "sample_n": int(len(strategy_returns)),
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


def _boundary_text(value: float) -> str:
    return f"{value:.6f}"


def calculate_median_threshold(scores: np.ndarray) -> float:
    """用两侧中央样本的中点计算精确中位数边界。"""
    ordered = np.sort(scores, kind="stable")
    n = len(ordered)
    if n == 0:
        raise ValueError("没有可用于计算中位数的样本。")
    if n % 2 == 1:
        return float(ordered[n // 2])
    return float((ordered[n // 2 - 1] + ordered[n // 2]) / 2.0)


def calculate_equal_frequency(
    scores: np.ndarray,
    returns: np.ndarray,
    before_cutoff: np.ndarray,
    divisions: int,
    threshold: float,
) -> dict:
    """按全体 score 排序后等样本分档，并按中位数左右计算多空指标。"""
    if divisions < 2 or divisions % 2 != 0:
        raise ValueError("等分数必须是至少为2的偶数，才能以中位数对称分开多空。")
    if len(scores) < divisions:
        raise ValueError("样本量小于等分数。")

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_returns = returns[order]
    sorted_before_cutoff = before_cutoff[order]
    n = len(sorted_scores)
    half = divisions // 2

    # 每个内部边界取相邻两档中央样本之间的中点。
    split_positions = [int(n * index / divisions) for index in range(1, divisions)]
    internal_boundaries = [
        float((sorted_scores[pos - 1] + sorted_scores[pos]) / 2.0)
        for pos in split_positions
    ]
    boundaries = [float(sorted_scores[0]), *internal_boundaries, float(sorted_scores[-1])]

    median_boundary = boundaries[half]
    if not np.isclose(median_boundary, threshold, rtol=0.0, atol=1e-15):
        raise AssertionError("等分中央边界与全体中位数不一致。")

    bucket_ids = (np.arange(n) * divisions // n) + 1
    rows: list[dict] = []
    for bucket in range(1, divisions + 1):
        mask = bucket_ids == bucket
        bucket_scores = sorted_scores[mask]
        bucket_returns = sorted_returns[mask]
        bucket_before_cutoff = sorted_before_cutoff[mask]
        is_short = bucket <= half
        strategy_returns = -bucket_returns if is_short else bucket_returns
        overall = _summarize_strategy(strategy_returns)
        before = _summarize_strategy(strategy_returns[bucket_before_cutoff])
        after = _summarize_strategy(strategy_returns[~bucket_before_cutoff])

        lower = boundaries[bucket - 1]
        upper = boundaries[bucket]
        if is_short:
            interval = f"[{_boundary_text(lower)}, {_boundary_text(upper)})"
            direction = "做空"
        else:
            interval = f"({_boundary_text(lower)}, {_boundary_text(upper)}]"
            direction = "做多"

        rows.append(
            {
                "bucket": bucket,
                "bucket_label": f"Q{bucket:02d}",
                "interval": interval,
                "lower_boundary": lower,
                "upper_boundary": upper,
                "direction": direction,
                "sample_n": overall["sample_n"],
                "sample_share": float(mask.mean()),
                "average_strategy_return": overall["average_strategy_return"],
                "win_rate": overall["win_rate"],
                "average_profit_loss_ratio": overall["average_profit_loss_ratio"],
                "long_share": 0.0 if is_short else 1.0,
                "short_share": 1.0 if is_short else 0.0,
                "before_sample_n": before["sample_n"],
                "before_average_strategy_return": before["average_strategy_return"],
                "before_win_rate": before["win_rate"],
                "before_average_profit_loss_ratio": before["average_profit_loss_ratio"],
                "after_sample_n": after["sample_n"],
                "after_average_strategy_return": after["average_strategy_return"],
                "after_win_rate": after["win_rate"],
                "after_average_profit_loss_ratio": after["average_profit_loss_ratio"],
            }
        )

    counts = [row["sample_n"] for row in rows]
    if max(counts) - min(counts) > 1:
        raise AssertionError(f"{divisions}等分未满足等样本要求。")
    if sum(counts) != n:
        raise AssertionError(f"{divisions}等分样本合计不一致。")

    return {
        "divisions": divisions,
        "n": n,
        "threshold": threshold,
        "short_n": int(np.sum(scores < threshold)),
        "neutral_n": int(np.sum(scores == threshold)),
        "long_n": int(np.sum(scores > threshold)),
        "min_bucket_n": min(counts),
        "max_bucket_n": max(counts),
        "rows": rows,
    }


def run_analysis(
    source_path: Path = SOURCE_PATH,
    division_counts: list[int] = DIVISION_COUNTS,
) -> dict:
    """生成全体、14:30前、14:30后三个数据集各自的中位数分析。"""
    df = load_source(source_path)
    scores = np.clip(
        df[PRED_COLUMN].to_numpy().astype(float) + SCORE_OFFSET,
        SCORE_MIN,
        SCORE_MAX,
    )
    returns = df[RETURN_COLUMN].to_numpy().astype(float)
    cutoff_hour, cutoff_minute = [int(value) for value in TIME_CUTOFF.split(":")]
    cutoff_total_minutes = cutoff_hour * 60 + cutoff_minute
    minute_of_day = (
        df.select(
            (
                pl.col(DATETIME_COLUMN).dt.hour().cast(pl.Int32) * 60
                + pl.col(DATETIME_COLUMN).dt.minute().cast(pl.Int32)
            ).alias("minute_of_day")
        )["minute_of_day"]
        .to_numpy()
        .astype(int)
    )
    before_cutoff = minute_of_day < cutoff_total_minutes
    def build_dataset(
        key: str,
        label: str,
        dataset_scores: np.ndarray,
        dataset_returns: np.ndarray,
        dataset_before_cutoff: np.ndarray,
    ) -> dict:
        dataset_threshold = calculate_median_threshold(dataset_scores)
        return {
            "key": key,
            "label": label,
            "total_n": int(len(dataset_scores)),
            "threshold": dataset_threshold,
            "threshold_display": round(dataset_threshold, 6),
            "analyses": {
                str(divisions): calculate_equal_frequency(
                    dataset_scores,
                    dataset_returns,
                    dataset_before_cutoff,
                    divisions,
                    dataset_threshold,
                )
                for divisions in division_counts
            },
        }

    datasets = {
        "global": build_dataset(
            "global",
            "全体样本",
            scores,
            returns,
            before_cutoff,
        ),
        "before": build_dataset(
            "before",
            "14:30以前",
            scores[before_cutoff],
            returns[before_cutoff],
            np.ones(int(np.sum(before_cutoff)), dtype=bool),
        ),
        "after": build_dataset(
            "after",
            "14:30及以后",
            scores[~before_cutoff],
            returns[~before_cutoff],
            np.zeros(int(np.sum(~before_cutoff)), dtype=bool),
        ),
    }
    global_dataset = datasets["global"]

    return {
        "source": str(source_path),
        "total_n": int(len(scores)),
        "threshold_quantile": THRESHOLD_QUANTILE,
        "threshold": global_dataset["threshold"],
        "threshold_display": global_dataset["threshold_display"],
        "time_cutoff": TIME_CUTOFF,
        "before_cutoff_n": int(np.sum(before_cutoff)),
        "after_cutoff_n": int(np.sum(~before_cutoff)),
        "analyses": global_dataset["analyses"],
        "datasets": datasets,
    }


def analysis_table(
    analysis: dict,
    divisions: int,
    dataset_key: str = "global",
) -> pl.DataFrame:
    """返回适合 Jupyter 查看的一张等分明细表。"""
    rows = analysis["datasets"][dataset_key]["analyses"][str(divisions)]["rows"]
    return pl.DataFrame(rows).select(
        "bucket_label",
        "interval",
        "direction",
        "sample_n",
        "sample_share",
        "average_strategy_return",
        "win_rate",
        "average_profit_loss_ratio",
        "long_share",
        "short_share",
        "before_sample_n",
        "before_average_strategy_return",
        "before_win_rate",
        "before_average_profit_loss_ratio",
        "after_sample_n",
        "after_average_strategy_return",
        "after_win_rate",
        "after_average_profit_loss_ratio",
    )


def display_analysis(analysis: dict) -> None:
    """依次显示三个数据集各自的20等分和50等分表。"""
    try:
        from IPython.display import display
    except ImportError:
        display = print

    for dataset_key in ["global", "before", "after"]:
        dataset = analysis["datasets"][dataset_key]
        print(
            f"\n===== {dataset['label']}｜中位数阈值 "
            f"{dataset['threshold']:.12f} ====="
        )
        for divisions in DIVISION_COUNTS:
            print(f"\n--- {divisions}等分 ---")
            display(analysis_table(analysis, divisions, dataset_key))


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--json", action="store_true")
    args, _unknown = parser.parse_known_args()

    result = run_analysis(args.source)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    else:
        display_analysis(result)
