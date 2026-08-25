from __future__ import annotations

"""将现有研究图封装为Streamlit可直接显示和下载的单组PNG。"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import polars as pl

from utils.adjusted_pred_analysis import (
    _build_zero_threshold_cumulative,
    _draw_zero_threshold_cumulative,
    build_paired_decile_cumulative_returns,
    build_zero_threshold_metric_rows,
    draw_paired_decile_cumulative_chart,
    draw_pred_distribution_chart,
    draw_zero_threshold_metric_chart,
)
from utils.experiment_pipeline import (
    INTERNAL_DATE,
    INTERNAL_DATETIME,
    INTERNAL_PRED,
    INTERNAL_TAG,
    PipelineResult,
)
from utils.plot_fonts import register_matplotlib_chinese_font


CHART_METRICS = "metrics"
CHART_CUMULATIVE = "cumulative"
CHART_PAIRED = "paired"
CHART_DISTRIBUTION = "distribution"
CHART_TYPES = (
    CHART_METRICS,
    CHART_CUMULATIVE,
    CHART_PAIRED,
    CHART_DISTRIBUTION,
)

SCOPE_OVERALL = "overall"
SCOPE_BEFORE = "before"
SCOPE_AFTER = "after"
CHART_SCOPES = (SCOPE_OVERALL, SCOPE_BEFORE, SCOPE_AFTER)

CHART_LABELS = {
    CHART_METRICS: "六指标图",
    CHART_CUMULATIVE: "零阈值单利累计收益",
    CHART_PAIRED: "对称配对单利累计收益",
    CHART_DISTRIBUTION: "Pred分布图",
}


@dataclass(frozen=True)
class RenderedChart:
    png: bytes
    chart_type: str
    scope: str
    cutoff: str
    rows: int
    trading_days: int


def _validate_cutoff(cutoff: str) -> str:
    try:
        return datetime.strptime(cutoff, "%H:%M").strftime("%H:%M")
    except ValueError as exc:
        raise ValueError(f"出图切点{cutoff!r}不是有效的HH:MM。") from exc


def prepare_chart_data(
    result: PipelineResult,
    *,
    scope: str,
    cutoff: str,
) -> tuple[pl.DataFrame, str]:
    if scope not in CHART_SCOPES:
        raise ValueError(f"未知出图范围：{scope!r}。")
    cutoff = _validate_cutoff(cutoff)
    prepared = result.data.select(
        pl.col(INTERNAL_DATE).alias("Date"),
        pl.col(INTERNAL_DATETIME).alias("Datetime"),
        pl.col(INTERNAL_PRED).alias("pred_value"),
        pl.col(INTERNAL_TAG).alias("tag"),
    )
    if scope == SCOPE_OVERALL:
        scope_label = result.output_scope_label
    elif scope == SCOPE_BEFORE:
        prepared = prepared.filter(pl.col("Datetime").dt.strftime("%H:%M") < cutoff)
        scope_label = f"{cutoff}前"
    else:
        prepared = prepared.filter(pl.col("Datetime").dt.strftime("%H:%M") >= cutoff)
        scope_label = f"{cutoff}及以后"
    if prepared.height == 0:
        raise ValueError(f"{scope_label}没有可用于出图的样本。")
    return prepared, scope_label


def render_experiment_chart(
    result: PipelineResult,
    *,
    experiment_name: str,
    chart_type: str,
    scope: str,
    cutoff: str,
) -> RenderedChart:
    """复用现有计算和绘图函数，返回内存PNG。"""

    if chart_type not in CHART_TYPES:
        raise ValueError(f"未知图表类型：{chart_type!r}。")
    register_matplotlib_chinese_font()
    data, scope_label = prepare_chart_data(result, scope=scope, cutoff=cutoff)
    minimum_rows = 20 if chart_type == CHART_METRICS else 10
    if data.height < minimum_rows:
        raise ValueError(
            f"{scope_label}仅有{data.height}条样本，{CHART_LABELS[chart_type]}至少需要"
            f"{minimum_rows}条。"
        )

    title_prefix = f"实验{experiment_name}"
    with tempfile.TemporaryDirectory(prefix="pred_web_chart_") as temporary_dir:
        output_path = Path(temporary_dir) / f"{chart_type}.png"
        if chart_type == CHART_METRICS:
            metrics, _ = build_zero_threshold_metric_rows(
                data, variant=title_prefix, scope=scope_label
            )
            draw_zero_threshold_metric_chart(
                metrics,
                title=f"{title_prefix} Pred · {scope_label} · 20等分零阈值六指标",
                output_path=output_path,
            )
        elif chart_type == CHART_CUMULATIVE:
            cumulative, summary = _build_zero_threshold_cumulative(
                data, scope=scope_label
            )
            _draw_zero_threshold_cumulative(
                cumulative,
                summary,
                title=f"{title_prefix} Pred · {scope_label} · 10等分零阈值单利累计收益",
                output_path=output_path,
            )
        elif chart_type == CHART_PAIRED:
            cumulative, summary = build_paired_decile_cumulative_returns(
                data,
                variant=title_prefix,
                scope=scope_label,
            )
            draw_paired_decile_cumulative_chart(
                cumulative,
                summary,
                variant=title_prefix,
                scope=scope_label,
                output_path=output_path,
            )
        else:
            draw_pred_distribution_chart(
                data,
                variant=title_prefix,
                scope=scope_label,
                output_path=output_path,
            )
        png = output_path.read_bytes()

    return RenderedChart(
        png=png,
        chart_type=chart_type,
        scope=scope,
        cutoff=_validate_cutoff(cutoff),
        rows=data.height,
        trading_days=int(data["Date"].n_unique()),
    )
