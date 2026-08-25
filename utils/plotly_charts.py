from __future__ import annotations

"""Streamlit Web 使用的 Plotly 图；数据口径与既有 Matplotlib 研究图一致。"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.adjusted_pred_analysis import (
    _build_zero_threshold_cumulative,
    _direction_color,
    _distribution_statistics,
    build_paired_decile_cumulative_returns,
    build_zero_threshold_metric_rows,
)
from utils.experiment_pipeline import PipelineResult
from utils.plot_fonts import PLOTLY_FONT_STACK
from utils.web_charts import (
    CHART_CUMULATIVE,
    CHART_DISTRIBUTION,
    CHART_LABELS,
    CHART_METRICS,
    CHART_PAIRED,
    CHART_TYPES,
    prepare_chart_data,
)


SHORT_COLOR = "#3498f0"
LONG_COLOR = "#f18436"
GRID_COLOR = "rgba(90, 105, 120, 0.16)"


@dataclass(frozen=True)
class PlotlyChart:
    figure: go.Figure
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


def _base_layout(figure: go.Figure, *, title: str, subtitle: str, height: int) -> None:
    figure.update_layout(
        template="plotly_white",
        title={
            "text": f"{title}<br><sup>{subtitle}</sup>",
            "x": 0.5,
            "xanchor": "center",
        },
        font={"family": PLOTLY_FONT_STACK, "color": "#263442", "size": 13},
        title_font={"family": PLOTLY_FONT_STACK, "size": 22},
        height=height,
        margin={"l": 70, "r": 35, "t": 120, "b": 70},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0.5, "xanchor": "center"},
    )
    figure.update_xaxes(gridcolor=GRID_COLOR, zeroline=False)
    figure.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)


def _metric_figure(data, *, experiment_name: str, scope_label: str) -> go.Figure:
    metrics, _ = build_zero_threshold_metric_rows(
        data, variant=f"实验{experiment_name}", scope=scope_label
    )
    panel_titles = (
        "平均策略收益",
        "每日开仓次数中位数",
        "平均每日收益",
        "胜率",
        "平均盈亏比",
        "期望R倍数",
    )
    figure = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        subplot_titles=panel_titles,
    )
    x = list(range(1, 21))
    labels = metrics["档位"].to_list()
    short_shares = metrics["做空占比"].to_numpy()
    colors = [_direction_color(float(value)) for value in short_shares]
    custom = np.column_stack(
        [
            metrics["Pred下界"].to_numpy(),
            metrics["Pred上界"].to_numpy(),
            metrics["样本量"].to_numpy(),
            short_shares,
        ]
    )
    bar_panels = (
        (1, "平均策略收益", ".3%"),
        (2, "每日开仓次数中位数", ".2f"),
        (3, "平均每日收益", ".3%"),
    )
    for row, column, value_format in bar_panels:
        values = metrics[column].to_list()
        figure.add_trace(
            go.Bar(
                x=x,
                y=values,
                marker_color=colors,
                customdata=custom,
                name=column,
                showlegend=False,
                hovertemplate=(
                    "档位=%{x}<br>数值=%{y:" + value_format + "}<br>"
                    "Pred=[%{customdata[0]:.6f}, %{customdata[1]:.6f}]<br>"
                    "样本量=%{customdata[2]:,.0f}<br>做空占比=%{customdata[3]:.2%}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )
    line_panels = (
        (4, "胜率_排除Tag零值", ".2%", 0.5),
        (5, "平均盈亏比", ".3f", 1.0),
        (6, "期望R倍数", ".3f", 0.0),
    )
    for row, column, value_format, baseline in line_panels:
        values = metrics[column].to_list()
        figure.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode="lines+markers",
                line={"color": "#8d98a6", "width": 1.5},
                marker={"color": colors, "size": 8},
                name=panel_titles[row - 1],
                showlegend=False,
                customdata=custom,
                hovertemplate=(
                    "档位=%{x}<br>数值=%{y:" + value_format + "}<br>"
                    "Pred=[%{customdata[0]:.6f}, %{customdata[1]:.6f}]<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )
        figure.add_hline(
            y=baseline,
            line={"color": "#6f7780", "width": 1, "dash": "dash"},
            row=row,
            col=1,
        )
    for row in (1, 3):
        figure.add_hline(
            y=0,
            line={"color": "#6f7780", "width": 1, "dash": "dash"},
            row=row,
            col=1,
        )

    short_rate = float(
        np.average(short_shares, weights=metrics["样本量"].to_numpy())
    )
    zero_axis_x = 0.5 + 20.0 * short_rate
    for row in range(1, 7):
        figure.add_vline(
            x=zero_axis_x,
            line={"color": "#59636e", "width": 1.2, "dash": "dot"},
            row=row,
            col=1,
        )
    for name, color in (("做空档", SHORT_COLOR), ("做多档", LONG_COLOR)):
        figure.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers", marker={"color": color, "size": 10}, name=name
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": "#59636e", "dash": "dot"},
            name=f"pred=0分界｜做空占比{short_rate:.2%}",
        ),
        row=1,
        col=1,
    )
    figure.update_yaxes(tickformat=".2%", row=1, col=1)
    figure.update_yaxes(title_text="次数/日", row=2, col=1)
    figure.update_yaxes(tickformat=".2%", row=3, col=1)
    figure.update_yaxes(tickformat=".1%", row=4, col=1)
    figure.update_xaxes(
        tickmode="array",
        tickvals=x,
        ticktext=labels,
        title_text="Pred由低到高的20个严格等样本档位（交易方向始终由0决定）",
        row=6,
        col=1,
    )
    _base_layout(
        figure,
        title=f"实验{experiment_name} Pred · {scope_label} · 20等分零阈值六指标",
        subtitle=(
            "Tag=0排除胜率、盈亏比和期望R计算｜平均每日收益=平均策略收益×每日开仓次数中位数；"
            "该项为典型频次下逐笔收益日内求和，不等同于累计曲线的日内资金均分账户收益"
        ),
        height=1550,
    )
    return figure


def _cumulative_figure(data, *, experiment_name: str, scope_label: str) -> go.Figure:
    cumulative, summary = _build_zero_threshold_cumulative(data, scope=scope_label)
    figure = go.Figure()
    for bin_index in range(10):
        label = f"Q{bin_index + 1:02d}"
        part = cumulative.filter(cumulative["档位"] == label).sort("Date")
        row = summary.filter(summary["档位"] == label).row(0, named=True)
        short_share = float(row["做空占比"])
        legend = f"{label}｜空{short_share:.1%}｜期末{row['期末累计收益率']:.2%}"
        hover = (
            f"<b>{label}</b><br>交易日=%{{x}}<br>单利累计收益=%{{y:.2%}}<br>"
            f"Pred=[{row['Pred下界']:.6f}, {row['Pred上界']:.6f}]<br>"
            f"做空占比={short_share:.2%}<br>样本量={row['样本量']:,}<extra></extra>"
        )
        figure.add_trace(
            go.Scatter(
                x=part["Date"].to_list(),
                y=part["累计收益率"].to_list(),
                mode="lines",
                line={"color": _direction_color(short_share), "width": 2},
                name=legend,
                hovertemplate=hover,
            )
        )
    figure.add_hline(y=0, line={"color": "#66717f", "width": 1.2})
    figure.update_yaxes(title_text="单利累计收益率", tickformat=".1%")
    figure.update_xaxes(title_text="交易日")
    _base_layout(
        figure,
        title=f"实验{experiment_name} Pred · {scope_label} · 10等分零阈值单利累计收益",
        subtitle="每条样本pred&lt;0做空、pred≥0做多｜每档日内等权后按交易日单利累计",
        height=760,
    )
    figure.update_layout(
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": -0.20,
            "yanchor": "top",
            "entrywidth": 145,
            "entrywidthmode": "pixels",
            "title": {"text": "档位｜做空占比｜期末单利累计收益"},
        },
        margin={"l": 75, "r": 35, "t": 115, "b": 220},
    )
    return figure


def _paired_figure(data, *, experiment_name: str, scope_label: str) -> go.Figure:
    cumulative, summary = build_paired_decile_cumulative_returns(
        data, variant=f"实验{experiment_name}", scope=scope_label
    )
    figure = go.Figure()
    colors = ["#174a7e", "#2f6fa8", "#4d91c5", "#7ab2d4", "#a6cfe3"]
    for index, row in enumerate(summary.iter_rows(named=True)):
        pair = str(row["配对"])
        part = cumulative.filter(cumulative["配对"] == pair).sort("Date")
        legend = f"{pair}｜期末{row['期末累计收益率']:.2%}"
        hover = (
            f"<b>{pair}</b><br>交易日=%{{x}}<br>组合单利累计收益=%{{y:.2%}}<br>"
            f"两腿样本量={row['低档样本量']:,}+{row['高档样本量']:,}<extra></extra>"
        )
        figure.add_trace(
            go.Scatter(
                x=part["Date"].to_list(),
                y=part["累计收益率"].to_list(),
                mode="lines",
                line={"color": colors[index], "width": 2.2},
                name=legend,
                hovertemplate=hover,
            )
        )
    figure.add_hline(y=0, line={"color": "#66717f", "width": 1.2})
    figure.update_yaxes(title_text="单利累计收益率", tickformat=".1%")
    figure.update_xaxes(title_text="交易日")
    _base_layout(
        figure,
        title=f"实验{experiment_name} Pred · {scope_label} · 10等分对称配对单利累计收益",
        subtitle="低档固定做空、高档固定做多｜两腿每日各占50%｜合并日收益后按交易日单利累计",
        height=760,
    )
    figure.update_layout(
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": -0.20,
            "yanchor": "top",
            "entrywidth": 190,
            "entrywidthmode": "pixels",
            "title": {"text": "对称配对｜期末单利累计收益"},
        },
        margin={"l": 75, "r": 35, "t": 115, "b": 165},
    )
    return figure


def _distribution_figure(data, *, experiment_name: str, scope_label: str) -> go.Figure:
    pred = data["pred_value"].to_numpy()
    stats = _distribution_statistics(pred)
    density, edges = np.histogram(pred, bins=100, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=centers,
            y=density,
            width=widths,
            marker={"color": "#4e91cf", "opacity": 0.72},
            name="Pred密度直方图",
            hovertemplate="Pred=%{x:.6f}<br>密度=%{y:.4f}<extra></extra>",
        )
    )
    x = np.linspace(float(edges[0]), float(edges[-1]), 600)
    if stats["std"] > 0:
        normal = np.exp(-0.5 * ((x - stats["mean"]) / stats["std"]) ** 2) / (
            stats["std"] * np.sqrt(2.0 * np.pi)
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=normal,
                mode="lines",
                line={"color": "#e58032", "width": 2, "dash": "dash"},
                name="同均值方差正态曲线",
                hovertemplate="Pred=%{x:.6f}<br>正态密度=%{y:.4f}<extra></extra>",
            )
        )
    figure.add_vline(x=0, line={"color": "#303942", "width": 1.3})
    figure.add_annotation(
        x=0.99,
        y=0.97,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        align="left",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#d5dce3",
        borderpad=8,
        text=(
            f"N={len(pred):,}<br>均值={stats['mean']:.6f}<br>标准差={stats['std']:.6f}<br>"
            f"偏度={stats['skewness']:.3f}<br>超额峰度={stats['excess_kurtosis']:.3f}<br>"
            f"做空占比={stats['short_rate']:.2%}"
        ),
    )
    figure.update_xaxes(title_text="Pred")
    figure.update_yaxes(title_text="密度")
    _base_layout(
        figure,
        title=f"实验{experiment_name} Pred · {scope_label} · 分布与正态曲线对照",
        subtitle="密度直方图｜橙色虚线为相同均值和标准差的正态分布｜黑色竖线为pred=0",
        height=720,
    )
    return figure


def build_experiment_plotly_chart(
    result: PipelineResult,
    *,
    experiment_name: str,
    chart_type: str,
    scope: str,
    cutoff: str,
) -> PlotlyChart:
    """建立可由浏览器交互显示并通过模式栏下载PNG的单组图。"""

    if chart_type not in CHART_TYPES:
        raise ValueError(f"未知图表类型：{chart_type!r}。")
    data, scope_label = prepare_chart_data(result, scope=scope, cutoff=cutoff)
    minimum_rows = 20 if chart_type == CHART_METRICS else 10
    if data.height < minimum_rows:
        raise ValueError(
            f"{scope_label}仅有{data.height}条样本，{CHART_LABELS[chart_type]}至少需要"
            f"{minimum_rows}条。"
        )
    if chart_type == CHART_METRICS:
        figure = _metric_figure(data, experiment_name=experiment_name, scope_label=scope_label)
    elif chart_type == CHART_CUMULATIVE:
        figure = _cumulative_figure(data, experiment_name=experiment_name, scope_label=scope_label)
    elif chart_type == CHART_PAIRED:
        figure = _paired_figure(data, experiment_name=experiment_name, scope_label=scope_label)
    else:
        figure = _distribution_figure(data, experiment_name=experiment_name, scope_label=scope_label)
    return PlotlyChart(
        figure=figure,
        chart_type=chart_type,
        scope=scope,
        cutoff=_validate_cutoff(cutoff),
        rows=data.height,
        trading_days=int(data["Date"].n_unique()),
    )
