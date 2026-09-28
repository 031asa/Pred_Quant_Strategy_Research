from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "tmp" / "pdfs"
OUTPUT_DIR = BASE_DIR / "pdf"
OUTPUT_PDF = OUTPUT_DIR / "固定阈值累计收益与回撤归因报告.pdf"
SOURCE = "C:/Users/Hello/Downloads/pred_eval.parquet"
DUCKDB = "D:/DevTools/DuckDB/duckdb.exe"
THRESHOLD = 0.5
PDF_FONT = "MicrosoftYaHei"
PDF_FONT_PATH = "C:/Windows/Fonts/msyh.ttc"

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#4472C4")
LIGHT_BLUE = colors.HexColor("#D9EAF7")
LIGHT_GRAY = colors.HexColor("#F2F2F2")
GREEN = colors.HexColor("#548235")
LIGHT_GREEN = colors.HexColor("#E2F0D9")
RED = colors.HexColor("#C00000")
LIGHT_RED = colors.HexColor("#F4CCCC")
ORANGE = colors.HexColor("#C65911")


def duckdb_json(sql: str) -> list[dict]:
    completed = subprocess.run(
        [DUCKDB, "-json", "-c", sql],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


BASE_SQL = f"""
WITH base AS MATERIALIZED (
  SELECT
    CAST(Date AS DATE) AS d,
    CAST(Contract AS VARCHAR) AS contract,
    Datetime AT TIME ZONE 'Asia/Shanghai' AS local_dt,
    LEAST(1.0, GREATEST(0.0, CAST(pred AS DOUBLE) + 0.5)) AS score,
    CASE
      WHEN LEAST(1.0, GREATEST(0.0, CAST(pred AS DOUBLE) + 0.5)) > {THRESHOLD}
        THEN CAST(Tag AS DOUBLE)
      WHEN LEAST(1.0, GREATEST(0.0, CAST(pred AS DOUBLE) + 0.5)) < {THRESHOLD}
        THEN -CAST(Tag AS DOUBLE)
      ELSE NULL
    END AS sr
  FROM read_parquet('{SOURCE}')
)
"""

BUCKET_CASE = """
CASE
  WHEN strftime(local_dt, '%H:%M') BETWEEN '09:30' AND '09:59' THEN '09:30-09:59'
  WHEN strftime(local_dt, '%H:%M') BETWEEN '10:00' AND '10:29' THEN '10:00-10:29'
  WHEN strftime(local_dt, '%H:%M') BETWEEN '10:30' AND '10:59' THEN '10:30-10:59'
  WHEN strftime(local_dt, '%H:%M') BETWEEN '11:00' AND '11:30' THEN '11:00-11:30'
  WHEN strftime(local_dt, '%H:%M') BETWEEN '13:01' AND '13:29' THEN '13:01-13:29'
  WHEN strftime(local_dt, '%H:%M') BETWEEN '13:30' AND '13:59' THEN '13:30-13:59'
  WHEN strftime(local_dt, '%H:%M') BETWEEN '14:00' AND '14:29' THEN '14:00-14:29'
  WHEN strftime(local_dt, '%H:%M') BETWEEN '14:30' AND '14:59' THEN '14:30-14:59'
END
"""


def load_data() -> dict[str, list[dict]]:
    daily = duckdb_json(
        BASE_SQL
        + """
, daily AS (
  SELECT d, avg(sr) AS daily_return, count(sr) AS trade_n,
    count(*) FILTER (WHERE sr > 0)::DOUBLE
      / NULLIF(count(*) FILTER (WHERE sr <> 0), 0) AS win_rate,
    avg(sr) FILTER (WHERE sr > 0)
      / NULLIF(abs(avg(sr) FILTER (WHERE sr < 0)), 0) AS pl_ratio,
    count(*) FILTER (WHERE score > 0.5)::DOUBLE / count(*) AS long_share
  FROM base GROUP BY d
), wealth AS (
  SELECT *, exp(sum(ln(1 + daily_return)) OVER (ORDER BY d)) AS wealth
  FROM daily
), curve AS (
  SELECT *, wealth - 1 AS cumulative_return,
    wealth / greatest(1.0, max(wealth) OVER (ORDER BY d)) - 1 AS drawdown,
    arg_max(d, wealth) OVER (
      ORDER BY d ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS peak_date
  FROM wealth
)
SELECT strftime(d, '%Y-%m-%d') AS date, daily_return, trade_n, win_rate,
  pl_ratio, long_share, cumulative_return, drawdown,
  strftime(peak_date, '%Y-%m-%d') AS peak_date
FROM curve ORDER BY d;
"""
    )

    episodes = duckdb_json(
        BASE_SQL
        + """
, daily AS (SELECT d, avg(sr) AS r FROM base GROUP BY d),
wealth AS (
  SELECT d, exp(sum(ln(1 + r)) OVER (ORDER BY d)) AS wealth FROM daily
), curve AS (
  SELECT d, wealth,
    greatest(1.0, max(wealth) OVER (ORDER BY d)) AS peak_wealth,
    arg_max(d, wealth) OVER (
      ORDER BY d ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS peak_date
  FROM wealth
), dd AS (
  SELECT *, wealth / peak_wealth - 1 AS drawdown FROM curve
)
SELECT strftime(peak_date, '%Y-%m-%d') AS peak_date,
  strftime(arg_min(d, drawdown), '%Y-%m-%d') AS trough_date,
  min(drawdown) AS max_drawdown,
  max(d) = (SELECT max(d) FROM dd) AS unrecovered_at_end
FROM dd WHERE drawdown < 0
GROUP BY peak_date
HAVING min(drawdown) < -0.001
ORDER BY max_drawdown ASC LIMIT 8;
"""
    )

    period_summary = duckdb_json(
        BASE_SQL
        + """
, periods(name, start_d, end_d) AS (VALUES
  ('全样本', DATE '2024-01-03', DATE '2026-06-30'),
  ('最大回撤', DATE '2026-06-02', DATE '2026-06-16'),
  ('长回撤', DATE '2025-05-07', DATE '2025-08-06'),
  ('尾部截至期末', DATE '2026-06-02', DATE '2026-06-30')
)
SELECT p.name, strftime(min(b.d), '%Y-%m-%d') AS actual_start,
  strftime(max(b.d), '%Y-%m-%d') AS actual_end, count(*) AS n,
  avg(sr) AS avg_return,
  count(*) FILTER (WHERE sr > 0)::DOUBLE
    / NULLIF(count(*) FILTER (WHERE sr <> 0), 0) AS win_rate,
  avg(sr) FILTER (WHERE sr > 0)
    / NULLIF(abs(avg(sr) FILTER (WHERE sr < 0)), 0) AS pl_ratio,
  count(*) FILTER (WHERE score > 0.5)::DOUBLE / count(*) AS long_share
FROM periods p JOIN base b ON b.d BETWEEN p.start_d AND p.end_d
GROUP BY p.name ORDER BY p.name;
"""
    )

    monthly = duckdb_json(
        BASE_SQL
        + """
SELECT strftime(d, '%Y-%m') AS month, count(sr) AS n, avg(sr) AS avg_return,
  count(*) FILTER (WHERE sr > 0)::DOUBLE
    / NULLIF(count(*) FILTER (WHERE sr <> 0), 0) AS win_rate,
  avg(sr) FILTER (WHERE sr > 0)
    / NULLIF(abs(avg(sr) FILTER (WHERE sr < 0)), 0) AS pl_ratio,
  count(*) FILTER (WHERE score > 0.5)::DOUBLE / count(*) AS long_share
FROM base WHERE d >= DATE '2025-07-01'
GROUP BY month ORDER BY month;
"""
    )

    contract_attribution = duckdb_json(
        BASE_SQL
        + """
, periods(name, start_d, end_d) AS (VALUES
  ('最大回撤', DATE '2026-06-02', DATE '2026-06-16'),
  ('长回撤', DATE '2025-05-07', DATE '2025-08-06'),
  ('尾部截至期末', DATE '2026-06-02', DATE '2026-06-30')
), daily_total AS (
  SELECT p.name, b.d, count(sr) AS total_n
  FROM periods p JOIN base b ON b.d BETWEEN p.start_d AND p.end_d
  GROUP BY p.name, b.d
)
SELECT p.name, b.contract,
  sum(coalesce(b.sr, 0) / dt.total_n) AS contribution,
  count(b.sr) AS n, avg(b.sr) AS avg_return,
  count(*) FILTER (WHERE b.sr > 0)::DOUBLE
    / NULLIF(count(*) FILTER (WHERE b.sr <> 0), 0) AS win_rate,
  avg(b.sr) FILTER (WHERE b.sr > 0)
    / NULLIF(abs(avg(b.sr) FILTER (WHERE b.sr < 0)), 0) AS pl_ratio,
  count(*) FILTER (WHERE b.score > 0.5)::DOUBLE / count(*) AS long_share
FROM periods p JOIN base b ON b.d BETWEEN p.start_d AND p.end_d
JOIN daily_total dt ON dt.name = p.name AND dt.d = b.d
GROUP BY p.name, b.contract
ORDER BY p.name, contribution ASC;
"""
    )

    worst_days = duckdb_json(
        BASE_SQL
        + """
SELECT strftime(d, '%Y-%m-%d') AS date, count(sr) AS n,
  avg(sr) AS avg_return,
  count(*) FILTER (WHERE sr > 0)::DOUBLE
    / NULLIF(count(*) FILTER (WHERE sr <> 0), 0) AS win_rate,
  avg(sr) FILTER (WHERE sr > 0)
    / NULLIF(abs(avg(sr) FILTER (WHERE sr < 0)), 0) AS pl_ratio,
  count(*) FILTER (WHERE score > 0.5)::DOUBLE / count(*) AS long_share
FROM base GROUP BY d ORDER BY avg_return ASC LIMIT 12;
"""
    )

    worst_pairs = duckdb_json(
        BASE_SQL
        + """
SELECT strftime(d, '%Y-%m-%d') AS date, contract, count(sr) AS n,
  avg(sr) AS avg_return,
  count(*) FILTER (WHERE sr > 0)::DOUBLE
    / NULLIF(count(*) FILTER (WHERE sr <> 0), 0) AS win_rate,
  avg(sr) FILTER (WHERE sr > 0)
    / NULLIF(abs(avg(sr) FILTER (WHERE sr < 0)), 0) AS pl_ratio,
  count(*) FILTER (WHERE score > 0.5)::DOUBLE / count(*) AS long_share
FROM base GROUP BY d, contract
HAVING count(sr) > 0
ORDER BY avg_return ASC LIMIT 15;
"""
    )

    tail_pairs = duckdb_json(
        BASE_SQL
        + """
SELECT strftime(d, '%Y-%m-%d') AS date, contract, count(sr) AS n,
  avg(sr) AS avg_return,
  count(*) FILTER (WHERE sr > 0)::DOUBLE
    / NULLIF(count(*) FILTER (WHERE sr <> 0), 0) AS win_rate,
  avg(sr) FILTER (WHERE sr > 0)
    / NULLIF(abs(avg(sr) FILTER (WHERE sr < 0)), 0) AS pl_ratio,
  count(*) FILTER (WHERE score > 0.5)::DOUBLE / count(*) AS long_share
FROM base
WHERE d BETWEEN DATE '2026-06-02' AND DATE '2026-06-16'
GROUP BY d, contract ORDER BY avg_return ASC LIMIT 15;
"""
    )

    return {
        "daily": daily,
        "episodes": episodes,
        "period_summary": period_summary,
        "monthly": monthly,
        "contract_attribution": contract_attribution,
        "worst_days": worst_days,
        "worst_pairs": worst_pairs,
        "tail_pairs": tail_pairs,
    }


def configure_charts() -> None:
    pdfmetrics.registerFont(
        TTFont(PDF_FONT, PDF_FONT_PATH, subfontIndex=0)
    )


def pct(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def mult(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def _label(drawing: Drawing, x: float, y: float, text: str, size=7, color="#555555", anchor="start") -> None:
    drawing.add(String(x, y, text, fontName=PDF_FONT, fontSize=size, fillColor=colors.HexColor(color), textAnchor=anchor))


def _scale(value: float, low: float, high: float, start: float, length: float) -> float:
    return start + (value - low) / (high - low) * length if high > low else start + length / 2


def save_curve_chart(data: dict[str, list[dict]]) -> Drawing:
    rows = data["daily"]
    values = [row["cumulative_return"] for row in rows]
    width, height = 500, 200
    x0, y0, plot_w, plot_h = 42, 25, 448, 145
    low = min(0.0, min(values))
    high = max(values)
    pad = max((high - low) * 0.08, 0.005)
    low -= pad
    high += pad
    d = Drawing(width, height)
    _label(d, width / 2, 184, "固定阈值 0.50 的累计收益率与最大回撤", 11, "#17365D", "middle")
    for i in range(5):
        value = low + (high - low) * i / 4
        y = _scale(value, low, high, y0, plot_h)
        d.add(Line(x0, y, x0 + plot_w, y, strokeColor=colors.HexColor("#D9E1F2"), strokeWidth=0.5))
        _label(d, x0 - 5, y - 2, f"{value:.0%}", 6.5, anchor="end")
    tick_indices = [round(i * (len(rows) - 1) / 6) for i in range(7)]
    for idx in tick_indices:
        x = x0 + idx / (len(rows) - 1) * plot_w
        d.add(Line(x, y0, x, y0 - 3, strokeColor=colors.HexColor("#A6A6A6"), strokeWidth=0.5))
        _label(d, x, 9, rows[idx]["date"][:7], 6.2, anchor="middle")
    start_idx = next(i for i, row in enumerate(rows) if row["date"] == "2026-06-02")
    trough_idx = next(i for i, row in enumerate(rows) if row["date"] == "2026-06-16")
    shade_x1 = x0 + start_idx / (len(rows) - 1) * plot_w
    shade_x2 = x0 + trough_idx / (len(rows) - 1) * plot_w
    d.add(Rect(shade_x1, y0, max(3, shade_x2 - shade_x1), plot_h, fillColor=colors.HexColor("#F4CCCC"), strokeColor=None, fillOpacity=0.65))
    points = [
        (x0 + i / (len(rows) - 1) * plot_w, _scale(value, low, high, y0, plot_h))
        for i, value in enumerate(values)
    ]
    d.add(PolyLine(points, strokeColor=BLUE, strokeWidth=1.35, fillColor=None))
    for idx, color_value in [(start_idx, GREEN), (trough_idx, RED)]:
        d.add(Circle(points[idx][0], points[idx][1], 2.4, fillColor=color_value, strokeColor=None))
    _label(d, shade_x1 - 4, points[start_idx][1] + 8, "前高10.67%", 6.5, "#548235", "end")
    _label(d, shade_x2 + 4, points[trough_idx][1] - 10, "低点9.81%", 6.5, "#C00000")
    d.add(Line(x0, y0, x0 + plot_w, y0, strokeColor=colors.HexColor("#808080"), strokeWidth=0.6))
    return d


def save_monthly_chart(data: dict[str, list[dict]]) -> Drawing:
    rows = data["monthly"]
    width, height = 500, 260
    d = Drawing(width, height)
    _label(d, width / 2, 246, "近12个月：收益、胜率和平均盈亏比", 11, "#17365D", "middle")
    panels = [
        ("平均收益", [row["avg_return"] for row in rows], -0.0002, 0.0005, BLUE, 169),
        ("胜率", [row["win_rate"] for row in rows], 0.46, 0.54, BLUE, 96),
        ("盈亏比", [row["pl_ratio"] for row in rows], 0.90, 1.16, colors.HexColor("#7030A0"), 23),
    ]
    x0, panel_w, panel_h = 46, 440, 52
    for title, values, low, high, line_color, y0 in panels:
        baseline = _scale(0 if title == "平均收益" else (0.5 if title == "胜率" else 1.0), low, high, y0, panel_h)
        d.add(Line(x0, baseline, x0 + panel_w, baseline, strokeColor=ORANGE, strokeWidth=0.7, strokeDashArray=[3, 2]))
        d.add(Line(x0, y0, x0, y0 + panel_h, strokeColor=colors.HexColor("#A6A6A6"), strokeWidth=0.5))
        _label(d, 4, y0 + panel_h / 2, title, 7, "#17365D")
        points = []
        for i, value in enumerate(values):
            x = x0 + i / (len(values) - 1) * panel_w
            y = _scale(value, low, high, y0, panel_h)
            if title == "平均收益":
                d.add(Rect(x - 7, min(y, baseline), 14, abs(y - baseline), fillColor=GREEN if value >= 0 else RED, strokeColor=None, fillOpacity=0.8))
            else:
                points.append((x, y))
                d.add(Circle(x, y, 1.7, fillColor=line_color, strokeColor=None))
        if points:
            d.add(PolyLine(points, strokeColor=line_color, strokeWidth=1.2, fillColor=None))
        _label(d, x0 - 4, y0 - 1, f"{low:.2%}" if title != "盈亏比" else f"{low:.2f}x", 5.8, anchor="end")
        _label(d, x0 - 4, y0 + panel_h - 2, f"{high:.2%}" if title != "盈亏比" else f"{high:.2f}x", 5.8, anchor="end")
    for i, row in enumerate(rows):
        x = x0 + i / (len(rows) - 1) * panel_w
        _label(d, x, 7, row["month"][2:], 5.2, anchor="middle")
    return d


def save_contract_chart(data: dict[str, list[dict]]) -> Drawing:
    rows = data["contract_attribution"]
    width, height = 500, 220
    d = Drawing(width, height)
    _label(d, width / 2, 206, "按合约拆解负收益贡献", 11, "#17365D", "middle")
    for panel_x, name, title in [
        (10, "最大回撤", "2026-06-02至06-16"),
        (258, "尾部截至期末", "2026-06-02至06-30"),
    ]:
        selected = sorted([row for row in rows if row["name"] == name], key=lambda row: row["contribution"])
        values = [row["contribution"] for row in selected]
        low, high = min(values + [0]), max(values + [0])
        plot_x, plot_w, y0, row_h = panel_x + 40, 195, 22, 13.5
        zero_x = _scale(0, low, high, plot_x, plot_w)
        d.add(Line(zero_x, y0, zero_x, y0 + row_h * len(selected), strokeColor=colors.HexColor("#808080"), strokeWidth=0.6))
        _label(d, panel_x + 120, 188, title, 8, "#17365D", "middle")
        for i, row in enumerate(selected):
            y = y0 + i * row_h
            end_x = _scale(row["contribution"], low, high, plot_x, plot_w)
            d.add(Rect(min(zero_x, end_x), y + 2, abs(end_x - zero_x), 8, fillColor=RED if row["contribution"] < 0 else GREEN, strokeColor=None, fillOpacity=0.82))
            _label(d, panel_x + 2, y + 3, row["contract"], 6.1)
        _label(d, plot_x, 8, f"{low:.2%}", 5.8, anchor="middle")
        _label(d, plot_x + plot_w, 8, f"{high:.2%}", 5.8, anchor="middle")
    return d


def save_worst_days_chart(data: dict[str, list[dict]]) -> Drawing:
    rows = list(reversed(data["worst_days"][:8]))
    width, height = 500, 140
    d = Drawing(width, height)
    _label(d, width / 2, 128, "全样本最差的8个交易日", 11, "#17365D", "middle")
    low = min(row["avg_return"] for row in rows)
    x0, plot_w, y0, row_h = 80, 400, 12, 13
    zero_x = x0 + plot_w
    for i, row in enumerate(rows):
        y = y0 + i * row_h
        end_x = _scale(row["avg_return"], low, 0, x0, plot_w)
        d.add(Rect(end_x, y + 2, zero_x - end_x, 8, fillColor=RED, strokeColor=None, fillOpacity=0.82))
        _label(d, 4, y + 3, row["date"], 6.2)
        _label(d, end_x - 3, y + 3, f"{row['avg_return']:.3%}", 5.8, "#C00000", "end")
    d.add(Line(zero_x, y0, zero_x, y0 + row_h * len(rows), strokeColor=colors.HexColor("#808080"), strokeWidth=0.6))
    return d


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN", parent=base["Title"], fontName=PDF_FONT,
            fontSize=20, leading=26, textColor=NAVY, alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN", parent=base["Normal"], fontName=PDF_FONT,
            fontSize=9, leading=14, textColor=colors.HexColor("#666666"),
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "H1CN", parent=base["Heading1"], fontName=PDF_FONT,
            fontSize=14, leading=19, textColor=NAVY, spaceBefore=5, spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "H2CN", parent=base["Heading2"], fontName=PDF_FONT,
            fontSize=11, leading=16, textColor=BLUE, spaceBefore=4, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyCN", parent=base["BodyText"], fontName=PDF_FONT,
            fontSize=9.2, leading=15, textColor=colors.HexColor("#222222"),
            alignment=TA_LEFT, spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "SmallCN", parent=base["BodyText"], fontName=PDF_FONT,
            fontSize=7.6, leading=11, textColor=colors.HexColor("#555555"),
        ),
        "callout": ParagraphStyle(
            "CalloutCN", parent=base["BodyText"], fontName=PDF_FONT,
            fontSize=10, leading=16, textColor=NAVY, leftIndent=7, rightIndent=7,
            borderColor=BLUE, borderWidth=0.7, borderPadding=7,
            backColor=LIGHT_BLUE, spaceAfter=8,
        ),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def make_table(rows: list[list], widths: list[float], header=True, font_size=7.5) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), PDF_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E1F2")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ]
    table.setStyle(TableStyle(commands))
    return table


def kpi_table(summary: dict[str, dict]) -> Table:
    overall = summary["全样本"]
    max_dd = summary["最大回撤"]
    cells = [
        ["期末累计收益", "最大回撤", "全样本胜率", "回撤期胜率", "回撤期盈亏比"],
        ["10.27%", "-0.78%", pct(overall["win_rate"]), pct(max_dd["win_rate"]), mult(max_dd["pl_ratio"])],
    ]
    table = Table(cells, colWidths=[35 * mm] * 5, rowHeights=[8 * mm, 12 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), PDF_FONT),
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, 1), LIGHT_BLUE),
                ("TEXTCOLOR", (0, 1), (0, 1), GREEN),
                ("TEXTCOLOR", (1, 1), (1, 1), RED),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("FONTSIZE", (0, 1), (-1, 1), 12),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9E1F2")),
            ]
        )
    )
    return table


def page_header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D9E1F2"))
    canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
    canvas.setFont(PDF_FONT, 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(18 * mm, 9 * mm, "固定阈值累计收益与回撤归因报告")
    canvas.drawRightString(width - 18 * mm, 9 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build_pdf(data: dict[str, list[dict]], charts: dict[str, Drawing]) -> None:
    styles = build_styles()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=16 * mm,
        title="固定阈值累计收益与回撤归因报告",
        author="Codex",
    )
    summary = {row["name"]: row for row in data["period_summary"]}
    flow = []

    flow += [
        p("固定阈值累计收益与回撤归因报告", styles["title"]),
        p(
            "数据：pred_eval.parquet｜固定阈值：0.50｜高于阈值做多，低于阈值做空｜"
            "日内等权，跨日复利｜分析截止：2026-06-30",
            styles["subtitle"],
        ),
        kpi_table(summary),
        Spacer(1, 5 * mm),
        charts["curve"],
        Spacer(1, 3 * mm),
        p("核心判断", styles["h1"]),
        p(
            "最大回撤发生在 <b>2026-06-02 至 2026-06-16</b>。累计收益从 10.67% 降到 9.81%，"
            "回撤为 -0.78%；截至 2026-06-30 仅修复到 10.27%，仍较前高低约 0.36%。",
            styles["callout"],
        ),
        p(
            "回撤期平均策略收益降至 <b>-0.0669%</b>，胜率从全样本的 51.66% 降至 "
            "<b>43.82%</b>，平均盈亏比从 1.04x 降至 <b>0.87x</b>。因此下跌不是只由输单变多，"
            "而是“命中率下降 + 赚钱单相对亏钱单变小”同时发生。",
            styles["body"],
        ),
        p(
            "说明：收益曲线尚未扣除手续费、滑点和资金容量影响；本报告能定位结果层面的时间和合约贡献，"
            "但仅凭 pred、Tag、合约和日期，不能证明宏观或行情层面的因果原因。",
            styles["small"],
        ),
        PageBreak(),
    ]

    episode_rows = [["前高日期", "低点日期", "最大回撤", "期末未修复"]]
    for row in data["episodes"][:6]:
        episode_rows.append([
            row["peak_date"], row["trough_date"], pct(row["max_drawdown"]),
            "是" if row["unrecovered_at_end"] else "否",
        ])
    flow += [
        p("1. 哪些时间段影响最大", styles["h1"]),
        make_table(episode_rows, [42 * mm, 42 * mm, 40 * mm, 40 * mm], font_size=8),
        Spacer(1, 4 * mm),
        charts["monthly"],
        Spacer(1, 2 * mm),
        p("尾部指标为何突然变差", styles["h2"]),
        p(
            "2026年5月仍是正收益月份：平均策略收益 0.0291%、胜率 52.67%、盈亏比 1.07x。"
            "进入2026年6月后，三项指标同时恶化为 -0.0134%、48.92% 和 0.97x。"
            "这说明尾部下滑是模型在该阶段失去方向优势，而不是单纯交易数量变化。",
            styles["body"],
        ),
        p(
            "历史上还存在一段更长的低迷期：2025-05-07 至 2025-08-06，最大回撤 -0.67%。"
            "该阶段胜率 48.45%、盈亏比 0.97x，表现为持续的小幅负优势；2026年6月则更集中、更突然。",
            styles["body"],
        ),
        PageBreak(),
    ]

    max_rows = [row for row in data["contract_attribution"] if row["name"] == "最大回撤"]
    tail_rows = [row for row in data["contract_attribution"] if row["name"] == "尾部截至期末"]
    max_table = [["合约", "收益贡献", "平均收益", "胜率", "盈亏比", "多单占比"]]
    for row in max_rows[:7]:
        max_table.append([
            row["contract"], pct(row["contribution"], 3), pct(row["avg_return"], 4),
            pct(row["win_rate"]), mult(row["pl_ratio"]), pct(row["long_share"]),
        ])
    tail_table = [["合约", "收益贡献", "平均收益", "胜率", "盈亏比", "多单占比"]]
    for row in tail_rows[:7]:
        tail_table.append([
            row["contract"], pct(row["contribution"], 3), pct(row["avg_return"], 4),
            pct(row["win_rate"]), mult(row["pl_ratio"]), pct(row["long_share"]),
        ])
    flow += [
        p("2. 哪些合约拖累最大", styles["h1"]),
        charts["contracts"],
        Spacer(1, 2 * mm),
        p("最大回撤区间：负贡献最大的合约", styles["h2"]),
        make_table(max_table, [23 * mm, 30 * mm, 30 * mm, 25 * mm, 25 * mm, 30 * mm], font_size=7.4),
        Spacer(1, 3 * mm),
        p("截至期末的尾部区间：持续拖累最大的合约", styles["h2"]),
        make_table(tail_table, [23 * mm, 30 * mm, 30 * mm, 25 * mm, 25 * mm, 30 * mm], font_size=7.4),
        Spacer(1, 3 * mm),
        p(
            "最大回撤的前半段主要由 <b>IM01、IC01、IC02、IM02、IC03</b> 共同拖累；"
            "这几组合约在 6月5日集中出现低胜率和低盈亏比。延长到6月底后，"
            "<b>IH02、IH03、IH01</b> 的持续负贡献变得更突出。也就是说，初始急跌由 IC/IM 共振触发，"
            "后续恢复不充分则更多受到 IH 系列拖累。",
            styles["callout"],
        ),
        p(
            "2025年的长回撤则主要集中在 IF/IH 系列，尤其是 IH03、IF03、IF01、IH01、IF02。"
            "不同回撤期由不同合约族主导，提示应做合约族级别的风险预算，而不能只设置一个全局阈值。",
            styles["body"],
        ),
        PageBreak(),
    ]

    worst_day_rows = [["日期", "日均收益", "胜率", "盈亏比", "多单占比"]]
    for row in data["worst_days"][:8]:
        worst_day_rows.append([
            row["date"], pct(row["avg_return"], 4), pct(row["win_rate"]),
            mult(row["pl_ratio"]), pct(row["long_share"]),
        ])
    tail_pair_rows = [["日期", "合约", "平均收益", "胜率", "盈亏比", "多单占比"]]
    for row in data["tail_pairs"][:8]:
        tail_pair_rows.append([
            row["date"], row["contract"], pct(row["avg_return"], 4),
            pct(row["win_rate"]), mult(row["pl_ratio"]), pct(row["long_share"]),
        ])
    flow += [
        p("3. 影响最大的具体样本", styles["h1"]),
        charts["worst_days"],
        Spacer(1, 2 * mm),
        make_table(worst_day_rows, [38 * mm, 34 * mm, 30 * mm, 30 * mm, 32 * mm], font_size=7.4),
        Spacer(1, 3 * mm),
        p("最大回撤区间内最差的“日期 × 合约”", styles["h2"]),
        make_table(tail_pair_rows, [31 * mm, 22 * mm, 30 * mm, 27 * mm, 27 * mm, 28 * mm], font_size=7.2),
        Spacer(1, 3 * mm),
        p(
            "<b>2026-06-05</b> 是本次回撤的核心冲击日：全体日均收益 -0.2650%，胜率 34.74%，"
            "盈亏比 0.58x；IM01、IC03、IC02、IM02、IC01 当日均明显为负。"
            "<b>2026-06-15</b> 是第二次集中冲击：胜率进一步降至 27.80%，盈亏比 0.57x，"
            "IC03、IC01、IF03、IF01 等表现最差。",
            styles["callout"],
        ),
        KeepTogether([
            p("建议的下一步验证", styles["h1"]),
            p(
                "1. 增加滚动20日监控：同时观察胜率、平均盈亏比和累计回撤。若胜率低于50%且盈亏比低于1，"
                "说明方向和赔率同时失效，应降低仓位或暂停。<br/>"
                "2. 对 IC、IF、IH、IM 分别设置合约族风险上限，防止多个近似合约在同一天产生共振损失。<br/>"
                "3. 对 2026-06-05、2026-06-15 进一步回看分钟级 Datetime，检查损失是否集中在开盘、午后或单一行情段。<br/>"
                "4. 加入手续费和滑点后重画曲线，并采用走样本外或滚动验证，确认10.27%的累计收益不是样本内结果。",
                styles["body"],
            ),
        ]),
    ]

    doc.build(flow, onFirstPage=page_header_footer, onLaterPages=page_header_footer)
    reader = PdfReader(str(OUTPUT_PDF))
    if len(reader.pages) != 4:
        raise RuntimeError(f"PDF页数异常：{len(reader.pages)}")
    embedded_font_objects = 0
    for page in reader.pages:
        resources = page["/Resources"].get_object()
        font_map = resources.get("/Font")
        if not font_map:
            continue
        for font_ref in font_map.get_object().values():
            font = font_ref.get_object()
            candidates = [font]
            descendants = font.get("/DescendantFonts")
            if descendants:
                candidates.extend(item.get_object() for item in descendants)
            for candidate in candidates:
                descriptor_ref = candidate.get("/FontDescriptor")
                if not descriptor_ref:
                    continue
                descriptor = descriptor_ref.get_object()
                if any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")):
                    embedded_font_objects += 1
    if embedded_font_objects == 0:
        raise RuntimeError("PDF字体未嵌入。")
    print(f"FONT_AUDIT=OK embedded_font_objects={embedded_font_objects}")


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    configure_charts()
    data = load_data()
    charts = {
        "curve": save_curve_chart(data),
        "monthly": save_monthly_chart(data),
        "contracts": save_contract_chart(data),
        "worst_days": save_worst_days_chart(data),
    }
    build_pdf(data, charts)
    print(f"OUTPUT={OUTPUT_PDF}")


if __name__ == "__main__":
    main()
