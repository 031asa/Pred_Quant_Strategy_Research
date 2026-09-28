from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import build_drawdown_pdf as core


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "pdf"
OUTPUT_PDF = OUTPUT_DIR / "关键回撤日分钟级归因报告.pdf"


def load_intraday_data() -> dict[str, list[dict]]:
    bucket_metrics = core.duckdb_json(
        core.BASE_SQL
        + f"""
, bucketed AS (
  SELECT *, {core.BUCKET_CASE} AS bucket FROM base
), groups(label, start_d, end_d) AS (VALUES
  ('全样本', DATE '2024-01-03', DATE '2026-06-30'),
  ('2026-06-05', DATE '2026-06-05', DATE '2026-06-05'),
  ('2026-06-15', DATE '2026-06-15', DATE '2026-06-15')
), metrics AS (
  SELECT g.label, b.bucket, count(sr) AS n, avg(sr) AS avg_return,
    count(*) FILTER (WHERE sr > 0)::DOUBLE
      / NULLIF(count(*) FILTER (WHERE sr <> 0), 0) AS win_rate,
    avg(sr) FILTER (WHERE sr > 0)
      / NULLIF(abs(avg(sr) FILTER (WHERE sr < 0)), 0) AS pl_ratio,
    count(*) FILTER (WHERE score > 0.5)::DOUBLE / count(*) AS long_share,
    CASE WHEN g.label <> '全样本' THEN sum(sr) / 2880.0 ELSE NULL END
      AS daily_contribution
  FROM groups g JOIN bucketed b ON b.d BETWEEN g.start_d AND g.end_d
  GROUP BY g.label, b.bucket
)
SELECT * FROM metrics ORDER BY label, bucket;
"""
    )

    minute_series = core.duckdb_json(
        core.BASE_SQL
        + """
, minute_metrics AS (
  SELECT d, strftime(local_dt, '%H:%M') AS minute_label,
    avg(sr) AS minute_return,
    count(*) FILTER (WHERE sr > 0)::DOUBLE
      / NULLIF(count(*) FILTER (WHERE sr <> 0), 0) AS win_rate,
    count(*) FILTER (WHERE sr < 0) AS loss_n,
    count(*) FILTER (WHERE score > 0.5)::DOUBLE / count(*) AS long_share
  FROM base
  WHERE d IN (DATE '2026-06-05', DATE '2026-06-15')
  GROUP BY d, minute_label
)
SELECT strftime(d, '%Y-%m-%d') AS date_label, minute_label,
  minute_return, win_rate, loss_n, long_share,
  sum(minute_return / 240.0) OVER (
    PARTITION BY d ORDER BY minute_label
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS cumulative_daily_contribution
FROM minute_metrics ORDER BY d, minute_label;
"""
    )

    worst_minutes = core.duckdb_json(
        core.BASE_SQL
        + """
SELECT strftime(d, '%Y-%m-%d') AS date_label,
  strftime(local_dt, '%H:%M') AS minute_label,
  avg(sr) AS avg_return,
  count(*) FILTER (WHERE sr > 0)::DOUBLE
    / NULLIF(count(*) FILTER (WHERE sr <> 0), 0) AS win_rate,
  count(*) FILTER (WHERE sr < 0) AS loss_n,
  count(*) FILTER (WHERE score > 0.5)::DOUBLE / count(*) AS long_share
FROM base
WHERE d IN (DATE '2026-06-05', DATE '2026-06-15')
GROUP BY d, minute_label
ORDER BY avg_return ASC LIMIT 14;
"""
    )

    contract_buckets = core.duckdb_json(
        core.BASE_SQL
        + f"""
, bucketed AS (
  SELECT *, {core.BUCKET_CASE} AS bucket FROM base
)
SELECT strftime(d, '%Y-%m-%d') AS date_label, bucket, contract,
  count(*) AS n, avg(sr) AS avg_return,
  sum(sr) / 2880.0 AS daily_contribution,
  count(*) FILTER (WHERE sr > 0)::DOUBLE
    / NULLIF(count(*) FILTER (WHERE sr <> 0), 0) AS win_rate,
  avg(sr) FILTER (WHERE sr > 0)
    / NULLIF(abs(avg(sr) FILTER (WHERE sr < 0)), 0) AS pl_ratio,
  count(*) FILTER (WHERE score > 0.5)::DOUBLE / count(*) AS long_share
FROM bucketed
WHERE d IN (DATE '2026-06-05', DATE '2026-06-15')
GROUP BY d, bucket, contract
ORDER BY date_label, daily_contribution ASC;
"""
    )
    return {
        "bucket_metrics": bucket_metrics,
        "minute_series": minute_series,
        "worst_minutes": worst_minutes,
        "contract_buckets": contract_buckets,
    }


def intraday_curve_chart(data: dict[str, list[dict]]) -> Drawing:
    width, height = 500, 215
    d = Drawing(width, height)
    core._label(d, width / 2, 200, "关键回撤日的分钟累计收益贡献", 11, "#17365D", "middle")
    x0, y0, plot_w, plot_h = 48, 30, 438, 145
    groups = {
        date: [row for row in data["minute_series"] if row["date_label"] == date]
        for date in ("2026-06-05", "2026-06-15")
    }
    values = [row["cumulative_daily_contribution"] for rows in groups.values() for row in rows]
    low, high = min(values) - 0.0001, max(0.0, max(values)) + 0.0001
    for i in range(5):
        value = low + (high - low) * i / 4
        y = core._scale(value, low, high, y0, plot_h)
        d.add(Line(x0, y, x0 + plot_w, y, strokeColor=colors.HexColor("#D9E1F2"), strokeWidth=0.5))
        core._label(d, x0 - 5, y - 2, f"{value:.2%}", 6, anchor="end")
    styles = [("2026-06-05", core.RED), ("2026-06-15", core.BLUE)]
    for date, color_value in styles:
        rows = groups[date]
        points = [
            (
                x0 + i / (len(rows) - 1) * plot_w,
                core._scale(row["cumulative_daily_contribution"], low, high, y0, plot_h),
            )
            for i, row in enumerate(rows)
        ]
        d.add(PolyLine(points, strokeColor=color_value, strokeWidth=1.4, fillColor=None))
        d.add(Circle(points[-1][0], points[-1][1], 2.3, fillColor=color_value, strokeColor=None))
        core._label(d, points[-1][0] - 2, points[-1][1] + (7 if date.endswith("05") else -11), f"{date[-5:]} {rows[-1]['cumulative_daily_contribution']:.3%}", 6.2, color_value.hexval(), "end")
    tick_indices = [0, 60, 120, 180, 239]
    labels = ["09:30", "10:30", "11:30", "13:59", "14:59"]
    for idx, label in zip(tick_indices, labels):
        x = x0 + idx / 239 * plot_w
        d.add(Line(x, y0, x, y0 - 3, strokeColor=colors.HexColor("#808080"), strokeWidth=0.5))
        core._label(d, x, 13, label, 6.2, anchor="middle")
    d.add(Line(x0, y0, x0 + plot_w, y0, strokeColor=colors.HexColor("#808080"), strokeWidth=0.6))
    core._label(d, 70, 183, "红线：06-05", 6.5, "#C00000")
    core._label(d, 145, 183, "蓝线：06-15", 6.5, "#4472C4")
    return d


def bucket_comparison_chart(data: dict[str, list[dict]]) -> Drawing:
    width, height = 500, 230
    d = Drawing(width, height)
    core._label(d, width / 2, 215, "30分钟时段表现：目标日与全样本基线", 11, "#17365D", "middle")
    baseline = {row["bucket"]: row for row in data["bucket_metrics"] if row["label"] == "全样本"}
    for panel_x, date, title in [(12, "2026-06-05", "6月5日"), (258, "2026-06-15", "6月15日")]:
        rows = [row for row in data["bucket_metrics"] if row["label"] == date]
        values = [row["avg_return"] for row in rows]
        base_values = [baseline[row["bucket"]]["avg_return"] for row in rows]
        low = min(values + base_values + [0]) * 1.08
        high = max(values + base_values + [0]) * 1.12
        x0, y0, plot_w, plot_h = panel_x + 31, 44, 202, 135
        zero_y = core._scale(0, low, high, y0, plot_h)
        d.add(Line(x0, zero_y, x0 + plot_w, zero_y, strokeColor=colors.HexColor("#808080"), strokeWidth=0.6))
        core._label(d, panel_x + 125, 193, title, 8, "#17365D", "middle")
        base_points = []
        for i, row in enumerate(rows):
            x = x0 + i / (len(rows) - 1) * plot_w
            y = core._scale(row["avg_return"], low, high, y0, plot_h)
            bar_w = 12
            d.add(Rect(x - bar_w / 2, min(y, zero_y), bar_w, abs(y - zero_y), fillColor=core.GREEN if row["avg_return"] >= 0 else core.RED, strokeColor=None, fillOpacity=0.82))
            base_y = core._scale(base_values[i], low, high, y0, plot_h)
            base_points.append((x, base_y))
            short_label = row["bucket"][:5]
            core._label(d, x, 27, short_label, 5.2, anchor="middle")
        d.add(PolyLine(base_points, strokeColor=core.BLUE, strokeWidth=1.0, fillColor=None))
        for x, y in base_points:
            d.add(Circle(x, y, 1.5, fillColor=core.BLUE, strokeColor=None))
        core._label(d, panel_x + 2, y0 + plot_h - 3, f"{high:.2%}", 5.5)
        core._label(d, panel_x + 2, y0, f"{low:.2%}", 5.5)
    core._label(d, 158, 8, "柱：目标日平均收益", 6.2, "#C00000")
    core._label(d, 278, 8, "蓝线：全样本同时间段基线", 6.2, "#4472C4")
    return d


def page_header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D9E1F2"))
    canvas.line(18 * mm, height - 14 * mm, width - 18 * mm, height - 14 * mm)
    canvas.setFont(core.PDF_FONT, 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(18 * mm, 9 * mm, "关键回撤日分钟级归因报告")
    canvas.drawRightString(width - 18 * mm, 9 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def metric_rows(data: dict[str, list[dict]], date: str) -> list[list[str]]:
    rows = [["时段", "平均收益", "日收益贡献", "胜率", "盈亏比", "多单占比"]]
    for row in data["bucket_metrics"]:
        if row["label"] != date:
            continue
        rows.append([
            row["bucket"], core.pct(row["avg_return"], 4), core.pct(row["daily_contribution"], 4),
            core.pct(row["win_rate"]), core.mult(row["pl_ratio"]), core.pct(row["long_share"]),
        ])
    return rows


def contract_rows(data: dict[str, list[dict]], date: str) -> list[list[str]]:
    selected = sorted(
        [row for row in data["contract_buckets"] if row["date_label"] == date],
        key=lambda row: row["daily_contribution"],
    )[:7]
    rows = [["时段", "合约", "平均收益", "日贡献", "胜率", "多单占比"]]
    for row in selected:
        rows.append([
            row["bucket"], row["contract"], core.pct(row["avg_return"], 4),
            core.pct(row["daily_contribution"], 4), core.pct(row["win_rate"]), core.pct(row["long_share"]),
        ])
    return rows


def build_pdf(data: dict[str, list[dict]]) -> None:
    core.configure_charts()
    styles = core.build_styles()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=16 * mm,
        title="关键回撤日分钟级归因报告", author="Codex",
    )
    flow = [
        core.p("关键回撤日分钟级归因报告", styles["title"]),
        core.p(
            "固定阈值0.50｜研究日期：2026-06-05、2026-06-15｜北京时间｜"
            "分钟数据按12个合约等权汇总",
            styles["subtitle"],
        ),
        intraday_curve_chart(data),
        Spacer(1, 3 * mm),
        core.p("结论先行", styles["h1"]),
        core.p(
            "<b>6月5日：</b>损失高度集中在13:30以后，最后90分钟贡献约94%的全天亏损。"
            "14:30-14:59平均策略收益-1.0438%，胜率7.78%，模型92.22%做多；"
            "14:30、14:31等多个分钟出现12个合约同时亏损。",
            styles["callout"],
        ),
        core.p(
            "<b>6月15日：</b>损失分成早盘和尾盘两段。10:00-10:29模型97.22%做多但平均收益-0.2210%；"
            "14:30-14:59模型仅0.83%做多，几乎全部做空，但平均收益仍为-0.3537%。"
            "这表明模型当天先在上涨/下跌方向上反复落后，尾盘又发生反向押注。",
            styles["callout"],
        ),
        core.p(
            "分钟累计曲线使用“分钟平均策略收益÷240”的累计和，因此终点与既有报告中的日均策略收益一致；"
            "它用于定位日内损失来源，不代表把240个分钟收益连续复利。",
            styles["small"],
        ),
        PageBreak(),
        core.p("1. 30分钟时段定位", styles["h1"]),
        bucket_comparison_chart(data),
        core.p("2026-06-05", styles["h2"]),
        core.make_table(metric_rows(data, "2026-06-05"), [31*mm, 28*mm, 29*mm, 24*mm, 24*mm, 28*mm], font_size=6.9),
        Spacer(1, 3 * mm),
        core.p("2026-06-15", styles["h2"]),
        core.make_table(metric_rows(data, "2026-06-15"), [31*mm, 28*mm, 29*mm, 24*mm, 24*mm, 28*mm], font_size=6.9),
        Spacer(1, 3 * mm),
        core.p(
            "全样本中14:30-14:59原本是表现最好的时段之一：平均收益0.0871%、胜率54.13%、盈亏比1.18x。"
            "但两个关键回撤日该时段均显著转负，因此不能解释为“尾盘本来就弱”，而是目标日发生了异常反转。",
            styles["callout"],
        ),
        PageBreak(),
        core.p("2. 具体分钟与合约", styles["h1"]),
    ]

    worst = [["日期", "分钟", "平均收益", "胜率", "亏损合约数", "多单占比"]]
    for row in data["worst_minutes"][:8]:
        worst.append([
            row["date_label"], row["minute_label"], core.pct(row["avg_return"], 3),
            core.pct(row["win_rate"]), str(row["loss_n"]), core.pct(row["long_share"]),
        ])
    flow += [
        core.p("最差分钟", styles["h2"]),
        core.make_table(worst, [31*mm, 22*mm, 29*mm, 25*mm, 29*mm, 28*mm], font_size=7.0),
        Spacer(1, 3 * mm),
        core.p("6月5日：负贡献最大的“时段 × 合约”", styles["h2"]),
        core.make_table(contract_rows(data, "2026-06-05"), [31*mm, 21*mm, 29*mm, 26*mm, 25*mm, 28*mm], font_size=6.8),
        Spacer(1, 3 * mm),
        core.p("6月15日：负贡献最大的“时段 × 合约”", styles["h2"]),
        core.make_table(contract_rows(data, "2026-06-15"), [31*mm, 21*mm, 29*mm, 26*mm, 25*mm, 28*mm], font_size=6.8),
        Spacer(1, 3 * mm),
        core.p("可执行的模型检查", styles["h1"]),
        core.p(
            "1. 按时段分别校准阈值，至少区分早盘、上午后段、午后开盘和尾盘；当前全局阈值无法处理日内方向切换。<br/>"
            "2. 增加“方向集中度”保护：当多单或空单占比超过85%，且滚动30分钟胜率低于20%时，先降仓而不是继续放大同向信号。该阈值是基于异常日的诊断起点，仍需样本外回测。<br/>"
            "3. 对IC、IF、IH、IM设置合约族风险上限。6月5日尾盘出现12合约同时亏损，单合约止损不能阻止组合共振。<br/>"
            "4. 下一步应把市场价格、波动率和趋势特征接入，验证模型是否在单边快速反转日存在系统性滞后。",
            styles["body"],
        ),
    ]
    doc.build(flow, onFirstPage=page_header_footer, onLaterPages=page_header_footer)

    reader = PdfReader(str(OUTPUT_PDF))
    if len(reader.pages) != 3:
        raise RuntimeError(f"PDF页数异常：{len(reader.pages)}")
    embedded_font_objects = 0
    for page in reader.pages:
        fonts = page["/Resources"].get_object().get("/Font")
        if not fonts:
            continue
        for font_ref in fonts.get_object().values():
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
    print(f"OUTPUT={OUTPUT_PDF}")


if __name__ == "__main__":
    build_pdf(load_intraday_data())
