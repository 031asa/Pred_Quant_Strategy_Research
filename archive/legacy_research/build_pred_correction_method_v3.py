from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
helper_source = (BASE_DIR / "build_pred_segment_brief.py").read_text(encoding="utf-8")
exec(helper_source.split("doc = Document()", 1)[0], globals())

OUTPUT = BASE_DIR / "14点30分段_Pred修正方案分析.docx"

doc = Document()
section = doc.sections[0]
section.page_width = Inches(8.5)
section.page_height = Inches(11)
section.top_margin = Inches(0.82)
section.bottom_margin = Inches(0.78)
section.left_margin = Inches(1.0)
section.right_margin = Inches(1.0)
section.header_distance = Inches(0.492)
section.footer_distance = Inches(0.492)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Arial"
normal._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.10
for style_name, size, before, after in (
    ("Heading 1", 16, 12, 6),
    ("Heading 2", 13, 10, 5),
    ("Heading 3", 12, 8, 4),
):
    style = styles[style_name]
    style.font.name = "Arial"
    style._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    style._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    style.font.size = Pt(size)
    style.font.bold = True
    style.font.color.rgb = RGBColor.from_string(BLUE if style_name != "Heading 3" else NAVY)
    style.paragraph_format.space_before = Pt(before)
    style.paragraph_format.space_after = Pt(after)

header = section.header
hp = header.paragraphs[0]
hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
set_font(hp.add_run("模型验证简报  |  Pred 修正方法"), size=8.5, color=MID_GRAY)

footer = section.footer
fp = footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_font(fp.add_run("内部研究材料 · 2026-08-05"), size=8.5, color=MID_GRAY)

title = doc.add_paragraph()
title.paragraph_format.space_before = Pt(10)
title.paragraph_format.space_after = Pt(5)
set_font(title.add_run("Pred 修正方法说明（14:30 分段）"), size=23, bold=True, color=NAVY)
subtitle = doc.add_paragraph()
subtitle.paragraph_format.space_after = Pt(12)
set_font(subtitle.add_run("重点回答：修正了什么、如何计算、每条样本怎样改变"), size=13, color=MID_GRAY)
meta = doc.add_paragraph()
meta.paragraph_format.space_after = Pt(14)
set_font(meta.add_run("数据：pred_eval.parquet  |  有效样本：1,670,400  |  历史窗口：20 个交易日"), size=9.5, color=MID_GRAY)

add_callout(
    doc,
    "一句话说明",
    "这次实际测试的修正，只处理“历史分布中心高于0.5”的多头偏置：偏多时统一向下平移；偏空时完全不动。它不是重新训练模型，也不是移动平均平滑。",
)

add_heading(doc, "一、这次实测究竟如何修正 Pred")
add_body(doc, "原始 parquet 中的 pred 以0为中心，因此先恢复为0～1附近的 score：")
add_callout(doc, "步骤1：恢复分数", "score = clip(pred + 0.5, 0, 1)", fill=LIGHT_BLUE)

steps = [
    ["1", "划分时段", "Datetime < 14:30 为盘中；Datetime ≥ 14:30 为尾盘，14:30整点归尾盘"],
    ["2", "估计分布中心", "对两个时段分别使用此前20个交易日的全部 score 计算中位数 m(t)"],
    ["3", "计算多头偏置", "bias(t) = max[m(t) − 0.5, 0]；m(t)≤0.5 时 bias=0"],
    ["4", "修正每条分数", "adjusted_score = clip[score − ρ×bias(t), 0, 1]"],
    ["5", "生成方向", "adjusted_score>0.5 做多；adjusted_score<0.5 做空；等于0.5记中性"],
]
add_table(doc, ["步骤", "操作", "具体计算"], steps, [900, 1900, 6560])

add_body(doc, "其中 ρ 是修正强度，本次测试 0%、25%、50%、75%、100%。历史中位数只使用当日之前的数据，因此当日信号不读取未来信息。")
add_callout(
    doc,
    "完整公式",
    "adjusted_score(t,i) = clip{score(t,i) − ρ × max[m(t) − 0.5, 0], 0, 1}",
    fill=GREEN,
)

add_heading(doc, "二、14:30 前后分别发生了什么")
add_heading(doc, "2.1 14:30 前：实际几乎没有修正", level=2)
add_body(doc, "14:30前滚动中位数的典型值为0.483094，低于0.5。代入公式：")
add_callout(doc, "盘中计算", "bias = max(0.483094 − 0.5, 0) = 0；因此 adjusted_score = score。无论ρ取多少，大多数交易日的盘中 Pred 都保持原值。", fill=GREEN)

before_examples = [
    ["0.480", "0", "0.480", "做空", "不变"],
    ["0.520", "0", "0.520", "做多", "不变"],
]
add_table(doc, ["原 score", "扣减值", "修正后 score", "最终方向", "变化"], before_examples, [1600, 1500, 2000, 1800, 2460])
add_body(doc, "580个有效交易日中，14:30前只有6天的历史中位数高于0.5。因此100%修正后，盘中只有389个样本改变方向，占盘中样本约0.03%。")

add_heading(doc, "2.2 14:30 后：按多头偏置幅度向下平移", level=2)
add_body(doc, "14:30后滚动中位数的典型值为0.540087，高于0.5，因此典型多头偏置为0.040087。不同ρ对应的扣减值和等效阈值如下：")

shift_table = [
    ["0%", "0.0000", "0.5000"],
    ["25%", "约0.0100", "约0.5100"],
    ["50%", "约0.0200", "约0.5200"],
    ["75%", "约0.0301", "约0.5301"],
    ["100%", "约0.0401", "约0.5401"],
]
add_table(doc, ["ρ", "每条尾盘 score 的典型扣减值", "等效原始分界线"], shift_table, [1400, 4400, 3560], highlight_rows={1: AMBER})

add_body(doc, "以ρ=25%为例，典型扣减值约为0.0100：")
after_examples = [
    ["0.480", "0.470", "做空→做空", "原本已低于0.5"],
    ["0.505", "0.495", "做多→做空", "靠近边界，被转为空头"],
    ["0.560", "0.550", "做多→做多", "高分多头仍保留"],
]
add_table(doc, ["原 score", "修正后 score", "方向变化", "解释"], after_examples, [1600, 2100, 2300, 3360], highlight_rows={1: AMBER})

add_callout(
    doc,
    "等效理解",
    "把尾盘 score 统一减去0.0100，再与0.5比较，等价于不改score、直接把尾盘做多门槛从0.5000提高到约0.5100。因此这次修正本质上是动态阈值调整。",
)

add_heading(doc, "三、修正后实际效果")
effect = [
    ["0%", 0.5955, 0.5110, 0.3088, 0.5086, "—"],
    ["25%", 0.6014, 0.5109, 0.3557, 0.5087, "50.92%"],
    ["50%", 0.6076, 0.5109, 0.4048, 0.5082, "50.68%"],
    ["75%", 0.6139, 0.5106, 0.4547, 0.5062, "50.11%"],
    ["100%", 0.6203, 0.5103, 0.5049, 0.5030, "49.41%"],
]
add_table(
    doc,
    ["ρ", "全体做空占比", "全体做空胜率", "尾盘做空占比", "尾盘做空胜率", "尾盘新增空单胜率"],
    effect,
    [1100, 1600, 1600, 1600, 1600, 1860],
    percent_cols={1, 2, 3, 4},
    highlight_rows={1: AMBER},
)
add_bullet(doc, "盘中基本没有变化，因为盘中历史中位数低于0.5，公式主动把修正量截为0。")
add_bullet(doc, "尾盘做空占比随ρ上升，说明向下平移确实把边界多头转成了空头。")
add_bullet(doc, "但是新增空单胜率随ρ增强而下降：ρ=25%时为50.92%，ρ=100%时降到49.41%。")
add_callout(doc, "实测结论", "当前修正能改变方向数量，但不能证明空头质量提高。ρ=25%伤害最小，可保留为尾盘对照；不建议为了多空各半使用更强修正。", fill=AMBER)

add_heading(doc, "四、当前修正能做什么、不能做什么")
status_rows = [
    ["已经实测", "尾盘多头偏置的单向扣减", "公式、ρ网格及胜率结果均已计算"],
    ["已经实测", "盘中偏空时不反向上调", "盘中方向和胜率基本保持原样"],
    ["没有做到", "提高14:30前空头胜率", "当前公式在盘中多数日期修正量为0"],
    ["尚未实测", "盘中严格做空阈值与不交易区间", "这是下一阶段建议，不应写成已有结果"],
]
add_table(doc, ["状态", "内容", "说明"], status_rows, [1700, 3100, 4560], highlight_rows={2: AMBER, 3: LIGHT_BLUE})

add_heading(doc, "五、如果下一步目标是提高14:30前空头胜率，准备怎样改")
add_body(doc, "下一步不建议把盘中所有 Pred 再统一向下平移，因为这只会增加边界空单。更直接的修改是在决策层加入严格做空阈值 L 和不交易区间：")
add_callout(doc, "拟测试的新规则（尚未运行）", "score < L：做空；L ≤ score ≤ 0.5：不交易；score > 0.5：做多。L 将从0.40到0.50逐档扫描。", fill=LIGHT_BLUE)
add_body(doc, "每个交易日只使用此前20或60个交易日的数据选择L，再把该阈值应用于下一交易日。选择标准同时考虑空头胜率、样本量、覆盖率以及按月/合约稳定性。")
add_body(doc, "这一方案与本次尾盘修正的区别是：尾盘修正负责减少整体多头偏置；盘中新规则负责从已有空头中筛选更强的空头。前者主要改变数量，后者才直接面向空头精准率。")

doc.core_properties.title = "Pred修正方法说明（14:30分段）"
doc.core_properties.subject = "Pred修正公式、样本影响与实测结果"
doc.core_properties.author = "策略研究"
doc.save(OUTPUT)
print(OUTPUT)
