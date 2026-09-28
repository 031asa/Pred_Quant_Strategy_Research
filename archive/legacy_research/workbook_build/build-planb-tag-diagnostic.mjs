import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = path.resolve("..");
const jsonPath = path.join(outputDir, "PlanB胜率与Tag周期诊断_中间数据.json");
const outputPath = path.join(outputDir, "PlanB胜率与Tag周期诊断.xlsx");
const previewDir = path.join(outputDir, "planb-tag-previews");
const payload = JSON.parse(await fs.readFile(jsonPath, "utf8"));

const workbook = Workbook.create();
const colors = {
  navy: "#17365D",
  blue: "#4472C4",
  lightBlue: "#D9EAF7",
  paleBlue: "#EEF5FB",
  green: "#70AD47",
  paleGreen: "#E2F0D9",
  red: "#C00000",
  paleRed: "#FCE4D6",
  amber: "#BF8F00",
  paleAmber: "#FFF2CC",
  gray: "#667085",
  lightGray: "#E7EAF0",
  white: "#FFFFFF",
  black: "#000000",
};

function columnName(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function rangeAddress(startRow, startCol, rowCount, colCount) {
  const start = `${columnName(startCol)}${startRow + 1}`;
  const end = `${columnName(startCol + colCount - 1)}${startRow + rowCount}`;
  return `${start}:${end}`;
}

function titleBand(sheet, title, subtitle, lastCol = 10) {
  const titleRange = sheet.getRange(rangeAddress(0, 0, 1, lastCol));
  titleRange.merge();
  titleRange.values = [[title]];
  titleRange.format = {
    fill: colors.navy,
    font: { bold: true, color: colors.white, size: 17 },
    verticalAlignment: "center",
  };
  titleRange.format.rowHeight = 30;

  const subtitleRange = sheet.getRange(rangeAddress(1, 0, 2, lastCol));
  subtitleRange.merge();
  subtitleRange.values = [[subtitle]];
  subtitleRange.format = {
    fill: colors.paleBlue,
    font: { color: colors.gray, size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  subtitleRange.format.rowHeight = 34;
}

function sectionBand(sheet, row, title, lastCol) {
  const section = sheet.getRange(rangeAddress(row, 0, 1, lastCol));
  section.merge();
  section.values = [[title]];
  section.format = {
    fill: colors.blue,
    font: { bold: true, color: colors.white },
    verticalAlignment: "center",
  };
  section.format.rowHeight = 23;
}

function normalizeValue(value) {
  if (value === undefined || Number.isNaN(value)) return null;
  return value;
}

function writeTable(sheet, startRow, startCol, headers, rows, options = {}) {
  const matrix = [
    headers.map((header) => header.label),
    ...rows.map((row) => headers.map((header) => normalizeValue(row[header.key]))),
  ];
  const address = rangeAddress(startRow, startCol, matrix.length, headers.length);
  const target = sheet.getRange(address);
  target.values = matrix;
  target.format = {
    verticalAlignment: "center",
    wrapText: false,
    borders: {
      insideHorizontal: { style: "thin", color: colors.lightGray },
      bottom: { style: "thin", color: colors.lightGray },
    },
  };
  const headerRange = sheet.getRange(rangeAddress(startRow, startCol, 1, headers.length));
  headerRange.format = {
    fill: options.headerFill ?? colors.navy,
    font: { bold: true, color: colors.white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: colors.navy },
  };
  headerRange.format.rowHeight = options.headerHeight ?? 28;

  headers.forEach((header, colOffset) => {
    const columnRange = sheet.getRange(
      rangeAddress(startRow + 1, startCol + colOffset, Math.max(rows.length, 1), 1),
    );
    if (header.format) columnRange.format.numberFormat = header.format;
    columnRange.format.horizontalAlignment = header.align ?? (header.format ? "right" : "left");
    sheet.getRange(rangeAddress(startRow, startCol + colOffset, matrix.length, 1)).format.columnWidth =
      header.width ?? 14;
  });
  return { address, startRow, endRow: startRow + rows.length, startCol, colCount: headers.length };
}

function addColorScale(sheet, address, low, mid, high) {
  sheet.getRange(address).conditionalFormats.add("colorScale", {
    colors: [low, mid, high],
    thresholds: ["min", "50%", "max"],
  });
}

function addStatusBlock(sheet, row, title, body, status = "warning", lastCol = 10) {
  const fill = status === "bad" ? colors.paleRed : status === "good" ? colors.paleGreen : colors.paleAmber;
  const fontColor = status === "bad" ? colors.red : status === "good" ? "#375623" : "#7F6000";
  const titleRange = sheet.getRange(rangeAddress(row, 0, 1, lastCol));
  titleRange.merge();
  titleRange.values = [[title]];
  titleRange.format = { fill, font: { bold: true, color: fontColor }, verticalAlignment: "center" };
  const bodyRange = sheet.getRange(rangeAddress(row + 1, 0, 3, lastCol));
  bodyRange.merge();
  bodyRange.values = [[body]];
  bodyRange.format = { fill, font: { color: colors.black }, wrapText: true, verticalAlignment: "center" };
  bodyRange.format.rowHeight = 48;
}

function baseSheet(name) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  return sheet;
}

const meta = payload.metadata;
const plan = payload.plan_b;
const diag = payload.tag_diagnostics;

// 1. 结论
{
  const sheet = baseSheet("结论");
  titleBand(
    sheet,
    "Plan B胜率与Tag周期诊断",
    `数据：${meta.source_path}｜分类中心：${meta.classification_center.toFixed(2)}｜过去${meta.rolling_days}个交易日滚动中位数｜本阶段不计算累计收益`,
    12,
  );

  const afterRows = plan.summary
    .filter((row) => row.segment === "14:30及以后")
    .sort((a, b) => a.strength - b.strength);
  const transitionRows = plan.transitions
    .filter((row) => row.segment === "14:30及以后" && row.transition === "多头→转为空头" && row.strength > 0)
    .sort((a, b) => a.strength - b.strength);
  const beforeRepeat = diag.repeat_summary.find(
    (row) => row.segment === "14:30以前" && row.contract === "全部合约",
  );
  const afterRepeat = diag.repeat_summary.find(
    (row) => row.segment === "14:30及以后" && row.contract === "全部合约",
  );
  const beforeAcf = Object.fromEntries(
    diag.acf
      .filter((row) => row.segment === "14:30以前" && row.contract === "全部合约")
      .map((row) => [row.lag_minutes, row.correlation]),
  );
  const afterAcf = Object.fromEntries(
    diag.acf
      .filter((row) => row.segment === "14:30及以后" && row.contract === "全部合约")
      .map((row) => [row.lag_minutes, row.correlation]),
  );

  addStatusBlock(
    sheet,
    4,
    "阶段结论：当前Plan B没有实现“提高尾盘空头胜率”的目标",
    `14:30后，修正强度从0%提高到100%时，空头占比由${(afterRows[0].short_share * 100).toFixed(2)}%升至${(afterRows.at(-1).short_share * 100).toFixed(2)}%，但空头胜率由${(afterRows[0].short_win_rate * 100).toFixed(2)}%降至${(afterRows.at(-1).short_win_rate * 100).toFixed(2)}%。多头转空头样本的胜率也由25%修正时的${(transitionRows[0].win_rate * 100).toFixed(2)}%逐步降至100%修正时的${(transitionRows.at(-1).win_rate * 100).toFixed(2)}%。因此增加的空头数量没有转化为更高的空头质量。`,
    "bad",
    12,
  );

  sectionBand(sheet, 9, "14:30后Plan B核心结果", 12);
  writeTable(
    sheet,
    10,
    0,
    [
      { key: "strength", label: "修正强度", format: "0%", width: 12 },
      { key: "long_share", label: "做多占比", format: "0.00%", width: 13 },
      { key: "long_win_rate", label: "做多胜率", format: "0.00%", width: 13 },
      { key: "short_share", label: "做空占比", format: "0.00%", width: 13 },
      { key: "short_win_rate", label: "做空胜率", format: "0.00%", width: 13 },
      { key: "direction_change_rate", label: "方向变化率", format: "0.00%", width: 14 },
    ],
    afterRows,
  );
  addColorScale(sheet, "E12:E16", colors.paleRed, colors.paleAmber, colors.paleGreen);

  sectionBand(sheet, 18, "Tag周期诊断核心证据", 12);
  const evidenceRows = [
    {
      item: "14:30前相邻Tag完全重复率",
      value: beforeRepeat.exact_repeat_rate,
      unit: "比例",
      interpretation: "仅约0.35%，大量完全重复并不是常态。",
    },
    {
      item: "14:30后相邻Tag完全重复率",
      value: afterRepeat.exact_repeat_rate,
      unit: "比例",
      interpretation: "仅约0.22%，此前看到的重复极端值属于少数异常段。",
    },
    {
      item: "14:30前Tag滞后1分钟相关",
      value: beforeAcf[1],
      unit: "相关系数",
      interpretation: "高度重叠。",
    },
    {
      item: "14:30前Tag滞后30分钟相关",
      value: beforeAcf[30],
      unit: "相关系数",
      interpretation: "由1分钟0.957逐步降到30分钟约0.026，形态高度疑似30分钟重叠收益标签。",
    },
    {
      item: "14:30后Tag滞后29分钟相关",
      value: afterAcf[29],
      unit: "相关系数",
      interpretation: "仍约0.575，且Tag波动在14:30处明显跳升，说明尾盘标签结构发生变化。",
    },
  ];
  writeTable(
    sheet,
    19,
    0,
    [
      { key: "item", label: "证据", width: 28 },
      { key: "value", label: "数值", format: "0.0000", width: 14 },
      { key: "unit", label: "单位", width: 12 },
      { key: "interpretation", label: "解释", width: 68 },
    ],
    evidenceRows,
  );
  sheet.getRange("D21:D25").format.wrapText = true;

  addStatusBlock(
    sheet,
    27,
    "Tag周期的当前推断",
    "14:30以前的滞后相关衰减形态高度疑似“未来30个交易分钟收益”。14:30整点后Tag波动突然放大，且29分钟滞后仍高度相关，结合典型日曲线，强烈怀疑标签采用按行shift(-30)之类的方式：14:30后的标签可能跨越收盘并连接到下一交易日分钟。这是数据内推断，不是最终确认；应使用标签生成代码或原始分钟价格复算。",
    "warning",
    12,
  );

  sectionBand(sheet, 32, "建议", 12);
  const recommendations = [
    ["1", "先暂停累计收益计算", "Plan B尚未改善尾盘空头胜率，暂不需要进入收益阶段。"],
    ["2", "确认Tag生成公式", "优先寻找shift、forward_return等标签代码；若有分钟价格，重算未来30分钟收益并逐行匹配。"],
    ["3", "重点检查14:30边界", "确认30分钟预测周期是否允许跨收盘，以及跨日标签是否符合真实交易与持仓设定。"],
    ["4", "胜率暂作描述性结果", "高度重叠的Tag使行级样本不独立，后续显著性检验应使用非重叠样本或按交易日分块。"],
  ].map(([order, action, detail]) => ({ order, action, detail }));
  writeTable(
    sheet,
    33,
    0,
    [
      { key: "order", label: "序号", width: 9, align: "center" },
      { key: "action", label: "动作", width: 25 },
      { key: "detail", label: "说明", width: 75 },
    ],
    recommendations,
  );
  sheet.getRange("C35:C38").format.wrapText = true;
  sheet.freezePanes.freezeRows(3);
}

// 2. PlanB总体胜率
{
  const sheet = baseSheet("PlanB总体胜率");
  titleBand(sheet, "Plan B总体胜率", "分别展示全体、14:30以前、14:30及以后的多空样本结构与胜率。最初20个交易日为预热期。", 14);
  const headers = [
    { key: "segment", label: "时段", width: 16 },
    { key: "strength", label: "修正强度", format: "0%", width: 12 },
    { key: "sample_n", label: "样本量", format: "#,##0", width: 14 },
    { key: "long_n", label: "做多样本", format: "#,##0", width: 14 },
    { key: "long_share", label: "做多占比", format: "0.00%", width: 13 },
    { key: "long_wins", label: "做多盈利数", format: "#,##0", width: 14 },
    { key: "long_win_rate", label: "做多胜率", format: "0.00%", width: 13 },
    { key: "short_n", label: "做空样本", format: "#,##0", width: 14 },
    { key: "short_share", label: "做空占比", format: "0.00%", width: 13 },
    { key: "short_wins", label: "做空盈利数", format: "#,##0", width: 14 },
    { key: "short_win_rate", label: "做空胜率", format: "0.00%", width: 13 },
    { key: "neutral_n", label: "中性样本", format: "#,##0", width: 13 },
    { key: "direction_change_n", label: "方向变化数", format: "#,##0", width: 15 },
    { key: "direction_change_rate", label: "方向变化率", format: "0.00%", width: 14 },
  ];
  writeTable(sheet, 4, 0, headers, plan.summary);
  addColorScale(sheet, `G6:G${5 + plan.summary.length}`, colors.paleRed, colors.paleAmber, colors.paleGreen);
  addColorScale(sheet, `K6:K${5 + plan.summary.length}`, colors.paleRed, colors.paleAmber, colors.paleGreen);
  sheet.freezePanes.freezeRows(5);
}

// 3. PlanB月度胜率
{
  const sheet = baseSheet("PlanB月度胜率");
  titleBand(sheet, "Plan B月度胜率", "用于检查胜率变化是否稳定存在于多数月份，而不是由少数异常月份贡献。", 13);
  const headers = [
    { key: "segment", label: "时段", width: 16 },
    { key: "month", label: "月份", width: 12 },
    { key: "strength", label: "修正强度", format: "0%", width: 12 },
    { key: "sample_n", label: "样本量", format: "#,##0", width: 13 },
    { key: "long_n", label: "做多样本", format: "#,##0", width: 13 },
    { key: "long_share", label: "做多占比", format: "0.00%", width: 13 },
    { key: "long_win_rate", label: "做多胜率", format: "0.00%", width: 13 },
    { key: "short_n", label: "做空样本", format: "#,##0", width: 13 },
    { key: "short_share", label: "做空占比", format: "0.00%", width: 13 },
    { key: "short_win_rate", label: "做空胜率", format: "0.00%", width: 13 },
    { key: "neutral_n", label: "中性样本", format: "#,##0", width: 13 },
    { key: "direction_change_n", label: "方向变化数", format: "#,##0", width: 14 },
    { key: "direction_change_rate", label: "方向变化率", format: "0.00%", width: 14 },
  ];
  writeTable(sheet, 4, 0, headers, plan.monthly);
  addColorScale(sheet, `G6:G${5 + plan.monthly.length}`, colors.paleRed, colors.paleAmber, colors.paleGreen);
  addColorScale(sheet, `J6:J${5 + plan.monthly.length}`, colors.paleRed, colors.paleAmber, colors.paleGreen);
  sheet.freezePanes.freezeRows(5);
}

// 4. 方向转换
{
  const sheet = baseSheet("方向转换");
  titleBand(sheet, "Plan B方向转换", "重点查看“多头转为空头”样本的做空胜率是否随修正增强而提高。", 9);
  const headers = [
    { key: "segment", label: "时段", width: 16 },
    { key: "strength", label: "修正强度", format: "0%", width: 12 },
    { key: "transition", label: "方向转换", width: 20 },
    { key: "sample_n", label: "样本量", format: "#,##0", width: 14 },
    { key: "sample_share", label: "样本占比", format: "0.00%", width: 13 },
    { key: "wins", label: "转换后盈利数", format: "#,##0", width: 15 },
    { key: "win_rate", label: "转换后胜率", format: "0.00%", width: 15 },
    { key: "tag_positive_rate", label: "Tag正值比例", format: "0.00%", width: 15 },
    { key: "tag_negative_rate", label: "Tag负值比例", format: "0.00%", width: 15 },
  ];
  writeTable(sheet, 4, 0, headers, plan.transitions);
  addColorScale(sheet, `G6:G${5 + plan.transitions.length}`, colors.paleRed, colors.paleAmber, colors.paleGreen);
  sheet.freezePanes.freezeRows(5);
}

// 5. 滚动中心
{
  const sheet = baseSheet("滚动中心");
  titleBand(sheet, "过去20个交易日预测中心", "所有偏置估计只使用当前交易日以前的数据；空白为最初20个交易日预热期。", 4);
  writeTable(
    sheet,
    4,
    0,
    [
      { key: "date", label: "日期", width: 14 },
      { key: "segment", label: "时段", width: 16 },
      { key: "rolling_center", label: "滚动score中位数", format: "0.000000", width: 20 },
      { key: "estimated_bias", label: "相对0.5偏置", format: "0.000000", width: 18 },
    ],
    plan.rolling_centers,
  );
  addColorScale(sheet, `D6:D${5 + plan.rolling_centers.length}`, colors.paleGreen, colors.paleAmber, colors.paleRed);
  sheet.freezePanes.freezeRows(5);
}

// 6. Tag周期摘要
{
  const sheet = baseSheet("Tag周期摘要");
  titleBand(sheet, "Tag重复结构与周期摘要", "重复率用于判断Tag是否被低频标签批量填充；滞后相关用于推断可能的重叠预测周期。", 10);
  sectionBand(sheet, 4, "Tag分位数", 10);
  writeTable(
    sheet,
    5,
    0,
    [
      { key: "segment", label: "时段", width: 16 },
      { key: "sample_n", label: "样本量", format: "#,##0", width: 14 },
      { key: "mean", label: "均值", format: "0.000000", width: 14 },
      { key: "min", label: "最小值", format: "0.000000", width: 14 },
      { key: "p01", label: "P1", format: "0.000000", width: 14 },
      { key: "p10", label: "P10", format: "0.000000", width: 14 },
      { key: "median", label: "中位数", format: "0.000000", width: 14 },
      { key: "p90", label: "P90", format: "0.000000", width: 14 },
      { key: "p99", label: "P99", format: "0.000000", width: 14 },
      { key: "max", label: "最大值", format: "0.000000", width: 14 },
    ],
    diag.tag_quantiles,
  );

  sectionBand(sheet, 10, "合约与时段重复摘要", 10);
  writeTable(
    sheet,
    11,
    0,
    [
      { key: "segment", label: "时段", width: 16 },
      { key: "contract", label: "合约", width: 13 },
      { key: "contract_day_groups", label: "合约日组数", format: "#,##0", width: 15 },
      { key: "total_rows", label: "总行数", format: "#,##0", width: 15 },
      { key: "weighted_unique_ratio", label: "Tag唯一值比例", format: "0.00%", width: 17 },
      { key: "exact_repeat_rate", label: "相邻完全重复率", format: "0.00%", width: 18 },
      { key: "near_repeat_rate", label: "相邻近似重复率", format: "0.00%", width: 18 },
      { key: "median_max_run", label: "最长重复中位数", format: "0.0", width: 18 },
      { key: "p95_max_run", label: "最长重复P95", format: "0.0", width: 17 },
      { key: "max_run", label: "最大连续分钟", format: "#,##0", width: 17 },
    ],
    diag.repeat_summary,
  );
  const acfStart = 14 + diag.repeat_summary.length;
  sectionBand(sheet, acfStart, "全部合约Tag滞后相关", 10);
  const acfAll = diag.acf.filter((row) => row.contract === "全部合约");
  writeTable(
    sheet,
    acfStart + 1,
    0,
    [
      { key: "segment", label: "时段", width: 16 },
      { key: "lag_minutes", label: "滞后分钟", format: "0", width: 13 },
      { key: "pair_n", label: "样本对数量", format: "#,##0", width: 16 },
      { key: "correlation", label: "相关系数", format: "0.0000", width: 14 },
    ],
    acfAll,
  );
  sheet.freezePanes.freezeRows(3);
}

// 7. Tag日内结构
{
  const sheet = baseSheet("Tag日内结构");
  titleBand(sheet, "Tag日内结构", "每一分钟的Tag分布、正值比例、相邻重复率；Tag单位保持parquet原始小数。", 13);
  writeTable(
    sheet,
    4,
    0,
    [
      { key: "segment", label: "时段", width: 16 },
      { key: "minute", label: "分钟序号", format: "0", width: 12 },
      { key: "time", label: "时间", width: 10 },
      { key: "sample_n", label: "样本量", format: "#,##0", width: 13 },
      { key: "tag_mean", label: "Tag均值", format: "0.000000", width: 14 },
      { key: "tag_std", label: "Tag标准差", format: "0.000000", width: 14 },
      { key: "tag_p10", label: "Tag P10", format: "0.000000", width: 14 },
      { key: "tag_median", label: "Tag中位数", format: "0.000000", width: 15 },
      { key: "tag_p90", label: "Tag P90", format: "0.000000", width: 14 },
      { key: "tag_positive_rate", label: "Tag正值比例", format: "0.00%", width: 16 },
      { key: "adjacent_pair_n", label: "相邻样本对", format: "#,##0", width: 14 },
      { key: "exact_repeat_rate", label: "完全重复率", format: "0.00%", width: 15 },
      { key: "near_repeat_rate", label: "近似重复率", format: "0.00%", width: 15 },
    ],
    diag.intraday,
  );
  addColorScale(sheet, `F6:F${5 + diag.intraday.length}`, colors.paleGreen, colors.paleAmber, colors.paleRed);
  sheet.freezePanes.freezeRows(5);
}

// 8. Tag合约日明细
{
  const sheet = baseSheet("Tag合约日明细");
  titleBand(sheet, "Tag合约日重复诊断明细", "每一行代表一个“日期×合约×时段”。用于定位唯一值比例低或连续重复时间长的异常组。", 12);
  writeTable(
    sheet,
    4,
    0,
    [
      { key: "date", label: "日期", width: 13 },
      { key: "contract", label: "合约", width: 11 },
      { key: "segment", label: "时段", width: 16 },
      { key: "row_n", label: "行数", format: "#,##0", width: 11 },
      { key: "unique_tag_n", label: "Tag唯一值数", format: "#,##0", width: 15 },
      { key: "unique_ratio", label: "唯一值比例", format: "0.00%", width: 14 },
      { key: "adjacent_pairs", label: "相邻样本对", format: "#,##0", width: 14 },
      { key: "exact_repeat_n", label: "完全重复数", format: "#,##0", width: 14 },
      { key: "exact_repeat_rate", label: "完全重复率", format: "0.00%", width: 15 },
      { key: "near_repeat_n", label: "近似重复数", format: "#,##0", width: 14 },
      { key: "near_repeat_rate", label: "近似重复率", format: "0.00%", width: 15 },
      { key: "max_run_length", label: "最大连续分钟", format: "#,##0", width: 16 },
    ],
    diag.contract_day,
  );
  addColorScale(sheet, `F6:F${5 + diag.contract_day.length}`, colors.paleRed, colors.paleAmber, colors.paleGreen);
  addColorScale(sheet, `I6:I${5 + diag.contract_day.length}`, colors.paleGreen, colors.paleAmber, colors.paleRed);
  addColorScale(sheet, `L6:L${5 + diag.contract_day.length}`, colors.paleGreen, colors.paleAmber, colors.paleRed);
  sheet.freezePanes.freezeRows(5);
}

// 9. Tag长重复段
{
  const sheet = baseSheet("Tag长重复段");
  titleBand(sheet, "Tag连续重复段", "仅保留连续长度最大的前500段。大量最长重复来自Tag=0，需要结合日期与行情状态判断。", 8);
  writeTable(
    sheet,
    4,
    0,
    [
      { key: "date", label: "日期", width: 13 },
      { key: "contract", label: "合约", width: 11 },
      { key: "segment", label: "时段", width: 16 },
      { key: "start_time", label: "开始时间", width: 12 },
      { key: "end_time", label: "结束时间", width: 12 },
      { key: "run_length", label: "连续分钟", format: "#,##0", width: 13 },
      { key: "tag_raw", label: "Tag原始值", format: "0.000000", width: 15 },
      { key: "tag_bp", label: "Tag（bp）", format: "0.00", width: 14 },
    ],
    diag.long_runs,
  );
  addColorScale(sheet, `F6:F${5 + diag.long_runs.length}`, colors.paleGreen, colors.paleAmber, colors.paleRed);
  sheet.freezePanes.freezeRows(5);
}

// Visual checks and export
await fs.mkdir(previewDir, { recursive: true });
const previewSpecs = [
  ["结论", "A1:L38"],
  ["PlanB总体胜率", "A1:N20"],
  ["PlanB月度胜率", "A1:M30"],
  ["方向转换", "A1:I30"],
  ["滚动中心", "A1:D30"],
  ["Tag周期摘要", "A1:J45"],
  ["Tag日内结构", "A1:M30"],
  ["Tag合约日明细", "A1:L30"],
  ["Tag长重复段", "A1:H30"],
];

for (const [sheetName, range] of previewSpecs) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const safeName = sheetName.replaceAll(":", "_");
  await fs.writeFile(
    path.join(previewDir, `${safeName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const inspect = await workbook.inspect({
  kind: "table",
  range: "结论!A1:L38",
  include: "values,formulas",
  tableMaxRows: 40,
  tableMaxCols: 12,
  maxChars: 12000,
});
await fs.writeFile(path.join(outputDir, "PlanB胜率与Tag周期诊断.xlsx.inspect.ndjson"), inspect.ndjson, "utf8");

const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errorScan.ndjson);

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);
console.log(`Saved ${outputPath}`);

