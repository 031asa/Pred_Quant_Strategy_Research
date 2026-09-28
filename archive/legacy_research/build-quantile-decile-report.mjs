import fs from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir =
  "C:/Users/Hello/Documents/基础环境配置/outputs/019fb20f-2fd0-7341-9e50-cb40298a48ce";
const outputPath = `${outputDir}/按分布十等分_阈值分析.xlsx`;
const previewDir = `${outputDir}/quantile-previews`;
const sourcePath = `${outputDir}/按分布十等分_Jupyter源码.py`;
const configPath = `${outputDir}/按分布十等分_参数.json`;
const pythonExe =
  "C:/Users/Hello/PycharmProjects/PythonProject/.venv/Scripts/python.exe";

const config = JSON.parse(await fs.readFile(configPath, "utf8"));
const analysis = JSON.parse(
  execFileSync(
    pythonExe,
    [sourcePath, "--quantile", "--quantile-bins", String(config.quantile_bins), "--json"],
    {
      encoding: "utf8",
      maxBuffer: 50 * 1024 * 1024,
      env: {
        ...process.env,
        PYTHONUTF8: "1",
        PYTHONIOENCODING: "utf-8",
      },
    },
  ),
);

const bins = Number(analysis.config.quantile_bins);
const threshold = Number(analysis.config.direction_threshold);
const allGroups = [analysis.global];

for (const group of allGroups) {
  if (group.rows.length !== bins) {
    throw new Error(`${group.group} 的分档数不是 ${bins}。`);
  }
  const counts = group.rows.map((row) => Number(row.sample_n));
  const total = counts.reduce((sum, value) => sum + value, 0);
  if (total !== Number(group.n)) {
    throw new Error(`${group.group} 的分档样本合计与组样本数不一致。`);
  }
  if (Math.max(...counts) - Math.min(...counts) > 1) {
    throw new Error(`${group.group} 的各档样本数相差超过 1。`);
  }
  const ordered = group.rows.every(
    (row, index) =>
      index === 0 || Number(row.lower_score) >= Number(group.rows[index - 1].lower_score),
  );
  if (!ordered) {
    throw new Error(`${group.group} 的分数边界没有按升序排列。`);
  }
}

const detailRows = [];
for (const [dimension, groups] of [
  ["全体", [analysis.global]],
  ["按合约", analysis.contracts],
  ["按月份", analysis.months],
]) {
  for (const group of groups) {
    for (const row of group.rows) {
      detailRows.push([
        dimension,
        group.group,
        Number(row.decile),
        Number(row.lower_score),
        Number(row.upper_score),
        row.candidate_boundary === null ? null : Number(row.candidate_boundary),
        row.direction,
        Number(row.sample_n),
        Number(row.group_share),
        row.average_strategy_return,
        row.win_rate,
        row.average_profit_loss_ratio,
        Number(row.long_share),
        Number(row.short_share),
      ]);
    }
  }
}

const workbook = Workbook.create();
const reportSheet = workbook.worksheets.add("全体十分位分析");

const navy = "#17365D";
const blue = "#4472C4";
const lightBlue = "#D9EAF7";
const lightGray = "#F2F2F2";
const border = "#D9E1F2";
const yellow = "#FFF2CC";
const redLight = "#FCE4D6";
const greenLight = "#E2F0D9";
const purpleLight = "#E4DFEC";
const bodyFont = "Microsoft YaHei";

function styleTitle(sheet, range, title) {
  sheet.mergeCells(range);
  const first = range.split(":")[0];
  sheet.getRange(first).values = [[title]];
  sheet.getRange(range).format = {
    fill: navy,
    font: { name: bodyFont, size: 16, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "left",
    verticalAlignment: "center",
  };
  sheet.getRange(range).format.rowHeight = 32;
}

function styleSection(sheet, row, lastColumn, title) {
  const range = `A${row}:${lastColumn}${row}`;
  sheet.mergeCells(range);
  sheet.getRange(`A${row}`).values = [[title]];
  sheet.getRange(range).format = {
    fill: lightBlue,
    font: { name: bodyFont, size: 11, bold: true, color: navy },
    verticalAlignment: "center",
    borders: { bottom: { style: "medium", color: blue } },
  };
  sheet.getRange(range).format.rowHeight = 24;
}

function styleHeader(sheet, range) {
  sheet.getRange(range).format = {
    fill: blue,
    font: { name: bodyFont, size: 9, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: border },
  };
  sheet.getRange(range).format.rowHeight = 28;
}

function styleData(sheet, range) {
  sheet.getRange(range).format = {
    font: { name: bodyFont, size: 9 },
    verticalAlignment: "center",
    borders: {
      insideHorizontal: { style: "thin", color: border },
      bottom: { style: "thin", color: border },
    },
  };
}

function addCard(sheet, labelRange, valueRange, label, value, fill, numberFormat) {
  sheet.mergeCells(labelRange);
  sheet.mergeCells(valueRange);
  const labelCell = labelRange.split(":")[0];
  const valueCell = valueRange.split(":")[0];
  sheet.getRange(labelCell).values = [[label]];
  if (typeof value === "string" && value.startsWith("=")) {
    sheet.getRange(valueCell).formulas = [[value]];
  } else {
    sheet.getRange(valueCell).values = [[value]];
  }
  sheet.getRange(labelRange).format = {
    fill,
    font: { name: bodyFont, size: 9, bold: true, color: navy },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: border },
  };
  sheet.getRange(valueRange).format = {
    fill: "#FFFFFF",
    font: { name: bodyFont, size: 15, bold: true, color: navy },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    numberFormat,
    borders: { preset: "outside", style: "thin", color: border },
  };
}

function rangeLabel(row) {
  return `[${Number(row.lower_score).toFixed(6)}, ${Number(row.upper_score).toFixed(6)}]`;
}

function bucketNumber(group) {
  return Number(group.rows[0].sample_n);
}

function crossBucket(group) {
  const row = group.rows.find((item) => item.direction === "跨阈值");
  return row ? `Q${row.decile}` : "无";
}

function boundary(group, bucket) {
  return Number(group.rows[bucket - 1].upper_score);
}

function buildReportSheet() {
  const sheet = reportSheet;
  sheet.showGridLines = false;
  styleTitle(sheet, "A1:J1", "按分布等样本十等分分析");
  sheet.mergeCells("A2:J2");
  sheet.getRange("A2").values = [[
    `原有规则保持不变：按全体、合约、月份分别分组；score> ${threshold.toFixed(2)} 做多取 Tag，score< ${threshold.toFixed(2)} 做空取 -Tag。唯一变化是每组按 score 排序后等样本切成 ${bins} 档，分数边界由各组实际分布决定。`,
  ]];
  sheet.getRange("A2:J2").format = {
    fill: lightGray,
    font: { name: bodyFont, size: 9, color: "#404040" },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange("A2:J2").format.rowHeight = 34;

  addCard(sheet, "A4:B4", "A5:B6", "等分档数", bins, lightBlue, "0");
  addCard(sheet, "C4:D4", "C5:D6", "全体样本", "=SUM(D10:D19)", greenLight, "#,##0");
  addCard(sheet, "E4:F4", "E5:F6", "每档样本", "=MIN(D10:D19)", lightBlue, "#,##0");
  addCard(sheet, "G4:H4", "G5:H6", "最大档差", "=MAX(D10:D19)-MIN(D10:D19)", yellow, "#,##0");
  addCard(sheet, "I4:J4", "I5:J6", "方向阈值", threshold, purpleLight, "0.00");

  styleSection(sheet, 8, "J", "全体：等样本十档表现");
  sheet.getRange("A9:J9").values = [[
    "档位",
    "实际分数范围",
    "方向口径",
    "样本量",
    "组内占比",
    "平均策略收益",
    "胜率",
    "平均盈亏比",
    "做多占比",
    "做空占比",
  ]];
  styleHeader(sheet, "A9:J9");
  const globalRows = analysis.global.rows.map((row) => [
    `Q${row.decile}`,
    rangeLabel(row),
    row.direction,
    Number(row.sample_n),
    Number(row.group_share),
    row.average_strategy_return,
    row.win_rate,
    row.average_profit_loss_ratio,
    Number(row.long_share),
    Number(row.short_share),
  ]);
  sheet.getRange("A10:J19").values = globalRows;
  styleData(sheet, "A10:J19");
  sheet.getRange("D10:D19").format.numberFormat = "#,##0";
  sheet.getRange("E10:E19").format.numberFormat = "0.00%";
  sheet.getRange("F10:F19").format.numberFormat = "0.0000%;[Red]-0.0000%;-";
  sheet.getRange("G10:G19").format.numberFormat = "0.00%";
  sheet.getRange("H10:H19").format.numberFormat = "0.00x";
  sheet.getRange("I10:J19").format.numberFormat = "0.00%";
  sheet.getRange("F10:F19").conditionalFormats.add("colorScale", {
    colors: ["#F4CCCC", "#FFFFFF", "#E2F0D9"],
    thresholds: ["min", "50%", "max"],
  });
  sheet.getRange("G10:G19").conditionalFormats.add("colorScale", {
    colors: ["#F4CCCC", "#FFF2CC", "#E2F0D9"],
    thresholds: ["min", "50%", "max"],
  });
  sheet.getRange("H10:H19").conditionalFormats.add("colorScale", {
    colors: ["#F4CCCC", "#FFF2CC", "#E2F0D9"],
    thresholds: ["min", "50%", "max"],
  });

  styleSection(sheet, 21, "J", "按合约：样本均衡与关键边界");
  sheet.getRange("A22:H22").values = [[
    "合约",
    "总样本",
    "每档最少",
    "每档最多",
    "最大档差",
    "Q5上界",
    "Q6上界",
    "跨0.5档",
  ]];
  styleHeader(sheet, "A22:H22");
  const contractRows = analysis.contracts.map((group) => [
    group.group,
    Number(group.n),
    Number(group.min_bucket_n),
    Number(group.max_bucket_n),
    Number(group.max_bucket_n) - Number(group.min_bucket_n),
    boundary(group, 5),
    boundary(group, 6),
    crossBucket(group),
  ]);
  const contractStart = 23;
  const contractEnd = contractStart + contractRows.length - 1;
  sheet.getRange(`A${contractStart}:H${contractEnd}`).values = contractRows;
  styleData(sheet, `A${contractStart}:H${contractEnd}`);
  sheet.getRange(`B${contractStart}:E${contractEnd}`).format.numberFormat = "#,##0";
  sheet.getRange(`F${contractStart}:G${contractEnd}`).format.numberFormat = "0.000000";
  sheet.getRange(`E${contractStart}:E${contractEnd}`).conditionalFormats.add("cellIs", {
    operator: "greaterThan",
    formula: 1,
    format: { fill: redLight, font: { color: "#9C0006", bold: true } },
  });

  const monthTitle = contractEnd + 2;
  styleSection(sheet, monthTitle, "J", "按月份：最近 12 个月样本均衡与关键边界");
  const monthHeader = monthTitle + 1;
  sheet.getRange(`A${monthHeader}:H${monthHeader}`).values = [[
    "月份",
    "总样本",
    "每档最少",
    "每档最多",
    "最大档差",
    "Q5上界",
    "Q6上界",
    "跨0.5档",
  ]];
  styleHeader(sheet, `A${monthHeader}:H${monthHeader}`);
  const recentMonths = analysis.months.slice(-12);
  const monthRows = recentMonths.map((group) => [
    group.group,
    Number(group.n),
    Number(group.min_bucket_n),
    Number(group.max_bucket_n),
    Number(group.max_bucket_n) - Number(group.min_bucket_n),
    boundary(group, 5),
    boundary(group, 6),
    crossBucket(group),
  ]);
  const monthStart = monthHeader + 1;
  const monthEnd = monthStart + monthRows.length - 1;
  sheet.getRange(`A${monthStart}:H${monthEnd}`).values = monthRows;
  styleData(sheet, `A${monthStart}:H${monthEnd}`);
  sheet.getRange(`B${monthStart}:E${monthEnd}`).format.numberFormat = "#,##0";
  sheet.getRange(`F${monthStart}:G${monthEnd}`).format.numberFormat = "0.000000";
  sheet.getRange(`E${monthStart}:E${monthEnd}`).conditionalFormats.add("cellIs", {
    operator: "greaterThan",
    formula: 1,
    format: { fill: redLight, font: { color: "#9C0006", bold: true } },
  });

  const ranked = analysis.global.rows
    .filter((row) => row.average_strategy_return !== null)
    .sort((a, b) => b.average_strategy_return - a.average_strategy_return);
  const best = ranked[0];
  const worst = ranked[ranked.length - 1];
  const cross = analysis.global.rows.find((row) => row.direction === "跨阈值");
  const conclusionTitle = monthEnd + 2;
  styleSection(sheet, conclusionTitle, "J", "关键结论");
  const conclusions = [
    `1. 全体、12 个合约、30 个月共 ${allGroups.length} 个分组均通过样本均衡审计；每组各档样本数最大差为 ${Math.max(...allGroups.map((group) => group.max_bucket_n - group.min_bucket_n))}。`,
    `2. 全体十档每档 ${bucketNumber(analysis.global).toLocaleString("zh-CN")} 条，占比 10%；分数边界由样本分布决定，不再使用写死的固定刻度。`,
    `3. 全体中平均策略收益最高为 Q${best.decile}（${rangeLabel(best)}），平均策略收益 ${(best.average_strategy_return * 100).toFixed(4)}%；最低为 Q${worst.decile}（${rangeLabel(worst)}），为 ${(worst.average_strategy_return * 100).toFixed(4)}%。`,
    cross
      ? `4. Q${cross.decile} 跨过 0.50；该档没有强行归为单一方向，而是逐笔按 score 与 0.50 的关系计算多空收益。`
      : "4. 全体十档没有跨过 0.50 的混合档。",
    "5. ‘十分位明细’保留全部合约和全部月份，可直接筛选维度、分组和档位查看收益、胜率与平均盈亏比。",
  ];
  conclusions.forEach((line, index) => {
    const row = conclusionTitle + 1 + index;
    sheet.mergeCells(`A${row}:J${row}`);
    sheet.getRange(`A${row}`).values = [[line]];
    sheet.getRange(`A${row}:J${row}`).format = {
      fill: index % 2 === 0 ? "#FFFFFF" : lightGray,
      font: { name: bodyFont, size: 9 },
      wrapText: true,
      verticalAlignment: "center",
      borders: { bottom: { style: "thin", color: border } },
    };
    sheet.getRange(`A${row}:J${row}`).format.rowHeight = 28;
  });

  const finalRow = conclusionTitle + conclusions.length;
  sheet.freezePanes.freezeRows(2);
  const widths = [10, 25, 13, 13, 13, 17, 13, 15, 13, 13];
  widths.forEach((width, index) => {
    sheet.getRangeByIndexes(0, index, finalRow, 1).format.columnWidth = width;
  });
  return { finalRow };
}

function buildDetailSheet() {
  const sheet = detailSheet;
  sheet.showGridLines = false;
  styleTitle(sheet, "A1:N1", "等样本十分位明细");
  sheet.mergeCells("A2:N2");
  sheet.getRange("A2").values = [[
    `每个全体/合约/月度分组独立排序并等样本切成 ${bins} 档。分数下界、上界和“下一档分界值”均来自该组实际分布；最后一档没有下一档，因此分界值留空。跨 0.50 的档位逐笔判定多空。`,
  ]];
  sheet.getRange("A2:N2").format = {
    fill: lightGray,
    font: { name: bodyFont, size: 9 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange("A2:N2").format.rowHeight = 34;
  sheet.getRange("A4:N4").values = [[
    "维度",
    "分组",
    "十分位",
    "分数下界",
    "分数上界",
    "下一档分界值",
    "方向口径",
    "样本量",
    "组内占比",
    "平均策略收益",
    "胜率",
    "平均盈亏比",
    "做多占比",
    "做空占比",
  ]];
  styleHeader(sheet, "A4:N4");
  const startRow = 5;
  const endRow = startRow + detailRows.length - 1;
  sheet.getRange(`A${startRow}:N${endRow}`).values = detailRows;
  styleData(sheet, `A${startRow}:N${endRow}`);
  const table = sheet.tables.add(`A4:N${endRow}`, true, "QuantileDetailTable");
  table.showFilterButton = true;
  sheet.getRange(`C${startRow}:C${endRow}`).format.numberFormat = "0";
  sheet.getRange(`D${startRow}:F${endRow}`).format.numberFormat = "0.000000";
  sheet.getRange(`H${startRow}:H${endRow}`).format.numberFormat = "#,##0";
  sheet.getRange(`I${startRow}:I${endRow}`).format.numberFormat = "0.00%";
  sheet.getRange(`J${startRow}:J${endRow}`).format.numberFormat =
    "0.0000%;[Red]-0.0000%;-";
  sheet.getRange(`K${startRow}:K${endRow}`).format.numberFormat = "0.00%";
  sheet.getRange(`L${startRow}:L${endRow}`).format.numberFormat = "0.00x";
  sheet.getRange(`M${startRow}:N${endRow}`).format.numberFormat = "0.00%";
  sheet.getRange(`J${startRow}:J${endRow}`).conditionalFormats.add("colorScale", {
    colors: ["#F4CCCC", "#FFFFFF", "#E2F0D9"],
    thresholds: ["min", "50%", "max"],
  });
  sheet.getRange(`K${startRow}:K${endRow}`).conditionalFormats.add("colorScale", {
    colors: ["#F4CCCC", "#FFF2CC", "#E2F0D9"],
    thresholds: ["min", "50%", "max"],
  });
  sheet.getRange(`L${startRow}:L${endRow}`).conditionalFormats.add("colorScale", {
    colors: ["#F4CCCC", "#FFF2CC", "#E2F0D9"],
    thresholds: ["min", "50%", "max"],
  });
  sheet.getRange(`G${startRow}:G${endRow}`).conditionalFormats.add("containsText", {
    text: "跨阈值",
    format: { fill: yellow, font: { color: "#7F6000", bold: true } },
  });
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(3);
  const widths = [11, 15, 10, 13, 13, 16, 13, 12, 12, 17, 12, 15, 12, 12];
  widths.forEach((width, index) => {
    sheet.getRangeByIndexes(0, index, endRow, 1).format.columnWidth = width;
  });
  return { startRow, endRow };
}

function buildGlobalOnlyReportSheet() {
  const sheet = reportSheet;
  sheet.showGridLines = false;
  styleTitle(sheet, "A1:J1", "全体按分布等样本十等分分析");
  sheet.mergeCells("A2:J2");
  sheet.getRange("A2").values = [[
    `只看全体样本：按 score 从小到大排序后等样本切成 ${bins} 档。score> ${threshold.toFixed(2)} 做多取 Tag，score< ${threshold.toFixed(2)} 做空取 -Tag；若某档跨过 0.50，则档内逐笔判定方向。`,
  ]];
  sheet.getRange("A2:J2").format = {
    fill: lightGray,
    font: { name: bodyFont, size: 9, color: "#404040" },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange("A2:J2").format.rowHeight = 32;

  addCard(sheet, "A4:B4", "A5:B6", "等分档数", bins, lightBlue, "0");
  addCard(sheet, "C4:D4", "C5:D6", "全体样本", "=SUM(D10:D19)", greenLight, "#,##0");
  addCard(sheet, "E4:F4", "E5:F6", "每档样本", "=MIN(D10:D19)", lightBlue, "#,##0");
  addCard(sheet, "G4:H4", "G5:H6", "最大档差", "=MAX(D10:D19)-MIN(D10:D19)", yellow, "#,##0");
  addCard(sheet, "I4:J4", "I5:J6", "方向阈值", threshold, purpleLight, "0.00");

  styleSection(sheet, 8, "J", "全体等样本十档表现");
  sheet.getRange("A9:J9").values = [[
    "档位",
    "实际分数范围",
    "方向口径",
    "样本量",
    "样本占比",
    "平均策略收益",
    "胜率",
    "平均盈亏比",
    "做多占比",
    "做空占比",
  ]];
  styleHeader(sheet, "A9:J9");
  const rows = analysis.global.rows.map((row) => [
    `Q${row.decile}`,
    rangeLabel(row),
    row.direction,
    Number(row.sample_n),
    Number(row.group_share),
    row.average_strategy_return,
    row.win_rate,
    row.average_profit_loss_ratio,
    Number(row.long_share),
    Number(row.short_share),
  ]);
  sheet.getRange("A10:J19").values = rows;
  styleData(sheet, "A10:J19");
  const table = sheet.tables.add("A9:J19", true, "GlobalQuantileTable");
  table.showFilterButton = true;
  sheet.getRange("D10:D19").format.numberFormat = "#,##0";
  sheet.getRange("E10:E19").format.numberFormat = "0.00%";
  sheet.getRange("F10:F19").format.numberFormat = "0.0000%;[Red]-0.0000%;-";
  sheet.getRange("G10:G19").format.numberFormat = "0.00%";
  sheet.getRange("H10:H19").format.numberFormat = "0.00x";
  sheet.getRange("I10:J19").format.numberFormat = "0.00%";
  sheet.getRange("F10:F19").conditionalFormats.add("colorScale", {
    colors: ["#F4CCCC", "#FFFFFF", "#E2F0D9"],
    thresholds: ["min", "50%", "max"],
  });
  sheet.getRange("G10:G19").conditionalFormats.add("colorScale", {
    colors: ["#F4CCCC", "#FFF2CC", "#E2F0D9"],
    thresholds: ["min", "50%", "max"],
  });
  sheet.getRange("H10:H19").conditionalFormats.add("colorScale", {
    colors: ["#F4CCCC", "#FFF2CC", "#E2F0D9"],
    thresholds: ["min", "50%", "max"],
  });
  sheet.getRange("C10:C19").conditionalFormats.add("containsText", {
    text: "跨阈值",
    format: { fill: yellow, font: { color: "#7F6000", bold: true } },
  });

  const ranked = analysis.global.rows
    .filter((row) => row.average_strategy_return !== null)
    .sort((a, b) => b.average_strategy_return - a.average_strategy_return);
  const best = ranked[0];
  const worst = ranked[ranked.length - 1];
  const cross = analysis.global.rows.find((row) => row.direction === "跨阈值");
  styleSection(sheet, 21, "J", "关键结论");
  const conclusions = [
    `1. 全体 ${Number(analysis.total_n).toLocaleString("zh-CN")} 条样本被等分为 ${bins} 档，每档 ${bucketNumber(analysis.global).toLocaleString("zh-CN")} 条、占比 10%，最大档差为 0。`,
    `2. 平均策略收益最高为 Q${best.decile}（${rangeLabel(best)}），为 ${(best.average_strategy_return * 100).toFixed(4)}%；最低为 Q${worst.decile}（${rangeLabel(worst)}），为 ${(worst.average_strategy_return * 100).toFixed(4)}%。`,
    cross
      ? `3. Q${cross.decile} 跨过 0.50；该档逐笔按 score 与 0.50 的关系计算多空，不强行归为单一方向。`
      : "3. 十档均未跨过 0.50。",
    "4. 分数边界完全由全体样本分布决定，没有再使用按合约或按月份的独立边界。",
  ];
  conclusions.forEach((line, index) => {
    const row = 22 + index;
    sheet.mergeCells(`A${row}:J${row}`);
    sheet.getRange(`A${row}`).values = [[line]];
    sheet.getRange(`A${row}:J${row}`).format = {
      fill: index % 2 === 0 ? "#FFFFFF" : lightGray,
      font: { name: bodyFont, size: 9 },
      wrapText: true,
      verticalAlignment: "center",
      borders: { bottom: { style: "thin", color: border } },
    };
    sheet.getRange(`A${row}:J${row}`).format.rowHeight = 28;
  });

  const finalRow = 25;
  sheet.freezePanes.freezeRows(9);
  const widths = [10, 25, 13, 13, 13, 17, 13, 15, 13, 13];
  widths.forEach((width, index) => {
    sheet.getRangeByIndexes(0, index, finalRow, 1).format.columnWidth = width;
  });
  return { finalRow };
}

const reportLayout = buildGlobalOnlyReportSheet();

await fs.mkdir(previewDir, { recursive: true });

async function saveRender(fileName, sheetName, range, scale = 1) {
  const preview = await workbook.render({ sheetName, range, scale, format: "png" });
  await fs.writeFile(
    `${previewDir}/${fileName}`,
    new Uint8Array(await preview.arrayBuffer()),
  );
}

await saveRender(
  "全体十分位分析.png",
  "全体十分位分析",
  `A1:J${reportLayout.finalRow}`,
  1.05,
);

const reportCheck = await workbook.inspect({
  kind: "table",
  range: `全体十分位分析!A1:J${reportLayout.finalRow}`,
  include: "values,formulas",
  tableMaxRows: 80,
  tableMaxCols: 10,
  maxChars: 9000,
});
console.log(reportCheck.ndjson);

const errorCheck = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errorCheck.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(
  `AUDIT=${JSON.stringify({
    totalN: analysis.total_n,
    groups: 1,
    bins,
    maxBucketDifference: Math.max(
      ...allGroups.map((group) => group.max_bucket_n - group.min_bucket_n),
    ),
    reportRows: analysis.global.rows.length,
  })}`,
);
console.log(`OUTPUT=${outputPath}`);
