import fs from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir =
  "C:/Users/Hello/Documents/基础环境配置/outputs/019fb20f-2fd0-7341-9e50-cb40298a48ce";
const outputPath = `${outputDir}/中位数阈值_20与50等分分析.xlsx`;
const previewDir = `${outputDir}/median-quantile-previews`;
const sourcePath = `${outputDir}/中位数阈值_20与50等分_Jupyter源码.py`;
const pythonExe =
  "C:/Users/Hello/PycharmProjects/PythonProject/.venv/Scripts/python.exe";

const analysis = JSON.parse(
  execFileSync(pythonExe, [sourcePath, "--json"], {
    encoding: "utf8",
    maxBuffer: 40 * 1024 * 1024,
    env: {
      ...process.env,
      PYTHONUTF8: "1",
      PYTHONIOENCODING: "utf-8",
    },
  }),
);

const datasetSpecs = [
  { key: "global", sheetName: "全体样本", scope: "全体Parquet数据" },
  { key: "before", sheetName: "14:30以前", scope: "Datetime < 14:30" },
  { key: "after", sheetName: "14:30及以后", scope: "Datetime >= 14:30" },
];
const divisionsList = [20, 50];

if (analysis.before_cutoff_n + analysis.after_cutoff_n !== analysis.total_n) {
  throw new Error("14:30前后样本数不能加回全体样本数。 ");
}

for (const spec of datasetSpecs) {
  const dataset = analysis.datasets[spec.key];
  for (const divisions of divisionsList) {
    const result = dataset.analyses[String(divisions)];
    const half = divisions / 2;
    const counts = result.rows.map((row) => Number(row.sample_n));
    if (result.rows.length !== divisions) {
      throw new Error(`${spec.sheetName} ${divisions}等分行数不正确。`);
    }
    if (counts.reduce((sum, value) => sum + value, 0) !== dataset.total_n) {
      throw new Error(`${spec.sheetName} ${divisions}等分样本合计不正确。`);
    }
    if (Math.max(...counts) - Math.min(...counts) > 1) {
      throw new Error(`${spec.sheetName} ${divisions}等分样本不均衡。`);
    }
    if (result.short_n !== dataset.total_n / 2 || result.long_n !== dataset.total_n / 2) {
      throw new Error(`${spec.sheetName} ${divisions}等分没有按中位数对半。`);
    }
    if (!result.rows.slice(0, half).every((row) => row.direction === "做空")) {
      throw new Error(`${spec.sheetName} ${divisions}等分左侧方向错误。`);
    }
    if (!result.rows.slice(half).every((row) => row.direction === "做多")) {
      throw new Error(`${spec.sheetName} ${divisions}等分右侧方向错误。`);
    }
    if (
      Math.abs(result.rows[half - 1].upper_boundary - dataset.threshold) > 1e-15 ||
      Math.abs(result.rows[half].lower_boundary - dataset.threshold) > 1e-15
    ) {
      throw new Error(`${spec.sheetName} ${divisions}等分中心边界错误。`);
    }
    if (spec.key === "global") {
      for (const row of result.rows) {
        if (row.before_sample_n + row.after_sample_n !== row.sample_n) {
          throw new Error(`${divisions}等分 ${row.bucket_label} 分时样本合计错误。`);
        }
      }
    }
  }
}

const workbook = Workbook.create();
const navy = "#17365D";
const blue = "#4472C4";
const lightBlue = "#D9EAF7";
const lightGray = "#F2F2F2";
const border = "#D9E1F2";
const yellow = "#FFF2CC";
const shortLight = "#FCE4D6";
const longLight = "#E2F0D9";
const purpleLight = "#E4DFEC";
const bodyFont = "Microsoft YaHei";

function styleTitle(sheet, range, title) {
  sheet.mergeCells(range);
  sheet.getRange(range.split(":")[0]).values = [[title]];
  sheet.getRange(range).format = {
    fill: navy,
    font: { name: bodyFont, size: 16, bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
  };
  sheet.getRange(range).format.rowHeight = 32;
}

function styleSection(sheet, row, title) {
  const range = `A${row}:K${row}`;
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

function styleHeader(sheet, range, fill = blue) {
  sheet.getRange(range).format = {
    fill,
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
  sheet.getRange(labelRange.split(":")[0]).values = [[label]];
  if (typeof value === "string" && value.startsWith("=")) {
    sheet.getRange(valueRange.split(":")[0]).formulas = [[value]];
  } else {
    sheet.getRange(valueRange.split(":")[0]).values = [[value]];
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
    font: { name: bodyFont, size: 14, bold: true, color: navy },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    numberFormat,
    borders: { preset: "outside", style: "thin", color: border },
  };
}

function addDirectionFormatting(sheet, startRow, endRow) {
  sheet.getRange(`C${startRow}:C${endRow}`).conditionalFormats.add("containsText", {
    text: "做空",
    format: { fill: shortLight, font: { color: "#9C5700", bold: true } },
  });
  sheet.getRange(`C${startRow}:C${endRow}`).conditionalFormats.add("containsText", {
    text: "做多",
    format: { fill: longLight, font: { color: "#375623", bold: true } },
  });
}

function addMetricScales(sheet, columns, startRow, endRow) {
  sheet.getRange(`${columns.avg}${startRow}:${columns.avg}${endRow}`).conditionalFormats.add("colorScale", {
    colors: ["#F4CCCC", "#FFFFFF", "#E2F0D9"],
    thresholds: ["min", "50%", "max"],
  });
  sheet.getRange(`${columns.win}${startRow}:${columns.win}${endRow}`).conditionalFormats.add("colorScale", {
    colors: ["#F4CCCC", "#FFF2CC", "#E2F0D9"],
    thresholds: ["min", "50%", "max"],
  });
  sheet.getRange(`${columns.ratio}${startRow}:${columns.ratio}${endRow}`).conditionalFormats.add("colorScale", {
    colors: ["#F4CCCC", "#FFF2CC", "#E2F0D9"],
    thresholds: ["min", "50%", "max"],
  });
}

function writeGlobalTable(sheet, dataset, divisions, groupRow, headerRow, dataStart) {
  const result = dataset.analyses[String(divisions)];
  const dataEnd = dataStart + divisions - 1;
  const half = divisions / 2;

  sheet.mergeCells(`A${groupRow}:C${groupRow}`);
  sheet.mergeCells(`D${groupRow}:G${groupRow}`);
  sheet.mergeCells(`H${groupRow}:K${groupRow}`);
  sheet.getRange(`A${groupRow}`).values = [["档位与方向"]];
  sheet.getRange(`D${groupRow}`).values = [["14:30以前"]];
  sheet.getRange(`H${groupRow}`).values = [["14:30及以后"]];
  styleHeader(sheet, `A${groupRow}:C${groupRow}`, navy);
  styleHeader(sheet, `D${groupRow}:G${groupRow}`, "#C65911");
  styleHeader(sheet, `H${groupRow}:K${groupRow}`, "#548235");
  sheet.getRange(`A${groupRow}:K${groupRow}`).format.rowHeight = 22;

  sheet.getRange(`A${headerRow}:K${headerRow}`).values = [[
    "档位",
    "概率区间",
    "方向",
    "样本量",
    "平均策略收益",
    "胜率",
    "平均盈亏比",
    "样本量",
    "平均策略收益",
    "胜率",
    "平均盈亏比",
  ]];
  styleHeader(sheet, `A${headerRow}:C${headerRow}`, blue);
  styleHeader(sheet, `D${headerRow}:G${headerRow}`, "#ED7D31");
  styleHeader(sheet, `H${headerRow}:K${headerRow}`, "#70AD47");

  const rows = result.rows.map((row) => [
    row.bucket_label,
    row.interval,
    row.direction,
    Number(row.before_sample_n),
    row.before_average_strategy_return,
    row.before_win_rate,
    row.before_average_profit_loss_ratio,
    Number(row.after_sample_n),
    row.after_average_strategy_return,
    row.after_win_rate,
    row.after_average_profit_loss_ratio,
  ]);
  sheet.getRange(`A${dataStart}:K${dataEnd}`).values = rows;
  styleData(sheet, `A${dataStart}:K${dataEnd}`);
  const table = sheet.tables.add(
    `A${headerRow}:K${dataEnd}`,
    true,
    `Global${divisions}Table`,
  );
  table.showFilterButton = true;
  sheet.getRange(`D${dataStart}:D${dataEnd}`).format.numberFormat = "#,##0";
  sheet.getRange(`E${dataStart}:E${dataEnd}`).format.numberFormat = "0.0000%;[Red]-0.0000%;-";
  sheet.getRange(`F${dataStart}:F${dataEnd}`).format.numberFormat = "0.00%";
  sheet.getRange(`G${dataStart}:G${dataEnd}`).format.numberFormat = "0.00x";
  sheet.getRange(`H${dataStart}:H${dataEnd}`).format.numberFormat = "#,##0";
  sheet.getRange(`I${dataStart}:I${dataEnd}`).format.numberFormat = "0.0000%;[Red]-0.0000%;-";
  sheet.getRange(`J${dataStart}:J${dataEnd}`).format.numberFormat = "0.00%";
  sheet.getRange(`K${dataStart}:K${dataEnd}`).format.numberFormat = "0.00x";
  addDirectionFormatting(sheet, dataStart, dataEnd);
  addMetricScales(sheet, { avg: "E", win: "F", ratio: "G" }, dataStart, dataEnd);
  addMetricScales(sheet, { avg: "I", win: "J", ratio: "K" }, dataStart, dataEnd);
  sheet.getRange(`A${dataStart + half - 1}:K${dataStart + half}`).format.borders = {
    top: { style: "medium", color: "#806000" },
    bottom: { style: "medium", color: "#806000" },
  };
  return { dataEnd };
}

function writeSubsetTable(sheet, dataset, divisions, groupRow, headerRow, dataStart, tablePrefix) {
  const result = dataset.analyses[String(divisions)];
  const dataEnd = dataStart + divisions - 1;
  const half = divisions / 2;
  sheet.mergeCells(`A${groupRow}:G${groupRow}`);
  sheet.getRange(`A${groupRow}`).values = [[`${dataset.label}：使用本数据集自己的中位数`]];
  styleHeader(sheet, `A${groupRow}:G${groupRow}`, navy);
  sheet.getRange(`A${groupRow}:G${groupRow}`).format.rowHeight = 22;
  sheet.getRange(`A${headerRow}:G${headerRow}`).values = [[
    "档位",
    "概率区间",
    "方向",
    "样本量",
    "平均策略收益",
    "胜率",
    "平均盈亏比",
  ]];
  styleHeader(sheet, `A${headerRow}:G${headerRow}`);
  const rows = result.rows.map((row) => [
    row.bucket_label,
    row.interval,
    row.direction,
    Number(row.sample_n),
    row.average_strategy_return,
    row.win_rate,
    row.average_profit_loss_ratio,
  ]);
  sheet.getRange(`A${dataStart}:G${dataEnd}`).values = rows;
  styleData(sheet, `A${dataStart}:G${dataEnd}`);
  const table = sheet.tables.add(
    `A${headerRow}:G${dataEnd}`,
    true,
    `${tablePrefix}${divisions}Table`,
  );
  table.showFilterButton = true;
  sheet.getRange(`D${dataStart}:D${dataEnd}`).format.numberFormat = "#,##0";
  sheet.getRange(`E${dataStart}:E${dataEnd}`).format.numberFormat = "0.0000%;[Red]-0.0000%;-";
  sheet.getRange(`F${dataStart}:F${dataEnd}`).format.numberFormat = "0.00%";
  sheet.getRange(`G${dataStart}:G${dataEnd}`).format.numberFormat = "0.00x";
  addDirectionFormatting(sheet, dataStart, dataEnd);
  addMetricScales(sheet, { avg: "E", win: "F", ratio: "G" }, dataStart, dataEnd);
  sheet.getRange(`A${dataStart + half - 1}:G${dataStart + half}`).format.borders = {
    top: { style: "medium", color: "#806000" },
    bottom: { style: "medium", color: "#806000" },
  };
  return { dataEnd };
}

function buildDatasetSheet(spec) {
  const dataset = analysis.datasets[spec.key];
  const sheet = workbook.worksheets.add(spec.sheetName);
  sheet.showGridLines = false;
  styleTitle(sheet, "A1:K1", `${spec.sheetName}：独立中位数阈值分析`);
  sheet.mergeCells("A2:K2");
  sheet.getRange("A2").values = [[
    spec.key === "global"
      ? `全体样本使用全体中位数 ${dataset.threshold.toFixed(12)}；表内直接比较14:30以前和14:30及以后。两个时间段均使用同一个全体中位数和相同区间边界。`
      : `${spec.scope}，先独立筛选数据，再以本数据集自己的中位数 ${dataset.threshold.toFixed(12)} 为方向阈值；阈值左侧做空取-Tag，右侧做多取Tag。`,
  ]];
  sheet.getRange("A2:K2").format = {
    fill: lightGray,
    font: { name: bodyFont, size: 9 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange("A2:K2").format.rowHeight = 38;

  addCard(sheet, "A4:B4", "A5:B6", "数据范围", spec.sheetName, lightBlue, "@");
  addCard(sheet, "C4:D4", "C5:D6", "独立中位数", dataset.threshold, purpleLight, "0.000000");
  addCard(sheet, "E4:F4", "E5:F6", "样本量", dataset.total_n, longLight, "#,##0");
  addCard(sheet, "G4:H4", "G5:H6", "20等分每档", dataset.analyses["20"].min_bucket_n, lightBlue, "#,##0");
  addCard(sheet, "I4:K4", "I5:K6", "50等分每档", dataset.analyses["50"].min_bucket_n, yellow, "#,##0");

  styleSection(sheet, 8, "20等分");
  const first = spec.key === "global"
    ? writeGlobalTable(sheet, dataset, 20, 9, 10, 11)
    : writeSubsetTable(sheet, dataset, 20, 9, 10, 11, spec.key === "before" ? "Before" : "After");

  const secondSection = first.dataEnd + 2;
  styleSection(sheet, secondSection, "50等分");
  const secondGroup = secondSection + 1;
  const secondHeader = secondSection + 2;
  const secondStart = secondSection + 3;
  const second = spec.key === "global"
    ? writeGlobalTable(sheet, dataset, 50, secondGroup, secondHeader, secondStart)
    : writeSubsetTable(
        sheet,
        dataset,
        50,
        secondGroup,
        secondHeader,
        secondStart,
        spec.key === "before" ? "Before" : "After",
      );

  const conclusionTitle = second.dataEnd + 2;
  styleSection(sheet, conclusionTitle, "关键结论");
  const rows50 = dataset.analyses["50"].rows;
  const ranked50 = [...rows50].sort(
    (a, b) => b.average_strategy_return - a.average_strategy_return,
  );
  const conclusions = [
    `1. ${spec.sheetName}共 ${Number(dataset.total_n).toLocaleString("zh-CN")} 条，独立中位数为 ${dataset.threshold.toFixed(6)}。`,
    `2. 20等分每档 ${Number(dataset.analyses["20"].min_bucket_n).toLocaleString("zh-CN")} 条；50等分每档 ${Number(dataset.analyses["50"].min_bucket_n).toLocaleString("zh-CN")} 条；所有档位最大样本差均为0。`,
    `3. 50等分平均策略收益最高为 ${ranked50[0].bucket_label}（${ranked50[0].interval}），最低为 ${ranked50[ranked50.length - 1].bucket_label}（${ranked50[ranked50.length - 1].interval}）。`,
    spec.key === "global"
      ? `4. 全体表内两组分时样本分别为14:30以前 ${Number(analysis.before_cutoff_n).toLocaleString("zh-CN")} 条、14:30及以后 ${Number(analysis.after_cutoff_n).toLocaleString("zh-CN")} 条。`
      : `4. 本表中位数只由${spec.sheetName}数据计算，不使用全体样本中位数。`,
  ];
  conclusions.forEach((line, index) => {
    const row = conclusionTitle + 1 + index;
    sheet.mergeCells(`A${row}:K${row}`);
    sheet.getRange(`A${row}`).values = [[line]];
    sheet.getRange(`A${row}:K${row}`).format = {
      fill: index % 2 === 0 ? "#FFFFFF" : lightGray,
      font: { name: bodyFont, size: 9 },
      wrapText: true,
      verticalAlignment: "center",
      borders: { bottom: { style: "thin", color: border } },
    };
    sheet.getRange(`A${row}:K${row}`).format.rowHeight = 28;
  });

  const finalRow = conclusionTitle + conclusions.length;
  sheet.freezePanes.freezeRows(10);
  const widths = spec.key === "global"
    ? [10, 27, 11, 13, 17, 13, 15, 13, 17, 13, 15]
    : [10, 27, 11, 13, 17, 13, 15, 4, 4, 4, 4];
  widths.forEach((width, index) => {
    sheet.getRangeByIndexes(0, index, finalRow, 1).format.columnWidth = width;
  });
  return { sheetName: spec.sheetName, finalRow };
}

const layouts = datasetSpecs.map(buildDatasetSheet);
await fs.mkdir(previewDir, { recursive: true });

for (const layout of layouts) {
  const previewFileName = layout.sheetName.replaceAll(":", "_");
  const preview = await workbook.render({
    sheetName: layout.sheetName,
    range: `A1:K${layout.finalRow}`,
    scale: layout.sheetName === "全体样本" ? 0.82 : 0.92,
    format: "png",
  });
  await fs.writeFile(
    `${previewDir}/${previewFileName}.png`,
    new Uint8Array(await preview.arrayBuffer()),
  );
}

for (const layout of layouts) {
  const check = await workbook.inspect({
    kind: "table",
    range: `${layout.sheetName}!A1:K${Math.min(layout.finalRow, 45)}`,
    include: "values,formulas",
    tableMaxRows: 45,
    tableMaxCols: 11,
    maxChars: 8000,
  });
  console.log(check.ndjson);
}

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
    globalMedian: analysis.datasets.global.threshold,
    beforeMedian: analysis.datasets.before.threshold,
    afterMedian: analysis.datasets.after.threshold,
    globalN: analysis.datasets.global.total_n,
    beforeN: analysis.datasets.before.total_n,
    afterN: analysis.datasets.after.total_n,
    sheets: layouts.map((layout) => layout.sheetName),
  })}`,
);
console.log(`OUTPUT=${outputPath}`);
