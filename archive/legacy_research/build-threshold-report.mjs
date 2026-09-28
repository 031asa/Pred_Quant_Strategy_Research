import fs from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "C:/Users/Hello/Documents/基础环境配置/outputs/019fb20f-2fd0-7341-9e50-cb40298a48ce";
const outputPath = `${outputDir}/全局阈值_ROC_AUC分析.xlsx`;
const previewDir = `${outputDir}/previews`;
const pythonExe = "C:/Users/Hello/PycharmProjects/PythonProject/.venv/Scripts/python.exe";

const analysisCode = String.raw`
from pathlib import Path
import json
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, roc_curve

path = Path.home() / "Downloads" / "pred_eval.parquet"
df = (
    pl.read_parquet(path)
    .select(["Date", "pred", "Tag"])
    .drop_nulls()
    .sort("Date")
)

dates = df["Date"].unique().sort()
split_index = int(len(dates) * 0.70)
split_date = dates[split_index]

segments = {
    "full": df,
    "validation": df.filter(pl.col("Date") < split_date),
    "test": df.filter(pl.col("Date") >= split_date),
}

thresholds = [round(0.40 + i * 0.01, 2) for i in range(21)]
result = {
    "source": str(path),
    "split_date": str(split_date),
    "total_dates": len(dates),
    "segments": {},
}

roc_raw = {}

for name, frame in segments.items():
    y = (frame["Tag"].to_numpy() > 0).astype(np.int8)
    p = np.clip(frame["pred"].to_numpy().astype(float) + 0.5, 0.0, 1.0)
    auc = float(roc_auc_score(y, p))
    fpr_curve, tpr_curve, _ = roc_curve(y, p)
    roc_raw[name] = (fpr_curve, tpr_curve)

    rows = []
    for threshold in thresholds:
        pred_class = p >= threshold
        positive = y == 1
        negative = ~positive
        tp = int(np.sum(pred_class & positive))
        fp = int(np.sum(pred_class & negative))
        tn = int(np.sum((~pred_class) & negative))
        fn = int(np.sum((~pred_class) & positive))
        rows.append({
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        })

    result["segments"][name] = {
        "auc": auc,
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "rows": rows,
    }

grid = np.linspace(0.0, 1.0, 101)
result["roc_grid"] = [
    {
        "fpr": float(x),
        "full_tpr": float(np.interp(x, roc_raw["full"][0], roc_raw["full"][1])),
        "validation_tpr": float(np.interp(x, roc_raw["validation"][0], roc_raw["validation"][1])),
        "test_tpr": float(np.interp(x, roc_raw["test"][0], roc_raw["test"][1])),
        "chance_tpr": float(x),
    }
    for x in grid
]

print(json.dumps(result, ensure_ascii=False))
`;

const analysis = JSON.parse(
  execFileSync(pythonExe, ["-c", analysisCode], {
    encoding: "utf8",
    maxBuffer: 20 * 1024 * 1024,
  }),
);

const workbook = Workbook.create();
const summary = workbook.worksheets.add("汇总");
const fullSheet = workbook.worksheets.add("全样本阈值表");
const validationSheet = workbook.worksheets.add("验证集阈值表");
const testSheet = workbook.worksheets.add("测试集阈值表");
const rocSheet = workbook.worksheets.add("ROC曲线数据");

const navy = "#17365D";
const blue = "#4472C4";
const lightBlue = "#D9EAF7";
const lightGray = "#F2F2F2";
const border = "#D9E1F2";
const green = "#E2F0D9";
const yellow = "#FFF2CC";
const red = "#FCE4D6";

for (const sheet of [summary, fullSheet, validationSheet, testSheet, rocSheet]) {
  sheet.showGridLines = false;
}

// 汇总页
summary.mergeCells("A1:J1");
summary.getRange("A1").values = [["全局分类阈值、ROC 与 AUC 分析"]];
summary.getRange("A1:J1").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
summary.getRange("A1:J1").format.rowHeight = 32;

summary.getRange("A3:B6").values = [
  ["数据源", analysis.source],
  ["评分定义", "prob_up = pred + 0.5"],
  ["真实标签", "y_true = 1(Tag > 0)"],
  ["时间切分", `前70%日期为验证段；${analysis.split_date}起为测试段`],
];
summary.getRange("A3:A6").format = {
  fill: lightBlue,
  font: { bold: true },
};
summary.getRange("B3:B6").format.wrapText = true;

summary.mergeCells("A8:D8");
summary.getRange("A8").values = [["整体区分能力"]];
summary.getRange("A8:D8").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF" },
};
summary.getRange("A9:D12").values = [
  ["数据段", "AUC", "样本量", "正样本率"],
  ["全样本", analysis.segments.full.auc, analysis.segments.full.n, analysis.segments.full.positive_rate],
  ["前70%验证段", analysis.segments.validation.auc, analysis.segments.validation.n, analysis.segments.validation.positive_rate],
  ["后30%测试段", analysis.segments.test.auc, analysis.segments.test.n, analysis.segments.test.positive_rate],
];
summary.getRange("A9:D9").format = {
  fill: blue,
  font: { bold: true, color: "#FFFFFF" },
};
summary.getRange("B10:B12").format.numberFormat = "0.0000";
summary.getRange("C10:C12").format.numberFormat = "#,##0";
summary.getRange("D10:D12").format.numberFormat = "0.00%";

summary.mergeCells("A14:J14");
summary.getRange("A14").values = [["口径说明与结论"]];
summary.getRange("A14:J14").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF" },
};
summary.mergeCells("A15:J15");
summary.mergeCells("A16:J16");
summary.mergeCells("A17:J17");
summary.mergeCells("A18:J18");
summary.getRange("A15:J18").values = [
  ["AUC评价整条ROC曲线，属于评分排序能力；同一数据段内不会因单个阈值改变。"],
  ["阈值决定ROC曲线上的具体操作点：TPR（召回率）与FPR（误报率）。"],
  [`全样本AUC=${analysis.segments.full.auc.toFixed(4)}；验证段AUC=${analysis.segments.validation.auc.toFixed(4)}；测试段AUC=${analysis.segments.test.auc.toFixed(4)}。`],
  ["测试段AUC更接近0.5，说明后段区分能力较弱；阈值比较应优先查看“测试集阈值表”。"],
];
summary.getRange("A15:J18").format = {
  fill: lightGray,
  wrapText: true,
  font: { size: 10 },
};
summary.getRange("A15:J18").format.rowHeight = 24;

// ROC数据和图
const rocRows = [
  ["FPR", "全样本TPR", "验证段TPR", "测试段TPR", "随机基准TPR"],
  ...analysis.roc_grid.filter((_, i) => i % 5 === 0).map((r) => [
    `${Math.round(r.fpr * 100)}%`,
    r.full_tpr,
    r.validation_tpr,
    r.test_tpr,
    r.chance_tpr,
  ]),
];
rocSheet.getRange(`A1:E${rocRows.length}`).values = rocRows;
rocSheet.getRange("A1:E1").format = {
  fill: blue,
  font: { bold: true, color: "#FFFFFF" },
};
rocSheet.getRange(`B2:E${rocRows.length}`).format.numberFormat = "0.00%";
rocSheet.freezePanes.freezeRows(1);
rocSheet.getRange("A:E").format.columnWidth = 16;

const rocChart = summary.charts.add("line", rocSheet.getRange(`A1:E${rocRows.length}`));
rocChart.title = "全局ROC曲线：测试段接近随机基准";
rocChart.titleTextStyle.fontSize = 13;
rocChart.hasLegend = true;
rocChart.xAxis = {
  axisType: "textAxis",
  title: { text: "假阳性率 FPR" },
  numberFormatCode: "0%",
};
rocChart.yAxis = {
  title: { text: "真阳性率 TPR" },
  numberFormatCode: "0%",
  min: 0,
  max: 1,
};
rocChart.setPosition("A20", "J42");

function buildThresholdSheet(sheet, title, segmentKey, summaryAucCell) {
  const segment = analysis.segments[segmentKey];
  sheet.mergeCells("A1:P1");
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1:P1").format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 15 },
    verticalAlignment: "center",
  };
  sheet.getRange("A1:P1").format.rowHeight = 30;

  sheet.mergeCells("A2:P2");
  sheet.getRange("A2").values = [[
    `阈值范围0.40–0.60，步长0.01；样本量${segment.n.toLocaleString()}。AUC为整条ROC曲线面积，因此本表各阈值行的AUC相同。`,
  ]];
  sheet.getRange("A2:P2").format = {
    fill: lightGray,
    wrapText: true,
    font: { size: 10 },
  };
  sheet.getRange("A2:P2").format.rowHeight = 28;

  const headers = [
    "概率阈值",
    "原始pred分界点",
    "AUC",
    "TP",
    "FP",
    "TN",
    "FN",
    "TPR/召回率",
    "FPR",
    "特异度",
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "F1",
    "预测正类比例",
    "Youden J",
  ];
  sheet.getRange("A4:P4").values = [headers];
  sheet.getRange("A4:P4").format = {
    fill: blue,
    font: { bold: true, color: "#FFFFFF", size: 10 },
    wrapText: true,
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: border },
  };
  sheet.getRange("A4:P4").format.rowHeight = 34;

  const startRow = 5;
  const endRow = startRow + segment.rows.length - 1;
  const valueRows = segment.rows.map((r) => [
    r.threshold,
    null,
    null,
    r.tp,
    r.fp,
    r.tn,
    r.fn,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
  ]);
  sheet.getRange(`A${startRow}:P${endRow}`).values = valueRows;

  const formulaRows = segment.rows.map((_, i) => {
    const row = startRow + i;
    return [
      `=A${row}-0.5`,
      `='汇总'!${summaryAucCell}`,
      `=D${row}/(D${row}+G${row})`,
      `=E${row}/(E${row}+F${row})`,
      `=F${row}/(E${row}+F${row})`,
      `=(D${row}+F${row})/SUM(D${row}:G${row})`,
      `=(H${row}+J${row})/2`,
      `=D${row}/(D${row}+E${row})`,
      `=2*M${row}*H${row}/(M${row}+H${row})`,
      `=(D${row}+E${row})/SUM(D${row}:G${row})`,
      `=H${row}-I${row}`,
    ];
  });
  for (let i = 0; i < formulaRows.length; i++) {
    const row = startRow + i;
    sheet.getRange(`B${row}:C${row}`).formulas = [[formulaRows[i][0], formulaRows[i][1]]];
    sheet.getRange(`H${row}:P${row}`).formulas = [[
      formulaRows[i][2],
      formulaRows[i][3],
      formulaRows[i][4],
      formulaRows[i][5],
      formulaRows[i][6],
      formulaRows[i][7],
      formulaRows[i][8],
      formulaRows[i][9],
      formulaRows[i][10],
    ]];
  }

  sheet.getRange(`A${startRow}:C${endRow}`).format.numberFormat = "0.0000";
  sheet.getRange(`D${startRow}:G${endRow}`).format.numberFormat = "#,##0";
  sheet.getRange(`H${startRow}:P${endRow}`).format.numberFormat = "0.00%";
  sheet.getRange(`C${startRow}:C${endRow}`).format.font = { color: "#008000" };
  sheet.getRange(`A${startRow}:P${endRow}`).format.borders = {
    insideHorizontal: { style: "thin", color: border },
    bottom: { style: "thin", color: border },
  };

  sheet.getRange(`A${startRow}:P${endRow}`).conditionalFormats.addCustom(
    `$A${startRow}=0.5`,
    { fill: yellow, font: { bold: true } },
  );
  sheet.getRange(`L${startRow}:L${endRow}`).conditionalFormats.add("colorScale", {
    colors: ["#FCE4D6", "#FFF2CC", "#E2F0D9"],
    thresholds: ["min", "50%", "max"],
  });

  sheet.freezePanes.freezeRows(4);
  sheet.getRange("A:P").format.font = { name: "Aptos", size: 10 };
  sheet.getRange("A:A").format.columnWidth = 12;
  sheet.getRange("B:B").format.columnWidth = 16;
  sheet.getRange("C:C").format.columnWidth = 10;
  sheet.getRange("D:G").format.columnWidth = 11;
  sheet.getRange("H:P").format.columnWidth = 16;
}

buildThresholdSheet(fullSheet, "全样本：全局阈值敏感性表", "full", "$B$10");
buildThresholdSheet(validationSheet, "前70%验证段：全局阈值敏感性表", "validation", "$B$11");
buildThresholdSheet(testSheet, "后30%测试段：全局阈值敏感性表", "test", "$B$12");

summary.getRange("A:J").format.font = { name: "Aptos", size: 10 };
summary.getRange("A:A").format.columnWidth = 22;
summary.getRange("B:B").format.columnWidth = 44;
summary.getRange("C:D").format.columnWidth = 16;
summary.getRange("E:J").format.columnWidth = 14;
summary.freezePanes.freezeRows(1);

await fs.mkdir(previewDir, { recursive: true });

const summaryPreview = await workbook.render({
  sheetName: "汇总",
  range: "A1:J42",
  scale: 1.2,
  format: "png",
});
await fs.writeFile(
  `${previewDir}/summary.png`,
  new Uint8Array(await summaryPreview.arrayBuffer()),
);

for (const sheetName of ["全样本阈值表", "验证集阈值表", "测试集阈值表"]) {
  const preview = await workbook.render({
    sheetName,
    range: "A1:P25",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    `${previewDir}/${sheetName}.png`,
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const tableCheck = await workbook.inspect({
  kind: "table",
  range: "测试集阈值表!A1:P25",
  include: "values,formulas",
  tableMaxRows: 25,
  tableMaxCols: 16,
});
console.log(tableCheck.ndjson);

const errorCheck = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errorCheck.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`OUTPUT=${outputPath}`);
