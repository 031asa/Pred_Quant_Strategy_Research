import fs from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir =
  "C:/Users/Hello/Documents/基础环境配置/outputs/019fb20f-2fd0-7341-9e50-cb40298a48ce";
const outputPath = `${outputDir}/按合约_按月份_阈值精确度分析.xlsx`;
const previewDir = `${outputDir}/segmented-previews`;
const configPath = `${outputDir}/阈值分析参数.json`;
const duckdbExe = "D:/DevTools/DuckDB/duckdb.exe";
const config = JSON.parse(await fs.readFile(configPath, "utf8"));

const navy = "#17365D";
const blue = "#4472C4";
const lightBlue = "#D9EAF7";
const lightGray = "#F2F2F2";
const border = "#D9E1F2";
const yellow = "#FFF2CC";
const shortColor = "#C65911";
const longColor = "#548235";
const shortLight = "#FCE4D6";
const longLight = "#E2F0D9";
const redLight = "#F4CCCC";

function roundStable(value) {
  return Math.round(value * 1e12) / 1e12;
}

function buildThresholds(configValue) {
  const denseLeft =
    configValue.direction_threshold - configValue.dense_half_width;
  const denseRight =
    configValue.direction_threshold + configValue.dense_half_width;
  const values = [
    configValue.score_min,
    denseLeft,
    configValue.direction_threshold,
    denseRight,
    configValue.score_max,
  ];
  for (
    let value = denseLeft;
    value <= denseRight + 1e-12;
    value += configValue.dense_step
  ) {
    values.push(roundStable(value));
  }
  return [...new Set(values.map(roundStable))].sort((a, b) => a - b);
}

function pointLabel(value) {
  const scaled = value * 100;
  const rounded = Math.round(scaled);
  return Math.abs(scaled - rounded) < 1e-9
    ? rounded
    : Math.round(scaled * 1e6) / 1e6;
}

function buildIntervalSpecs(thresholds, configValue) {
  const left = [];
  const right = [];
  const center = configValue.direction_threshold;
  for (let index = 0; index < thresholds.length - 1; index++) {
    const lower = thresholds[index];
    const upper = thresholds[index + 1];
    if (upper <= center + 1e-12) {
      left.push({
        label: `[${pointLabel(lower)},${pointLabel(upper)})`,
        direction: "short",
        lower,
        upper,
      });
    } else if (lower >= center - 1e-12) {
      right.push({
        label: `(${pointLabel(lower)},${pointLabel(upper)}]`,
        direction: "long",
        lower,
        upper,
      });
    } else {
      throw new Error("区间网格跨越方向阈值，请检查参数。");
    }
  }
  return [
    ...left,
    {
      label: String(pointLabel(center)),
      direction: "neutral",
      lower: center,
      upper: center,
    },
    ...right,
  ];
}

function quoteIdentifier(identifier) {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(identifier)) {
    throw new Error(`不安全的字段名：${identifier}`);
  }
  return `"${identifier}"`;
}

function sqlString(value) {
  return `'${String(value).replaceAll("'", "''")}'`;
}

function buildAnalysis(configValue) {
  const thresholds = buildThresholds(configValue);
  const intervalSpecs = buildIntervalSpecs(thresholds, configValue);
  const thresholdValues = thresholds
    .map((value, index) => `(${index}, ${value})`)
    .join(",\n");
  const intervalValues = intervalSpecs
    .map(
      (spec, index) =>
        `(${index}, ${sqlString(spec.label)}, ${sqlString(spec.direction)}, ${spec.lower}, ${spec.upper})`,
    )
    .join(",\n");

  const source = sqlString(configValue.source_path.replaceAll("\\", "/"));
  const predColumn = quoteIdentifier(configValue.pred_column);
  const returnColumn = quoteIdentifier(configValue.return_column);
  const dateColumn = quoteIdentifier(configValue.date_column);
  const contractColumn = quoteIdentifier(configValue.contract_column);
  const center = configValue.direction_threshold;

  const sql = `
WITH
base AS MATERIALIZED (
  SELECT
    CAST(${contractColumn} AS VARCHAR) AS contract_group,
    strftime(${dateColumn}, '%Y-%m') AS month_group,
    LEAST(
      GREATEST(CAST(${predColumn} AS DOUBLE) + ${configValue.score_offset}, ${configValue.score_min}),
      ${configValue.score_max}
    ) AS score,
    CAST(${returnColumn} AS DOUBLE) AS return_value
  FROM read_parquet(${source})
  WHERE
    ${contractColumn} IS NOT NULL
    AND ${dateColumn} IS NOT NULL
    AND ${predColumn} IS NOT NULL
    AND ${returnColumn} IS NOT NULL
),
expanded AS MATERIALIZED (
  SELECT 'contract' AS dimension, contract_group AS group_name, score, return_value FROM base
  UNION ALL
  SELECT 'month' AS dimension, month_group AS group_name, score, return_value FROM base
  UNION ALL
  SELECT 'global' AS dimension, '全体' AS group_name, score, return_value FROM base
),
thresholds(idx, threshold) AS (
  VALUES
  ${thresholdValues}
),
intervals(idx, label, direction, lower_bound, upper_bound) AS (
  VALUES
  ${intervalValues}
),
moving AS (
  SELECT
    'moving' AS record_type,
    e.dimension,
    e.group_name,
    t.idx,
    CAST(t.threshold * 100 AS DOUBLE) AS label,
    count(*) AS n,
    count(*) FILTER (WHERE e.return_value > 0) AS positive_n,
    count(*) FILTER (WHERE e.return_value < 0) AS negative_n,
    (
      count(*) FILTER (WHERE e.score > t.threshold AND e.return_value > 0)
    )::DOUBLE
      / NULLIF(count(*) FILTER (WHERE e.score > t.threshold), 0) AS long_precision,
    (
      count(*) FILTER (WHERE e.score < t.threshold AND e.return_value < 0)
    )::DOUBLE
      / NULLIF(count(*) FILTER (WHERE e.score < t.threshold), 0) AS short_precision,
    (
      count(*) FILTER (WHERE e.score > t.threshold)
    )::DOUBLE / count(*) AS long_coverage,
    (
      count(*) FILTER (WHERE e.score < t.threshold)
    )::DOUBLE / count(*) AS short_coverage
  FROM expanded e
  CROSS JOIN thresholds t
  GROUP BY e.dimension, e.group_name, t.idx, t.threshold
),
group_names AS MATERIALIZED (
  SELECT DISTINCT dimension, group_name FROM expanded
),
interval_metrics AS (
  SELECT
    'interval' AS record_type,
    g.dimension,
    g.group_name,
    i.idx,
    i.label,
    avg(
      CASE
        WHEN i.direction = 'short' THEN -e.return_value
        WHEN i.direction = 'long' THEN e.return_value
        ELSE NULL
      END
    ) AS average_strategy_return,
    (
      count(*) FILTER (
        WHERE
          (i.direction = 'short' AND -e.return_value > 0)
          OR (i.direction = 'long' AND e.return_value > 0)
      )
    )::DOUBLE
      / NULLIF(
          count(*) FILTER (
            WHERE
              (i.direction = 'short' AND -e.return_value <> 0)
              OR (i.direction = 'long' AND e.return_value <> 0)
          ),
          0
        ) AS interval_win_rate,
    avg(
      CASE
        WHEN i.direction = 'short' AND -e.return_value > 0 THEN -e.return_value
        WHEN i.direction = 'long' AND e.return_value > 0 THEN e.return_value
        ELSE NULL
      END
    )
      / NULLIF(
          abs(
            avg(
              CASE
                WHEN i.direction = 'short' AND -e.return_value < 0 THEN -e.return_value
                WHEN i.direction = 'long' AND e.return_value < 0 THEN e.return_value
                ELSE NULL
              END
            )
          ),
          0
        ) AS average_profit_loss_ratio,
    count(e.score) AS interval_sample_n
  FROM group_names g
  CROSS JOIN intervals i
  LEFT JOIN expanded e
    ON e.dimension = g.dimension
    AND e.group_name = g.group_name
    AND (
      (i.direction = 'short' AND e.score >= i.lower_bound AND e.score < i.upper_bound)
      OR (i.direction = 'long' AND e.score > i.lower_bound AND e.score <= i.upper_bound)
      OR (i.direction = 'neutral' AND e.score = ${center})
    )
  GROUP BY g.dimension, g.group_name, i.idx, i.label
),
strategy_rows AS MATERIALIZED (
  SELECT
    dimension,
    group_name,
    score,
    CASE
      WHEN score > ${center} THEN return_value
      WHEN score < ${center} THEN -return_value
      ELSE NULL
    END AS strategy_return,
    CASE
      WHEN score > ${center} THEN 'long'
      WHEN score < ${center} THEN 'short'
      ELSE 'neutral'
    END AS side
  FROM expanded
),
fixed_summary AS (
  SELECT
    'summary' AS record_type,
    dimension,
    group_name,
    count(*) AS n,
    count(strategy_return) AS fixed_trade_n,
    avg(strategy_return) AS fixed_average_strategy_return,
    (
      count(*) FILTER (WHERE strategy_return > 0)
    )::DOUBLE
      / NULLIF(count(*) FILTER (WHERE strategy_return <> 0), 0) AS fixed_win_rate,
    avg(strategy_return) FILTER (WHERE strategy_return > 0)
      / NULLIF(abs(avg(strategy_return) FILTER (WHERE strategy_return < 0)), 0)
      AS fixed_average_profit_loss_ratio,
    count(*) FILTER (WHERE side = 'long') AS long_trade_n,
    avg(strategy_return) FILTER (WHERE side = 'long') AS long_average_strategy_return,
    (
      count(*) FILTER (WHERE side = 'long' AND strategy_return > 0)
    )::DOUBLE
      / NULLIF(count(*) FILTER (WHERE side = 'long' AND strategy_return <> 0), 0)
      AS long_win_rate,
    avg(strategy_return) FILTER (WHERE side = 'long' AND strategy_return > 0)
      / NULLIF(
          abs(avg(strategy_return) FILTER (WHERE side = 'long' AND strategy_return < 0)),
          0
        ) AS long_average_profit_loss_ratio,
    count(*) FILTER (WHERE side = 'short') AS short_trade_n,
    avg(strategy_return) FILTER (WHERE side = 'short') AS short_average_strategy_return,
    (
      count(*) FILTER (WHERE side = 'short' AND strategy_return > 0)
    )::DOUBLE
      / NULLIF(count(*) FILTER (WHERE side = 'short' AND strategy_return <> 0), 0)
      AS short_win_rate,
    avg(strategy_return) FILTER (WHERE side = 'short' AND strategy_return > 0)
      / NULLIF(
          abs(avg(strategy_return) FILTER (WHERE side = 'short' AND strategy_return < 0)),
          0
        ) AS short_average_profit_loss_ratio
  FROM strategy_rows
  GROUP BY dimension, group_name
)
SELECT * FROM moving
UNION ALL BY NAME
SELECT * FROM interval_metrics
UNION ALL BY NAME
SELECT * FROM fixed_summary
ORDER BY record_type, dimension, group_name, idx;
`;

  const rows = JSON.parse(
    execFileSync(duckdbExe, ["-json", "-c", sql], {
      encoding: "utf8",
      maxBuffer: 100 * 1024 * 1024,
    }),
  );

  const dimensions = {
    contract: new Map(),
    month: new Map(),
    global: new Map(),
  };

  function getGroup(dimension, groupName) {
    const dimensionMap = dimensions[dimension];
    if (!dimensionMap.has(groupName)) {
      dimensionMap.set(groupName, {
        group: groupName,
        n: 0,
        positive_n: 0,
        negative_n: 0,
        long_precision: Array(thresholds.length).fill(null),
        short_precision: Array(thresholds.length).fill(null),
        long_coverage: Array(thresholds.length).fill(null),
        short_coverage: Array(thresholds.length).fill(null),
        average_strategy_return: Array(intervalSpecs.length).fill(null),
        interval_win_rate: Array(intervalSpecs.length).fill(null),
        average_profit_loss_ratio: Array(intervalSpecs.length).fill(null),
        interval_sample_n: Array(intervalSpecs.length).fill(0),
        fixed_summary: null,
      });
    }
    return dimensionMap.get(groupName);
  }

  for (const row of rows) {
    const group = getGroup(row.dimension, row.group_name);
    if (row.record_type === "moving") {
      const index = Number(row.idx);
      group.n = Number(row.n);
      group.positive_n = Number(row.positive_n);
      group.negative_n = Number(row.negative_n);
      group.long_precision[index] = row.long_precision;
      group.short_precision[index] = row.short_precision;
      group.long_coverage[index] = row.long_coverage;
      group.short_coverage[index] = row.short_coverage;
    } else if (row.record_type === "interval") {
      const index = Number(row.idx);
      group.average_strategy_return[index] = row.average_strategy_return;
      group.interval_win_rate[index] = row.interval_win_rate;
      group.average_profit_loss_ratio[index] = row.average_profit_loss_ratio;
      group.interval_sample_n[index] = Number(row.interval_sample_n);
    } else {
      group.n = Number(row.n);
      group.fixed_summary = {
        overall: {
          trade_n: Number(row.fixed_trade_n),
          average_strategy_return: row.fixed_average_strategy_return,
          win_rate: row.fixed_win_rate,
          average_profit_loss_ratio: row.fixed_average_profit_loss_ratio,
        },
        long: {
          trade_n: Number(row.long_trade_n),
          average_strategy_return: row.long_average_strategy_return,
          win_rate: row.long_win_rate,
          average_profit_loss_ratio: row.long_average_profit_loss_ratio,
        },
        short: {
          trade_n: Number(row.short_trade_n),
          average_strategy_return: row.short_average_strategy_return,
          win_rate: row.short_win_rate,
          average_profit_loss_ratio: row.short_average_profit_loss_ratio,
        },
      };
    }
  }

  const contracts = [...dimensions.contract.values()].sort((a, b) =>
    a.group.localeCompare(b.group),
  );
  const months = [...dimensions.month.values()].sort((a, b) =>
    a.group.localeCompare(b.group),
  );
  const global = dimensions.global.get("全体");

  return {
    source: configValue.source_path,
    config: configValue,
    total_n: global.n,
    total_positive_n: global.positive_n,
    total_negative_n: global.negative_n,
    thresholds,
    threshold_labels: thresholds.map(pointLabel),
    interval_specs: intervalSpecs,
    interval_labels: intervalSpecs.map((spec) => spec.label),
    global,
    contracts,
    months,
  };
}

const analysis = buildAnalysis(config);

function buildEquityCurve(configValue) {
  const source = sqlString(configValue.source_path.replaceAll("\\", "/"));
  const predColumn = quoteIdentifier(configValue.pred_column);
  const returnColumn = quoteIdentifier(configValue.return_column);
  const dateColumn = quoteIdentifier(configValue.date_column);
  const center = configValue.direction_threshold;
  const sql = `
WITH base AS (
  SELECT
    CAST(${dateColumn} AS DATE) AS trade_date,
    LEAST(
      GREATEST(CAST(${predColumn} AS DOUBLE) + ${configValue.score_offset}, ${configValue.score_min}),
      ${configValue.score_max}
    ) AS score,
    CAST(${returnColumn} AS DOUBLE) AS return_value
  FROM read_parquet(${source})
  WHERE ${dateColumn} IS NOT NULL
    AND ${predColumn} IS NOT NULL
    AND ${returnColumn} IS NOT NULL
),
daily AS (
  SELECT
    trade_date,
    avg(
      CASE
        WHEN score > ${center} THEN return_value
        WHEN score < ${center} THEN -return_value
        ELSE NULL
      END
    ) AS daily_strategy_return,
    count(*) FILTER (WHERE score <> ${center}) AS trade_n
  FROM base
  GROUP BY trade_date
)
SELECT
  strftime(trade_date, '%Y-%m-%d') AS date_label,
  daily_strategy_return,
  trade_n
FROM daily
WHERE daily_strategy_return IS NOT NULL
ORDER BY trade_date;
`;
  return JSON.parse(
    execFileSync(duckdbExe, ["-json", "-c", sql], {
      encoding: "utf8",
      maxBuffer: 20 * 1024 * 1024,
    }),
  ).map((row) => ({
    date: row.date_label,
    dailyReturn: Number(row.daily_strategy_return),
    tradeN: Number(row.trade_n),
  }));
}

const equityCurve = buildEquityCurve(config);

function auditAnalysis(result) {
  const contractTotal = result.contracts.reduce((sum, item) => sum + item.n, 0);
  const monthTotal = result.months.reduce((sum, item) => sum + item.n, 0);
  if (contractTotal !== result.total_n || monthTotal !== result.total_n) {
    throw new Error("审计失败：合约/月度样本量未与全体样本量对齐。");
  }

  for (const group of [...result.contracts, ...result.months, result.global]) {
    const intervalTotal = group.interval_sample_n.reduce(
      (sum, value) => sum + value,
      0,
    );
    if (intervalTotal !== group.n) {
      throw new Error(`审计失败：${group.group} 区间样本未完整覆盖。`);
    }
    if (
      group.fixed_summary.overall.trade_n !==
      group.fixed_summary.long.trade_n + group.fixed_summary.short.trade_n
    ) {
      throw new Error(`审计失败：${group.group} 多空交易数不一致。`);
    }
    for (const value of group.interval_win_rate) {
      if (value !== null && (value < 0 || value > 1)) {
        throw new Error(`审计失败：${group.group} 出现非法胜率。`);
      }
    }
  }

  console.log(
    `AUDIT=OK total=${result.total_n} thresholds=${result.thresholds.length} intervals=${result.interval_specs.length}`,
  );
}

auditAnalysis(analysis);

function pct(value, digits = 2) {
  return value === null || value === undefined
    ? "—"
    : `${(value * 100).toFixed(digits)}%`;
}

function multiple(value) {
  return value === null || value === undefined ? "—" : `${value.toFixed(2)}x`;
}

function advantageDirection(group) {
  const longRate = group.fixed_summary.long.win_rate;
  const shortRate = group.fixed_summary.short.win_rate;
  if (longRate === null || shortRate === null) return "样本不足";
  if (Math.abs(longRate - shortRate) < 0.0025) return "接近";
  return longRate > shortRate ? "做多" : "做空";
}

function directionText(direction) {
  if (direction === "long") return "做多";
  if (direction === "short") return "做空";
  return "不交易";
}

const workbook = Workbook.create();
const reportSheet = workbook.worksheets.add("分析报告");
const curveSheet = workbook.worksheets.add("累计收益曲线");
const intervalSheet = workbook.worksheets.add("区间明细");
const thresholdSheet = workbook.worksheets.add("阈值明细");

function styleTitle(sheet, range, title) {
  sheet.mergeCells(range);
  const topLeft = range.split(":")[0];
  sheet.getRange(topLeft).values = [[title]];
  sheet.getRange(range).format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 16 },
    verticalAlignment: "center",
  };
  sheet.getRange(range).format.rowHeight = 32;
}

function styleSection(sheet, row, lastColumn, title) {
  sheet.mergeCells(`A${row}:${lastColumn}${row}`);
  sheet.getRange(`A${row}`).values = [[title]];
  sheet.getRange(`A${row}:${lastColumn}${row}`).format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 11 },
    verticalAlignment: "center",
  };
  sheet.getRange(`A${row}:${lastColumn}${row}`).format.rowHeight = 23;
}

function styleHeader(sheet, range) {
  sheet.getRange(range).format = {
    fill: blue,
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: border },
  };
}

function styleDataBlock(sheet, range) {
  sheet.getRange(range).format.borders = {
    insideHorizontal: { style: "thin", color: border },
    bottom: { style: "thin", color: border },
  };
}

function addMetricColorScale(sheet, range, centerValue) {
  sheet.getRange(range).conditionalFormats.add("colorScale", {
    colors: [redLight, yellow, longLight],
    thresholds: [
      "min",
      { type: "num", value: centerValue },
      "max",
    ],
  });
}

function addDirectionFormatting(sheet, range) {
  const target = sheet.getRange(range);
  target.conditionalFormats.add("containsText", {
    text: "做空",
    format: { fill: shortLight, font: { color: shortColor } },
  });
  target.conditionalFormats.add("containsText", {
    text: "做多",
    format: { fill: longLight, font: { color: longColor } },
  });
  target.conditionalFormats.add("containsText", {
    text: "不交易",
    format: { fill: yellow },
  });
}

function addKpiCard(sheet, labelRange, valueRange, label, value, fill, format) {
  sheet.mergeCells(labelRange);
  sheet.mergeCells(valueRange);
  sheet.getRange(labelRange.split(":")[0]).values = [[label]];
  sheet.getRange(valueRange.split(":")[0]).values = [[value]];
  sheet.getRange(labelRange).format = {
    fill,
    font: { bold: true, color: navy },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: border },
  };
  sheet.getRange(valueRange).format = {
    fill,
    font: { bold: true, color: "#000000", size: 18 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    numberFormat: format,
    borders: { preset: "outside", style: "thin", color: border },
  };
}

function buildReportSheet() {
  const sheet = reportSheet;
  const centerLabel = pointLabel(config.direction_threshold);
  const minStableShare = config.report_min_interval_share ?? 0.005;
  const globalSummary = analysis.global.fixed_summary;
  sheet.showGridLines = false;

  styleTitle(sheet, "A1:H1", "预测分数多空策略分析报告");
  sheet.mergeCells("A2:H2");
  sheet.getRange("A2").values = [[
    `数据源：${analysis.source}｜score=pred+${config.score_offset}｜方向阈值=${centerLabel}｜低于阈值做空，高于阈值做多，等于阈值不交易｜区间稳定性提示门槛：样本占比≥${pct(minStableShare, 1)}`,
  ]];
  sheet.getRange("A2:H2").format = {
    fill: lightGray,
    font: { size: 9 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange("A2:H2").format.rowHeight = 28;

  addKpiCard(
    sheet,
    "A4:B4",
    "A5:B6",
    "整体胜率",
    globalSummary.overall.win_rate,
    lightBlue,
    "0.00%",
  );
  addKpiCard(
    sheet,
    "C4:D4",
    "C5:D6",
    "平均策略收益",
    globalSummary.overall.average_strategy_return,
    longLight,
    "0.0000%;[Red]-0.0000%;-",
  );
  addKpiCard(
    sheet,
    "E4:F4",
    "E5:F6",
    "平均盈亏比",
    globalSummary.overall.average_profit_loss_ratio,
    yellow,
    "0.00x",
  );
  addKpiCard(
    sheet,
    "G4:H4",
    "G5:H6",
    "有效交易数",
    globalSummary.overall.trade_n,
    lightGray,
    "#,##0",
  );

  styleSection(sheet, 8, "H", `多空概览（固定方向阈值 ${centerLabel}）`);
  sheet.getRange("A9:F9").values = [[
    "方向",
    "样本量",
    "样本占比",
    "胜率",
    "平均策略收益",
    "平均盈亏比",
  ]];
  styleHeader(sheet, "A9:F9");
  sheet.getRange("A10:F12").values = [
    [
      "做多",
      globalSummary.long.trade_n,
      globalSummary.long.trade_n / analysis.total_n,
      globalSummary.long.win_rate,
      globalSummary.long.average_strategy_return,
      globalSummary.long.average_profit_loss_ratio,
    ],
    [
      "做空",
      globalSummary.short.trade_n,
      globalSummary.short.trade_n / analysis.total_n,
      globalSummary.short.win_rate,
      globalSummary.short.average_strategy_return,
      globalSummary.short.average_profit_loss_ratio,
    ],
    [
      "合计",
      globalSummary.overall.trade_n,
      globalSummary.overall.trade_n / analysis.total_n,
      globalSummary.overall.win_rate,
      globalSummary.overall.average_strategy_return,
      globalSummary.overall.average_profit_loss_ratio,
    ],
  ];
  styleDataBlock(sheet, "A10:F12");
  sheet.getRange("B10:B12").format.numberFormat = "#,##0";
  sheet.getRange("C10:D12").format.numberFormat = "0.00%";
  sheet.getRange("E10:E12").format.numberFormat =
    "0.0000%;[Red]-0.0000%;-";
  sheet.getRange("F10:F12").format.numberFormat = "0.00x";
  sheet.getRange("A12:F12").format = {
    fill: lightBlue,
    font: { bold: true },
    borders: { top: { style: "medium", color: navy } },
  };
  addDirectionFormatting(sheet, "A10:A11");

  styleSection(sheet, 14, "H", "全体区间表现");
  sheet.getRange("A15:H15").values = [[
    "方向",
    "分数区间",
    "样本量",
    "样本占比",
    "平均策略收益",
    "胜率",
    "平均盈亏比",
    "样本提示",
  ]];
  styleHeader(sheet, "A15:H15");

  const intervalRows = analysis.interval_specs.map((spec, index) => {
    const share = analysis.global.interval_sample_n[index] / analysis.total_n;
    return [
      directionText(spec.direction),
      spec.label,
      analysis.global.interval_sample_n[index],
      share,
      analysis.global.average_strategy_return[index],
      analysis.global.interval_win_rate[index],
      analysis.global.average_profit_loss_ratio[index],
      spec.direction === "neutral"
        ? "不交易"
        : share >= minStableShare
          ? "可参考"
          : "小样本",
    ];
  });
  const intervalStartRow = 16;
  const intervalEndRow = intervalStartRow + intervalRows.length - 1;
  sheet.getRange(`A${intervalStartRow}:H${intervalEndRow}`).values = intervalRows;
  styleDataBlock(sheet, `A${intervalStartRow}:H${intervalEndRow}`);
  sheet.getRange(`C${intervalStartRow}:C${intervalEndRow}`).format.numberFormat =
    "#,##0";
  sheet.getRange(`D${intervalStartRow}:D${intervalEndRow}`).format.numberFormat =
    "0.00%";
  sheet.getRange(`E${intervalStartRow}:E${intervalEndRow}`).format.numberFormat =
    "0.0000%;[Red]-0.0000%;-";
  sheet.getRange(`F${intervalStartRow}:F${intervalEndRow}`).format.numberFormat =
    "0.00%";
  sheet.getRange(`G${intervalStartRow}:G${intervalEndRow}`).format.numberFormat =
    "0.00x";
  addDirectionFormatting(sheet, `A${intervalStartRow}:A${intervalEndRow}`);
  addMetricColorScale(sheet, `E${intervalStartRow}:E${intervalEndRow}`, 0);
  addMetricColorScale(sheet, `F${intervalStartRow}:F${intervalEndRow}`, 0.5);
  addMetricColorScale(sheet, `G${intervalStartRow}:G${intervalEndRow}`, 1);
  sheet
    .getRange(`H${intervalStartRow}:H${intervalEndRow}`)
    .conditionalFormats.add("containsText", {
      text: "小样本",
      format: { fill: redLight, font: { color: "#9C0006" } },
    });

  const contractTitleRow = intervalEndRow + 2;
  styleSection(sheet, contractTitleRow, "H", "合约表现概览");
  const contractHeaderRow = contractTitleRow + 1;
  sheet.getRange(`A${contractHeaderRow}:H${contractHeaderRow}`).values = [[
    "合约",
    "样本量",
    "多单占比",
    "空单占比",
    "整体胜率",
    "平均策略收益",
    "平均盈亏比",
    "优势方向",
  ]];
  styleHeader(sheet, `A${contractHeaderRow}:H${contractHeaderRow}`);
  const contractRows = analysis.contracts.map((group) => [
    group.group,
    group.n,
    group.fixed_summary.long.trade_n / group.n,
    group.fixed_summary.short.trade_n / group.n,
    group.fixed_summary.overall.win_rate,
    group.fixed_summary.overall.average_strategy_return,
    group.fixed_summary.overall.average_profit_loss_ratio,
    advantageDirection(group),
  ]);
  const contractStartRow = contractHeaderRow + 1;
  const contractEndRow = contractStartRow + contractRows.length - 1;
  sheet.getRange(`A${contractStartRow}:H${contractEndRow}`).values = contractRows;
  styleDataBlock(sheet, `A${contractStartRow}:H${contractEndRow}`);
  sheet.getRange(`B${contractStartRow}:B${contractEndRow}`).format.numberFormat =
    "#,##0";
  sheet.getRange(`C${contractStartRow}:E${contractEndRow}`).format.numberFormat =
    "0.00%";
  sheet.getRange(`F${contractStartRow}:F${contractEndRow}`).format.numberFormat =
    "0.0000%;[Red]-0.0000%;-";
  sheet.getRange(`G${contractStartRow}:G${contractEndRow}`).format.numberFormat =
    "0.00x";
  addMetricColorScale(sheet, `E${contractStartRow}:E${contractEndRow}`, 0.5);
  addMetricColorScale(sheet, `F${contractStartRow}:F${contractEndRow}`, 0);
  addMetricColorScale(sheet, `G${contractStartRow}:G${contractEndRow}`, 1);
  addDirectionFormatting(sheet, `H${contractStartRow}:H${contractEndRow}`);

  const monthTitleRow = contractEndRow + 2;
  styleSection(sheet, monthTitleRow, "H", "近期12个月表现");
  const monthHeaderRow = monthTitleRow + 1;
  sheet.getRange(`A${monthHeaderRow}:H${monthHeaderRow}`).values = [[
    "月份",
    "样本量",
    "多单占比",
    "空单占比",
    "整体胜率",
    "平均策略收益",
    "平均盈亏比",
    "优势方向",
  ]];
  styleHeader(sheet, `A${monthHeaderRow}:H${monthHeaderRow}`);
  const recentMonths = analysis.months.slice(-12);
  const monthRows = recentMonths.map((group) => [
    group.group,
    group.n,
    group.fixed_summary.long.trade_n / group.n,
    group.fixed_summary.short.trade_n / group.n,
    group.fixed_summary.overall.win_rate,
    group.fixed_summary.overall.average_strategy_return,
    group.fixed_summary.overall.average_profit_loss_ratio,
    advantageDirection(group),
  ]);
  const monthStartRow = monthHeaderRow + 1;
  const monthEndRow = monthStartRow + monthRows.length - 1;
  sheet.getRange(`A${monthStartRow}:H${monthEndRow}`).values = monthRows;
  styleDataBlock(sheet, `A${monthStartRow}:H${monthEndRow}`);
  sheet.getRange(`B${monthStartRow}:B${monthEndRow}`).format.numberFormat =
    "#,##0";
  sheet.getRange(`C${monthStartRow}:E${monthEndRow}`).format.numberFormat =
    "0.00%";
  sheet.getRange(`F${monthStartRow}:F${monthEndRow}`).format.numberFormat =
    "0.0000%;[Red]-0.0000%;-";
  sheet.getRange(`G${monthStartRow}:G${monthEndRow}`).format.numberFormat =
    "0.00x";
  addMetricColorScale(sheet, `E${monthStartRow}:E${monthEndRow}`, 0.5);
  addMetricColorScale(sheet, `F${monthStartRow}:F${monthEndRow}`, 0);
  addMetricColorScale(sheet, `G${monthStartRow}:G${monthEndRow}`, 1);
  addDirectionFormatting(sheet, `H${monthStartRow}:H${monthEndRow}`);

  const globalIntervals = analysis.interval_specs
    .map((spec, index) => ({
      spec,
      index,
      share: analysis.global.interval_sample_n[index] / analysis.total_n,
      averageReturn: analysis.global.average_strategy_return[index],
    }))
    .filter(
      (item) => item.spec.direction !== "neutral" && item.averageReturn !== null,
    );
  const stableIntervals = globalIntervals
    .filter((item) => item.share >= minStableShare)
    .sort((a, b) => b.averageReturn - a.averageReturn);
  const allRankedIntervals = [...globalIntervals].sort(
    (a, b) => b.averageReturn - a.averageReturn,
  );
  const bestStable = stableIntervals[0];
  const bestRaw = allRankedIntervals[0];
  const directionLeader =
    globalSummary.long.win_rate >= globalSummary.short.win_rate ? "做多" : "做空";
  const directionDelta = Math.abs(
    globalSummary.long.win_rate - globalSummary.short.win_rate,
  );

  const conclusionTitleRow = monthEndRow + 2;
  styleSection(sheet, conclusionTitleRow, "H", "关键结论");
  const conclusions = [
    `1. 固定阈值 ${centerLabel} 下，整体胜率为 ${pct(globalSummary.overall.win_rate)}，平均策略收益为 ${pct(globalSummary.overall.average_strategy_return, 4)}，平均盈亏比为 ${multiple(globalSummary.overall.average_profit_loss_ratio)}。`,
    `2. ${directionLeader}方向胜率更高，较另一方向高 ${pct(directionDelta)}；但方向选择仍应同时观察样本占比与平均收益。`,
    bestStable
      ? `3. 在样本占比≥${pct(minStableShare, 1)}的区间中，${bestStable.spec.label}（${directionText(bestStable.spec.direction)}）平均策略收益最高，为 ${pct(bestStable.averageReturn, 4)}，样本占比 ${pct(bestStable.share)}。`
      : `3. 没有区间达到样本占比≥${pct(minStableShare, 1)}的稳定性门槛。`,
    bestRaw && bestStable && bestRaw.spec.label !== bestStable.spec.label
      ? `4. 未限制样本量时，最高收益区间为 ${bestRaw.spec.label}，平均收益 ${pct(bestRaw.averageReturn, 4)}，但样本占比仅 ${pct(bestRaw.share)}，应防止小样本误判。`
      : "4. 高收益区间仍需结合胜率、平均盈亏比和样本量判断，不能只看单一收益指标。",
    "5. 胜率衡量赚钱频率；平均盈亏比衡量单笔赚亏幅度。只有两者结合，才能判断策略是否具备稳定优势。",
  ];
  conclusions.forEach((line, index) => {
    const row = conclusionTitleRow + 1 + index;
    sheet.mergeCells(`A${row}:H${row}`);
    sheet.getRange(`A${row}`).values = [[line]];
    sheet.getRange(`A${row}:H${row}`).format = {
      fill: index % 2 === 0 ? "#FFFFFF" : lightGray,
      wrapText: true,
      verticalAlignment: "center",
      borders: { bottom: { style: "thin", color: border } },
    };
    sheet.getRange(`A${row}:H${row}`).format.rowHeight = 25;
  });

  const finalRow = conclusionTitleRow + conclusions.length;
  sheet.freezePanes.freezeRows(2);
  sheet.getRange(`A1:H${finalRow}`).format.font = { name: "Aptos", size: 10 };
  sheet.getRange("A:A").format.columnWidth = 15;
  sheet.getRange("B:B").format.columnWidth = 15;
  sheet.getRange("C:C").format.columnWidth = 13;
  sheet.getRange("D:D").format.columnWidth = 13;
  sheet.getRange("E:E").format.columnWidth = 17;
  sheet.getRange("F:F").format.columnWidth = 13;
  sheet.getRange("G:G").format.columnWidth = 14;
  sheet.getRange("H:H").format.columnWidth = 15;

  return { finalRow };
}

function buildIntervalDetailSheet() {
  const sheet = intervalSheet;
  sheet.showGridLines = false;
  styleTitle(sheet, "A1:I1", "区间表现明细");
  sheet.mergeCells("A2:I2");
  sheet.getRange("A2").values = [[
    "固定方向阈值下的区间明细：左侧做空、右侧做多；50单列不交易。胜率不含收益为0的样本。可使用表头筛选维度、分组和方向。",
  ]];
  sheet.getRange("A2:I2").format = {
    fill: lightGray,
    wrapText: true,
    font: { size: 9 },
  };
  sheet.getRange("A4:I4").values = [[
    "维度",
    "分组",
    "方向",
    "分数区间",
    "样本量",
    "组内占比",
    "平均策略收益",
    "胜率",
    "平均盈亏比",
  ]];

  const rows = [];
  for (const [dimensionLabel, groups] of [
    ["全体", [analysis.global]],
    ["按合约", analysis.contracts],
    ["按月份", analysis.months],
  ]) {
    for (const group of groups) {
      analysis.interval_specs.forEach((spec, index) => {
        rows.push([
          dimensionLabel,
          group.group,
          directionText(spec.direction),
          spec.label,
          group.interval_sample_n[index],
          group.interval_sample_n[index] / group.n,
          group.average_strategy_return[index],
          group.interval_win_rate[index],
          group.average_profit_loss_ratio[index],
        ]);
      });
    }
  }

  const startRow = 5;
  const endRow = startRow + rows.length - 1;
  sheet.getRange(`A${startRow}:I${endRow}`).values = rows;
  const table = sheet.tables.add(`A4:I${endRow}`, true, "IntervalDetailTable");
  table.showFilterButton = true;
  styleHeader(sheet, "A4:I4");
  styleDataBlock(sheet, `A${startRow}:I${endRow}`);
  sheet.getRange(`E${startRow}:E${endRow}`).format.numberFormat = "#,##0";
  sheet.getRange(`F${startRow}:F${endRow}`).format.numberFormat = "0.00%";
  sheet.getRange(`G${startRow}:G${endRow}`).format.numberFormat =
    "0.0000%;[Red]-0.0000%;-";
  sheet.getRange(`H${startRow}:H${endRow}`).format.numberFormat = "0.00%";
  sheet.getRange(`I${startRow}:I${endRow}`).format.numberFormat = "0.00x";
  addDirectionFormatting(sheet, `C${startRow}:C${endRow}`);
  addMetricColorScale(sheet, `G${startRow}:G${endRow}`, 0);
  addMetricColorScale(sheet, `H${startRow}:H${endRow}`, 0.5);
  addMetricColorScale(sheet, `I${startRow}:I${endRow}`, 1);
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(4);
  sheet.getRange(`A1:I${endRow}`).format.font = { name: "Aptos", size: 10 };
  sheet.getRange("A:A").format.columnWidth = 12;
  sheet.getRange("B:B").format.columnWidth = 14;
  sheet.getRange("C:C").format.columnWidth = 11;
  sheet.getRange("D:D").format.columnWidth = 14;
  sheet.getRange("E:E").format.columnWidth = 12;
  sheet.getRange("F:F").format.columnWidth = 12;
  sheet.getRange("G:G").format.columnWidth = 17;
  sheet.getRange("H:H").format.columnWidth = 12;
  sheet.getRange("I:I").format.columnWidth = 14;
  return { endRow };
}

function buildThresholdDetailSheet() {
  const sheet = thresholdSheet;
  sheet.showGridLines = false;
  styleTitle(sheet, "A1:G1", "移动阈值明细");
  sheet.mergeCells("A2:G2");
  sheet.getRange("A2").values = [[
    "每一行代表一个候选阈值：score>threshold做多，score<threshold做空，相等不交易。Precision衡量方向命中率，Coverage衡量信号覆盖率。",
  ]];
  sheet.getRange("A2:G2").format = {
    fill: lightGray,
    wrapText: true,
    font: { size: 9 },
  };
  sheet.getRange("A4:G4").values = [[
    "维度",
    "分组",
    "阈值",
    "做多Precision",
    "做空Precision",
    "做多Coverage",
    "做空Coverage",
  ]];

  const rows = [];
  for (const [dimensionLabel, groups] of [
    ["全体", [analysis.global]],
    ["按合约", analysis.contracts],
    ["按月份", analysis.months],
  ]) {
    for (const group of groups) {
      analysis.threshold_labels.forEach((label, index) => {
        rows.push([
          dimensionLabel,
          group.group,
          label,
          group.long_precision[index],
          group.short_precision[index],
          group.long_coverage[index],
          group.short_coverage[index],
        ]);
      });
    }
  }

  const startRow = 5;
  const endRow = startRow + rows.length - 1;
  sheet.getRange(`A${startRow}:G${endRow}`).values = rows;
  const table = sheet.tables.add(`A4:G${endRow}`, true, "ThresholdDetailTable");
  table.showFilterButton = true;
  styleHeader(sheet, "A4:G4");
  styleDataBlock(sheet, `A${startRow}:G${endRow}`);
  sheet.getRange(`C${startRow}:C${endRow}`).format.numberFormat = "0";
  sheet.getRange(`D${startRow}:G${endRow}`).format.numberFormat = "0.00%";
  addMetricColorScale(sheet, `D${startRow}:E${endRow}`, 0.5);
  sheet.getRange(`F${startRow}:G${endRow}`).conditionalFormats.add("colorScale", {
    colors: ["#FFFFFF", lightBlue, blue],
    thresholds: ["min", "50%", "max"],
  });
  sheet.getRange(`C${startRow}:C${endRow}`).conditionalFormats.add("cellIs", {
    operator: "equal",
    formula: pointLabel(config.direction_threshold),
    format: { fill: yellow, font: { bold: true } },
  });
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(3);
  sheet.getRange(`A1:G${endRow}`).format.font = { name: "Aptos", size: 10 };
  sheet.getRange("A:A").format.columnWidth = 12;
  sheet.getRange("B:B").format.columnWidth = 14;
  sheet.getRange("C:C").format.columnWidth = 10;
  sheet.getRange("D:G").format.columnWidth = 16;
  return { endRow };
}

function buildEquityCurveSheet() {
  const sheet = curveSheet;
  const centerLabel = pointLabel(config.direction_threshold);
  const startRow = 31;
  const endRow = startRow + equityCurve.length - 1;
  sheet.showGridLines = false;

  styleTitle(sheet, "A1:H1", "固定阈值累计收益曲线");
  sheet.mergeCells("A2:H2");
  sheet.getRange("A2").values = [[
    `阈值=${centerLabel}｜score>阈值做多取Tag，score<阈值做空取-Tag，等于阈值不交易｜先按交易日对全部有效信号等权平均，再按复利口径累计：cumprod(1+日均策略收益)-1`,
  ]];
  sheet.getRange("A2:H2").format = {
    fill: lightGray,
    font: { size: 9 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange("A2:H2").format.rowHeight = 30;

  addKpiCard(sheet, "A4:B4", "A5:B6", "方向阈值", config.direction_threshold, lightBlue, "0.00");
  addKpiCard(sheet, "C4:D4", "C5:D6", "期末累计收益", null, longLight, "0.00%;[Red]-0.00%;-");
  addKpiCard(sheet, "E4:F4", "E5:F6", "最大回撤", null, redLight, "0.00%;[Red]-0.00%;-");
  addKpiCard(sheet, "G4:H4", "G5:H6", "统计交易日", equityCurve.length, lightGray, "#,##0");
  sheet.getRange("C5").formulas = [[`=D${endRow}`]];
  sheet.getRange("E5").formulas = [[`=MIN(E${startRow}:E${endRow})`]];

  sheet.getRange("A30:E30").values = [[
    "日期",
    "日均策略收益",
    "交易样本数",
    "累计收益率",
    "回撤",
  ]];
  styleHeader(sheet, "A30:E30");
  sheet.getRange(`A${startRow}:C${endRow}`).values = equityCurve.map((row) => [
    row.date,
    row.dailyReturn,
    row.tradeN,
  ]);
  sheet.getRange(`D${startRow}`).formulas = [[`=B${startRow}`]];
  if (equityCurve.length > 1) {
    sheet.getRange(`D${startRow + 1}`).formulas = [[
      `=(1+D${startRow})*(1+B${startRow + 1})-1`,
    ]];
    sheet.getRange(`D${startRow + 1}:D${endRow}`).fillDown();
  }
  sheet.getRange(`E${startRow}`).formulas = [[
    `=(1+D${startRow})/(1+MAX(0,$D$${startRow}:D${startRow}))-1`,
  ]];
  if (equityCurve.length > 1) {
    sheet.getRange(`E${startRow + 1}`).formulas = [[
      `=(1+D${startRow + 1})/(1+MAX(0,$D$${startRow}:D${startRow + 1}))-1`,
    ]];
    sheet.getRange(`E${startRow + 1}:E${endRow}`).fillDown();
  }
  const table = sheet.tables.add(`A30:E${endRow}`, true, "EquityCurveTable");
  table.showFilterButton = true;
  styleDataBlock(sheet, `A${startRow}:E${endRow}`);
  sheet.getRange(`B${startRow}:B${endRow}`).format.numberFormat =
    "0.0000%;[Red]-0.0000%;-";
  sheet.getRange(`C${startRow}:C${endRow}`).format.numberFormat = "#,##0";
  sheet.getRange(`D${startRow}:E${endRow}`).format.numberFormat =
    "0.00%;[Red]-0.00%;-";

  const chart = sheet.charts.add("line", {
    chartType: "line",
    title: `固定阈值 ${centerLabel}：累计收益率（日频等权、复利）`,
    hasLegend: false,
  });
  const series = chart.series.add("累计收益率");
  series.categoryFormula = `'累计收益曲线'!$A$${startRow}:$A$${endRow}`;
  series.formula = `'累计收益曲线'!$D$${startRow}:$D$${endRow}`;
  series.fill = blue;
  chart.setPosition("A8", "H27");
  chart.titleTextStyle.fontSize = 13;
  chart.hasLegend = false;
  chart.xAxis = {
    axisType: "textAxis",
    tickLabelInterval: Math.max(1, Math.floor(equityCurve.length / 12)),
    textStyle: { fontSize: 9 },
  };
  chart.yAxis = { numberFormatCode: "0%;[Red]-0%" };

  sheet.freezePanes.freezeRows(30);
  sheet.getRange(`A1:H${endRow}`).format.font = { name: "Aptos", size: 10 };
  sheet.getRange("A:A").format.columnWidth = 13;
  sheet.getRange("B:B").format.columnWidth = 17;
  sheet.getRange("C:C").format.columnWidth = 14;
  sheet.getRange("D:E").format.columnWidth = 16;
  sheet.getRange("F:H").format.columnWidth = 13;
  return { startRow, endRow };
}

const reportLayout = buildReportSheet();
const curveLayout = buildEquityCurveSheet();
const intervalLayout = buildIntervalDetailSheet();
const thresholdLayout = buildThresholdDetailSheet();

await fs.mkdir(previewDir, { recursive: true });

async function saveRender(fileName, sheetName, range, scale) {
  const preview = await workbook.render({
    sheetName,
    range,
    scale,
    format: "png",
  });
  await fs.writeFile(
    `${previewDir}/${fileName}`,
    new Uint8Array(await preview.arrayBuffer()),
  );
}

await saveRender(
  "分析报告.png",
  "分析报告",
  `A1:H${reportLayout.finalRow}`,
  1.1,
);
await saveRender("累计收益曲线.png", "累计收益曲线", "A1:H35", 1.1);
await saveRender("区间明细.png", "区间明细", "A1:I35", 1.0);
await saveRender("阈值明细.png", "阈值明细", "A1:G35", 1.0);

const reportCheck = await workbook.inspect({
  kind: "table",
  range: `分析报告!A1:H${reportLayout.finalRow}`,
  include: "values,formulas",
  tableMaxRows: 90,
  tableMaxCols: 8,
  maxChars: 9000,
});
console.log(reportCheck.ndjson);

const curveCheck = await workbook.inspect({
  kind: "table",
  range: `累计收益曲线!A1:H35`,
  include: "values,formulas",
  tableMaxRows: 35,
  tableMaxCols: 8,
  maxChars: 5000,
});
console.log(curveCheck.ndjson);

const intervalCheck = await workbook.inspect({
  kind: "table",
  range: "区间明细!A1:I15",
  include: "values,formulas",
  tableMaxRows: 15,
  tableMaxCols: 9,
  maxChars: 5000,
});
console.log(intervalCheck.ndjson);

const thresholdCheck = await workbook.inspect({
  kind: "table",
  range: "阈值明细!A1:G15",
  include: "values,formulas",
  tableMaxRows: 15,
  tableMaxCols: 7,
  maxChars: 5000,
});
console.log(thresholdCheck.ndjson);

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
