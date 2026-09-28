# Gitea 与本地研究材料核对

核对日期：2026-09-28。目标为既有私有仓库 `http://172.16.20.14:3000/hjz/Pred_Quant_Strategy_Research`。核对前 main 提交为 `0425b8e45a4e14973e2ba37d893a26b0767a0b72`。

## 对比结论

远程原有70个文件与本机 `Pred_Quant_Strategy_Research` 新版项目内容相同：65个SHA256一致，另外5个只有CRLF/LF差异。Windows新版和WSL旧副本均得到相同结果。对应详细路径见 `source_comparison.json`；其中 `changed` 表示字节不同，`line_ending_only` 表示规范化换行后完全相同。

因此此次没有覆盖现有算法、Streamlit应用、参数或研究结果，也没有把较早中文代码当成新版。远程原本已有最终零中心Parquet、26张当前图、业务README、字体及许可证、脚本和测试；主要缺口是原始数据和完整历史证据。

## 本次补齐

| 材料 | 保存位置 | 处理 |
|---|---|---|
| 原始172.8万行结果集 | `data/pred_eval.parquet` | 与下载原件SHA256一致，23,638,240字节 |
| 早期研究源码、参数、CSV、Excel、PNG、PDF、Word、HTML与交付ZIP | `archive/legacy_research/` | 99文件、41,788,393字节，保留原文件名与内容 |
| 较早中文工作汇总 | `archive/chinese_summary_snapshot/` | 43文件；两份重复Parquet引用当前目录中的同内容原件 |
| 较早中文仓库历史 | `archive/chinese_summary_history.bundle` | 完整Git历史，起点提交`ae1cc94e71f0b457b30fdde0447daa3ceeb53c42` |
| 来源、完整性与口径说明 | `docs/` | 文件来源/SHA256、业务定义、版本错配说明 |
| 复现环境与验收 | `environment.yml`、`scripts/reproduce_handover.py`、`result/verification/` | Linux Conda配置、隔离复现脚本与运行证据 |

逐文件来源和去重映射在 `source_manifest.json`。未归档虚拟环境、node_modules、Python缓存、约93MB的inspect.ndjson，以及渲染预览和a11y检查文件；它们不是原始数据或研究结论。历史ZIP作为已交付版本保留，即使与旁边的单文件存在重复。

## 版本与证据边界

旧中文版本与当前英文目录版不等价：当前版新增A/B可组合流水线、Plotly、字体资源与更多测试，并采用日内等权、跨日单利累计。旧复利结果保留供追溯，不能替换当前口径。

历史 `README_总体Score时序漂移_加减修正.md` 曾更新为分时说明，但旁边旧总体脚本、参数和CSV仍属于全体混合版本。归档保留当时实际文件；其对应关系与解释以 `HANDOVER.md` 的专门说明为准。

未在此次项目材料中找到模型训练代码、模型权重、特征表、原始分钟行情或Tag生成源码；本次补齐的是模型结果集后处理研究的交接。Tag约30分钟的预测周期仍为分析推断。

首次补齐时，仓库页面核验为私有，Mirror Settings显示无push mirror；该次仅上传Gitea，形成完整交接提交 `bcd2725ccb4285d98c9ff287cfb8efc1e36a8afb`。

2026-09-28用户随后明确要求将完整项目同步到 GitHub `031asa/Pred_Quant_Strategy_Research`。核验该仓库为private，main仍为最初提交 `0425b8e45a4e14973e2ba37d893a26b0767a0b72`，是完整交接版本的祖先，可快进同步。此次保留原始Parquet、所有已归档研究材料及提交历史，仅补充GitHub存放说明与对应哈希清单，不改变研究算法和结果。此前2026-09-23的同步记录不代表当时已经包含本次补齐内容。
