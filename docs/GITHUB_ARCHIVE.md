# GitHub完整归档说明

私有仓库：<https://github.com/031asa/Pred_Quant_Strategy_Research>。

本副本以已验证的Gitea完整交接提交 `bcd2725ccb4285d98c9ff287cfb8efc1e36a8afb` 为基线，包含该提交的全部文件和此前提交历史。2026-09-28用户明确授权将原始数据与全部研究资料保存到本私有GitHub；新增的GitHub说明不改变算法、参数或结果。

## 保存内容

| 材料 | 仓库路径 |
|---|---|
| 原始1,728,000条模型结果集 | `data/pred_eval.parquet` |
| 最终1,667,520条零中心Pred | `result/data/pred_eval_segmented_drift_adjusted.parquet` |
| 当前三步分析、公共计算、Web A/B实验与测试 | `scripts/`、`utils/`、`streamlit_app.py`、`tests/` |
| 参数、当前26张图、原研究报告与验收 | `result/` |
| 99份历史源码、参数、CSV/Excel、图与交付包 | `archive/legacy_research/` |
| 较早中文项目43份文件快照 | `archive/chinese_summary_snapshot/` |
| 较早中文项目完整Git历史 | `archive/chinese_summary_history.bundle` |
| 业务口径、历史版本差异及未知事项 | `docs/HANDOVER.md`、`docs/REPOSITORY_AUDIT.md` |
| 可移植环境和实际验收依赖快照 | `environment.yml`、`result/verification/environment_snapshot.json` |
| 每个来源文件与完整性校验 | `docs/source_manifest.json`、`docs/checksums.sha256` |

上述数据文件直接保存在Git中，不是LFS指针，也不依赖原电脑或Gitea下载链接。`git clone`可以取回完整仓库和提交历史；GitHub的Download ZIP只包含文件快照，不包含当前仓库的`.git`历史，但仍包含上述单独保存的中文旧历史bundle。

## 获取与校验

使用有该私有仓库访问权限的GitHub账号：

```bash
git clone https://github.com/031asa/Pred_Quant_Strategy_Research.git
cd Pred_Quant_Strategy_Research
conda env create -f environment.yml
conda activate pred-quant-research
python scripts/verify_handover.py
```

建议先读 `docs/HANDOVER.md`，再按根README运行分析。`python scripts/reproduce_handover.py`在临时目录完整重跑三步分析并核对原交付结果，只更新 `result/verification/` 中的运行记录。校验清单描述本次交付状态，重新运行实验后部分输出文件发生变化属于预期行为。

交接基线已通过43项测试、全量参数与Pred逐值复现，并重新生成26张图；证据位于 `result/verification/`。GitHub同步保留同一套数据和计算实现。

本归档覆盖模型输出之后的分析与修正。没有凭空补造原始模型训练代码、模型权重、特征或Tag生成源码；这些缺项及历史标签周期的不确定性已在交接说明中记录。
