# 原始数据说明

`pred_eval.parquet` 是原始模型输出结果集，现已随本私有 Gitea 仓库提交。它与下载目录原件逐字节一致，未经漂移修正，不是原始市场行情或训练特征。

- 字节数：23,638,240；样本数：1,728,000；交易日：600；合约数：12。
- 日期：2024-01-03 至 2026-06-30。
- SHA256：`be9297c4e60ac6eefe73bcbfe2cf84f5d3ba4cc3912bda16e67ae7dba61c0305`。
- 字段：`Date`、`Contract`、`Datetime`、`pred`、`Tag`。
- `Datetime` 为 Asia/Shanghai 时间；14:30 整点及以后归后半段。
- 原始 `pred` 为原预测概率减0.5，当前脚本使用 `score = clip(pred + 0.5, 0, 1)`。
- `Tag` 为带符号的真实收益率；做多收益取 `Tag`，做空收益取 `-Tag`。乘10,000才转换为bp。

`result/data/pred_eval_segmented_drift_adjusted.parquet` 是另一种语义的最终零中心信号，包含1,667,520条有效样本。对该文件直接使用 `pred < 0` 做空、`pred >= 0` 做多；不要把它当原始概率再送入原始输入流程。

原始文件由 `.gitignore` 中的明确例外 `!data/pred_eval.parquet` 纳入版本控制，其他新增Parquet默认仍忽略。全部文件校验见 `docs/checksums.sha256`，业务口径与未知标签周期见 `docs/HANDOVER.md`。
