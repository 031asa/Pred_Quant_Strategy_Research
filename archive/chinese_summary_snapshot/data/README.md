# 数据说明

`pred_eval.parquet`为项目原始分钟数据，共1,728,000行、600个交易日和12个合约。

核心字段：

- `Datetime`：Asia/Shanghai分钟时间戳；
- `Contract`：合约代码；
- `pred`：原始模型输出；
- `Tag`：实际收益率。

项目脚本计算：

- `score = clip(pred + 0.5, 0, 1)`；
- `Datetime < 14:30`为盘中；
- `Datetime >= 14:30`为尾盘；
- 阈值右侧做多，左侧做空；
- 做多收益为Tag，做空收益为-Tag。

parquet文件被`.gitignore`排除，不写入Git历史。
