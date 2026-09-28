# 最终结果目录

- `数据/pred_eval_分时漂移修正后.parquet`：1,667,520条零中心有效样本；`pred < 0`做空，`pred >= 0`做多。
- `参数/分时Score时序漂移_选定参数.json`：两个时段的半衰期、alpha、固定平衡偏移、冷启动和测试期配置。
- `可视化/`：当前汇报使用的9张核心PNG、图片索引和核心指标表。
- `报告/`：既有PDF、Word、项目交接MD及报告README。
- `历史说明/`：总体Score漂移和Plan B等旧实验README，仅用于追溯。

本目录不保存CSV、Excel或中间状态文件。状态由`utils/segmented_drift.py`从原始Parquet和参数JSON在内存中重建。
