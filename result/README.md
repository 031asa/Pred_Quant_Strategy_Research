# 最终结果目录

- `data/pred_eval_segmented_drift_adjusted.parquet`：1,667,520条零中心有效样本；`pred < 0`做空，`pred >= 0`做多。
- `parameters/segmented_score_drift_selected_parameters.json`：两个时段的半衰期、alpha、固定平衡偏移、冷启动和测试期配置。
- `visualizations/before_adjustment/`：调整前（未平滑）的三张零阈值六联图、三张零阈值单利累计收益图，以及按总体、14:30前、14:30后拆分的配对单利累计收益图和Pred分布图。
- `visualizations/after_adjustment/`：调整后的三张零阈值六联图、三张零阈值单利累计收益图，以及按总体、14:30前、14:30后拆分的配对单利累计收益图和Pred分布图。
- `visualizations/drift_comparison/`：14:30前后两张月度漂移对比图。
- `reports/`：既有PDF、Word、项目交接MD及报告README。
- `historical_notes/`：总体Score漂移和Plan B等旧实验README，仅用于追溯。

本目录不保存CSV、Excel或中间状态文件。状态由`utils/segmented_drift.py`从原始Parquet和参数JSON在内存中重建。
