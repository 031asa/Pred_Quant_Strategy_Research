# 当前复现入口

`config.json`是原始数据路径、字段名、14:30切点和Score换算参数的唯一配置来源。`scripts/`只保留三个可读入口，每个入口明确列出配置文件、额外实验参数和输出路径，在`main()`中读取配置、用命名参数调用`utils/`接口并打印结果摘要；计算、统计和绘图工具函数仍位于`utils/`：

1. `analyze_segmented_score_drift.py`：输入原始Parquet，输出参数JSON和两张月度漂移图。
2. `export_zero_centered_pred.py`：输入原始Parquet与参数JSON，输出零中心最终Parquet。
3. `analyze_adjusted_pred_returns.py`：输入原始与最终Parquet，先对齐相同有效样本，再输出六张六联图、六张零阈值单利累计收益图、六张配对单利累计收益图和六张Pred分布图。

实现代码位于`utils/`：

- `score_drift_workflow.py`：漂移选参、评价和两张月度漂移图。
- `zero_centered_export.py`：因果状态重建及最终Parquet导出。
- `adjusted_pred_analysis.py`：同样本对齐、逐样本零阈值方向、六指标、等样本分组、单利累计收益、对称档位配对和Pred分布，共二十四张图。
- `segmented_drift.py`：共享分段、EWMA和每日因果状态函数。
- 三个工作流模块分别暴露有类型标注的高层函数和返回对象，导入模块不会执行数据读写。

所有入口和实现均不读写CSV或Excel。
