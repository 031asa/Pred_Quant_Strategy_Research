# 当前复现入口

`scripts/`只保留三个薄入口，每个入口负责确定项目根目录、导入工作流接口并执行，不包含计算、统计或绘图工具函数：

1. `总体Score时序漂移.py`：调用分时漂移选参与月度图工作流。
2. `导出分时漂移修正后Pred.py`：调用零中心Pred导出工作流。
3. `修正后Pred分组收益.py`：调用分组、累计收益和盈亏比图工作流。

实现代码位于`utils/`：

- `score_drift_workflow.py`：漂移选参、评价和两张月度漂移图。
- `zero_centered_export.py`：因果状态重建及最终Parquet导出。
- `adjusted_pred_analysis.py`：等样本分组、累计收益、盈亏比和七张图。
- `segmented_drift.py`：共享分段、EWMA和每日因果状态函数。
- `pipelines.py`：向三个入口暴露稳定的高层调用接口。

所有入口和实现均不读写CSV或Excel。
