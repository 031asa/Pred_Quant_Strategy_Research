# 按分布等样本十等分分析

## 口径

- 原始文件：`C:/Users/Hello/Downloads/pred_eval.parquet`
- 预估分数：`score = pred + 0.5`，并限制在 `[0, 1]`
- 固定方向阈值：`0.50`
- `score > 0.50`：做多，策略收益取 `Tag`
- `score < 0.50`：做空，策略收益取 `-Tag`
- `score == 0.50`：不交易
- 只对全体样本按 score 排序，再按样本量等分为 10 档
- 每档样本量最多相差 1；本次全体样本的十档样本量完全相等
- 若某一档跨过 0.50，档内每条样本仍按自己的 score 逐笔判定多空

## Excel 报告

- 工作簿仅保留一张 `全体十分位分析`：展示全体十档表现、样本均衡检查和关键结论

## Jupyter / PyCharm 用法

十等分使用独立源码 `按分布十等分_Jupyter源码.py`；原来的
`阈值与区间分析_Jupyter源码.py` 已恢复为固定阈值版本，两者互不覆盖。

在 Jupyter 中运行源码单元后：

```python
quantile_result = run_quantile_analysis(bins=10)
display_quantile_analysis(quantile_result)

# display_quantile_analysis 会同时生成全体十档明细表 quantile_detail
```

命令行或 PyCharm Terminal：

```powershell
python "按分布十等分_Jupyter源码.py" --quantile --quantile-bins 10
```

不加 `--quantile` 时仍执行原来的固定刻度分析流程。

如需改成五等分、二十等分等，只需修改 `按分布十等分_参数.json` 中的
`quantile_bins`，或在函数调用中传入 `bins`，不需要改计算源码。
