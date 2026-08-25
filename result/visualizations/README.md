# 当前可视化索引

本目录保留26张汇报图，路径全部为ASCII英文，图中文字保持中文。

## `before_adjustment/`：调整前（未平滑）

- `overall/overall_ventile_zero_threshold_metrics.png`
- `before_1430/before_1430_ventile_zero_threshold_metrics.png`
- `after_1430/after_1430_ventile_zero_threshold_metrics.png`
- `overall/overall_decile_zero_threshold_cumulative_return.png`
- `before_1430/before_1430_decile_zero_threshold_cumulative_return.png`
- `after_1430/after_1430_decile_zero_threshold_cumulative_return.png`
- `overall/paired_decile_cumulative_return.png`、`before_1430/paired_decile_cumulative_return.png`、`after_1430/paired_decile_cumulative_return.png`：三个范围各自的5组对称档位累计收益曲线。
- `overall/pred_distribution.png`、`before_1430/pred_distribution.png`、`after_1430/pred_distribution.png`：三个范围各自的Pred密度直方图与正态曲线对照。

三张六联图均展示20等分的平均策略收益、每日开仓次数中位数、平均每日收益、胜率、平均盈亏比和期望R倍数；三张累计收益图与调整后使用相同的10等分、零阈值方向和按交易日单利累计口径。

## `after_adjustment/`：调整后（已平滑）

- `overall/overall_ventile_zero_threshold_metrics.png`
- `before_1430/before_1430_ventile_zero_threshold_metrics.png`
- `after_1430/after_1430_ventile_zero_threshold_metrics.png`
- `overall/overall_decile_zero_threshold_cumulative_return.png`
- `before_1430/before_1430_decile_zero_threshold_cumulative_return.png`
- `after_1430/after_1430_decile_zero_threshold_cumulative_return.png`
- `overall/paired_decile_cumulative_return.png`、`before_1430/paired_decile_cumulative_return.png`、`after_1430/paired_decile_cumulative_return.png`：三个范围各自的5组对称档位累计收益曲线。
- `overall/pred_distribution.png`、`before_1430/pred_distribution.png`、`after_1430/pred_distribution.png`：三个范围各自的Pred密度直方图与正态曲线对照。

## `drift_comparison/`：平滑前后月度漂移

- `before_1430_monthly_score_drift.png`
- `after_1430_monthly_score_drift.png`

## 统一口径

- 调整前后使用完全相同的1,667,520条有效样本，期间为2024-02-01至2026-06-30。
- 每条样本严格以`pred < 0`做空、`pred >= 0`做多；不在分析阶段读取或比较动态慢速基准。
- 20/10等分只负责按Pred强弱分组，不负责决定交易方向。
- 策略收益：做空为`-Tag`，做多为`Tag`。
- 胜率排除`Tag=0`；盈亏比为平均正策略收益除以平均负策略收益绝对值；期望R倍数为`胜率×盈亏比-(1-胜率)`，0为盈亏平衡。
- 每日开仓次数中位数基于范围内全部交易日计算，无交易日按0次计入；平均每日收益为平均策略收益乘每日开仓次数中位数，表示典型频次下的逐笔收益日内求和，不等同于资金均分账户收益。
- 累计收益为档内样本日内等权后按交易日累计求和；某档无交易的日期以0收益补齐。
- 配对收益按Q01空+Q10多、Q02空+Q09多、Q03空+Q08多、Q04空+Q07多、Q05空+Q06多构造；两腿日均收益每日各占50%，合并后按交易日单利累计。
- 分布图使用100档密度直方图，并叠加与样本具有相同均值和标准差的正态密度曲线。
- 未扣除手续费、滑点和资金占用。
