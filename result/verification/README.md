# 交接验收记录

验收日期：2026-09-28。基于Gitea原提交 `0425b8e45a4e14973e2ba37d893a26b0767a0b72`，在独立Linux工作目录补齐数据后验证。环境为WSL 2 Ubuntu 24.04、Conda `pred-quant-research`、Python 3.12.14。

- 原有43项unittest通过，包含算法因果性、分段独立性、阈值等号、多空收益符号、零Tag分母、字体和Streamlit上传/A/B交互测试。
- 三个批量入口均在独立临时目录退出码为0，原交付研究结果未被覆盖。
- 原始Parquet确认1,728,000行、600交易日、12合约，2024-01-03至2026-06-30；五个字段均无null。
- 再生成的参数JSON与原参数完全一致。
- 再生成的最终Parquet为1,667,520行，与原结果逐字段、逐行相符；Pred最大绝对误差0、方向差异0，完整文件SHA256也一致。
- 再生成26张PNG，路径集合与交付图一致，全部通过图像文件完整性验证。未要求跨平台PNG逐字节一致。

完整数据核对在 `reproduction.json`，三个脚本的输出在同目录 `.log`，实际安装依赖版本在 `environment_snapshot.json`。后者只是验收快照，项目可移植配置为根目录 `environment.yml`。

复现指令：`python scripts/reproduce_handover.py`。该工具保留当前研究结果，仅更新本目录下的运行记录。再次运行会改变耗时和运行路径等记录，因此 `docs/checksums.sha256` 是本次交付时点的完整性快照；建议克隆后先执行 `python scripts/verify_handover.py`，再运行实验。

本次验证对象是当前三步分析与Streamlit流程；历史99份材料保留原貌，未逐一重跑。没有训练模型源码或Tag生成代码，因此验收不代表原模型训练与真实交易净值可以复现。
