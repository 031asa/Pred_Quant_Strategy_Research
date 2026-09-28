# 历史研究归档

当前可执行版本在仓库根目录。此处保留历史文件原貌，不把旧口径与当前结果混合。

- [legacy_research](legacy_research/)：99份原始研究资料，包括源码、参数、CSV、Excel、图、PDF、Word、HTML和原交付ZIP。
- [chinese_summary_snapshot](chinese_summary_snapshot/)：较早中文项目43份文件，包含尚未提交到旧Git历史中的本地状态。
- [chinese_summary_history.bundle](chinese_summary_history.bundle)：旧中文仓库的完整Git历史。可在新目录使用 `git clone chinese_summary_history.bundle recovered-chinese-history` 读取；该历史不包含后来未提交的改动，需结合上面的快照。

中文快照中两份重复Parquet不再复制；`docs/source_manifest.json` 的 `deduplicated_assets` 提供旧路径到当前原件的映射。若确需运行该旧版本，应在另一份工作副本内按映射放回文件，并核对其旧依赖和路径。

主题索引、阈值端点、不同胜率分母、单复利、以及旧“总体漂移”README与输出错配的说明，都在 [交接说明](../docs/HANDOVER.md)。历史脚本可能含原电脑绝对路径，部分Excel构建依赖专用工具，未宣称全部旧报告在新环境可一键重建。
