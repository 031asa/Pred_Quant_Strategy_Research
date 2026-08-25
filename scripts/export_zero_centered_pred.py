from __future__ import annotations

"""步骤二：输出可以直接用 0 判断做多/做空的最终 Pred Parquet。

输入、输出和字段口径集中列在本文件顶部；状态重建位于 utils。
"""

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.zero_centered_export import export_zero_centered_pred


# 输入配置
CONFIG_PATH = PROJECT_ROOT / "config.json"
SELECTED_PARAMETERS_PATH = (
    PROJECT_ROOT / "result" / "parameters"
    / "segmented_score_drift_selected_parameters.json"
)
DATE_COLUMN = "Date"
# 原始数据路径、字段名、14:30切点和Score换算参数统一从CONFIG_PATH读取。
EXPECTED_EFFECTIVE_SAMPLE_COUNT = 1_667_520

# 输出配置
OUTPUT_PARQUET_PATH = (
    PROJECT_ROOT / "result" / "data"
    / "pred_eval_segmented_drift_adjusted.parquet"
)


def main() -> None:
    """读取项目配置并通过命名参数调用最终 Pred 导出接口。"""

    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    source_parquet_path = PROJECT_ROOT / config["source_path"]

    result = export_zero_centered_pred(
        source_path=source_parquet_path,
        selected_parameters_path=SELECTED_PARAMETERS_PATH,
        output_path=OUTPUT_PARQUET_PATH,
        date_column=DATE_COLUMN,
        datetime_column=config["datetime_column"],
        pred_column=config["pred_column"],
        time_cutoff=config["time_cutoff"],
        score_offset=float(config["score_offset"]),
        score_min=float(config["score_min"]),
        score_max=float(config["score_max"]),
        expected_sample_count=EXPECTED_EFFECTIVE_SAMPLE_COUNT,
    )

    print("\n步骤二完成：零中心最终 Pred 导出")
    print(f"输入 Parquet：{source_parquet_path}")
    print(f"输入参数：{SELECTED_PARAMETERS_PATH}")
    print(f"输出 Parquet：{result.output_path}")
    print(f"有效样本量：{result.sample_count:,}")
    print(f"pred 范围：[{result.pred_min:.12f}, {result.pred_max:.12f}]")
    print(f"做空占比（pred < 0）：{result.short_share:.12%}")


if __name__ == "__main__":
    main()
