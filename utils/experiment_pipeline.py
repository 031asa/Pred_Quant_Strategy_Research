from __future__ import annotations

"""Streamlit实验页使用的纯内存可组合Pred处理流水线。"""

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import polars as pl

from utils.score_drift_workflow import monthly_curve_metrics
from utils.segmented_drift import alpha_from_half_life, build_causal_state


MODULE_GROUP_CONTRACT = "group_contract"
MODULE_GROUP_TIME = "group_time"
MODULE_SMOOTH = "smooth"
MODULE_TYPES = (MODULE_GROUP_CONTRACT, MODULE_GROUP_TIME, MODULE_SMOOTH)

DEFAULT_HALF_LIFE_CANDIDATES = (1, 2, 5, 10, 20, 40, 60, 90, 120)
DEFAULT_TIME_CUTOFF = "14:30"
TIME_SELECTION_BOTH = "both"
TIME_SELECTION_BEFORE = "before"
TIME_SELECTION_AFTER = "after"
TIME_SELECTIONS = (
    TIME_SELECTION_BOTH,
    TIME_SELECTION_BEFORE,
    TIME_SELECTION_AFTER,
)

INTERNAL_ROW_ID = "__web_row_id"
INTERNAL_DATETIME = "__web_datetime"
INTERNAL_CONTRACT = "__web_contract"
INTERNAL_PRED = "__web_pred"
INTERNAL_TAG = "__web_tag"
INTERNAL_DATE = "Date"
INTERNAL_TIME_GROUP = "__web_time_group"


class PipelineValidationError(ValueError):
    """输入或模块配置不满足流水线约束。"""


class PipelineStepError(RuntimeError):
    """带步骤和分组上下文的流水线执行错误。"""


@dataclass(frozen=True)
class SchemaMapping:
    datetime: str
    contract: str
    pred: str
    tag: str


@dataclass(frozen=True)
class GroupSmoothingSummary:
    group: dict[str, str]
    input_rows: int
    output_rows: int
    input_trading_days: int
    output_trading_days: int
    selected_half_life: int
    alpha: float
    balance_offset: float
    development_short_share: float
    target_met: bool
    selection_mode: str
    cold_start_days_effective: int
    test_days: int


@dataclass(frozen=True)
class StepSummary:
    step_number: int
    module_type: str
    active_group_dimensions: tuple[str, ...]
    input_rows: int
    output_rows: int
    dropped_rows: int
    groups: tuple[GroupSmoothingSummary, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    data: pl.DataFrame
    input_rows: int
    output_rows: int
    input_start_date: str
    input_end_date: str
    output_start_date: str
    output_end_date: str
    input_trading_days: int
    output_trading_days: int
    output_scope_label: str
    step_summaries: tuple[StepSummary, ...]
    normalized_modules: tuple[dict[str, object], ...]

    @property
    def dropped_rows(self) -> int:
        return self.input_rows - self.output_rows

    def metadata(self) -> dict[str, object]:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "dropped_rows": self.dropped_rows,
            "input_date_range": [self.input_start_date, self.input_end_date],
            "output_date_range": [self.output_start_date, self.output_end_date],
            "input_trading_days": self.input_trading_days,
            "output_trading_days": self.output_trading_days,
            "output_scope": self.output_scope_label,
            "modules": list(self.normalized_modules),
            "steps": [
                {
                    **{
                        key: value
                        for key, value in asdict(step).items()
                        if key != "groups"
                    },
                    "groups": [asdict(group) for group in step.groups],
                }
                for step in self.step_summaries
            ],
        }


ProgressCallback = Callable[[int, int, str], None]


_FIELD_ALIASES = {
    "datetime": ("datetime", "date_time", "timestamp", "time"),
    "contract": ("contract", "symbol", "instrument", "ticker"),
    "pred": ("pred", "prediction", "signal", "score"),
    "tag": ("tag", "return", "ret", "target", "label"),
}


def infer_schema_mapping(columns: Sequence[str]) -> dict[str, str | None]:
    """按大小写不敏感别名推断四个必需字段，不确定时返回None。"""

    by_lower: dict[str, list[str]] = {}
    for column in columns:
        by_lower.setdefault(column.casefold(), []).append(column)
    inferred: dict[str, str | None] = {}
    for logical_name, aliases in _FIELD_ALIASES.items():
        matches = [value for alias in aliases for value in by_lower.get(alias, [])]
        inferred[logical_name] = matches[0] if len(matches) == 1 else None
    return inferred


def _parse_cutoff(value: object) -> str:
    cutoff = str(value)
    try:
        parsed = datetime.strptime(cutoff, "%H:%M")
    except ValueError as exc:
        raise PipelineValidationError(f"分时切点{cutoff!r}不是有效的HH:MM。") from exc
    return parsed.strftime("%H:%M")


def _normal_smoothing_config(module: Mapping[str, object]) -> dict[str, object]:
    mode = str(module.get("selection_mode", "auto"))
    if mode not in {"auto", "fixed"}:
        raise PipelineValidationError("平滑选参模式只能是auto或fixed。")
    candidates = tuple(
        sorted({int(value) for value in module.get("half_life_candidates", DEFAULT_HALF_LIFE_CANDIDATES)})
    )
    if not candidates or candidates[0] <= 0:
        raise PipelineValidationError("半衰期候选必须是正整数。")
    fixed_half_life = int(module.get("fixed_half_life", candidates[0]))
    if fixed_half_life <= 0:
        raise PipelineValidationError("固定半衰期必须是正整数。")
    cold_start_days = int(module.get("cold_start_days", 20))
    test_fraction = float(module.get("test_fraction", 0.10))
    short_rate_target = float(module.get("short_rate_target", 0.50))
    short_rate_tolerance = float(module.get("short_rate_tolerance", 0.01))
    if cold_start_days <= 0:
        raise PipelineValidationError("冷启动交易日数必须为正整数。")
    if not 0 < test_fraction < 0.5:
        raise PipelineValidationError("测试集比例必须在0与0.5之间。")
    if not 0 <= short_rate_target <= 1:
        raise PipelineValidationError("目标做空比例必须在0与1之间。")
    if not 0 <= short_rate_tolerance <= 0.5:
        raise PipelineValidationError("做空比例容差必须在0与0.5之间。")
    score_min = float(module.get("score_min", 0.0))
    score_max = float(module.get("score_max", 1.0))
    if score_min >= score_max:
        raise PipelineValidationError("Score下限必须小于Score上限。")
    return {
        "selection_mode": mode,
        "half_life_candidates": list(candidates),
        "fixed_half_life": fixed_half_life,
        "cold_start_days": cold_start_days,
        "test_fraction": test_fraction,
        "short_rate_target": short_rate_target,
        "short_rate_tolerance": short_rate_tolerance,
        "score_offset": float(module.get("score_offset", 0.5)),
        "score_min": score_min,
        "score_max": score_max,
    }


def normalize_pipeline_modules(
    modules: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """校验并规范化模块；UI标识不参与缓存键。"""

    normalized: list[dict[str, object]] = []
    seen_group_modules: set[str] = set()
    for module in modules:
        module_type = str(module.get("type", ""))
        if module_type not in MODULE_TYPES:
            raise PipelineValidationError(f"未知模块类型：{module_type!r}。")
        enabled = bool(module.get("enabled", True))
        item: dict[str, object] = {"type": module_type, "enabled": enabled}
        if module_type in {MODULE_GROUP_CONTRACT, MODULE_GROUP_TIME}:
            if module_type in seen_group_modules:
                label = "按合约" if module_type == MODULE_GROUP_CONTRACT else "按时段"
                raise PipelineValidationError(f"每条流水线最多添加一次{label}预处理。")
            seen_group_modules.add(module_type)
        if module_type == MODULE_GROUP_TIME:
            item["cutoff"] = _parse_cutoff(module.get("cutoff", DEFAULT_TIME_CUTOFF))
            selection = str(module.get("time_selection", TIME_SELECTION_BOTH))
            if selection not in TIME_SELECTIONS:
                raise PipelineValidationError(
                    "按时段预处理的数据选择只能是both、before或after。"
                )
            item["time_selection"] = selection
        elif module_type == MODULE_SMOOTH:
            item.update(_normal_smoothing_config(module))
        normalized.append(item)
    return tuple(normalized)


def pipeline_cache_key(
    file_hash: str,
    mapping: SchemaMapping,
    modules: Iterable[Mapping[str, object]],
) -> str:
    payload = {
        "file_hash": file_hash,
        "mapping": asdict(mapping),
        "modules": normalize_pipeline_modules(modules),
        "engine_version": 1,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def hash_uploaded_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def move_pipeline_module(
    modules: Sequence[Mapping[str, object]], index: int, delta: int
) -> list[dict[str, object]]:
    """返回模块顺序副本；越界移动保持原顺序。"""

    copied = [dict(module) for module in modules]
    target = index + delta
    if 0 <= index < len(copied) and 0 <= target < len(copied):
        copied[index], copied[target] = copied[target], copied[index]
    return copied


def delete_pipeline_module(
    modules: Sequence[Mapping[str, object]], index: int
) -> list[dict[str, object]]:
    """返回删除指定位置后的模块副本。"""

    if not 0 <= index < len(modules):
        return [dict(module) for module in modules]
    return [dict(module) for position, module in enumerate(modules) if position != index]


def prepare_uploaded_frame(source: pl.DataFrame, mapping: SchemaMapping) -> pl.DataFrame:
    """验证字段并转换为不依赖上传列名的内部结构。"""

    mapped_columns = [mapping.datetime, mapping.contract, mapping.pred, mapping.tag]
    missing = [column for column in mapped_columns if column not in source.columns]
    if missing:
        raise PipelineValidationError(f"上传数据缺少字段：{', '.join(missing)}。")
    if len(set(mapped_columns)) != 4:
        raise PipelineValidationError("Datetime、Contract、pred和Tag必须映射到四个不同字段。")
    forbidden = [column for column in source.columns if column.startswith("__web_")]
    if forbidden:
        raise PipelineValidationError(
            "上传数据包含保留前缀__web_字段：" + ", ".join(forbidden)
        )

    selected = source.select(mapped_columns).with_row_index(INTERNAL_ROW_ID)
    datetime_dtype = selected.schema[mapping.datetime]
    if datetime_dtype == pl.String:
        datetime_expr = pl.col(mapping.datetime).str.to_datetime(strict=False)
    elif datetime_dtype == pl.Date:
        datetime_expr = pl.col(mapping.datetime).cast(pl.Datetime)
    elif datetime_dtype.base_type() == pl.Datetime:
        datetime_expr = pl.col(mapping.datetime)
    else:
        raise PipelineValidationError(
            f"Datetime字段类型为{datetime_dtype}，必须是Datetime、Date或可解析字符串。"
        )

    prepared = selected.select(
        pl.col(INTERNAL_ROW_ID),
        datetime_expr.alias(INTERNAL_DATETIME),
        pl.col(mapping.contract).cast(pl.String, strict=False).alias(INTERNAL_CONTRACT),
        pl.col(mapping.pred).cast(pl.Float64, strict=False).alias(INTERNAL_PRED),
        pl.col(mapping.tag).cast(pl.Float64, strict=False).alias(INTERNAL_TAG),
    ).with_columns(pl.col(INTERNAL_DATETIME).dt.date().alias(INTERNAL_DATE))

    null_counts = prepared.select(
        [pl.col(column).is_null().sum().alias(column) for column in prepared.columns]
    ).row(0, named=True)
    invalid = {column: count for column, count in null_counts.items() if count}
    if invalid:
        detail = ", ".join(f"{column}={count}" for column, count in invalid.items())
        raise PipelineValidationError(f"必需字段存在空值或无法转换的值：{detail}。")
    if prepared.height == 0:
        raise PipelineValidationError("上传数据没有有效行。")
    if not np.isfinite(prepared[INTERNAL_PRED].to_numpy()).all():
        raise PipelineValidationError("pred包含NaN或无穷值。")
    if not np.isfinite(prepared[INTERNAL_TAG].to_numpy()).all():
        raise PipelineValidationError("Tag包含NaN或无穷值。")
    return prepared.sort(INTERNAL_ROW_ID)


def _date_summary(data: pl.DataFrame) -> tuple[str, str, int]:
    dates = data[INTERNAL_DATE]
    return str(dates.min()), str(dates.max()), int(dates.n_unique())


def _effective_cold_start(dates: Sequence[object], minimum_days: int) -> int:
    first = dates[0]
    first_month = (first.year, first.month)
    next_month_index = next(
        (index for index, value in enumerate(dates) if (value.year, value.month) != first_month),
        len(dates),
    )
    return max(minimum_days, next_month_index)


def _group_label(group_columns: Sequence[str], key: object) -> dict[str, str]:
    if not group_columns:
        return {"总体": "全部"}
    values = key if isinstance(key, tuple) else (key,)
    labels = {
        INTERNAL_CONTRACT: "合约",
        INTERNAL_TIME_GROUP: "时段",
    }
    return {labels[column]: str(value) for column, value in zip(group_columns, values)}


def _attach_candidate_state(
    part: pl.DataFrame,
    state: pl.DataFrame,
    *,
    balance_offset: float,
    test_start_index: int,
) -> pl.DataFrame:
    return (
        part.join(state, on=INTERNAL_DATE, how="left", validate="m:1")
        .filter(pl.col("slow_baseline").is_finite() & pl.col("recent_center").is_finite())
        .with_columns(
            (pl.col("recent_center") + pl.lit(balance_offset)).alias("effective_center"),
            (pl.col("date_index") >= test_start_index).alias("is_test"),
        )
        .with_columns(
            (pl.col("score") - pl.col("effective_center")).alias("adjusted_residual"),
            (pl.col("score") < pl.col("effective_center")).alias("adjusted_short"),
        )
    )


def _smooth_one_group(
    part: pl.DataFrame,
    *,
    group: dict[str, str],
    config: Mapping[str, object],
) -> tuple[pl.DataFrame, GroupSmoothingSummary]:
    score_offset = float(config["score_offset"])
    score_min = float(config["score_min"])
    score_max = float(config["score_max"])
    prepared = part.with_columns(
        (pl.col(INTERNAL_PRED) + score_offset)
        .clip(score_min, score_max)
        .alias("score"),
        pl.col(INTERNAL_DATETIME).dt.strftime("%Y-%m").alias("Month"),
    )
    daily = (
        prepared.group_by(INTERNAL_DATE)
        .agg(pl.col("score").median().alias("daily_score_median"))
        .sort(INTERNAL_DATE)
    )
    dates = daily[INTERNAL_DATE].to_list()
    n_dates = len(dates)
    cold_start = _effective_cold_start(dates, int(config["cold_start_days"]))
    test_days = int(math.ceil(n_dates * float(config["test_fraction"])))
    test_start_index = n_dates - test_days
    if n_dates <= cold_start + test_days:
        raise PipelineValidationError(
            f"交易日只有{n_dates}天，无法同时容纳{cold_start}天冷启动和{test_days}天测试集。"
        )

    candidate_half_lives = (
        [int(config["fixed_half_life"])]
        if config["selection_mode"] == "fixed"
        else [int(value) for value in config["half_life_candidates"]]
    )
    candidate_rows: list[dict[str, object]] = []
    states: dict[int, pl.DataFrame] = {}
    for half_life in candidate_half_lives:
        arrays = build_causal_state(
            daily["daily_score_median"].to_numpy(), cold_start, half_life
        )
        state = pl.DataFrame(
            {
                INTERNAL_DATE: dates,
                "date_index": np.arange(n_dates, dtype=np.int64),
                **arrays,
            }
        ).with_columns(pl.col(INTERNAL_DATE).cast(pl.Date))
        states[half_life] = state
        uncalibrated = _attach_candidate_state(
            prepared,
            state,
            balance_offset=0.0,
            test_start_index=test_start_index,
        ).filter(pl.col("date_index") < test_start_index)
        if uncalibrated.height == 0:
            raise PipelineValidationError("冷启动后没有可用于开发段选参的样本。")
        balance_offset = float(
            (uncalibrated["score"] - uncalibrated["recent_center"]).median()
        )
        candidate = _attach_candidate_state(
            prepared,
            state,
            balance_offset=balance_offset,
            test_start_index=test_start_index,
        ).filter(pl.col("date_index") < test_start_index)
        short_share = float(candidate["adjusted_short"].mean())
        area, slope, _ = monthly_curve_metrics(candidate, "adjusted_residual")
        target_distance = abs(short_share - float(config["short_rate_target"]))
        candidate_rows.append(
            {
                "half_life": half_life,
                "balance_offset": balance_offset,
                "short_share": short_share,
                "target_distance": target_distance,
                "target_met": target_distance <= float(config["short_rate_tolerance"]),
                "area": area,
                "slope": slope,
            }
        )

    if config["selection_mode"] == "fixed":
        selected = candidate_rows[0]
    else:
        feasible = [row for row in candidate_rows if row["target_met"]]
        selected = min(
            feasible,
            key=lambda row: (row["area"], row["slope"], row["half_life"]),
        ) if feasible else min(
            candidate_rows,
            key=lambda row: (
                row["target_distance"],
                row["area"],
                row["slope"],
                row["half_life"],
            ),
        )

    selected_half_life = int(selected["half_life"])
    final = _attach_candidate_state(
        prepared,
        states[selected_half_life],
        balance_offset=float(selected["balance_offset"]),
        test_start_index=test_start_index,
    ).with_columns(pl.col("adjusted_residual").alias(INTERNAL_PRED))
    output = final.select(part.columns).sort(INTERNAL_ROW_ID)
    summary = GroupSmoothingSummary(
        group=group,
        input_rows=part.height,
        output_rows=output.height,
        input_trading_days=n_dates,
        output_trading_days=int(output[INTERNAL_DATE].n_unique()),
        selected_half_life=selected_half_life,
        alpha=alpha_from_half_life(selected_half_life),
        balance_offset=float(selected["balance_offset"]),
        development_short_share=float(selected["short_share"]),
        target_met=bool(selected["target_met"]),
        selection_mode=str(config["selection_mode"]),
        cold_start_days_effective=cold_start,
        test_days=test_days,
    )
    return output, summary


def _apply_smoothing(
    data: pl.DataFrame,
    *,
    group_columns: Sequence[str],
    config: Mapping[str, object],
) -> tuple[pl.DataFrame, tuple[GroupSmoothingSummary, ...]]:
    if group_columns:
        partitions = data.partition_by(
            list(group_columns), as_dict=True, maintain_order=True
        )
    else:
        partitions = {(): data}
    outputs: list[pl.DataFrame] = []
    summaries: list[GroupSmoothingSummary] = []
    for key, part in partitions.items():
        group = _group_label(group_columns, key)
        try:
            output, summary = _smooth_one_group(part, group=group, config=config)
        except Exception as exc:
            group_text = "、".join(f"{name}={value}" for name, value in group.items())
            raise PipelineValidationError(f"分组[{group_text}]平滑失败：{exc}") from exc
        outputs.append(output)
        summaries.append(summary)
    combined = pl.concat(outputs, how="vertical").sort(INTERNAL_ROW_ID)
    if combined[INTERNAL_ROW_ID].n_unique() != combined.height:
        raise AssertionError("平滑后行号发生重复。")
    return combined, tuple(summaries)


def execute_pipeline(
    prepared: pl.DataFrame,
    modules: Iterable[Mapping[str, object]],
    *,
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    """在同一份内部数据上按顺序执行模块，不读写文件。"""

    normalized = normalize_pipeline_modules(modules)
    active_modules = [module for module in normalized if module["enabled"]]
    data = prepared.clone()
    input_start, input_end, input_days = _date_summary(data)
    input_rows = data.height
    active_group_columns: list[str] = []
    output_scope_label = "总体"
    step_summaries: list[StepSummary] = []

    for active_index, module in enumerate(active_modules, start=1):
        module_type = str(module["type"])
        before_rows = data.height
        if progress_callback:
            progress_callback(
                active_index - 1,
                len(active_modules),
                f"步骤{active_index}/{len(active_modules)}：{module_type}",
            )
        try:
            group_summaries: tuple[GroupSmoothingSummary, ...] = ()
            if module_type == MODULE_GROUP_CONTRACT:
                active_group_columns.append(INTERNAL_CONTRACT)
            elif module_type == MODULE_GROUP_TIME:
                cutoff = str(module["cutoff"])
                selection = str(module["time_selection"])
                data = data.with_columns(
                    pl.when(pl.col(INTERNAL_DATETIME).dt.strftime("%H:%M") < cutoff)
                    .then(pl.lit(f"{cutoff}前"))
                    .otherwise(pl.lit(f"{cutoff}及以后"))
                    .alias(INTERNAL_TIME_GROUP)
                )
                if selection == TIME_SELECTION_BEFORE:
                    data = data.filter(pl.col(INTERNAL_TIME_GROUP) == f"{cutoff}前")
                    output_scope_label = f"{cutoff}前"
                elif selection == TIME_SELECTION_AFTER:
                    data = data.filter(
                        pl.col(INTERNAL_TIME_GROUP) == f"{cutoff}及以后"
                    )
                    output_scope_label = f"{cutoff}及以后"
                else:
                    output_scope_label = f"总体（{cutoff}前后均保留）"
                if data.height == 0:
                    raise PipelineValidationError(
                        f"按{cutoff}选择{output_scope_label}后没有剩余样本。"
                    )
                active_group_columns.append(INTERNAL_TIME_GROUP)
            elif module_type == MODULE_SMOOTH:
                data, group_summaries = _apply_smoothing(
                    data,
                    group_columns=active_group_columns,
                    config=module,
                )
        except Exception as exc:
            raise PipelineStepError(
                f"第{active_index}步({module_type})执行失败：{exc}"
            ) from exc
        step_summaries.append(
            StepSummary(
                step_number=active_index,
                module_type=module_type,
                active_group_dimensions=tuple(active_group_columns),
                input_rows=before_rows,
                output_rows=data.height,
                dropped_rows=before_rows - data.height,
                groups=group_summaries,
            )
        )
        if progress_callback:
            progress_callback(
                active_index,
                len(active_modules),
                f"步骤{active_index}/{len(active_modules)}完成",
            )

    output_start, output_end, output_days = _date_summary(data)
    return PipelineResult(
        data=data.sort(INTERNAL_ROW_ID),
        input_rows=input_rows,
        output_rows=data.height,
        input_start_date=input_start,
        input_end_date=input_end,
        output_start_date=output_start,
        output_end_date=output_end,
        input_trading_days=input_days,
        output_trading_days=output_days,
        output_scope_label=output_scope_label,
        step_summaries=tuple(step_summaries),
        normalized_modules=normalized,
    )
