from __future__ import annotations

"""Pred双组可组合流水线实验页。运行：streamlit run streamlit_app.py"""

from dataclasses import asdict
from datetime import datetime, time
import io
import json
from typing import Any
from uuid import uuid4

import polars as pl
import streamlit as st

from utils.experiment_pipeline import (
    DEFAULT_HALF_LIFE_CANDIDATES,
    DEFAULT_TIME_CUTOFF,
    INTERNAL_CONTRACT,
    MODULE_GROUP_CONTRACT,
    MODULE_GROUP_TIME,
    MODULE_SMOOTH,
    PipelineResult,
    PipelineValidationError,
    SchemaMapping,
    TIME_SELECTION_AFTER,
    TIME_SELECTION_BEFORE,
    TIME_SELECTION_BOTH,
    delete_pipeline_module,
    execute_pipeline,
    hash_uploaded_bytes,
    infer_schema_mapping,
    normalize_pipeline_modules,
    pipeline_cache_key,
    prepare_uploaded_frame,
    move_pipeline_module,
)
from utils.web_charts import (
    CHART_CUMULATIVE,
    CHART_DISTRIBUTION,
    CHART_LABELS,
    CHART_METRICS,
    CHART_PAIRED,
    SCOPE_OVERALL,
)
from utils.plotly_charts import build_experiment_plotly_chart


st.set_page_config(
    page_title="Pred双组流水线实验",
    page_icon="📈",
    layout="wide",
)


MODULE_LABELS = {
    MODULE_GROUP_CONTRACT: "按合约预处理",
    MODULE_GROUP_TIME: "按时段预处理",
    MODULE_SMOOTH: "因果动态基准平滑",
}
TIME_SELECTION_LABELS = {
    TIME_SELECTION_BOTH: "前后都保留并分别处理",
    TIME_SELECTION_BEFORE: "仅保留切点前",
    TIME_SELECTION_AFTER: "仅保留切点及以后",
}
CHART_OPTIONS = (
    CHART_METRICS,
    CHART_CUMULATIVE,
    CHART_PAIRED,
    CHART_DISTRIBUTION,
)
RESULT_LAYOUT_STACKED = "上下完整"
RESULT_LAYOUT_SIDE_BY_SIDE = "左右并排"


def _new_module(module_type: str) -> dict[str, Any]:
    module: dict[str, Any] = {
        "id": uuid4().hex,
        "type": module_type,
        "enabled": True,
    }
    if module_type == MODULE_GROUP_TIME:
        module["cutoff"] = DEFAULT_TIME_CUTOFF
        module["time_selection"] = TIME_SELECTION_BOTH
    elif module_type == MODULE_SMOOTH:
        module.update(
            {
                "selection_mode": "auto",
                "half_life_candidates": list(DEFAULT_HALF_LIFE_CANDIDATES),
                "fixed_half_life": 1,
                "cold_start_days": 20,
                "test_fraction": 0.10,
                "short_rate_target": 0.50,
                "short_rate_tolerance": 0.01,
                "score_offset": 0.5,
                "score_min": 0.0,
                "score_max": 1.0,
            }
        )
    return module


def _ensure_state() -> None:
    defaults = {
        "pipelines": {"A": [], "B": []},
        "pipeline_results": {},
        "pipeline_result_keys": {},
        "pipeline_cache": {},
        "chart_cache": {},
        "executed_snapshot": None,
        "last_file_hash": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(show_spinner=False, max_entries=2)
def _read_uploaded_parquet(file_bytes: bytes) -> pl.DataFrame:
    return pl.read_parquet(io.BytesIO(file_bytes))


def _time_value(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def _move_module(experiment: str, index: int, delta: int) -> None:
    st.session_state.pipelines[experiment] = move_pipeline_module(
        st.session_state.pipelines[experiment], index, delta
    )


def _delete_module(experiment: str, index: int) -> None:
    st.session_state.pipelines[experiment] = delete_pipeline_module(
        st.session_state.pipelines[experiment], index
    )


def _render_smoothing_settings(module: dict[str, Any], key_prefix: str) -> None:
    mode = st.selectbox(
        "选参方式",
        options=("auto", "fixed"),
        index=0 if module.get("selection_mode", "auto") == "auto" else 1,
        format_func=lambda value: "候选集自动选择" if value == "auto" else "固定半衰期",
        key=f"{key_prefix}_selection_mode",
    )
    module["selection_mode"] = mode
    if mode == "auto":
        candidates = st.multiselect(
            "候选半衰期（交易日）",
            options=list(DEFAULT_HALF_LIFE_CANDIDATES),
            default=[
                value
                for value in module.get(
                    "half_life_candidates", DEFAULT_HALF_LIFE_CANDIDATES
                )
                if value in DEFAULT_HALF_LIFE_CANDIDATES
            ],
            key=f"{key_prefix}_candidates",
        )
        module["half_life_candidates"] = candidates
    else:
        module["fixed_half_life"] = int(
            st.number_input(
                "固定半衰期（交易日）",
                min_value=1,
                max_value=500,
                value=int(module.get("fixed_half_life", 1)),
                step=1,
                key=f"{key_prefix}_fixed_half_life",
            )
        )
    with st.expander("高级参数", expanded=False):
        left, right = st.columns(2)
        with left:
            module["cold_start_days"] = int(
                st.number_input(
                    "最少冷启动交易日",
                    min_value=1,
                    max_value=500,
                    value=int(module.get("cold_start_days", 20)),
                    step=1,
                    key=f"{key_prefix}_cold_start",
                )
            )
            module["short_rate_target"] = float(
                st.number_input(
                    "目标做空比例",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(module.get("short_rate_target", 0.50)),
                    step=0.01,
                    format="%.2f",
                    key=f"{key_prefix}_target",
                )
            )
            module["score_offset"] = float(
                st.number_input(
                    "Pred转Score偏移",
                    value=float(module.get("score_offset", 0.5)),
                    step=0.01,
                    format="%.4f",
                    key=f"{key_prefix}_score_offset",
                )
            )
        with right:
            module["test_fraction"] = float(
                st.number_input(
                    "测试集比例",
                    min_value=0.01,
                    max_value=0.49,
                    value=float(module.get("test_fraction", 0.10)),
                    step=0.01,
                    format="%.2f",
                    key=f"{key_prefix}_test_fraction",
                )
            )
            module["short_rate_tolerance"] = float(
                st.number_input(
                    "做空比例容差",
                    min_value=0.0,
                    max_value=0.5,
                    value=float(module.get("short_rate_tolerance", 0.01)),
                    step=0.005,
                    format="%.3f",
                    key=f"{key_prefix}_tolerance",
                )
            )
            score_limits = st.columns(2)
            module["score_min"] = float(
                score_limits[0].number_input(
                    "Score下限",
                    value=float(module.get("score_min", 0.0)),
                    step=0.01,
                    format="%.2f",
                    key=f"{key_prefix}_score_min",
                )
            )
            module["score_max"] = float(
                score_limits[1].number_input(
                    "Score上限",
                    value=float(module.get("score_max", 1.0)),
                    step=0.01,
                    format="%.2f",
                    key=f"{key_prefix}_score_max",
                )
            )


def _render_pipeline_editor(experiment: str) -> None:
    modules: list[dict[str, Any]] = st.session_state.pipelines[experiment]
    st.markdown(f"#### 实验 {experiment}")
    add_columns = st.columns([3, 1])
    module_type = add_columns[0].selectbox(
        "新增模块",
        options=tuple(MODULE_LABELS),
        format_func=lambda value: MODULE_LABELS[value],
        key=f"add_type_{experiment}",
        label_visibility="collapsed",
    )
    if add_columns[1].button(
        "新增",
        key=f"add_{experiment}",
        width="stretch",
        type="secondary",
    ):
        existing = {module["type"] for module in modules}
        if module_type in {MODULE_GROUP_CONTRACT, MODULE_GROUP_TIME} and module_type in existing:
            st.warning(f"实验{experiment}中{MODULE_LABELS[module_type]}最多添加一次。")
        else:
            modules.append(_new_module(module_type))
            st.rerun()

    if not modules:
        st.info("当前为空流水线：运行后直接使用原始Pred，可作为对照组。")
        return

    for index, module in enumerate(modules):
        module_id = str(module["id"])
        with st.container(border=True):
            header = st.columns([0.8, 4.2, 0.7, 0.7, 0.8])
            module["enabled"] = header[0].checkbox(
                "启用",
                value=bool(module.get("enabled", True)),
                key=f"enabled_{experiment}_{module_id}",
            )
            header[1].markdown(
                f"**{index + 1}. {MODULE_LABELS[str(module['type'])]}**"
            )
            if header[2].button(
                "↑",
                disabled=index == 0,
                key=f"up_{experiment}_{module_id}",
                help="上移",
                width="stretch",
            ):
                _move_module(experiment, index, -1)
                st.rerun()
            if header[3].button(
                "↓",
                disabled=index == len(modules) - 1,
                key=f"down_{experiment}_{module_id}",
                help="下移",
                width="stretch",
            ):
                _move_module(experiment, index, 1)
                st.rerun()
            if header[4].button(
                "删除",
                key=f"delete_{experiment}_{module_id}",
                width="stretch",
            ):
                _delete_module(experiment, index)
                st.rerun()

            if module["type"] == MODULE_GROUP_CONTRACT:
                st.caption("从此步骤起，后续平滑增加合约分组维度。")
            elif module["type"] == MODULE_GROUP_TIME:
                selected_time = st.time_input(
                    "分时切点",
                    value=_time_value(str(module.get("cutoff", DEFAULT_TIME_CUTOFF))),
                    step=60,
                    key=f"cutoff_{experiment}_{module_id}",
                )
                module["cutoff"] = selected_time.strftime("%H:%M")
                module["time_selection"] = st.selectbox(
                    "数据选择",
                    options=(
                        TIME_SELECTION_BOTH,
                        TIME_SELECTION_BEFORE,
                        TIME_SELECTION_AFTER,
                    ),
                    index=(
                        TIME_SELECTION_BOTH,
                        TIME_SELECTION_BEFORE,
                        TIME_SELECTION_AFTER,
                    ).index(
                        str(module.get("time_selection", TIME_SELECTION_BOTH))
                    ),
                    format_func=lambda value: TIME_SELECTION_LABELS[value],
                    key=f"time_selection_{experiment}_{module_id}",
                )
                st.caption(
                    "选择在此步骤立即生效，后续平滑只使用保留数据；"
                    "切点时刻归入后半段，时间维度与已有合约维度叠加。"
                )
            else:
                _render_smoothing_settings(module, f"smooth_{experiment}_{module_id}")


def _render_chart_settings(experiment: str) -> dict[str, str]:
    with st.expander(f"实验{experiment}出图设置", expanded=True):
        chart_type = st.selectbox(
            "图片类型",
            options=CHART_OPTIONS,
            format_func=lambda value: CHART_LABELS[value],
            key=f"chart_type_{experiment}",
        )
        st.caption("图片自动使用该组流水线最终保留下来的全部数据。")
    return {"chart_type": chart_type}


def _mapping_select(
    logical_name: str,
    label: str,
    columns: list[str],
    inferred: dict[str, str | None],
    file_hash: str,
) -> str | None:
    options: list[str | None] = [None, *columns]
    default = inferred[logical_name]
    index = options.index(default) if default in options else 0
    return st.selectbox(
        label,
        options=options,
        index=index,
        format_func=lambda value: "— 请选择 —" if value is None else value,
        key=f"mapping_{file_hash}_{logical_name}",
    )


def _run_experiments(
    prepared: pl.DataFrame,
    *,
    file_hash: str,
    mapping: SchemaMapping,
    file_name: str,
) -> None:
    normalized = {
        experiment: normalize_pipeline_modules(st.session_state.pipelines[experiment])
        for experiment in ("A", "B")
    }
    active_counts = {
        experiment: sum(bool(module["enabled"]) for module in normalized[experiment])
        for experiment in ("A", "B")
    }
    total_steps = max(sum(active_counts.values()), 1)
    completed_before = 0
    progress = st.progress(0.0, text="准备执行实验")
    results: dict[str, PipelineResult] = {}
    result_keys: dict[str, str] = {}

    for experiment in ("A", "B"):
        cache_key = pipeline_cache_key(
            file_hash,
            mapping,
            st.session_state.pipelines[experiment],
        )
        result_keys[experiment] = cache_key
        if cache_key in st.session_state.pipeline_cache:
            results[experiment] = st.session_state.pipeline_cache[cache_key]
            completed_before += active_counts[experiment]
            progress.progress(
                min(completed_before / total_steps, 1.0),
                text=f"实验{experiment}命中缓存",
            )
            continue

        base = completed_before

        def update(current: int, total: int, message: str, *, _base: int = base) -> None:
            denominator = max(total, 1)
            fraction = (_base + current / denominator * active_counts[experiment]) / total_steps
            progress.progress(min(fraction, 1.0), text=f"实验{experiment}：{message}")

        result = execute_pipeline(
            prepared,
            st.session_state.pipelines[experiment],
            progress_callback=update,
        )
        st.session_state.pipeline_cache[cache_key] = result
        results[experiment] = result
        completed_before += active_counts[experiment]

    progress.progress(1.0, text="A/B实验执行完成")
    st.session_state.pipeline_results = results
    st.session_state.pipeline_result_keys = result_keys
    st.session_state.executed_snapshot = {
        "engine": "Pred双组可组合流水线",
        "engine_version": 1,
        "file": {"name": file_name, "sha256": file_hash},
        "mapping": asdict(mapping),
        "experiments": {
            experiment: {
                "cache_key": result_keys[experiment],
                "pipeline": list(normalized[experiment]),
                "result": results[experiment].metadata(),
            }
            for experiment in ("A", "B")
        },
    }


def _result_is_stale(
    experiment: str,
    *,
    file_hash: str,
    mapping: SchemaMapping,
) -> bool:
    executed_key = st.session_state.pipeline_result_keys.get(experiment)
    if not executed_key:
        return True
    return executed_key != pipeline_cache_key(
        file_hash, mapping, st.session_state.pipelines[experiment]
    )


def _render_result_card(
    experiment: str,
    *,
    result: PipelineResult,
    chart_settings: dict[str, str],
    file_hash: str,
    mapping: SchemaMapping,
) -> dict[str, object]:
    st.markdown(f"### 实验 {experiment}")
    if _result_is_stale(experiment, file_hash=file_hash, mapping=mapping):
        st.warning("流水线配置已修改；当前展示仍是上一次点击“执行A/B实验”的结果。")
    with st.expander("步骤与自动选参明细", expanded=False):
        st.json(result.metadata())

    result_key = st.session_state.pipeline_result_keys[experiment]
    chart_key_payload = {
        "result_key": result_key,
        **chart_settings,
        "experiment": experiment,
    }
    chart_key = json.dumps(chart_key_payload, sort_keys=True, ensure_ascii=False)
    if chart_key not in st.session_state.chart_cache:
        with st.spinner(f"正在生成实验{experiment}图片……"):
            st.session_state.chart_cache[chart_key] = build_experiment_plotly_chart(
                result,
                experiment_name=experiment,
                chart_type=chart_settings["chart_type"],
                scope=SCOPE_OVERALL,
                cutoff=DEFAULT_TIME_CUTOFF,
            )
    chart = st.session_state.chart_cache[chart_key]
    st.plotly_chart(
        chart.figure,
        width="stretch",
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"experiment_{experiment}_{chart.chart_type}",
                "scale": 2,
            },
        },
        key=f"plotly_{experiment}_{chart_key}",
    )
    st.caption(
        f"图片样本量 {chart.rows:,}｜交易日 {chart.trading_days:,}｜"
        f"数据范围 {result.output_scope_label}｜PNG请使用图表右上角相机按钮下载"
    )
    return {
        "type": chart.chart_type,
        "data_scope": result.output_scope_label,
        "rows": chart.rows,
        "trading_days": chart.trading_days,
    }


def main() -> None:
    _ensure_state()
    st.title("Pred 双组流水线实验")
    st.caption(
        "同一份原始Parquet分别进入A/B流水线；预处理增加后续平滑的分组维度，模块顺序会改变结果。"
    )

    uploaded = st.file_uploader("上传原始 Parquet", type=("parquet", "pq"))
    if uploaded is None:
        st.info("请先上传包含Datetime、Contract、pred和Tag的Parquet文件。")
        return

    file_bytes = uploaded.getvalue()
    file_hash = hash_uploaded_bytes(file_bytes)
    if st.session_state.last_file_hash != file_hash:
        st.session_state.last_file_hash = file_hash
        st.session_state.pipeline_results = {}
        st.session_state.pipeline_result_keys = {}
        st.session_state.pipeline_cache = {}
        st.session_state.chart_cache = {}
        st.session_state.executed_snapshot = None
    try:
        source = _read_uploaded_parquet(file_bytes)
    except Exception as exc:
        st.error(f"Parquet读取失败：{exc}")
        return

    inferred = infer_schema_mapping(source.columns)
    st.markdown("### 字段确认")
    mapping_columns = st.columns(4)
    with mapping_columns[0]:
        datetime_column = _mapping_select(
            "datetime", "Datetime", source.columns, inferred, file_hash
        )
    with mapping_columns[1]:
        contract_column = _mapping_select(
            "contract", "Contract", source.columns, inferred, file_hash
        )
    with mapping_columns[2]:
        pred_column = _mapping_select("pred", "pred", source.columns, inferred, file_hash)
    with mapping_columns[3]:
        tag_column = _mapping_select("tag", "Tag", source.columns, inferred, file_hash)

    if None in (datetime_column, contract_column, pred_column, tag_column):
        st.warning("请确认四个字段映射后继续。")
        return
    mapping = SchemaMapping(
        datetime=str(datetime_column),
        contract=str(contract_column),
        pred=str(pred_column),
        tag=str(tag_column),
    )
    try:
        prepared = prepare_uploaded_frame(source, mapping)
    except PipelineValidationError as exc:
        st.error(str(exc))
        return

    input_dates = prepared["Date"]
    summary = st.columns(5)
    summary[0].metric("行数", f"{prepared.height:,}")
    summary[1].metric("交易日", f"{input_dates.n_unique():,}")
    summary[2].metric("合约数", f"{prepared[INTERNAL_CONTRACT].n_unique():,}")
    summary[3].metric("起始日期", str(input_dates.min()))
    summary[4].metric("结束日期", str(input_dates.max()))
    st.caption(f"SHA256：`{file_hash}`")

    st.divider()
    editor_columns = st.columns(2)
    with editor_columns[0]:
        _render_pipeline_editor("A")
    with editor_columns[1]:
        _render_pipeline_editor("B")

    st.markdown("### 出图设置")
    chart_setting_columns = st.columns(2)
    with chart_setting_columns[0]:
        chart_settings_a = _render_chart_settings("A")
    with chart_setting_columns[1]:
        chart_settings_b = _render_chart_settings("B")

    if st.button(
        "执行 A/B 实验",
        type="primary",
        width="stretch",
    ):
        try:
            _run_experiments(
                prepared,
                file_hash=file_hash,
                mapping=mapping,
                file_name=uploaded.name,
            )
        except Exception as exc:
            st.exception(exc)

    results = st.session_state.pipeline_results
    if set(results) != {"A", "B"}:
        return

    st.divider()
    st.markdown("## 结果对照")
    st.dataframe(
        pl.DataFrame(
            {
                "指标": (
                    "最终样本",
                    "丢弃样本",
                    "交易日",
                    "起始日期",
                    "结束日期",
                    "数据范围",
                ),
                "实验 A": (
                    f"{results['A'].output_rows:,}",
                    f"{results['A'].dropped_rows:,}",
                    f"{results['A'].output_trading_days:,}",
                    str(results["A"].output_start_date),
                    str(results["A"].output_end_date),
                    results["A"].output_scope_label,
                ),
                "实验 B": (
                    f"{results['B'].output_rows:,}",
                    f"{results['B'].dropped_rows:,}",
                    f"{results['B'].output_trading_days:,}",
                    str(results["B"].output_start_date),
                    str(results["B"].output_end_date),
                    results["B"].output_scope_label,
                ),
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "指标": st.column_config.TextColumn(width="medium"),
            "实验 A": st.column_config.TextColumn(width="large"),
            "实验 B": st.column_config.TextColumn(width="large"),
        },
    )
    result_layout = st.radio(
        "结果布局",
        options=(RESULT_LAYOUT_STACKED, RESULT_LAYOUT_SIDE_BY_SIDE),
        index=0,
        horizontal=True,
        help="上下完整适合普通窗口；左右并排适合宽屏快速对照。切换布局不会重新运行流水线。",
    )
    chart_metadata: dict[str, object] = {}
    try:
        if result_layout == RESULT_LAYOUT_STACKED:
            chart_metadata["A"] = _render_result_card(
                "A",
                result=results["A"],
                chart_settings=chart_settings_a,
                file_hash=file_hash,
                mapping=mapping,
            )
            chart_metadata["B"] = _render_result_card(
                "B",
                result=results["B"],
                chart_settings=chart_settings_b,
                file_hash=file_hash,
                mapping=mapping,
            )
        else:
            result_columns = st.columns(2)
            with result_columns[0]:
                chart_metadata["A"] = _render_result_card(
                    "A",
                    result=results["A"],
                    chart_settings=chart_settings_a,
                    file_hash=file_hash,
                    mapping=mapping,
                )
            with result_columns[1]:
                chart_metadata["B"] = _render_result_card(
                    "B",
                    result=results["B"],
                    chart_settings=chart_settings_b,
                    file_hash=file_hash,
                    mapping=mapping,
                )
    except Exception as exc:
        st.error(f"图片生成失败：{exc}")
        return

    snapshot = dict(st.session_state.executed_snapshot)
    snapshot["charts"] = chart_metadata
    configuration = json.dumps(
        snapshot,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    st.download_button(
        "下载完整实验配置 JSON",
        data=configuration,
        file_name="pred_dual_pipeline_experiment.json",
        mime="application/json",
        width="stretch",
    )


if __name__ == "__main__":
    main()
