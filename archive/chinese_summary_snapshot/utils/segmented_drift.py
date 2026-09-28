from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl


SEGMENT_BEFORE = "14:30以前"
SEGMENT_AFTER = "14:30及以后"
SEGMENT_ORDER = (SEGMENT_BEFORE, SEGMENT_AFTER)


def direction_from_slow_baseline(scores, slow_baselines) -> np.ndarray:
    """Return -1 below the slow baseline and +1 on or above it."""
    score_values = np.asarray(scores, dtype=float)
    baseline_values = np.asarray(slow_baselines, dtype=float)
    if score_values.shape != baseline_values.shape:
        raise ValueError("scores and slow_baselines must have the same shape")
    return np.where(score_values < baseline_values, -1, 1)


def directional_hit_rate(direction, tags) -> float | None:
    """Measure sign accuracy, excluding zero tags from the denominator."""
    direction_values = np.asarray(direction, dtype=int)
    tag_values = np.asarray(tags, dtype=float)
    if direction_values.shape != tag_values.shape:
        raise ValueError("direction and tags must have the same shape")
    valid = tag_values != 0
    if not np.any(valid):
        return None
    hit = ((direction_values == -1) & (tag_values < 0)) | (
        (direction_values == 1) & (tag_values > 0)
    )
    return float(np.mean(hit[valid]))


def segment_from_clock(
    hour: int,
    minute: int,
    cutoff_hour: int = 14,
    cutoff_minute: int = 30,
) -> str:
    """Split a clock time; the exact cutoff belongs to the later segment."""
    return (
        SEGMENT_BEFORE
        if (int(hour), int(minute)) < (int(cutoff_hour), int(cutoff_minute))
        else SEGMENT_AFTER
    )


def alpha_from_half_life(half_life_days: int) -> float:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    return float(1.0 - 0.5 ** (1.0 / half_life_days))


def build_causal_state(
    daily_medians: Sequence[float],
    cold_start_days: int,
    half_life_days: int,
) -> dict[str, np.ndarray]:
    """Build states where date t uses medians strictly earlier than t."""
    medians = np.asarray(daily_medians, dtype=float)
    if medians.ndim != 1:
        raise ValueError("daily_medians must be one-dimensional")
    if not 0 < cold_start_days < len(medians):
        raise ValueError("cold_start_days must leave at least one evaluation day")
    if not np.all(np.isfinite(medians)):
        raise ValueError("daily_medians must be finite")

    alpha = alpha_from_half_life(half_life_days)
    baseline = np.full(len(medians), np.nan, dtype=float)
    recent_center = np.full(len(medians), np.nan, dtype=float)
    current_center = None

    for index in range(cold_start_days, len(medians)):
        current_baseline = float(np.median(medians[:index]))
        if index == cold_start_days:
            current_center = current_baseline
        else:
            current_center = (
                (1.0 - alpha) * current_center + alpha * float(medians[index - 1])
            )
        baseline[index] = current_baseline
        recent_center[index] = current_center

    return {
        "slow_baseline": baseline,
        "recent_center": recent_center,
        "time_offset": recent_center - baseline,
    }


def build_independent_segment_states(
    daily_medians_by_segment: Mapping[str, Sequence[float]],
    cold_start_days: int,
    half_life_days_by_segment: Mapping[str, int],
) -> dict[str, dict[str, np.ndarray]]:
    """Build each segment without pooling information across segments."""
    return {
        segment: build_causal_state(
            medians,
            cold_start_days,
            half_life_days_by_segment[segment],
        )
        for segment, medians in daily_medians_by_segment.items()
    }


def build_selected_daily_states(
    samples: pl.DataFrame,
    selected_config: Mapping,
) -> pl.DataFrame:
    """Rebuild the selected per-segment causal state entirely in memory."""
    required = {"Date", "时段", "score"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"samples missing columns: {sorted(missing)}")

    dates = sorted(samples["Date"].unique().to_list())
    cold_start_days = int(selected_config["cold_start_effective_trading_days"])
    segment_configs = selected_config["segments"]
    frames = []
    for segment in SEGMENT_ORDER:
        daily = (
            samples.filter(pl.col("时段") == segment)
            .group_by("Date")
            .agg(
                pl.col("score").median().alias("daily_score_median"),
                pl.len().alias("daily_samples"),
            )
            .sort("Date")
        )
        if daily["Date"].to_list() != dates:
            raise ValueError(f"{segment} does not contain every trading date")
        segment_config = segment_configs[segment]
        arrays = build_causal_state(
            daily["daily_score_median"].to_numpy(),
            cold_start_days,
            int(segment_config["selected_half_life_trading_days"]),
        )
        balance_offset = float(segment_config["selected_development_balance_offset"])
        frame = (
            daily.with_row_index("date_index")
            .with_columns(
                pl.lit(segment).alias("时段"),
                pl.Series("slow_baseline", arrays["slow_baseline"]),
                pl.Series("recent_center", arrays["recent_center"]),
            )
            .with_columns(
                (pl.col("recent_center") + balance_offset).alias("effective_center")
            )
            .with_columns(
                (pl.col("effective_center") - pl.col("slow_baseline")).alias(
                    "effective_time_offset"
                )
            )
        )
        frames.append(frame)
    return (
        pl.concat(frames)
        .filter(
            pl.col("slow_baseline").is_finite()
            & pl.col("recent_center").is_finite()
            & pl.col("effective_time_offset").is_finite()
        )
        .sort(["时段", "Date"])
    )
