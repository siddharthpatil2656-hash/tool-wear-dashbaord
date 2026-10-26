"""Fit Taylor tool-life coefficients from measured, setup-matched trials."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


CONTEXT_COLUMNS = (
    "pairing_key",
    "workpiece_name",
    "tool_material",
    "coating",
    "machine_name",
    "coolant_name",
    "operation_name",
    "milling_tooling_name",
    "holder_name",
    "overhang_ratio",
    "is_roughing",
)
MEASUREMENT_COLUMNS = (
    "cutting_speed_m_min",
    "feed_rate_mm",
    "depth_of_cut_mm",
    "measured_tool_life_min",
)
CALIBRATION_COLUMNS = CONTEXT_COLUMNS + MEASUREMENT_COLUMNS
MIN_CALIBRATION_RUNS = 8


@dataclass(frozen=True)
class TaylorFit:
    taylor_C: float
    taylor_n: float
    taylor_x: float
    taylor_y: float
    speed_min: float
    speed_max: float
    feed_min: float
    feed_max: float
    depth_min: float
    depth_max: float
    runs_used: int
    r_squared: float
    log_rmse: float
    loo_r_squared: float
    loo_log_rmse: float


def calibration_template(context: Mapping[str, object]) -> pd.DataFrame:
    """Return a one-row CSV template with the current non-cutting setup filled in."""
    row = {column: context.get(column, "") for column in CONTEXT_COLUMNS}
    row.update({column: "" for column in MEASUREMENT_COLUMNS})
    return pd.DataFrame([row], columns=CALIBRATION_COLUMNS)


def _context_value(column: str, value: object) -> object:
    if column == "overhang_ratio":
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            return None
    if column == "is_roughing":
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        return None
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def matching_calibration_trials(
    trials: pd.DataFrame,
    context: Mapping[str, object],
) -> pd.DataFrame:
    """Return complete positive measurement rows for exactly the selected setup."""
    missing = [column for column in CALIBRATION_COLUMNS if column not in trials.columns]
    if missing:
        raise ValueError("Calibration CSV is missing required columns: " + ", ".join(missing))

    data = trials.loc[:, CALIBRATION_COLUMNS].copy()
    for column in CONTEXT_COLUMNS:
        expected = _context_value(column, context.get(column))
        data = data[
            data[column].map(lambda value: _context_value(column, value)) == expected
        ]
    if data.empty:
        raise ValueError(
            "No trial rows match the currently selected material, machine, coolant, "
            "operation, tooling, holder, overhang, and roughing mode."
        )

    numeric = list(MEASUREMENT_COLUMNS)
    raw_numeric = data[numeric].copy()
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        invalid_numeric = (
            data[column].isna()
            & raw_numeric[column].notna()
            & raw_numeric[column].astype(str).str.strip().ne("")
        )
        if invalid_numeric.any():
            raise ValueError(f"Column '{column}' contains a non-numeric value.")
    complete_rows = data[numeric].notna().all(axis=1)
    blank_rows = data[numeric].isna().all(axis=1)
    if (~complete_rows & ~blank_rows).any():
        raise ValueError(
            "A matching trial row has some, but not all, cutting parameters and measured "
            "tool life. Complete the row or remove it."
        )
    data = data.loc[complete_rows].copy()
    if (data[numeric] <= 0).any().any():
        raise ValueError("Cutting speed, feed, depth, and measured life must all be positive.")
    return data.reset_index(drop=True)


def fit_taylor_model(
    trials: pd.DataFrame,
    context: Mapping[str, object],
    life_multiplier: float,
) -> TaylorFit:
    """Fit extended Taylor coefficients using only identical machining setups.

    Observed effective tool life is divided by the known, setup-specific model
    multiplier before fitting so machine/coolant/holder effects are not counted
    twice when the fitted coefficients are used by the physics model.
    """
    if life_multiplier <= 0 or not np.isfinite(life_multiplier):
        raise ValueError("The selected setup has an invalid tool-life multiplier.")

    data = matching_calibration_trials(trials, context)
    numeric = list(MEASUREMENT_COLUMNS)
    if len(data) < MIN_CALIBRATION_RUNS:
        raise ValueError(
            f"At least {MIN_CALIBRATION_RUNS} complete, setup-matched trial runs are required; "
            f"found {len(data)}."
        )

    speed = data["cutting_speed_m_min"].to_numpy(dtype=float)
    feed = data["feed_rate_mm"].to_numpy(dtype=float)
    depth = data["depth_of_cut_mm"].to_numpy(dtype=float)
    nominal_life = data["measured_tool_life_min"].to_numpy(dtype=float) / life_multiplier
    design = np.column_stack((
        np.ones(len(data)),
        np.log(speed),
        np.log(feed),
        np.log(depth),
    ))
    response = np.log(nominal_life)
    if np.linalg.matrix_rank(design) < 4:
        raise ValueError(
            "The trials do not vary cutting speed, feed, and depth enough to identify "
            "all four Taylor coefficients. Collect trials spanning these parameters."
        )
    if np.linalg.cond(design) > 1e6:
        raise ValueError(
            "The trial parameter values are too strongly correlated to fit stable Taylor "
            "coefficients. Vary speed, feed, and depth independently."
        )

    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    intercept, speed_slope, feed_slope, depth_slope = coefficients
    if speed_slope >= 0:
        raise ValueError("The trial data do not show tool life decreasing as speed increases.")
    taylor_n = -1.0 / speed_slope
    taylor_x = -feed_slope * taylor_n
    taylor_y = -depth_slope * taylor_n
    taylor_C = float(np.exp(intercept * taylor_n))
    if not (0.05 <= taylor_n <= 0.9):
        raise ValueError(f"Fitted Taylor exponent n={taylor_n:.3f} is outside 0.05–0.90.")
    if not (0.0 <= taylor_x <= 1.0 and 0.0 <= taylor_y <= 1.0):
        raise ValueError(
            f"Fitted feed/depth exponents x={taylor_x:.3f}, y={taylor_y:.3f} "
            "must each be within 0–1."
        )
    if not np.isfinite(taylor_C) or taylor_C <= 0:
        raise ValueError("The fitted Taylor constant is invalid.")

    residuals = response - design @ coefficients
    total_sum_squares = float(np.sum((response - response.mean()) ** 2))
    r_squared = (
        1.0 - float(np.sum(residuals ** 2)) / total_sum_squares
        if total_sum_squares > 0
        else 0.0
    )
    loo_predictions = np.empty(len(data), dtype=float)
    for index in range(len(data)):
        training_mask = np.arange(len(data)) != index
        training_design = design[training_mask]
        if np.linalg.matrix_rank(training_design) < 4:
            raise ValueError(
                "The trials do not support leave-one-out validation. Add more independently "
                "varied runs before calibrating."
            )
        loo_coefficients, _, _, _ = np.linalg.lstsq(
            training_design, response[training_mask], rcond=None
        )
        loo_predictions[index] = design[index] @ loo_coefficients
    loo_residuals = response - loo_predictions
    loo_sum_squares = float(np.sum(loo_residuals ** 2))
    loo_r_squared = (
        1.0 - loo_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else 0.0
    )
    return TaylorFit(
        taylor_C=taylor_C,
        taylor_n=float(taylor_n),
        taylor_x=float(taylor_x),
        taylor_y=float(taylor_y),
        speed_min=float(speed.min()),
        speed_max=float(speed.max()),
        feed_min=float(feed.min()),
        feed_max=float(feed.max()),
        depth_min=float(depth.min()),
        depth_max=float(depth.max()),
        runs_used=len(data),
        r_squared=float(r_squared),
        log_rmse=float(np.sqrt(np.mean(residuals ** 2))),
        loo_r_squared=float(loo_r_squared),
        loo_log_rmse=float(np.sqrt(np.mean(loo_residuals ** 2))),
    )
