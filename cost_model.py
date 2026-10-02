"""Simple per-part cost model for comparing cutting-speed candidates."""

from dataclasses import dataclass
import math
from typing import Callable, Iterable


@dataclass(frozen=True)
class CostPoint:
    cutting_speed_m_min: float
    tool_life_min: float
    mrr_cm3_min: float
    machining_time_min: float
    total_cycle_time_min: float
    cost_per_part: float


def cost_per_part(
    *,
    part_volume_cm3: float,
    mrr_cm3_min: float,
    tool_life_min: float,
    tool_edge_cost: float,
    machine_rate_per_hour: float,
    operator_rate_per_hour: float,
    tool_change_min: float,
) -> tuple[float, float, float]:
    """Return (cost/part, cutting time/part, amortized cycle time/part)."""
    values = (
        part_volume_cm3,
        mrr_cm3_min,
        tool_life_min,
        tool_edge_cost,
        machine_rate_per_hour,
        operator_rate_per_hour,
        tool_change_min,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Cost inputs must be finite numbers.")
    if part_volume_cm3 <= 0 or mrr_cm3_min <= 0 or tool_life_min <= 0:
        raise ValueError("Part volume, MRR, and tool life must be greater than zero.")
    if min(tool_edge_cost, machine_rate_per_hour, operator_rate_per_hour, tool_change_min) < 0:
        raise ValueError("Costs and tool-change time cannot be negative.")

    machining_time = part_volume_cm3 / mrr_cm3_min
    edge_fraction_per_part = machining_time / tool_life_min
    cycle_time = machining_time + edge_fraction_per_part * tool_change_min
    hourly_rate = machine_rate_per_hour + operator_rate_per_hour
    per_part_cost = hourly_rate * cycle_time / 60.0 + tool_edge_cost * edge_fraction_per_part
    return per_part_cost, machining_time, cycle_time


def build_cost_curve(
    *,
    cutting_speeds: Iterable[float],
    feed: float,
    depth_of_cut: float,
    part_volume_cm3: float,
    tool_edge_cost: float,
    machine_rate_per_hour: float,
    operator_rate_per_hour: float,
    tool_change_min: float,
    tool_life_fn: Callable[[float, float, float], float],
    mrr_fn: Callable[[float, float, float], float],
) -> list[CostPoint]:
    """Evaluate speed candidates and return them ordered by cutting speed."""
    points = []
    for speed in cutting_speeds:
        life = float(tool_life_fn(float(speed), feed, depth_of_cut))
        mrr = float(mrr_fn(float(speed), feed, depth_of_cut))
        cost, cutting_time, cycle_time = cost_per_part(
            part_volume_cm3=part_volume_cm3,
            mrr_cm3_min=mrr,
            tool_life_min=life,
            tool_edge_cost=tool_edge_cost,
            machine_rate_per_hour=machine_rate_per_hour,
            operator_rate_per_hour=operator_rate_per_hour,
            tool_change_min=tool_change_min,
        )
        points.append(CostPoint(
            cutting_speed_m_min=float(speed),
            tool_life_min=life,
            mrr_cm3_min=mrr,
            machining_time_min=cutting_time,
            total_cycle_time_min=cycle_time,
            cost_per_part=cost,
        ))
    if not points:
        raise ValueError("At least one cutting-speed candidate is required.")
    return points
