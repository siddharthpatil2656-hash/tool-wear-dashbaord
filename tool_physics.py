"""
Machining Physics & Tool Wear Calculation Engine.
Implements Extended Taylor's Tool Life Equation, 3-Stage Flank Wear (VB) Progression,
Machine Dynamic Rigidity Factor, and Material Removal Rate (MRR).
Calibrated against Springer IJAMT empirical data.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from springer_database import (
    COOLANT_DATABASE,
    MACHINE_DATABASE,
    MILLING_TOOLING_DATABASE,
    OPERATION_DATABASE,
    SPRINGER_DATABASE,
    CoolantCharacteristics,
    MachineCharacteristics,
    MaterialToolPairing,
)


@dataclass
class ToolWearPrediction:
    pairing_key: str
    pairing: MaterialToolPairing
    machine_name: str
    machine: MachineCharacteristics
    coolant_name: str
    coolant: CoolantCharacteristics
    # Cutting inputs
    vc: float  # m/min
    feed: float  # mm/rev or mm/tooth
    ap: float  # mm
    current_time_min: float
    is_roughing: bool
    operation_name: Optional[str]
    milling_tooling_name: Optional[str]
    # Output metrics
    tool_life_minutes: float
    mrr_cm3_min: float
    total_volume_cut_cm3: float
    cutting_distance_meters: float
    current_vb_mm: float
    vb_threshold_mm: float
    rul_minutes: float
    rul_percentage: float
    rul_status: str  # Safe, Warning, Critical
    confidence_score: float  # 0 to 100%
    confidence_label: str    # High, Moderate, Extrapolated
    confidence_notes: List[str]


def evaluate_confidence(
    pairing: MaterialToolPairing, vc: float, feed: float, ap: float
) -> Tuple[float, str, List[str]]:
    """
    Evaluates whether the user's cutting parameters fall within the verified
    empirical domain of the Springer research publication.
    """
    score = 100.0
    notes = []

    # Speed check
    if vc < pairing.v_min:
        penalty = min(35.0, ((pairing.v_min - vc) / pairing.v_min) * 50.0)
        score -= penalty
        notes.append(f"Cutting speed ({vc:.1f} m/min) is below Springer study baseline ({pairing.v_min:.1f} m/min). Risk of unstable Built-Up Edge (BUE).")
    elif vc > pairing.v_max:
        penalty = min(40.0, ((vc - pairing.v_max) / pairing.v_max) * 60.0)
        score -= penalty
        notes.append(f"Cutting speed ({vc:.1f} m/min) exceeds Springer tested limit ({pairing.v_max:.1f} m/min). High thermal softening risk.")

    # Feed check
    if feed < pairing.f_min:
        penalty = min(25.0, ((pairing.f_min - feed) / pairing.f_min) * 40.0)
        score -= penalty
        notes.append(f"Feed ({feed:.3f} mm/rev) is below minimum chip formation threshold ({pairing.f_min:.3f} mm/rev). Risk of tool rubbing / burnishing.")
    elif feed > pairing.f_max:
        penalty = min(30.0, ((feed - pairing.f_max) / pairing.f_max) * 50.0)
        score -= penalty
        notes.append(f"Feed ({feed:.3f} mm/rev) exceeds Springer tested range ({pairing.f_max:.3f} mm/rev). Increased risk of insert chipping.")

    # Depth of cut check
    if ap < pairing.ap_min:
        penalty = min(20.0, ((pairing.ap_min - ap) / pairing.ap_min) * 35.0)
        score -= penalty
        notes.append(f"Depth of cut ({ap:.2f} mm) is smaller than tool nose radius engage ({pairing.ap_min:.2f} mm).")
    elif ap > pairing.ap_max:
        penalty = min(25.0, ((ap - pairing.ap_max) / pairing.ap_max) * 40.0)
        score -= penalty
        notes.append(f"Depth of cut ({ap:.2f} mm) exceeds tested boundary ({pairing.ap_max:.2f} mm). Heavy load on toolholder.")

    score = max(10.0, min(100.0, score))

    if score >= 85:
        label = "High Confidence (Empirically Verified)"
    elif score >= 60:
        label = "Moderate Confidence (Near Research Boundary)"
    else:
        label = "Extrapolated (Outside Empirical Literature Range)"

    if not notes:
        notes.append("All machining parameters are strictly within the experimental window verified by the Springer publication.")

    return score, label, notes


def calculate_tool_life(
    pairing: MaterialToolPairing,
    machine: MachineCharacteristics,
    coolant: CoolantCharacteristics,
    vc: float,
    feed: float,
    ap: float,
    is_roughing: bool = False,
    operation_name: Optional[str] = None,
    milling_tooling_name: Optional[str] = None,
) -> float:
    """
    Computes nominal and effective tool life using Extended Taylor's Law:
    T_nominal = (C / (Vc * feed^x * ap^y))^(1/n)
    T_effective = T_nominal * k_machine * k_coolant * k_operation * k_tooling
    """
    # Guard against invalid negative/zero inputs
    vc = max(1.0, vc)
    feed = max(0.001, feed)
    ap = max(0.05, ap)

    # Denominator in Taylor's equation
    denom = vc * (feed ** pairing.taylor_x) * (ap ** pairing.taylor_y)
    ratio = pairing.taylor_C / denom

    # Prevent negative or infinite life
    if ratio <= 0:
        return 0.1

    t_nominal = ratio ** (1.0 / pairing.taylor_n)

    # Coolant synergy adjustments:
    # If using ceramic tools, dry air blast is often superior to flood coolant
    # because flood causes cyclical thermal shock and comb micro-cracking.
    coolant_factor = coolant.life_multiplier
    if "Ceramic" in pairing.tool_material:
        if "Dry" in coolant.name:
            coolant_factor = 1.15  # Ceramics excel in continuous high heat dry
        elif "Flood" in coolant.name:
            coolant_factor = 0.65  # Flood induces catastrophic thermal shock

    # Roughing factor: Roughing allows higher acceptable VB limit (0.5-0.6mm vs 0.3mm),
    # which extends allowable cut duration by ~30-40% before reaching failure limit.
    roughing_factor = 1.35 if is_roughing else 1.0

    # Operation & tooling factors
    operation_factor = 1.0
    if operation_name and operation_name in OPERATION_DATABASE:
        op = OPERATION_DATABASE[operation_name]
        operation_factor *= op.life_multiplier
        # Ceramic inserts suffer comb micro-cracking under the cyclic thermal/mechanical
        # shock of interrupted milling engagement — heavy derate required.
        if "Ceramic" in pairing.tool_material and op.family == "milling":
            operation_factor *= 0.50

    tooling_factor = 1.0
    if milling_tooling_name and milling_tooling_name in MILLING_TOOLING_DATABASE:
        tooling_factor = MILLING_TOOLING_DATABASE[milling_tooling_name].life_multiplier

    t_effective = (
        t_nominal
        * machine.rigidity_factor
        * coolant_factor
        * roughing_factor
        * operation_factor
        * tooling_factor
    )

    # Floor at 0.5 minutes, ceiling at 2000 minutes
    return float(np.clip(t_effective, 0.5, 2000.0))


def compute_flank_wear_curve(
    pairing: MaterialToolPairing,
    tool_life_min: float,
    is_roughing: bool = False,
    time_steps: int = 150,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates the 3-stage Flank Wear (VB) degradation curve over time:
    Stage I (Run-in): Rapid rounding of edge micro-roughness.
    Stage II (Steady-state): Linear abrasive flank wear.
    Stage III (Tertiary): Exponential runaway wear toward failure threshold.

    Returns:
    t_array (minutes), vb_total (mm), vb_stage1, vb_stage2_3
    """
    vb_crit = pairing.vb_critical_roughing if is_roughing else pairing.vb_critical_finishing
    vb_0 = pairing.run_in_vb

    # Extend time domain slightly beyond tool life (up to 115% of tool life) to illustrate failure
    max_time = tool_life_min * 1.15
    t = np.linspace(0.0, max_time, time_steps)

    # Stage 1: Run-in exponential approach to initial steady wear land
    # VB1(t) = vb_0 * (1 - exp(-k_runin * t / (t_life * 0.1)))
    tau_runin = max(0.2, tool_life_min * 0.08)
    vb_stage1 = vb_0 * (1.0 - np.exp(-t / tau_runin))

    # Stage 2: Steady state wear
    # Rate alpha is calibrated so total VB reaches vb_crit at t = tool_life_min
    # Let tertiary wear start contributing around 0.85 * tool_life_min
    t_tertiary_start = 0.82 * tool_life_min
    steady_slope = (vb_crit * 0.82 - vb_0) / (t_tertiary_start)
    steady_slope = max(0.0001, steady_slope)

    vb_stage2 = steady_slope * t

    # Stage 3: Tertiary accelerated wear (runaway temperature / clearance collapse)
    excess_time = np.maximum(0.0, t - t_tertiary_start)
    tertiary_scale = (vb_crit - (vb_0 + steady_slope * tool_life_min))
    tertiary_scale = max(0.02, tertiary_scale)

    tau_tertiary = tool_life_min * 0.18
    vb_stage3 = tertiary_scale * (np.exp(excess_time / tau_tertiary) - 1.0) / (np.exp((tool_life_min - t_tertiary_start) / tau_tertiary) - 1.0)
    vb_stage3 = np.maximum(0.0, vb_stage3)

    # Combined flank wear curve
    vb_total = vb_stage1 + vb_stage2 + vb_stage3

    return t, vb_total, vb_stage1, (vb_stage2 + vb_stage3)


def get_current_flank_wear(
    pairing: MaterialToolPairing,
    tool_life_min: float,
    current_time_min: float,
    is_roughing: bool = False,
) -> float:
    """Computes exact instantaneous flank wear VB at elapsed time t."""
    if current_time_min <= 0:
        return 0.0

    t_arr, vb_arr, _, _ = compute_flank_wear_curve(pairing, tool_life_min, is_roughing, 300)
    current_vb = np.interp(current_time_min, t_arr, vb_arr)
    return float(current_vb)


def calculate_mrr(
    vc: float,
    feed: float,
    ap: float,
    operation_name: Optional[str] = None,
    milling_tooling_name: Optional[str] = None,
) -> float:
    """
    Computes Material Removal Rate (MRR) in cm^3/min.

    Turning/Boring model:  MRR = Vc * feed * ap
    Milling model:         n = 1000*Vc/(pi*D) rpm;  vf = fz*z*n mm/min;
                           MRR = ap * ae * vf / 1000,  ae = ae_fraction * D
    Drilling model:        MRR = (pi*D^2/4) * f * n / 1000  (D = 10 mm reference)
    """
    op = OPERATION_DATABASE.get(operation_name) if operation_name else None
    model = op.mrr_model if op else "turning"

    if model == "milling":
        tool = (
            MILLING_TOOLING_DATABASE.get(milling_tooling_name)
            if milling_tooling_name
            else None
        )
        z = tool.teeth if tool else 4
        d = tool.diameter_mm if tool else 12.0
        ae_frac = op.ae_fraction_of_d if op else 0.35
        n_rpm = 1000.0 * vc / (math.pi * d)
        vf = feed * z * n_rpm  # feed is fz in mm/tooth
        ae = d * ae_frac
        return float(ap * ae * vf / 1000.0)

    if model == "drilling":
        d = 10.0  # reference twist-drill diameter
        n_rpm = 1000.0 * vc / (math.pi * d)
        return float((math.pi * d * d / 4.0) * feed * n_rpm / 1000.0)

    # Turning / boring baseline
    return float(vc * feed * ap)


def predict_tool_wear(
    pairing_key: Optional[str] = None,
    machine_name: str = "5-Axis High-Precision CNC Machining Center",
    coolant_name: str = "Standard Flood Emulsion (7-10% oil)",
    vc: float = 100.0,
    feed: float = 0.15,
    ap: float = 1.0,
    current_time_min: float = 0.0,
    is_roughing: bool = False,
    custom_pairing: Optional[MaterialToolPairing] = None,
    operation_name: Optional[str] = None,
    milling_tooling_name: Optional[str] = None,
) -> ToolWearPrediction:
    """
    Executes full predictive tool wear and life calculations.
    Accepts either a standard pairing_key or a custom synthesized pairing.
    """
    if custom_pairing is not None:
        pairing = custom_pairing
        eff_pairing_key = pairing_key or f"{pairing.workpiece_name} | {pairing.tool_material}"
    else:
        assert pairing_key is not None, "Either pairing_key or custom_pairing must be provided"
        pairing = SPRINGER_DATABASE[pairing_key]
        eff_pairing_key = pairing_key

    machine = MACHINE_DATABASE[machine_name]
    coolant = COOLANT_DATABASE[coolant_name]

    # 1. Tool life
    t_life = calculate_tool_life(
        pairing, machine, coolant, vc, feed, ap, is_roughing,
        operation_name=operation_name,
        milling_tooling_name=milling_tooling_name,
    )

    # 2. Material removal rate
    mrr = calculate_mrr(
        vc, feed, ap,
        operation_name=operation_name,
        milling_tooling_name=milling_tooling_name,
    )
    total_volume = mrr * t_life
    cutting_distance = vc * t_life  # meters

    # 3. Flank wear & RUL
    vb_threshold = pairing.vb_critical_roughing if is_roughing else pairing.vb_critical_finishing
    current_vb = get_current_flank_wear(pairing, t_life, current_time_min, is_roughing)

    rul_minutes = max(0.0, t_life - current_time_min)
    rul_pct = max(0.0, min(100.0, (rul_minutes / t_life) * 100.0))

    if rul_pct > 35.0 and current_vb < (vb_threshold * 0.70):
        rul_status = "Safe / Nominal Wear Zone"
    elif rul_pct > 12.0 and current_vb < (vb_threshold * 0.90):
        rul_status = "Caution: Advancing Wear (Monitor Surface Finish)"
    else:
        rul_status = "Critical: End of Tool Life (Replace Insert Now)"

    # 4. Confidence scoring against Springer literature
    conf_score, conf_label, conf_notes = evaluate_confidence(pairing, vc, feed, ap)

    # 5. Operation-specific physics warnings
    if operation_name and operation_name in OPERATION_DATABASE:
        op = OPERATION_DATABASE[operation_name]
        if "Ceramic" in pairing.tool_material and op.family == "milling":
            conf_notes.append(
                "⚠️ **THERMAL-SHOCK DERATE**: Ceramic inserts in interrupted milling suffer comb "
                "micro-cracking from cyclic engagement shock — tool life has been derated 50%. "
                "Prefer carbide or PCBN tooling, or switch to continuous turning."
            )
        if op.family == "milling" and "HSS" in pairing.tool_material and vc > 120.0:
            conf_notes.append(
                f"⚠️ **HSS SPEED LIMIT**: HSS tooling is generally capped near 120 m/min in milling; "
                f"the selected {vc:.0f} m/min risks rapid edge softening. Reduce speed or switch to carbide."
            )

    return ToolWearPrediction(
        pairing_key=eff_pairing_key,
        pairing=pairing,
        machine_name=machine_name,
        machine=machine,
        coolant_name=coolant_name,
        coolant=coolant,
        vc=vc,
        feed=feed,
        ap=ap,
        current_time_min=current_time_min,
        is_roughing=is_roughing,
        operation_name=operation_name,
        milling_tooling_name=milling_tooling_name,
        tool_life_minutes=t_life,
        mrr_cm3_min=mrr,
        total_volume_cut_cm3=total_volume,
        cutting_distance_meters=cutting_distance,
        current_vb_mm=current_vb,
        vb_threshold_mm=vb_threshold,
        rul_minutes=rul_minutes,
        rul_percentage=rul_pct,
        rul_status=rul_status,
        confidence_score=conf_score,
        confidence_label=conf_label,
        confidence_notes=conf_notes,
    )
