"""
Tool Wear Minimization & Multi-Objective Optimization Engine.
Provides physics-grounded optimization algorithms to minimize tool wear while preserving
Material Removal Rate (MRR), along with concrete, actionable engineering recommendations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from springer_database import (
    COOLANT_DATABASE,
    MACHINE_DATABASE,
    MILLING_TOOLING_DATABASE,
    OPERATION_DATABASE,
    SPRINGER_DATABASE,
    TOOL_HOLDER_DATABASE,
    MaterialToolPairing,
)
from tool_physics import calculate_mrr, calculate_tool_life, evaluate_confidence


@dataclass
class OptimizedParameters:
    vc: float
    feed: float
    ap: float
    tool_life_min: float
    mrr_cm3_min: float
    tool_life_gain_pct: float
    mrr_change_pct: float
    strategy_name: str
    rationale: str


@dataclass
class WearMinimizationReport:
    current_vc: float
    current_feed: float
    current_ap: float
    current_tool_life: float
    current_mrr: float
    operation_name: Optional[str]
    milling_tooling_name: Optional[str]
    holder_name: Optional[str]
    overhang_ratio: Optional[float]

    # Optimized strategies
    balanced_strategy: OptimizedParameters   # Maintains MRR, vastly improves life
    max_life_strategy: OptimizedParameters   # Maximizes tool life (lights-out machining)
    high_efficiency_strategy: OptimizedParameters  # Modest life gain, boosts MRR

    # Categorized recommendations
    cutting_parameter_tips: List[str]
    tooling_coating_tips: List[str]
    machine_vibration_tips: List[str]
    coolant_tips: List[str]

    # Pareto trade-off curve (Tool Life vs MRR)
    pareto_mrr: List[float]
    pareto_tool_life: List[float]


def generate_pareto_curve(
    pairing: MaterialToolPairing,
    machine_name: str,
    coolant_name: str,
    is_roughing: bool = False,
    steps: int = 40,
    operation_name: Optional[str] = None,
    milling_tooling_name: Optional[str] = None,
    holder_name: Optional[str] = None,
    overhang_ratio: Optional[float] = None,
) -> Tuple[List[float], List[float]]:
    """
    Generates Pareto optimal frontier between Tool Life (min) and Material Removal Rate (cm3/min).
    Varies cutting speed across the Springer empirical boundary.
    """
    machine = MACHINE_DATABASE[machine_name]
    coolant = COOLANT_DATABASE[coolant_name]

    vc_range = np.linspace(pairing.v_min, pairing.v_max * 1.05, steps)
    # Use median feed and ap
    nominal_feed = (pairing.f_min + pairing.f_max) * 0.5
    nominal_ap = (pairing.ap_min + pairing.ap_max) * 0.5

    pareto_mrr = []
    pareto_life = []

    for v in vc_range:
        life = calculate_tool_life(
            pairing, machine, coolant, v, nominal_feed, nominal_ap, is_roughing,
            operation_name=operation_name,
            milling_tooling_name=milling_tooling_name,
            holder_name=holder_name,
            overhang_ratio=overhang_ratio,
        )
        mrr = calculate_mrr(
            v, nominal_feed, nominal_ap,
            operation_name=operation_name,
            milling_tooling_name=milling_tooling_name,
        )
        pareto_life.append(round(life, 2))
        pareto_mrr.append(round(mrr, 2))

    return pareto_mrr, pareto_life


def optimize_tool_wear(
    pairing_key: Optional[str] = None,
    machine_name: str = "5-Axis High-Precision CNC Machining Center",
    coolant_name: str = "Standard Flood Emulsion (7-10% oil)",
    current_vc: float = 100.0,
    current_feed: float = 0.15,
    current_ap: float = 1.0,
    is_roughing: bool = False,
    custom_pairing: Optional[MaterialToolPairing] = None,
    operation_name: Optional[str] = None,
    milling_tooling_name: Optional[str] = None,
    holder_name: Optional[str] = None,
    overhang_ratio: Optional[float] = None,
) -> WearMinimizationReport:
    """
    Analyzes current machining parameters and derives mathematically optimized
    alternatives to minimize tool wear while maintaining production goals.
    Accepts either a standard pairing_key or a custom synthesized pairing.
    """
    if custom_pairing is not None:
        pairing = custom_pairing
    else:
        assert pairing_key is not None, "Either pairing_key or custom_pairing must be provided"
        pairing = SPRINGER_DATABASE[pairing_key]

    machine = MACHINE_DATABASE[machine_name]
    coolant = COOLANT_DATABASE[coolant_name]

    # Current baseline
    base_life = calculate_tool_life(
        pairing, machine, coolant, current_vc, current_feed, current_ap, is_roughing,
        operation_name=operation_name,
        milling_tooling_name=milling_tooling_name,
        holder_name=holder_name,
        overhang_ratio=overhang_ratio,
    )
    base_mrr = calculate_mrr(
        current_vc, current_feed, current_ap,
        operation_name=operation_name,
        milling_tooling_name=milling_tooling_name,
    )

    # Strategy 1: Productivity-Neutral Optimization (The "Golden Trade-off")
    # In Taylor's law, Vc has exponent 1/n (~3.5 to 5.0), whereas ap has exponent y/n (~0.7 to 1.1).
    # Reducing Vc by 18% and increasing ap by 22% yields identical MRR, but doubles tool life!
    opt_vc_balanced = max(pairing.v_min, current_vc * 0.82)
    # Compensate with depth of cut to preserve MRR:
    target_ap = (base_mrr) / (opt_vc_balanced * current_feed)
    opt_ap_balanced = min(pairing.ap_max, max(pairing.ap_min, target_ap))
    # If ap capped, compensate slightly with feed
    target_feed = (base_mrr) / (opt_vc_balanced * opt_ap_balanced)
    opt_feed_balanced = min(pairing.f_max, max(pairing.f_min, target_feed))

    life_balanced = calculate_tool_life(
        pairing, machine, coolant, opt_vc_balanced, opt_feed_balanced, opt_ap_balanced, is_roughing,
        operation_name=operation_name,
        milling_tooling_name=milling_tooling_name,
        holder_name=holder_name,
        overhang_ratio=overhang_ratio,
    )
    mrr_balanced = calculate_mrr(
        opt_vc_balanced, opt_feed_balanced, opt_ap_balanced,
        operation_name=operation_name,
        milling_tooling_name=milling_tooling_name,
    )

    balanced_strat = OptimizedParameters(
        vc=round(opt_vc_balanced, 1),
        feed=round(opt_feed_balanced, 3),
        ap=round(opt_ap_balanced, 2),
        tool_life_min=round(life_balanced, 1),
        mrr_cm3_min=round(mrr_balanced, 2),
        tool_life_gain_pct=round(((life_balanced - base_life) / base_life) * 100.0, 1),
        mrr_change_pct=round(((mrr_balanced - base_mrr) / base_mrr) * 100.0, 1),
        strategy_name="Productivity-Neutral Wear Reduction (Recommended)",
        rationale="Exploits the differential Taylor exponents: reduces thermal wear by decreasing cutting speed by 18%, while increasing depth of cut to maintain 100% of material removal productivity."
    )

    # Strategy 2: Maximum Tool Life (Lights-Out / High-Reliability Mode)
    # For critical components or overnight machining where an unexpected tool break is catastrophic.
    opt_vc_maxlife = max(pairing.v_min * 1.05, current_vc * 0.70)
    opt_feed_maxlife = max(pairing.f_min * 1.1, current_feed * 0.85)
    opt_ap_maxlife = max(pairing.ap_min, current_ap * 0.90)

    life_maxlife = calculate_tool_life(
        pairing, machine, coolant, opt_vc_maxlife, opt_feed_maxlife, opt_ap_maxlife, is_roughing,
        operation_name=operation_name,
        milling_tooling_name=milling_tooling_name,
        holder_name=holder_name,
        overhang_ratio=overhang_ratio,
    )
    mrr_maxlife = calculate_mrr(
        opt_vc_maxlife, opt_feed_maxlife, opt_ap_maxlife,
        operation_name=operation_name,
        milling_tooling_name=milling_tooling_name,
    )

    max_life_strat = OptimizedParameters(
        vc=round(opt_vc_maxlife, 1),
        feed=round(opt_feed_maxlife, 3),
        ap=round(opt_ap_maxlife, 2),
        tool_life_min=round(life_maxlife, 1),
        mrr_cm3_min=round(mrr_maxlife, 2),
        tool_life_gain_pct=round(((life_maxlife - base_life) / base_life) * 100.0, 1),
        mrr_change_pct=round(((mrr_maxlife - base_mrr) / base_mrr) * 100.0, 1),
        strategy_name="Maximum Endurance Mode (Lights-Out Machining)",
        rationale="Prioritizes operational stability and zero premature insert failure over cycle time. Drastically lowers interface temperatures below the chemical diffusion threshold."
    )

    # Strategy 3: High-Efficiency (Boost MRR while holding wear steady)
    opt_vc_he = current_vc * 0.92
    opt_feed_he = min(pairing.f_max, current_feed * 1.15)
    opt_ap_he = min(pairing.ap_max, current_ap * 1.15)

    life_he = calculate_tool_life(
        pairing, machine, coolant, opt_vc_he, opt_feed_he, opt_ap_he, is_roughing,
        operation_name=operation_name,
        milling_tooling_name=milling_tooling_name,
        holder_name=holder_name,
        overhang_ratio=overhang_ratio,
    )
    mrr_he = calculate_mrr(
        opt_vc_he, opt_feed_he, opt_ap_he,
        operation_name=operation_name,
        milling_tooling_name=milling_tooling_name,
    )

    he_strat = OptimizedParameters(
        vc=round(opt_vc_he, 1),
        feed=round(opt_feed_he, 3),
        ap=round(opt_ap_he, 2),
        tool_life_min=round(life_he, 1),
        mrr_cm3_min=round(mrr_he, 2),
        tool_life_gain_pct=round(((life_he - base_life) / base_life) * 100.0, 1),
        mrr_change_pct=round(((mrr_he - base_mrr) / base_mrr) * 100.0, 1),
        strategy_name="High-Efficiency Balanced Boost",
        rationale="Increases chip load and volumetric output while trimming speed slightly to prevent the thermal runaway barrier."
    )

    # Categorized Recommendations
    cutting_tips = [
        f"**Cutting Speed Leverage**: In Taylor's tool life equation, cutting speed ($V_c$) has an exponent of $1/n = {1.0 / pairing.taylor_n:.2f}$. A 15% drop in $V_c$ increases tool life by approximately {((1.0 / (0.85 ** (1.0 / pairing.taylor_n))) - 1.0) * 100.0:.0f}%.",
        f"**Depth-of-Cut Utilization**: Depth of cut ($a_p$) exponent is only {pairing.taylor_y:.2f}. You can safely increase $a_p$ to compensate for lower speeds with minimal impact on flank wear.",
        f"**Minimum Chip Load Warning**: Never drop feed below {pairing.f_min:.3f} mm/rev; insufficient feed causes the cutting edge to rub and burnish rather than shear, accelerating flank wear.",
    ]

    # Operation-specific guidance
    if operation_name and operation_name in OPERATION_DATABASE:
        op = OPERATION_DATABASE[operation_name]
        feed_unit = op.feed_unit
        cutting_tips.append(f"**Operation Modality ({op.name})**: {op.best_practices}")
        if op.family == "milling" and milling_tooling_name and milling_tooling_name in MILLING_TOOLING_DATABASE:
            mt = MILLING_TOOLING_DATABASE[milling_tooling_name]
            cutting_tips.append(
                f"**Cutter Engagement ({mt.name.split(',')[0]})**: {mt.teeth} cutting edges, "
                f"{mt.diameter_mm:.0f} mm diameter. {mt.best_practices}"
            )
            if op.ae_fraction_of_d <= 0.15:
                cutting_tips.append(
                    f"**Chip-Thinning Compensation**: At $a_e = {op.ae_fraction_of_d*100:.0f}\\%$ of diameter, apply the feed-correction "
                    "$f_z' = f_z \\cdot (D/(2\\cdot a_e))$ to maintain true chip thickness and avoid rubbing."
                )

    # Tooling & coating advice based on ISO group
    tooling_tips = []
    iso = pairing.workpiece_iso
    if iso == "S":  # Titanium / Inconel
        tooling_tips.append(r"**AlTiN / AlCrN Physical Vapor Deposition (PVD)**: Use PVD coating rather than CVD. PVD retains sharp cutting edges (edge radius $r_n < 15\,\mu\text{m}$), preventing work hardening in heat-resistant superalloys.")
        tooling_tips.append("**Round Insert Geometry (R-Geometry)**: For Inconel 718, switch from rhombic (C/W) to round (RCMT) inserts. Round inserts thin the chip entering and exiting the cut, distributing notch wear across a much larger curved arc.")
    elif iso == "P":  # Steel
        tooling_tips.append("**Multi-Layer CVD Al2O3 Barrier**: For continuous turning of AISI 1045/4140, use CVD multi-layer coatings with an aluminum oxide ($Al_2O_3$) thermal barrier to resist crater wear at temperatures > 750°C.")
    elif iso == "M":  # Stainless Steel
        tooling_tips.append("**Positive Rake & Sharp Honing**: Use positive rake angle inserts (> +12°) with sharp edge preparation to shear austenitic stainless steel below the work-hardened deformation boundary.")
    elif iso == "N":  # Aluminum
        tooling_tips.append("**Mirror-Polished Flutes (Ra < 0.05 µm) or PCD**: Polished un-coated submicron carbide or PCD completely stops aluminum adhesion (Built-Up Edge) without needing thick coatings.")
    elif iso == "H":  # Hardened Steel
        tooling_tips.append("**Negative Chamfered T-Land PCBN**: Use PCBN inserts with a negative chamfer (0.1 mm x 20°) to strengthen the cutting edge against high compressive stresses.")
    elif iso == "K":  # Cast Iron
        tooling_tips.append("**Silicon Nitride Ceramics ($Si_3N_4$)**: Si3N4 ceramic or thick TiCN CVD coating resists abrasive scratching from hard cementite phases in grey cast iron.")

    # Machine-specific vibration & dynamics advice
    machine_tips = [
        f"**Machine Rigidity Profile ({machine.category})**: Current rigidity multiplier is **{machine.rigidity_factor:.2f}x**.",
        f"**Chatter & Vibration Control**: {machine.best_practices}",
    ]
    if "Manual" in machine_name:
        machine_tips.append("**Backlash Precaution**: Avoid climb/down milling on manual machinery. Use conventional up-milling to keep leadscrew threads seated under tension and prevent sudden insert chipping.")
    elif "5-Axis" in machine_name:
        machine_tips.append("**Continuous 5-Axis Angle Optimization**: Tilt the tool 10°–15° relative to the surface normal to ensure the cutting contact point never drops to the zero-surface-speed center of the tool.")
    elif "High-Speed" in machine_name:
        machine_tips.append("**Dynamic Stability Lobes**: In high-speed milling, use tap-testing or harmonic speed tuning to place the spindle speed at a stability lobe peak, eliminating chatter-induced micro-fractures.")

    # Tool holder & overhang advice
    if holder_name and holder_name in TOOL_HOLDER_DATABASE:
        holder = TOOL_HOLDER_DATABASE[holder_name]
        machine_tips.append(
            f"**Tool Holding System ({holder.name.split('(')[0].strip()})**: "
            f"Runout accuracy {holder.runout_accuracy}, rigidity multiplier **{holder.rigidity_multiplier:.2f}x**. "
            f"{holder.best_practices}"
        )
    if overhang_ratio and overhang_ratio > 3.0:
        holder_is_dampened = holder_name and "Dampened" in TOOL_HOLDER_DATABASE[holder_name].name
        dampened_note = (
            "Dampened holder selected — overhang penalty halved."
            if holder_is_dampened
            else "Switching to a dampened anti-vibration holder halves the overhang derate rate (0.070 → 0.035 per unit L/D)."
        )
        machine_tips.append(
            f"**Overhang Management (L/D = {overhang_ratio:.1f})**: Exposed length beyond 3×D amplifies "
            f"chatter and edge chipping; tool life is derated accordingly. {dampened_note}"
        )

    # Coolant advice
    coolant_tips = [
        f"**Selected Cooling**: {coolant.name} (Multiplier: {coolant.life_multiplier:.2f}x).",
        coolant.description,
        f"**Material Compatibility**: {coolant.recommended_materials}",
    ]
    if "Ceramic" in pairing.tool_material and "Flood" in coolant_name:
        coolant_tips.append("⚠️ **CRITICAL WARNING**: Flood coolant on Ceramic inserts causes cyclical thermal shocks leading to rapid comb cracking and catastrophic edge fracture! Switch to Dry Machining or Air Blast immediately.")

    # Pareto curve
    pareto_mrr, pareto_life = generate_pareto_curve(
        pairing, machine_name, coolant_name, is_roughing,
        operation_name=operation_name,
        milling_tooling_name=milling_tooling_name,
        holder_name=holder_name,
        overhang_ratio=overhang_ratio,
    )

    return WearMinimizationReport(
        current_vc=current_vc,
        current_feed=current_feed,
        current_ap=current_ap,
        current_tool_life=round(base_life, 1),
        current_mrr=round(base_mrr, 2),
        operation_name=operation_name,
        milling_tooling_name=milling_tooling_name,
        holder_name=holder_name,
        overhang_ratio=overhang_ratio,
        balanced_strategy=balanced_strat,
        max_life_strategy=max_life_strat,
        high_efficiency_strategy=he_strat,
        cutting_parameter_tips=cutting_tips,
        tooling_coating_tips=tooling_tips,
        machine_vibration_tips=machine_tips,
        coolant_tips=coolant_tips,
        pareto_mrr=pareto_mrr,
        pareto_tool_life=pareto_life,
    )
