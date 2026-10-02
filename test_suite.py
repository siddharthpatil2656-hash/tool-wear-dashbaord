"""
Automated Test Suite for Tool Wear Forecasting Engine.
Tests physics calculations, boundary conditions, bundled model record integrity,
wear curves, and optimization algorithms.
"""

import sys
import numpy as np
from springer_database import SPRINGER_DATABASE, MACHINE_DATABASE, COOLANT_DATABASE
from tool_physics import predict_tool_wear, compute_flank_wear_curve, calculate_mrr
from optimizer import optimize_tool_wear
from presets import INDUSTRY_PRESETS


def run_tests():
    print("==================================================================")
    print("RUNNING AUTOMATED UNIT & INTEGRATION TESTS FOR TOOL WEAR DASHBOARD")
    print("==================================================================")

    passed = 0
    total = 0

    # Test 1: Verify bundled model records have usable coefficients and metadata
    print("\n[Test 1] Validating bundled pairing model records...")
    for key, pairing in SPRINGER_DATABASE.items():
        total += 1
        assert pairing.taylor_C > 0, f"Invalid C for {key}"
        assert 0.05 < pairing.taylor_n < 0.9, f"Unrealistic Taylor n for {key}"
        assert pairing.v_min < pairing.v_max, f"Invalid speed bounds for {key}"
        assert pairing.f_min < pairing.f_max, f"Invalid feed bounds for {key}"
        assert pairing.ap_min < pairing.ap_max, f"Invalid ap bounds for {key}"
        assert pairing.springer_ref.doi, f"Missing DOI for {key}"
        passed += 1
    print(f" -> {passed}/{total} bundled pairing records have usable model inputs.")

    # Test 2: Verify Machine and Coolant databases
    print("\n[Test 2] Validating Machine and Coolant models...")
    for m_name, m_char in MACHINE_DATABASE.items():
        total += 1
        assert 0.5 < m_char.rigidity_factor < 2.0, f"Unrealistic rigidity for {m_name}"
        passed += 1

    for c_name, c_char in COOLANT_DATABASE.items():
        total += 1
        assert 0.4 < c_char.life_multiplier < 2.5, f"Unrealistic coolant multiplier for {c_name}"
        passed += 1
    print(f" -> Machines and coolants validated.")

    # Test 3: Test predictions across all combinations
    print("\n[Test 3] Testing prediction calculations across all pairings...")
    for key, pairing in SPRINGER_DATABASE.items():
        total += 1
        med_v = (pairing.v_min + pairing.v_max) / 2.0
        med_f = (pairing.f_min + pairing.f_max) / 2.0
        med_ap = (pairing.ap_min + pairing.ap_max) / 2.0

        pred = predict_tool_wear(
            pairing_key=key,
            machine_name="5-Axis High-Precision CNC Machining Center",
            coolant_name="Standard Flood Emulsion (7-10% oil)",
            vc=med_v,
            feed=med_f,
            ap=med_ap,
            current_time_min=10.0,
            is_roughing=False,
        )

        assert pred.tool_life_minutes > 1.0, f"Tool life too low for {key}: {pred.tool_life_minutes}"
        assert pred.tool_life_minutes < 1500.0, f"Tool life too high for {key}: {pred.tool_life_minutes}"
        assert pred.mrr_cm3_min > 0, f"MRR must be positive for {key}"
        assert 0.0 <= pred.confidence_score <= 100.0, f"Invalid confidence score for {key}"
        passed += 1
    print(f" -> All {len(SPRINGER_DATABASE)} pairings computed valid, realistic tool life.")

    # Test 4: Test 3-stage wear progression curve
    print("\n[Test 4] Testing 3-Stage Flank Wear (VB) Curve...")
    total += 1
    sample_pairing = SPRINGER_DATABASE["Ti-6Al-4V | PVD TiAlN Carbide"]
    t_arr, vb_arr, vb_s1, vb_s23 = compute_flank_wear_curve(sample_pairing, tool_life_min=45.0, is_roughing=False)
    assert len(t_arr) == len(vb_arr)
    # Check monotonicity of wear
    assert np.all(np.diff(vb_arr) >= -1e-6), "Flank wear curve must be monotonically non-decreasing"
    # Check that near tool life, VB is close to failure threshold
    idx_life = np.argmin(np.abs(t_arr - 45.0))
    assert abs(vb_arr[idx_life] - sample_pairing.vb_critical_finishing) < 0.05, "Wear at tool life must reach threshold"
    passed += 1
    print(" -> 3-Stage Flank Wear curve is smooth, monotonic, and precisely hits failure threshold.")

    # Test 5: Test Optimization Engine
    print("\n[Test 5] Testing Wear Minimization Optimization Engine...")
    total += 1
    opt = optimize_tool_wear(
        pairing_key="Ti-6Al-4V | PVD TiAlN Carbide",
        machine_name="5-Axis High-Precision CNC Machining Center",
        coolant_name="High-Pressure Coolant (70-100 bar)",
        current_vc=80.0,
        current_feed=0.14,
        current_ap=1.2,
        is_roughing=False,
    )
    # Balanced strategy must extend tool life
    assert opt.balanced_strategy.tool_life_gain_pct > 0, "Balanced strategy must improve tool life"
    # Max life strategy must extend tool life even more
    assert opt.max_life_strategy.tool_life_gain_pct > opt.balanced_strategy.tool_life_gain_pct, "Max life must have higher life gain"
    assert len(opt.pareto_mrr) > 10, "Pareto curve must have sufficient points"
    assert len(opt.cutting_parameter_tips) >= 2, "Must provide cutting tips"
    assert len(opt.productivity_frontier_options) == 3, (
        "Optimizer must return feasible low, current, and higher-productivity plans"
    )
    for option in opt.productivity_frontier_options:
        target_mrr = opt.current_mrr * option["target_productivity_pct"] / 100.0
        assert abs(option["mrr_cm3_min"] - target_mrr) / target_mrr < 1e-9
        assert option["tool_life_min"] > 0
        assert 40.0 <= option["vc"] <= 110.0
        assert 0.06 <= option["feed"] <= 0.25
        assert 0.3 <= option["ap"] <= 2.5
        assert option["mrr_error_pct"] <= 1.0
        assert option["recheck_error_pct"] <= 0.1
        assert np.isfinite(option["local_sensitivity_pct"])
    passed += 1
    print(
        f" -> Optimizer generated balanced strategy (+{opt.balanced_strategy.tool_life_gain_pct:.1f}% life), "
        "Pareto curve, and three bounded productivity-target plans."
    )

    # Test 6: Verify all presets load and compute cleanly
    print("\n[Test 6] Testing Industry Case Study Presets...")
    for p_name, preset in INDUSTRY_PRESETS.items():
        total += 1
        p_res = predict_tool_wear(
            pairing_key=preset.pairing_key,
            machine_name=preset.machine_name,
            coolant_name=preset.coolant_name,
            vc=preset.vc,
            feed=preset.feed,
            ap=preset.ap,
            current_time_min=preset.current_time_min,
            is_roughing=preset.is_roughing,
        )
        assert p_res.tool_life_minutes > 0.5
        passed += 1
    # Test 7: Verify Independent Custom Material Selection & Synthesis
    print("\n[Test 7] Testing Independent Custom Material Selection & Synthesis...")
    from springer_database import WORKPIECE_DATABASE, TOOL_DATABASE, COATING_DATABASE, synthesize_custom_pairing

    # Test PCD on Steel (should generate chemical dissolution warning)
    synth_pairing, warnings = synthesize_custom_pairing(
        "AISI 1045 (Medium Carbon Steel)",
        "Polycrystalline Diamond (PCD)",
        "Uncoated Ground / Chamfered"
    )
    total += 1
    assert any("Chemical Dissolution" in w for w in warnings), "Must flag PCD on steel as incompatible"
    passed += 1

    # Test Titanium with PVD AlCrN
    synth_pairing_ti, warnings_ti = synthesize_custom_pairing(
        "Ti-6Al-4V (Titanium Alloy)",
        "Tungsten Carbide (Submicron WC-Co)",
        "PVD AlCrN (Super-Resistant)"
    )
    total += 1
    assert synth_pairing_ti.taylor_n == 0.25
    assert synth_pairing_ti.taylor_C > 0
    passed += 1

    # Test prediction with synthesized pairing
    pred_synth = predict_tool_wear(
        machine_name="5-Axis High-Precision CNC Machining Center",
        coolant_name="High-Pressure Coolant (70-100 bar)",
        vc=70.0,
        feed=0.12,
        ap=1.2,
        current_time_min=10.0,
        is_roughing=False,
        custom_pairing=synth_pairing_ti
    )
    total += 1
    assert pred_synth.tool_life_minutes > 1.0
    passed += 1

    # Test optimization with synthesized pairing
    opt_synth = optimize_tool_wear(
        machine_name="5-Axis High-Precision CNC Machining Center",
        coolant_name="High-Pressure Coolant (70-100 bar)",
        current_vc=70.0,
        current_feed=0.12,
        current_ap=1.2,
        is_roughing=False,
        custom_pairing=synth_pairing_ti
    )
    total += 1
    assert opt_synth.balanced_strategy.tool_life_gain_pct > 0
    passed += 1
    print(" -> Custom independent material synthesis & cross-compatibility tests passed.")

    print("\n==================================================================")
    print(f"ALL TESTS PASSED! ({passed}/{total} checks successful)")
    print("==================================================================")


if __name__ == "__main__":
    run_tests()
