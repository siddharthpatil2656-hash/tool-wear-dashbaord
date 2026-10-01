"""
Machining Intelligence & Tool Wear Forecasting Dashboard.
Grounded in empirical datasets and Taylor constants from Springer IJAMT research papers.
Interactive Streamlit application with Plotly analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from springer_database import (
    SPRINGER_DATABASE,
    MACHINE_DATABASE,
    COOLANT_DATABASE,
    WORKPIECE_DATABASE,
    TOOL_DATABASE,
    COATING_DATABASE,
    synthesize_custom_pairing,
)
from tool_physics import (
    predict_tool_wear,
    compute_flank_wear_curve,
    calculate_tool_life,
    calculate_mrr,
)
from optimizer import optimize_tool_wear
from presets import INDUSTRY_PRESETS


# Page configuration
st.set_page_config(
    page_title="Tool Wear Forecasting & Life Optimization",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Visual system: a compact control rail and a calm, high-contrast analysis surface.
st.markdown("""
<style>
    :root { --ink: #14213d; --muted: #64748b; --line: #e6eaf0; --accent: #1d4ed8; --surface: #ffffff; --canvas: #f7f8fb; }
    .stApp { background: var(--canvas); color: var(--ink); }
    .block-container { padding: 2.4rem 2.8rem 3rem; max-width: 1480px; }
    .main-header {
        font-size: 2.35rem;
        line-height: 1.05;
        font-weight: 720;
        color: var(--ink);
        letter-spacing: -0.045em;
        margin: 0.2rem 0 0.35rem;
    }
    .sub-header {
        font-size: 0.98rem;
        color: var(--muted);
        margin-bottom: 1.6rem;
        max-width: 680px;
        line-height: 1.55;
    }
    .eyebrow {
        color: var(--accent);
        font-size: 0.72rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.14em;
    }
    .kpi-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1.15rem 1.1rem 1.05rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.035);
        min-height: 118px;
    }
    .kpi-title {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: var(--muted);
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 720;
        letter-spacing: -0.035em;
        color: var(--ink);
    }
    .kpi-subtext {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 0.35rem;
    }
    .springer-badge {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #dbeafe;
        border-radius: 999px;
        padding: 3px 9px;
        font-size: 0.73rem;
        font-weight: 650;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 26px;
        border-bottom: 1px solid var(--line);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.65rem 0.1rem 0.7rem;
        font-size: 0.88rem;
        font-weight: 650;
        color: var(--muted);
    }
    .stTabs [aria-selected="true"] { color: var(--ink) !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; height: 2px !important; }
    div[data-testid="stPlotlyChart"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.35rem 0.55rem 0.15rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.025);
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.92rem;
        background-color: #FFFFFF;
        border-radius: 12px;
        overflow: hidden;
    }
    th {
        background-color: #f8fafc !important;
        color: var(--ink) !important;
        font-weight: 600;
        padding: 10px 14px;
        border: 1px solid var(--line);
        text-align: left;
    }
    td {
        padding: 10px 14px;
        border: 1px solid var(--line);
        color: #334155;
    }
    tr:nth-child(even) {
        background-color: #F8FAFC;
    }
    [data-testid="stSidebar"] { background: #111c34; border-right: 0; }
    [data-testid="stSidebar"] > div:first-child { background: #111c34; }
    [data-testid="stSidebar"] .stMarkdown h2 { font-size: 1.3rem; color: #f8fafc; letter-spacing: -0.025em; }
    [data-testid="stSidebar"] .stMarkdown h3 { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.11em; margin-top: 1.45rem; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stCaption { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
    [data-testid="stSidebar"] [data-testid="stTextInput"] input { background: #1b2a49 !important; border-color: #314366 !important; color: #f8fafc !important; border-radius: 8px !important; }
    [data-testid="stSidebar"] [data-testid="stExpander"] { border: 1px solid #314366; border-radius: 10px; background: #162440; }
    [data-testid="stSidebar"] hr { border-color: #314366; }
    [data-testid="stSidebar"] .stAlert { background: #172a4c; color: #dbeafe; border: 1px solid #2e4c7a; border-radius: 10px; }
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div { background-color: #60a5fa; }
    [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: var(--surface); }
    [data-testid="stAlert"] { border-radius: 12px; border: 1px solid #bfdbfe; }
    [data-testid="stDownloadButton"] button { border-radius: 9px; border: 0; background: var(--accent); color: white; font-weight: 650; }
    [data-testid="stDownloadButton"] button:hover { background: #1e40af; color: white; }
    @media (max-width: 900px) { .block-container { padding: 1.35rem 1rem 2rem; } .main-header { font-size: 2rem; } }
</style>
""", unsafe_allow_html=True)


# Sidebar: Presets and User Inputs
with st.sidebar:
    st.markdown("## Machining setup")

    # 1-Click Industry Presets
    st.markdown("### Case library")
    preset_keys = ["-- Custom User Parameters --"] + list(INDUSTRY_PRESETS.keys())
    selected_preset = st.selectbox("Load Industry Case Study:", preset_keys)

    # Defaults
    if selected_preset != "-- Custom User Parameters --":
        p_data = INDUSTRY_PRESETS[selected_preset]
        default_pairing = p_data.pairing_key
        default_machine = p_data.machine_name
        default_coolant = p_data.coolant_name
        default_vc = float(p_data.vc)
        default_feed = float(p_data.feed)
        default_ap = float(p_data.ap)
        default_time = float(p_data.current_time_min)
        default_roughing = p_data.is_roughing
        st.info(f"Loaded preset: **{p_data.target_industry}**\n\n{p_data.description}")
    else:
        default_pairing = "Ti-6Al-4V | PVD TiAlN Carbide"
        default_machine = "5-Axis High-Precision CNC Machining Center"
        default_coolant = "High-Pressure Coolant (70-100 bar)"
        default_vc = 75.0
        default_feed = 0.12
        default_ap = 1.20
        default_time = 15.0
        default_roughing = False

    st.markdown("---")
    st.markdown("### Tool & material")

    # Mode toggle: Independent vs Curated Pairings
    material_mode = st.radio(
        "Material Selection Mode:",
        [
            "🔧 Independent Custom Selection (Tool & Workpiece Separately)",
            "📖 Curated Springer IJAMT Research Pairings"
        ],
        index=0 if selected_preset == "-- Custom User Parameters --" else 1,
        help="Choose whether to select workpiece and tool independently or use curated research literature pairings."
    )

    custom_warnings = []

    if "Independent" in material_mode:
        st.markdown("#### Workpiece Material")
        wp_keys = list(WORKPIECE_DATABASE.keys())
        selected_wp_key = st.selectbox("Select Workpiece Material:", wp_keys, index=0)
        wp_info = WORKPIECE_DATABASE[selected_wp_key]

        st.markdown(
            f"<span class='springer-badge'>ISO Group {wp_info.iso_group}</span> "
            f"<span class='springer-badge'>{wp_info.hardness}</span>",
            unsafe_allow_html=True
        )
        st.caption(f"🔥 **Thermal Conduct.** {wp_info.thermal_conductivity}")

        st.markdown("#### Tool Material & Coating")
        tool_keys = list(TOOL_DATABASE.keys())
        selected_tool_key = st.selectbox("Select Tool Material (Substrate):", tool_keys, index=0)
        tool_info = TOOL_DATABASE[selected_tool_key]

        st.markdown(
            f"<span class='springer-badge'>{tool_info.category}</span> "
            f"<span class='springer-badge'>Taylor n: {tool_info.taylor_n:.3f}</span> "
            f"<span class='springer-badge'>Max Temp: {tool_info.max_temp_c:.0f}°C</span>",
            unsafe_allow_html=True
        )

        coating_keys = list(COATING_DATABASE.keys())
        selected_coating_key = st.selectbox("Select Tool Coating / Prep:", coating_keys, index=0)
        coating_info = COATING_DATABASE[selected_coating_key]
        st.caption(f"🛡️ **Process**: {coating_info.process} | Life Multiplier: {coating_info.life_multiplier:.2f}x")

        # Synthesize dynamic pairing
        pairing_info, custom_warnings = synthesize_custom_pairing(
            selected_wp_key, selected_tool_key, selected_coating_key
        )
        selected_pairing = f"{wp_info.name} | {tool_info.name}"

        # Display chemical/process compatibility warnings
        for warn in custom_warnings:
            st.warning(warn)

        # Optional Advanced Taylor Constant Customizer
        with st.expander("⚙️ Fine-Tune Taylor Constants (Expert Overrides)", expanded=False):
            override_c = st.number_input(
                "Taylor Constant (C):",
                min_value=1.0,
                max_value=25000.0,
                value=float(pairing_info.taylor_C),
                step=5.0
            )
            override_n = st.number_input(
                "Taylor Exponent (n):",
                min_value=0.05,
                max_value=0.90,
                value=float(pairing_info.taylor_n),
                step=0.01,
                format="%.3f"
            )
            pairing_info.taylor_C = override_c
            pairing_info.taylor_n = override_n

    else:
        pairing_list = list(SPRINGER_DATABASE.keys())
        pairing_index = pairing_list.index(default_pairing) if default_pairing in pairing_list else 0
        selected_pairing = st.selectbox(
            "Curated Workpiece & Tool Combination:",
            pairing_list,
            index=pairing_index,
            help="Calibrated empirical pairings from peer-reviewed Springer IJAMT research studies."
        )
        pairing_info = SPRINGER_DATABASE[selected_pairing]

        # Display material badges
        st.markdown(
            f"<span class='springer-badge'>ISO Group {pairing_info.workpiece_iso}</span> "
            f"<span class='springer-badge'>{pairing_info.tool_material}</span>",
            unsafe_allow_html=True
        )
        st.caption(f"**Coating**: {pairing_info.coating}")

    st.markdown("---")
    st.markdown("### Machine & environment")

    machine_list = list(MACHINE_DATABASE.keys())
    machine_index = machine_list.index(default_machine) if default_machine in machine_list else 0
    selected_machine = st.selectbox(
        "Which Machine is Used for Machining?",
        machine_list,
        index=machine_index,
        help="Machine rigidity directly scales dynamic chatter vibration and tool degradation rate."
    )

    coolant_list = list(COOLANT_DATABASE.keys())
    coolant_index = coolant_list.index(default_coolant) if default_coolant in coolant_list else 0
    selected_coolant = st.selectbox(
        "Cooling / Lubrication Method:",
        coolant_list,
        index=coolant_index,
    )

    is_roughing = st.checkbox(
        "Heavy Roughing Cut (VB limit = 0.5-0.6 mm)",
        value=default_roughing,
        help="ISO 3685 defines VB=0.3mm for precision finishing, or 0.5-0.6mm for roughing."
    )

    st.markdown("---")
    st.markdown("### Cutting parameters")

    with st.expander("Case-based setup", expanded=False):
        case_name = st.text_input("Case name", value=selected_preset if selected_preset != "-- Custom User Parameters --" else "Custom machining case")
        operation_type = st.selectbox("Operation", ["Turning", "Milling", "Drilling / boring", "General workshop"], index=0)
        production_goal = st.radio("Production priority", ["Balanced", "Tool life", "Throughput"], horizontal=True)
        st.caption("These details are saved in the report so each run can be traced to its practical use case.")

    # Reference bounds provide context, while each case may select its own safe maximum.
    v_rec_min = float(pairing_info.v_min)
    v_rec_max = float(pairing_info.v_max)

    machine_speed_cap = st.number_input(
        "Maximum cutting speed for this case (m/min):",
        min_value=1.0,
        max_value=10000.0,
        value=float(max(v_rec_max * 1.8, default_vc, 100.0)),
        step=10.0,
        help="Set this to the capability or process limit for the current job. Cutting speed is selectable from 0 to this maximum."
    )

    vc_input = st.slider(
        "Cutting Speed (Vc, m/min):",
        min_value=0.0,
        max_value=float(machine_speed_cap),
        value=float(min(default_vc, machine_speed_cap)),
        step=1.0,
        help=f"0 means the machine is stopped. Verified reference range: {v_rec_min:.0f} – {v_rec_max:.0f} m/min."
    )

    feed_input = st.slider(
        "Feed Rate (f, mm/rev or mm/tooth):",
        min_value=max(0.01, float(pairing_info.f_min * 0.5)),
        max_value=float(pairing_info.f_max * 1.8),
        value=float(min(max(default_feed, pairing_info.f_min * 0.6), pairing_info.f_max * 1.6)),
        step=0.01,
        format="%.3f",
        help=f"Empirical research range: {pairing_info.f_min:.3f} – {pairing_info.f_max:.3f} mm/rev"
    )

    ap_input = st.slider(
        "Depth of Cut (ap, mm):",
        min_value=max(0.1, float(pairing_info.ap_min * 0.5)),
        max_value=float(pairing_info.ap_max * 1.8),
        value=float(min(max(default_ap, pairing_info.ap_min * 0.6), pairing_info.ap_max * 1.6)),
        step=0.1,
        help=f"Empirical research range: {pairing_info.ap_min:.1f} – {pairing_info.ap_max:.1f} mm"
    )

    st.markdown("---")
    st.markdown("### Tool in-service monitor")
    current_time_input = st.number_input(
        "Current Elapsed Cutting Time (minutes):",
        min_value=0.0,
        max_value=500.0,
        value=float(default_time),
        step=1.0,
        help="Enter current machining time on this tool edge to estimate Remaining Useful Life (RUL)."
    )


# Perform Predictions & Optimization
pred = predict_tool_wear(
    pairing_key=selected_pairing,
    machine_name=selected_machine,
    coolant_name=selected_coolant,
    vc=vc_input,
    feed=feed_input,
    ap=ap_input,
    current_time_min=current_time_input,
    is_roughing=is_roughing,
    custom_pairing=pairing_info,
)

is_idle = vc_input <= 0

opt_report = optimize_tool_wear(
    pairing_key=selected_pairing,
    machine_name=selected_machine,
    coolant_name=selected_coolant,
    current_vc=vc_input,
    current_feed=feed_input,
    current_ap=ap_input,
    is_roughing=is_roughing,
    custom_pairing=pairing_info,
)


# Main Content Area Header
st.markdown("<div class='eyebrow'>Machining intelligence</div><div class='main-header'>Machining tool wear & life</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='sub-header'>A concise tool-life forecast using peer-reviewed research and clearly labelled handbook starting guidance.</div>",
    unsafe_allow_html=True
)

if is_idle:
    st.info("Machine stopped — set a positive cutting speed to generate an active wear and RUL forecast. The optimisation panel uses the lowest validated speed as a restart reference.")


# Key Metrics Row
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Predicted Tool Life</div>
            <div class='kpi-value'>{pred.tool_life_minutes:.1f} <span style='font-size:1.1rem;'>min</span></div>
            <div class='kpi-subtext'>Until VB reaches {pred.vb_threshold_mm:.2f} mm</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    status_color = "#16A34A" if pred.rul_percentage > 35 else ("#CA8A04" if pred.rul_percentage > 12 else "#DC2626")
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Remaining Useful Life</div>
            <div class='kpi-value' style='color:{status_color};'>{pred.rul_minutes:.1f} <span style='font-size:1.1rem;'>min</span></div>
            <div class='kpi-subtext'>{pred.rul_percentage:.0f}% Life Left ({pred.current_time_min:.0f}m elapsed)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Flank Wear (VB)</div>
            <div class='kpi-value'>{pred.current_vb_mm:.3f} <span style='font-size:1.1rem;'>mm</span></div>
            <div class='kpi-subtext'>Limit: {pred.vb_threshold_mm:.2f} mm (ISO 3685)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Removal Rate (MRR)</div>
            <div class='kpi-value'>{pred.mrr_cm3_min:.1f} <span style='font-size:1.1rem;'>cm³/m</span></div>
            <div class='kpi-subtext'>Total Vol: {pred.total_volume_cut_cm3:.0f} cm³</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c5:
    conf_color = "#16A34A" if pred.confidence_score >= 85 else ("#CA8A04" if pred.confidence_score >= 60 else "#EA580C")
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Data Confidence</div>
            <div class='kpi-value' style='color:{conf_color};'>{pred.confidence_score:.0f}%</div>
            <div class='kpi-subtext' style='color:{conf_color};'>{pred.confidence_label.split('(')[0]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)


# Tab Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "Wear progression",
    "Optimisation",
    "Evidence",
    "Compare & export"
])


# ==============================================================================
# TAB 1: Graphical Forecasting & Wear Curves
# ==============================================================================
with tab1:
    st.subheader("1. Progressive Flank Wear (VB) vs. Machining Time")
    st.caption("Standard 3-stage wear curve: Stage I (Initial Break-in) ➔ Stage II (Uniform Steady-State) ➔ Stage III (Catastrophic Failure Runaway).")

    # Generate Wear Curve
    t_arr, vb_total, vb_s1, vb_s23 = compute_flank_wear_curve(
        pred.pairing, pred.tool_life_minutes, pred.is_roughing, time_steps=200
    )

    fig_wear = go.Figure()

    # Flank wear curve
    fig_wear.add_trace(go.Scatter(
        x=t_arr,
        y=vb_total,
        mode='lines',
        name='Predicted Flank Wear VB(t)',
        line=dict(color='#0284C7', width=3.5),
        hovertemplate='Time: %{x:.1f} min<br>Flank Wear VB: %{y:.3f} mm<extra></extra>'
    ))

    # Failure Threshold Line
    fig_wear.add_hline(
        y=pred.vb_threshold_mm,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"ISO 3685 Failure Criterion: VB = {pred.vb_threshold_mm:.2f} mm",
        annotation_position="top left",
        annotation_font_color="#DC2626"
    )

    # Caution boundary (80% of limit)
    fig_wear.add_hrect(
        y0=pred.vb_threshold_mm * 0.80,
        y1=pred.vb_threshold_mm,
        fillcolor="#FEF08A",
        opacity=0.25,
        line_width=0,
        annotation_text="Accelerated Wear Warning Zone (Stage III)",
        annotation_position="bottom right"
    )

    # Current time marker
    if pred.current_time_min > 0:
        fig_wear.add_vline(
            x=pred.current_time_min,
            line_dash="dot",
            line_color="#475569",
            line_width=2,
            annotation_text=f"Current: {pred.current_time_min:.1f} min",
            annotation_position="top right"
        )
        fig_wear.add_trace(go.Scatter(
            x=[pred.current_time_min],
            y=[pred.current_vb_mm],
            mode='markers+text',
            name='Current Operating State',
            text=[f"VB={pred.current_vb_mm:.3f}mm"],
            textposition="top center",
            marker=dict(size=13, color=status_color, symbol='diamond')
        ))

    # Tool Life end marker
    fig_wear.add_trace(go.Scatter(
        x=[pred.tool_life_minutes],
        y=[pred.vb_threshold_mm],
        mode='markers+text',
        name='Tool Life Limit (T)',
        text=[f"Life: {pred.tool_life_minutes:.1f}m"],
        textposition="bottom right",
        marker=dict(size=11, color='#DC2626', symbol='x')
    ))

    fig_wear.update_layout(
        xaxis_title="Machining Time (minutes)",
        yaxis_title="Flank Wear Land Width VB (mm)",
        template="plotly_white",
        hovermode="x unified",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_wear, use_container_width=True)

    # Section 2: Sensitivity Analysis (Two Columns)
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("2. Wear Rate vs. Cutting Speed")
        st.caption(f"Average flank wear rate (µm/min) across the cutting speed range. Higher wear rate = faster tool degradation. The reference range is shaded green.")

        v_sweep = np.linspace(pred.pairing.v_min * 0.7, pred.pairing.v_max * 1.4, 60)
        t_sweep = [
            calculate_tool_life(pred.pairing, pred.machine, pred.coolant, v, pred.feed, pred.ap, pred.is_roughing)
            for v in v_sweep
        ]
        # Wear rate = VB_critical / tool_life  (µm/min)
        vb_crit_um = pred.vb_threshold_mm * 1000.0  # convert mm → µm
        wear_rate_sweep = [vb_crit_um / max(t, 0.01) for t in t_sweep]

        fig_taylor = go.Figure()
        fig_taylor.add_trace(go.Scatter(
            x=v_sweep,
            y=wear_rate_sweep,
            mode='lines',
            name='Wear Rate (µm/min)',
            line=dict(color='#8B5CF6', width=3),
            fill='tozeroy',
            fillcolor='rgba(139,92,246,0.08)',
            hovertemplate='Speed: %{x:.1f} m/min<br>Wear Rate: %{y:.2f} µm/min<extra></extra>'
        ))

        # Mark current speed wear rate
        current_wear_rate = vb_crit_um / max(pred.tool_life_minutes, 0.01)
        fig_taylor.add_trace(go.Scatter(
            x=[pred.vc],
            y=[current_wear_rate],
            mode='markers+text',
            name='Current Operating Point',
            text=[f"{current_wear_rate:.2f} µm/min"],
            textposition="top right",
            marker=dict(size=12, color='#DC2626', symbol='circle')
        ))

        # Add empirical boundaries
        fig_taylor.add_vrect(
            x0=pred.pairing.v_min,
            x1=pred.pairing.v_max,
            fillcolor="#22C55E",
            opacity=0.1,
            line_width=0,
            annotation_text="Reference Range",
            annotation_position="top left"
        )

        fig_taylor.update_layout(
            xaxis_title="Cutting Speed Vc (m/min)",
            yaxis_title="Avg. Flank Wear Rate (µm/min)",
            template="plotly_white",
            height=380,
            margin=dict(l=30, r=30, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_taylor, use_container_width=True)

    with col_g2:
        st.subheader("3. 2D Tool Life Contour (Speed vs. Feed)")
        st.caption("Contour map showing predicted tool life (minutes) across varying cutting speeds and feed rates.")

        v_grid = np.linspace(pred.pairing.v_min * 0.8, pred.pairing.v_max * 1.25, 25)
        f_grid = np.linspace(pred.pairing.f_min * 0.8, pred.pairing.f_max * 1.25, 25)
        V_mesh, F_mesh = np.meshgrid(v_grid, f_grid)

        Z_life = np.zeros(V_mesh.shape)
        for i in range(V_mesh.shape[0]):
            for j in range(V_mesh.shape[1]):
                Z_life[i, j] = calculate_tool_life(
                    pred.pairing, pred.machine, pred.coolant,
                    V_mesh[i, j], F_mesh[i, j], pred.ap, pred.is_roughing
                )

        fig_contour = go.Figure(data=go.Contour(
            z=Z_life,
            x=v_grid,
            y=f_grid,
            colorscale='Viridis',
            contours=dict(showlabels=True, labelfont=dict(size=11, color='white')),
            colorbar=dict(title="Life (min)")
        ))

        # Mark current point
        fig_contour.add_trace(go.Scatter(
            x=[pred.vc],
            y=[pred.feed],
            mode='markers+text',
            name='Operating Point',
            text=["Current"],
            textposition="top center",
            marker=dict(size=12, color='#EF4444', symbol='cross')
        ))

        fig_contour.update_layout(
            xaxis_title="Cutting Speed Vc (m/min)",
            yaxis_title="Feed Rate f (mm/rev)",
            template="plotly_white",
            height=380,
            margin=dict(l=30, r=30, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_contour, use_container_width=True)


# ==============================================================================
# TAB 2: How to Minimize Tool Wear (Optimization Engine)
# ==============================================================================
with tab2:
    st.subheader("🎯 How to Minimize Tool Wear: Optimization Strategies")
    st.markdown("""
    Tool wear is governed by friction, interface temperatures, mechanical fatigue, and chemical diffusion. 
    Below are **mathematically optimized strategies** and physics-backed methods to extend tool life without sacrificing production throughput.
    """)

    # Strategy Comparison Table
    opt_col1, opt_col2, opt_col3 = st.columns(3)

    # Strategy 1: Balanced
    with opt_col1:
        st.markdown("### 🏆 Recommended Strategy")
        st.markdown(f"**{opt_report.balanced_strategy.strategy_name}**")
        st.info(opt_report.balanced_strategy.rationale)

        b_life_gain = opt_report.balanced_strategy.tool_life_gain_pct
        st.metric(
            label="Predicted Tool Life",
            value=f"{opt_report.balanced_strategy.tool_life_min} min",
            delta=f"+{b_life_gain:.1f}% Life Extension"
        )
        st.metric(
            label="Material Removal Rate (MRR)",
            value=f"{opt_report.balanced_strategy.mrr_cm3_min} cm³/min",
            delta=f"{opt_report.balanced_strategy.mrr_change_pct:+.1f}% Productivity",
            delta_color="off"
        )

        st.markdown(f"""
        - **Speed ($V_c$)**: `{opt_report.balanced_strategy.vc} m/min` *(from {pred.vc:.0f})*
        - **Feed ($f$)**: `{opt_report.balanced_strategy.feed} mm/rev` *(from {pred.feed:.3f})*
        - **Depth ($a_p$)**: `{opt_report.balanced_strategy.ap} mm` *(from {pred.ap:.2f})*
        """)

    # Strategy 2: Max Life
    with opt_col2:
        st.markdown("### 🛡️ Lights-Out Endurance Mode")
        st.markdown(f"**{opt_report.max_life_strategy.strategy_name}**")
        st.info(opt_report.max_life_strategy.rationale)

        m_life_gain = opt_report.max_life_strategy.tool_life_gain_pct
        st.metric(
            label="Predicted Tool Life",
            value=f"{opt_report.max_life_strategy.tool_life_min} min",
            delta=f"+{m_life_gain:.1f}% Maximum Life"
        )
        st.metric(
            label="Material Removal Rate (MRR)",
            value=f"{opt_report.max_life_strategy.mrr_cm3_min} cm³/min",
            delta=f"{opt_report.max_life_strategy.mrr_change_pct:+.1f}% MRR Trade-off",
            delta_color="normal"
        )

        st.markdown(f"""
        - **Speed ($V_c$)**: `{opt_report.max_life_strategy.vc} m/min`
        - **Feed ($f$)**: `{opt_report.max_life_strategy.feed} mm/rev`
        - **Depth ($a_p$)**: `{opt_report.max_life_strategy.ap} mm`
        """)

    # Strategy 3: High Efficiency
    with opt_col3:
        st.markdown("### ⚡ High Volumetric Output")
        st.markdown(f"**{opt_report.high_efficiency_strategy.strategy_name}**")
        st.info(opt_report.high_efficiency_strategy.rationale)

        h_life_gain = opt_report.high_efficiency_strategy.tool_life_gain_pct
        st.metric(
            label="Predicted Tool Life",
            value=f"{opt_report.high_efficiency_strategy.tool_life_min} min",
            delta=f"{h_life_gain:+.1f}% Life Change"
        )
        st.metric(
            label="Material Removal Rate (MRR)",
            value=f"{opt_report.high_efficiency_strategy.mrr_cm3_min} cm³/min",
            delta=f"+{opt_report.high_efficiency_strategy.mrr_change_pct:.1f}% Throughput Boost",
            delta_color="normal"
        )

        st.markdown(f"""
        - **Speed ($V_c$)**: `{opt_report.high_efficiency_strategy.vc} m/min`
        - **Feed ($f$)**: `{opt_report.high_efficiency_strategy.feed} mm/rev`
        - **Depth ($a_p$)**: `{opt_report.high_efficiency_strategy.ap} mm`
        """)

    st.markdown("---")

    # Pareto Frontier Plot
    col_p1, col_p2 = st.columns([3, 2])

    with col_p1:
        st.subheader("Pareto Trade-Off Curve: Tool Life vs. Productivity")
        st.caption("Visualizing the non-dominated Pareto front between Tool Life (min) and Material Removal Rate ($cm^3/min$).")

        fig_pareto = go.Figure()

        fig_pareto.add_trace(go.Scatter(
            x=opt_report.pareto_mrr,
            y=opt_report.pareto_tool_life,
            mode='lines+markers',
            name='Pareto Optimal Frontier',
            line=dict(color='#0D9488', width=3),
            marker=dict(size=6, color='#0D9488'),
            hovertemplate='MRR: %{x:.1f} cm³/min<br>Tool Life: %{y:.1f} min<extra></extra>'
        ))

        # Mark current point
        fig_pareto.add_trace(go.Scatter(
            x=[pred.mrr_cm3_min],
            y=[pred.tool_life_minutes],
            mode='markers+text',
            name='Current Setting',
            text=["Current"],
            textposition="top right",
            marker=dict(size=13, color='#DC2626', symbol='circle')
        ))

        # Mark Balanced point
        fig_pareto.add_trace(go.Scatter(
            x=[opt_report.balanced_strategy.mrr_cm3_min],
            y=[opt_report.balanced_strategy.tool_life_min],
            mode='markers+text',
            name='Balanced Low-Wear Point',
            text=["Optimized (Same MRR)"],
            textposition="bottom left",
            marker=dict(size=13, color='#16A34A', symbol='star')
        ))

        fig_pareto.update_layout(
            xaxis_title="Material Removal Rate MRR (cm³/min)",
            yaxis_title="Tool Life (minutes)",
            template="plotly_white",
            height=400,
            margin=dict(l=30, r=30, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    with col_p2:
        st.subheader("Physics Rationale & Actionable Guidelines")

        with st.expander("1. Cutting Speed ($V_c$) & Feed ($f$) Rules", expanded=True):
            for tip in opt_report.cutting_parameter_tips:
                st.markdown(f"- {tip}")

        with st.expander("2. Tool Coating & Geometry Matching", expanded=True):
            for tip in opt_report.tooling_coating_tips:
                st.markdown(f"- {tip}")

        with st.expander("3. Machine Dynamics & Chatter Control", expanded=False):
            for tip in opt_report.machine_vibration_tips:
                st.markdown(f"- {tip}")

        with st.expander("4. Cooling & Lubrication Strategy", expanded=False):
            for tip in opt_report.coolant_tips:
                st.markdown(f"- {tip}")


# ==============================================================================
# TAB 3: Springer Research Reference & Evidence
# ==============================================================================
with tab3:
    st.subheader("Reference evidence")
    ref = pred.pairing.springer_ref

    source_link = f"[Source]({ref.source_url})" if ref.source_url else ""
    identifier = f"DOI: [{ref.doi}](https://doi.org/{ref.doi})" if ref.doi else "No DOI — handbook reference"
    st.caption(ref.evidence_level)
    st.markdown(f"""
    ### {ref.title}
    **Authors**: {ref.authors}  
    **Journal**: *{ref.journal}* | {ref.volume_issue} ({ref.year})  
    **Reference**: {identifier} {source_link}
    """)

    st.markdown("---")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("#### 🔬 Experimental Methodology & Setup")
        st.info(ref.experimental_setup)

        st.markdown("#### 🔍 Observed Wear Mechanisms (SEM Analysis)")
        st.warning(ref.observed_wear_mechanisms)

    with col_r2:
        st.markdown("#### 💡 Key Research Findings & Taylor Constants")
        st.success(ref.key_findings)

        st.markdown("#### 📐 Calibrated Extended Taylor Model Constants")
        st.markdown(f"""
        $$V_c \\cdot T^{{{pred.pairing.taylor_n:.3f}}} \\cdot f^{{{pred.pairing.taylor_x:.3f}}} \\cdot a_p^{{{pred.pairing.taylor_y:.3f}}} = {pred.pairing.taylor_C:.1f}$$
        - **Taylor Constant ($C$)**: `{pred.pairing.taylor_C:.1f}`
        - **Taylor Exponent ($n$)**: `{pred.pairing.taylor_n:.3f}` *(Sensitivity = {1.0/pred.pairing.taylor_n:.2f})*
        - **Feed Exponent ($x$)**: `{pred.pairing.taylor_x:.3f}`
        - **Depth Exponent ($y$)**: `{pred.pairing.taylor_y:.3f}`
        """)

    st.markdown("---")
    st.subheader("Experimental Validity Envelope & Confidence Audit")
    
    st.markdown(f"**Overall Prediction Confidence**: `{pred.confidence_score:.0f}%` ({pred.confidence_label})")
    
    # Boundary validation table
    boundary_df = pd.DataFrame([
        {
            "Parameter": "Cutting Speed (Vc)",
            "User Value": f"{pred.vc:.1f} m/min",
            "Reference Range": f"{pred.pairing.v_min:.1f} – {pred.pairing.v_max:.1f} m/min",
            "In Boundary": "✅ YES" if (pred.pairing.v_min <= pred.vc <= pred.pairing.v_max) else "⚠️ EXTRAPOLATED"
        },
        {
            "Parameter": "Feed Rate (f)",
            "User Value": f"{pred.feed:.3f} mm/rev",
            "Reference Range": f"{pred.pairing.f_min:.3f} – {pred.pairing.f_max:.3f} mm/rev",
            "In Boundary": "✅ YES" if (pred.pairing.f_min <= pred.feed <= pred.pairing.f_max) else "⚠️ EXTRAPOLATED"
        },
        {
            "Parameter": "Depth of Cut (ap)",
            "User Value": f"{pred.ap:.2f} mm",
            "Reference Range": f"{pred.pairing.ap_min:.2f} – {pred.pairing.ap_max:.2f} mm",
            "In Boundary": "✅ YES" if (pred.pairing.ap_min <= pred.ap <= pred.pairing.ap_max) else "⚠️ EXTRAPOLATED"
        }
    ])
    st.markdown(boundary_df.to_html(index=False, escape=False), unsafe_allow_html=True)

    if pred.confidence_notes:
        st.markdown("##### 📝 Engineering Audit Notes:")
        for note in pred.confidence_notes:
            st.markdown(f"- {note}")


# ==============================================================================
# TAB 4: Batch Comparison & Export
# ==============================================================================
with tab4:
    st.subheader("📊 Scenario Simulation Comparison & Export")
    st.caption("Compare your current machining condition against the mathematically optimized low-wear scenarios.")

    summary_df = pd.DataFrame([
        {
            "Scenario": "Current Setting",
            "Speed (m/min)": pred.vc,
            "Feed (mm/rev)": pred.feed,
            "Depth ap (mm)": pred.ap,
            "Tool Life (min)": round(pred.tool_life_minutes, 1),
            "MRR (cm³/min)": round(pred.mrr_cm3_min, 2),
            "Life Gain (%)": "0.0%",
            "Productivity Change (%)": "0.0%"
        },
        {
            "Scenario": "Optimized (Balanced / Low-Wear)",
            "Speed (m/min)": opt_report.balanced_strategy.vc,
            "Feed (mm/rev)": opt_report.balanced_strategy.feed,
            "Depth ap (mm)": opt_report.balanced_strategy.ap,
            "Tool Life (min)": opt_report.balanced_strategy.tool_life_min,
            "MRR (cm³/min)": opt_report.balanced_strategy.mrr_cm3_min,
            "Life Gain (%)": f"+{opt_report.balanced_strategy.tool_life_gain_pct:.1f}%",
            "Productivity Change (%)": f"{opt_report.balanced_strategy.mrr_change_pct:+.1f}%"
        },
        {
            "Scenario": "Max Life (Lights-Out Machining)",
            "Speed (m/min)": opt_report.max_life_strategy.vc,
            "Feed (mm/rev)": opt_report.max_life_strategy.feed,
            "Depth ap (mm)": opt_report.max_life_strategy.ap,
            "Tool Life (min)": opt_report.max_life_strategy.tool_life_min,
            "MRR (cm³/min)": opt_report.max_life_strategy.mrr_cm3_min,
            "Life Gain (%)": f"+{opt_report.max_life_strategy.tool_life_gain_pct:.1f}%",
            "Productivity Change (%)": f"{opt_report.max_life_strategy.mrr_change_pct:+.1f}%"
        },
        {
            "Scenario": "High Efficiency (MRR Boost)",
            "Speed (m/min)": opt_report.high_efficiency_strategy.vc,
            "Feed (mm/rev)": opt_report.high_efficiency_strategy.feed,
            "Depth ap (mm)": opt_report.high_efficiency_strategy.ap,
            "Tool Life (min)": opt_report.high_efficiency_strategy.tool_life_min,
            "MRR (cm³/min)": opt_report.high_efficiency_strategy.mrr_cm3_min,
            "Life Gain (%)": f"{opt_report.high_efficiency_strategy.tool_life_gain_pct:+.1f}%",
            "Productivity Change (%)": f"+{opt_report.high_efficiency_strategy.mrr_change_pct:.1f}%"
        },
    ])

    st.markdown(summary_df.to_html(index=False, escape=False), unsafe_allow_html=True)

    # Download CSV
    csv_data = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Machining Optimization Report (CSV)",
        data=csv_data,
        file_name=f"tool_wear_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("### 📋 Complete Machining Specification Summary")
    st.json({
        "case_name": case_name,
        "operation_type": operation_type,
        "production_priority": production_goal,
        "workpiece_material": pred.pairing.workpiece_name,
        "workpiece_iso_group": pred.pairing.workpiece_iso,
        "tool_material": pred.pairing.tool_material,
        "tool_coating": pred.pairing.coating,
        "machine_tool_selected": pred.machine_name,
        "machine_rigidity_multiplier": pred.machine.rigidity_factor,
        "coolant_applied": pred.coolant_name,
        "cutting_speed_vc_m_min": pred.vc,
        "feed_rate_mm_rev": pred.feed,
        "depth_of_cut_ap_mm": pred.ap,
        "roughing_mode": pred.is_roughing,
        "predicted_tool_life_min": pred.tool_life_minutes,
        "remaining_useful_life_min": pred.rul_minutes,
        "current_flank_wear_vb_mm": pred.current_vb_mm,
        "critical_flank_wear_vb_mm": pred.vb_threshold_mm,
        "material_removal_rate_cm3_min": pred.mrr_cm3_min,
        "total_volume_removed_cm3": pred.total_volume_cut_cm3,
        "springer_reference_doi": pred.pairing.springer_ref.doi
    })
