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
from io import BytesIO
import re
from pathlib import Path
from types import SimpleNamespace
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from springer_database import (
    SPRINGER_DATABASE,
    MACHINE_DATABASE,
    COOLANT_DATABASE,
    WORKPIECE_DATABASE,
    TOOL_DATABASE,
    COATING_DATABASE,
    OPERATION_DATABASE,
    MILLING_TOOLING_DATABASE,
    TOOL_HOLDER_DATABASE,
    synthesize_custom_pairing,
)
from tool_physics import (
    predict_tool_wear,
    compute_flank_wear_curve,
    calculate_tool_life,
    calculate_mrr,
    effective_coolant_factor,
)
from optimizer import optimize_tool_wear
from presets import INDUSTRY_PRESETS


# "-- No Selection --" means that field was left empty: modifiers fall back to
# neutral physics (1.0x / None), while mandatory material inputs prompt the user.
NO_SELECTION = "-- No Selection --"
NEUTRAL_MACHINE_KEY = "Unspecified Machine (Neutral Rigidity 1.0x)"
NEUTRAL_COOLANT_KEY = "No Selection (Neutral 1.0x)"
RESET_OPTION = "Reset to default"


# Page configuration
st.set_page_config(
    page_title="Tool Wear Forecasting & Life Optimization",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern industrial styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0F172A;
    }
    .kpi-subtext {
        font-size: 0.82rem;
        color: #0284C7;
        margin-top: 0.2rem;
    }
    .springer-badge {
        background-color: #EEF2FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.92rem;
        background-color: #FFFFFF;
        border-radius: 8px;
        overflow: hidden;
    }
    th {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        font-weight: 600;
        padding: 10px 14px;
        border: 1px solid #CBD5E1;
        text-align: left;
    }
    td {
        padding: 10px 14px;
        border: 1px solid #E2E8F0;
        color: #334155;
    }
    tr:nth-child(even) {
        background-color: #F8FAFC;
    }
    [data-testid="stSidebar"] { border-right: 1px solid #E5E7EB; }
    [data-testid="stSidebar"] .stMarkdown h2 { font-size: 1.15rem; }
</style>
""", unsafe_allow_html=True)


CUSTOM_PARAMS = "-- Custom User Parameters --"


def _reset_dropdown(key, default_value):
    """Restore one selectbox to its current dashboard default."""
    if st.session_state.get(key) == RESET_OPTION:
        st.session_state[key] = default_value


def _on_preset_change():
    _reset_dropdown("preset_choice", CUSTOM_PARAMS)
    _apply_industry_preset()


def _on_machine_change(default_machine):
    _reset_dropdown("machine_choice", default_machine)
    _sync_dependent_widgets()


def _pdf_text(value):
    """Convert dashboard text to characters supported by ReportLab's base font."""
    substitutions = {
        "µ": "u",
        "³": "3",
        "–": "-",
        "—": "-",
        "×": "x",
        "°": " deg",
        "₂": "2",
        "₃": "3",
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "→": "->",
        "≥": ">=",
        "≤": "<=",
        "≈": "~",
    }
    text = str(value)
    for source, replacement in substitutions.items():
        text = text.replace(source, replacement)
    return text.encode("cp1252", errors="replace").decode("cp1252")


def _valid_doi(doi):
    return bool(re.fullmatch(r"10\.\d{4,9}/\S+", str(doi or "")))


def _build_pdf_report(summary_df, prediction):
    """Create a compact, portrait-oriented PDF report for the current forecast."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Machining Tool Wear Forecast",
        author="Machining Tool Wear Dashboard",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0F172A"),
        fontSize=17,
        leading=21,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="ReportSection",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1D4ED8"),
        fontSize=11,
        leading=14,
        spaceBefore=9,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="ReportBody",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="ReportSmall",
        parent=styles["BodyText"],
        fontSize=6.5,
        leading=8,
    ))
    styles.add(ParagraphStyle(
        name="ReportHeader",
        parent=styles["BodyText"],
        fontSize=6.5,
        leading=8,
        textColor=colors.white,
    ))

    def paragraph(value, style="ReportBody"):
        return Paragraph(escape(_pdf_text(value)), styles[style])

    story = [
        Paragraph("Machining Tool Wear Forecast", styles["ReportTitle"]),
        paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
        Paragraph("Machining setup and current forecast", styles["ReportSection"]),
    ]

    setup_rows = [
        ["Workpiece", _pdf_text(prediction.pairing.workpiece_name)],
        ["Tool / coating", _pdf_text(f"{prediction.pairing.tool_material} / {prediction.pairing.coating}")],
        ["Machine", _pdf_text(prediction.machine_name)],
        ["Operation", _pdf_text(prediction.operation_name or "Not specified")],
        ["Coolant", _pdf_text(prediction.coolant_name)],
        ["Cutting parameters", _pdf_text(
            f"Vc {prediction.vc:.1f} m/min; feed {prediction.feed:.3f}; "
            f"ap {prediction.ap:.2f} mm; overhang L/D {prediction.overhang_ratio:.1f}"
        )],
        ["Predicted tool life", f"{prediction.tool_life_minutes:.1f} min"],
        ["Remaining useful life", f"{prediction.rul_minutes:.1f} min ({prediction.rul_percentage:.0f}%)"],
        ["Current flank wear", f"{prediction.current_vb_mm:.3f} mm (limit {prediction.vb_threshold_mm:.2f} mm)"],
        ["Material removal rate", f"{prediction.mrr_cm3_min:.2f} cm3/min"],
        ["Prediction confidence", f"{prediction.confidence_score:.0f}% - {_pdf_text(prediction.confidence_label)}"],
    ]
    setup_table = Table(
        [[paragraph(label), paragraph(value)] for label, value in setup_rows],
        colWidths=[1.45 * inch, 5.45 * inch],
        hAlign="LEFT",
    )
    setup_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EFF6FF")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([setup_table, Paragraph("Scenario comparison", styles["ReportSection"])])

    headers = [paragraph(column, "ReportHeader") for column in summary_df.columns]
    comparison_rows = [headers]
    for row in summary_df.itertuples(index=False, name=None):
        comparison_rows.append([paragraph(value, "ReportSmall") for value in row])
    comparison_table = Table(
        comparison_rows,
        colWidths=[1.42 * inch, 0.72 * inch, 0.58 * inch, 0.55 * inch,
                   0.70 * inch, 0.64 * inch, 0.70 * inch, 0.78 * inch],
        repeatRows=1,
        hAlign="LEFT",
    )
    comparison_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([comparison_table, Paragraph("Evidence and references", styles["ReportSection"])])

    reference = prediction.pairing.springer_ref
    source_text = (
        f"{reference.authors} ({reference.year}). {reference.title}. "
        f"{reference.journal}, {reference.volume_issue}."
    )
    if _valid_doi(reference.doi) and "empirical-synthesis" not in reference.doi:
        source_text += f" DOI: {reference.doi}."
    story.append(paragraph(f"Database evidence ({reference.evidence_level}): {source_text}"))
    story.append(paragraph(
        "Standards reference: ISO 3685:1993, Tool-life testing with single-point turning tools. "
        "Used as a tool-life and flank-wear test-method reference; it is not the source of the "
        "pairing-specific Taylor constants."
    ))
    story.append(paragraph(
        "Standards reference: ASME B94.55M-1985, Tool Life Testing with Single-Point Turning Tools. "
        "Consult the applicable published edition and verify the standard's scope for the operation."
    ))
    story.extend([
        Spacer(1, 5),
        paragraph(
            "Engineering note: model outputs are estimates based on the selected database record and "
            "physics model. Confirm the cited publication/handbook and validate recommendations with "
            "machine-specific trials before production use.",
            "ReportSmall",
        ),
    ])
    document.build(story)
    return output.getvalue()


def _init(key, **defaults):
    """Pass widget defaults only on the first render, so preset-driven session
    state never collides with a declared default (avoids Streamlit warnings)."""
    return {} if key in st.session_state else defaults


def _sync_dependent_widgets():
    """After a machine change, keep operation/holder selections that are still
    valid for the new family and re-point invalid ones to that family's default.
    Also clamps the cutting-parameter sliders into the new machine's envelope."""
    name = st.session_state.get("machine_choice", NO_SELECTION)
    mkey = NEUTRAL_MACHINE_KEY if name == NO_SELECTION else name
    machine = MACHINE_DATABASE[mkey]
    fam = machine.family
    # Streamlit resets a keyed slider whenever min/max/step change (documented
    # behavior), so a machine switch would zero the cutting parameters. Stash
    # the current values clamped into the new envelope and drop the keys; the
    # cutting-parameter section re-seeds the sliders from the stash on the
    # very next render, preserving the user's values across the switch.
    _stash = {}
    for _k, _cap in (
        ("vc_input", machine.max_vc_m_per_min),
        ("feed_input", machine.max_feed_mm),
        ("ap_input", machine.max_ap_mm),
    ):
        if _k in st.session_state:
            _stash[_k] = min(float(st.session_state[_k]), _cap)
            del st.session_state[_k]
    if _stash:
        st.session_state["_param_stash"] = _stash
    if fam == "milling":
        allowed = {"milling", "drilling", "boring"}
    elif fam == "turning":
        allowed = {"turning", "drilling"}
    else:
        allowed = {"milling", "turning", "drilling", "boring"}
    op_list = [k for k, v in OPERATION_DATABASE.items() if v.family in allowed]
    op_default = "End Milling / Shoulder Milling"
    if st.session_state.get("operation_choice") not in ([NO_SELECTION] + op_list):
        st.session_state["operation_choice"] = op_default if op_default in op_list else op_list[0]
    if fam == "milling":
        holder_keys = [k for k, h in TOOL_HOLDER_DATABASE.items() if h.applicable_family in ("milling", "both")]
        holder_default = "Shrink-Fit Holder (HSK, Balanced G2.5)"
    elif fam == "turning":
        holder_keys = [k for k, h in TOOL_HOLDER_DATABASE.items() if h.applicable_family in ("turning", "both")]
        holder_default = "Standard ISO Turning Tool Holder (Rigid Clamp)"
    else:
        holder_keys = list(TOOL_HOLDER_DATABASE.keys())
        holder_default = "Standard ER Collet Chuck"
    if st.session_state.get("holder_choice") not in ([NO_SELECTION] + holder_keys):
        st.session_state["holder_choice"] = holder_default


def _apply_industry_preset():
    """Push preset values into the keyed sidebar widgets; runs before the rerun."""
    name = st.session_state.get("preset_choice", CUSTOM_PARAMS)
    if name == CUSTOM_PARAMS:
        return
    p = INDUSTRY_PRESETS[name]
    st.session_state["material_mode"] = "📖 Curated Springer IJAMT Research Pairings"
    st.session_state["pairing_choice"] = p.pairing_key
    st.session_state["machine_choice"] = p.machine_name
    st.session_state["coolant_choice"] = p.coolant_name
    if p.operation_name:
        st.session_state["operation_choice"] = p.operation_name
    if p.milling_tooling_name:
        st.session_state["tooling_choice"] = p.milling_tooling_name
    pairing = SPRINGER_DATABASE.get(p.pairing_key)
    if pairing is not None:
        st.session_state["vc_input"] = min(max(float(p.vc), pairing.v_min * 0.5), pairing.v_max * 1.7)
        st.session_state["feed_input"] = min(max(float(p.feed), pairing.f_min * 0.6), pairing.f_max * 1.6)
        st.session_state["ap_input"] = min(max(float(p.ap), pairing.ap_min * 0.6), pairing.ap_max * 1.6)
    else:
        st.session_state["vc_input"] = float(p.vc)
        st.session_state["feed_input"] = float(p.feed)
        st.session_state["ap_input"] = float(p.ap)
    st.session_state["time_input"] = float(p.current_time_min)
    st.session_state["roughing_flag"] = bool(p.is_roughing)
    _sync_dependent_widgets()


# Sidebar: Presets and User Inputs
with st.sidebar:
    st.markdown("## Machining Setup")
    st.caption("Runs locally; no internet is needed for dashboard calculations.")

    # 1-Click Industry Presets
    st.markdown("### ⚡ Quick-Load Presets")
    preset_keys = [CUSTOM_PARAMS] + list(INDUSTRY_PRESETS.keys())
    selected_preset = st.selectbox(
        "Load Industry Case Study:",
        preset_keys + [RESET_OPTION],
        key="preset_choice",
        on_change=_on_preset_change,
    )

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
        default_operation = p_data.operation_name
        default_milling_tooling = p_data.milling_tooling_name
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
        default_operation = "End Milling / Shoulder Milling"
        default_milling_tooling = None

    st.markdown("---")
    st.markdown("### Tool & material")

    # Mode toggle: Independent vs Curated Pairings
    material_mode = st.radio(
        "Material Selection Mode:",
        [
            "🔧 Independent Custom Selection (Tool & Workpiece Separately)",
            "📖 Curated Springer IJAMT Research Pairings"
        ],
        key="material_mode",
        help="Choose whether to select workpiece and tool independently or use curated research literature pairings.",
        **_init("material_mode", index=0 if selected_preset == CUSTOM_PARAMS else 1),
    )

    custom_warnings = []

    if "Independent" in material_mode:
        st.markdown("#### Workpiece Material")
        wp_keys = [NO_SELECTION] + list(WORKPIECE_DATABASE.keys())
        selected_wp_key = st.selectbox(
            "Select Workpiece Material:",
            wp_keys + [RESET_OPTION],
            index=1,
            key="wp_choice",
            on_change=lambda: _reset_dropdown("wp_choice", wp_keys[1]),
        )
        if selected_wp_key == NO_SELECTION:
            st.warning("⚠️ No selection has been made for the workpiece material — please select a material to generate predictions.")
            st.stop()
        wp_info = WORKPIECE_DATABASE[selected_wp_key]

        st.markdown(
            f"<span class='springer-badge'>ISO Group {wp_info.iso_group}</span> "
            f"<span class='springer-badge'>{wp_info.hardness}</span>",
            unsafe_allow_html=True
        )
        st.caption(f"🔥 **Thermal Conduct.** {wp_info.thermal_conductivity}")

        st.markdown("#### Tool Material & Coating")
        tool_keys = [NO_SELECTION] + list(TOOL_DATABASE.keys())
        selected_tool_key = st.selectbox(
            "Select Tool Material (Substrate):",
            tool_keys + [RESET_OPTION],
            index=1,
            key="tool_choice",
            on_change=lambda: _reset_dropdown("tool_choice", tool_keys[1]),
        )
        if selected_tool_key == NO_SELECTION:
            st.warning("⚠️ No selection has been made for the tool material — please select a substrate to generate predictions.")
            st.stop()
        tool_info = TOOL_DATABASE[selected_tool_key]

        st.markdown(
            f"<span class='springer-badge'>{tool_info.category}</span> "
            f"<span class='springer-badge'>Taylor n: {tool_info.taylor_n:.3f}</span> "
            f"<span class='springer-badge'>Max Temp: {tool_info.max_temp_c:.0f}°C</span>",
            unsafe_allow_html=True
        )

        coating_keys = [NO_SELECTION] + list(COATING_DATABASE.keys())
        selected_coating_key = st.selectbox(
            "Select Tool Coating / Prep:",
            coating_keys + [RESET_OPTION],
            index=1,
            key="coating_choice",
            on_change=lambda: _reset_dropdown("coating_choice", coating_keys[1]),
        )
        if selected_coating_key == NO_SELECTION:
            st.warning("⚠️ No selection has been made for the tool coating — please select a coating/preparation to generate predictions.")
            st.stop()
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
        # Keys include the material identity so switching workpiece/tool/coating
        # re-seeds the inputs from the newly synthesized pairing instead of
        # keeping stale overrides.
        _taylor_key = f"{selected_wp_key} | {selected_tool_key} | {selected_coating_key}"
        with st.expander("⚙️ Fine-Tune Taylor Constants (Expert Overrides)", expanded=False):
            override_c = st.number_input(
                "Taylor Constant (C):",
                min_value=1.0,
                max_value=25000.0,
                step=5.0,
                key=f"taylor_c_{_taylor_key}",
                **_init(f"taylor_c_{_taylor_key}", value=float(pairing_info.taylor_C)),
            )
            override_n = st.number_input(
                "Taylor Exponent (n):",
                min_value=0.05,
                max_value=0.90,
                step=0.01,
                format="%.3f",
                key=f"taylor_n_{_taylor_key}",
                **_init(f"taylor_n_{_taylor_key}", value=float(pairing_info.taylor_n)),
            )
            pairing_info.taylor_C = override_c
            pairing_info.taylor_n = override_n

    else:
        pairing_list = list(SPRINGER_DATABASE.keys())
        pairing_display = [NO_SELECTION] + pairing_list
        pairing_index = (pairing_list.index(default_pairing) + 1) if default_pairing in pairing_list else 1
        pairing_choice = st.selectbox(
            "Curated Workpiece & Tool Combination:",
            pairing_display + [RESET_OPTION],
            key="pairing_choice",
            on_change=lambda: _reset_dropdown("pairing_choice", default_pairing if default_pairing in pairing_list else pairing_list[0]),
            help="Calibrated empirical pairings from peer-reviewed Springer IJAMT research studies.",
            **_init("pairing_choice", index=pairing_index),
        )
        if pairing_choice == NO_SELECTION:
            st.warning("⚠️ No selection has been made for the material pairing — please choose a curated pairing to generate predictions.")
            st.stop()
        selected_pairing = pairing_choice
        pairing_info = SPRINGER_DATABASE[selected_pairing]

        # Display material badges
        st.markdown(
            f"<span class='springer-badge'>ISO Group {pairing_info.workpiece_iso}</span> "
            f"<span class='springer-badge'>{pairing_info.tool_material}</span>",
            unsafe_allow_html=True
        )
        st.caption(f"**Coating**: {pairing_info.coating}")

    st.markdown("---")
    st.markdown("### 2. Machine Tool & Environment")

    machine_list = list(MACHINE_DATABASE.keys())
    machine_display = [NO_SELECTION] + machine_list
    machine_index = (machine_list.index(default_machine) + 1) if default_machine in machine_list else 1
    machine_choice = st.selectbox(
        "Which Machine is Used for Machining?",
        machine_display + [RESET_OPTION],
        key="machine_choice",
        on_change=lambda: _on_machine_change(default_machine if default_machine in machine_list else machine_list[0]),
        help="Machine rigidity directly scales dynamic chatter vibration and tool degradation rate. The machine family (milling / turning / universal) determines which operations, cutters and holders are available.",
        **_init("machine_choice", index=machine_index),
    )
    if machine_choice == NO_SELECTION:
        selected_machine = NEUTRAL_MACHINE_KEY
        st.caption("⚠️ No machine selected — neutral 1.0x rigidity applied.")
    else:
        selected_machine = machine_choice
    machine_info = MACHINE_DATABASE[selected_machine]
    m_family = machine_info.family

    # Consume the parameter stash written by _sync_dependent_widgets on a
    # machine switch (slider keys were deleted there and are re-seeded below).
    _param_stash = st.session_state.pop("_param_stash", None)

    # Operations are filtered by the selected machine's family
    if m_family == "milling":
        allowed_op_families = {"milling", "drilling", "boring"}
    elif m_family == "turning":
        allowed_op_families = {"turning", "drilling"}
    else:  # universal
        allowed_op_families = {"milling", "turning", "drilling", "boring"}
    operation_list = [k for k, v in OPERATION_DATABASE.items() if v.family in allowed_op_families]
    operation_display = [NO_SELECTION] + operation_list
    operation_index = (operation_list.index(default_operation) + 1) if default_operation in operation_list else 1
    operation_choice = st.selectbox(
        "Machining Operation:",
        operation_display + [RESET_OPTION],
        key="operation_choice",
        on_change=lambda: _reset_dropdown(
            "operation_choice",
            default_operation if default_operation in operation_list else operation_list[0],
        ),
        help="Operation type scales tool life (chip thinning, interrupted cuts, thermal cycling) and switches the MRR model.",
        **_init("operation_choice", index=operation_index),
    )
    if operation_choice == NO_SELECTION:
        selected_operation = None
        op_info = SimpleNamespace(
            feed_unit="mm/rev", family="none", life_multiplier=1.0,
            description="No operation selected — neutral 1.0x life factor, turning-model MRR."
        )
        st.caption("⚠️ No operation selected — neutral physics applied (1.0x life, turning-model MRR).")
    else:
        selected_operation = operation_choice
        op_info = OPERATION_DATABASE[selected_operation]
        st.caption(f"🏭 **{op_info.family.capitalize()}** | Life Factor: {op_info.life_multiplier:.2f}x | {op_info.description}")

    # Milling cutter is only accessible on milling machines running milling operations
    selected_milling_tooling = None
    if m_family == "milling" and op_info.family == "milling":
        tooling_list = list(MILLING_TOOLING_DATABASE.keys())
        tooling_display = [NO_SELECTION] + tooling_list
        tooling_index = (tooling_list.index(default_milling_tooling) + 1) if default_milling_tooling in tooling_list else 1
        tooling_choice = st.selectbox(
            "Milling Cutter / Tooling:",
            tooling_display + [RESET_OPTION],
            key="tooling_choice",
            on_change=lambda: _reset_dropdown(
                "tooling_choice",
                default_milling_tooling if default_milling_tooling in tooling_list else tooling_list[0],
            ),
            help="Cutter geometry (teeth, diameter, edge prep) scales tool life and drives the MRR spindle-speed model.",
            **_init("tooling_choice", index=tooling_index),
        )
        if tooling_choice == NO_SELECTION:
            st.caption("⚠️ No cutter selected — neutral 1.0x tooling factor, default 4-flute Ø12 mm MRR model.")
        else:
            selected_milling_tooling = tooling_choice
            mt_info = MILLING_TOOLING_DATABASE[selected_milling_tooling]
            st.markdown(
                f"<span class='springer-badge'>{mt_info.family.replace('_', ' ').title()}</span> "
                f"<span class='springer-badge'>{mt_info.teeth} Flutes</span> "
                f"<span class='springer-badge'>Ø {mt_info.diameter_mm:.0f} mm</span> "
                f"<span class='springer-badge'>Life: {mt_info.life_multiplier:.2f}x</span>",
                unsafe_allow_html=True
            )
            st.caption(f"💡 **Best practice**: {mt_info.best_practices}")

    # Tool holder filtered by machine family
    if m_family == "milling":
        holder_keys = [k for k, h in TOOL_HOLDER_DATABASE.items() if h.applicable_family in ("milling", "both")]
        default_holder = "Shrink-Fit Holder (HSK, Balanced G2.5)"
    elif m_family == "turning":
        holder_keys = [k for k, h in TOOL_HOLDER_DATABASE.items() if h.applicable_family in ("turning", "both")]
        default_holder = "Standard ISO Turning Tool Holder (Rigid Clamp)"
    else:
        holder_keys = list(TOOL_HOLDER_DATABASE.keys())
        default_holder = "Standard ER Collet Chuck"
    holder_display = [NO_SELECTION] + holder_keys
    holder_index = (holder_keys.index(default_holder) + 1) if default_holder in holder_keys else 1
    holder_choice = st.selectbox(
        "Tool Holder / Chucking System:",
        holder_display + [RESET_OPTION],
        key="holder_choice",
        on_change=lambda: _reset_dropdown(
            "holder_choice",
            default_holder if default_holder in holder_keys else holder_keys[0],
        ),
        help="Holder rigidity and runout accuracy scale tool life. Dampened anti-vibration holders halve the overhang penalty.",
        **_init("holder_choice", index=holder_index),
    )
    if holder_choice == NO_SELECTION:
        selected_holder = None
        st.caption("⚠️ No holder selected — neutral 1.0x holder rigidity.")
    else:
        selected_holder = holder_choice
        holder_info = TOOL_HOLDER_DATABASE[selected_holder]
        st.caption(f"🔩 **Runout**: {holder_info.runout_accuracy} | Rigidity Multiplier: {holder_info.rigidity_multiplier:.2f}x")

    overhang_input = st.slider(
        "Tool Overhang Ratio (L/D):",
        min_value=0.5,
        max_value=8.0,
        value=3.0,
        step=0.1,
        key="overhang_ld",
        help="Exposed tool length divided by tool diameter. Beyond L/D = 3.0, chatter and edge chipping accelerate — tool life is derated. Dampened holders halve the derate rate."
    )
    if overhang_input > 3.0:
        st.caption(f"⚠️ L/D = {overhang_input:.1f} exceeds 3×D — overhang derate active.")

    coolant_list = list(COOLANT_DATABASE.keys())
    coolant_display = [NO_SELECTION] + coolant_list
    coolant_index = (coolant_list.index(default_coolant) + 1) if default_coolant in coolant_list else 1
    coolant_choice = st.selectbox(
        "Cooling / Lubrication Method:",
        coolant_display + [RESET_OPTION],
        key="coolant_choice",
        on_change=lambda: _reset_dropdown(
            "coolant_choice",
            default_coolant if default_coolant in coolant_list else coolant_list[0],
        ),
        help="Select 'No Coolant (Bare Dry Cut)' to simulate dry machining — tool life is derated and thermal-risk warnings appear automatically.",
        **_init("coolant_choice", index=coolant_index),
    )
    if coolant_choice == NO_SELECTION:
        selected_coolant = NEUTRAL_COOLANT_KEY
        st.caption("⚠️ No coolant selection — neutral 1.0x multiplier (distinct from 'No Coolant' dry cutting).")
    else:
        selected_coolant = coolant_choice
    if "No Coolant" in selected_coolant:
        st.warning("⚠️ **Bare Dry Cut selected**: tool life reduced ~38%. Suitable for cast iron, ceramic tooling, or light finishing passes only.")

        # Live dry-cut impact assessment at the current cutting parameters
        # (sliders/flags live in session state; read directly so this section
        # re-assesses on every dashboard change even though it renders earlier).
        def _live_param(key, fallback):
            if _param_stash and key in _param_stash:
                return float(_param_stash[key])
            return float(st.session_state.get(key, fallback))

        _vc_now = _live_param("vc_input", default_vc)
        _feed_now = _live_param("feed_input", default_feed)
        _ap_now = _live_param("ap_input", default_ap)
        _rough_now = bool(st.session_state.get("roughing_flag", default_roughing))
        _dry_cool = COOLANT_DATABASE[selected_coolant]
        _flood_cool = COOLANT_DATABASE["Standard Flood Emulsion (7-10% oil)"]
        _common = dict(
            is_roughing=_rough_now,
            operation_name=selected_operation,
            milling_tooling_name=selected_milling_tooling,
            holder_name=selected_holder,
            overhang_ratio=overhang_input,
        )
        _dry_life = calculate_tool_life(
            pairing_info, machine_info, _dry_cool, _vc_now, _feed_now, _ap_now, **_common
        )
        _flood_life = calculate_tool_life(
            pairing_info, machine_info, _flood_cool, _vc_now, _feed_now, _ap_now, **_common
        )
        _dry_eff = effective_coolant_factor(pairing_info, _dry_cool)
        _flood_eff = effective_coolant_factor(pairing_info, _flood_cool)

        if _dry_life >= _flood_life:
            st.success(
                f"✅ **Dry cutting is optimal for {pairing_info.tool_material}**: "
                f"{_dry_life:.1f} min dry vs {_flood_life:.1f} min with flood "
                f"(+{((_dry_life - _flood_life) / max(_flood_life, 1e-9)) * 100:.0f}%). "
                "Flood coolant would thermally shock this tool material — keep it dry."
            )
        else:
            _loss = _flood_life - _dry_life
            _pct = (_loss / _flood_life) * 100.0
            st.error(
                f"📉 **Real tool-life impact at current parameters**: "
                f"**{_dry_life:.1f} min dry** vs **{_flood_life:.1f} min with flood** "
                f"→ **−{_loss:.1f} min (−{_pct:.0f}%)**"
            )

            # Alternative-coolant recovery table at the exact same parameters
            _rows = []
            for _cname, _cinfo in COOLANT_DATABASE.items():
                if "No Selection" in _cname or "No Coolant" in _cname:
                    continue
                _life = calculate_tool_life(
                    pairing_info, machine_info, _cinfo, _vc_now, _feed_now, _ap_now, **_common
                )
                _rows.append((_cinfo.name, _life, effective_coolant_factor(pairing_info, _cinfo)))
            _rows.sort(key=lambda r: r[1], reverse=True)
            _tbl = "\n".join(
                f"| {'**' if n == _rows[0][0] else ''}{n}{'**' if n == _rows[0][0] else ''} "
                f"| {l:.1f} min | {((l - _dry_life) / max(_dry_life, 1e-9)) * 100:+.0f}% |"
                for n, l, _f in _rows
            )
            st.markdown(
                "**Recovery — same cut under each coolant:**\n\n"
                "| Coolant | Tool Life | vs Dry |\n|---|---|---|\n" + _tbl
            )

            # Dry-optimized parameter suggestions
            _n = float(pairing_info.taylor_n)
            _vc_comp = _vc_now * (_dry_eff / _flood_eff) ** _n
            _life_comp = calculate_tool_life(
                pairing_info, machine_info, _dry_cool, _vc_comp, _feed_now, _ap_now, **_common
            )
            _best_name, _best_life, _best_eff = _rows[0]
            st.markdown(
                "**Dry-cut optimization options:**\n"
                f"1. 🎛️ **Speed compensation** — drop Vc to **{_vc_comp:.0f} m/min** "
                f"({_dry_eff / _flood_eff:+.0%} factor → matches flood life ≈ {_life_comp:.1f} min, MRR −{100 * (1 - _vc_comp / _vc_now):.0f}%).\n"
                f"2. 💧 **Best coolant switch** — {_best_name} restores ≈ **{_best_life:.1f} min** "
                f"({_best_eff / _dry_eff - 1:+.0%} vs dry) with zero parameter change.\n"
                f"3. 🧱 **Ceramic tooling** — Si₃N₄/Al₂O₃ thrives dry (+15% vs dry carbide baseline, "
                "flood would crack it); see Tool Material selector.\n"
                f"4. 📐 **Lighter engagement** — reduce ap/feed and use the wear-minimization "
                "strategies in the '🎯 How to Minimize Tool Wear' tab."
            )

    is_roughing = st.checkbox(
        "Heavy Roughing Cut (VB limit = 0.5-0.6 mm)",
        key="roughing_flag",
        help="ISO 3685 defines VB=0.3mm for precision finishing, or 0.5-0.6mm for roughing.",
        **_init("roughing_flag", value=default_roughing),
    )

    st.markdown("---")
    st.markdown("### 3. Cutting Parameters")

    # Slider ceilings follow the selected machine's capability envelope
    # (min is 0 for all three; physics engine floors tiny inputs internally).
    v_rec_min = float(pairing_info.v_min)
    v_rec_max = float(pairing_info.v_max)
    vc_cap = float(machine_info.max_vc_m_per_min)
    feed_cap = float(machine_info.max_feed_mm)
    ap_cap = float(machine_info.max_ap_mm)

    # Seed from the machine-switch stash when present (values already clamped
    # to this machine's envelope by _sync_dependent_widgets); otherwise use
    # the pairing/preset-driven defaults.
    _seed_vc = float(_param_stash["vc_input"]) if (_param_stash and "vc_input" in _param_stash) else float(min(max(default_vc, v_rec_min * 0.5), vc_cap))
    _seed_feed = float(_param_stash["feed_input"]) if (_param_stash and "feed_input" in _param_stash) else float(min(max(default_feed, pairing_info.f_min * 0.6), feed_cap))
    _seed_ap = float(_param_stash["ap_input"]) if (_param_stash and "ap_input" in _param_stash) else float(min(max(default_ap, pairing_info.ap_min * 0.6), ap_cap))

    vc_input = st.slider(
        "Cutting Speed (Vc, m/min):",
        min_value=0.0,
        max_value=vc_cap,
        step=1.0,
        key="vc_input",
        help=f"Machine envelope: 0 – {vc_cap:.0f} m/min (this machine) | Springer empirical range: {v_rec_min:.0f} – {v_rec_max:.0f} m/min",
        **_init("vc_input", value=_seed_vc),
    )

    feed_input = st.slider(
        f"Feed Rate (f, {op_info.feed_unit}):",
        min_value=0.0,
        max_value=feed_cap,
        step=0.01,
        format="%.3f",
        key="feed_input",
        help=f"Machine envelope: 0 – {feed_cap:.3f} {op_info.feed_unit} (this machine) | Springer empirical range: {pairing_info.f_min:.3f} – {pairing_info.f_max:.3f} {op_info.feed_unit}",
        **_init("feed_input", value=_seed_feed),
    )

    ap_input = st.slider(
        "Depth of Cut (ap, mm):",
        min_value=0.0,
        max_value=ap_cap,
        step=0.1,
        key="ap_input",
        help=f"Machine envelope: 0 – {ap_cap:.1f} mm (this machine) | Springer empirical range: {pairing_info.ap_min:.1f} – {pairing_info.ap_max:.1f} mm",
        **_init("ap_input", value=_seed_ap),
    )

    st.markdown("---")
    st.markdown("### 4. Tool In-Service Monitor")
    current_time_input = st.number_input(
        "Current Elapsed Cutting Time (minutes):",
        min_value=0.0,
        max_value=500.0,
        step=1.0,
        key="time_input",
        help="Enter current machining time on this tool edge to estimate Remaining Useful Life (RUL).",
        **_init("time_input", value=float(default_time)),
    )
    st.caption(f"Running app file: `{Path(__file__).resolve()}`")


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
    operation_name=selected_operation,
    milling_tooling_name=selected_milling_tooling,
    holder_name=selected_holder,
    overhang_ratio=overhang_input,
)

opt_report = optimize_tool_wear(
    pairing_key=selected_pairing,
    machine_name=selected_machine,
    coolant_name=selected_coolant,
    current_vc=vc_input,
    current_feed=feed_input,
    current_ap=ap_input,
    is_roughing=is_roughing,
    custom_pairing=pairing_info,
    operation_name=selected_operation,
    milling_tooling_name=selected_milling_tooling,
    holder_name=selected_holder,
    overhang_ratio=overhang_input,
)


# Main Content Area Header
st.markdown("<div class='main-header'>⚙️ Machining Tool Wear & Life Intelligence Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='sub-header'>Forecasting flank wear ($VB$), Remaining Useful Life (RUL), and physics-driven wear minimization grounded in peer-reviewed <b>Springer IJAMT</b> experimental research.</div>",
    unsafe_allow_html=True
)


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
            <div class='kpi-title'>Springer Confidence</div>
            <div class='kpi-value' style='color:{conf_color};'>{pred.confidence_score:.0f}%</div>
            <div class='kpi-subtext' style='color:{conf_color};'>{pred.confidence_label.split('(')[0]}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)


# Tab Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tool Life & Wear Progression (Graphical)",
    "🎯 How to Minimize Tool Wear (Optimization)",
    "📚 Springer Research Reference & Evidence",
    "📊 Scenario Comparison & Data Export"
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
        st.caption(f"Average flank wear rate (µm/min) across the cutting speed range. Higher wear rate = faster tool degradation. Springer verified range shaded green.")

        v_sweep = np.linspace(pred.pairing.v_min * 0.7, pred.pairing.v_max * 1.4, 60)
        t_sweep = [
            calculate_tool_life(
                pred.pairing, pred.machine, pred.coolant, v, pred.feed, pred.ap, pred.is_roughing,
                operation_name=pred.operation_name, milling_tooling_name=pred.milling_tooling_name,
                holder_name=pred.holder_name, overhang_ratio=pred.overhang_ratio
            )
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
            annotation_text="Springer Verified Range",
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
                    V_mesh[i, j], F_mesh[i, j], pred.ap, pred.is_roughing,
                    operation_name=pred.operation_name, milling_tooling_name=pred.milling_tooling_name,
                    holder_name=pred.holder_name, overhang_ratio=pred.overhang_ratio
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
    st.subheader("📚 Springer Research & Handbook Reference")
    ref = pred.pairing.springer_ref
    if "Peer-reviewed" in ref.evidence_level:
        range_label = "Springer Tested Range"
    elif "handbook" in ref.evidence_level.lower():
        range_label = "Handbook Starting Range"
    else:
        range_label = "Database Model Range"
    is_synthesized_reference = "empirical-synthesis" in ref.doi or "synthesized" in ref.title.lower()
    st.caption(ref.evidence_level)

    if is_synthesized_reference:
        st.markdown("### 📖 Composite model evidence")
        st.info(
            "This independent material/tool/coating combination is synthesized by the dashboard; "
            "it is not a single Springer experiment. Its estimated parameters are derived from "
            "the selected material/tool records and the physics model, so no study-specific DOI "
            "is asserted for this combination."
        )
        st.markdown(
            f"**Model record**: {ref.title}  \n"
            f"**Evidence classification**: {ref.evidence_level}"
        )
    else:
        citation = (
            f"### 📖 {ref.title}\n"
            f"**Authors**: {ref.authors}  \n"
            f"**Journal / publisher**: *{ref.journal}* | {ref.volume_issue} ({ref.year})"
        )
        if _valid_doi(ref.doi):
            citation += f"  \n**DOI**: [{ref.doi}](https://doi.org/{ref.doi})"
        else:
            citation += "  \n**DOI**: No valid DOI recorded for this database entry."
        st.markdown(citation)

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
    st.subheader("Standards and handbook methodology")
    st.markdown(
        "- **ISO 3685:1993**, *Tool-life testing with single-point turning tools* — "
        "reference for tool-life testing and flank-wear criteria in its stated scope.\n"
        "- **ASME B94.55M-1985**, *Tool Life Testing with Single-Point Turning Tools* — "
        "related tool-life test-method reference.\n"
        "- **Springer handbook starting guidance**, when identified as the selected evidence "
        "level above — treat its parameter window as a starting point and validate it on the machine."
    )
    st.warning(
        "The standards describe test methods and criteria; they do not supply this pairing's "
        "Taylor constants. Those are taken from the selected database record or synthesized "
        "model. Check the published standard/handbook edition and applicability before using "
        "these estimates for production decisions."
    )

    st.markdown("---")
    st.subheader("Experimental Validity Envelope & Confidence Audit")
    
    st.markdown(f"**Overall Prediction Confidence**: `{pred.confidence_score:.0f}%` ({pred.confidence_label})")
    
    # Boundary validation table
    boundary_df = pd.DataFrame([
        {
            "Parameter": "Cutting Speed (Vc)",
            "User Value": f"{pred.vc:.1f} m/min",
            range_label: f"{pred.pairing.v_min:.1f} – {pred.pairing.v_max:.1f} m/min",
            "In Boundary": "✅ YES" if (pred.pairing.v_min <= pred.vc <= pred.pairing.v_max) else "⚠️ EXTRAPOLATED"
        },
        {
            "Parameter": "Feed Rate (f)",
            "User Value": f"{pred.feed:.3f} mm/rev",
            range_label: f"{pred.pairing.f_min:.3f} – {pred.pairing.f_max:.3f} mm/rev",
            "In Boundary": "✅ YES" if (pred.pairing.f_min <= pred.feed <= pred.pairing.f_max) else "⚠️ EXTRAPOLATED"
        },
        {
            "Parameter": "Depth of Cut (ap)",
            "User Value": f"{pred.ap:.2f} mm",
            range_label: f"{pred.pairing.ap_min:.2f} – {pred.pairing.ap_max:.2f} mm",
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
    pdf_data = _build_pdf_report(summary_df, pred)
    st.download_button(
        label="📄 Download Portrait PDF Report",
        data=pdf_data,
        file_name=f"tool_wear_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
    )

    st.markdown("---")
    st.markdown("### 📋 Complete Machining Specification Summary")
    st.json({
        "workpiece_material": pred.pairing.workpiece_name,
        "workpiece_iso_group": pred.pairing.workpiece_iso,
        "tool_material": pred.pairing.tool_material,
        "tool_coating": pred.pairing.coating,
        "machine_tool_selected": pred.machine_name,
        "machine_rigidity_multiplier": pred.machine.rigidity_factor,
        "coolant_applied": pred.coolant_name,
        "machining_operation": pred.operation_name,
        "milling_cutter": pred.milling_tooling_name,
        "tool_holder": pred.holder_name,
        "tool_overhang_ratio_ld": pred.overhang_ratio,
        "cutting_speed_vc_m_min": pred.vc,
        "feed_rate_" + op_info.feed_unit.replace("/", "_").replace(" ", "_"): pred.feed,
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
