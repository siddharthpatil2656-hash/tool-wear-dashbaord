"""
Machining Intelligence & Tool Wear Forecasting Dashboard.
Includes literature-referenced default model records and optional measured-trial calibration.
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
from dataclasses import replace
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
    calculate_tool_life_multiplier,
    calculate_mrr,
    effective_coolant_factor,
)
from optimizer import optimize_tool_wear
from presets import INDUSTRY_PRESETS
from calibration import (
    calibration_template,
    fit_taylor_model,
    matching_calibration_trials,
)
from crossref_evidence import search_research_articles
from springer_library import (
    load_reference_library,
    merge_references,
    parse_reference_export,
    relevant_references,
    save_reference_library,
)


# "-- No Selection --" means that field was left empty: modifiers fall back to
# neutral physics (1.0x / None), while mandatory material inputs prompt the user.
NO_SELECTION = "-- No Selection --"
NEUTRAL_MACHINE_KEY = "Unspecified Machine (Neutral Rigidity 1.0x)"
NEUTRAL_COOLANT_KEY = "No Selection (Neutral 1.0x)"
RESET_OPTION = "Reset to default"
KNOWN_DOI_MISMATCHES = {
    "10.1007/s00170-022-09415-z": "The bundled DOI was not found in Crossref and does not verify the listed Springer citation; the paper metadata is not treated as verified.",
    "10.1115/1.4035567": "DOI resolves to 'A Simple Model for Identifying the Flutter Bite of Fan Blades', not the turning study named in the local record.",
    "10.1115/1.4038989": "DOI resolves to 'Numerical Simulation of Forced Convective Boiling in a Microchannel', not the turning study named in the local record.",
}

EVIDENCE_CATALOGS = (
    (
        "Springer Nature Link",
        "Publisher catalog for locating peer-reviewed papers; this app does not download article full text or experimental datasets.",
        "https://link.springer.com/journal/170",
    ),
    (
        "ASME B94.55M",
        "ASME catalog entry for tool-life testing with single-point turning tools; a test-method standard, not a prediction dataset.",
        "https://www.asme.org/codes-standards/find-codes-standards/b94-55m-tool-life-testing-single-point-turning-tools",
    ),
    (
        "ISO 3685:1993",
        "Tool-life testing with single-point turning tools; use within the standard's stated scope, not as Taylor-constant data.",
        "https://www.iso.org/obp/ui/#iso:std:iso:3685:ed-2:v1:en",
    ),
    (
        "ASTM International",
        "Official standards catalog for checking relevant material and test methods; no ASTM tool-life dataset is bundled or used to fit this model.",
        "https://www.astm.org/",
    ),
    (
        "Sandvik Coromant machining formulas",
        "Public cutting-data formulas, terminology, and definitions. Manufacturer guidance; verify the selected tool's current catalog limits.",
        "https://www.sandvik.coromant.com/en-gb/knowledge/machining-formulas-definitions",
    ),
    (
        "Sandvik Coromant CoroPlus ToolGuide",
        "Manufacturer cutting-data/tool selection guidance; not imported into the dashboard's Taylor coefficients.",
        "https://www.sandvik.coromant.com/en-gb/software/coroplus-tool-guide",
    ),
    (
        "PHM Society 2010 CNC cutter wear dataset",
        "Public sensor/wear records for a 6 mm carbide cutter (forces, vibration, acoustic emission); a research dataset with a specific setup, not a universal cutting-data handbook.",
        "https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/",
    ),
    (
        "Handbook of Advanced Ceramics Machining",
        "CRC Press book record (DOI 10.1201/9781420005547); relevant to ceramic machining, not a general Taylor-constant dataset. Full text may require access.",
        "https://doi.org/10.1201/9781420005547",
    ),
)


# Page configuration
st.set_page_config(
    page_title="Tool Wear Forecasting & Life Optimization",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme-aware card and HTML table styling. Streamlit supplies these CSS tokens
# from the active light or dark theme.
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: var(--text-color);
        opacity: 0.78;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, var(--background-color) 0%, var(--secondary-background-color) 100%);
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.25));
        border-radius: 10px;
        padding: 1.1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-color);
        opacity: 0.72;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--text-color);
    }
    .kpi-subtext {
        font-size: 0.82rem;
        color: var(--primary-color);
        margin-top: 0.2rem;
    }
    .springer-badge {
        background-color: var(--secondary-background-color);
        color: var(--primary-color);
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.25));
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
        background-color: var(--background-color);
        border-radius: 8px;
        overflow: hidden;
    }
    th {
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        font-weight: 600;
        padding: 10px 14px;
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.25));
        text-align: left;
    }
    td {
        padding: 10px 14px;
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.25));
        color: var(--text-color);
    }
    tr:nth-child(even) {
        background-color: var(--secondary-background-color);
    }
    [data-testid="stSidebar"] { border-right: 1px solid var(--border-color, rgba(128, 128, 128, 0.25)); }
    [data-testid="stSidebar"] .stMarkdown h2 { font-size: 1.15rem; }
</style>
""", unsafe_allow_html=True)


DARK_THEME = st.context.theme.type == "dark"
CHART_COLORS = {
    "primary": "#67B7C7" if DARK_THEME else "#287E8C",
    "danger": "#EF8A8A" if DARK_THEME else "#C65D63",
    "warning": "#E7B866" if DARK_THEME else "#B7791F",
    "purple": "#B4A7E8" if DARK_THEME else "#7667A8",
    "green": "#76B79A" if DARK_THEME else "#4C8B71",
    "blue": "#7FB3D5" if DARK_THEME else "#4C83A6",
    "muted": "#8793A3" if DARK_THEME else "#64748B",
}


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


def _polish_chart(fig, height=420, hovermode="closest"):
    """Apply consistent readable Plotly styling while preserving chart-specific axes."""
    if DARK_THEME:
        colors = {
            "template": "plotly_dark",
            "paper": "#171D27",
            "plot": "#141A23",
            "text": "#E5EAF1",
            "muted_text": "#AAB5C4",
            "grid": "#2C3543",
            "axis": "#465365",
            "hover": "#222B38",
            "hover_border": "#465365",
            "legend": "rgba(23,29,39,0.94)",
        }
    else:
        colors = {
            "template": "plotly_white",
            "paper": "#FBFCFE",
            "plot": "#FFFFFF",
            "text": "#263445",
            "muted_text": "#536274",
            "grid": "#E8EDF3",
            "axis": "#CBD5E1",
            "hover": "#263445",
            "hover_border": "#475569",
            "legend": "rgba(251,252,254,0.94)",
        }
    fig.update_layout(
        template=colors["template"],
        height=height,
        hovermode=hovermode,
        font=dict(family="Arial, sans-serif", size=12, color=colors["text"]),
        paper_bgcolor=colors["paper"],
        plot_bgcolor=colors["plot"],
        margin=dict(l=58, r=28, t=42, b=56),
        hoverlabel=dict(
            bgcolor=colors["hover"],
            bordercolor=colors["hover_border"],
            font=dict(color="#F8FAFC", size=12),
            namelength=32,
        ),
        legend=dict(
            bgcolor=colors["legend"],
            bordercolor=colors["grid"],
            borderwidth=1,
            font=dict(size=11, color=colors["text"]),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=colors["grid"],
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=colors["axis"],
        ticks="outside",
        tickcolor=colors["axis"],
        title_font=dict(size=13, color=colors["muted_text"]),
        tickfont=dict(size=11, color=colors["muted_text"]),
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=colors["grid"],
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor=colors["axis"],
        ticks="outside",
        tickcolor=colors["axis"],
        title_font=dict(size=13, color=colors["muted_text"]),
        tickfont=dict(size=11, color=colors["muted_text"]),
        automargin=True,
    )
    return fig


PLOTLY_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}


@st.cache_data(ttl=15 * 60, max_entries=64, show_spinner=False)
def _cached_research_search(query, refresh_token=0):
    try:
        return search_research_articles(query), ""
    except (RuntimeError, ValueError) as exc:
        return [], str(exc)


def _build_pdf_report(
    summary_df, prediction, calibration_fit=None, additional_evidence=None
):
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
    if _valid_doi(reference.doi):
        if reference.doi not in KNOWN_DOI_MISMATCHES:
            source_text += f" DOI: {reference.doi}."
        else:
            source_text = KNOWN_DOI_MISMATCHES[reference.doi]
    if reference.doi in KNOWN_DOI_MISMATCHES:
        source_text = (
            "The bundled citation metadata was excluded because its DOI is mismatched: "
            + KNOWN_DOI_MISMATCHES[reference.doi]
        )
    story.append(paragraph(
        f"Bundled model-record citation (verify metadata independently): {source_text}"
    ))
    story.append(paragraph(
        "Standards reference: ISO 3685:1993, Tool-life testing with single-point turning tools. "
        "Used as a tool-life and flank-wear test-method reference; it is not the source of the "
        "pairing-specific Taylor constants."
    ))
    story.append(paragraph(
        "Standards reference: ASME B94.55M-1985, Tool Life Testing with Single-Point Turning Tools. "
        "Consult the applicable published edition and verify the standard's scope for the operation."
    ))
    story.append(paragraph(
        "ASTM International: consult the official standards catalog for standards applicable to "
        "the selected tool/workpiece and test method. No ASTM Taylor coefficient dataset is bundled."
    ))
    for catalog_name, catalog_scope, catalog_url in EVIDENCE_CATALOGS:
        story.append(paragraph(f"{catalog_name} catalog: {catalog_scope} {catalog_url}", "ReportSmall"))
    if additional_evidence:
        story.append(Paragraph("Research citation evidence", styles["ReportSection"]))
        story.append(paragraph(
            "These saved/imported citation records and abstracts are metadata, not a substitute "
            "for licensed full text. They provide context only and do not alter the prediction."
        ))
        for item in additional_evidence:
            citation = (
                f"{item.get('title', 'Untitled record')}. "
                f"{item.get('authors', '')} "
                f"{item.get('publication', '')} "
                f"{item.get('date', '')}. "
                f"DOI: {item.get('doi', 'not supplied')}. "
                f"{item.get('url', '')}"
            )
            story.append(paragraph(citation, "ReportSmall"))
            if item.get("abstract"):
                story.append(paragraph(f"Abstract: {item['abstract']}", "ReportSmall"))
    if calibration_fit is not None:
        story.append(paragraph(
            f"Measured-trial calibration: {calibration_fit.runs_used} setup-matched runs; "
            f"in-sample R-squared {calibration_fit.r_squared:.3f}; "
            f"in-sample log-RMSE {calibration_fit.log_rmse:.3f}; "
            f"leave-one-out R-squared {calibration_fit.loo_r_squared:.3f}; "
            f"leave-one-out log-RMSE {calibration_fit.loo_log_rmse:.3f}. "
            f"Fitted C={calibration_fit.taylor_C:.4g}, n={calibration_fit.taylor_n:.4f}, "
            f"x={calibration_fit.taylor_x:.4f}, y={calibration_fit.taylor_y:.4f}."
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
    st.session_state["material_mode"] = "📖 Curated Model Pairings"
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
            "📖 Curated Model Pairings"
        ],
        key="material_mode",
        help="Choose between independent materials or bundled pairings with literature-reference metadata.",
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
            help="Bundled Taylor-model defaults with literature-reference metadata. Verify citations in the evidence tab.",
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
        help=f"Machine envelope: 0 – {vc_cap:.0f} m/min (this machine) | Built-in model range: {v_rec_min:.0f} – {v_rec_max:.0f} m/min",
        **_init("vc_input", value=_seed_vc),
    )

    feed_input = st.slider(
        f"Feed Rate (f, {op_info.feed_unit}):",
        min_value=0.0,
        max_value=feed_cap,
        step=0.01,
        format="%.3f",
        key="feed_input",
        help=f"Machine envelope: 0 – {feed_cap:.3f} {op_info.feed_unit} (this machine) | Built-in model range: {pairing_info.f_min:.3f} – {pairing_info.f_max:.3f} {op_info.feed_unit}",
        **_init("feed_input", value=_seed_feed),
    )

    ap_input = st.slider(
        "Depth of Cut (ap, mm):",
        min_value=0.0,
        max_value=ap_cap,
        step=0.1,
        key="ap_input",
        help=f"Machine envelope: 0 – {ap_cap:.1f} mm (this machine) | Built-in model range: {pairing_info.ap_min:.1f} – {pairing_info.ap_max:.1f} mm",
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

    calibration_context = {
        "pairing_key": selected_pairing,
        "workpiece_name": pairing_info.workpiece_name,
        "tool_material": pairing_info.tool_material,
        "coating": pairing_info.coating,
        "machine_name": selected_machine,
        "coolant_name": selected_coolant,
        "operation_name": selected_operation or "",
        "milling_tooling_name": selected_milling_tooling or "",
        "holder_name": selected_holder or "",
        "overhang_ratio": overhang_input,
        "is_roughing": is_roughing,
    }
    calibration_fit = None
    calibration_trials = None
    with st.expander("Calibrate with measured tool-life trials", expanded=False):
        st.caption(
            "Upload measured runs for this exact material/tool and machine setup. "
            "Use the same failure criterion for every measured life. At least 8 runs "
            "with independently varied speed, feed, and depth are required."
        )
        calibration_template_csv = calibration_template(calibration_context).to_csv(index=False)
        st.download_button(
            "Download calibration CSV template",
            data=calibration_template_csv,
            file_name="tool_life_calibration_template.csv",
            mime="text/csv",
            key="calibration_template_download",
        )
        st.caption(
            "Fill one row per completed tool-life trial. Keep the setup columns unchanged; "
            "enter cutting speed (m/min), feed (mm/rev or mm/tooth), depth (mm), and measured "
            "life (min). Only matching setup rows are used."
        )
        calibration_file = st.file_uploader(
            "Upload completed tool-life trials CSV",
            type=["csv"],
            key="tool_life_calibration_upload",
        )
        if calibration_file is not None:
            try:
                trial_data = pd.read_csv(BytesIO(calibration_file.getvalue()))
                setup_life_multiplier = calculate_tool_life_multiplier(
                    pairing_info,
                    machine_info,
                    COOLANT_DATABASE[selected_coolant],
                    is_roughing=is_roughing,
                    operation_name=selected_operation,
                    milling_tooling_name=selected_milling_tooling,
                    holder_name=selected_holder,
                    overhang_ratio=overhang_input,
                )
                calibration_fit = fit_taylor_model(
                    trial_data,
                    calibration_context,
                    setup_life_multiplier,
                )
                calibration_trials = matching_calibration_trials(
                    trial_data,
                    calibration_context,
                )
                pairing_info = replace(
                    pairing_info,
                    taylor_C=calibration_fit.taylor_C,
                    taylor_n=calibration_fit.taylor_n,
                    taylor_x=calibration_fit.taylor_x,
                    taylor_y=calibration_fit.taylor_y,
                    v_min=calibration_fit.speed_min,
                    v_max=calibration_fit.speed_max,
                    f_min=calibration_fit.feed_min,
                    f_max=calibration_fit.feed_max,
                    ap_min=calibration_fit.depth_min,
                    ap_max=calibration_fit.depth_max,
                )
                st.success(
                    f"Applied fit to {calibration_fit.runs_used} matching runs "
                    f"(in-sample R²={calibration_fit.r_squared:.3f}; "
                    f"in-sample log-RMSE={calibration_fit.log_rmse:.3f}; "
                    f"leave-one-out R²={calibration_fit.loo_r_squared:.3f}; "
                    f"leave-one-out log-RMSE={calibration_fit.loo_log_rmse:.3f}). "
                    "Predictions and optimized scenarios now use these fitted Taylor constants."
                )
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(
                    f"Calibration was not applied: {exc} The built-in model remains active."
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
what_if_result = None
verified_trial_options = []


# Main Content Area Header
st.markdown("<div class='main-header'>⚙️ Machining Tool Wear & Life Intelligence Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='sub-header'>Forecasting flank wear ($VB$), Remaining Useful Life (RUL), and physics-driven wear minimization. Calibrate the model with your measured tool-life trials for your setup.</div>",
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
    status_color = (
        CHART_COLORS["green"]
        if pred.rul_percentage > 35
        else (CHART_COLORS["warning"] if pred.rul_percentage > 12 else CHART_COLORS["danger"])
    )
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
    conf_color = (
        CHART_COLORS["green"]
        if pred.confidence_score >= 85
        else (
            CHART_COLORS["warning"]
            if pred.confidence_score >= 60
            else CHART_COLORS["danger"]
        )
    )
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Model Range Score</div>
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
    "📚 Research, standards & evidence",
    "📊 Scenario Comparison & Data Export"
])


# ==============================================================================
# TAB 1: Graphical Forecasting & Wear Curves
# ==============================================================================
with tab1:
    st.subheader("1. Progressive Flank Wear (VB) vs. Machining Time")
    st.caption(
        "Use the range slider below the plot to zoom in on a time interval. Hover for "
        "wear values; drag to zoom, double-click to reset, or use the toolbar to pan."
    )

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
        line=dict(color=CHART_COLORS["primary"], width=3, shape="spline", smoothing=0.35),
        hovertemplate='Time: %{x:.1f} min<br>Flank Wear VB: %{y:.3f} mm<extra></extra>'
    ))

    # Failure Threshold Line
    fig_wear.add_hline(
        y=pred.vb_threshold_mm,
        line_dash="dash",
        line_color=CHART_COLORS["danger"],
        annotation_text=f"ISO 3685 Failure Criterion: VB = {pred.vb_threshold_mm:.2f} mm",
        annotation_position="top left",
        annotation_font_color=CHART_COLORS["danger"]
    )

    # Caution boundary (80% of limit)
    fig_wear.add_hrect(
        y0=pred.vb_threshold_mm * 0.80,
        y1=pred.vb_threshold_mm,
        fillcolor=CHART_COLORS["warning"],
        opacity=0.12,
        line_width=0,
        annotation_text="Accelerated Wear Warning Zone (Stage III)",
        annotation_position="bottom right"
    )

    # Current time marker
    if pred.current_time_min > 0:
        fig_wear.add_vline(
            x=pred.current_time_min,
            line_dash="dot",
            line_color=CHART_COLORS["muted"],
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
            marker=dict(size=13, color=status_color, symbol='diamond',
                        line=dict(color="white", width=1.5)),
            hovertemplate=(
                "Current operating state<br>Time: %{x:.1f} min<br>"
                "Flank wear: %{y:.3f} mm<extra></extra>"
            ),
        ))

    # Tool Life end marker
    fig_wear.add_trace(go.Scatter(
        x=[pred.tool_life_minutes],
        y=[pred.vb_threshold_mm],
        mode='markers+text',
        name='Tool Life Limit (T)',
        text=[f"Life: {pred.tool_life_minutes:.1f}m"],
        textposition="bottom right",
        marker=dict(size=12, color=CHART_COLORS["danger"], symbol='x',
                    line=dict(color="white", width=1.5)),
        hovertemplate=(
            "Predicted tool-life threshold<br>Time: %{x:.1f} min<br>"
            "Wear threshold: %{y:.3f} mm<extra></extra>"
        ),
    ))

    fig_wear.update_layout(
        xaxis_title="Machining time (min)",
        yaxis_title="Flank wear land width VB (mm)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    _polish_chart(fig_wear, height=490, hovermode="x unified")
    fig_wear.update_xaxes(rangeslider=dict(visible=True, thickness=0.08))
    st.plotly_chart(
        fig_wear,
        width="stretch",
        config=PLOTLY_CONFIG,
        key="flank_wear_chart",
    )

    # Section 2: Sensitivity Analysis (Two Columns)
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("2. Wear Rate vs. Cutting Speed")
        st.caption(
            "Average wear-rate proxy derived from the predicted wear limit and tool life. "
            "The shaded band shows the active model range; hover to inspect values."
        )

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
            name='Model wear-rate proxy',
            line=dict(color=CHART_COLORS["purple"], width=3, shape="spline", smoothing=0.3),
            fill='tozeroy',
            fillcolor=(
                "rgba(180,167,232,0.12)" if DARK_THEME else "rgba(118,103,168,0.10)"
            ),
            hovertemplate=(
                "Cutting speed: %{x:.1f} m/min<br>"
                "Wear-rate proxy: %{y:.2f} µm/min<extra></extra>"
            )
        ))

        # Mark current speed wear rate
        current_wear_rate = vb_crit_um / max(pred.tool_life_minutes, 0.01)
        fig_taylor.add_trace(go.Scatter(
            x=[pred.vc],
            y=[current_wear_rate],
            mode='markers',
            name='Current setting',
            marker=dict(size=13, color=CHART_COLORS["danger"], symbol='circle',
                        line=dict(color="white", width=1.8)),
            hovertemplate=(
                "Current setting<br>Cutting speed: %{x:.1f} m/min<br>"
                "Wear-rate proxy: %{y:.2f} µm/min<extra></extra>"
            ),
        ))

        # Add empirical boundaries
        fig_taylor.add_vrect(
            x0=pred.pairing.v_min,
            x1=pred.pairing.v_max,
            fillcolor=CHART_COLORS["green"],
            opacity=0.10,
            line_width=0,
            annotation_text="Active model range",
            annotation_position="top left"
        )

        fig_taylor.update_layout(
            xaxis_title="Cutting speed Vc (m/min)",
            yaxis_title="Average wear-rate proxy (µm/min)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        _polish_chart(fig_taylor, height=410)
        st.plotly_chart(
            fig_taylor,
            width="stretch",
            config=PLOTLY_CONFIG,
            key="wear_rate_speed_chart",
        )

    with col_g2:
        st.subheader("3. 2D Tool Life Contour (Speed vs. Feed)")
        contour_metric = st.selectbox(
            "Contour map metric",
            ["Predicted tool life (min)", "Material removal rate (cm³/min)"],
            key="contour_metric",
        )
        st.caption(
            f"Hover for exact values; drag to zoom, double-click to reset, or use the toolbar. "
            f"Current point shown; depth is fixed at {pred.ap:.3f} mm."
        )

        v_min = max(1.0, pred.pairing.v_min)
        v_max = min(pred.pairing.v_max, pred.machine.max_vc_m_per_min)
        f_min = max(0.001, pred.pairing.f_min)
        f_max = min(pred.pairing.f_max, pred.machine.max_feed_mm)
        ap_supported = (
            pred.pairing.ap_min <= pred.ap <= pred.pairing.ap_max
            and pred.ap <= pred.machine.max_ap_mm
        )
        if v_min >= v_max or f_min >= f_max or not ap_supported:
            st.info(
                "No overlapping active model/machine range is available, or the current depth "
                "is outside its supported range. No out-of-range surface is drawn."
            )
        else:
            v_grid = np.linspace(v_min, v_max, 25)
            f_grid = np.linspace(f_min, f_max, 25)
            V_mesh, F_mesh = np.meshgrid(v_grid, f_grid)

            Z_life = np.zeros(V_mesh.shape)
            Z_mrr = np.zeros(V_mesh.shape)
            for i in range(V_mesh.shape[0]):
                for j in range(V_mesh.shape[1]):
                    Z_life[i, j] = calculate_tool_life(
                        pred.pairing, pred.machine, pred.coolant,
                        V_mesh[i, j], F_mesh[i, j], pred.ap, pred.is_roughing,
                        operation_name=pred.operation_name, milling_tooling_name=pred.milling_tooling_name,
                        holder_name=pred.holder_name, overhang_ratio=pred.overhang_ratio
                    )
                    Z_mrr[i, j] = calculate_mrr(
                        V_mesh[i, j],
                        F_mesh[i, j],
                        pred.ap,
                        operation_name=pred.operation_name,
                        milling_tooling_name=pred.milling_tooling_name,
                    )

            contour_values, colorbar_title = (
                (Z_life, "Life (min)")
                if contour_metric == "Predicted tool life (min)"
                else (Z_mrr, "MRR (cm³/min)")
            )

            fig_contour = go.Figure(data=go.Contour(
                z=contour_values,
                x=v_grid,
                y=f_grid,
                colorscale=(
                    [
                        [0.0, "#203044"],
                        [0.25, "#2C5362"],
                        [0.5, "#3F7C7A"],
                        [0.75, "#77A89A"],
                        [1.0, "#C6D6B4"],
                    ]
                    if DARK_THEME
                    else [
                        [0.0, "#E8F1F4"],
                        [0.25, "#BDD8DE"],
                        [0.5, "#83B9BE"],
                        [0.75, "#4E9298"],
                        [1.0, "#286A72"],
                    ]
                ),
                ncontours=12,
                contours=dict(showlabels=False),
                colorbar=dict(
                    title=dict(text=colorbar_title, side="right"),
                    thickness=14,
                    outlinewidth=0,
                ),
                hovertemplate=(
                    f"Speed: %{{x:.1f}} m/min<br>Feed: %{{y:.4f}} {op_info.feed_unit}<br>"
                    f"{colorbar_title}: %{{z:.2f}}<extra></extra>"
                ),
            ))

            fig_contour.add_trace(go.Scatter(
                x=[pred.vc],
                y=[pred.feed],
                mode='markers+text',
                name='Current setting',
                text=["Current"],
                textposition="top center",
                marker=dict(size=14, color=CHART_COLORS["danger"], symbol='x',
                            line=dict(color="white", width=1.5)),
                hovertemplate=(
                    f"Current setting<br>Vc: {pred.vc:.1f} m/min<br>"
                    f"Feed: {pred.feed:.4f} {op_info.feed_unit}<br>"
                    f"Depth: {pred.ap:.3f} mm<extra></extra>"
                ),
            ))

            fig_contour.update_layout(
                xaxis_title="Cutting speed Vc (m/min)",
                yaxis_title=f"Feed rate f ({op_info.feed_unit})",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            _polish_chart(fig_contour, height=410)
            st.plotly_chart(
                fig_contour,
                width="stretch",
                config=PLOTLY_CONFIG,
                key="parameter_contour_chart",
            )


# ==============================================================================
# TAB 2: How to Minimize Tool Wear (Optimization Engine)
# ==============================================================================
with tab2:
    st.subheader("🎯 How to Minimize Tool Wear: Optimization Strategies")
    st.markdown("""
    Tool wear is governed by friction, interface temperatures, mechanical fatigue, and chemical diffusion. 
    Below are model-calculated strategies, selected database guidance, and machine-/operation-specific checks.
    """)
    st.caption(
        f"Active pairing: {pred.pairing.workpiece_name} with {pred.pairing.tool_material} "
        f"({pred.pairing.coating}); machine: {pred.machine_name}; coolant: {pred.coolant_name}; "
        f"operation: {pred.operation_name or 'not specified'}. "
        + (
            f"Taylor coefficients fitted from {calibration_fit.runs_used} matching measured trials."
            if calibration_fit is not None
            else "Using bundled Taylor coefficients; exact setup performance is not independently verified."
        )
    )

    st.markdown("---")
    st.subheader("Interactive what-if: adjust your cutting parameters")
    st.caption(
        "Change these controls to immediately recalculate the forecast. They start at your "
        "current settings and follow the active model limits and selected machine envelope; "
        "the current setting remains available for comparison if it is outside the model range."
    )
    def _what_if_control_bounds(model_min, model_max, machine_max, current, floor, step):
        machine_max = max(float(machine_max), floor + step)
        current = min(machine_max, max(floor, float(current)))
        lower = max(floor, min(float(model_min), current))
        upper = min(machine_max, max(float(model_max), current))
        if upper <= lower:
            lower = max(floor, min(current, machine_max - step))
            upper = min(machine_max, lower + step)
        return lower, upper, current, step

    what_if_bounds = {
        "speed": _what_if_control_bounds(
            pred.pairing.v_min, pred.pairing.v_max,
            machine_info.max_vc_m_per_min, pred.vc, 1.0, 1.0,
        ),
        "feed": _what_if_control_bounds(
            pred.pairing.f_min, pred.pairing.f_max,
            machine_info.max_feed_mm, pred.feed, 0.001, 0.001,
        ),
        "depth": _what_if_control_bounds(
            pred.pairing.ap_min, pred.pairing.ap_max,
            machine_info.max_ap_mm, pred.ap, 0.05, 0.05,
        ),
    }
    for name, (minimum, maximum, current, step) in what_if_bounds.items():
        if maximum <= minimum:
            maximum = minimum + step
        state_key = f"what_if_{name}"
        if state_key in st.session_state:
            st.session_state[state_key] = min(
                maximum, max(minimum, float(st.session_state[state_key]))
            )
        else:
            st.session_state[state_key] = min(maximum, max(minimum, current))

    what_if_col1, what_if_col2, what_if_col3 = st.columns(3)
    with what_if_col1:
        what_if_vc = st.slider(
            "What-if cutting speed (m/min)",
            min_value=what_if_bounds["speed"][0],
            max_value=what_if_bounds["speed"][1],
            step=what_if_bounds["speed"][3],
            key="what_if_speed",
        )
    with what_if_col2:
        what_if_feed = st.slider(
            f"What-if feed ({op_info.feed_unit})",
            min_value=what_if_bounds["feed"][0],
            max_value=what_if_bounds["feed"][1],
            step=what_if_bounds["feed"][3],
            format="%.4f",
            key="what_if_feed",
        )
    with what_if_col3:
        what_if_ap = st.slider(
            "What-if depth of cut (mm)",
            min_value=what_if_bounds["depth"][0],
            max_value=what_if_bounds["depth"][1],
            step=what_if_bounds["depth"][3],
            format="%.3f",
            key="what_if_depth",
        )

    what_if_life = calculate_tool_life(
        pairing_info,
        machine_info,
        COOLANT_DATABASE[selected_coolant],
        what_if_vc,
        what_if_feed,
        what_if_ap,
        is_roughing=is_roughing,
        operation_name=selected_operation,
        milling_tooling_name=selected_milling_tooling,
        holder_name=selected_holder,
        overhang_ratio=overhang_input,
    )
    what_if_mrr = calculate_mrr(
        what_if_vc,
        what_if_feed,
        what_if_ap,
        operation_name=selected_operation,
        milling_tooling_name=selected_milling_tooling,
    )
    what_if_result = {
        "vc": what_if_vc,
        "feed": what_if_feed,
        "ap": what_if_ap,
        "tool_life_min": what_if_life,
        "mrr_cm3_min": what_if_mrr,
        "tool_life_gain_pct": (
            (what_if_life - pred.tool_life_minutes)
            / max(pred.tool_life_minutes, 1e-9)
            * 100.0
        ),
        "mrr_change_pct": (
            (what_if_mrr - pred.mrr_cm3_min)
            / max(pred.mrr_cm3_min, 1e-9)
            * 100.0
        ),
    }
    what_if_out_of_range = [
        label for label, value, lower, upper in (
            ("speed", what_if_vc, pred.pairing.v_min, pred.pairing.v_max),
            ("feed", what_if_feed, pred.pairing.f_min, pred.pairing.f_max),
            ("depth", what_if_ap, pred.pairing.ap_min, pred.pairing.ap_max),
        )
        if not lower <= value <= upper
    ]
    if what_if_out_of_range:
        st.warning(
            "The what-if candidate is outside the active model range for "
            f"{', '.join(what_if_out_of_range)}; those estimates are extrapolations."
        )
    st.caption(
        "Both forecasts use the same material, machine, coolant, holder, overhang, and roughing "
        "settings; only speed, feed, and depth differ."
    )
    st.dataframe(pd.DataFrame([
        {
            "Scenario": "Current",
            "Speed (m/min)": pred.vc,
            f"Feed ({op_info.feed_unit})": pred.feed,
            "Depth of cut (mm)": pred.ap,
        },
        {
            "Scenario": "What-if",
            "Speed (m/min)": what_if_vc,
            f"Feed ({op_info.feed_unit})": what_if_feed,
            "Depth of cut (mm)": what_if_ap,
        },
    ]), hide_index=True, width="stretch")
    what_if_metric_cols = st.columns(3)
    what_if_metric_cols[0].metric(
        "What-if predicted tool life",
        f"{what_if_life:.1f} min",
        f"{what_if_result['tool_life_gain_pct']:+.1f}% vs current",
    )
    what_if_metric_cols[1].metric(
        "What-if material removal rate",
        f"{what_if_mrr:.2f} cm³/min",
        f"{what_if_result['mrr_change_pct']:+.1f}% vs current",
    )
    what_if_metric_cols[2].metric(
        "What-if remaining life",
        f"{max(0.0, what_if_life - current_time_input):.1f} min",
        help="Uses the same elapsed tool time as the current forecast.",
    )
    comparison_fig = go.Figure()
    comparison_fig.add_trace(go.Bar(
        x=["Current", "What-if"],
        y=[pred.tool_life_minutes, what_if_life],
        name="Predicted tool life (min)",
        marker_color=["#8793A3" if DARK_THEME else "#8895A4", CHART_COLORS["primary"]],
        text=[f"{pred.tool_life_minutes:.1f}", f"{what_if_life:.1f}"],
        textposition="auto",
        marker_line_color="white",
        marker_line_width=1.5,
        hovertemplate="%{x}<br>Predicted tool life: %{y:.1f} min<extra></extra>",
    ))
    comparison_fig.update_layout(
        title="Current vs what-if predicted tool life",
        yaxis_title="Predicted tool life (min)",
        showlegend=False,
    )
    _polish_chart(comparison_fig, height=320)
    comparison_fig.update_xaxes(title_text=None)
    st.plotly_chart(
        comparison_fig,
        width="stretch",
        config=PLOTLY_CONFIG,
        key="what_if_life_comparison",
    )
    if calibration_fit is None:
        st.caption(
            "What-if values are model estimates, not verified outcomes. Upload measured "
            "setup-matched trials to calibrate the model, then validate changes on the machine."
        )
    else:
        st.caption(
            f"Using {calibration_fit.runs_used} setup-matched calibration trials "
            f"(leave-one-out log-RMSE {calibration_fit.loo_log_rmse:.3f}); "
            "validate candidate changes with controlled shop trials."
        )

    # Strategy Comparison Table
    st.markdown("---")
    st.subheader("Heuristic strategy previews — model estimates")
    st.caption(
        "These quick previews apply fixed parameter adjustments to the Taylor model. They are "
        "not the audited database search below and are not trial-verified recommendations."
    )
    opt_col1, opt_col2, opt_col3 = st.columns(3)

    # Strategy 1: Balanced
    with opt_col1:
        st.markdown("### Productivity-neutral model candidate")
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
        - **Feed ($f$)**: `{opt_report.balanced_strategy.feed} {op_info.feed_unit}` *(from {pred.feed:.3f})*
        - **Depth ($a_p$)**: `{opt_report.balanced_strategy.ap} mm` *(from {pred.ap:.2f})*
        """)

    # Strategy 2: Max Life
    with opt_col2:
        st.markdown("### Maximum predicted life (model estimate)")
        st.caption("Not a breakage-risk model; do not use this estimate alone to justify unattended operation.")
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
        - **Feed ($f$)**: `{opt_report.max_life_strategy.feed} {op_info.feed_unit}`
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
        - **Feed ($f$)**: `{opt_report.high_efficiency_strategy.feed} {op_info.feed_unit}`
        - **Depth ($a_p$)**: `{opt_report.high_efficiency_strategy.ap} mm`
        """)

    st.markdown("---")
    st.subheader("Database-bounded plans for different productivity targets")
    st.caption(
        "Offline model screening: for each target the search evaluates 41 × 41 speed/feed "
        "combinations inside the active model and machine ranges, solves depth for the target "
        "MRR, then checks range membership, finite outputs, MRR agreement, and a repeated life "
        "calculation. A ±2% one-at-a-time perturbation check reports how sensitive each selected "
        "candidate is near its operating point."
    )
    st.warning(
        "MODEL-SCREENED ESTIMATES — NOT VERIFIED OPTIMIZATIONS. Numerical checks can find "
        "inconsistent calculations and fragile candidates, but cannot establish real cutting "
        "accuracy. Bundled coefficients do not prove performance for your tool edge. Live/offline "
        "article citation metadata is not numerical cutting-test data. Use measured, setup-matched "
        "trials as the separate evidence source below, then validate one change at a time."
    )
    if calibration_fit is not None:
        if np.isfinite(calibration_fit.loo_log_rmse) and calibration_fit.loo_log_rmse <= np.log(100.0):
            loo_error_summary = (
                f"about {np.expm1(calibration_fit.loo_log_rmse) * 100.0:.1f}% "
                "multiplicative error"
            )
        else:
            loo_error_summary = "more than 9,900% multiplicative error"
        st.success(
            f"Search uses Taylor constants fitted to {calibration_fit.runs_used} matching runs. "
            f"Leave-one-out log-RMSE = {calibration_fit.loo_log_rmse:.3f} "
            f"({loo_error_summary} on held-out trials in this CSV); "
            "this is sample-specific and not a production "
            "guarantee."
        )
    else:
        st.info(
            "Search uses bundled Taylor model records. Predictive error for this exact setup is "
            "unknown until measured trials are collected."
        )
    if opt_report.productivity_frontier_options:
        frontier_df = pd.DataFrame([
            {
                "Target MRR vs current (%)": option["target_productivity_pct"],
                "Speed (m/min)": round(option["vc"], 1),
                f"Feed ({op_info.feed_unit})": round(option["feed"], 4),
                "Depth (mm)": round(option["ap"], 3),
                "Predicted tool life (min)": round(option["tool_life_min"], 1),
                "Life change vs current (%)": round(option["tool_life_gain_pct"], 1),
                "Target MRR (cm³/min)": round(option["mrr_cm3_min"], 2),
                "MRR check error (%)": round(option["mrr_error_pct"], 4),
                "Life recheck error (%)": round(option["recheck_error_pct"], 6),
                "Life sensitivity at ±2% (%)": round(option["local_sensitivity_pct"], 1),
            }
            for option in opt_report.productivity_frontier_options
        ])
        st.dataframe(frontier_df, hide_index=True, width="stretch")
        for option in opt_report.productivity_frontier_options:
            with st.expander(
                f"{option['target_productivity_pct']:.0f}% productivity plan — "
                f"predicted tool-life change {option['tool_life_gain_pct']:+.1f}%"
            ):
                st.markdown(
                    f"Try **Vc {option['vc']:.1f} m/min**, "
                    f"**feed {option['feed']:.4f} {op_info.feed_unit}**, "
                    f"**depth {option['ap']:.3f} mm**. Predicted MRR is "
                    f"{option['mrr_cm3_min']:.2f} cm³/min and predicted life is "
                    f"{option['tool_life_min']:.1f} min."
                )
                st.caption(
                    f"Numerical audit passed (MRR error {option['mrr_error_pct']:.4f}%; "
                    f"life recheck error {option['recheck_error_pct']:.6f}%). A ±2% change in "
                    f"one input changes the nearest life estimate by up to "
                    f"{option['local_sensitivity_pct']:.1f}%. This sensitivity is not an accuracy "
                    "confidence interval. Check power, chip load, tool-maker limits, workholding, "
                    "and quality requirements; validate in controlled trials."
                )
    else:
        st.warning(
            "No candidate met all selected model-range and machine-envelope limits at the "
            "requested productivity targets. Check the active ranges, machine selection, "
            "and current MRR."
        )

    st.markdown("---")
    st.subheader("Measured-trial-backed optimization")
    st.caption(
        "This section ranks only actual measured tool-life runs for the exact selected setup. "
        "It reads the uploaded CSV locally and works offline; live article metadata is citation "
        "information, not numerical cutting-test data."
    )
    if calibration_trials is None:
        st.info(
            "No valid setup-matched trial CSV is active. Upload measured tool-life runs in the "
            "sidebar calibration panel to show observed optimization options. Model-only "
            "strategies above remain estimates and are not shown here as verified."
        )
    else:
        measured_rows = []
        for trial_number, trial in enumerate(
            calibration_trials.to_dict(orient="records"),
            start=1,
        ):
            trial_vc = float(trial["cutting_speed_m_min"])
            trial_feed = float(trial["feed_rate_mm"])
            trial_ap = float(trial["depth_of_cut_mm"])
            if (
                trial_vc > machine_info.max_vc_m_per_min
                or trial_feed > machine_info.max_feed_mm
                or trial_ap > machine_info.max_ap_mm
            ):
                continue
            trial_mrr = calculate_mrr(
                trial_vc,
                trial_feed,
                trial_ap,
                operation_name=selected_operation,
                milling_tooling_name=selected_milling_tooling,
            )
            measured_rows.append({
                "Trial": f"Matched trial {trial_number}",
                "Speed (m/min)": trial_vc,
                f"Feed ({op_info.feed_unit})": trial_feed,
                "Depth (mm)": trial_ap,
                "Measured tool life (min)": float(trial["measured_tool_life_min"]),
                "Calculated MRR (cm³/min)": float(trial_mrr),
            })

        if not measured_rows:
            st.warning(
                "None of the matching measured runs are within the selected machine's stated "
                "speed/feed/depth envelope, so no trial-backed candidate is recommended."
            )
        else:
            target_levels = (80, 100, 120)
            for target_pct in target_levels:
                target_mrr = pred.mrr_cm3_min * target_pct / 100.0
                eligible = [
                    row for row in measured_rows
                    if row["Calculated MRR (cm³/min)"] >= target_mrr
                ]
                if eligible:
                    best_trial = max(
                        eligible,
                        key=lambda row: row["Measured tool life (min)"],
                    )
                    verified_trial_options.append({
                        **best_trial,
                        "Scenario": f"Measured trial (≥{target_pct}% current MRR)",
                        "Target MRR (% current)": target_pct,
                    })

            st.success(
                f"Analyzed {len(measured_rows)} complete, setup-matched measured runs within "
                "the selected machine envelope. Candidates below are observed test points, "
                "not generated predictions."
            )
            if verified_trial_options:
                st.dataframe(
                    pd.DataFrame([{
                        "Minimum MRR target (% current)": row["Target MRR (% current)"],
                        "Observed trial": row["Trial"],
                        "Speed (m/min)": row["Speed (m/min)"],
                        f"Feed ({op_info.feed_unit})": row[f"Feed ({op_info.feed_unit})"],
                        "Depth (mm)": row["Depth (mm)"],
                        "Measured tool life (min)": row["Measured tool life (min)"],
                        "Calculated MRR (cm³/min)": row["Calculated MRR (cm³/min)"],
                    } for row in verified_trial_options]),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info(
                    "No measured run met the 80% of current MRR target. Review the measured "
                    "runs below and collect additional trials at relevant productivity levels."
                )
            with st.expander("Review all matched measured runs"):
                st.dataframe(
                    pd.DataFrame(measured_rows).sort_values(
                        "Measured tool life (min)",
                        ascending=False,
                    ),
                    hide_index=True,
                    width="stretch",
                )
            st.caption(
                "Tool life in this table is measured in the uploaded trials; MRR is calculated "
                "from their recorded parameters and the selected operation/tool geometry. "
                "Repeat trials and confirm quality, chip load, workholding, and tool-maker "
                "limits before applying a setting to production."
            )

    st.markdown("---")

    # Pareto Frontier Plot
    col_p1, col_p2 = st.columns([3, 2])

    with col_p1:
        st.subheader("Model trade-off: tool life vs. productivity")
        st.caption(
            "The line is a model sweep at fixed median feed and depth (not a measured "
            "Pareto frontier). Points are shown separately; hover for details and use the "
            "legend to hide/show series."
        )

        fig_pareto = go.Figure()

        fig_pareto.add_trace(go.Scatter(
            x=opt_report.pareto_mrr,
            y=opt_report.pareto_tool_life,
            mode='lines+markers',
            name='Taylor-model speed sweep',
            line=dict(color=CHART_COLORS["primary"], width=3),
            marker=dict(size=6, color=CHART_COLORS["primary"], line=dict(color="#F8FAFC", width=1)),
            hovertemplate=(
                "Model sweep<br>MRR: %{x:.2f} cm³/min<br>"
                "Predicted life: %{y:.1f} min<extra></extra>"
            ),
        ))

        # Mark current point
        fig_pareto.add_trace(go.Scatter(
            x=[pred.mrr_cm3_min],
            y=[pred.tool_life_minutes],
            mode='markers+text',
            name='Current setting',
            text=["Current"],
            textposition="top right",
            marker=dict(size=14, color=CHART_COLORS["danger"], symbol='circle',
                        line=dict(color="white", width=1.8)),
            hovertemplate=(
                "Current setting<br>MRR: %{x:.2f} cm³/min<br>"
                "Predicted life: %{y:.1f} min<extra></extra>"
            ),
        ))

        fig_pareto.add_trace(go.Scatter(
            x=[opt_report.balanced_strategy.mrr_cm3_min],
            y=[opt_report.balanced_strategy.tool_life_min],
            mode='markers+text',
            name='Productivity-Neutral Model Estimate',
            text=["Model estimate"],
            textposition="bottom left",
            marker=dict(size=15, color=CHART_COLORS["purple"], symbol='star',
                        line=dict(color="white", width=1.5)),
            hovertemplate=(
                "Productivity-neutral model estimate<br>MRR: %{x:.2f} cm³/min<br>"
                "Predicted life: %{y:.1f} min<extra></extra>"
            ),
        ))

        if what_if_result is not None:
            fig_pareto.add_trace(go.Scatter(
                x=[what_if_result["mrr_cm3_min"]],
                y=[what_if_result["tool_life_min"]],
                mode="markers",
                name="Interactive what-if",
                marker=dict(size=13, color=CHART_COLORS["warning"], symbol="diamond",
                            line=dict(color="white", width=1.5)),
                hovertemplate=(
                    "What-if model estimate<br>MRR: %{x:.2f} cm³/min<br>"
                    "Predicted life: %{y:.1f} min<extra></extra>"
                ),
            ))
        if verified_trial_options:
            fig_pareto.add_trace(go.Scatter(
                x=[
                    option["Calculated MRR (cm³/min)"]
                    for option in verified_trial_options
                ],
                y=[
                    option["Measured tool life (min)"]
                    for option in verified_trial_options
                ],
                mode="markers",
                name="Measured trial",
                marker=dict(size=12, color=CHART_COLORS["blue"], symbol="square",
                            line=dict(color="white", width=1.5)),
                text=[option["Trial"] for option in verified_trial_options],
                hovertemplate=(
                    "%{text}<br>Calculated MRR: %{x:.2f} cm³/min<br>"
                    "Measured tool life: %{y:.1f} min<extra></extra>"
                ),
            ))

        fig_pareto.update_layout(
            xaxis_title="Material removal rate (cm³/min)",
            yaxis_title="Tool life (min)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        _polish_chart(fig_pareto, height=450)
        st.plotly_chart(
            fig_pareto,
            width="stretch",
            config=PLOTLY_CONFIG,
            key="pareto_tradeoff_chart",
        )

    with col_p2:
        st.subheader("Database-informed guidance and setup checks")

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
# TAB 3: Research, standards, and evidence
# ==============================================================================
research_evidence_results = []
research_report_evidence = []
research_live_error = ""
with tab3:
    st.subheader("📚 Research, standards, and model evidence")
    ref = pred.pairing.springer_ref
    if calibration_fit is not None:
        range_label = "Measured Trial Range"
    else:
        range_label = "Bundled Model Range"
    is_synthesized_reference = (
        ref.doi.startswith("model:")
        or "empirical-synthesis" in ref.doi
        or "synthesized" in ref.title.lower()
        or "composite model" in ref.title.lower()
    )
    if ref.doi in KNOWN_DOI_MISMATCHES:
        st.caption("Bundled citation metadata: unverified")
    else:
        st.caption(ref.evidence_level)
    st.info(
        "Publisher and standards catalogs below are reference sources, not live databases "
        "feeding this model. The built-in coefficients come from the bundled model record. "
        "Uploading setup-matched measured trials is the only source here that recalibrates "
        "the numeric prediction."
    )

    st.markdown("### Live machining research across publishers")
    st.caption(
        "Searches Crossref's public journal-article metadata across publishers for records "
        "related to the selected workpiece, tool, coating, and operation. No API key is "
        "needed; an internet connection is required. Results are citation metadata—not "
        "full-text data or validated cutting tables—and do not change numerical predictions."
    )
    setup_terms = [
        pred.pairing.workpiece_name,
        pred.pairing.tool_material,
        pred.pairing.coating,
        pred.operation_name or "",
        pred.coolant_name or "",
        "machining tool wear tool life",
    ]
    research_query = " ".join(term.strip() for term in setup_terms if term.strip())
    research_query = research_query[:250]
    st.caption(
        f"Current simulated conditions: Vc {pred.vc:.1f} m/min, feed {pred.feed:.3f}, "
        f"depth {pred.ap:.2f} mm, coolant {pred.coolant_name}."
    )
    st.caption(
        "The live bibliographic query follows the selected workpiece/tool/operation. "
        "Numeric cut values are shown as simulation context because publisher metadata "
        "indexes generally do not expose paper-specific cutting-test tables."
    )
    refresh_token = st.session_state.get("research_reference_refresh_token", 0)
    try:
        with st.spinner("Searching research metadata across publishers..."):
            live_records, live_error = _cached_research_search(
                research_query, refresh_token
            )
        if not live_error:
            research_evidence_results = live_records
            st.session_state["crossref_research_evidence_records"] = live_records
            st.session_state["crossref_research_evidence_query"] = research_query
        else:
            research_live_error = live_error
            research_evidence_results = st.session_state.get(
                "crossref_research_evidence_records",
                st.session_state.get("crossref_springer_evidence_records", []),
            )
            st.warning(
                f"Live research lookup is unavailable: {live_error} "
                "The dashboard simulation and locally saved citation library remain available."
            )
            if st.button("Retry live reference lookup", key="retry_research_reference_lookup"):
                st.session_state["research_reference_refresh_token"] = refresh_token + 1
                st.rerun()
    except (RuntimeError, ValueError) as exc:
        research_live_error = str(exc)
        st.warning(
            f"Live research lookup is unavailable: {exc} "
            "The dashboard simulation and locally saved citation library remain available."
        )
    searched_query = st.session_state.get(
        "crossref_research_evidence_query",
        st.session_state.get("crossref_springer_evidence_query", ""),
    )
    if research_evidence_results:
        if searched_query != research_query:
            st.caption(
                "Showing saved results from the previous setup because the live lookup "
                "could not complete."
            )
        st.caption(
            f"{len(research_evidence_results)} live journal-article record(s) "
            "across publishers (Crossref metadata)"
        )
        for index, item in enumerate(research_evidence_results, start=1):
            publisher = item.get("publisher") or "Publisher not specified"
            with st.expander(f"{index}. {item['title']} — {publisher}"):
                if item.get("authors"):
                    st.write(f"**Authors:** {item['authors']}")
                if item.get("publication") or item.get("date"):
                    st.write(
                        f"**Publication:** {item.get('publication', '')} "
                        f"{item.get('date', '')}"
                    )
                if item.get("publisher"):
                    st.write(f"**Publisher:** {item['publisher']}")
                if item.get("doi"):
                    st.write(f"**DOI:** {item['doi']}")
                if item.get("abstract"):
                    st.write(item["abstract"])
                if item.get("url"):
                    st.link_button("Open article record", item["url"])
        evidence_csv = pd.DataFrame(research_evidence_results).to_csv(index=False)
        st.download_button(
            "Download current research references (CSV)",
            data=evidence_csv,
            file_name="machining_research_references.csv",
            mime="text/csv",
            key="download_research_live_evidence",
        )
    elif not research_live_error:
        st.info(
            "No journal article metadata matched the current setup in Crossref. The simulation "
            "is unaffected; use the local citation library below or adjust the selected setup."
        )

    st.markdown("### Public handbooks, data, and standards catalogs")
    st.caption(
        "Curated links below were selected for machining formulas, tool-life test methods, "
        "or machining/wear research. Some publisher handbooks and standards require purchase "
        "or institutional access; links are discovery sources, not locally copied content."
    )
    for catalog_name, catalog_scope, catalog_url in EVIDENCE_CATALOGS:
        st.markdown(f"- [{catalog_name}]({catalog_url}) — {catalog_scope}")

    st.markdown("### Saved research citations for offline use")
    st.caption(
        "Export citation metadata from a publisher or library in RIS, BibTeX, or CSV format, "
        "then import it here. Citation metadata is saved beside this app on this computer and "
        "remains available after restart. Article PDFs/full text are not imported."
    )
    reference_library_path = Path(__file__).resolve().with_name(
        "springer_references.json"
    )
    try:
        offline_references = load_reference_library(reference_library_path)
    except ValueError as exc:
        offline_references = []
        st.error(f"Could not load saved offline references: {exc}")

    with st.form("import_springer_reference_export", clear_on_submit=False):
        citation_upload = st.file_uploader(
            "Choose a citation export",
            type=["ris", "bib", "bibtex", "csv"],
            key="springer_reference_import_file",
        )
        import_references = st.form_submit_button("Import and save references offline")
    if import_references:
        if citation_upload is None:
            st.error("Choose a RIS, BibTeX, or CSV citation export before importing.")
        else:
            try:
                imported_references = parse_reference_export(
                    citation_upload.name, citation_upload.getvalue()
                )
                offline_references = merge_references(
                    offline_references, imported_references
                )
                save_reference_library(reference_library_path, offline_references)
                st.success(
                    f"Saved {len(offline_references)} citation record(s) to the local "
                    "offline reference library."
                )
            except ValueError as exc:
                st.error(f"References were not imported: {exc}")

    offline_relevant_evidence = relevant_references(
        offline_references, research_query, limit=20
    )
    research_report_evidence = merge_references(
        offline_relevant_evidence, research_evidence_results
    )[:20]
    if offline_references:
        st.caption(
            f"{len(offline_references)} locally saved citation record(s); "
            f"{len(offline_relevant_evidence)} match this simulation's terms."
        )
        if offline_relevant_evidence:
            for index, item in enumerate(offline_relevant_evidence, start=1):
                with st.expander(f"{index}. {item['title']}"):
                    if item.get("authors"):
                        st.write(f"**Authors:** {item['authors']}")
                    if item.get("publication") or item.get("date"):
                        st.write(
                            f"**Publication:** {item.get('publication', '')} "
                            f"{item.get('date', '')}"
                        )
                    if item.get("doi"):
                        st.write(f"**DOI:** {item['doi']}")
                    if item.get("abstract"):
                        st.write(item["abstract"])
                    if item.get("url"):
                        st.link_button("Open article record", item["url"])
        with st.expander(
            f"View complete offline reference library ({len(offline_references)} records)"
        ):
            st.dataframe(
                pd.DataFrame(offline_references),
                hide_index=True,
                width="stretch",
            )
        st.download_button(
            "Download complete offline reference library (CSV)",
            data=pd.DataFrame(offline_references).to_csv(index=False),
            file_name="springer_offline_references.csv",
            mime="text/csv",
            key="download_offline_springer_library",
        )
    else:
        st.info(
            "No citation export has been saved on this computer yet. Import citations "
            "while online or transfer an RIS, BibTeX, or CSV export here by USB; after "
            "import, references remain visible when this system is offline."
        )

    if ref.doi in KNOWN_DOI_MISMATCHES:
        st.error(
            "Citation verification warning: the bundled DOI does not verify the paper details "
            f"stored for this pairing. Those details are not presented as verified evidence. {KNOWN_DOI_MISMATCHES[ref.doi]}"
        )
    elif is_synthesized_reference:
        st.markdown("### 📖 Composite model evidence")
        st.info(
            "This independent material/tool/coating combination is synthesized by the dashboard; "
            "it is not a single publisher experiment. Its estimated parameters are derived from "
            "the selected material/tool records and the physics model, so no study-specific DOI "
            "is asserted for this combination."
        )
        st.markdown(
            f"**Model record**: {ref.title}  \n"
            f"**Evidence classification**: {ref.evidence_level}"
        )
    else:
        st.caption(
            "The citation shown here is bundled metadata. This offline app does not query "
            "publisher indexes to independently verify every article record."
        )
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
        if ref.doi not in KNOWN_DOI_MISMATCHES:
            st.markdown("#### 🔬 Experimental Methodology & Setup")
            st.info(ref.experimental_setup)

            st.markdown("#### 🔍 Observed Wear Mechanisms")
            st.warning(ref.observed_wear_mechanisms)

    with col_r2:
        if ref.doi not in KNOWN_DOI_MISMATCHES:
            st.markdown("#### 💡 Key Research Findings")
            st.success(ref.key_findings)

        st.markdown("#### 📐 Active Extended Taylor Model Constants")
        st.markdown(f"""
        $$V_c \\cdot T^{{{pred.pairing.taylor_n:.3f}}} \\cdot f^{{{pred.pairing.taylor_x:.3f}}} \\cdot a_p^{{{pred.pairing.taylor_y:.3f}}} = {pred.pairing.taylor_C:.1f}$$
        - **Taylor Constant ($C$)**: `{pred.pairing.taylor_C:.1f}`
        - **Taylor Exponent ($n$)**: `{pred.pairing.taylor_n:.3f}` *(Sensitivity = {1.0/pred.pairing.taylor_n:.2f})*
        - **Feed Exponent ($x$)**: `{pred.pairing.taylor_x:.3f}`
        - **Depth Exponent ($y$)**: `{pred.pairing.taylor_y:.3f}`
        """)
        if calibration_fit is not None:
            st.success(
                f"Taylor constants are fitted to {calibration_fit.runs_used} measured runs "
                f"for the exact selected setup (leave-one-out R² = {calibration_fit.loo_r_squared:.3f}; "
                f"leave-one-out log-RMSE = {calibration_fit.loo_log_rmse:.3f}). "
                "The empirical boundary below is limited to the uploaded trials' min/max values."
            )
        else:
            st.caption(
                "Built-in Taylor constants are model defaults, not independently validated "
                "against your machine. Upload measured trials in the sidebar to calibrate."
            )

    st.markdown("---")
    st.subheader("Source catalogs and standards")
    for catalog_name, catalog_scope, catalog_url in EVIDENCE_CATALOGS:
        st.markdown(f"- [{catalog_name}]({catalog_url}) — {catalog_scope}")
    st.warning(
        "ISO, ASME, and ASTM documents are standards/catalogs, not openly available prediction "
        "databases. Their full texts may require purchase or subscription. Confirm the selected "
        "edition and applicability; do not infer Taylor constants from a test-method standard."
    )

    st.markdown("---")
    st.subheader("Experimental Validity Envelope & Confidence Audit")
    
    st.markdown(f"**Heuristic Model Range Score**: `{pred.confidence_score:.0f}%` ({pred.confidence_label})")
    st.caption("This score checks whether inputs fall inside the active parameter range; it is not a statistical confidence interval or measured accuracy.")
    
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
            f"Feed ({op_info.feed_unit})": pred.feed,
            "Depth ap (mm)": pred.ap,
            "Tool Life (min)": round(pred.tool_life_minutes, 1),
            "MRR (cm³/min)": round(pred.mrr_cm3_min, 2),
            "Life Gain (%)": "0.0%",
            "Productivity Change (%)": "0.0%"
        },
        {
            "Scenario": "Optimized (Balanced / Low-Wear)",
            "Speed (m/min)": opt_report.balanced_strategy.vc,
            f"Feed ({op_info.feed_unit})": opt_report.balanced_strategy.feed,
            "Depth ap (mm)": opt_report.balanced_strategy.ap,
            "Tool Life (min)": opt_report.balanced_strategy.tool_life_min,
            "MRR (cm³/min)": opt_report.balanced_strategy.mrr_cm3_min,
            "Life Gain (%)": f"+{opt_report.balanced_strategy.tool_life_gain_pct:.1f}%",
            "Productivity Change (%)": f"{opt_report.balanced_strategy.mrr_change_pct:+.1f}%"
        },
        {
            "Scenario": "Maximum Predicted Life (Model Estimate)",
            "Speed (m/min)": opt_report.max_life_strategy.vc,
            f"Feed ({op_info.feed_unit})": opt_report.max_life_strategy.feed,
            "Depth ap (mm)": opt_report.max_life_strategy.ap,
            "Tool Life (min)": opt_report.max_life_strategy.tool_life_min,
            "MRR (cm³/min)": opt_report.max_life_strategy.mrr_cm3_min,
            "Life Gain (%)": f"+{opt_report.max_life_strategy.tool_life_gain_pct:.1f}%",
            "Productivity Change (%)": f"{opt_report.max_life_strategy.mrr_change_pct:+.1f}%"
        },
        {
            "Scenario": "High Efficiency (MRR Boost)",
            "Speed (m/min)": opt_report.high_efficiency_strategy.vc,
            f"Feed ({op_info.feed_unit})": opt_report.high_efficiency_strategy.feed,
            "Depth ap (mm)": opt_report.high_efficiency_strategy.ap,
            "Tool Life (min)": opt_report.high_efficiency_strategy.tool_life_min,
            "MRR (cm³/min)": opt_report.high_efficiency_strategy.mrr_cm3_min,
            "Life Gain (%)": f"{opt_report.high_efficiency_strategy.tool_life_gain_pct:+.1f}%",
            "Productivity Change (%)": f"+{opt_report.high_efficiency_strategy.mrr_change_pct:.1f}%"
        },
    ])
    if opt_report.productivity_frontier_options:
        frontier_rows = pd.DataFrame([
            {
                "Scenario": (
                    "Model search ({:.0f}% current MRR)".format(
                        option["target_productivity_pct"]
                    )
                ),
                "Speed (m/min)": round(option["vc"], 1),
                f"Feed ({op_info.feed_unit})": round(option["feed"], 4),
                "Depth ap (mm)": round(option["ap"], 3),
                "Tool Life (min)": round(option["tool_life_min"], 1),
                "MRR (cm³/min)": round(option["mrr_cm3_min"], 2),
                "Life Gain (%)": f"{option['tool_life_gain_pct']:+.1f}%",
                "Productivity Change (%)": (
                    f"{option['target_productivity_pct'] - 100:+.0f}%"
                ),
            }
            for option in opt_report.productivity_frontier_options
        ])
        summary_df = pd.concat([summary_df, frontier_rows], ignore_index=True)
    if what_if_result is not None:
        what_if_row = pd.DataFrame([{
            "Scenario": "Interactive What-if",
            f"Speed (m/min)": round(what_if_result["vc"], 1),
            f"Feed ({op_info.feed_unit})": round(what_if_result["feed"], 4),
            "Depth ap (mm)": round(what_if_result["ap"], 3),
            "Tool Life (min)": round(what_if_result["tool_life_min"], 1),
            "MRR (cm³/min)": round(what_if_result["mrr_cm3_min"], 2),
            "Life Gain (%)": f"{what_if_result['tool_life_gain_pct']:+.1f}%",
            "Productivity Change (%)": f"{what_if_result['mrr_change_pct']:+.1f}%",
        }])
        summary_df = pd.concat([summary_df, what_if_row], ignore_index=True)
    if verified_trial_options:
        trial_option_rows = pd.DataFrame([{
            "Scenario": option["Scenario"],
            "Speed (m/min)": round(option["Speed (m/min)"], 1),
            f"Feed ({op_info.feed_unit})": round(
                option[f"Feed ({op_info.feed_unit})"], 4
            ),
            "Depth ap (mm)": round(option["Depth (mm)"], 3),
            "Tool Life (min)": round(option["Measured tool life (min)"], 1),
            "MRR (cm³/min)": round(option["Calculated MRR (cm³/min)"], 2),
            "Life Gain (%)": "No measured current-setting baseline",
            "Productivity Change (%)": (
                f"{(option['Calculated MRR (cm³/min)'] / max(pred.mrr_cm3_min, 1e-9) - 1.0) * 100.0:+.1f}%"
            ),
        } for option in verified_trial_options])
        summary_df = pd.concat([summary_df, trial_option_rows], ignore_index=True)

    st.markdown(summary_df.to_html(index=False, escape=False), unsafe_allow_html=True)

    # Download CSV
    csv_data = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Machining Optimization Report (CSV)",
        data=csv_data,
        file_name=f"tool_wear_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    pdf_data = _build_pdf_report(
        summary_df, pred, calibration_fit, research_report_evidence
    )
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
