"""
Portrait PDF Report Generator for the Tool Wear Forecasting Dashboard.
Produces a self-contained, printable A4-portrait machining specification report
summarizing current prediction, optimization strategies, and Springer evidence.
Uses reportlab (pure-Python, no browser/Chrome dependency) so it works fully offline.
"""

from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
)
from reportlab.lib.enums import TA_CENTER


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle(
        name="ReportTitle", parent=ss["Title"], fontSize=18, textColor=colors.HexColor("#0F172A"),
        spaceAfter=2,
    ))
    ss.add(ParagraphStyle(
        name="ReportSubtitle", parent=ss["Normal"], fontSize=9.5, textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER, spaceAfter=10,
    ))
    ss.add(ParagraphStyle(
        name="SectionHeader", parent=ss["Heading2"], fontSize=12.5, textColor=colors.HexColor("#0F172A"),
        spaceBefore=12, spaceAfter=4, borderPadding=0,
    ))
    ss.add(ParagraphStyle(
        name="BodySmall", parent=ss["Normal"], fontSize=8.8, textColor=colors.HexColor("#334155"),
        leading=12,
    ))
    ss.add(ParagraphStyle(
        name="Caption", parent=ss["Normal"], fontSize=7.8, textColor=colors.HexColor("#64748B"),
        leading=10,
    ))
    return ss


def _kpi_table(pred, styles):
    data = [
        ["Predicted Tool Life", "Remaining Useful Life", "Flank Wear (VB)", "MRR", "Springer Confidence"],
        [
            f"{pred.tool_life_minutes:.1f} min",
            f"{pred.rul_minutes:.1f} min ({pred.rul_percentage:.0f}%)",
            f"{pred.current_vb_mm:.3f} mm / {pred.vb_threshold_mm:.2f} mm",
            f"{pred.mrr_cm3_min:.1f} cm³/min",
            f"{pred.confidence_score:.0f}% ({pred.confidence_label.split('(')[0].strip()})",
        ],
    ]
    t = Table(data, colWidths=[33 * mm] * 5)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _spec_table(pred):
    rows = [
        ["Parameter", "Value"],
        ["Workpiece Material", pred.pairing.workpiece_name],
        ["Tool Material", pred.pairing.tool_material],
        ["Coating / Prep", pred.pairing.coating],
        ["Machine Tool", pred.machine_name],
        ["Coolant / Lubrication", pred.coolant_name],
        ["Cutting Speed (Vc)", f"{pred.vc:.1f} m/min"],
        ["Feed Rate (f)", f"{pred.feed:.3f} mm/rev"],
        ["Depth of Cut (ap)", f"{pred.ap:.2f} mm"],
        ["Mode", "Roughing (VB=0.5-0.6mm)" if pred.is_roughing else "Finishing (VB=0.3mm)"],
        ["Elapsed Cutting Time", f"{pred.current_time_min:.1f} min"],
    ]
    t = Table(rows, colWidths=[60 * mm, 100 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _strategy_table(opt_report):
    strategies = [
        ("Current Setting", opt_report.current_vc, opt_report.current_feed, opt_report.current_ap,
         opt_report.current_tool_life, opt_report.current_mrr, 0.0, 0.0),
        ("Balanced (Recommended)", opt_report.balanced_strategy.vc, opt_report.balanced_strategy.feed,
         opt_report.balanced_strategy.ap, opt_report.balanced_strategy.tool_life_min,
         opt_report.balanced_strategy.mrr_cm3_min, opt_report.balanced_strategy.tool_life_gain_pct,
         opt_report.balanced_strategy.mrr_change_pct),
        ("Max Life (Lights-Out)", opt_report.max_life_strategy.vc, opt_report.max_life_strategy.feed,
         opt_report.max_life_strategy.ap, opt_report.max_life_strategy.tool_life_min,
         opt_report.max_life_strategy.mrr_cm3_min, opt_report.max_life_strategy.tool_life_gain_pct,
         opt_report.max_life_strategy.mrr_change_pct),
        ("High Efficiency", opt_report.high_efficiency_strategy.vc, opt_report.high_efficiency_strategy.feed,
         opt_report.high_efficiency_strategy.ap, opt_report.high_efficiency_strategy.tool_life_min,
         opt_report.high_efficiency_strategy.mrr_cm3_min, opt_report.high_efficiency_strategy.tool_life_gain_pct,
         opt_report.high_efficiency_strategy.mrr_change_pct),
    ]
    header = ["Scenario", "Vc (m/min)", "f (mm/rev)", "ap (mm)", "Life (min)", "MRR (cm³/min)", "Life Δ%", "MRR Δ%"]
    rows = [header]
    for name, vc, f, ap, life, mrr, life_pct, mrr_pct in strategies:
        rows.append([
            name, f"{vc:.1f}", f"{f:.3f}", f"{ap:.2f}", f"{life:.1f}", f"{mrr:.2f}",
            f"{life_pct:+.1f}%" if name != "Current Setting" else "—",
            f"{mrr_pct:+.1f}%" if name != "Current Setting" else "—",
        ])
    t = Table(rows, colWidths=[32 * mm, 18 * mm, 18 * mm, 15 * mm, 18 * mm, 24 * mm, 16 * mm, 16 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D9488")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.3),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def generate_pdf_report(pred, opt_report, evidence: dict) -> bytes:
    """
    Builds a portrait A4 PDF report summarizing the current prediction,
    optimization strategies, and supporting Springer / standards evidence.
    Returns the PDF as raw bytes, suitable for st.download_button.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=16 * mm, bottomMargin=14 * mm, leftMargin=16 * mm, rightMargin=16 * mm,
        title="Tool Wear Forecasting Report",
    )
    styles = _styles()
    story = []

    story.append(Paragraph("Machining Tool Wear &amp; Life Forecasting Report", styles["ReportTitle"]))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; "
        f"Grounded in peer-reviewed Springer IJAMT and related empirical literature",
        styles["ReportSubtitle"]
    ))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#CBD5E1"), thickness=1))

    story.append(Paragraph("Key Performance Indicators", styles["SectionHeader"]))
    story.append(_kpi_table(pred, styles))

    story.append(Paragraph("Machining Specification", styles["SectionHeader"]))
    story.append(_spec_table(pred))

    story.append(Paragraph("Wear Minimization — Scenario Comparison", styles["SectionHeader"]))
    story.append(_strategy_table(opt_report))

    story.append(Paragraph("Springer Research Reference & Evidence", styles["SectionHeader"]))
    primary = evidence["primary"]
    lit_tag = "Peer-reviewed empirical study" if evidence["is_literature_backed"] else "Synthesized / handbook-derived estimate"
    story.append(Paragraph(f"<b>Evidence basis:</b> {lit_tag}", styles["BodySmall"]))
    story.append(Paragraph(f"<b>{primary.title}</b>", styles["BodySmall"]))
    story.append(Paragraph(f"{primary.authors} — {primary.journal}, {primary.volume_issue} ({primary.year})", styles["Caption"]))
    story.append(Paragraph(f"DOI: {primary.doi}", styles["Caption"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Key findings:</b> {primary.key_findings}", styles["BodySmall"]))

    if evidence["related_studies"]:
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Cross-validated against related curated Springer pairings (same ISO group):</b>", styles["BodySmall"]))
        for key, cand in evidence["related_studies"]:
            story.append(Paragraph(
                f"• {key} — {cand.springer_ref.authors.split(',')[0]} et al. ({cand.springer_ref.year}), "
                f"DOI: {cand.springer_ref.doi}", styles["Caption"]
            ))

    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Governing standards & handbooks:</b>", styles["BodySmall"]))
    for std in evidence["standards"]:
        story.append(Paragraph(f"• <b>{std.code}</b> — {std.title} ({std.body}): {std.relevance}", styles["Caption"]))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#CBD5E1"), thickness=0.6))
    story.append(Paragraph(
        "This report is generated for engineering reference only. Predictions outside the verified "
        "Springer experimental boundary are extrapolated — see Confidence Score above.",
        styles["Caption"]
    ))

    doc.build(story)
    return buf.getvalue()
