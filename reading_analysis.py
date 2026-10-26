"""Import machining readings and prepare a concise, shareable batch report."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from io import BytesIO, StringIO
from typing import Callable

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from xml.sax.saxutils import escape

REQUIRED_FIELDS = ("cutting_speed_m_min", "feed_rate_mm", "depth_of_cut_mm")
OPTIONAL_FIELDS = ("elapsed_time_min", "overhang_ratio", "is_roughing", "measured_tool_life_min")
_ALIASES = {
    "cutting_speed_m_min": ("cutting_speed_m_min", "cutting speed", "cutting_speed", "surface speed", "speed", "vc", "vc m min"),
    "spindle_speed_rpm": ("spindle_speed_rpm", "spindle speed", "rpm", "n rpm", "n-rpm"),
    "feed_rate_mm": ("feed_rate_mm", "feed", "feed rate", "feed_rate", "feed mm rev", "feed mm tooth", "f"),
    "depth_of_cut_mm": ("depth_of_cut_mm", "depth", "depth of cut", "depth_of_cut", "ap", "doc"),
    "elapsed_time_min": ("elapsed time", "time", "cutting time", "runtime", "time min"),
    "overhang_ratio": ("overhang", "overhang ratio", "l d", "l/d"),
    "is_roughing": ("roughing", "is roughing", "operation mode"),
    "measured_tool_life_min": ("measured life", "tool life", "actual life", "life min"),
}


def _normalise(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def load_readings(filename: str, payload: bytes) -> pd.DataFrame:
    """Load tabular files plus text-table PDFs and screenshots/scans via OCR."""
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    stream = BytesIO(payload)
    if suffix == "csv":
        return pd.read_csv(stream)
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(stream)
    if suffix == "json":
        raw = json.loads(payload.decode("utf-8-sig"))
        if isinstance(raw, dict):
            raw = raw.get("readings", raw.get("data", raw))
        return pd.DataFrame(raw)
    if suffix in {"txt", "tsv"}:
        return pd.read_csv(stream, sep=None, engine="python")
    if suffix == "pdf":
        try:
            import pdfplumber
            with pdfplumber.open(stream) as document:
                tables = [table for page in document.pages for table in page.extract_tables() if table]
                if tables:
                    frames = []
                    for table in tables:
                        header, *rows = table
                        if header and rows:
                            frames.append(pd.DataFrame(rows, columns=header))
                    if frames:
                        return pd.concat(frames, ignore_index=True)
                text = "\n".join(page.extract_text() or "" for page in document.pages)
        except Exception as exc:
            raise ValueError(f"Could not extract a table from this PDF: {exc}") from exc
        try:
            return _read_extracted_text(text, "PDF")
        except ValueError:
            # Scanned PDFs have no text layer. Render the pages and use the
            # locally installed OCR engine, never a cloud service.
            with pdfplumber.open(BytesIO(payload)) as document:
                frames = []
                for page in document.pages:
                    frames.append(_ocr_image(page.to_image(resolution=220).original))
            return pd.concat(frames, ignore_index=True)
    if suffix in {"jpg", "jpeg", "png"}:
        from PIL import Image
        return _ocr_image(Image.open(stream))
    raise ValueError("Use CSV, Excel, JSON, TSV/TXT, PDF, JPG, JPEG, or PNG.")


def manual_readings_template() -> pd.DataFrame:
    """An explicit safe fallback when protected devices block document extraction."""
    return pd.DataFrame(columns=[
        "cutting_speed_m_min", "feed_rate_mm", "depth_of_cut_mm",
        "elapsed_time_min", "overhang_ratio", "is_roughing", "measured_tool_life_min",
    ])


def _ocr_image(image) -> pd.DataFrame:
    """Read a machining table using the locally installed Tesseract executable."""
    try:
        import pytesseract
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        installed = os.path.join(program_files, "Tesseract-OCR", "tesseract.exe")
        executable = installed if os.path.exists(installed) else shutil.which("tesseract")
        if not executable:
            raise RuntimeError("Tesseract OCR is not installed.")
        pytesseract.pytesseract.tesseract_cmd = executable
        prepared = _prepare_table_for_ocr(image)
        text = pytesseract.image_to_string(prepared, config="--psm 4")
    except Exception as exc:
        raise ValueError(f"Could not read this image automatically: {exc}") from exc
    return _read_ocr_table(text)


def _prepare_table_for_ocr(image):
    """Crop a light table from a screenshot, remove ruling, then enlarge it for OCR."""
    from PIL import Image

    gray = np.asarray(image.convert("L")).copy()
    bright = gray > 220

    def merged_runs(active, maximum_gap=6):
        raw, start = [], None
        for index, value in enumerate(active):
            if value and start is None:
                start = index
            if start is not None and (not value or index == len(active) - 1):
                raw.append((start, index if not value else index + 1))
                start = None
        merged = []
        for run in raw:
            if merged and run[0] - merged[-1][1] <= maximum_gap:
                merged[-1] = (merged[-1][0], run[1])
            else:
                merged.append(run)
        return merged

    row_runs = merged_runs(bright.sum(axis=1) > max(40, bright.shape[1] * 0.05))
    if row_runs:
        y0, y1 = max(row_runs, key=lambda run: run[1] - run[0])
        column_runs = merged_runs(bright[y0:y1].sum(axis=0) > max(30, (y1 - y0) * 0.05))
        if column_runs:
            x0, x1 = max(column_runs, key=lambda run: run[1] - run[0])
            gray = gray[y0:y1, x0:x1]

    # Remove long horizontal/vertical rules while preserving the text inside cells.
    ink = gray < 100
    for index in np.where(ink.mean(axis=1) > 0.55)[0]:
        gray[max(0, index - 1):min(gray.shape[0], index + 2), :] = 255
    for index in np.where(ink.mean(axis=0) > 0.55)[0]:
        gray[:, max(0, index - 1):min(gray.shape[1], index + 2)] = 255
    rendered = Image.fromarray(gray)
    return rendered.resize((rendered.width * 4, rendered.height * 4), Image.Resampling.LANCZOS)


def _read_ocr_table(text: str) -> pd.DataFrame:
    """Convert OCR lines from common machining experiment tables into model inputs."""
    numeric_rows = []
    for line in (text or "").splitlines():
        values = [float(value) for value in re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", line)]
        if len(values) >= 3:
            numeric_rows.append(values)
    if not numeric_rows:
        raise ValueError("No numeric machining-reading rows were found in the image.")

    readings = []
    for row in numeric_rows:
        # Ignore the experiment number and locate the first plausible cutting
        # speed. This tolerates OCR dropping/misreading the experiment index.
        for index in range(len(row) - 2):
            speed, feed, depth = row[index:index + 3]
            if 50 <= speed <= 100000 and 0 < feed <= 100 and 0 < depth <= 100:
                readings.append({
                    "spindle_speed_rpm": speed,
                    "feed_rate_mm": feed,
                    "depth_of_cut_mm": depth,
                })
                break
    if not readings:
        raise ValueError("OCR found numbers but could not identify speed, feed, and depth columns.")
    return pd.DataFrame(readings)


def _read_extracted_text(text: str, source: str) -> pd.DataFrame:
    """Parse a delimited or visibly column-aligned table extracted from a document."""
    if not text or not text.strip():
        raise ValueError(f"The {source} does not contain readable text or a table.")
    for separator in (None, r"\s{2,}"):
        try:
            candidate = pd.read_csv(StringIO(text), sep=separator, engine="python")
            if len(candidate.columns) >= 3:
                return candidate
        except (pd.errors.ParserError, UnicodeDecodeError):
            continue
    raise ValueError(
        f"A readable table was not found in the {source}. Use a clear, horizontal table "
        "with a header row, or upload the original CSV/Excel file for best accuracy."
    )


def standardize_readings(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Map reasonable user column labels into the reading schema and validate values."""
    mapped: dict[str, str] = {}
    available = {_normalise(column): str(column) for column in data.columns}
    for target, names in _ALIASES.items():
        for name in names:
            if _normalise(name) in available:
                mapped[target] = available[_normalise(name)]
                break
    missing = [field for field in REQUIRED_FIELDS if field not in mapped]
    if "cutting_speed_m_min" in missing and "spindle_speed_rpm" in mapped:
        missing.remove("cutting_speed_m_min")
    if missing:
        friendly = ", ".join(field.replace("_", " ") for field in missing)
        raise ValueError(f"Missing required readings: {friendly}. Add columns such as Vc/speed, feed, and ap/depth.")
    result = pd.DataFrame({field: data[source] for field, source in mapped.items()})
    for field in set(REQUIRED_FIELDS + OPTIONAL_FIELDS) & set(result.columns):
        if field != "is_roughing":
            result[field] = pd.to_numeric(result[field], errors="coerce")
    if "is_roughing" in result:
        result["is_roughing"] = result["is_roughing"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "rough", "roughing"})
    for field in REQUIRED_FIELDS:
        if field not in result.columns:
            continue
        result = result[result[field].notna() & (result[field] > 0)]
    if result.empty:
        raise ValueError("No usable rows remain: speed, feed, and depth must be positive numbers.")
    return result.reset_index(drop=True), mapped


def analyze_readings(readings: pd.DataFrame, predict_ml: Callable[[float, float, float, float | None, bool], object], fallback_overhang: float | None, fallback_roughing: bool) -> pd.DataFrame:
    """Score each reading with the dashboard's already-trained hybrid ML model."""
    results = readings.copy()
    rows = []
    for row in results.itertuples(index=False):
        values = row._asdict()
        overhang = values.get("overhang_ratio")
        overhang = fallback_overhang if pd.isna(overhang) else float(overhang)
        roughing = bool(values.get("is_roughing", fallback_roughing)) if "is_roughing" in results else fallback_roughing
        estimate = predict_ml(float(values["cutting_speed_m_min"]), float(values["feed_rate_mm"]), float(values["depth_of_cut_mm"]), overhang, roughing)
        elapsed = values.get("elapsed_time_min", 0.0)
        elapsed = 0.0 if pd.isna(elapsed) else max(0.0, float(elapsed))
        measured = values.get("measured_tool_life_min", np.nan)
        life = float(estimate.life_min)
        rows.append({"Predicted life (min)": round(life, 1), "80% low (min)": round(float(estimate.band_low_min), 1), "80% high (min)": round(float(estimate.band_high_min), 1), "Remaining life (min)": round(max(0.0, life - elapsed), 1), "Status": "Replace / inspect" if elapsed >= life * 0.85 else "Monitor" if elapsed >= life * 0.65 else "Healthy", "Measured-life error (%)": (round((life - float(measured)) / float(measured) * 100, 1) if pd.notna(measured) and float(measured) > 0 else np.nan)})
    return pd.concat([results.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def build_batch_pdf(report: pd.DataFrame, source_name: str, setup_label: str) -> bytes:
    """Build a polished portable report for uploaded readings."""
    output = BytesIO()
    document = SimpleDocTemplate(output, pagesize=letter, leftMargin=.55*inch, rightMargin=.55*inch, topMargin=.5*inch, bottomMargin=.5*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BatchTitle", parent=styles["Title"], textColor=colors.HexColor("#0F3D56"), fontSize=18, leading=22))
    styles.add(ParagraphStyle(name="BatchHeading", parent=styles["Heading2"], textColor=colors.HexColor("#137B8C"), fontSize=11, spaceBefore=10, spaceAfter=5))
    styles.add(ParagraphStyle(name="BatchBody", parent=styles["BodyText"], fontSize=8, leading=10))
    para = lambda text, style="BatchBody": Paragraph(escape(str(text)), styles[style])
    critical = int((report["Status"] == "Replace / inspect").sum())
    story = [Paragraph("Machining Readings Analysis", styles["BatchTitle"]), para(f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')} | Source: {source_name}"), para(setup_label), Spacer(1, 8), Paragraph("Executive summary", styles["BatchHeading"])]
    summary = [["Readings analyzed", str(len(report))], ["Average predicted life", f"{report['Predicted life (min)'].mean():.1f} min"], ["Inspection / replacement alerts", str(critical)], ["Average remaining life", f"{report['Remaining life (min)'].mean():.1f} min"]]
    table = Table([[para(a), para(b)] for a, b in summary], colWidths=[2.4*inch, 4.3*inch])
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (0,-1), colors.HexColor("#E6F4F6")), ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#B8C8D0")), ("PADDING", (0,0), (-1,-1), 5)]))
    story.extend([table, Paragraph("ML assessment by reading", styles["BatchHeading"])])
    columns = ["cutting_speed_m_min", "feed_rate_mm", "depth_of_cut_mm", "Predicted life (min)", "Remaining life (min)", "Status"]
    headers = ["Vc", "Feed", "ap", "Pred. life", "Remaining", "Action"]
    table_rows = [[para(x) for x in headers]]
    for _, record in report.loc[:, columns].iterrows():
        table_rows.append([para(f"{record.iloc[0]:.1f}"), para(f"{record.iloc[1]:.3f}"), para(f"{record.iloc[2]:.2f}"), para(record.iloc[3]), para(record.iloc[4]), para(record.iloc[5])])
    detail = Table(table_rows, colWidths=[.72*inch, .72*inch, .62*inch, 1.0*inch, 1.1*inch, 1.45*inch], repeatRows=1)
    detail.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0F3D56")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5FAFB")]), ("GRID", (0,0), (-1,-1), .25, colors.HexColor("#C7D6DC")), ("PADDING", (0,0), (-1,-1), 4), ("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
    story.extend([detail, Spacer(1, 7), para("Method note: predictions use the dashboard's active hybrid physics + ML model. They are decision-support estimates and should be validated against tool-maker guidance and controlled shop trials.")])
    document.build(story)
    return output.getvalue()
