"""Import cutting-parameter readings (speed, feed, depth of cut, optional tool life)
from CSV, Excel, PDF tables, or images (OCR when Tesseract is installed)."""

from io import BytesIO
from pathlib import Path
import re

import numpy as np
import pandas as pd

SUPPORTED_TYPES = ["csv", "tsv", "txt", "xlsx", "xlsm", "xls", "pdf", "jpg", "jpeg", "png", "bmp", "tif", "tiff"]
ROLES = ("speed", "feed", "depth", "life")
CANONICAL = {
    "speed": "cutting_speed_m_min",
    "feed": "feed_rate_mm",
    "depth": "depth_of_cut_mm",
    "life": "measured_tool_life_min",
}
_PATTERNS = {
    "speed": r"(^|[^a-z])(vc|v|speed|cutting speed|rpm|sfm|m/min)([^a-z]|$)",
    "feed": r"(^|[^a-z])(f|fz|feed|feed rate|mm/rev|mm/tooth)([^a-z]|$)",
    "depth": r"(^|[^a-z])(ap|doc|depth|depth of cut|d\.o\.c)([^a-z]|$)",
    "life": r"(life|tool life|t\s*\(min|minutes|time to|wear time)",
}
_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _looks_numeric(value):
    return bool(re.fullmatch(r"\s*[-+]?\d+(?:[.,]\d+)?\s*", str(value)))


def _promote_header(df):
    """If the first row is text (and later rows are numeric), use it as the header."""
    if df.empty:
        return df
    first = df.iloc[0].astype(str)
    rest = df.iloc[1:]
    first_numeric = sum(_looks_numeric(v) for v in first)
    if first_numeric <= len(first) // 3 and len(rest):
        out = rest.copy()
        names = []
        for i, v in enumerate(first):
            v = re.sub(r"\s+", " ", str(v)).strip()
            names.append(v if v and v.lower() != "nan" else f"Column {i + 1}")
        out.columns = _dedupe(names)
        return out.reset_index(drop=True)
    df = df.copy()
    df.columns = [f"Column {i + 1}" for i in range(df.shape[1])]
    return df.reset_index(drop=True)


def _dedupe(names):
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n} ({seen[n]})")
        else:
            seen[n] = 0
            out.append(n)
    return out


def _tables_from_pdf(content):
    import pdfplumber

    frames = []
    with pdfplumber.open(BytesIO(content)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                rows = [[("" if c is None else str(c)) for c in row] for row in table if row]
                if len(rows) >= 2:
                    frames.append(pd.DataFrame(rows))
    if not frames:
        # No ruled tables: fall back to whitespace-separated numeric lines in the text layer.
        lines = []
        with pdfplumber.open(BytesIO(content)) as pdf:
            for page in pdf.pages:
                lines.extend((page.extract_text() or "").splitlines())
        return _frame_from_text_lines(lines)
    width = max(f.shape[1] for f in frames)
    frames = [f.reindex(columns=range(width)) for f in frames]
    return pd.concat(frames, ignore_index=True)


def _frame_from_text_lines(lines):
    rows = []
    for line in lines:
        tokens = _NUM.findall(line)
        if len(tokens) >= 3:
            rows.append([t.replace(",", ".") for t in tokens])
    if not rows:
        return pd.DataFrame()
    width = max(len(r) for r in rows)
    return pd.DataFrame([r + [""] * (width - len(r)) for r in rows])


def _tables_from_image(content):
    from PIL import Image
    import pytesseract

    image = Image.open(BytesIO(content)).convert("L")
    text = pytesseract.image_to_string(image, config="--psm 6")
    return _frame_from_text_lines(text.splitlines())


def parse_readings(filename, content):
    """Return (DataFrame, notes). The DataFrame may be empty if nothing could be read."""
    ext = Path(filename).suffix.lower().lstrip(".")
    notes = []
    if ext in ("csv", "tsv", "txt"):
        sep = "\t" if ext == "tsv" else None
        df = pd.read_csv(BytesIO(content), sep=sep, engine="python", header=None, dtype=str)
        df = _promote_header(df)
    elif ext in ("xlsx", "xlsm", "xls"):
        sheets = pd.read_excel(BytesIO(content), sheet_name=None, header=None, dtype=str)
        sheets = {k: v.dropna(how="all").dropna(axis=1, how="all") for k, v in sheets.items()}
        sheets = {k: v for k, v in sheets.items() if not v.empty}
        if not sheets:
            return pd.DataFrame(), ["The workbook has no data."]
        name, df = next(iter(sheets.items()))
        if len(sheets) > 1:
            notes.append(f"Workbook has {len(sheets)} sheets; using the first non-empty sheet '{name}'.")
        df = _promote_header(df.reset_index(drop=True))
    elif ext == "pdf":
        try:
            df = _tables_from_pdf(content)
        except ImportError:
            return pd.DataFrame(), ["PDF reading needs the 'pdfplumber' package (pip install pdfplumber)."]
        if df.empty:
            return df, ["No table or numeric rows were found in this PDF. If it is a scan, "
                        "photograph the page as an image or type the values into the table below."]
        df = _promote_header(df)
        notes.append("Values were extracted from the PDF automatically - check them against the original.")
    else:
        try:
            df = _tables_from_image(content)
        except ImportError:
            return pd.DataFrame(), ["Image reading needs 'pillow' and 'pytesseract' plus the Tesseract OCR program. "
                                    "Without it, type the values from the image into the table below."]
        except Exception as exc:  # TesseractNotFoundError etc.
            return pd.DataFrame(), [f"OCR is unavailable ({type(exc).__name__}). Install Tesseract OCR, "
                                    "or type the values from the image into the table below."]
        if df.empty:
            return df, ["No rows with three or more numbers were detected in the image."]
        df = _promote_header(df)
        notes.append("Values were read by OCR - OCR often misreads digits, so check every row.")
    df = df.dropna(how="all").reset_index(drop=True)
    return df, notes


def guess_columns(df):
    """Map roles -> column names using header text, then numeric column order."""
    mapping = {}
    used = set()
    for role in ROLES:
        for col in df.columns:
            if col in used:
                continue
            if re.search(_PATTERNS[role], str(col).lower()):
                mapping[role] = col
                used.add(col)
                break
    numeric_cols = [c for c in df.columns
                    if c not in used and pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce").notna().mean() > 0.6]
    for role in ROLES:
        if role not in mapping and numeric_cols and role != "life":
            mapping[role] = numeric_cols.pop(0)
        elif role == "life" and role not in mapping and numeric_cols:
            mapping[role] = numeric_cols.pop(0)
    return mapping


def clean_readings(df, mapping):
    """Build the canonical numeric frame; returns (clean_df, dropped_row_count)."""
    out = pd.DataFrame()
    for role, col in mapping.items():
        if col is None or col not in df.columns:
            continue
        out[CANONICAL[role]] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce"
        )
    required = [CANONICAL[r] for r in ("speed", "feed", "depth")]
    for col in required:
        if col not in out:
            return pd.DataFrame(columns=list(CANONICAL.values())), len(df)
    keep = out[required].notna().all(axis=1) & (out[required] > 0).all(axis=1)
    dropped = int((~keep).sum())
    out = out.loc[keep].reset_index(drop=True)
    life_col = CANONICAL["life"]
    if life_col in out:
        out.loc[out[life_col] <= 0, life_col] = np.nan
    else:
        out[life_col] = np.nan
    return out[list(CANONICAL.values())], dropped
