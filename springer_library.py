"""Import and persist SpringerLink citation exports for offline evidence."""

import csv
from io import StringIO
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable


MAX_IMPORT_BYTES = 10 * 1024 * 1024
MAX_LIBRARY_RECORDS = 5000
REFERENCE_FIELDS = (
    "title",
    "authors",
    "publication",
    "date",
    "doi",
    "url",
    "abstract",
)

FIELD_ALIASES = {
    "title": {"title", "ti", "t1", "bt"},
    "authors": {"authors", "author", "au", "a1", "creators"},
    "publication": {"publication", "publicationname", "journal", "jo", "jf", "t2"},
    "date": {"date", "year", "py", "y1", "publicationdate", "publicationyear"},
    "doi": {"doi", "do", "digitalobjectidentifier"},
    "url": {"url", "link", "ur"},
    "abstract": {"abstract", "ab", "n2", "summary"},
}


def _clean(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _field_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _normalize_record(record: dict) -> dict[str, str]:
    normalized = {}
    for target, aliases in FIELD_ALIASES.items():
        value = ""
        for key, candidate in record.items():
            if _field_name(key) in aliases and candidate:
                if isinstance(candidate, (list, tuple)):
                    value = "; ".join(_clean(item) for item in candidate if item)
                else:
                    value = _clean(candidate)
                break
        normalized[target] = value
    normalized["doi"] = re.sub(
        r"^https?://(?:dx\.)?doi\.org/", "", normalized["doi"], flags=re.I
    ).strip()
    if normalized["doi"] and not normalized["url"]:
        normalized["url"] = f"https://doi.org/{normalized['doi']}"
    if not normalized["title"]:
        return {}
    return normalized


def _parse_ris(text: str) -> list[dict[str, str]]:
    records = []
    record = {}
    last_tag = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line[:1].isspace() and last_tag:
            record[last_tag] = f"{record.get(last_tag, '')} {line.strip()}".strip()
            continue
        match = re.match(r"^([A-Z0-9]{2})\s*-\s?(.*)$", line)
        if not match:
            continue
        tag, value = match.groups()
        if tag == "TY":
            record = {}
        elif tag == "ER":
            normalized = _normalize_record(record)
            if normalized:
                records.append(normalized)
            record = {}
        elif tag in {"AU", "A1"}:
            record.setdefault("authors", []).append(value.strip())
        else:
            record[tag] = value.strip()
        last_tag = tag
    normalized = _normalize_record(record)
    if normalized:
        records.append(normalized)
    return records


def _read_bib_value(text: str, start: int) -> tuple[str, int]:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text):
        return "", start
    if text[start] == "{":
        depth = 1
        cursor = start + 1
        value_start = cursor
        while cursor < len(text) and depth:
            if text[cursor] == "{" and (cursor == 0 or text[cursor - 1] != "\\"):
                depth += 1
            elif text[cursor] == "}" and (cursor == 0 or text[cursor - 1] != "\\"):
                depth -= 1
            cursor += 1
        if depth:
            raise ValueError("BibTeX entry has an unclosed brace.")
        return text[value_start:cursor - 1], cursor
    if text[start] == '"':
        cursor = start + 1
        value = []
        while cursor < len(text):
            char = text[cursor]
            if char == '"' and (cursor == 0 or text[cursor - 1] != "\\"):
                return "".join(value), cursor + 1
            value.append(char)
            cursor += 1
        raise ValueError("BibTeX entry has an unclosed quoted field.")
    cursor = start
    while cursor < len(text) and text[cursor] not in ",\r\n":
        cursor += 1
    return text[start:cursor].strip(), cursor


def _parse_bibtex(text: str) -> list[dict[str, str]]:
    records = []
    entry_start = re.compile(r"@\w+\s*[\{\(]", re.I)
    field_name = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
    position = 0
    while match := entry_start.search(text, position):
        cursor = match.end()
        depth = 1
        closing = ")" if text[match.end() - 1] == "(" else "}"
        body_start = cursor
        while cursor < len(text) and depth:
            if text[cursor] == text[match.end() - 1]:
                depth += 1
            elif text[cursor] == closing:
                depth -= 1
            cursor += 1
        if depth:
            raise ValueError("BibTeX entry has an unclosed record.")
        body = text[body_start:cursor - 1]
        comma = body.find(",")
        fields = {}
        if comma >= 0:
            index = comma + 1
            while index < len(body):
                while index < len(body) and body[index] in " \r\n\t,":
                    index += 1
                name_match = field_name.match(body, index)
                if not name_match:
                    break
                name = name_match.group()
                index = name_match.end()
                while index < len(body) and body[index].isspace():
                    index += 1
                if index >= len(body) or body[index] != "=":
                    raise ValueError(f"BibTeX field '{name}' is missing '='.")
                value, index = _read_bib_value(body, index + 1)
                fields[name] = value
        normalized = _normalize_record(fields)
        if normalized:
            if "author" in fields:
                normalized["authors"] = re.sub(
                    r"\s+and\s+", "; ", fields["author"], flags=re.I
                )
            records.append(normalized)
        position = cursor
    if not records and text.strip():
        raise ValueError("No BibTeX article records with titles were found.")
    return records


def _parse_csv(text: str) -> list[dict[str, str]]:
    try:
        rows = csv.DictReader(StringIO(text))
        if not rows.fieldnames:
            raise ValueError("CSV file must have a header row.")
        records = [_normalize_record(row) for row in rows]
    except csv.Error as exc:
        raise ValueError(f"Could not parse citation CSV: {exc}") from exc
    return [record for record in records if record]


def parse_reference_export(filename: str, content: bytes) -> list[dict[str, str]]:
    """Parse a RIS, BibTeX, or CSV citation export (never article full text)."""
    if len(content) > MAX_IMPORT_BYTES:
        raise ValueError("Citation export exceeds the 10 MB import limit.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = content.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise ValueError("Citation export must be UTF-8 or Windows-1252 text.") from exc
    extension = Path(filename).suffix.lower()
    if extension == ".ris":
        records = _parse_ris(text)
    elif extension in {".bib", ".bibtex"}:
        records = _parse_bibtex(text)
    elif extension == ".csv":
        records = _parse_csv(text)
    else:
        raise ValueError("Supported citation export formats are RIS, BibTeX, and CSV.")
    if not records:
        raise ValueError("No citation records with titles were found in the export.")
    return records


def merge_references(
    existing: Iterable[dict[str, str]],
    incoming: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    """Merge by DOI when available, otherwise by normalized title."""
    records = []
    seen = set()
    for record in [*existing, *incoming]:
        normalized = _normalize_record(record)
        if not normalized:
            continue
        key = normalized["doi"].lower() or re.sub(
            r"[^a-z0-9]", "", normalized["title"].lower()
        )
        if key in seen:
            continue
        seen.add(key)
        records.append(normalized)
        if len(records) > MAX_LIBRARY_RECORDS:
            raise ValueError(
                f"Offline reference library is limited to {MAX_LIBRARY_RECORDS} records."
            )
    return records


def load_reference_library(path: Path) -> list[dict[str, str]]:
    """Load the local citation library, reporting corrupt files explicitly."""
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read local reference library: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("Local reference library must contain a list of records.")
    if len(payload) > MAX_LIBRARY_RECORDS:
        raise ValueError(
            f"Local reference library exceeds the {MAX_LIBRARY_RECORDS}-record limit."
        )
    return merge_references([], (record for record in payload if isinstance(record, dict)))


def save_reference_library(path: Path, records: Iterable[dict[str, str]]) -> None:
    """Atomically persist citation metadata beside the local dashboard."""
    merged = merge_references([], records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(merged, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ValueError(f"Could not save local reference library: {exc}") from exc


def relevant_references(
    records: Iterable[dict[str, str]], query: str, limit: int = 20
) -> list[dict[str, str]]:
    """Rank local citations by query-term overlap, preserving all records separately."""
    terms = {
        term
        for term in re.findall(r"[a-z0-9]+", query.lower())
        if len(term) > 2
    }
    ranked = []
    for record in records:
        searchable = " ".join(
            str(record.get(field, "")) for field in ("title", "abstract", "publication")
        ).lower()
        words = set(re.findall(r"[a-z0-9]+", searchable))
        score = len(terms & words)
        if score:
            ranked.append((score, str(record.get("title", "")).lower(), record))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [record for _, _, record in ranked[:limit]]
