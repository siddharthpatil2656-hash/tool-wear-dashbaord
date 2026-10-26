"""Guarded S/F-only preview helper for a single metric G-code block."""

import math
import re


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_S_WORD = re.compile(rf"(?i)(?<![A-Z])S\s*{_NUMBER}")
_F_WORD = re.compile(rf"(?i)(?<![A-Z])F\s*{_NUMBER}")
_G_WORD = re.compile(r"(?i)(?<![A-Z])G\s*(\d+(?:\.\d*)?)")
_PAREN_COMMENT = re.compile(r"\([^)]*\)")


def cutting_targets(
    *,
    cutting_speed_m_min: float,
    diameter_mm: float,
    feed_value: float,
    feed_unit: str,
    teeth: int,
    feed_mode: str,
) -> tuple[int, float]:
    """Convert surface speed and dashboard feed to metric spindle RPM and G-code F."""
    values = (cutting_speed_m_min, diameter_mm, feed_value)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("G-code target inputs must be finite numbers.")
    if cutting_speed_m_min <= 0 or diameter_mm <= 0 or feed_value <= 0:
        raise ValueError("Speed, diameter, and feed must be greater than zero.")
    if feed_mode not in {"G94", "G95"}:
        raise ValueError("Choose G94 (units/min) or G95 (units/rev).")
    if teeth < 1:
        raise ValueError("Tool tooth count must be at least one.")
    if feed_unit not in {"mm/rev", "mm/tooth"}:
        raise ValueError(f"Unsupported dashboard feed unit: {feed_unit}")

    rpm = round(1000.0 * cutting_speed_m_min / (math.pi * diameter_mm))
    feed_per_revolution = feed_value * teeth if feed_unit == "mm/tooth" else feed_value
    feed_command = feed_per_revolution if feed_mode == "G95" else feed_per_revolution * rpm
    return rpm, feed_command


def _code_without_comments(line: str) -> str:
    code = line.split(";", 1)[0]
    return _PAREN_COMMENT.sub("", code)


def update_spindle_and_feed(
    line: str,
    spindle_rpm: float,
    feed_command: float,
    feed_mode: str,
) -> str:
    """Replace or append S/F in one line, preserving comments and all other words."""
    if not line or "\n" in line or "\r" in line:
        raise ValueError("Enter exactly one non-empty G-code block.")
    if not math.isfinite(spindle_rpm) or not math.isfinite(feed_command):
        raise ValueError("S and F targets must be finite numbers.")
    if spindle_rpm <= 0 or feed_command <= 0:
        raise ValueError("S and F targets must be greater than zero.")

    code = _code_without_comments(line)
    if not code.strip():
        raise ValueError("The line contains no G-code outside its comments.")
    g_codes = [int(float(value)) for value in _G_WORD.findall(code)]
    if 20 in g_codes:
        raise ValueError("G20 inch mode is not supported; use a verified G21 metric program.")
    if 96 in g_codes:
        raise ValueError("G96 constant-surface-speed mode is not supported by this S-word preview.")
    if 1 not in g_codes or any(code_number in {0, 2, 3} for code_number in g_codes):
        raise ValueError("Include a G01 feed-motion block only; rapid moves, arcs, canned cycles, and modal-only lines are not rewritten.")
    if any(80 <= code_number <= 89 for code_number in g_codes):
        raise ValueError("Canned-cycle blocks are not rewritten.")
    if feed_mode not in {"G94", "G95"}:
        raise ValueError("Choose G94 (units/min) or G95 (units/rev).")
    explicit_feed_modes = {f"G{code_number}" for code_number in g_codes if code_number in {94, 95}}
    if explicit_feed_modes and explicit_feed_modes != {feed_mode}:
        raise ValueError("Selected feed mode conflicts with the G94/G95 word in this block.")

    for pattern, label in ((_S_WORD, "S"), (_F_WORD, "F")):
        if len(pattern.findall(code)) > 1:
            raise ValueError(f"The block has multiple {label} words; edit it manually before previewing.")

    comment_start = len(line)
    semicolon_start = line.find(";")
    if semicolon_start >= 0:
        comment_start = min(comment_start, semicolon_start)
    for match in _PAREN_COMMENT.finditer(line):
        comment_start = min(comment_start, match.start())
    code_part, comment_part = line[:comment_start], line[comment_start:]

    def replace_or_append(pattern: re.Pattern, label: str, value: float, digits: int) -> None:
        nonlocal code_part
        formatted_value = f"{value:.{digits}f}"
        if "." in formatted_value:
            formatted_value = formatted_value.rstrip("0").rstrip(".")
        replacement = f"{label}{formatted_value}"
        match = pattern.search(_code_without_comments(code_part))
        if match:
            code_part = code_part[:match.start()] + replacement + code_part[match.end():]
        else:
            code_part = code_part.rstrip() + f" {replacement}"

    replace_or_append(_S_WORD, "S", float(round(spindle_rpm)), 0)
    replace_or_append(_F_WORD, "F", feed_command, 3)
    if comment_part and not code_part.endswith((" ", "\t")):
        code_part += " "
    return code_part + comment_part
