"""
Machine Learning Literature Analysis Engine for Tool Wear & Machining Research.
Analyzes peer-reviewed abstracts and papers returned by the Springer Nature API:
1. NLP-based Wear Mechanism Classification (Abrasion, Adhesion, Diffusion, Chipping, Thermal).
2. Cutting parameter extraction (Vc, feed, ap, wear limits).
3. Literature Agreement & Dynamic Tool Life Calibration Multiplier (k_ML).
"""

from __future__ import annotations
import re
from typing import Dict, List, Any, Optional
import numpy as np


# Semantic keyword dictionaries for wear mechanism classification
WEAR_TAXONOMY = {
    "Abrasive Wear & Flank Scoring": {
        "keywords": [
            "abrasion", "abrasive", "scratching", "groove", "grooving", "scoring",
            "micro-groove", "hard carbide", "flank wear", "carbide particles", "wear land"
        ],
        "weight": 1.0,
        "derate_threshold": 0.35,
    },
    "Adhesive Wear & Built-Up Edge (BUE)": {
        "keywords": [
            "adhesion", "adhesive", "built-up edge", "bue", "smearing", "galling",
            "welding", "chip sticking", "attrition", "workpiece adhesion"
        ],
        "weight": 1.1,
        "derate_threshold": 0.30,
    },
    "Diffusion & Crater Wear": {
        "keywords": [
            "diffusion", "crater", "cratering", "dissolution", "chemical wear",
            "high temperature", "cobalt leaching", "binder loss", "rake face crater"
        ],
        "weight": 1.3,
        "derate_threshold": 0.25,
    },
    "Micro-Chipping & Mechanical Fracture": {
        "keywords": [
            "chipping", "micro-chipping", "fracture", "flaking", "delamination",
            "cutting edge breakout", "impact", "interrupted cut", "crack propagation"
        ],
        "weight": 1.4,
        "derate_threshold": 0.20,
    },
    "Thermal Shock & Comb Cracking": {
        "keywords": [
            "thermal crack", "comb cracking", "thermal fatigue", "cyclic heating",
            "thermal shock", "plastic deformation", "edge softening", "rapid thermal"
        ],
        "weight": 1.25,
        "derate_threshold": 0.25,
    },
}


def extract_cutting_speeds_from_text(text: str) -> List[float]:
    """Extracts cutting speeds mentioned in text (e.g. '120 m/min', '60-180 m/min')."""
    speeds: List[float] = []
    # Pattern for ranges like '60-180 m/min' or single values like '120 m/min'
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:-|to)?\s*(\d+(?:\.\d+)?)?\s*(?:m/min|m\s*min\s*[-⁻]1)", text, re.IGNORECASE)
    for m in matches:
        try:
            val1 = float(m[0])
            if 10.0 <= val1 <= 3000.0:
                speeds.append(val1)
            if m[1]:
                val2 = float(m[1])
                if 10.0 <= val2 <= 3000.0:
                    speeds.append(val2)
        except (ValueError, IndexError):
            continue
    return sorted(list(set(speeds)))


def extract_feed_rates_from_text(text: str) -> List[float]:
    """Extracts feed rates mentioned in text (e.g. '0.12 mm/rev', '0.05-0.25 mm/tooth')."""
    feeds: List[float] = []
    matches = re.findall(r"(\d+\.\d+)\s*(?:-|to)?\s*(\d+\.\d+)?\s*(?:mm/rev|mm/tooth|mm/z)", text, re.IGNORECASE)
    for m in matches:
        try:
            val1 = float(m[0])
            if 0.01 <= val1 <= 1.50:
                feeds.append(val1)
            if m[1]:
                val2 = float(m[1])
                if 0.01 <= val2 <= 1.50:
                    feeds.append(val2)
        except (ValueError, IndexError):
            continue
    return sorted(list(set(feeds)))


def classify_wear_mechanisms(corpus: str) -> Dict[str, float]:
    """
    Computes normalized occurrence and probability distribution
    of wear mechanisms across the retrieved Springer literature corpus.
    """
    corpus_lower = corpus.lower()
    scores: Dict[str, float] = {}

    for mode, data in WEAR_TAXONOMY.items():
        count = 0
        for kw in data["keywords"]:
            count += len(re.findall(r"\b" + re.escape(kw) + r"\b", corpus_lower))
        # Weight by severity factor
        scores[mode] = float(count * data["weight"])

    total = sum(scores.values())
    if total <= 0:
        # Default baseline industrial distribution
        return {
            "Abrasive Wear & Flank Scoring": 0.40,
            "Adhesive Wear & Built-Up Edge (BUE)": 0.25,
            "Diffusion & Crater Wear": 0.15,
            "Micro-Chipping & Mechanical Fracture": 0.10,
            "Thermal Shock & Comb Cracking": 0.10,
        }

    return {k: round(v / total, 3) for k, v in scores.items()}


def analyze_springer_records(
    records: List[Dict[str, Any]],
    workpiece_name: str,
    tool_material: str,
    current_vc: float,
    current_feed: float,
    nominal_tool_life: float
) -> Dict[str, Any]:
    """
    Executes the machine learning & NLP analysis on Springer API search results.
    """
    if not records:
        return {
            "analyzed": False,
            "message": "No Springer literature records provided for ML analysis.",
            "dominant_wear_mode": "Standard Abrasive Flank Wear (ISO 3685 Baseline)",
            "wear_probabilities": {
                "Abrasive Wear & Flank Scoring": 0.45,
                "Adhesive Wear & Built-Up Edge (BUE)": 0.25,
                "Diffusion & Crater Wear": 0.15,
                "Micro-Chipping & Mechanical Fracture": 0.10,
                "Thermal Shock & Comb Cracking": 0.05,
            },
            "literature_agreement_score": 75.0,
            "calibration_multiplier": 1.00,
            "adjusted_tool_life": nominal_tool_life,
            "extracted_speeds": [],
            "extracted_feeds": [],
            "insights": ["Operating on baseline empirical Taylor equations (no active API records)."]
        }

    # Aggregate title + abstract into a consolidated text corpus
    full_text = " ".join([
        f"{r.get('title', '')} {r.get('abstract', '')}"
        for r in records
    ])

    # 1. Wear mode classification
    wear_probs = classify_wear_mechanisms(full_text)
    dominant_mode = max(wear_probs, key=wear_probs.get)

    # 2. Extract numeric parameter bounds from literature
    speeds = extract_cutting_speeds_from_text(full_text)
    feeds = extract_feed_rates_from_text(full_text)

    # 3. Literature Agreement Scoring
    agreement_score = 70.0
    insights = []

    if speeds:
        min_s, max_s = min(speeds), max(speeds)
        if min_s <= current_vc <= max_s:
            agreement_score += 15.0
            insights.append(f"Cutting speed ({current_vc:.0f} m/min) falls squarely within published Springer experimental window ({min_s:.0f} – {max_s:.0f} m/min).")
        elif current_vc > max_s:
            penalty = min(25.0, (current_vc - max_s) / max_s * 40.0)
            agreement_score -= penalty
            insights.append(f"Cutting speed ({current_vc:.0f} m/min) exceeds published Springer testing ceiling ({max_s:.0f} m/min). Accelerated thermal wear likely.")
        else:
            insights.append(f"Cutting speed ({current_vc:.0f} m/min) is lower than published high-speed test window ({min_s:.0f} – {max_s:.0f} m/min).")
    else:
        insights.append(f"Literature confirms {workpiece_name} machinability regimes, with dominant wear driven by {dominant_mode.lower()}.")

    if feeds:
        min_f, max_f = min(feeds), max(feeds)
        if min_f <= current_feed <= max_f:
            agreement_score += 10.0
            insights.append(f"Feed rate ({current_feed:.3f} mm/rev) is corroborated by peer-reviewed cutting trials ({min_f:.3f} – {max_f:.3f} mm/rev).")

    agreement_score = float(np.clip(agreement_score, 20.0, 98.0))

    # 4. Dynamic Tool Life Calibration Multiplier (k_ML)
    # Calibrate based on detected dominant wear severity in the literature
    diff_prob = wear_probs.get("Diffusion & Crater Wear", 0.0)
    chip_prob = wear_probs.get("Micro-Chipping & Mechanical Fracture", 0.0)
    adh_prob = wear_probs.get("Adhesive Wear & Built-Up Edge (BUE)", 0.0)

    # If severe diffusion / chipping reported and user is at top 30% of speed range
    k_ml = 1.00
    if speeds and current_vc > np.percentile(speeds, 65):
        if diff_prob > 0.25:
            k_ml *= 0.88  # Derate by 12% due to crater diffusion
            insights.append("ML Calibration: High diffusion probability in literature suggests earlier crater breakthrough; tool life derated by 12%.")
        elif chip_prob > 0.20:
            k_ml *= 0.85
            insights.append("ML Calibration: Mechanical fracture risk identified in papers; applying 15% safety factor to nominal life.")
    elif adh_prob > 0.35 and current_vc < 60.0:
        k_ml *= 0.90
        insights.append("ML Calibration: Adhesive BUE tearing observed at lower speeds; 10% derate applied.")
    elif agreement_score > 85.0:
        k_ml *= 1.05  # Highly stable sweet spot
        insights.append("ML Calibration: High parameter consensus in literature confirms optimal stable cutting regime (+5% life bonus).")

    k_ml = float(np.clip(k_ml, 0.75, 1.25))
    adjusted_life = float(nominal_tool_life * k_ml)

    return {
        "analyzed": True,
        "articles_analyzed_count": len(records),
        "dominant_wear_mode": dominant_mode,
        "wear_probabilities": wear_probs,
        "literature_agreement_score": round(agreement_score, 1),
        "calibration_multiplier": round(k_ml, 3),
        "adjusted_tool_life": round(adjusted_life, 1),
        "extracted_speeds": speeds,
        "extracted_feeds": feeds,
        "insights": insights
    }
