"""Check every bundled citation (DOI, title, year) against Crossref.

Run on a PC with internet access:   python verify_references.py
Writes reference_audit.csv with one row per database entry and a verdict:
  VERIFIED  - DOI exists and its registered title closely matches the bundled title
  MISMATCH  - DOI exists but belongs to a different paper
  NOT FOUND - DOI is not registered in Crossref
  ERROR     - network/API problem (retry)
A VERIFIED citation confirms the paper exists; it does NOT confirm the Taylor
constants in the record. Constants must be traced to the paper's own tables.
"""
import csv, difflib, json, re, time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from springer_database import SPRINGER_DATABASE


def norm(text):
    return re.sub(r"[^a-z0-9 ]", "", str(text).lower())


def lookup(doi):
    req = Request(f"https://api.crossref.org/works/{quote(doi)}",
                  headers={"User-Agent": "ToolWearDashboard/1.0 (reference audit)"})
    with urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))["message"]


rows = []
for key, p in SPRINGER_DATABASE.items():
    ref = p.springer_ref
    verdict, found_title, ratio = "ERROR", "", 0.0
    try:
        msg = lookup(ref.doi)
        found_title = (msg.get("title") or [""])[0]
        ratio = difflib.SequenceMatcher(None, norm(ref.title), norm(found_title)).ratio()
        verdict = "VERIFIED" if ratio >= 0.75 else "MISMATCH"
    except HTTPError as e:
        verdict = "NOT FOUND" if e.code == 404 else "ERROR"
    except (URLError, TimeoutError, OSError):
        verdict = "ERROR"
    rows.append([key, ref.doi, ref.title, found_title, f"{ratio:.2f}", verdict])
    print(f"{verdict:10} {key}")
    time.sleep(0.3)

with open("reference_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["pairing", "doi", "bundled_title", "crossref_title", "similarity", "verdict"])
    w.writerows(rows)
bad = [r for r in rows if r[5] != "VERIFIED"]
print(f"\n{len(rows) - len(bad)}/{len(rows)} verified. Details in reference_audit.csv")
