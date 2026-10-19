"""
Springer Nature API Client for Tool Wear & Machining Research.
Interfaces with Springer Nature Meta API and OpenAccess API to query
peer-reviewed literature (e.g. The International Journal of Advanced Manufacturing Technology).
"""

from __future__ import annotations
import os
import re
from typing import Dict, List, Optional, Any
import requests
import streamlit as st


SPRINGER_META_URL = "https://api.springernature.com/meta/v1/json"
SPRINGER_OPENACCESS_URL = "https://api.springernature.com/openaccess/json"


def get_stored_api_key() -> Optional[str]:
    """
    Safely retrieves the Springer API key from:
    1. Streamlit Session State (entered via UI)
    2. Streamlit Secrets (secrets.toml on local or cloud)
    3. System Environment Variable (SPRINGER_API_KEY)
    4. Local .env file
    """
    # 1. UI session state
    if st.session_state.get("springer_api_key"):
        key = str(st.session_state["springer_api_key"]).strip()
        if key:
            return key

    # 2. Streamlit secrets
    try:
        if hasattr(st, "secrets") and "SPRINGER_API_KEY" in st.secrets:
            key = str(st.secrets["SPRINGER_API_KEY"]).strip()
            if key:
                return key
    except Exception:
        pass

    # 3. Environment variable
    env_key = os.environ.get("SPRINGER_API_KEY", "").strip()
    if env_key:
        return env_key

    # 4. Local .env file fallback
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("SPRINGER_API_KEY="):
                        k = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if k:
                            return k
        except Exception:
            pass

    return None


def clean_query_term(term: str) -> str:
    """Extracts the cleanest technical keyword from a material string."""
    # Remove parenthetical details
    cleaned = re.sub(r"\(.*?\)", "", term).strip()
    # If it's a known alloy, keep the key identifier
    for pattern in ["Ti-6Al-4V", "Inconel 718", "AISI 316L", "AISI 304", "AISI 1045", "AISI 4140", "AISI 4340", "AISI H13", "D2", "CFRP", "CGI-400", "7075-T6", "6061-T6", "Hastelloy", "AZ91D"]:
        if pattern.lower() in term.lower():
            return pattern
    return cleaned.split("/")[0].strip()


@st.cache_data(ttl=3600, show_spinner=False)
def query_springer_api(
    query_str: str,
    api_key: Optional[str] = None,
    max_results: int = 5,
    endpoint: str = "meta"
) -> Dict[str, Any]:
    """
    Executes a query against the Springer Nature API.
    Cached for 1 hour to conserve API quota.
    """
    effective_key = api_key or get_stored_api_key()
    if not effective_key:
        return {
            "status": "no_key",
            "message": "No Springer API key configured. Provide a key in the sidebar or Streamlit Secrets to enable live literature search.",
            "records": []
        }

    url = SPRINGER_OPENACCESS_URL if endpoint == "openaccess" else SPRINGER_META_URL
    params = {
        "api_key": effective_key,
        "q": query_str,
        "p": min(max_results, 20),
        "s": 1
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        
        if resp.status_code == 401 or resp.status_code == 403:
            return {
                "status": "auth_error",
                "message": f"Springer API authentication failed (HTTP {resp.status_code}). Please verify your API key.",
                "records": []
            }
        elif resp.status_code == 429:
            return {
                "status": "rate_limited",
                "message": "Springer API rate limit reached. Using local curated engineering dataset.",
                "records": []
            }
        elif resp.status_code != 200:
            return {
                "status": "error",
                "message": f"Springer API returned status code {resp.status_code}.",
                "records": []
            }

        data = resp.json()
        records_raw = data.get("records", [])
        
        formatted_records = []
        for r in records_raw:
            # Extract creators
            creators = r.get("creators", [])
            if isinstance(creators, list) and creators:
                author_names = [c.get("creator", "") for c in creators if isinstance(c, dict) and c.get("creator")]
                authors_str = ", ".join(author_names[:3])
                if len(author_names) > 3:
                    authors_str += " et al."
            else:
                authors_str = "Springer Research Group"

            # URLs
            url_entries = r.get("url", [])
            web_url = ""
            if isinstance(url_entries, list):
                for u in url_entries:
                    if isinstance(u, dict) and u.get("format") == "html":
                        web_url = u.get("value", "")
                        break
                if not web_url and url_entries and isinstance(url_entries[0], dict):
                    web_url = url_entries[0].get("value", "")

            # Abstract
            abstract = r.get("abstract", "")
            if isinstance(abstract, list):
                abstract = " ".join([str(a) for a in abstract])
            
            # Shorten abstract
            abstract_str = str(abstract).strip()
            if len(abstract_str) > 400:
                abstract_str = abstract_str[:397] + "..."

            formatted_records.append({
                "title": r.get("title", "Untitled Publication"),
                "authors": authors_str,
                "journal": r.get("publicationName", "Springer Journal"),
                "publication_date": r.get("publicationDate", "N/A"),
                "doi": r.get("doi", ""),
                "url": web_url or (f"https://doi.org/{r.get('doi')}" if r.get("doi") else "https://link.springer.com"),
                "abstract": abstract_str or "No abstract text provided in meta response.",
                "open_access": str(r.get("openaccess", "")).lower() == "true",
            })

        total_results = 0
        result_meta = data.get("result", [])
        if result_meta and isinstance(result_meta, list):
            total_results = int(result_meta[0].get("total", len(formatted_records)))

        return {
            "status": "success",
            "total_available": total_results,
            "query": query_str,
            "records": formatted_records
        }

    except requests.exceptions.Timeout:
        return {
            "status": "timeout",
            "message": "Springer API request timed out after 10s. Showing local cached benchmarks.",
            "records": []
        }
    except Exception as exc:
        return {
            "status": "exception",
            "message": f"Connection error querying Springer API: {str(exc)}",
            "records": []
        }


def search_papers_for_material_and_tool(
    workpiece_name: str,
    tool_material: str,
    api_key: Optional[str] = None,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Builds an optimized query targeted at IJAMT tool wear studies.
    """
    mat_term = clean_query_term(workpiece_name)
    tool_term = "carbide" if "carbide" in tool_material.lower() else ("ceramic" if "ceramic" in tool_material.lower() else ("pcd" if "pcd" in tool_material.lower() else "tool"))
    
    # Springer query syntax: field:"value" AND field:"value"
    # Target The International Journal of Advanced Manufacturing Technology or machining subjects
    query = f'("tool wear" OR "tool life" OR "flank wear") AND "{mat_term}" AND "{tool_term}"'
    
    return query_springer_api(query, api_key=api_key, max_results=max_results)
