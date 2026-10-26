"""Small, metadata-only client for the Springer Nature Metadata API."""

import json
import re
from html import unescape
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


METADATA_API_URL = "https://api.springernature.com/meta/v2/json"
MAX_RESULTS = 10


def _plain_text(value: object) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _first_url(urls: object) -> str:
    if isinstance(urls, str):
        return urls
    if isinstance(urls, list):
        for item in urls:
            if isinstance(item, dict) and item.get("value"):
                return str(item["value"])
    return ""


def _normalize_record(record: dict) -> dict[str, object]:
    creators = record.get("creators") or record.get("authors") or []
    authors = []
    if isinstance(creators, list):
        for creator in creators:
            name = creator.get("creator", "") if isinstance(creator, dict) else creator
            if name:
                authors.append(_plain_text(name))
    elif isinstance(creators, str):
        authors.append(_plain_text(creators))
    doi = _plain_text(record.get("doi"))
    return {
        "title": _plain_text(record.get("title")) or "Untitled record",
        "authors": "; ".join(authors),
        "publication": _plain_text(record.get("publicationName")),
        "date": _plain_text(record.get("publicationDate")),
        "doi": doi,
        "url": _first_url(record.get("url")) or (
            f"https://doi.org/{doi}" if doi else ""
        ),
        "abstract": _plain_text(record.get("abstract")),
        "open_access": str(record.get("openaccess", "")).lower() == "true",
    }


def search_springer_metadata(
    query: str,
    api_key: str,
    limit: int = 8,
    timeout_seconds: float = 12,
) -> list[dict[str, str]]:
    """Search article metadata; this endpoint does not return licensed full text."""
    clean_query = str(query or "").strip()
    clean_key = str(api_key or "").strip()
    if not clean_query:
        raise ValueError("Enter a search query.")
    if not clean_key:
        raise ValueError("Configure a Springer Nature API key before searching.")
    if len(clean_query) > 250:
        raise ValueError("Search queries must be 250 characters or fewer.")
    result_count = max(1, min(int(limit), MAX_RESULTS))
    parameters = {
        'q': clean_query,
        'api_key': clean_key,
        'p': result_count,
    }
    request_url = f"{METADATA_API_URL}?{urlencode(parameters)}"
    request = Request(
        request_url,
        headers={"Accept": "application/json", "User-Agent": "ToolWearDashboard/1.0"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError(
                "Springer Nature rejected the API key or this API is not enabled for it."
            ) from None
        if exc.code == 429:
            raise RuntimeError("Springer Nature API rate limit reached; try again later.") from None
        raise RuntimeError(
            f"Springer Nature metadata search failed with HTTP status {exc.code}."
        ) from None
    except URLError:
        raise RuntimeError(
            "Could not connect to Springer Nature. Check the internet connection and try again."
        ) from None
    except TimeoutError:
        raise RuntimeError(
            "Springer Nature metadata search timed out. Try again when the connection is stable."
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("Springer Nature returned an unreadable metadata response.") from None

    if not isinstance(payload, dict):
        raise RuntimeError("Springer Nature returned an unexpected metadata response.")
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Springer Nature returned an unexpected metadata response.")
    return [_normalize_record(item) for item in records if isinstance(item, dict)]
