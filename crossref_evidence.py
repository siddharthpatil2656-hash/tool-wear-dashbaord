"""Keyless Crossref metadata search for machining research publications."""

import json
import re
from html import unescape
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


CROSSREF_API_URL = "https://api.crossref.org/works"
MAX_RESULTS = 8


def _plain_text(value: object) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _first(value: object) -> str:
    if isinstance(value, list):
        return _plain_text(value[0]) if value else ""
    return _plain_text(value)


def _normalize_record(record: dict) -> dict[str, object]:
    authors = []
    for author in record.get("author", []):
        if not isinstance(author, dict):
            continue
        name = " ".join(
            part for part in (_plain_text(author.get("given")), _plain_text(author.get("family")))
            if part
        )
        if name:
            authors.append(name)
    published = (
        record.get("published")
        or record.get("published-print")
        or record.get("published-online")
        or record.get("created")
        or {}
    )
    date_parts = published.get("date-parts", [[]]) if isinstance(published, dict) else [[]]
    date = "-".join(str(part) for part in date_parts[0]) if date_parts and date_parts[0] else ""
    doi = _plain_text(record.get("DOI"))
    return {
        "title": _first(record.get("title")) or "Untitled record",
        "authors": "; ".join(authors),
        "publication": _first(record.get("container-title")),
        "date": date,
        "doi": doi,
        "url": _plain_text(record.get("URL")) or (
            f"https://doi.org/{doi}" if doi else ""
        ),
        "abstract": _plain_text(record.get("abstract")),
        "publisher": _plain_text(record.get("publisher")),
        "source": "Crossref bibliographic metadata",
        "open_access": False,
    }


def search_research_articles(
    query: str,
    limit: int = 8,
    publisher: str | None = None,
    timeout_seconds: float = 10,
) -> list[dict[str, object]]:
    """Search Crossref metadata for journal articles, optionally by publisher."""
    clean_query = str(query or "").strip()
    if not clean_query:
        raise ValueError("A machining setup search query is required.")
    clean_publisher = str(publisher or "").strip()
    result_count = max(1, min(int(limit), MAX_RESULTS))
    parameters = {
        "query.bibliographic": clean_query,
        "filter": "type:journal-article",
        "rows": result_count,
        "select": "DOI,title,author,published,published-print,published-online,created,container-title,URL,abstract,publisher,type",
    }
    if clean_publisher:
        parameters["query.publisher-name"] = clean_publisher
    request = Request(
        f"{CROSSREF_API_URL}?{urlencode(parameters)}",
        headers={
            "Accept": "application/json",
            "User-Agent": "ToolWearDashboard/1.0 (contact: Crossref public API)",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError(
                "Crossref rate limit reached; wait briefly before trying again."
            ) from None
        raise RuntimeError(
            f"Crossref metadata search failed with HTTP status {exc.code}."
        ) from None
    except URLError:
        raise RuntimeError(
            "Could not connect to Crossref. Check the internet connection."
        ) from None
    except OSError:
        raise RuntimeError(
            "Could not complete the Crossref network request. Check the internet connection."
        ) from None
    except TimeoutError:
        raise RuntimeError("Crossref metadata search timed out; try again.") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("Crossref returned an unreadable metadata response.") from None

    if not isinstance(payload, dict):
        raise RuntimeError("Crossref returned an unexpected metadata response.")
    message = payload.get("message")
    items = message.get("items") if isinstance(message, dict) else None
    if not isinstance(items, list):
        raise RuntimeError("Crossref returned an unexpected metadata response.")
    return [
        _normalize_record(item)
        for item in items
        if isinstance(item, dict)
        and item.get("type") in (None, "journal-article")
        and (
            not clean_publisher
            or clean_publisher.casefold() in _plain_text(item.get("publisher")).casefold()
        )
    ]


def search_springer_articles(
    query: str,
    limit: int = 8,
    timeout_seconds: float = 10,
) -> list[dict[str, object]]:
    """Backwards-compatible Springer-only Crossref search."""
    return search_research_articles(
        query,
        limit=limit,
        publisher="Springer",
        timeout_seconds=timeout_seconds,
    )
