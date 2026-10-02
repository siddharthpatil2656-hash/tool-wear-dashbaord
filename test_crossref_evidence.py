import json
import unittest
from io import BytesIO
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from crossref_evidence import search_research_articles, search_springer_articles


class CrossrefEvidenceTests(unittest.TestCase):
    def test_search_returns_springer_citations_without_api_key(self):
        payload = {
            "message": {
                "items": [
                    {
                        "title": ["Tool wear in titanium machining"],
                        "author": [
                            {"given": "A.", "family": "Researcher"},
                            {"given": "B.", "family": "Engineer"},
                        ],
                        "container-title": ["Manufacturing Journal"],
                        "published": {"date-parts": [[2025, 2, 1]]},
                        "DOI": "10.1000/example",
                        "URL": "https://doi.org/10.1000/example",
                        "abstract": "<p>Wear <i>model</i> study.</p>",
                        "publisher": "Springer Nature",
                    },
                    {
                        "title": ["Other publisher record"],
                        "publisher": "Other Publisher",
                    },
                ]
            }
        }
        with patch(
            "crossref_evidence.urlopen",
            return_value=BytesIO(json.dumps(payload).encode("utf-8")),
        ) as mocked:
            results = search_springer_articles("titanium carbide turning", limit=50)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["authors"], "A. Researcher; B. Engineer")
        self.assertEqual(results[0]["date"], "2025-2-1")
        self.assertEqual(results[0]["abstract"], "Wear model study.")
        self.assertEqual(results[0]["publisher"], "Springer Nature")
        parameters = parse_qs(urlparse(mocked.call_args.args[0].full_url).query)
        self.assertEqual(parameters["query.publisher-name"], ["Springer"])
        self.assertEqual(parameters["rows"], ["8"])
        self.assertNotIn("api_key", parameters)

    def test_rejects_empty_search(self):
        with self.assertRaisesRegex(ValueError, "search query"):
            search_springer_articles("")

    def test_search_can_include_multiple_publishers_and_preserves_sources(self):
        payload = {
            "message": {
                "items": [
                    {
                        "title": ["Metal cutting research"],
                        "publisher": "Elsevier BV",
                        "type": "journal-article",
                    },
                    {
                        "title": ["Tool life study"],
                        "publisher": "Springer Nature",
                        "type": "journal-article",
                    },
                    {
                        "title": ["Book result"],
                        "publisher": "CRC Press",
                        "type": "book",
                    },
                ]
            }
        }
        with patch(
            "crossref_evidence.urlopen",
            return_value=BytesIO(json.dumps(payload).encode("utf-8")),
        ) as mocked:
            results = search_research_articles("metal cutting tool wear", limit=5)

        self.assertEqual(
            [item["publisher"] for item in results],
            ["Elsevier BV", "Springer Nature"],
        )
        self.assertTrue(all(item["source"] == "Crossref bibliographic metadata" for item in results))
        parameters = parse_qs(urlparse(mocked.call_args.args[0].full_url).query)
        self.assertNotIn("query.publisher-name", parameters)

    def test_search_can_filter_public_metadata_to_ieee(self):
        payload = {
            "message": {
                "items": [
                    {
                        "title": ["IEEE machining study"],
                        "publisher": "Institute of Electrical and Electronics Engineers (IEEE)",
                        "type": "journal-article",
                    },
                    {
                        "title": ["Non-IEEE record"],
                        "publisher": "Elsevier",
                        "type": "journal-article",
                    },
                ]
            }
        }
        with patch(
            "crossref_evidence.urlopen",
            return_value=BytesIO(json.dumps(payload).encode("utf-8")),
        ) as mocked:
            results = search_research_articles(
                "machining tool wear", publisher="IEEE"
            )

        self.assertEqual([item["title"] for item in results], ["IEEE machining study"])
        parameters = parse_qs(urlparse(mocked.call_args.args[0].full_url).query)
        self.assertEqual(parameters["query.publisher-name"], ["IEEE"])


if __name__ == "__main__":
    unittest.main()
