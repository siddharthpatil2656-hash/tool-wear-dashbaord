import tempfile
import unittest
from pathlib import Path

from springer_library import (
    merge_references,
    parse_reference_export,
    relevant_references,
    load_reference_library,
    save_reference_library,
)


class SpringerReferenceLibraryTests(unittest.TestCase):
    def test_parses_ris_and_multiple_authors(self):
        data = (
            "TY  - JOUR\n"
            "TI  - Tool wear in titanium machining\n"
            "AU  - A. Researcher\n"
            "AU  - B. Engineer\n"
            "JO  - Manufacturing Journal\n"
            "PY  - 2024\n"
            "DO  - https://doi.org/10.1000/tool-wear\n"
            "AB  - A study of cutting tool wear.\n"
            "ER  - \n"
        ).encode()
        records = parse_reference_export("citations.ris", data)
        self.assertEqual(records[0]["title"], "Tool wear in titanium machining")
        self.assertEqual(records[0]["authors"], "A. Researcher; B. Engineer")
        self.assertEqual(records[0]["doi"], "10.1000/tool-wear")

    def test_parses_bibtex_with_nested_braces_and_author_list(self):
        data = b"""@article{sample,
  title = {Tool wear in {Ti-6Al-4V} machining},
  author = {A. Researcher and B. Engineer},
  journal = {Manufacturing Journal},
  year = {2023},
  doi = {10.1000/sample}
}
"""
        record = parse_reference_export("papers.bib", data)[0]
        self.assertEqual(record["title"], "Tool wear in {Ti-6Al-4V} machining")
        self.assertEqual(record["authors"], "A. Researcher; B. Engineer")
        self.assertEqual(record["publication"], "Manufacturing Journal")

    def test_parses_csv_and_requires_a_title(self):
        records = parse_reference_export(
            "papers.csv",
            b"Title,Authors,Journal,Year,DOI\nTool wear study,A. Researcher,Journal,2022,10.1000/csv\n",
        )
        self.assertEqual(records[0]["date"], "2022")
        with self.assertRaisesRegex(ValueError, "No citation records"):
            parse_reference_export("empty.ris", b"TY  - JOUR\nER  - \n")

    def test_merges_by_doi_and_ranks_by_relevance(self):
        records = merge_references(
            [],
            [
                {"title": "Tool wear in titanium", "doi": "10.1000/a"},
                {"title": "Duplicate citation", "doi": "10.1000/a"},
                {"title": "General machining study", "doi": "10.1000/b"},
            ],
        )
        ranked = relevant_references(records, "titanium tool wear", limit=1)
        self.assertEqual(len(records), 2)
        self.assertEqual(ranked[0]["doi"], "10.1000/a")

    def test_library_persists_and_loads_across_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "springer_references.json"
            records = [{"title": "Offline citation", "doi": "10.1000/offline"}]
            save_reference_library(path, records)
            self.assertEqual(load_reference_library(path), merge_references([], records))


if __name__ == "__main__":
    unittest.main()
