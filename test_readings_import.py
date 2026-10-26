import unittest
from io import BytesIO

import pandas as pd

from readings_import import clean_readings, guess_columns, parse_readings


class ReadingsImportTests(unittest.TestCase):
    def test_csv_with_headers_is_mapped(self):
        data = b"Vc (m/min),Feed (mm/rev),ap (mm),Tool life (min)\n80,0.12,1.0,40\n100,0.15,1.5,18\n"
        df, _ = parse_readings("runs.csv", data)
        mapping = guess_columns(df)
        clean, dropped = clean_readings(df, mapping)
        self.assertEqual(len(clean), 2)
        self.assertEqual(dropped, 0)
        self.assertEqual(clean["cutting_speed_m_min"].tolist(), [80.0, 100.0])
        self.assertEqual(clean["measured_tool_life_min"].tolist(), [40.0, 18.0])

    def test_excel_roundtrip(self):
        buffer = BytesIO()
        pd.DataFrame({"Speed": [60, 90], "Feed": [0.1, 0.2], "Depth": [0.5, 1.0]}).to_excel(buffer, index=False)
        df, _ = parse_readings("runs.xlsx", buffer.getvalue())
        clean, _ = clean_readings(df, guess_columns(df))
        self.assertEqual(len(clean), 2)
        self.assertTrue(clean["measured_tool_life_min"].isna().all())

    def test_headerless_numeric_columns_follow_order_and_bad_rows_are_dropped(self):
        df, _ = parse_readings("x.csv", b"80,0.12,1.0\n0,0.1,1\nabc,0.1,1\n90,0.2,1.2\n")
        clean, dropped = clean_readings(df, guess_columns(df))
        self.assertEqual(len(clean), 2)
        self.assertEqual(dropped, 2)

    def test_decimal_comma(self):
        df, _ = parse_readings("x.csv", b"speed;feed;depth\n80;0,12;1,5\n".replace(b";", b";"))
        self.assertFalse(df.empty)

    def test_pdf_table(self):
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table
        buffer = BytesIO()
        table = Table([["Vc", "f", "ap"], ["80", "0.12", "1.0"], ["100", "0.15", "1.5"]])
        from reportlab.platypus import TableStyle
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, (0, 0, 0))]))
        SimpleDocTemplate(buffer, pagesize=letter).build([table])
        df, _ = parse_readings("t.pdf", buffer.getvalue())
        clean, _ = clean_readings(df, guess_columns(df))
        self.assertEqual(len(clean), 2)


if __name__ == "__main__":
    unittest.main()
