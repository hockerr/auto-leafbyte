"""Tests for the summary workbook/CSV builder (tools/make_summary_table.py)."""

from __future__ import annotations

import csv
import json
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import make_summary_table as summary  # noqa: E402

SCRATCH = ROOT / "out" / "tests"
SCRATCH.mkdir(parents=True, exist_ok=True)


def _scratch(tag: str) -> Path:
    path = SCRATCH / f"{tag}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fake_records() -> list[dict]:
    base = {
        "status": "auto",
        "confidence": 0.95,
        "percent_consumed": 1.0,
        "measure": {"mode": "color", "filled_dark_px": 0},
        "scale": {"scale_px_length": 750},
        "flags": [],
        "reason": "",
        "traits": {"perimeter_cm": 25.0, "length_cm": 11.0, "width_cm": 4.0,
                   "aspect_ratio": 2.75, "circularity": 0.5, "solidity": 0.95,
                   "intact_area_cm2": 20.0, "gcc": 0.4, "exg": 0.2},
    }
    first = dict(base, file="Herbivory Data 140盐肤木-1 (2026.07.08).png",
                 total_cm2=20.0, consumed_cm2=0.2)
    second = dict(base, file="Herbivory Data 140盐肤木-2 (2026.07.08).png",
                  total_cm2=22.0, consumed_cm2=0.0, status="review", confidence=0.55,
                  flags=["area_unstable"])
    return [first, second]


def _write_run(folder: Path, records: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "results.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class SummaryTableTest(unittest.TestCase):
    def test_csv_has_all_columns_and_rows(self) -> None:
        run = _scratch("run")
        _write_run(run, _fake_records())
        out = _scratch("out")
        self.assertEqual(0, summary.main(["--run", str(run), "--out", str(out)]))
        csv_path = out / "叶片性状汇总.csv"
        self.assertTrue(csv_path.exists())
        with open(csv_path, encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual([label for _, label in summary.SAMPLE_COLUMNS], rows[0])
        self.assertEqual(3, len(rows))                      # header + 2 samples
        self.assertIn("盐肤木", rows[1][1])                 # dataset column

    def test_xlsx_when_openpyxl_is_available(self) -> None:
        run = _scratch("run")
        _write_run(run, _fake_records())
        out = _scratch("out")
        summary.main(["--run", str(run), "--out", str(out)])
        xlsx_path = out / "叶片性状汇总.xlsx"
        try:
            from openpyxl import load_workbook
        except ImportError:  # pragma: no cover
            self.assertFalse(xlsx_path.exists())
            return
        self.assertTrue(xlsx_path.exists())
        workbook = load_workbook(xlsx_path)
        self.assertEqual(["样本数据", "性状统计", "说明"], workbook.sheetnames)
        sheet = workbook["样本数据"]
        self.assertEqual(3, sheet.max_row)
        self.assertGreaterEqual(sheet.max_column, 30)
        stats = workbook["性状统计"]
        self.assertGreater(stats.max_row, 10)
        labels = {row[0] for row in stats.iter_rows(min_row=2, values_only=True)}
        self.assertIn("全部样本", labels)

    def test_two_runs_are_merged(self) -> None:
        run_a, run_b = _scratch("a"), _scratch("b")
        _write_run(run_a, _fake_records()[:1])
        _write_run(run_b, _fake_records()[1:])
        out = _scratch("out")
        summary.main(["--run", str(run_a), "--run", str(run_b), "--out", str(out)])
        with open(out / "叶片性状汇总.csv", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(3, len(rows))
        self.assertEqual(["1", "2"], [rows[1][0], rows[2][0]])


if __name__ == "__main__":
    unittest.main()
