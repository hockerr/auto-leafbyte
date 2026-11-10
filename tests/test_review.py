"""Tests for :mod:`leafbyte.review` (offline review queue + decisions merge).

Run from the repository root::

    PYTHONPATH=src python -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from leafbyte import review  # noqa: E402
from leafbyte.config import Config  # noqa: E402

#: The session sandbox may not allow writing to the OS temp directory, so the
#: tests use a scratch directory inside the repository workspace instead.
_SCRATCH_ROOT = ROOT / "out" / "_test_review"


def _make_temp_dir(case: unittest.TestCase) -> Path:
    _SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    path = _SCRATCH_ROOT / ("tmp_" + uuid.uuid4().hex[:12])
    path.mkdir(parents=True)
    case.addCleanup(shutil.rmtree, path, ignore_errors=True)
    return path


def _png(path: Path, colour=(240, 240, 240), size=(64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, format="PNG")


def _percent(consumed, total) -> float:
    return 100.0 * consumed / total


class ReviewBuildTest(unittest.TestCase):
    """``review.build`` writes the queue JSON and a fully offline page."""

    def setUp(self) -> None:
        self.root = _make_temp_dir(self)
        self.out = self.root / "out"
        self.src = self.root / "src"

        # Original photos (the batch step's `path` values are relative to out/).
        _png(self.src / "CO1.png", (25, 120, 35))
        _png(self.src / "YX1.png", (30, 30, 200))

        # Overlays written by batch: paths are relative to out/.
        _png(self.out / "overlays" / "CO1_original.jpg", (200, 200, 200))
        _png(self.out / "overlays" / "CO1_rectified.jpg", (255, 60, 60))
        _png(self.out / "overlays" / "candidates" / "CO1__t1.jpg", (255, 80, 80))
        _png(self.out / "overlays" / "candidates" / "CO1__t2.jpg", (80, 80, 255))

        self.records = [
            {
                "file": "CO1.png",
                "path": "../src/CO1.png",
                "ok": True,
                "stage": "done",
                "status": "review",
                "threshold": 0.53,
                "otsu_threshold": 0.55,
                "scale": {
                    "found": True,
                    "method": "quad",
                    "quality": 0.92,
                    "points": [[10, 10], [890, 10], [890, 1190], [10, 1190]],
                    "scale_px_length": 806,
                    "flags": [],
                    "details": {},
                },
                "total_cm2": 23.951,
                "consumed_cm2": 0.992,
                "percent_consumed": 4.141,
                "confidence": 0.42,
                "flags": ["low_stability"],
                "reason": "阈值附近面积变化 3%",
                "overlay_original": "overlays/CO1_original.jpg",
                "overlay_rectified": "overlays/CO1_rectified.jpg",
                "candidates": [
                    {
                        "candidate_id": "t1",
                        "threshold": 0.51,
                        "total_cm2": 23.8,
                        "consumed_cm2": 0.9,
                        "percent_consumed": 3.78,
                        "selected": False,
                        "overlay": "overlays/candidates/CO1__t1.jpg",
                    },
                    {
                        "candidate_id": "t2",
                        "threshold": 0.53,
                        "total_cm2": 23.951,
                        "consumed_cm2": 0.992,
                        "percent_consumed": 4.141,
                        "selected": True,
                        "overlay": "overlays/candidates/CO1__t2.jpg",
                    },
                    {
                        "candidate_id": "t3",
                        "threshold": 0.55,
                        "total_cm2": None,
                        "consumed_cm2": None,
                        "percent_consumed": 4.0,
                        "selected": False,
                        "overlay": None,
                        "error": "threshold too high",
                    },
                ],
            },
            {
                "file": "CA1 反光.png",
                "path": "../src/missing.png",  # deliberately absent
                "ok": False,
                "stage": "failed",
                "status": "failed",
                "threshold": 0.4,
                "confidence": 0.12,
                "flags": ["no_scale"],
                "reason": "比例尺标记不足",
                "error": "no scale marks",
                "overlay_original": "overlays/does_not_exist.jpg",
                "overlay_rectified": None,
                "candidates": [],
            },
            {
                "file": "YX1.png",
                "path": "../src/YX1.png",
                "ok": True,
                "stage": "done",
                "status": "auto",
                "threshold": 0.5,
                "total_cm2": 10.0,
                "consumed_cm2": 0.5,
                "percent_consumed": 5.0,
                "confidence": 0.93,
                "candidates": [],
            },
        ]
        self.index = review.build(self.out, self.records, Config())

    def _queue(self) -> dict:
        return json.loads((self.out / "review" / "queue.json").read_text(encoding="utf-8"))

    def test_build_returns_index_and_writes_queue(self) -> None:
        self.assertIsInstance(self.index, Path)
        self.assertEqual(self.index, self.out / "review" / "index.html")
        self.assertTrue(self.index.is_file())
        self.assertTrue((self.out / "review" / "queue.json").is_file())

    def test_queue_contains_only_flagged_records(self) -> None:
        queue = self._queue()
        self.assertEqual(queue["schema"], review.QUEUE_SCHEMA)
        self.assertEqual(queue["count"], 2)
        self.assertEqual(queue["total_records"], 3)
        self.assertEqual([item["file"] for item in queue["items"]], ["CO1.png", "CA1 反光.png"])
        self.assertEqual(queue["scale_cm"], 17.0)
        self.assertNotIn("YX1.png", (self.out / "review" / "queue.json").read_text(encoding="utf-8"))

    def test_queue_item_numbers_overlays_and_candidates(self) -> None:
        item = self._queue()["items"][0]
        self.assertEqual(item["threshold"], 0.53)
        self.assertEqual(item["total_cm2"], 23.951)
        self.assertEqual(item["consumed_cm2"], 0.992)
        self.assertEqual(item["percent_consumed"], 4.141)
        self.assertEqual(item["confidence"], 0.42)
        self.assertEqual(item["flags"], ["low_stability"])
        self.assertIn("阈值附近", item["reason"])
        self.assertTrue(item["scale_found"])
        self.assertEqual(item["scale_method"], "quad")

        # Image paths must resolve from out/review/ with plain relative math.
        self.assertTrue(item["image"].startswith("originals/"))
        self.assertTrue((self.index.parent / item["image"]).is_file())
        self.assertEqual(item["overlay"], "../overlays/CO1_rectified.jpg")
        self.assertEqual(item["overlay_original"], "../overlays/CO1_original.jpg")
        self.assertEqual(
            (self.index.parent / item["overlay"]).resolve(),
            (self.out / "overlays" / "CO1_rectified.jpg").resolve(),
        )
        self.assertEqual(
            (self.index.parent / item["overlay_original"]).resolve(),
            (self.out / "overlays" / "CO1_original.jpg").resolve(),
        )

        candidates = item["candidates"]
        self.assertEqual([c["candidate_id"] for c in candidates], ["t1", "t2", "t3"])
        self.assertEqual([c["threshold"] for c in candidates], [0.51, 0.53, 0.55])
        self.assertEqual(item["default_selected"], 2)
        self.assertEqual(candidates[1]["overlay"], "../overlays/candidates/CO1__t2.jpg")
        self.assertTrue((self.index.parent / candidates[1]["overlay"]).is_file())
        self.assertIsNone(candidates[2]["overlay"])  # missing overlay -> placeholder
        self.assertEqual(candidates[2]["error"], "threshold too high")
        self.assertEqual(candidates[2]["percent_consumed"], 4.0)

    def test_missing_images_degrade_gracefully(self) -> None:
        item = self._queue()["items"][1]
        self.assertEqual(item["status"], "failed")
        self.assertIsNone(item["image"])  # original not found -> placeholder
        self.assertIsNone(item["overlay"])  # overlay path not found -> placeholder
        self.assertEqual(item["candidates"], [])
        self.assertIsNone(item["total_cm2"])

    def test_html_is_offline_and_contains_data(self) -> None:
        html = self.index.read_text(encoding="utf-8")
        self.assertIn("CO1.png", html)
        self.assertIn("CA1 反光.png", html)  # UTF-8 file name
        self.assertNotIn("YX1.png", html)  # auto record never becomes a card
        self.assertIn("decisions.json", html)
        self.assertIn("reviewed", html)
        self.assertIn("candidate_id", html)
        self.assertIn("t2", html)
        self.assertIn("0.51", html)

        # No network of any kind: no absolute URLs, no external script/link.
        self.assertIsNone(re.search(r"https?://", html), "page must not contain http(s) URLs")
        self.assertIsNone(re.search(r"<script\b[^>]*\bsrc\s*=", html, re.IGNORECASE))
        self.assertIsNone(re.search(r"<link\b", html, re.IGNORECASE))
        self.assertNotIn("fetch(", html)
        self.assertNotIn("XMLHttpRequest", html)

    def test_embedded_queue_json_parses(self) -> None:
        html = self.index.read_text(encoding="utf-8")
        marker = "const QUEUE = "
        start = html.index(marker) + len(marker)
        parsed, _ = json.JSONDecoder().raw_decode(html, start)
        self.assertEqual(parsed, self._queue())

    def test_build_is_repeatable_and_handles_empty_queue(self) -> None:
        review.build(self.out, self.records, Config())  # second run keeps working
        self.assertEqual(self._queue()["count"], 2)

        empty_index = review.build(self.out, [], Config())
        empty_queue = json.loads((self.out / "review" / "queue.json").read_text(encoding="utf-8"))
        self.assertEqual(empty_queue["count"], 0)
        self.assertIn("没有待审核样本", empty_index.read_text(encoding="utf-8"))


class ReviewApplyTest(unittest.TestCase):
    """``review.apply`` merges decisions, audit fields and failure modes."""

    def _records(self):
        return [
            {
                "file": "CO1.png",
                "status": "review",
                "ok": True,
                "threshold": 0.53,
                "total_cm2": 23.951,
                "consumed_cm2": 0.992,
                "percent_consumed": 4.141,
                "confidence": 0.42,
                "flags": ["low_stability"],
                "candidates": [
                    {
                        "candidate_id": "t1",
                        "threshold": 0.51,
                        "total_cm2": 23.8,
                        "consumed_cm2": 0.9,
                        "percent_consumed": 3.78,
                    },
                    {
                        "candidate_id": "t2",
                        "threshold": 0.53,
                        "total_cm2": 23.951,
                        "consumed_cm2": 0.992,
                        "percent_consumed": 4.141,
                    },
                ],
            },
            {
                "file": "AI2.png",
                "status": "failed",
                "ok": False,
                "threshold": 0.4,
                "total_cm2": None,
                "consumed_cm2": None,
                "percent_consumed": 4.0,
                "confidence": 0.1,
                "error": "no scale",
                "candidates": [
                    {
                        "candidate_id": "t1",
                        "threshold": None,
                        "total_cm2": None,
                        "consumed_cm2": None,
                        "percent_consumed": 4.0,
                        "selected": True,
                    }
                ],
            },
            {
                "file": "CA1.png",
                "status": "review",
                "ok": True,
                "threshold": 0.5,
                "total_cm2": 20.0,
                "consumed_cm2": 1.0,
                "percent_consumed": 5.0,
                "confidence": 0.5,
                "candidates": [],
            },
            {
                "file": "AUTO.png",
                "status": "auto",
                "ok": True,
                "threshold": 0.5,
                "total_cm2": 10.0,
                "consumed_cm2": 0.5,
                "percent_consumed": 5.0,
                "confidence": 0.93,
            },
        ]

    def _by_file(self, records):
        return {record["file"]: record for record in records}

    def test_accept_uses_candidate_numbers_and_audit_fields(self) -> None:
        decisions = {
            "CO1.png": {
                "action": "accept",
                "candidate_id": "t1",
                "threshold": 0.51,
                "total_cm2": 99.0,  # deliberately wrong: candidate wins
                "consumed_cm2": 9.0,
                "note": "候选更贴合洞的边缘",
            }
        }
        out = review.apply(self._records(), decisions, Config(), reviewer="zs")
        record = self._by_file(out)["CO1.png"]
        self.assertEqual(record["status"], "done")
        self.assertEqual(record["source"], "review:zs")
        self.assertEqual(record["reviewed_by"], "zs")
        self.assertEqual(record["review_action"], "accept")
        self.assertEqual(record["review_note"], "候选更贴合洞的边缘")
        self.assertEqual(record["threshold"], 0.51)
        self.assertEqual(record["total_cm2"], 23.8)
        self.assertEqual(record["consumed_cm2"], 0.9)
        self.assertAlmostEqual(record["percent_consumed"], 3.78151, places=4)
        # audit: originals preserved
        self.assertEqual(record["original_confidence"], 0.42)
        self.assertEqual(record["original_threshold"], 0.53)
        # unreviewed records keep their status/numbers
        by_file = self._by_file(out)
        self.assertEqual(by_file["CA1.png"]["status"], "review")
        self.assertEqual(by_file["CA1.png"]["total_cm2"], 20.0)
        self.assertEqual(by_file["AUTO.png"]["status"], "auto")
        self.assertEqual(by_file["AUTO.png"]["source"], "auto")

    def test_accept_matches_candidate_by_threshold_when_id_absent(self) -> None:
        decisions = {
            "CO1.png": {
                "action": "accept",
                "threshold": 0.53,
                "total_cm2": 1.0,
                "consumed_cm2": 1.0,
                "note": "",
            }
        }
        out = review.apply(self._records(), decisions, Config(), reviewer="zs")
        record = self._by_file(out)["CO1.png"]
        self.assertEqual(record["threshold"], 0.53)
        self.assertEqual(record["total_cm2"], 23.951)
        self.assertEqual(record["consumed_cm2"], 0.992)

    def test_accept_percent_only_when_scale_missing(self) -> None:
        decisions = {
            "AI2.png": {
                "action": "accept",
                "candidate_id": "t1",
                "threshold": None,
                "total_cm2": None,
                "consumed_cm2": None,
                "percent_consumed": 4.0,
                "note": "无比例尺，仅确认啃食比例",
            }
        }
        out = review.apply(self._records(), decisions, Config(), reviewer="wb")
        record = self._by_file(out)["AI2.png"]
        self.assertEqual(record["status"], "done")
        self.assertEqual(record["source"], "review:wb")
        self.assertEqual(record["reviewed_by"], "wb")
        self.assertIsNone(record["total_cm2"])
        self.assertIsNone(record["consumed_cm2"])
        self.assertEqual(record["percent_consumed"], 4.0)
        self.assertEqual(record["original_confidence"], 0.1)
        self.assertEqual(record["original_threshold"], 0.4)

    def test_override_uses_human_numbers(self) -> None:
        decisions = {
            "CA1.png": {
                "action": "override",
                "total_cm2": 22.565,
                "consumed_cm2": 2.279,
                "note": "ImageJ 手测",
            }
        }
        out = review.apply(self._records(), decisions, Config(), reviewer="zs")
        record = self._by_file(out)["CA1.png"]
        self.assertEqual(record["status"], "done")
        self.assertEqual(record["source"], "review:zs")
        self.assertEqual(record["total_cm2"], 22.565)
        self.assertEqual(record["consumed_cm2"], 2.279)
        self.assertAlmostEqual(record["percent_consumed"], _percent(2.279, 22.565), places=4)
        self.assertEqual(record["threshold"], 0.5)  # unchanged without a decision threshold
        self.assertEqual(record["review_note"], "ImageJ 手测")
        self.assertEqual(record["original_threshold"], 0.5)
        self.assertEqual(record["original_confidence"], 0.5)

    def test_reject_keeps_no_final_measurement(self) -> None:
        decisions = {
            "CA1.png": {"action": "reject", "note": "反光，需重拍"},
        }
        out = review.apply(self._records(), decisions, Config(), reviewer="zs")
        record = self._by_file(out)["CA1.png"]
        self.assertEqual(record["status"], "rejected")
        self.assertEqual(record["source"], "review:zs")
        self.assertIsNone(record["total_cm2"])
        self.assertIsNone(record["consumed_cm2"])
        self.assertIsNone(record["percent_consumed"])
        self.assertEqual(record["review_note"], "反光，需重拍")
        self.assertEqual(record["review_action"], "reject")
        self.assertEqual(record["original_threshold"], 0.5)

    def test_unknown_file_and_unknown_action_do_not_crash(self) -> None:
        decisions = {
            "CO1.png": {"action": "accept", "threshold": 0.51, "note": ""},
            "CA1.png": {"action": "warp", "note": "unknown action"},
            "NOPE.png": {"action": "reject", "note": "never seen"},
        }
        out_records = self._records()
        before = json.loads(json.dumps(out_records))
        out = review.apply(out_records, decisions, Config(), reviewer="zs")
        # input untouched
        self.assertEqual(out_records, before)
        # unknown action: record still there, still unreviewed, with a warning
        record = self._by_file(out)["CA1.png"]
        self.assertEqual(record["status"], "review")
        self.assertEqual(record["source"], "auto")
        self.assertTrue(record["review_warnings"])
        self.assertTrue(any("warp" in warning for warning in record["review_warnings"]))
        # unknown file: mentioned in summary, not fatal
        summary_path = self._summary_path()
        summary = review.apply(
            out_records,
            decisions,
            Config(),
            reviewer="zs",
            summary_out=summary_path,
        )
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["unknown_files"], ["NOPE.png"])
        self.assertEqual(payload["counts"]["accept"], 1)
        self.assertGreaterEqual(payload["warning_count"], 2)
        self.assertTrue(any("NOPE.png" in warning["message"] for warning in payload["warnings"]))
        self.assertEqual(summary[0]["status"], "done")

    def _summary_path(self) -> Path:
        return _make_temp_dir(self) / "apply_summary.json"

    def test_apply_is_idempotent(self) -> None:
        decisions = {
            "CO1.png": {"action": "accept", "candidate_id": "t2", "threshold": 0.53, "note": ""},
            "CA1.png": {"action": "override", "total_cm2": 20.5, "consumed_cm2": 1.2, "note": ""},
            "AI2.png": {"action": "reject", "note": "no scale"},
        }
        first = review.apply(self._records(), decisions, Config(), reviewer="zs")
        second = review.apply(first, decisions, Config(), reviewer="zs")
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        # the original confidence must still be the batch value, not the reviewed one
        self.assertEqual(self._by_file(second)["CO1.png"]["original_confidence"], 0.42)
        self.assertEqual(self._by_file(second)["CO1.png"]["source"], "review:zs")

    def test_full_payload_and_reviewer_precedence(self) -> None:
        payload = {
            "reviewer": "payload-user",
            "reviewed_at": "2026-11-10T12:00:00",
            "decisions": {"CO1.png": {"action": "reject", "note": ""}},
        }
        out = review.apply(self._records(), payload, Config())
        self.assertEqual(self._by_file(out)["CO1.png"]["source"], "review:payload-user")
        out_cli = review.apply(self._records(), payload, Config(), reviewer="cli-user")
        self.assertEqual(self._by_file(out_cli)["CO1.png"]["source"], "review:cli-user")
        self.assertEqual(self._by_file(out_cli)["CO1.png"]["reviewed_at"], "2026-11-10T12:00:00")

    def test_apply_output_is_json_serialisable(self) -> None:
        decisions = {"CO1.png": {"action": "accept", "candidate_id": "t1", "threshold": 0.51}}
        out = review.apply(self._records(), decisions, Config(), reviewer="zs")
        json.dumps(out)  # must not raise

    def test_export_payload_schema(self) -> None:
        payload = review.export_payload(
            "zs",
            {
                "CO1.png": {
                    "action": "accept",
                    "candidate_id": "t2",
                    "threshold": 0.53,
                    "total_cm2": 23.951,
                    "consumed_cm2": 0.992,
                    "note": "",
                },
                "AI2.png": {
                    "action": "accept",
                    "candidate_id": "t1",
                    "threshold": None,
                    "total_cm2": None,
                    "consumed_cm2": None,
                    "percent_consumed": 4.0,
                    "note": "无比例尺",
                },
                "CA1.png": {"action": "reject", "note": "反光，需重拍"},
            },
            reviewed_at="2026-11-10T12:00:00",
        )
        self.assertEqual(set(payload), {"reviewer", "reviewed_at", "decisions"})
        self.assertEqual(payload["reviewer"], "zs")
        self.assertEqual(payload["reviewed_at"], "2026-11-10T12:00:00")
        accept = payload["decisions"]["CO1.png"]
        self.assertEqual(
            set(accept),
            {"action", "threshold", "total_cm2", "consumed_cm2", "candidate_id", "note"},
        )
        self.assertEqual(accept["action"], "accept")
        self.assertEqual(accept["candidate_id"], "t2")
        self.assertEqual(payload["decisions"]["CA1.png"], {"action": "reject", "note": "反光，需重拍"})
        relative_only = payload["decisions"]["AI2.png"]
        self.assertIsNone(relative_only["total_cm2"])
        self.assertEqual(relative_only["percent_consumed"], 4.0)
        # round-trips through JSON/UTF-8
        restored = json.loads(json.dumps(payload, ensure_ascii=False))
        self.assertEqual(restored["decisions"]["CA1.png"]["note"], "反光，需重拍")


if __name__ == "__main__":
    unittest.main()
