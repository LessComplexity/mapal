#!/usr/bin/env python3
"""The evidence reports' token claims against the token-agreement table, and the table against its raw files."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import token_agree as ta  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "benches" / "results" / "evidence" / "local-rtx-5080"
TABLE = EVIDENCE / "token-agreement.tsv"
DOCS = [EVIDENCE / "full-model.md", EVIDENCE / "prefill-precision.md"]
MARK = "<!-- tokens prefix(hf-f32,hf-bf16-eager)@128,512,2048 -->"


class TokenAgreeTests(unittest.TestCase):
    def test_measures_and_formats(self):
        a, b = [1, 2, 3, 4, 5], [1, 2, 9, 4, 5]
        self.assertEqual((ta.prefix(a, b), ta.equal(a, b)), (2, 4))
        self.assertEqual((ta.prefix(a, a[:3]), ta.equal(a, a[:3])), (3, 3))
        with tempfile.TemporaryDirectory() as d:
            ref, run, txt = Path(d, "ref_n8.json"), Path(d, "run_n8.json"), Path(d, "tool_n8_b1.txt")
            ref.write_text(json.dumps({"tokens": a, "math": "f32"}))
            run.write_text(json.dumps({"tool": "mapal", "tokens": [b, a]}))
            txt.write_text("1 2 3 7 5\n")
            self.assertEqual(ta.load(run), b)
            rows = ta.build_rows([("ref", "ref_n{n}.json"), ("run", "run_n{n}.json"), ("tool", "tool_n{n}_b1.txt"),
                                  ("absent", "absent_n{n}.json")], [8], d)
            self.assertEqual(rows, [(8, "ref", "run", 2, 4), (8, "ref", "tool", 3, 4), (8, "run", "tool", 2, 3)])

    def test_numbers_before_a_marker(self):
        self.assertEqual(ta.numbers_before("at 512: 3.67 / 3.57 | 128 / **128** | 32 "), [128, 128, 32])
        self.assertEqual(ta.numbers_before("in 128 / 32 / 61"), [128, 32, 61])
        self.assertEqual(ta.numbers_before("in 128 / 32 / 61 tokens"), [])

    def test_old_precision_claim_is_rejected(self):
        # The sentence prefill-precision.md carried: HF f32 against the harness's SDPA `hf-eager` (49 / 32 / 57),
        # written as HF f32 against HF bf16 eager attention (`hf_reference.py`: 128 / 32 / 61).
        _, _, values = ta.read_table(TABLE)
        bad, count = ta.check_text(f"HF f32 itself equals HF bf16 eager in 49 / 32 / 57{MARK} tokens", values)
        self.assertEqual(count, 1)
        self.assertEqual(bad, [("doc", 1, [128, 32, 61], [49, 32, 57])])
        bad, _ = ta.check_text(f"HF f32 itself equals HF bf16 eager in 128 / 32 / 61{MARK} tokens", values)
        self.assertEqual(bad, [])

    def test_reports_match_the_table(self):
        _, _, values = ta.read_table(TABLE)
        for doc in DOCS:
            bad, count = ta.check_text(doc.read_text(), values, doc.name)
            self.assertGreater(count, 0, f"{doc.name}: no token claim is marked")
            self.assertEqual(bad, [], f"{doc.name}: token claims differ from {TABLE.name}")

    def test_table_matches_the_raw_tokens(self):
        runs, ns, _ = ta.read_table(TABLE)
        root = Path(os.environ.get("MAPAL_RAW_ROOT", REPO))
        if not all((root / p.format(n=ns[0])).exists() for _, p in runs[:1]):
            self.skipTest("raw token files are not on this machine (never in git)")
        self.assertEqual(ta.render(runs, ns, ta.build_rows(runs, ns, root)), TABLE.read_text())


if __name__ == "__main__":
    unittest.main()
