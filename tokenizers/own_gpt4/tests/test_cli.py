from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


class CommandLineTests(unittest.TestCase):
    def _run(self, *arguments: str) -> dict[str, object]:
        repository_root = Path(__file__).resolve().parents[3]
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tokenizers.own_gpt4.cli",
                *arguments,
            ],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def test_gpt4_special_mode_prints_golden_json(self) -> None:
        payload = self._run(
            "--text",
            "<|endoftext|>hello world",
            "--special-policy",
            "all",
        )

        self.assertEqual(payload["tokenizer"]["class"], "GPT4Tokenizer")
        self.assertEqual(payload["encoding"]["ids"], [100257, 15339, 1917])
        self.assertTrue(payload["encoding"]["round_trip"])

    def test_train_mode_reports_learned_merges(self) -> None:
        payload = self._run(
            "--mode",
            "train",
            "--train-text",
            "abababab",
            "--vocab-size",
            "259",
            "--text",
            "abab",
        )

        self.assertEqual(payload["tokenizer"]["class"], "RegexTokenizer")
        self.assertGreater(payload["training"]["merge_count"], 0)
        self.assertEqual(payload["encoding"]["decoded_text"], "abab")


if __name__ == "__main__":
    unittest.main()
