from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from tokenizers.tiktoken_gpt4 import GPT4Tokenizer


class GPT4TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = GPT4Tokenizer()

    def test_hello_world_matches_cl100k_base_golden_tokens(self) -> None:
        self.assertEqual(self.tokenizer.encode("hello world"), [15339, 1917])
        self.assertEqual(self.tokenizer.model_name, "gpt-4")
        self.assertEqual(self.tokenizer.encoding_name, "cl100k_base")

    def test_unicode_emoji_and_newline_round_trip(self) -> None:
        text = "你好，world 🙂\n第二行🚀"
        tokens = self.tokenizer.encode(text)

        self.assertEqual(self.tokenizer.decode(tokens), text)

    def test_special_token_literal_is_encoded_as_ordinary_text(self) -> None:
        literal = "<|endoftext|>"

        self.assertEqual(
            self.tokenizer.encode(literal),
            [27, 91, 8862, 728, 428, 91, 29],
        )
        self.assertNotIn(100257, self.tokenizer.encode(literal))
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(literal)), literal)

    def test_token_bytes_reconstruct_original_utf8(self) -> None:
        text = "A你🙂\n"
        tokens = self.tokenizer.encode(text)

        reconstructed = b"".join(
            self.tokenizer.token_bytes(token_id) for token_id in tokens
        )
        self.assertEqual(reconstructed, text.encode("utf-8"))

    def test_analyze_has_contiguous_byte_offsets_and_safe_displays(self) -> None:
        text = "你🙂\n"
        report = self.tokenizer.analyze(text)
        original_bytes = text.encode("utf-8")

        self.assertEqual(report.text, text)
        self.assertEqual(report.tokens, tuple(self.tokenizer.encode(text)))
        self.assertEqual(report.token_count, len(report.token_details))
        self.assertEqual(report.utf8_byte_count, len(original_bytes))
        self.assertEqual(report.decoded_text, text)

        cursor = 0
        reconstructed = bytearray()
        for index, detail in enumerate(report):
            with self.subTest(index=index):
                self.assertEqual(detail.index, index)
                self.assertEqual(detail.byte_start, cursor)
                self.assertEqual(
                    detail.byte_end - detail.byte_start,
                    len(detail.raw_bytes),
                )
                self.assertEqual(detail.bytes, detail.raw_bytes)
                self.assertEqual(detail.bytes_hex, detail.raw_bytes.hex())
                self.assertEqual(detail.as_dict()["bytes"], list(detail.raw_bytes))
                reconstructed.extend(detail.raw_bytes)
                cursor = detail.byte_end

        self.assertEqual(bytes(reconstructed), original_bytes)
        self.assertEqual(cursor, len(original_bytes))
        newline_details = [
            detail for detail in report.token_details if detail.raw_bytes == b"\n"
        ]
        self.assertEqual(len(newline_details), 1)
        self.assertEqual(newline_details[0].display, "\\n")

    def test_empty_text_has_empty_results_and_zero_metrics(self) -> None:
        report = self.tokenizer.analyze("")

        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")
        self.assertEqual(report.tokens, ())
        self.assertEqual(report.token_details, ())
        self.assertEqual(report.utf8_byte_count, 0)
        self.assertEqual(report.token_count, 0)
        self.assertEqual(report.compression_ratio, 0.0)
        self.assertEqual(report.reduction_ratio, 0.0)

    def test_encode_rejects_non_string_inputs(self) -> None:
        for invalid in (b"text", None, 123, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(TypeError, "text must be a str"):
                    self.tokenizer.encode(invalid)  # type: ignore[arg-type]

    def test_encode_and_analyze_reject_lone_surrogates_consistently(self) -> None:
        invalid_text = "\ud800"

        for operation in (self.tokenizer.encode, self.tokenizer.analyze):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    "valid Unicode scalar values",
                ):
                    operation(invalid_text)

    def test_decode_strictly_checks_container_and_items(self) -> None:
        for invalid in ("15339", b"15339", iter([15339])):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(TypeError, "tokens must be a sequence"):
                    self.tokenizer.decode(invalid)  # type: ignore[arg-type]

        for invalid in ([15339, True], [15339, 1917.0], [15339, "1917"]):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(TypeError, r"tokens\[1\]"):
                    self.tokenizer.decode(invalid)  # type: ignore[arg-type]

    def test_invalid_and_reserved_gap_ids_have_friendly_errors(self) -> None:
        for invalid_token_id in (-1, 100256, 100261, 999999):
            with self.subTest(token_id=invalid_token_id):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"token_id={invalid_token_id} is not a valid token id",
                ):
                    self.tokenizer.token_bytes(invalid_token_id)
                with self.assertRaisesRegex(
                    ValueError,
                    rf"tokens\[0\]={invalid_token_id} is not a valid token id",
                ):
                    self.tokenizer.decode([invalid_token_id])

    def test_token_bytes_strictly_checks_type(self) -> None:
        for invalid in (True, 1.0, "1", None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(TypeError, "token_id must be an int"):
                    self.tokenizer.token_bytes(invalid)  # type: ignore[arg-type]


class CommandLineTests(unittest.TestCase):
    def test_cli_prints_json_analysis(self) -> None:
        repository_root = Path(__file__).resolve().parents[3]
        environment = os.environ.copy()
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tokenizers.tiktoken_gpt4.tokenizer",
                "--text",
                "hello world",
            ],
            cwd=repository_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)

        self.assertEqual(payload["tokenizer"]["name"], "tiktoken-gpt-4")
        self.assertEqual(payload["tokenizer"]["model_name"], "gpt-4")
        self.assertEqual(payload["tokenizer"]["encoding_name"], "cl100k_base")
        self.assertEqual(payload["analysis"]["tokens"], [15339, 1917])
        self.assertEqual(payload["analysis"]["decoded_text"], "hello world")
        self.assertEqual(len(payload["analysis"]["token_details"]), 2)


if __name__ == "__main__":
    unittest.main()
