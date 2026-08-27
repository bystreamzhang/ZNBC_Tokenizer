import unittest

from bpe import BytePairEncoder
from visualizations.adapters.bpe_trace import (
    VisualizationInputError,
    build_bpe_trace,
)
from visualizations.server import STATIC_ROOT, resolve_static_file


class BpeTraceTests(unittest.TestCase):
    def test_trace_comes_from_the_actual_encoder_merges(self) -> None:
        payload = {
            "corpus": ["aaaa"],
            "vocab_size": 258,
            "text": "aaaa",
        }
        trace = build_bpe_trace(payload)

        encoder = BytePairEncoder()
        encoder.train(payload["corpus"], vocab_size=payload["vocab_size"])

        self.assertEqual(
            [step["pair"] for step in trace["trace"]["steps"]],
            [list(merge.pair) for merge in encoder.merges],
        )
        self.assertEqual(trace["trace"]["steps"][0]["before"], [[97, 97, 97, 97]])
        self.assertEqual(trace["trace"]["steps"][0]["after"], [[256, 256]])
        self.assertEqual(trace["trace"]["steps"][1]["after"], [[257]])
        self.assertEqual(trace["encoding"]["tokens"], [257])
        self.assertTrue(trace["invariants"]["trace_matches_encoder"])
        self.assertTrue(trace["invariants"]["encoded_bytes_match_input"])

    def test_separate_samples_do_not_create_a_boundary_pair(self) -> None:
        trace = build_bpe_trace(
            {
                "corpus": ["a", "a"],
                "vocab_size": 257,
                "text": "aa",
            }
        )

        self.assertEqual(trace["training"]["actual_vocab_size"], 256)
        self.assertEqual(trace["trace"]["steps"], [])
        self.assertEqual(trace["trace"]["final_sequences"], [[97], [97]])
        self.assertEqual(trace["encoding"]["tokens"], [97, 97])

    def test_unicode_encoding_bytes_are_preserved(self) -> None:
        text = "你好🙂"
        trace = build_bpe_trace(
            {
                "corpus": ["你好你好", "你好🙂你好🙂"],
                "vocab_size": 270,
                "text": text,
            }
        )

        vocabulary = {
            int(token_id): bytes(token_bytes)
            for token_id, token_bytes in trace["vocabulary"].items()
        }
        rebuilt = b"".join(
            vocabulary[token_id] for token_id in trace["encoding"]["tokens"]
        )
        self.assertEqual(rebuilt.decode("utf-8"), text)

    def test_invalid_payloads_are_rejected(self) -> None:
        invalid_payloads = (
            None,
            {},
            {"corpus": [], "vocab_size": 256},
            {"corpus": ["ok", 1], "vocab_size": 256},
            {"corpus": ["ok"], "vocab_size": 255},
            {"corpus": ["ok"], "vocab_size": True},
            {"corpus": ["ok"], "vocab_size": 256, "text": b"bad"},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(VisualizationInputError):
                    build_bpe_trace(payload)


class StaticFileTests(unittest.TestCase):
    def test_root_and_assets_resolve_inside_static_directory(self) -> None:
        self.assertEqual(resolve_static_file("/"), STATIC_ROOT / "index.html")
        self.assertEqual(
            resolve_static_file("/styles/system.css"),
            STATIC_ROOT / "styles" / "system.css",
        )

    def test_directory_traversal_is_rejected(self) -> None:
        self.assertIsNone(resolve_static_file("/../bpe.py"))
        self.assertIsNone(resolve_static_file("/%2e%2e/bpe.py"))


if __name__ == "__main__":
    unittest.main()

