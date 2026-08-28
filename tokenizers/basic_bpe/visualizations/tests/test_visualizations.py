import unittest

from bpe import BytePairEncoder
from visualizations.adapters.bpe_decode import build_bpe_decode
from visualizations.adapters.bpe_encode import build_bpe_encode
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


class BpeEncodeTests(unittest.TestCase):
    def test_encoder_trace_shows_rules_cascading_in_order(self) -> None:
        result = build_bpe_encode(
            {
                "corpus": ["ababab"],
                "vocab_size": 259,
                "text": "ababab",
            }
        )

        steps = result["trace"]["steps"]
        self.assertEqual(
            [step["pair"] for step in steps],
            [[97, 98], [256, 256], [257, 256]],
        )
        self.assertEqual(steps[0]["before"], [97, 98, 97, 98, 97, 98])
        self.assertEqual(steps[0]["after"], [256, 256, 256])
        self.assertEqual(steps[1]["after"], [257, 256])
        self.assertEqual(steps[2]["after"], [258])
        self.assertFalse(steps[0]["depends_on_previous_token"])
        self.assertTrue(steps[1]["depends_on_previous_token"])
        self.assertEqual(result["encoding"]["tokens"], [258])
        self.assertEqual(result["trace"]["final_tokens"], [258])
        self.assertTrue(result["invariants"]["trace_matches_encoder"])
        self.assertTrue(result["invariants"]["rules_follow_training_order"])
        self.assertTrue(result["invariants"]["parents_precede_children"])

    def test_encoder_keeps_noop_rules_in_their_original_order(self) -> None:
        result = build_bpe_encode(
            {
                "corpus": ["bcbcbcbc", "ababab"],
                "vocab_size": 258,
                "text": "abc",
            }
        )

        first, second = result["trace"]["steps"]
        self.assertEqual(first["pair"], [98, 99])
        self.assertEqual(first["after"], [97, 256])
        self.assertTrue(first["applied"])
        self.assertEqual(second["pair"], [97, 98])
        self.assertEqual(second["before"], [97, 256])
        self.assertEqual(second["after"], [97, 256])
        self.assertFalse(second["applied"])
        self.assertEqual(result["encoding"]["tokens"], [97, 256])

    def test_encoder_does_not_learn_frequent_pairs_from_new_text(self) -> None:
        result = build_bpe_encode(
            {
                "corpus": ["abab"],
                "vocab_size": 257,
                "text": "zzzz",
            }
        )

        self.assertEqual(result["trace"]["rules_applied"], 0)
        self.assertEqual(result["encoding"]["tokens"], [122, 122, 122, 122])

    def test_encoder_trace_marks_non_overlapping_pair_starts(self) -> None:
        result = build_bpe_encode(
            {
                "corpus": ["aaaaa"],
                "vocab_size": 257,
                "text": "aaaaa",
            }
        )

        step = result["trace"]["steps"][0]
        self.assertEqual(step["merged_pair_starts"], [0, 2])
        self.assertEqual(step["applied_merge_count"], 2)
        self.assertEqual(step["after"], [256, 256, 97])

    def test_encoder_preserves_unicode_bytes_and_empty_text(self) -> None:
        text = "你好🙂"
        result = build_bpe_encode(
            {
                "corpus": ["你好你好", "hello hello"],
                "vocab_size": 264,
                "text": text,
            }
        )
        empty = build_bpe_encode(
            {
                "corpus": ["sample"],
                "vocab_size": 260,
                "text": "",
            }
        )

        vocabulary = {
            int(token_id): bytes(token_bytes)
            for token_id, token_bytes in result["vocabulary"].items()
        }
        rebuilt = b"".join(
            vocabulary[token_id] for token_id in result["encoding"]["tokens"]
        )
        self.assertEqual(rebuilt, text.encode("utf-8"))
        self.assertTrue(result["invariants"]["encoded_bytes_match_input"])
        self.assertEqual(empty["encoding"]["initial_tokens"], [])
        self.assertEqual(empty["encoding"]["tokens"], [])

    def test_encoder_output_can_be_passed_directly_to_decoder(self) -> None:
        payload = {
            "corpus": ["你好🙂你好🙂", "hello hello"],
            "vocab_size": 270,
            "text": "你好🙂 hello",
        }
        encoded = build_bpe_encode(payload)
        decoded = build_bpe_decode(
            {
                "corpus": payload["corpus"],
                "vocab_size": payload["vocab_size"],
                "tokens": encoded["encoding"]["tokens"],
            }
        )

        self.assertEqual(decoded["decoding"]["text"], payload["text"])
        self.assertTrue(decoded["decoding"]["is_canonical_encoding"])

    def test_invalid_encoder_payloads_are_rejected(self) -> None:
        invalid_payloads = (
            None,
            {},
            {"corpus": [], "vocab_size": 256, "text": "ok"},
            {"corpus": ["ok", 1], "vocab_size": 256, "text": "ok"},
            {"corpus": ["ok"], "vocab_size": 255, "text": "ok"},
            {"corpus": ["ok"], "vocab_size": True, "text": "ok"},
            {"corpus": ["ok"], "vocab_size": 256},
            {"corpus": ["ok"], "vocab_size": 256, "text": b"bad"},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(VisualizationInputError):
                    build_bpe_encode(payload)

    def test_encoder_rejects_a_trace_that_would_be_too_large(self) -> None:
        with self.assertRaisesRegex(VisualizationInputError, "trace 过大"):
            build_bpe_encode(
                {
                    "corpus": ["a" * 1024],
                    "vocab_size": 276,
                    "text": "b" * 20_000,
                }
            )


class BpeDecodeTests(unittest.TestCase):
    def test_decoder_uses_the_same_vocabulary_as_encoder(self) -> None:
        corpus = ["你好🙂你好🙂", "hello hello"]
        encoder = BytePairEncoder()
        encoder.train(corpus, vocab_size=270)
        text = "你好🙂 hello"
        tokens = encoder.encode(text)

        result = build_bpe_decode(
            {
                "corpus": corpus,
                "vocab_size": 270,
                "tokens": tokens,
            }
        )

        self.assertEqual(result["source"]["method"], "decode")
        self.assertEqual(result["decoding"]["text"], text)
        self.assertEqual(result["decoding"]["tokens"], tokens)
        self.assertEqual(
            result["decoding"]["decoded_bytes"],
            list(text.encode("utf-8")),
        )
        self.assertTrue(result["decoding"]["is_canonical_encoding"])
        self.assertFalse(result["decoding"]["used_replacement"])
        self.assertTrue(result["invariants"]["decode_matches_replace_policy"])
        self.assertTrue(result["invariants"]["decoded_bytes_match_text"])

    def test_known_noncanonical_ids_still_decode(self) -> None:
        result = build_bpe_decode(
            {
                "corpus": ["aaaa"],
                "vocab_size": 258,
                "tokens": [97, 97, 97, 97],
            }
        )

        self.assertEqual(result["decoding"]["text"], "aaaa")
        self.assertEqual(result["decoding"]["reencoded_tokens"], [257])
        self.assertFalse(result["decoding"]["is_canonical_encoding"])

    def test_empty_token_list_decodes_to_empty_string(self) -> None:
        result = build_bpe_decode(
            {
                "corpus": ["sample"],
                "vocab_size": 256,
                "tokens": [],
            }
        )

        self.assertEqual(result["decoding"]["text"], "")
        self.assertEqual(result["decoding"]["byte_count"], 0)

    def test_unknown_token_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(VisualizationInputError, r"tokens\[0\]=256"):
            build_bpe_decode(
                {
                    "corpus": ["sample"],
                    "vocab_size": 256,
                    "tokens": [256],
                }
            )

    def test_invalid_utf8_token_uses_replacement_character(self) -> None:
        result = build_bpe_decode(
            {
                "corpus": ["sample"],
                "vocab_size": 256,
                "tokens": [128],
            }
        )

        self.assertEqual(result["decoding"]["text"], "�")
        self.assertFalse(result["decoding"]["utf8_input_valid"])
        self.assertTrue(result["decoding"]["used_replacement"])
        self.assertTrue(result["invariants"]["decode_matches_replace_policy"])
        self.assertFalse(result["invariants"]["decoded_bytes_match_text"])

    def test_invalid_decoder_payloads_are_rejected(self) -> None:
        invalid_payloads = (
            None,
            {},
            {"corpus": [], "vocab_size": 256, "tokens": []},
            {"corpus": ["ok", 1], "vocab_size": 256, "tokens": []},
            {"corpus": ["ok"], "vocab_size": 255, "tokens": []},
            {"corpus": ["ok"], "vocab_size": True, "tokens": []},
            {"corpus": ["ok"], "vocab_size": 256},
            {"corpus": ["ok"], "vocab_size": 256, "tokens": [True]},
            {"corpus": ["ok"], "vocab_size": 256, "tokens": [1.5]},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(VisualizationInputError):
                    build_bpe_decode(payload)


class StaticFileTests(unittest.TestCase):
    def test_root_and_assets_resolve_inside_static_directory(self) -> None:
        self.assertEqual(resolve_static_file("/"), STATIC_ROOT / "index.html")
        self.assertEqual(
            resolve_static_file("/styles/system.css"),
            STATIC_ROOT / "styles" / "system.css",
        )
        self.assertEqual(
            resolve_static_file("/scripts/views/decoder.js"),
            STATIC_ROOT / "scripts" / "views" / "decoder.js",
        )
        self.assertEqual(
            resolve_static_file("/scripts/views/encoder.js"),
            STATIC_ROOT / "scripts" / "views" / "encoder.js",
        )
        self.assertEqual(
            resolve_static_file("/styles/encoder.css"),
            STATIC_ROOT / "styles" / "encoder.css",
        )

    def test_directory_traversal_is_rejected(self) -> None:
        self.assertIsNone(resolve_static_file("/../bpe.py"))
        self.assertIsNone(resolve_static_file("/%2e%2e/bpe.py"))


if __name__ == "__main__":
    unittest.main()
