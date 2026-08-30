from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from unittest import mock
import unittest

import tiktoken

from tokenizers.own_gpt4 import gpt4 as gpt4_module
from tokenizers.own_gpt4.gpt4 import GPT4Tokenizer, recover_merges


EXERCISE_TEXT = "hello world!!!? (안녕하세요!) lol123 😉"
EXERCISE_IDS = [
    15339,
    1917,
    12340,
    30,
    320,
    31495,
    230,
    75265,
    243,
    92245,
    16715,
    28509,
    4513,
    57037,
]


class RecoverMergesTests(unittest.TestCase):
    def test_recovers_parent_pairs_from_ranked_token_bytes(self) -> None:
        mergeable_ranks = {
            b"a": 0,
            b"b": 1,
            b"c": 2,
            b"ab": 3,
            b"abc": 4,
        }

        self.assertEqual(
            recover_merges(mergeable_ranks),
            {(0, 1): 3, (3, 2): 4},
        )

    def test_cl100k_cache_is_single_flight_during_concurrent_cold_start(self) -> None:
        sentinel = {"test": "cl100k-data"}
        calls = 0

        def delayed_recover() -> dict[str, str]:
            nonlocal calls
            calls += 1
            time.sleep(0.02)
            return sentinel

        with gpt4_module._CL100K_DATA_LOCK:
            previous = gpt4_module._CL100K_DATA
            gpt4_module._CL100K_DATA = None
        try:
            with (
                mock.patch.object(
                    gpt4_module,
                    "_recover_cl100k_data",
                    side_effect=delayed_recover,
                ),
                ThreadPoolExecutor(max_workers=8) as executor,
            ):
                results = list(
                    executor.map(lambda _: gpt4_module._load_cl100k_data(), range(8))
                )
        finally:
            with gpt4_module._CL100K_DATA_LOCK:
                gpt4_module._CL100K_DATA = previous

        self.assertEqual(calls, 1)
        self.assertTrue(all(result is sentinel for result in results))


class GPT4TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 恢复 cl100k_base 的十万条规则较重，整个测试类只执行一次。
        cls.tokenizer = GPT4Tokenizer()
        cls.reference = tiktoken.get_encoding("cl100k_base")

    def test_exercise_example_matches_the_documented_golden_ids(self) -> None:
        ids = self.tokenizer.encode(EXERCISE_TEXT)

        self.assertEqual(ids, EXERCISE_IDS)
        self.assertEqual(self.tokenizer.decode(ids), EXERCISE_TEXT)

    def test_multiple_inputs_match_tiktoken_encode_and_decode(self) -> None:
        texts = (
            "",
            "hello world",
            "你好，world 🙂\n第二行🚀",
            "I'm testing 1234567...\r\n",
            "e\u0301 and é are different Unicode sequences",
            "\t leading and trailing spaces   ",
        )

        for text in texts:
            with self.subTest(text=text):
                expected_ids = self.reference.encode(text)
                actual_ids = self.tokenizer.encode(text)
                self.assertEqual(actual_ids, expected_ids)
                self.assertEqual(
                    self.tokenizer.decode(actual_ids),
                    self.reference.decode(expected_ids),
                )

    def test_byte_shuffle_is_a_complete_bijection(self) -> None:
        shuffle = dict(self.tokenizer.byte_shuffle)
        inverse = dict(self.tokenizer.inverse_byte_shuffle)

        self.assertEqual(set(shuffle), set(range(256)))
        self.assertEqual(set(shuffle.values()), set(range(256)))
        self.assertEqual(set(inverse), set(range(256)))
        self.assertEqual(set(inverse.values()), set(range(256)))
        for raw_byte, shuffled_byte in shuffle.items():
            with self.subTest(raw_byte=raw_byte):
                self.assertEqual(inverse[shuffled_byte], raw_byte)

    def test_all_mergeable_token_bytes_match_tiktoken(self) -> None:
        # 逐个检查词表可同时发现 merge 恢复和 byte unshuffle 的细小偏差。
        for token_id in range(100256):
            with self.subTest(token_id=token_id):
                self.assertEqual(
                    self.tokenizer.token_bytes(token_id),
                    self.reference.decode_single_token_bytes(token_id),
                )

    def test_special_token_ids_and_all_mode_match_tiktoken(self) -> None:
        expected_specials = {
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        }
        self.assertEqual(self.tokenizer.special_tokens, expected_specials)

        text = "<|endoftext|>hello<|fim_middle|>world<|endofprompt|>"
        actual = self.tokenizer.encode(text, allowed_special="all")
        expected = self.reference.encode(text, allowed_special="all")

        self.assertEqual(actual, expected)
        self.assertEqual(self.tokenizer.decode(actual), text)

    def test_special_literal_is_rejected_by_default(self) -> None:
        with self.assertRaises(ValueError):
            self.tokenizer.encode("<|endoftext|>hello world")

    def test_explicit_special_set_matches_tiktoken(self) -> None:
        text = "prefix <|fim_prefix|>suffix"
        allowed = {"<|fim_prefix|>"}

        self.assertEqual(
            self.tokenizer.encode(text, allowed_special=allowed),
            self.reference.encode(text, allowed_special=allowed),
        )

    def test_special_literal_can_be_encoded_as_ordinary_text(self) -> None:
        text = "<|endoftext|>hello world"

        actual = self.tokenizer.encode(text, disallowed_special=())
        expected = self.reference.encode(text, disallowed_special=())

        self.assertEqual(actual, expected)
        self.assertNotIn(100257, actual)
        self.assertEqual(self.tokenizer.decode(actual), text)

    def test_encode_and_decode_do_not_delegate_to_tiktoken_methods(self) -> None:
        with (
            mock.patch.object(
                tiktoken.Encoding,
                "encode",
                side_effect=AssertionError("must not delegate encode"),
            ),
            mock.patch.object(
                tiktoken.Encoding,
                "decode",
                side_effect=AssertionError("must not delegate decode"),
            ),
        ):
            ids = self.tokenizer.encode("hello world")
            text = self.tokenizer.decode(ids)

        self.assertEqual(ids, [15339, 1917])
        self.assertEqual(text, "hello world")

    def test_fixed_gpt4_tokenizer_rejects_training(self) -> None:
        with self.assertRaises(RuntimeError):
            self.tokenizer.train("new corpus", vocab_size=300)

    def test_invalid_and_reserved_gap_ids_are_rejected(self) -> None:
        for token_id in (-1, 100256, 100261, 999_999):
            with self.subTest(token_id=token_id):
                with self.assertRaises(ValueError):
                    self.tokenizer.token_bytes(token_id)
                with self.assertRaises(ValueError):
                    self.tokenizer.decode([token_id])

    def test_decode_and_token_bytes_reject_invalid_types(self) -> None:
        for invalid in (True, 1.0, "1", None):
            with self.subTest(token_id=invalid):
                with self.assertRaises(TypeError):
                    self.tokenizer.token_bytes(invalid)  # type: ignore[arg-type]

        for invalid in ("15339", b"15339", iter([15339])):
            with self.subTest(tokens=invalid):
                with self.assertRaises(TypeError):
                    self.tokenizer.decode(invalid)  # type: ignore[arg-type]

        for invalid in ([15339, True], [15339, 1917.0], [15339, "1917"]):
            with self.subTest(tokens=invalid):
                with self.assertRaises(TypeError):
                    self.tokenizer.decode(invalid)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
