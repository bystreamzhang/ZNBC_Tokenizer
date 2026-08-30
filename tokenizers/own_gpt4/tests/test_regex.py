from __future__ import annotations

import unittest

import regex

from tokenizers.own_gpt4.regex import GPT4_SPLIT_PATTERN, RegexTokenizer


class RegexBoundaryTests(unittest.TestCase):
    def test_gpt4_pattern_has_the_expected_boundary_examples(self) -> None:
        text = "I'm dog.dog 1234!!!\r\n中文🙂  "

        self.assertEqual(
            regex.findall(GPT4_SPLIT_PATTERN, text),
            ["I", "'m", " dog", ".dog", " ", "123", "4", "!!!\r\n", "中文", "🙂", "  "],
        )

    def test_training_never_merges_across_regex_piece_boundaries(self) -> None:
        tokenizer = RegexTokenizer()

        tokenizer.train("123412341234 dog.dog.dog", vocab_size=320)

        learned_bytes = tuple(
            token_bytes
            for token_id, token_bytes in tokenizer.vocab.items()
            if token_id >= 256
        )
        # 数字 piece 最长三个字符，因此 123+4 的边界不能被训练抹掉。
        self.assertNotIn(b"1234", learned_bytes)
        self.assertNotIn(b"dog.dog", learned_bytes)
        self.assertEqual(
            tokenizer.decode(tokenizer.encode("1234 dog.dog")),
            "1234 dog.dog",
        )

    def test_training_and_encoding_use_the_same_regex_boundaries(self) -> None:
        tokenizer = RegexTokenizer()
        tokenizer.train("ab ab", vocab_size=258)

        self.assertEqual(
            tokenizer.merges,
            {(ord("a"), ord("b")): 256, (ord(" "), 256): 257},
        )
        self.assertEqual(tokenizer.encode("ab ab"), [256, 257])

    def test_unicode_emoji_and_newlines_round_trip(self) -> None:
        tokenizer = RegexTokenizer()
        tokenizer.train("你好你好🙂🙂 안녕하세요\r\n", vocab_size=300)

        for text in ("", "你好🙂", "안녕하세요!\n第二行🚀", "e\u0301é  "):
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)


class SpecialTokenTests(unittest.TestCase):
    END = "<|endoftext|>"
    PREFIX = "<|fim_prefix|>"

    def setUp(self) -> None:
        self.tokenizer = RegexTokenizer()
        self.tokenizer.train("hello world", vocab_size=270)
        self.tokenizer.register_special_tokens(
            {
                self.END: 100257,
                self.PREFIX: 100258,
            }
        )

    def test_special_literal_is_rejected_by_default(self) -> None:
        with self.assertRaises(ValueError):
            self.tokenizer.encode(f"{self.END}hello")

    def test_allowed_special_all_recognizes_registered_tokens(self) -> None:
        ordinary_hello = self.tokenizer.encode("hello")

        ids = self.tokenizer.encode(
            f"{self.END}{self.PREFIX}hello{self.END}",
            allowed_special="all",
        )

        self.assertEqual(ids, [100257, 100258, *ordinary_hello, 100257])
        self.assertEqual(
            self.tokenizer.decode(ids),
            f"{self.END}{self.PREFIX}hello{self.END}",
        )

    def test_special_split_exposes_the_actual_encode_path(self) -> None:
        pieces = self.tokenizer.split_with_special_tokens(
            f"{self.END}hello world{self.PREFIX}",
            allowed_special="all",
        )

        self.assertEqual(
            pieces,
            [
                (self.END, "special"),
                ("hello", "ordinary"),
                (" world", "ordinary"),
                (self.PREFIX, "special"),
            ],
        )
        self.assertEqual("".join(piece for piece, _ in pieces), f"{self.END}hello world{self.PREFIX}")

    def test_explicit_allowed_set_only_recognizes_the_selected_token(self) -> None:
        ids = self.tokenizer.encode(
            f"hello{self.PREFIX}",
            allowed_special={self.PREFIX},
        )

        self.assertEqual(ids[-1], 100258)
        self.assertEqual(self.tokenizer.decode(ids), f"hello{self.PREFIX}")

    def test_disallowed_empty_tuple_encodes_special_as_ordinary_text(self) -> None:
        ids = self.tokenizer.encode(self.END, disallowed_special=())

        self.assertNotIn(100257, ids)
        self.assertEqual(self.tokenizer.decode(ids), self.END)

    def test_unlisted_registered_special_remains_disallowed(self) -> None:
        with self.assertRaises(ValueError):
            self.tokenizer.encode(
                f"{self.END}{self.PREFIX}",
                allowed_special={self.END},
            )

    def test_registration_exposes_forward_and_inverse_mappings(self) -> None:
        self.assertEqual(
            self.tokenizer.special_tokens,
            {self.END: 100257, self.PREFIX: 100258},
        )
        self.assertEqual(
            self.tokenizer.inverse_special_tokens,
            {100257: self.END, 100258: self.PREFIX},
        )


if __name__ == "__main__":
    unittest.main()
