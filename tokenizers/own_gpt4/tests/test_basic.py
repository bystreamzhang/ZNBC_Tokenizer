from __future__ import annotations

import unittest

from tokenizers.own_gpt4.basic import BasicTokenizer


class BasicTokenizerTrainingTests(unittest.TestCase):
    def test_train_selects_the_most_frequent_pair(self) -> None:
        tokenizer = BasicTokenizer()

        tokenizer.train("aaab", vocab_size=257)

        self.assertEqual(tokenizer.merges, {(ord("a"), ord("a")): 256})
        self.assertEqual(tokenizer.vocab[256], b"aa")

    def test_frequency_tie_has_a_deterministic_lexicographic_winner(self) -> None:
        tokenizer = BasicTokenizer()

        # 三个 pair 都只出现一次；结果不能依赖 hash seed 或进程环境。
        tokenizer.train("baab", vocab_size=257)

        self.assertEqual(tokenizer.merges, {(ord("a"), ord("a")): 256})

    def test_train_builds_a_merge_cascade_in_rank_order(self) -> None:
        tokenizer = BasicTokenizer()

        tokenizer.train("ababab", vocab_size=259)

        self.assertEqual(
            tokenizer.merges,
            {
                (ord("a"), ord("b")): 256,
                (256, 256): 257,
                (257, 256): 258,
            },
        )
        self.assertEqual(tokenizer.vocab[258], b"ababab")
        self.assertEqual(tokenizer.encode("ababab"), [258])

    def test_retraining_discards_old_merges_and_vocab_entries(self) -> None:
        tokenizer = BasicTokenizer()
        tokenizer.train("aaaa", vocab_size=258)

        tokenizer.train("bb", vocab_size=257)

        self.assertEqual(tokenizer.merges, {(ord("b"), ord("b")): 256})
        self.assertEqual(tokenizer.vocab[256], b"bb")
        self.assertNotIn(257, tokenizer.vocab)
        self.assertEqual(tokenizer.encode("aaaa"), [ord("a")] * 4)

    def test_training_stops_when_there_are_no_pairs(self) -> None:
        tokenizer = BasicTokenizer()

        tokenizer.train("x", vocab_size=300)

        self.assertEqual(tokenizer.merges, {})
        self.assertEqual(len(tokenizer.vocab), 256)

    def test_train_validates_text_and_vocab_size(self) -> None:
        tokenizer = BasicTokenizer()

        for invalid_text in (b"text", None, 123, True):
            with self.subTest(text=invalid_text):
                with self.assertRaises(TypeError):
                    tokenizer.train(invalid_text, 256)  # type: ignore[arg-type]

        for invalid_size in (255, 256.0, "257", True):
            with self.subTest(vocab_size=invalid_size):
                with self.assertRaises((TypeError, ValueError)):
                    tokenizer.train("text", invalid_size)  # type: ignore[arg-type]


class BasicTokenizerCodecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = BasicTokenizer()
        self.tokenizer.train(
            "你好你好🙂🙂 hello hello\r\ne\u0301é",
            vocab_size=280,
        )

    def test_unicode_emoji_whitespace_and_normalization_round_trip(self) -> None:
        texts = (
            "",
            "你好🙂 hello",
            "안녕하세요!\r\n第二行🚀",
            "e\u0301é",
            "\t  trailing  ",
        )

        for text in texts:
            with self.subTest(text=text):
                ids = self.tokenizer.encode(text)
                rebuilt = b"".join(self.tokenizer.vocab[token_id] for token_id in ids)
                self.assertEqual(rebuilt, text.encode("utf-8"))
                self.assertEqual(self.tokenizer.decode(ids), text)

    def test_decode_joins_bytes_before_decoding_utf8(self) -> None:
        self.assertEqual(self.tokenizer.decode(list("你".encode("utf-8"))), "你")

    def test_invalid_utf8_uses_the_replacement_character(self) -> None:
        self.assertEqual(self.tokenizer.decode([ord("A"), 0x80, ord("B")]), "A�B")

    def test_encode_rejects_non_string_input(self) -> None:
        for invalid in (b"text", None, 123, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    self.tokenizer.encode(invalid)  # type: ignore[arg-type]

    def test_decode_rejects_invalid_container_and_item_types(self) -> None:
        for invalid in ("97", b"97", iter([97])):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    self.tokenizer.decode(invalid)  # type: ignore[arg-type]

        for invalid in ([97, True], [97, 98.0], [97, "98"]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    self.tokenizer.decode(invalid)  # type: ignore[arg-type]

    def test_decode_rejects_unknown_and_negative_ids(self) -> None:
        for invalid in (-1, 280, 999_999):
            with self.subTest(token_id=invalid):
                with self.assertRaises(ValueError):
                    self.tokenizer.decode([invalid])


if __name__ == "__main__":
    unittest.main()
