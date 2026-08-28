import unittest

from bpe import SplitAwareBytePairEncoder


class SplitAwareBytePairEncoderTests(unittest.TestCase):
    def test_training_and_encode_share_the_same_protected_boundary(self) -> None:
        encoder = SplitAwareBytePairEncoder(protected_characters="|")
        report = encoder.train("ab|ab", vocab_size=258)

        self.assertEqual([merge.pair for merge in encoder.merges], [(97, 98)])
        self.assertEqual(encoder.encode("ab|ab"), [256, 124, 256])
        self.assertEqual(report.original_token_count, 5)
        self.assertEqual(report.final_token_count, 3)
        self.assertEqual(report.protected_piece_count, 1)

    def test_merge_cascade_stays_inside_each_piece(self) -> None:
        encoder = SplitAwareBytePairEncoder(protected_characters="|")
        encoder.train("abab|abab", vocab_size=258)

        self.assertEqual(
            [merge.pair for merge in encoder.merges],
            [(97, 98), (256, 256)],
        )
        self.assertEqual(encoder.encode("abab|abab"), [257, 124, 257])

    def test_default_policy_prevents_dog_punctuation_and_space_merges(self) -> None:
        encoder = SplitAwareBytePairEncoder()
        encoder.train(
            ["dog.dog.", "dog?dog?", "dog dog "],
            vocab_size=280,
        )

        for text in ("dog.dog.", "dog?dog?", "dog dog "):
            with self.subTest(text=text):
                pieces = encoder.encode_pieces(text)
                self.assertEqual("".join(piece.piece.text for piece in pieces), text)
                for piece in pieces:
                    if not piece.piece.merge_allowed:
                        self.assertEqual(
                            list(piece.tokens),
                            list(piece.piece.text.encode("utf-8")),
                        )
                self.assertEqual(encoder.decode(encoder.encode(text)), text)

    def test_protected_unicode_bypasses_an_existing_matching_byte_rule(self) -> None:
        encoder = SplitAwareBytePairEncoder(protected_characters="你")
        encoder.train("佀佁佂佃佄", vocab_size=257)

        self.assertEqual(encoder.merges[0].pair, (228, 189))
        self.assertEqual(encoder.encode("你"), [228, 189, 160])

    def test_protected_unicode_internal_bytes_never_merge(self) -> None:
        encoder = SplitAwareBytePairEncoder(protected_characters="你")
        report = encoder.train("你你", vocab_size=270)

        self.assertEqual(encoder.merges, ())
        self.assertEqual(report.actual_vocab_size, 256)
        self.assertEqual(
            encoder.encode("你你"),
            list("你你".encode("utf-8")),
        )

    def test_category_boundaries_apply_even_without_protected_characters(self) -> None:
        encoder = SplitAwareBytePairEncoder(protected_characters="")
        encoder.train("dog.dog.", vocab_size=270)

        composite_bytes = [
            value for token_id, value in encoder.vocabulary.items() if token_id >= 256
        ]
        self.assertNotIn(b"dog.", composite_bytes)
        self.assertEqual(
            [(piece.piece.text, piece.piece.kind) for piece in encoder.encode_pieces("dog.")],
            [("dog", "letter"), (".", "other")],
        )

    def test_round_trip_preserves_unicode_whitespace_and_normalization(self) -> None:
        encoder = SplitAwareBytePairEncoder(protected_characters=" .?\n你")
        encoder.train(
            ["hello hello", "中文中文", "🙂🙂", "e\u0301é"],
            vocab_size=280,
        )
        texts = ("", " dog.\tdog?\r\n中文42🙂  ", "你你", "e\u0301é")

        for text in texts:
            with self.subTest(text=text):
                tokens = encoder.encode(text)
                rebuilt = b"".join(encoder.vocabulary[token] for token in tokens)
                self.assertEqual(rebuilt, text.encode("utf-8"))
                self.assertEqual(encoder.decode(tokens), text)

    def test_encode_does_not_learn_from_new_input(self) -> None:
        encoder = SplitAwareBytePairEncoder(protected_characters="|")
        encoder.train("abab|abab", vocab_size=258)
        merges_before = encoder.merges
        vocabulary_before = encoder.vocabulary

        self.assertEqual(encoder.encode("zzzz|zzzz"), [122] * 4 + [124] + [122] * 4)
        self.assertEqual(encoder.merges, merges_before)
        self.assertEqual(encoder.vocabulary, vocabulary_before)


if __name__ == "__main__":
    unittest.main()
