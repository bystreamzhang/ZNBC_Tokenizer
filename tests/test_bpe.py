import unittest

from bpe import (
    BytePairEncoder,
    count_adjacent_pairs,
    merge_pair,
    utf8_tokens,
)


class Utf8TokenTests(unittest.TestCase):
    def test_unicode_text_becomes_utf8_byte_ids(self) -> None:
        text = "A你🙂"
        self.assertEqual(utf8_tokens(text), list(text.encode("utf-8")))
        self.assertTrue(all(0 <= token <= 255 for token in utf8_tokens(text)))

    def test_rejects_non_string_input(self) -> None:
        with self.assertRaises(TypeError):
            utf8_tokens(b"not text")  # type: ignore[arg-type]


class PairOperationTests(unittest.TestCase):
    def test_pair_counts_do_not_cross_sample_boundaries(self) -> None:
        counts = count_adjacent_pairs([[1, 2], [3, 4]])
        self.assertEqual(counts, {(1, 2): 1, (3, 4): 1})
        self.assertNotIn((2, 3), counts)

    def test_merge_is_non_overlapping_and_left_to_right(self) -> None:
        self.assertEqual(merge_pair([1, 1, 1], (1, 1), 256), [256, 1])


class BytePairEncoderTests(unittest.TestCase):
    def test_training_chooses_the_most_frequent_pair(self) -> None:
        encoder = BytePairEncoder()
        report = encoder.train("aaab", vocab_size=257)

        self.assertEqual(encoder.merges[0].pair, (ord("a"), ord("a")))
        self.assertEqual(encoder.merges[0].token_id, 256)
        self.assertEqual(encoder.merges[0].training_frequency, 2)
        self.assertEqual(report.merges_learned, 1)

    def test_frequency_tie_uses_lexicographically_smallest_pair(self) -> None:
        encoder = BytePairEncoder()
        encoder.train(["ba", "ab"], vocab_size=257)

        self.assertEqual(encoder.merges[0].pair, (ord("a"), ord("b")))

    def test_training_never_merges_across_strings(self) -> None:
        encoder = BytePairEncoder()
        report = encoder.train(["a", "a"], vocab_size=257)

        self.assertEqual(report.actual_vocab_size, 256)
        self.assertEqual(encoder.merges, ())

    def test_learned_merges_are_replayed_in_order(self) -> None:
        encoder = BytePairEncoder()
        report = encoder.train("aaaa", vocab_size=258)

        self.assertEqual(
            [merge.pair for merge in encoder.merges],
            [(ord("a"), ord("a")), (256, 256)],
        )
        self.assertEqual(encoder.encode("aaaa"), [257])
        self.assertEqual(encoder.vocabulary[257], b"aaaa")
        self.assertEqual(report.original_token_count, 4)
        self.assertEqual(report.final_token_count, 1)

    def test_unicode_bytes_can_be_reconstructed_from_vocabulary(self) -> None:
        encoder = BytePairEncoder()
        encoder.train(["你好你好", "你好"], vocab_size=264)

        tokens = encoder.encode("你好🙂")
        encoded_bytes = b"".join(encoder.vocabulary[token] for token in tokens)
        self.assertEqual(encoded_bytes.decode("utf-8"), "你好🙂")

    def test_train_resets_previous_merges(self) -> None:
        encoder = BytePairEncoder()
        encoder.train("aaaa", vocab_size=258)
        encoder.train("bb", vocab_size=257)

        self.assertEqual(len(encoder.merges), 1)
        self.assertEqual(encoder.merges[0].pair, (ord("b"), ord("b")))
        self.assertEqual(encoder.merges[0].token_id, 256)

    def test_analysis_can_hide_or_show_tokens(self) -> None:
        encoder = BytePairEncoder()
        encoder.train("aaaa", vocab_size=258)

        hidden = encoder.analyze("aaaa")
        shown = encoder.analyze("aaaa", include_tokens=True)

        self.assertEqual(hidden.utf8_byte_count, 4)
        self.assertEqual(hidden.token_count, 1)
        self.assertEqual(hidden.saved_token_count, 3)
        self.assertEqual(hidden.compression_ratio, 4.0)
        self.assertEqual(hidden.reduction_ratio, 0.75)
        self.assertIsNone(hidden.tokens)
        self.assertEqual(shown.tokens, (257,))
        self.assertNotIn("tokens", hidden.as_dict())
        self.assertEqual(shown.as_dict()["tokens"], [257])

    def test_empty_text_has_well_defined_metrics(self) -> None:
        encoder = BytePairEncoder()
        encoder.train([], vocab_size=260)
        report = encoder.analyze("")

        self.assertEqual(encoder.vocab_size, 256)
        self.assertEqual(report.compression_ratio, 1.0)
        self.assertEqual(report.reduction_ratio, 0.0)

    def test_vocab_size_is_validated(self) -> None:
        encoder = BytePairEncoder()
        for invalid_size in (255, 256.0, True):
            with self.subTest(invalid_size=invalid_size):
                with self.assertRaises(ValueError):
                    encoder.train("text", vocab_size=invalid_size)  # type: ignore[arg-type]

    def test_corpus_items_must_be_strings(self) -> None:
        encoder = BytePairEncoder()
        with self.assertRaisesRegex(TypeError, "corpus item 1"):
            encoder.train(["valid", b"invalid"], vocab_size=256)  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
