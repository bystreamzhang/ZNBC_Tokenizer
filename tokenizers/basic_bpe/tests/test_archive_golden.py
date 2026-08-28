import unittest

from bpe import BytePairEncoder


class BasicBpeArchiveGoldenTests(unittest.TestCase):
    def test_archive_preserves_cross_category_merge_behavior(self) -> None:
        encoder = BytePairEncoder()
        encoder.train("dog.dog.", vocab_size=260)

        self.assertEqual(encoder.encode("dog.dog."), [259])
        self.assertEqual(encoder.vocabulary[259], b"dog.dog.")

    def test_archive_still_round_trips_unicode(self) -> None:
        encoder = BytePairEncoder()
        encoder.train(["你好你好", "hello hello"], vocab_size=270)
        text = "你好🙂 hello"

        self.assertEqual(encoder.decode(encoder.encode(text)), text)


if __name__ == "__main__":
    unittest.main()
