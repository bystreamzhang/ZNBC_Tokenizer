import unittest

from pretokenizer import RegexPretokenizer


class RegexPretokenizerTests(unittest.TestCase):
    def test_split_is_continuous_lossless_and_explains_each_piece(self) -> None:
        splitter = RegexPretokenizer(" .?\n")
        text = " dog.\tdog?\r\n中文42🙂  "

        pieces = splitter.split(text)

        self.assertEqual("".join(piece.text for piece in pieces), text)
        self.assertEqual([piece.start for piece in pieces], [0, 1, 4, 5, 6, 9, 10, 11, 12, 14, 16, 17, 18])
        self.assertEqual(pieces[-1].end, len(text))
        self.assertTrue(all(piece.text for piece in pieces))
        self.assertTrue(all(
            left.end == right.start for left, right in zip(pieces, pieces[1:])
        ))

    def test_protected_characters_are_each_an_atomic_barrier(self) -> None:
        splitter = RegexPretokenizer(".? ")
        pieces = splitter.split("dog..??  dog")

        protected = [piece for piece in pieces if not piece.merge_allowed]
        self.assertEqual([piece.text for piece in protected], list("..??  "))
        self.assertTrue(all(piece.kind == "protected" for piece in protected))

    def test_regex_metacharacters_are_escaped(self) -> None:
        splitter = RegexPretokenizer("]-\\")

        pieces = splitter.split("a]-\\b")

        self.assertEqual("".join(piece.text for piece in pieces), "a]-\\b")
        self.assertEqual(
            [piece.text for piece in pieces if not piece.merge_allowed],
            ["]", "-", "\\"],
        )

    def test_no_english_contraction_presets_are_hidden_in_the_policy(self) -> None:
        splitter = RegexPretokenizer("")

        pieces = splitter.split("dog's I'M")

        self.assertEqual(
            [(piece.text, piece.kind) for piece in pieces],
            [
                ("dog", "letter"),
                ("'", "other"),
                ("s", "letter"),
                (" ", "whitespace"),
                ("I", "letter"),
                ("'", "other"),
                ("M", "letter"),
            ],
        )

    def test_empty_text_and_invalid_input(self) -> None:
        splitter = RegexPretokenizer()

        self.assertEqual(splitter.split(""), ())
        with self.assertRaises(TypeError):
            splitter.split(b"not text")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            RegexPretokenizer(["."])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
