import re
import unittest

from visualizations.adapters.split_overview import (
    VisualizationInputError,
    build_split_overview,
)
from visualizations.server import STATIC_ROOT, resolve_static_file


class SplitOverviewTests(unittest.TestCase):
    def test_response_exposes_policy_pieces_and_real_encoding(self) -> None:
        result = build_split_overview(
            {
                "corpus": ["dog.dog.", "abab|abab", "你好。你好。"],
                "vocab_size": 272,
                "protected_characters": " .?|。",
                "text": "dog.dog? | 你好。",
            }
        )

        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(
            result["source"]["class"],
            "SplitAwareBytePairEncoder",
        )
        self.assertIn("protected_pattern", result["split_policy"])
        self.assertIn("category_pattern", result["split_policy"])
        self.assertTrue(all(result["invariants"].values()))

        encoding = result["encoding"]
        self.assertEqual(
            "".join(piece["text"] for piece in encoding["pieces"]),
            encoding["text"],
        )
        self.assertEqual(
            [
                token
                for piece in encoding["pieces"]
                for token in piece["tokens"]
            ],
            encoding["tokens"],
        )
        for piece in encoding["pieces"]:
            if not piece["merge_allowed"]:
                self.assertEqual(piece["initial_tokens"], piece["tokens"])

    def test_unicode_protected_piece_bypasses_matching_merge(self) -> None:
        result = build_split_overview(
            {
                "corpus": ["佀佁佂佃佄"],
                "vocab_size": 257,
                "protected_characters": "你",
                "text": "你",
            }
        )

        self.assertEqual(result["merges"][0]["pair"], [228, 189])
        self.assertEqual(result["encoding"]["tokens"], [228, 189, 160])
        self.assertTrue(
            result["invariants"]["protected_pieces_unchanged"]
        )

    def test_invalid_payload_is_rejected(self) -> None:
        invalid_payloads = (
            None,
            {},
            {
                "corpus": ["ok"],
                "vocab_size": 256,
                "protected_characters": [],
                "text": "ok",
            },
            {
                "corpus": ["ok"],
                "vocab_size": 255,
                "protected_characters": ".",
                "text": "ok",
            },
            {
                "corpus": [""] * 201,
                "vocab_size": 256,
                "protected_characters": ".",
                "text": "ok",
            },
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(VisualizationInputError):
                    build_split_overview(payload)


class StaticFileTests(unittest.TestCase):
    def test_root_and_new_assets_resolve_inside_static_directory(self) -> None:
        self.assertEqual(resolve_static_file("/"), STATIC_ROOT / "split.html")
        self.assertEqual(
            resolve_static_file("/scripts/split-app.js"),
            STATIC_ROOT / "scripts" / "split-app.js",
        )
        self.assertEqual(
            resolve_static_file("/styles/split.css"),
            STATIC_ROOT / "styles" / "split.css",
        )

    def test_unknown_files_and_directory_traversal_are_rejected(self) -> None:
        self.assertIsNone(resolve_static_file("/index.html"))
        self.assertIsNone(resolve_static_file("/../bpe.py"))
        self.assertIsNone(resolve_static_file("/%2e%2e/bpe.py"))

    def test_javascript_dom_contract_and_api_route_match_page(self) -> None:
        html = (STATIC_ROOT / "split.html").read_text(encoding="utf-8")
        javascript = (
            STATIC_ROOT / "scripts" / "split-app.js"
        ).read_text(encoding="utf-8")
        html_ids = re.findall(r'\bid="([^"]+)"', html)
        referenced_ids = re.findall(r'byId\("([^"]+)"\)', javascript)

        self.assertEqual(len(html_ids), len(set(html_ids)))
        self.assertTrue(set(referenced_ids).issubset(html_ids))
        self.assertIn('fetch("/api/split-bpe/overview"', javascript)
        self.assertIn('id="protected-pattern"', html)
        self.assertIn('id="category-pattern"', html)
        self.assertIn('id="encoding-pieces"', html)

    def test_page_uses_compact_three_stage_information_architecture(
        self,
    ) -> None:
        html = (STATIC_ROOT / "split.html").read_text(encoding="utf-8")

        stage_markers = [
            'id="training-stage"',
            'id="encoding-stage"',
            'id="decoding-stage"',
        ]
        positions = [html.index(marker) for marker in stage_markers]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn('class="panel contract"', html)

        for details_id in (
            "strategy-details",
            "training-details",
            "merge-details",
        ):
            with self.subTest(details_id=details_id):
                match = re.search(
                    rf'<details\b[^>]*\bid="{details_id}"[^>]*>',
                    html,
                )
                self.assertIsNotNone(match)
                self.assertNotIn(" open", match.group(0))

    def test_large_results_have_explicit_frontend_display_limits(self) -> None:
        html = (STATIC_ROOT / "split.html").read_text(encoding="utf-8")
        javascript = (
            STATIC_ROOT / "scripts" / "split-app.js"
        ).read_text(encoding="utf-8")

        for constant in (
            "MAX_VISIBLE_PROTECTED_CHARACTERS",
            "MAX_VISIBLE_TRAINING_SAMPLES",
            "MAX_VISIBLE_PIECES_PER_SAMPLE",
            "MAX_VISIBLE_MERGES",
            "MAX_VISIBLE_ENCODING_PIECES",
        ):
            with self.subTest(constant=constant):
                self.assertRegex(
                    javascript,
                    rf"const {constant} = \d+;",
                )
                self.assertGreater(javascript.count(constant), 1)

        self.assertIn("samples.slice(0, MAX_VISIBLE_TRAINING_SAMPLES)", javascript)
        self.assertIn("merges.slice(0, MAX_VISIBLE_MERGES)", javascript)
        for note_id in (
            "protected-omitted-note",
            "training-omitted-note",
            "merge-omitted-note",
            "encoding-omitted-note",
        ):
            self.assertIn(f'id="{note_id}"', html)

    def test_styles_guard_against_page_level_horizontal_overflow(self) -> None:
        css = (STATIC_ROOT / "styles" / "split.css").read_text(
            encoding="utf-8"
        )

        self.assertRegex(
            css,
            r"(?s)\.panel\s*\{[^}]*min-width:\s*0",
        )
        self.assertRegex(
            css,
            r"(?s)\.piece-strip\s*\{[^}]*grid-template-columns:",
        )
        self.assertIn("overflow-x: clip", css)
        self.assertIn("@media (max-width: 760px)", css)
        self.assertIn("overflow-wrap: anywhere", css)


if __name__ == "__main__":
    unittest.main()
