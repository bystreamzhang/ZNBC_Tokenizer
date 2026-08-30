from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from io import BytesIO
import json
import re
import time
import unittest
from unittest.mock import patch

from tokenizers.own_gpt4.visualizations.adapters import own_gpt4_overview as overview_module
from tokenizers.own_gpt4.visualizations.adapters.own_gpt4_overview import (
    MAX_MERGES_IN_RESPONSE,
    MAX_TEXT_UTF8_BYTES,
    MAX_VOCAB_SIZE,
    VisualizationInputError,
    build_own_gpt4_overview,
)
from tokenizers.own_gpt4.visualizations.server import (
    STATIC_ROOT,
    VisualizationRequestHandler,
    resolve_static_file,
)


class OwnGPT4OverviewTests(unittest.TestCase):
    def test_fixed_frontend_tokenizer_is_single_flight(self) -> None:
        sentinel = object()
        calls = 0

        def delayed_constructor() -> object:
            nonlocal calls
            calls += 1
            time.sleep(0.02)
            return sentinel

        with overview_module._FIXED_TOKENIZER_LOCK:
            previous = overview_module._FIXED_TOKENIZER
            overview_module._FIXED_TOKENIZER = None
        try:
            with (
                patch.object(
                    overview_module,
                    "GPT4Tokenizer",
                    side_effect=delayed_constructor,
                ),
                ThreadPoolExecutor(max_workers=8) as executor,
            ):
                results = list(
                    executor.map(lambda _: overview_module._fixed_tokenizer(), range(8))
                )
        finally:
            with overview_module._FIXED_TOKENIZER_LOCK:
                overview_module._FIXED_TOKENIZER = previous

        self.assertEqual(calls, 1)
        self.assertTrue(all(result is sentinel for result in results))

    def test_gpt4_mode_matches_reference_and_exposes_special_token(self) -> None:
        text = "<|endoftext|>hello world"
        result = build_own_gpt4_overview(
            {
                "mode": "gpt4",
                "special_policy": "all",
                "text": text,
            }
        )

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["source"]["class"], "GPT4Tokenizer")
        self.assertEqual(result["encoding"]["ids"], [100257, 15339, 1917])
        self.assertEqual(result["encoding"]["decoded_text"], text)
        self.assertEqual(result["encoding"]["tokens"][0]["kind"], "special")
        self.assertEqual(result["metrics"]["special_token_count"], 1)
        self.assertTrue(result["reference"]["ids_match"])
        self.assertTrue(result["reference"]["decode_match"])
        self.assertTrue(all(result["invariants"].values()))
        self.assertEqual(
            [(piece["text"], piece["kind"]) for piece in result["pieces"]],
            [
                ("<|endoftext|>", "special"),
                ("hello", "ordinary"),
                (" world", "ordinary"),
            ],
        )

    def test_train_mode_uses_real_train_and_has_no_tiktoken_reference(self) -> None:
        text = "ab ab 你好🙂"
        result = build_own_gpt4_overview(
            {
                "mode": "train",
                "special_policy": "none_raise",
                "training_text": "ab ab ab 你好你好",
                "vocab_size": 264,
                "text": text,
            }
        )

        self.assertEqual(result["source"]["class"], "RegexTokenizer")
        self.assertEqual(result["training"]["requested_vocab_size"], 264)
        self.assertGreater(result["training"]["merge_count"], 0)
        self.assertEqual(result["configuration"]["merge_count"], len(result["merges"]))
        self.assertEqual(result["reference"], None)
        self.assertEqual(result["encoding"]["decoded_text"], text)
        self.assertEqual("".join(row["text"] for row in result["pieces"]), text)
        self.assertTrue(all(result["invariants"].values()))

    def test_special_literal_can_be_ordinary_or_rejected(self) -> None:
        text = "<|endoftext|>hello"
        ordinary = build_own_gpt4_overview(
            {
                "mode": "gpt4",
                "special_policy": "ordinary",
                "text": text,
            }
        )

        self.assertNotIn(100257, ordinary["encoding"]["ids"])
        self.assertEqual(ordinary["encoding"]["decoded_text"], text)
        self.assertTrue(ordinary["reference"]["ids_match"])
        with self.assertRaisesRegex(ValueError, "disallowed special token"):
            build_own_gpt4_overview(
                {
                    "mode": "gpt4",
                    "special_policy": "none_raise",
                    "text": text,
                }
            )

    def test_merge_response_is_bounded_but_reports_full_count(self) -> None:
        result = build_own_gpt4_overview(
            {"mode": "gpt4", "special_policy": "none_raise", "text": "hello"}
        )

        self.assertEqual(len(result["merges"]), MAX_MERGES_IN_RESPONSE)
        self.assertEqual(result["configuration"]["merge_count"], 100_000)
        self.assertGreater(result["configuration"]["merge_count"], len(result["merges"]))

    def test_token_rows_have_contiguous_byte_offsets(self) -> None:
        text = "A你🙂\n"
        result = build_own_gpt4_overview(
            {"mode": "gpt4", "special_policy": "none_raise", "text": text}
        )
        cursor = 0
        rebuilt = bytearray()
        for index, row in enumerate(result["encoding"]["tokens"]):
            self.assertEqual(row["index"], index)
            self.assertEqual(row["byte_start"], cursor)
            rebuilt.extend(row["bytes"])
            cursor = row["byte_end"]
        self.assertEqual(bytes(rebuilt), text.encode("utf-8"))
        self.assertEqual(cursor, len(text.encode("utf-8")))

    def test_invalid_payloads_are_rejected(self) -> None:
        invalid_payloads = (
            None,
            {},
            {"mode": "bad", "special_policy": "all", "text": "ok"},
            {"mode": "gpt4", "special_policy": "bad", "text": "ok"},
            {"mode": "gpt4", "special_policy": "all", "text": "ok", "vocab_size": 300},
            {
                "mode": "train",
                "special_policy": "all",
                "text": "ok",
                "training_text": "ok",
                "vocab_size": 255,
            },
            {
                "mode": "train",
                "special_policy": "all",
                "text": "ok",
                "training_text": "ok",
                "vocab_size": True,
            },
            {
                "mode": "train",
                "special_policy": "all",
                "text": "ok",
                "training_text": "ok",
                "vocab_size": MAX_VOCAB_SIZE + 1,
            },
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(VisualizationInputError):
                    build_own_gpt4_overview(payload)

    def test_text_byte_limit_is_enforced(self) -> None:
        with self.assertRaisesRegex(VisualizationInputError, str(MAX_TEXT_UTF8_BYTES)):
            build_own_gpt4_overview(
                {
                    "mode": "gpt4",
                    "special_policy": "all",
                    "text": "a" * (MAX_TEXT_UTF8_BYTES + 1),
                }
            )


class StaticFileTests(unittest.TestCase):
    def test_root_and_assets_resolve_inside_static_directory(self) -> None:
        self.assertEqual(resolve_static_file("/"), STATIC_ROOT / "index.html")
        self.assertEqual(
            resolve_static_file("/scripts/app.js"),
            STATIC_ROOT / "scripts" / "app.js",
        )
        self.assertEqual(
            resolve_static_file("/styles/own-gpt4.css"),
            STATIC_ROOT / "styles" / "own-gpt4.css",
        )

    def test_unknown_files_and_directory_traversal_are_rejected(self) -> None:
        self.assertIsNone(resolve_static_file("/missing.html"))
        self.assertIsNone(resolve_static_file("/../gpt4.py"))
        self.assertIsNone(resolve_static_file("/%2e%2e/gpt4.py"))
        self.assertIsNone(resolve_static_file("/styles/%2e%2e/%2e%2e/gpt4.py"))

    def test_dom_contract_and_api_route_match(self) -> None:
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        javascript = (STATIC_ROOT / "scripts" / "app.js").read_text(encoding="utf-8")
        html_ids = re.findall(r'\bid="([^"]+)"', html)
        referenced_ids = re.findall(r'byId\("([^"]+)"\)', javascript)

        self.assertEqual(len(html_ids), len(set(html_ids)))
        self.assertTrue(set(referenced_ids).issubset(html_ids))
        self.assertIn('fetch("/api/own-gpt4/overview"', javascript)
        for required_id in (
            "training-config",
            "piece-list",
            "merge-table-body",
            "token-list",
            "reference-stage",
            "decoded-output",
        ):
            self.assertIn(f'id="{required_id}"', html)

    def test_frontend_only_renders_python_data_and_limits_details(self) -> None:
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        javascript = (STATIC_ROOT / "scripts" / "app.js").read_text(encoding="utf-8")
        for constant in (
            "MAX_VISIBLE_PIECES",
            "MAX_VISIBLE_MERGES",
            "MAX_VISIBLE_TOKENS",
        ):
            self.assertRegex(javascript, rf"const {constant} = \d+;")
            self.assertGreater(javascript.count(constant), 1)
        self.assertIn("pieces.slice(0, MAX_VISIBLE_PIECES)", javascript)
        self.assertIn("merges.slice(0, MAX_VISIBLE_MERGES)", javascript)
        self.assertIn("encoding.tokens.slice(0, MAX_VISIBLE_TOKENS)", javascript)
        self.assertNotIn("TextEncoder", javascript)
        self.assertNotIn("innerHTML", javascript)
        self.assertIn('id="token-omitted-note"', html)

    def test_styles_are_responsive_and_guard_horizontal_overflow(self) -> None:
        css = (STATIC_ROOT / "styles" / "own-gpt4.css").read_text(encoding="utf-8")

        self.assertRegex(css, r"(?s)\.panel\s*\{[^}]*min-width:\s*0")
        self.assertRegex(css, r"(?s)\.token-grid\s*\{[^}]*grid-template-columns:")
        self.assertIn("overflow-x: clip", css)
        self.assertIn("overflow-wrap: anywhere", css)
        self.assertIn("@media (max-width: 760px)", css)

    def test_run_script_treats_port_as_decimal(self) -> None:
        run_script = (STATIC_ROOT.parent / "run.sh").read_text(encoding="utf-8")

        self.assertIn("port_number=$((10#${port}))", run_script)
        self.assertIn('--port "${port_number}"', run_script)


class _QuietHandler(VisualizationRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def log_error(self, format: str, *args: object) -> None:
        pass


class ServerTests(unittest.TestCase):
    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], dict[str, object]]:
        request_body = body or b""
        request_headers = Message()
        for key, value in (headers or {}).items():
            request_headers[key] = value
        if "Content-Length" not in request_headers and method == "POST":
            request_headers["Content-Length"] = str(len(request_body))

        handler = _QuietHandler.__new__(_QuietHandler)
        handler.command = method
        handler.path = path
        handler.request_version = "HTTP/1.1"
        handler.requestline = f"{method} {path} HTTP/1.1"
        handler.client_address = ("127.0.0.1", 0)
        handler.headers = request_headers
        handler.rfile = BytesIO(request_body)
        handler.wfile = BytesIO()

        if method == "GET":
            handler.do_GET()
        else:
            handler.do_POST()

        raw_headers, response_body = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
        header_lines = raw_headers.decode("iso-8859-1").split("\r\n")
        status = int(header_lines[0].split()[1])
        response_headers = {
            key.lower(): value.strip()
            for key, value in (line.split(":", 1) for line in header_lines[1:])
        }
        return status, response_headers, json.loads(response_body.decode("utf-8"))

    def test_health_has_security_and_no_store_headers(self) -> None:
        status, headers, payload = self._request("GET", "/api/health")

        self.assertEqual(status, 200)
        self.assertEqual(payload["variant"], "own-gpt4")
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertIn("default-src 'self'", headers["content-security-policy"])

    def test_overview_and_expected_error_statuses(self) -> None:
        status, _, payload = self._request(
            "POST",
            "/api/own-gpt4/overview",
            body=json.dumps(
                {"mode": "gpt4", "special_policy": "none_raise", "text": "hello"}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["encoding"]["ids"], [15339])

        status, _, _ = self._request(
            "POST",
            "/api/own-gpt4/overview",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 422)

        status, _, _ = self._request(
            "POST",
            "/api/own-gpt4/overview",
            body=b"{}",
            headers={"Content-Length": "not-an-integer"},
        )
        self.assertEqual(status, 400)

        with patch(
            "tokenizers.own_gpt4.visualizations.server.build_own_gpt4_overview",
            side_effect=RuntimeError("test-only failure"),
        ):
            status, _, payload = self._request(
                "POST",
                "/api/own-gpt4/overview",
                body=json.dumps(
                    {"mode": "gpt4", "special_policy": "all", "text": "hello"}
                ).encode(),
            )
        self.assertEqual(status, 500)
        self.assertIn("内部错误", payload["error"])


if __name__ == "__main__":
    unittest.main()
