from email.message import Message
from io import BytesIO
import json
import re
import unittest
from unittest.mock import patch

from visualizations.adapters.tiktoken_overview import (
    MAX_TEXT_UTF8_BYTES,
    VisualizationInputError,
    build_tiktoken_overview,
)
from visualizations.server import (
    STATIC_ROOT,
    VisualizationRequestHandler,
    resolve_static_file,
)


class TiktokenOverviewTests(unittest.TestCase):
    def test_schema_and_known_cl100k_golden_token(self) -> None:
        result = build_tiktoken_overview({"text": "hello"})

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(
            result["source"],
            {
                "module": "tokenizers/tiktoken_gpt4/tokenizer.py",
                "class": "GPT4Tokenizer",
                "method": "analyze",
            },
        )
        self.assertEqual(result["tokenizer"]["model"], "gpt-4")
        self.assertEqual(result["tokenizer"]["encoding"], "cl100k_base")
        self.assertGreater(result["tokenizer"]["vocab_size"], 100_000)
        self.assertEqual(result["metrics"]["utf8_byte_count"], 5)
        self.assertEqual(result["metrics"]["token_count"], 1)
        self.assertEqual(result["metrics"]["bytes_per_token"], 5.0)
        self.assertEqual(result["metrics"]["tokens_per_byte"], 0.2)

        encoding = result["encoding"]
        self.assertEqual(encoding["ids"], [15339])
        self.assertEqual(encoding["decoded_text"], "hello")
        self.assertEqual(
            encoding["tokens"],
            [
                {
                    "index": 0,
                    "token_id": 15339,
                    "byte_start": 0,
                    "byte_end": 5,
                    "bytes": [104, 101, 108, 108, 111],
                    "bytes_hex": "68656c6c6f",
                    "display": "hello",
                }
            ],
        )
        self.assertTrue(all(result["invariants"].values()))

    def test_unicode_rows_have_contiguous_byte_offsets(self) -> None:
        text = "A你🙂\n café"
        result = build_tiktoken_overview({"text": text})
        rows = result["encoding"]["tokens"]

        self.assertEqual(
            b"".join(bytes(row["bytes"]) for row in rows),
            text.encode("utf-8"),
        )
        cursor = 0
        for index, row in enumerate(rows):
            self.assertEqual(row["index"], index)
            self.assertEqual(row["byte_start"], cursor)
            cursor = row["byte_end"]
            self.assertEqual(cursor - row["byte_start"], len(row["bytes"]))
            self.assertIsInstance(row["bytes_hex"], str)
            self.assertIsInstance(row["display"], str)
        self.assertEqual(cursor, len(text.encode("utf-8")))
        self.assertEqual(result["encoding"]["decoded_text"], text)
        self.assertTrue(result["invariants"]["token_bytes_match_input"])
        self.assertTrue(result["invariants"]["decode_round_trip"])

    def test_special_token_literal_is_encoded_as_ordinary_text(self) -> None:
        text = "before <|endoftext|> after"
        result = build_tiktoken_overview({"text": text})

        self.assertEqual(result["encoding"]["decoded_text"], text)
        self.assertNotEqual(result["encoding"]["ids"], [100257])
        self.assertTrue(all(result["invariants"].values()))

    def test_empty_text_has_explicit_zero_metrics(self) -> None:
        result = build_tiktoken_overview({"text": ""})

        self.assertEqual(result["metrics"]["utf8_byte_count"], 0)
        self.assertEqual(result["metrics"]["token_count"], 0)
        self.assertEqual(result["metrics"]["bytes_per_token"], 0.0)
        self.assertEqual(result["metrics"]["tokens_per_byte"], 0.0)
        self.assertEqual(result["encoding"]["tokens"], [])
        self.assertEqual(result["encoding"]["ids"], [])
        self.assertTrue(all(result["invariants"].values()))

    def test_payload_contract_and_utf8_byte_limit(self) -> None:
        exact = build_tiktoken_overview({"text": "a" * MAX_TEXT_UTF8_BYTES})
        self.assertEqual(
            exact["metrics"]["utf8_byte_count"],
            MAX_TEXT_UTF8_BYTES,
        )

        invalid_payloads = (
            None,
            {},
            {"text": 1},
            {"text": "\ud800"},
            {"text": "ok", "model": "gpt-4"},
            {"text": "a" * (MAX_TEXT_UTF8_BYTES + 1)},
            {"text": "你" * (MAX_TEXT_UTF8_BYTES // 3 + 1)},
        )
        for payload in invalid_payloads:
            with self.subTest(payload_type=type(payload).__name__):
                with self.assertRaises(VisualizationInputError):
                    build_tiktoken_overview(payload)


class StaticFileTests(unittest.TestCase):
    def test_root_and_assets_resolve_inside_static_directory(self) -> None:
        self.assertEqual(resolve_static_file("/"), STATIC_ROOT / "index.html")
        self.assertEqual(
            resolve_static_file("/scripts/app.js"),
            STATIC_ROOT / "scripts" / "app.js",
        )
        self.assertEqual(
            resolve_static_file("/styles/system.css"),
            STATIC_ROOT / "styles" / "system.css",
        )
        self.assertEqual(
            resolve_static_file("/styles/tiktoken.css?version=1"),
            STATIC_ROOT / "styles" / "tiktoken.css",
        )

    def test_unknown_files_and_directory_traversal_are_rejected(self) -> None:
        self.assertIsNone(resolve_static_file("/missing.html"))
        self.assertIsNone(resolve_static_file("/../tokenizer.py"))
        self.assertIsNone(resolve_static_file("/%2e%2e/tokenizer.py"))
        self.assertIsNone(resolve_static_file("/styles/%2e%2e/%2e%2e/tokenizer.py"))

    def test_dom_contract_and_api_route_match(self) -> None:
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        javascript = (STATIC_ROOT / "scripts" / "app.js").read_text(
            encoding="utf-8"
        )
        html_ids = re.findall(r'\bid="([^"]+)"', html)
        referenced_ids = re.findall(r'byId\("([^"]+)"\)', javascript)

        self.assertEqual(len(html_ids), len(set(html_ids)))
        self.assertTrue(set(referenced_ids).issubset(html_ids))
        self.assertIn('fetch("/api/tiktoken-gpt-4/overview"', javascript)
        self.assertIn('id="token-list"', html)
        self.assertIn('id="token-ids-output"', html)
        self.assertIn('id="decoded-output"', html)

    def test_page_explains_fixed_mapping_and_raw_text_scope(self) -> None:
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

        self.assertIn("gpt-4 → cl100k_base", html)
        self.assertIn("ChatML", html)
        self.assertIn("OpenAI API", html)
        stage_markers = [
            'id="model-stage"',
            'id="encoding-stage"',
            'id="decoding-stage"',
        ]
        positions = [html.index(marker) for marker in stage_markers]
        self.assertEqual(positions, sorted(positions))

    def test_frontend_only_renders_python_data_and_limits_details(self) -> None:
        html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        javascript = (STATIC_ROOT / "scripts" / "app.js").read_text(
            encoding="utf-8"
        )

        self.assertRegex(javascript, r"const MAX_VISIBLE_TOKENS = 300;")
        self.assertGreater(javascript.count("MAX_VISIBLE_TOKENS"), 1)
        self.assertIn(
            "tokens.slice(0, MAX_VISIBLE_TOKENS)",
            javascript,
        )
        self.assertIn('id="token-omitted-note"', html)
        self.assertNotIn("TextEncoder", javascript)
        self.assertNotIn("innerHTML", javascript)

    def test_styles_are_responsive_and_guard_horizontal_overflow(self) -> None:
        system_css = (STATIC_ROOT / "styles" / "system.css").read_text(
            encoding="utf-8"
        )
        view_css = (STATIC_ROOT / "styles" / "tiktoken.css").read_text(
            encoding="utf-8"
        )

        self.assertRegex(system_css, r"(?s)\.panel\s*\{[^}]*min-width:\s*0")
        self.assertIn("overflow-x: clip", system_css)
        self.assertIn("overflow-wrap: anywhere", system_css)
        self.assertRegex(
            view_css,
            r"(?s)\.token-grid\s*\{[^}]*grid-template-columns:",
        )
        self.assertIn("@media (max-width: 760px)", view_css)

    def test_run_script_treats_port_as_decimal(self) -> None:
        run_script = (STATIC_ROOT.parent / "run.sh").read_text(
            encoding="utf-8"
        )

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

        raw_response = handler.wfile.getvalue()
        raw_headers, response_body = raw_response.split(b"\r\n\r\n", 1)
        header_lines = raw_headers.decode("iso-8859-1").split("\r\n")
        status = int(header_lines[0].split()[1])
        response_headers = {
            key.lower(): value.strip()
            for key, value in (
                line.split(":", 1) for line in header_lines[1:]
            )
        }
        return (
            status,
            response_headers,
            json.loads(response_body.decode("utf-8")),
        )

    def test_health_has_security_and_no_store_headers(self) -> None:
        status, headers, payload = self._request("GET", "/api/health")

        self.assertEqual(status, 200)
        self.assertEqual(payload["model"], "gpt-4")
        self.assertEqual(payload["encoding"], "cl100k_base")
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertIn("default-src 'self'", headers["content-security-policy"])

    def test_overview_and_expected_error_statuses(self) -> None:
        status, _, payload = self._request(
            "POST",
            "/api/tiktoken-gpt-4/overview",
            body=b'{"text":"hello"}',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["encoding"]["ids"], [15339])

        status, _, _ = self._request(
            "POST",
            "/api/tiktoken-gpt-4/overview",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 422)

        status, _, payload = self._request(
            "POST",
            "/api/tiktoken-gpt-4/overview",
            body=b'{"text":"\\ud800"}',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 422)
        self.assertIn("Unicode scalar values", payload["error"])

        status, _, _ = self._request(
            "POST",
            "/api/tiktoken-gpt-4/overview",
            body=b"{}",
            headers={"Content-Length": "not-an-integer"},
        )
        self.assertEqual(status, 400)

        with patch(
            "visualizations.server.build_tiktoken_overview",
            side_effect=RuntimeError("test-only failure"),
        ):
            status, _, payload = self._request(
                "POST",
                "/api/tiktoken-gpt-4/overview",
                body=b'{"text":"hello"}',
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(status, 500)
        self.assertIn("内部错误", payload["error"])


if __name__ == "__main__":
    unittest.main()
