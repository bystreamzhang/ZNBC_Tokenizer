"""只依赖 Python 标准库的 tiktoken GPT-4 本地可视化 server。"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import traceback
from typing import Any
from urllib.parse import unquote, urlsplit

from .adapters.tiktoken_overview import (
    VisualizationInputError,
    build_tiktoken_overview,
)


VISUALIZATIONS_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = VISUALIZATIONS_ROOT / "static"
MAX_REQUEST_BYTES = 1_000_000


def resolve_static_file(url_path: str) -> Path | None:
    """把 URL 映射到 static 内部文件，并拒绝目录穿越。"""

    decoded_path = unquote(urlsplit(url_path).path)
    relative_path = "index.html" if decoded_path == "/" else decoded_path.lstrip("/")
    candidate = (STATIC_ROOT / relative_path).resolve()
    if not candidate.is_relative_to(STATIC_ROOT) or not candidate.is_file():
        return None
    return candidate


class VisualizationRequestHandler(BaseHTTPRequestHandler):
    """提供静态页面和调用真实 GPT-4 tokenizer 的 JSON API。"""

    server_version = "ZNBCTokenizerVisualization/1.0"

    def _send_bytes(
        self,
        status: HTTPStatus,
        content: bytes,
        content_type: str,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; base-uri 'none'; "
            "form-action 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlsplit(self.path).path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model": "gpt-4",
                    "encoding": "cl100k_base",
                    "default_view": "tiktoken-gpt-4",
                },
            )
            return

        static_file = resolve_static_file(self.path)
        if static_file is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "页面不存在。"})
            return

        guessed_type, _ = mimetypes.guess_type(static_file.name)
        content_type = guessed_type or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {
            "application/javascript",
            "application/json",
        }:
            content_type += "; charset=utf-8"
        self._send_bytes(HTTPStatus.OK, static_file.read_bytes(), content_type)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        request_path = urlsplit(self.path).path
        if request_path != "/api/tiktoken-gpt-4/overview":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "API 不存在。"})
            return

        try:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise VisualizationInputError("请求缺少 Content-Length。")
            content_length = int(raw_length)
            if not 0 <= content_length <= MAX_REQUEST_BYTES:
                raise VisualizationInputError("请求内容过大。")

            raw_body = self.rfile.read(content_length)
            try:
                payload: Any = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise VisualizationInputError(
                    "请求不是有效的 UTF-8 JSON。"
                ) from error

            result = build_tiktoken_overview(payload)
        except VisualizationInputError as error:
            self._send_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": str(error)},
            )
            return
        except (TypeError, ValueError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except Exception:
            self.log_error(
                "unexpected error while building visualization data for %s\n%s",
                request_path,
                traceback.format_exc(),
            )
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "生成可视化数据时发生内部错误。"},
            )
            return

        self._send_json(HTTPStatus.OK, result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="启动 tiktoken GPT-4 本地可视化界面。"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址，默认仅本机可访问：127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8012,
        help="监听端口，默认：8012",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("port 必须是 1~65535 之间的整数。")

    server = ThreadingHTTPServer(
        (args.host, args.port),
        VisualizationRequestHandler,
    )
    url_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"Tokenizer 可视化界面：http://{url_host}:{args.port}/", flush=True)
    print("按 Ctrl+C 停止。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止可视化 server。", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
