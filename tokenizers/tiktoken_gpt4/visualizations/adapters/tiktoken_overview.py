"""把 ``GPT4Tokenizer.analyze`` 的结果转换成浏览器展示数据。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

try:
    from ...tokenizer import GPT4Tokenizer
except ImportError:  # 支持在 tiktoken_gpt4 目录内运行本地测试。
    from tokenizer import GPT4Tokenizer  # type: ignore[no-redef]


MAX_TEXT_UTF8_BYTES = 20_000


class VisualizationInputError(ValueError):
    """用户输入无法安全地生成 GPT-4 tokenizer 展示数据。"""


def _validate_payload(payload: object) -> str:
    if not isinstance(payload, Mapping):
        raise VisualizationInputError("请求内容必须是一个 JSON object。")

    unexpected_fields = set(payload) - {"text"}
    if unexpected_fields:
        names = ", ".join(sorted(str(field) for field in unexpected_fields))
        raise VisualizationInputError(f"请求只接受 text 字段；未知字段：{names}。")

    text = payload.get("text")
    if not isinstance(text, str):
        raise VisualizationInputError("text 必须是 string。")
    try:
        text_bytes = text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise VisualizationInputError(
            "text 必须只包含有效的 Unicode scalar values。"
        ) from error
    if len(text_bytes) > MAX_TEXT_UTF8_BYTES:
        raise VisualizationInputError(
            f"text 最多包含 {MAX_TEXT_UTF8_BYTES} 个 UTF-8 bytes。"
        )
    return text


def _as_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        result = as_dict()
        if isinstance(result, Mapping):
            return result
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError("GPT4Tokenizer.analyze 返回了无法展示的 token 记录。")


def _analysis_rows(analysis: object) -> Iterable[object]:
    if isinstance(analysis, Mapping):
        rows = analysis.get("token_details", analysis.get("details"))
    else:
        rows = getattr(
            analysis,
            "token_details",
            getattr(analysis, "details", analysis),
        )
    if isinstance(rows, (str, bytes, bytearray, Mapping)) or not isinstance(
        rows, Iterable
    ):
        raise TypeError("GPT4Tokenizer.analyze 必须返回可迭代的 token 记录。")
    return rows


def _normalise_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, int)
        and not isinstance(item, bool)
        and 0 <= item <= 255
        for item in value
    ):
        return bytes(value)
    raise TypeError("analyze token 的 bytes 必须是 bytes 或 byte 值列表。")


def _normalise_token_rows(analysis: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for expected_index, raw_row in enumerate(_analysis_rows(analysis)):
        row = _as_mapping(raw_row)
        token_id = row.get("token_id", row.get("id"))
        byte_start = row.get("byte_start")
        byte_end = row.get("byte_end")
        token_bytes = _normalise_bytes(row.get("bytes"))

        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TypeError("analyze token 缺少有效的 token_id。")
        if not isinstance(byte_start, int) or not isinstance(byte_end, int):
            raise TypeError("analyze token 缺少有效的 byte_start/byte_end。")
        if byte_start < 0 or byte_end < byte_start:
            raise ValueError("analyze token 返回了无效的 byte offsets。")

        raw_index = row.get("index", expected_index)
        if raw_index != expected_index:
            raise ValueError("analyze token index 必须从 0 连续递增。")

        bytes_hex = row.get("bytes_hex", token_bytes.hex(" "))
        display = row.get(
            "display",
            token_bytes.decode("utf-8", errors="replace"),
        )
        if not isinstance(bytes_hex, str) or not isinstance(display, str):
            raise TypeError("analyze token 的 bytes_hex/display 必须是 string。")

        rows.append(
            {
                "index": expected_index,
                "token_id": token_id,
                "byte_start": byte_start,
                "byte_end": byte_end,
                "bytes": list(token_bytes),
                "bytes_hex": bytes_hex,
                "display": display,
            }
        )
    return rows


def build_tiktoken_overview(payload: object) -> dict[str, Any]:
    """编码一次文本，并返回模型信息、逐 token bytes 与 round-trip。"""

    text = _validate_payload(payload)
    tokenizer = GPT4Tokenizer()

    # 逐 token 的边界和安全显示必须来自核心公开分析方法，而不是在前端重算。
    analysis = tokenizer.analyze(text)
    token_rows = _normalise_token_rows(analysis)
    token_ids = list(analysis.tokens)
    decoded_text = analysis.decoded_text
    input_bytes = text.encode("utf-8")
    rebuilt_bytes = b"".join(bytes(row["bytes"]) for row in token_rows)
    byte_count = len(input_bytes)
    token_count = len(token_ids)

    return {
        "schema_version": 1,
        "source": {
            "module": "tokenizers/tiktoken_gpt4/tokenizer.py",
            "class": "GPT4Tokenizer",
            "method": "analyze",
        },
        "tokenizer": {
            "model": tokenizer.model_name,
            "encoding": tokenizer.encoding_name,
            "vocab_size": tokenizer.vocab_size,
        },
        "metrics": {
            "utf8_byte_count": byte_count,
            "token_count": token_count,
            "bytes_per_token": (
                byte_count / token_count if token_count else 0.0
            ),
            "tokens_per_byte": (
                token_count / byte_count if byte_count else 0.0
            ),
        },
        "encoding": {
            "text": text,
            "tokens": token_rows,
            "ids": token_ids,
            "decoded_text": decoded_text,
        },
        "invariants": {
            "token_bytes_match_input": rebuilt_bytes == input_bytes,
            "decode_round_trip": decoded_text == text,
        },
    }
