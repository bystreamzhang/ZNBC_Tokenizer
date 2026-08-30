"""把自研 tokenizer 的训练、分块、编解码和 golden 对照转换成 JSON。"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from threading import Lock
from typing import Any

try:
    from ...gpt4 import CL100K_SPECIAL_TOKENS, GPT4Tokenizer
    from ...regex import RegexTokenizer
except ImportError:  # 支持从仓库根目录按绝对包名加载。
    from tokenizers.own_gpt4.gpt4 import (  # type: ignore[no-redef]
        CL100K_SPECIAL_TOKENS,
        GPT4Tokenizer,
    )
    from tokenizers.own_gpt4.regex import RegexTokenizer  # type: ignore[no-redef]


# 训练实现为教学优先的逐轮全量扫描；前端限制计算量，避免一个合法请求长时间
# 占用 ThreadingHTTPServer 的 Python 执行线程。核心 ``train()`` 本身不受此限制。
MAX_VOCAB_SIZE = 512
MAX_TRAINING_UTF8_BYTES = 5_000
MAX_TEXT_UTF8_BYTES = 20_000
MAX_MERGES_IN_RESPONSE = 160
VALID_MODES = frozenset({"train", "gpt4"})
VALID_SPECIAL_POLICIES = frozenset({"none_raise", "all", "ordinary"})

_FIXED_TOKENIZER: GPT4Tokenizer | None = None
_FIXED_TOKENIZER_LOCK = Lock()


class VisualizationInputError(ValueError):
    """用户输入无法安全生成可视化数据。"""


def _strict_utf8(value: object, *, name: str, limit: int) -> tuple[str, bytes]:
    if not isinstance(value, str):
        raise VisualizationInputError(f"{name} 必须是 string。")
    try:
        raw_bytes = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise VisualizationInputError(
            f"{name} 必须只包含有效的 Unicode scalar values。"
        ) from error
    if len(raw_bytes) > limit:
        raise VisualizationInputError(
            f"{name} 最多包含 {limit} 个 UTF-8 bytes。"
        )
    return value, raw_bytes


def _validate_payload(
    payload: object,
) -> tuple[str, str, str, str | None, int | None]:
    if not isinstance(payload, Mapping):
        raise VisualizationInputError("请求内容必须是一个 JSON object。")

    mode = payload.get("mode")
    if mode not in VALID_MODES:
        raise VisualizationInputError("mode 必须是 train 或 gpt4。")

    special_policy = payload.get("special_policy")
    if special_policy not in VALID_SPECIAL_POLICIES:
        raise VisualizationInputError(
            "special_policy 必须是 none_raise、all 或 ordinary。"
        )

    text, _ = _strict_utf8(
        payload.get("text"),
        name="text",
        limit=MAX_TEXT_UTF8_BYTES,
    )

    common_fields = {"mode", "special_policy", "text"}
    if mode == "gpt4":
        unexpected = set(payload) - common_fields
        if unexpected:
            names = ", ".join(sorted(str(field) for field in unexpected))
            raise VisualizationInputError(
                f"gpt4 模式不接受这些字段：{names}。"
            )
        return mode, special_policy, text, None, None

    expected_fields = common_fields | {"training_text", "vocab_size"}
    unexpected = set(payload) - expected_fields
    if unexpected:
        names = ", ".join(sorted(str(field) for field in unexpected))
        raise VisualizationInputError(f"train 模式包含未知字段：{names}。")

    training_text, _ = _strict_utf8(
        payload.get("training_text"),
        name="training_text",
        limit=MAX_TRAINING_UTF8_BYTES,
    )
    vocab_size = payload.get("vocab_size")
    if (
        not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or not 256 <= vocab_size <= MAX_VOCAB_SIZE
    ):
        raise VisualizationInputError(
            f"vocab_size 必须是 256~{MAX_VOCAB_SIZE} 之间的整数。"
        )
    return mode, special_policy, text, training_text, vocab_size


def _safe_display(raw_bytes: bytes) -> str:
    """让 UTF-8 fragment 和控制字符都能安全出现在卡片中。"""

    text = raw_bytes.decode("utf-8", errors="replace")
    common_escapes = {"\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    escaped: list[str] = []
    for character in text:
        if character in common_escapes:
            escaped.append(common_escapes[character])
        elif character.isprintable():
            escaped.append(character)
        else:
            code_point = ord(character)
            if code_point <= 0xFF:
                escaped.append(f"\\x{code_point:02x}")
            elif code_point <= 0xFFFF:
                escaped.append(f"\\u{code_point:04x}")
            else:
                escaped.append(f"\\U{code_point:08x}")
    return "".join(escaped)


def _special_kwargs(policy: str) -> dict[str, object]:
    if policy == "all":
        return {"allowed_special": "all"}
    if policy == "ordinary":
        return {"disallowed_special": ()}
    return {}


def _fixed_tokenizer() -> GPT4Tokenizer:
    """single-flight 构造固定实例；server 请求只读取它，不修改训练状态。"""

    global _FIXED_TOKENIZER

    cached = _FIXED_TOKENIZER
    if cached is not None:
        return cached
    with _FIXED_TOKENIZER_LOCK:
        if _FIXED_TOKENIZER is None:
            _FIXED_TOKENIZER = GPT4Tokenizer()
        return _FIXED_TOKENIZER


@lru_cache(maxsize=1)
def _reference_encoding() -> Any:
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _piece_rows(
    tokenizer: RegexTokenizer,
    text: str,
    encode_kwargs: Mapping[str, object],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = 0
    actual_pieces = tokenizer.split_with_special_tokens(text, **encode_kwargs)
    for index, (piece, kind) in enumerate(actual_pieces):
        start = cursor
        cursor += len(piece)
        raw_bytes = piece.encode("utf-8")
        rows.append(
            {
                "index": index,
                "kind": kind,
                "text": piece,
                "display": _safe_display(raw_bytes),
                "char_start": start,
                "char_end": cursor,
                "utf8_byte_count": len(raw_bytes),
            }
        )
    return rows


def _merge_rows(tokenizer: RegexTokenizer) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered_merges = sorted(tokenizer.merges.items(), key=lambda item: item[1])
    for rank, (pair, token_id) in enumerate(
        ordered_merges[:MAX_MERGES_IN_RESPONSE],
        start=1,
    ):
        raw_bytes = tokenizer.token_bytes(token_id)
        rows.append(
            {
                "rank": rank,
                "pair": list(pair),
                "token_id": token_id,
                "bytes": list(raw_bytes),
                "bytes_hex": raw_bytes.hex(),
                "display": _safe_display(raw_bytes),
            }
        )
    return rows


def _token_rows(
    tokenizer: RegexTokenizer,
    token_ids: list[int],
) -> tuple[list[dict[str, Any]], bytes]:
    rows: list[dict[str, Any]] = []
    reconstructed = bytearray()
    for index, token_id in enumerate(token_ids):
        raw_bytes = tokenizer.token_bytes(token_id)
        byte_start = len(reconstructed)
        reconstructed.extend(raw_bytes)
        rows.append(
            {
                "index": index,
                "kind": (
                    "special"
                    if token_id in tokenizer.inverse_special_tokens
                    else "ordinary"
                ),
                "token_id": token_id,
                "byte_start": byte_start,
                "byte_end": len(reconstructed),
                "bytes": list(raw_bytes),
                "bytes_hex": raw_bytes.hex(),
                "display": _safe_display(raw_bytes),
            }
        )
    return rows, bytes(reconstructed)


def build_own_gpt4_overview(payload: object) -> dict[str, Any]:
    """执行一次训练或固定 GPT-4 编码，并返回页面需要的可验证数据。"""

    mode, special_policy, text, training_text, requested_vocab_size = (
        _validate_payload(payload)
    )
    if mode == "gpt4":
        tokenizer: RegexTokenizer = _fixed_tokenizer()
        training = None
    else:
        assert training_text is not None
        assert requested_vocab_size is not None
        tokenizer = RegexTokenizer()
        tokenizer.train(training_text, requested_vocab_size)
        # 高位的 cl100k special ids 不会与页面允许的自训练普通词表冲突。
        tokenizer.register_special_tokens(CL100K_SPECIAL_TOKENS)
        training = {
            "requested_vocab_size": requested_vocab_size,
            "actual_vocab_size": tokenizer.mergeable_vocab_size,
            "merge_count": len(tokenizer.merges),
            "training_utf8_byte_count": len(training_text.encode("utf-8")),
        }

    kwargs = _special_kwargs(special_policy)
    token_ids = tokenizer.encode(text, **kwargs)
    decoded_text = tokenizer.decode(token_ids)
    token_rows, reconstructed_bytes = _token_rows(tokenizer, token_ids)
    pieces = _piece_rows(tokenizer, text, kwargs)
    input_bytes = text.encode("utf-8")
    token_count = len(token_ids)

    reference: dict[str, Any] | None = None
    if mode == "gpt4":
        reference_encoding = _reference_encoding()
        reference_ids = list(reference_encoding.encode(text, **kwargs))
        reference_decoded = reference_encoding.decode(reference_ids)
        reference = {
            "ids": reference_ids,
            "decoded_text": reference_decoded,
            "ids_match": reference_ids == token_ids,
            "decode_match": reference_decoded == decoded_text,
        }

    return {
        "schema_version": 1,
        "source": {
            "module": (
                "tokenizers/own_gpt4/gpt4.py"
                if mode == "gpt4"
                else "tokenizers/own_gpt4/regex.py"
            ),
            "class": type(tokenizer).__name__,
            "method": "encode",
        },
        "configuration": {
            "mode": mode,
            "pattern": tokenizer.pattern,
            "vocab_size": tokenizer.vocab_size,
            "mergeable_vocab_size": tokenizer.mergeable_vocab_size,
            "merge_count": len(tokenizer.merges),
            "special_policy": special_policy,
            "special_tokens": dict(tokenizer.special_tokens),
        },
        "training": training,
        "pieces": pieces,
        "merges": _merge_rows(tokenizer),
        "metrics": {
            "utf8_byte_count": len(input_bytes),
            "token_count": token_count,
            "special_token_count": sum(
                token_id in tokenizer.inverse_special_tokens
                for token_id in token_ids
            ),
            "bytes_per_token": (
                len(input_bytes) / token_count if token_count else 0.0
            ),
        },
        "encoding": {
            "text": text,
            "ids": token_ids,
            "tokens": token_rows,
            "decoded_text": decoded_text,
        },
        "reference": reference,
        "invariants": {
            "pieces_rebuild_input": "".join(piece["text"] for piece in pieces)
            == text,
            "token_bytes_match_input": reconstructed_bytes == input_bytes,
            "decode_round_trip": decoded_text == text,
        },
    }


__all__ = [
    "MAX_MERGES_IN_RESPONSE",
    "MAX_TEXT_UTF8_BYTES",
    "MAX_TRAINING_UTF8_BYTES",
    "MAX_VOCAB_SIZE",
    "VisualizationInputError",
    "build_own_gpt4_overview",
]
