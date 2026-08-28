"""使用真实 BPE 词表生成 decoder 演示所需的数据。

浏览器只提交训练配置和 int token 列表。本 adapter 会重建与 encoder 运行
一致的确定性词表，再调用 ``BytePairEncoder.decode()``；JavaScript 不复制
Python decoder 的任何核心逻辑。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from ...bpe import BytePairEncoder
except ImportError:  # 支持在 basic_bpe 目录内运行本地测试。
    from bpe import BytePairEncoder  # type: ignore[no-redef]
from .bpe_trace import (
    MAX_CORPUS_UTF8_BYTES,
    MAX_VOCAB_SIZE,
    VisualizationInputError,
)


MAX_DECODE_TOKEN_COUNT = 20_000
MAX_DECODED_UTF8_BYTES = 100_000


def _validate_payload(payload: object) -> tuple[list[str], int, list[int]]:
    """验证 decoder API 输入，并返回已收窄类型的字段。"""

    if not isinstance(payload, Mapping):
        raise VisualizationInputError("请求内容必须是一个 JSON object。")

    corpus = payload.get("corpus")
    if not isinstance(corpus, list) or not corpus:
        raise VisualizationInputError("corpus 必须是至少包含一个 string 的列表。")
    if any(not isinstance(sample, str) for sample in corpus):
        raise VisualizationInputError("corpus 中的每一项都必须是 string。")

    vocab_size = payload.get("vocab_size")
    if (
        not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or not 256 <= vocab_size <= MAX_VOCAB_SIZE
    ):
        raise VisualizationInputError(
            f"vocab_size 必须是 256~{MAX_VOCAB_SIZE} 之间的整数。"
        )

    tokens = payload.get("tokens")
    if not isinstance(tokens, list):
        raise VisualizationInputError("tokens 必须是一个 JSON int 列表。")
    if len(tokens) > MAX_DECODE_TOKEN_COUNT:
        raise VisualizationInputError(
            f"tokens 最多包含 {MAX_DECODE_TOKEN_COUNT} 个 id。"
        )
    for index, token_id in enumerate(tokens):
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise VisualizationInputError(f"tokens[{index}] 必须是 int。")

    corpus_byte_count = sum(len(sample.encode("utf-8")) for sample in corpus)
    if corpus_byte_count > MAX_CORPUS_UTF8_BYTES:
        raise VisualizationInputError(
            f"训练样本最多包含 {MAX_CORPUS_UTF8_BYTES} 个 UTF-8 bytes。"
        )

    return corpus, vocab_size, tokens


def build_bpe_decode(payload: object) -> dict[str, Any]:
    """训练同一套 BPE 词表，并展示 int 列表的真实解码结果。"""

    corpus, vocab_size, tokens = _validate_payload(payload)

    encoder = BytePairEncoder()
    training_report = encoder.train(corpus, vocab_size=vocab_size)
    vocabulary = encoder.vocabulary

    token_pieces: list[dict[str, int | list[int]]] = []
    decoded_byte_count = 0
    for index, token_id in enumerate(tokens):
        try:
            piece = vocabulary[token_id]
        except KeyError as error:
            raise VisualizationInputError(
                f"tokens[{index}]={token_id} 不在当前 BPE 词表中。"
            ) from error

        decoded_byte_count += len(piece)
        if decoded_byte_count > MAX_DECODED_UTF8_BYTES:
            raise VisualizationInputError(
                f"token 展开后最多允许 {MAX_DECODED_UTF8_BYTES} 个 UTF-8 bytes。"
            )
        token_pieces.append(
            {
                "index": index,
                "token_id": token_id,
                "bytes": list(piece),
            }
        )

    decoded_bytes = b"".join(
        bytes(piece["bytes"]) for piece in token_pieces
    )
    # 最终 string 必须来自核心实现，而不是 adapter 自己重新实现 decoder。
    decoded_text = encoder.decode(tokens)

    # 严格解码只用于给演示层标记是否触发 replace，不改变核心返回结果。
    try:
        decoded_bytes.decode("utf-8")
        utf8_input_valid = True
    except UnicodeDecodeError:
        utf8_input_valid = False

    reencoded_tokens = encoder.encode(decoded_text)

    return {
        "schema_version": 1,
        "source": {
            "module": "bpe.py",
            "class": "BytePairEncoder",
            "method": "decode",
        },
        "training": training_report.as_dict(),
        "vocabulary": {
            str(token_id): list(token_bytes)
            for token_id, token_bytes in vocabulary.items()
        },
        "decoding": {
            "tokens": tokens,
            "token_count": len(tokens),
            "token_pieces": token_pieces,
            "decoded_bytes": list(decoded_bytes),
            "byte_count": len(decoded_bytes),
            "text": decoded_text,
            "utf8_input_valid": utf8_input_valid,
            "used_replacement": not utf8_input_valid,
            # Python len(str) 统计 Unicode code point，不等同于视觉字形数量。
            "python_length": len(decoded_text),
            "reencoded_tokens": reencoded_tokens,
            "is_canonical_encoding": reencoded_tokens == tokens,
        },
        "invariants": {
            "decode_matches_replace_policy": decoded_text
            == decoded_bytes.decode("utf-8", errors="replace"),
            "decoded_bytes_match_text": decoded_bytes
            == decoded_text.encode("utf-8"),
        },
    }
