"""把 split-aware BPE 的真实策略、pieces 和输出转换成前端 JSON。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from ...bpe import SplitAwareBytePairEncoder
except ImportError:  # 支持在 split_bpe 目录内运行本地测试。
    from bpe import SplitAwareBytePairEncoder  # type: ignore[no-redef]


MAX_VOCAB_SIZE = 1024
MAX_CORPUS_UTF8_BYTES = 20_000
MAX_TEXT_UTF8_BYTES = 20_000
MAX_PROTECTED_CHARACTERS = 128
MAX_CORPUS_SAMPLES = 200


class VisualizationInputError(ValueError):
    """用户输入无法安全地生成 split-aware BPE 展示数据。"""


def _validate_payload(
    payload: object,
) -> tuple[list[str], int, str, str]:
    if not isinstance(payload, Mapping):
        raise VisualizationInputError("请求内容必须是一个 JSON object。")

    corpus = payload.get("corpus")
    if not isinstance(corpus, list) or not corpus:
        raise VisualizationInputError("corpus 必须是至少包含一个 string 的列表。")
    if len(corpus) > MAX_CORPUS_SAMPLES:
        raise VisualizationInputError(
            f"corpus 最多包含 {MAX_CORPUS_SAMPLES} 个 samples。"
        )
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

    text = payload.get("text")
    if not isinstance(text, str):
        raise VisualizationInputError("text 必须是 string。")

    protected_characters = payload.get("protected_characters")
    if not isinstance(protected_characters, str):
        raise VisualizationInputError(
            "protected_characters 必须是一个 string。"
        )
    if len(protected_characters) > MAX_PROTECTED_CHARACTERS:
        raise VisualizationInputError(
            "protected_characters 最多包含 "
            f"{MAX_PROTECTED_CHARACTERS} 个 Unicode code points。"
        )

    corpus_byte_count = sum(len(sample.encode("utf-8")) for sample in corpus)
    if corpus_byte_count > MAX_CORPUS_UTF8_BYTES:
        raise VisualizationInputError(
            f"训练样本最多包含 {MAX_CORPUS_UTF8_BYTES} 个 UTF-8 bytes。"
        )
    if len(text.encode("utf-8")) > MAX_TEXT_UTF8_BYTES:
        raise VisualizationInputError(
            f"待编码文本最多包含 {MAX_TEXT_UTF8_BYTES} 个 UTF-8 bytes。"
        )

    return corpus, vocab_size, text, protected_characters


def _encoded_piece_payload(
    encoder: SplitAwareBytePairEncoder,
    text: str,
) -> list[dict[str, Any]]:
    return [piece.as_dict() for piece in encoder.encode_pieces(text)]


def build_split_overview(payload: object) -> dict[str, Any]:
    """训练一次 split-aware BPE，并返回页面所需的完整、可验证数据。"""

    corpus, vocab_size, text, protected_characters = _validate_payload(payload)
    encoder = SplitAwareBytePairEncoder(
        protected_characters=protected_characters,
    )
    training_report = encoder.train(corpus, vocab_size=vocab_size)
    vocabulary = encoder.vocabulary

    training_samples = []
    all_training_pieces: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(corpus):
        pieces = _encoded_piece_payload(encoder, sample)
        all_training_pieces.extend(pieces)
        training_samples.append(
            {
                "sample_index": sample_index,
                "text": sample,
                "pieces": pieces,
                "initial_token_count": len(sample.encode("utf-8")),
                "final_token_count": sum(
                    len(piece["tokens"]) for piece in pieces
                ),
            }
        )

    encoded_pieces = _encoded_piece_payload(encoder, text)
    encoding_report = encoder.analyze(text, include_tokens=True)
    encoded_tokens = list(encoding_report.tokens or ())
    decoded_text = encoder.decode(encoded_tokens)
    encoded_bytes = b"".join(
        vocabulary[token_id] for token_id in encoded_tokens
    )

    merge_rows = []
    for rank, learned_merge in enumerate(encoder.merges, start=1):
        token_bytes = vocabulary[learned_merge.token_id]
        merge_rows.append(
            {
                "rank": rank,
                "pair": list(learned_merge.pair),
                "token_id": learned_merge.token_id,
                "training_frequency": learned_merge.training_frequency,
                "bytes": list(token_bytes),
                "text": token_bytes.decode("utf-8", errors="replace"),
            }
        )

    protected_pieces_unchanged = all(
        piece["initial_tokens"] == piece["tokens"]
        for piece in [*all_training_pieces, *encoded_pieces]
        if not piece["merge_allowed"]
    )
    split_rebuilds_inputs = all(
        "".join(piece["text"] for piece in sample["pieces"])
        == sample["text"]
        for sample in training_samples
    ) and "".join(piece["text"] for piece in encoded_pieces) == text
    flattened_piece_tokens = [
        token_id
        for piece in encoded_pieces
        for token_id in piece["tokens"]
    ]

    return {
        "schema_version": 2,
        "source": {
            "module": "tokenizers/split_bpe/bpe.py",
            "class": "SplitAwareBytePairEncoder",
        },
        "configuration": {
            "vocab_size": vocab_size,
            "protected_characters": protected_characters,
        },
        "split_policy": encoder.split_policy,
        "training": training_report.as_dict(),
        "training_samples": training_samples,
        "merges": merge_rows,
        "encoding": {
            "text": text,
            "pieces": encoded_pieces,
            "decoded_text": decoded_text,
            **encoding_report.as_dict(),
        },
        "invariants": {
            "split_rebuilds_inputs": split_rebuilds_inputs,
            "protected_pieces_unchanged": protected_pieces_unchanged,
            "pieces_flatten_to_encoding": flattened_piece_tokens
            == encoded_tokens,
            "encoded_bytes_match_input": encoded_bytes
            == text.encode("utf-8"),
            "decode_round_trip": decoded_text == text,
        },
    }
