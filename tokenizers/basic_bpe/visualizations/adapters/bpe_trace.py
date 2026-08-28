"""调用真实 BPE 实现，并生成浏览器可以渲染的训练 trace。

本模块是核心算法与展示层之间的单向 adapter：它可以 import ``bpe``，
但 ``bpe.py`` 不需要知道本模块或浏览器界面的存在。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from ...bpe import (
        BytePairEncoder,
        count_adjacent_pairs,
        merge_pair,
        utf8_tokens,
    )
except ImportError:  # 支持在 basic_bpe 目录内运行本地测试。
    from bpe import (  # type: ignore[no-redef]
        BytePairEncoder,
        count_adjacent_pairs,
        merge_pair,
        utf8_tokens,
    )


MAX_VOCAB_SIZE = 1024
MAX_CORPUS_UTF8_BYTES = 20_000
MAX_TEXT_UTF8_BYTES = 20_000
MAX_TRACE_TOKEN_CELLS = 300_000
PAIR_ROWS_PER_STEP = 12


class VisualizationInputError(ValueError):
    """用户输入无法安全地生成可视化 trace。"""


def validate_bpe_payload(payload: object) -> tuple[list[str], int, str]:
    """验证训练配置与可选待编码 string，供 BPE/Encoder adapter 共用。"""

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

    text = payload.get("text", "")
    if not isinstance(text, str):
        raise VisualizationInputError("text 必须是 string。")

    corpus_byte_count = sum(len(sample.encode("utf-8")) for sample in corpus)
    if corpus_byte_count > MAX_CORPUS_UTF8_BYTES:
        raise VisualizationInputError(
            f"训练样本最多包含 {MAX_CORPUS_UTF8_BYTES} 个 UTF-8 bytes。"
        )
    if len(text.encode("utf-8")) > MAX_TEXT_UTF8_BYTES:
        raise VisualizationInputError(
            f"待编码文本最多包含 {MAX_TEXT_UTF8_BYTES} 个 UTF-8 bytes。"
        )

    return corpus, vocab_size, text


def _sorted_pair_counts(
    token_sequences: list[list[int]],
) -> list[dict[str, int | list[int]]]:
    pair_counts = count_adjacent_pairs(token_sequences)
    sorted_pairs = sorted(
        pair_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return [
        {"pair": list(pair), "frequency": frequency}
        for pair, frequency in sorted_pairs
    ]


def _merged_pair_starts(tokens: list[int], pair: tuple[int, int]) -> list[int]:
    """返回本轮从左到右实际参与非重叠合并的 pair 起点。"""

    starts: list[int] = []
    index = 0
    while index + 1 < len(tokens):
        if tokens[index] == pair[0] and tokens[index + 1] == pair[1]:
            starts.append(index)
            index += 2
        else:
            index += 1
    return starts


def _total_token_count(token_sequences: list[list[int]]) -> int:
    return sum(len(tokens) for tokens in token_sequences)


def _tokens_to_bytes(tokens: list[int], vocabulary: Mapping[int, bytes]) -> bytes:
    return b"".join(vocabulary[token] for token in tokens)


def build_bpe_trace(payload: object) -> dict[str, Any]:
    """用 ``bpe.py`` 训练、编码，并把真实结果转换成展示数据。"""

    corpus, vocab_size, text = validate_bpe_payload(payload)

    encoder = BytePairEncoder()
    training_report = encoder.train(corpus, vocab_size=vocab_size)
    vocabulary = encoder.vocabulary
    initial_sequences = [utf8_tokens(sample) for sample in corpus]
    token_sequences = [tokens.copy() for tokens in initial_sequences]
    steps: list[dict[str, Any]] = []
    trace_token_cells = _total_token_count(token_sequences)

    for round_number, learned_merge in enumerate(encoder.merges, start=1):
        all_pair_counts = _sorted_pair_counts(token_sequences)
        if not all_pair_counts:
            raise RuntimeError("BPE merge 存在，但 replay 时没有找到相邻 pair。")

        observed_best_pair = tuple(all_pair_counts[0]["pair"])
        observed_frequency = all_pair_counts[0]["frequency"]
        if (
            observed_best_pair != learned_merge.pair
            or observed_frequency != learned_merge.training_frequency
        ):
            raise RuntimeError("可视化 trace 与 BPE 实际训练结果不一致。")

        starts_by_sample = [
            _merged_pair_starts(tokens, learned_merge.pair)
            for tokens in token_sequences
        ]
        after = [
            merge_pair(tokens, learned_merge.pair, learned_merge.token_id)
            for tokens in token_sequences
        ]
        trace_token_cells += _total_token_count(token_sequences)
        trace_token_cells += _total_token_count(after)
        if trace_token_cells > MAX_TRACE_TOKEN_CELLS:
            raise VisualizationInputError(
                "训练 trace 过大，请减少输入文本或目标词表大小后再试。"
            )

        steps.append(
            {
                "round": round_number,
                "pair": list(learned_merge.pair),
                "token_id": learned_merge.token_id,
                "training_frequency": learned_merge.training_frequency,
                "applied_merge_count": sum(map(len, starts_by_sample)),
                "before": [tokens.copy() for tokens in token_sequences],
                "after": [tokens.copy() for tokens in after],
                "merged_pair_starts": starts_by_sample,
                "pair_counts": all_pair_counts[:PAIR_ROWS_PER_STEP],
                "pair_type_count": len(all_pair_counts),
                "pair_counts_truncated": len(all_pair_counts) > PAIR_ROWS_PER_STEP,
            }
        )
        token_sequences = after

    encoded_corpus = [encoder.encode(sample) for sample in corpus]
    trace_matches_encoder = token_sequences == encoded_corpus
    if not trace_matches_encoder:
        raise RuntimeError("最终 trace 与 BytePairEncoder.encode() 的结果不一致。")

    encoding_report = encoder.analyze(text, include_tokens=True)
    encoded_tokens = list(encoding_report.tokens or ())
    encoded_bytes_match_input = (
        _tokens_to_bytes(encoded_tokens, vocabulary) == text.encode("utf-8")
    )

    return {
        "schema_version": 1,
        "source": {
            "module": "bpe.py",
            "class": "BytePairEncoder",
        },
        "training": training_report.as_dict(),
        "trace": {
            "initial_sequences": initial_sequences,
            "steps": steps,
            "final_sequences": token_sequences,
        },
        "vocabulary": {
            str(token_id): list(token_bytes)
            for token_id, token_bytes in vocabulary.items()
        },
        "encoding": {
            "text": text,
            **encoding_report.as_dict(),
        },
        "invariants": {
            "trace_matches_encoder": trace_matches_encoder,
            "encoded_bytes_match_input": encoded_bytes_match_input,
        },
    }
