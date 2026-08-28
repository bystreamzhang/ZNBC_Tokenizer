"""使用已经构建好的 BPE 规则生成 Encoder 演示数据。

浏览器只负责提交训练配置和待编码 string。本 adapter 会先重建同一个
确定性的 Tokenizer，再严格按照 ``encoder.merges`` 的既定顺序重放规则。
它不会根据待编码文本重新统计 pair，也不会在 JavaScript 中复制编码算法。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from ...bpe import BytePairEncoder, merge_pair, utf8_tokens
except ImportError:  # 支持在 basic_bpe 目录内运行本地测试。
    from bpe import BytePairEncoder, merge_pair, utf8_tokens  # type: ignore[no-redef]
from .bpe_trace import (
    MAX_TRACE_TOKEN_CELLS,
    VisualizationInputError,
    validate_bpe_payload,
)


def _merged_pair_starts(
    tokens: list[int], pair: tuple[int, int]
) -> list[int]:
    """返回本条规则从左到右实际合并的非重叠 pair 起点。"""

    starts: list[int] = []
    index = 0
    while index + 1 < len(tokens):
        if tokens[index] == pair[0] and tokens[index + 1] == pair[1]:
            starts.append(index)
            index += 2
        else:
            index += 1
    return starts


def build_bpe_encode(payload: object) -> dict[str, Any]:
    """用固定 Tokenizer 调用真实 ``encode()``，并记录规则重放过程。"""

    if not isinstance(payload, Mapping) or "text" not in payload:
        raise VisualizationInputError("Encoder 请求必须包含 text string。")
    corpus, vocab_size, text = validate_bpe_payload(payload)

    # HTTP 请求之间不共享 Python 对象，所以用训练配置确定性地重建同一套
    # 有序规则。待编码的 text 不参与 train()，只在下面的 encode 阶段使用。
    encoder = BytePairEncoder()
    training_report = encoder.train(corpus, vocab_size=vocab_size)
    vocabulary = encoder.vocabulary

    initial_tokens = utf8_tokens(text)
    replayed_tokens = initial_tokens.copy()
    steps: list[dict[str, Any]] = []
    trace_token_cells = len(initial_tokens)
    total_merge_operations = 0

    for rule_number, learned_merge in enumerate(encoder.merges, start=1):
        before = replayed_tokens.copy()
        merged_pair_starts = _merged_pair_starts(
            before,
            learned_merge.pair,
        )
        replayed_tokens = merge_pair(
            before,
            learned_merge.pair,
            learned_merge.token_id,
        )
        applied_merge_count = len(merged_pair_starts)
        total_merge_operations += applied_merge_count

        trace_token_cells += len(before) + len(replayed_tokens)
        if trace_token_cells > MAX_TRACE_TOKEN_CELLS:
            raise VisualizationInputError(
                "Encoder trace 过大，请缩短待编码文本或减少目标词表大小后再试。"
            )

        steps.append(
            {
                "rule_number": rule_number,
                "pair": list(learned_merge.pair),
                "token_id": learned_merge.token_id,
                "training_frequency": learned_merge.training_frequency,
                "before": before,
                "after": replayed_tokens.copy(),
                "merged_pair_starts": merged_pair_starts,
                "applied_merge_count": applied_merge_count,
                "applied": applied_merge_count > 0,
                # pair 中出现 learned token，说明本规则依赖前序规则的产物。
                "depends_on_previous_token": any(
                    token_id >= 256 for token_id in learned_merge.pair
                ),
            }
        )

    # 最终结果必须直接来自核心 encode()；上面的 replay 只为生成可视化步骤。
    encoded_tokens = encoder.encode(text)
    encoding_report = encoder.analyze(text, include_tokens=True)
    encoded_bytes = b"".join(vocabulary[token] for token in encoded_tokens)
    ordered_merge_ids = [
        learned_merge.token_id for learned_merge in encoder.merges
    ]
    parents_precede_children = all(
        left < learned_merge.token_id and right < learned_merge.token_id
        for learned_merge in encoder.merges
        for left, right in [learned_merge.pair]
    )

    return {
        "schema_version": 1,
        "source": {
            "module": "bpe.py",
            "class": "BytePairEncoder",
            "method": "encode",
        },
        "training": training_report.as_dict(),
        "vocabulary": {
            str(token_id): list(token_bytes)
            for token_id, token_bytes in vocabulary.items()
        },
        "encoding": {
            "text": text,
            "initial_tokens": initial_tokens,
            **encoding_report.as_dict(),
        },
        "trace": {
            "steps": steps,
            "rules_checked": len(steps),
            "rules_applied": sum(step["applied"] for step in steps),
            "merge_operations": total_merge_operations,
            "final_tokens": replayed_tokens,
        },
        "invariants": {
            "rules_follow_training_order": ordered_merge_ids
            == list(range(256, 256 + len(ordered_merge_ids))),
            "parents_precede_children": parents_precede_children,
            "trace_matches_encoder": replayed_tokens == encoded_tokens,
            "encoded_bytes_match_input": encoded_bytes == text.encode("utf-8"),
        },
    }
