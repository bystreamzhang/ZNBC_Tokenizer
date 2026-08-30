"""不做预切分的、可训练的 byte-level BPE tokenizer。"""

from __future__ import annotations

from collections.abc import Sequence

from .base import (
    BASE_VOCAB_SIZE,
    Pair,
    Tokenizer,
    get_stats,
    merge,
    text_to_bytes,
)


def _validate_training_options(vocab_size: object, verbose: object) -> tuple[int, bool]:
    """集中校验训练参数，使 Basic/Regex 的边界行为完全一致。"""

    if (
        not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size < BASE_VOCAB_SIZE
    ):
        raise ValueError("vocab_size must be an integer >= 256")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")
    return vocab_size, verbose


class BasicTokenizer(Tokenizer):
    """直接在整个 UTF-8 byte stream 上学习 BPE merge rules。"""

    def _train_sequences(
        self,
        sequences: Sequence[Sequence[int]],
        vocab_size: int,
        verbose: bool,
    ) -> None:
        """在互相独立的 token sequences 上学习规则。

        该辅助函数由 RegexTokenizer 复用。每条 sequence 都单独统计，因此
        不会产生跨 regex piece 的 merge。
        """

        # 复制训练数据；每轮 merge 都生成新 list，不修改调用者持有的对象。
        working = [list(sequence) for sequence in sequences]
        learned_merges: dict[Pair, int] = {}

        for new_token_id in range(BASE_VOCAB_SIZE, vocab_size):
            pair_counts: dict[Pair, int] = {}
            for ids in working:
                get_stats(ids, pair_counts)
            if not pair_counts:
                # 空语料或所有 piece 长度都小于 2 时，词表允许提前停止增长。
                break

            # 同频时选择 token-id pair 字典序最小者，保证跨进程可复现。
            best_pair = min(
                pair_counts,
                key=lambda pair: (-pair_counts[pair], pair),
            )
            frequency = pair_counts[best_pair]
            working = [merge(ids, best_pair, new_token_id) for ids in working]
            learned_merges[best_pair] = new_token_id

            if verbose:
                print(
                    f"merge {new_token_id - 255}/{vocab_size - 256}: "
                    f"{best_pair} -> {new_token_id}，训练频率 {frequency}"
                )

        # 一次性替换状态，确保重复 train() 不会残留上一次的规则。
        self.merges = learned_merges
        self._build_vocab()

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """从一个 string 学习至多 ``vocab_size - 256`` 条规则。

        ``vocab_size`` 包含最初的 256 个 byte token。再次调用会完整清空旧
        merge 状态，从同一输入和参数训练总能得到相同结果。
        """

        requested_size, verbose_flag = _validate_training_options(
            vocab_size,
            verbose,
        )
        raw_bytes = text_to_bytes(text)

        # 已注册 special id 不得进入本次可能生成的普通 id 区间。
        conflicting = sorted(
            token_id
            for token_id in self.inverse_special_tokens
            if token_id < requested_size
        )
        if conflicting:
            raise ValueError(
                "vocab_size would conflict with registered special token ids: "
                f"{conflicting}"
            )

        self._train_sequences([list(raw_bytes)], requested_size, verbose_flag)

    def _encode_token_ids(self, ids: Sequence[int]) -> list[int]:
        """按最早可用规则反复合并一条已经 byte 化的 sequence。"""

        encoded = list(ids)
        while len(encoded) >= 2:
            pair_counts = get_stats(encoded)
            best_pair = min(
                pair_counts,
                key=lambda pair: self.merges.get(pair, float("inf")),
            )
            if best_pair not in self.merges:
                break
            encoded = merge(encoded, best_pair, self.merges[best_pair])
        return encoded

    def _encode_chunk(self, chunk_bytes: bytes) -> list[int]:
        """编码一个不允许跨边界合并的 bytes chunk。"""

        return self._encode_token_ids(list(chunk_bytes))

    def encode(self, text: str) -> list[int]:
        """用当前固定规则编码 string；该操作绝不会现场学习新规则。"""

        return self._encode_chunk(text_to_bytes(text))


__all__ = ["BasicTokenizer"]
