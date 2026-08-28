"""一个带可解释预切分边界的 byte-level BPE 实现。

训练与 encode 都先使用同一个 ``RegexPretokenizer``。BPE 只统计并合并单个
普通 piece 内的 pair，不能跨 piece；protected piece 则完全跳过 BPE。这样
配置为 protected 的字符既不会与左右内容合并，其自身 UTF-8 bytes 也不会
互相合并。
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping, Sequence, TypeAlias

try:
    from .pretokenizer import (
        DEFAULT_PROTECTED_CHARACTERS,
        RegexPretokenizer,
        SplitPiece,
    )
except ImportError:  # 允许在当前目录直接执行 ``python3 bpe.py``。
    from pretokenizer import (  # type: ignore[no-redef]
        DEFAULT_PROTECTED_CHARACTERS,
        RegexPretokenizer,
        SplitPiece,
    )


BASE_VOCAB_SIZE = 256

Pair: TypeAlias = tuple[int, int]


def utf8_tokens(text: str) -> list[int]:
    """把 Unicode string 编码成 UTF-8，并让每个 byte 成为一个 int token。

    返回值中的每个 int 都位于 0~255。一个 Unicode 字符可能对应多个
    byte token，例如中文字符通常会得到 3 个初始 token。
    """

    if not isinstance(text, str):
        raise TypeError("text must be a str")
    return list(text.encode("utf-8"))


def count_adjacent_pairs(
    token_sequences: Iterable[Sequence[int]],
) -> Counter[Pair]:
    """统计每条 token sequence 中所有相邻 pair 的出现次数。

    每条 sequence 独立统计，不会把上一条的末尾 token 和下一条的开头
    token 组成一个实际不存在的跨样本 pair。
    """

    counts: Counter[Pair] = Counter()
    for tokens in token_sequences:
        counts.update(zip(tokens, tokens[1:]))
    return counts


def merge_pair(
    tokens: Sequence[int], pair: Pair, new_token_id: int
) -> list[int]:
    """从左向右，把不重叠的 ``pair`` 替换成 ``new_token_id``。

    例如把 ``(1, 1)`` 合并为 256 时，输入 ``[1, 1, 1]`` 会得到
    ``[256, 1]``；中间的 1 已被第一次合并使用，不能再参与重叠合并。
    """

    merged: list[int] = []
    index = 0
    while index < len(tokens):
        if (
            index + 1 < len(tokens)
            and tokens[index] == pair[0]
            and tokens[index + 1] == pair[1]
        ):
            merged.append(new_token_id)
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return merged


@dataclass(frozen=True, slots=True)
class Merge:
    """训练过程中学到的一条规则：pair、对应的新 id 和当轮频率。"""

    pair: Pair
    token_id: int
    training_frequency: int


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """一次训练的摘要，包括目标/实际词表大小和语料压缩结果。"""

    requested_vocab_size: int
    actual_vocab_size: int
    merges_learned: int
    original_token_count: int
    final_token_count: int
    split_piece_count: int
    protected_piece_count: int
    mergeable_piece_count: int

    @property
    def compression_ratio(self) -> float:
        """返回原始 byte token 数 / 最终 BPE token 数。"""

        if self.final_token_count == 0:
            return 1.0
        return self.original_token_count / self.final_token_count

    @property
    def reduction_ratio(self) -> float:
        """返回训练语料经过 merge 后减少的 token 比例。"""

        if self.original_token_count == 0:
            return 0.0
        return 1.0 - self.final_token_count / self.original_token_count

    def as_dict(self) -> dict[str, int | float]:
        """转换成可以直接输出为 JSON 的 dictionary。"""

        return {
            "requested_vocab_size": self.requested_vocab_size,
            "actual_vocab_size": self.actual_vocab_size,
            "merges_learned": self.merges_learned,
            "original_token_count": self.original_token_count,
            "final_token_count": self.final_token_count,
            "split_piece_count": self.split_piece_count,
            "protected_piece_count": self.protected_piece_count,
            "mergeable_piece_count": self.mergeable_piece_count,
            "compression_ratio": self.compression_ratio,
            "reduction_ratio": self.reduction_ratio,
        }


@dataclass(frozen=True, slots=True)
class EncodingReport:
    """一个输入 string 编码后的长度、压缩比例和可选 token 内容。"""

    utf8_byte_count: int
    token_count: int
    tokens: tuple[int, ...] | None = None

    @property
    def saved_token_count(self) -> int:
        return self.utf8_byte_count - self.token_count

    @property
    def compression_ratio(self) -> float:
        """返回 UTF-8 byte 数 / BPE token 数，即每个 token 平均承载几 bytes。"""

        if self.token_count == 0:
            return 1.0
        return self.utf8_byte_count / self.token_count

    @property
    def reduction_ratio(self) -> float:
        """返回相对原始 UTF-8 byte token 数减少的比例。"""

        if self.utf8_byte_count == 0:
            return 0.0
        return 1.0 - self.token_count / self.utf8_byte_count

    def as_dict(self) -> dict[str, int | float | list[int]]:
        """转换成 dictionary；未要求展示 token 时不加入 ``tokens`` 字段。"""

        result: dict[str, int | float | list[int]] = {
            "utf8_byte_count": self.utf8_byte_count,
            "token_count": self.token_count,
            "saved_token_count": self.saved_token_count,
            "compression_ratio": self.compression_ratio,
            "reduction_ratio": self.reduction_ratio,
        }
        if self.tokens is not None:
            result["tokens"] = list(self.tokens)
        return result


@dataclass(frozen=True, slots=True)
class EncodedPiece:
    """一个 split piece 在应用固定 merge rules 前后的 token。"""

    piece: SplitPiece
    initial_tokens: tuple[int, ...]
    tokens: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.piece.as_dict(),
            "initial_tokens": list(self.initial_tokens),
            "tokens": list(self.tokens),
        }


class SplitAwareBytePairEncoder:
    """在不可跨越的 split pieces 内训练并使用 byte-level BPE。"""

    def __init__(
        self,
        *,
        protected_characters: str = DEFAULT_PROTECTED_CHARACTERS,
    ) -> None:
        self._pretokenizer = RegexPretokenizer(protected_characters)
        self._merges: list[Merge] = []
        self._vocabulary: dict[int, bytes] = self._base_vocabulary()

    @staticmethod
    def _base_vocabulary() -> dict[int, bytes]:
        return {token_id: bytes([token_id]) for token_id in range(256)}

    @property
    def merges(self) -> tuple[Merge, ...]:
        """按新 token id 从小到大返回所有已学习的 merge 规则。"""

        return tuple(self._merges)

    @property
    def vocabulary(self) -> Mapping[int, bytes]:
        """返回 token id 到其所代表原始 bytes 的映射副本。"""

        return self._vocabulary.copy()

    @property
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    @property
    def protected_characters(self) -> str:
        return self._pretokenizer.protected_characters

    @property
    def split_policy(self) -> Mapping[str, Any]:
        return self._pretokenizer.policy_as_dict()

    def split(self, text: str) -> tuple[SplitPiece, ...]:
        """返回训练和 encode 共同使用的无损 split 结果。"""

        return self._pretokenizer.split(text)

    def train(
        self, corpus: str | Iterable[str], *, vocab_size: int
    ) -> TrainingReport:
        """从 ``corpus`` 学习 merge，词表最多增长到 ``vocab_size``。

        ``vocab_size`` 包含最初的 256 个 byte token，因此目标 260 表示最多
        学习 4 条 merge。每个 corpus item 会先被切成 pieces；pair 既不会跨
        item，也不会跨 piece。protected pieces 不进入 pair 统计或 merge。
        再次调用 ``train`` 会清空旧规则，开始一次完全独立的新训练。
        """

        if (
            not isinstance(vocab_size, int)
            or isinstance(vocab_size, bool)
            or vocab_size < BASE_VOCAB_SIZE
        ):
            raise ValueError("vocab_size must be an integer >= 256")

        if isinstance(corpus, str):
            texts = [corpus]
        else:
            texts = list(corpus)

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"corpus item {index} must be a str")

        pieces = [piece for text in texts for piece in self.split(text)]
        token_sequences = [
            utf8_tokens(piece.text) for piece in pieces if piece.merge_allowed
        ]
        protected_token_count = sum(
            len(piece.utf8_bytes) for piece in pieces if not piece.merge_allowed
        )
        original_token_count = sum(len(text.encode("utf-8")) for text in texts)

        # 重置所有可学习状态，确保本次训练不受上一次 train() 的结果影响。
        self._merges = []
        self._vocabulary = self._base_vocabulary()

        for new_token_id in range(BASE_VOCAB_SIZE, vocab_size):
            pair_counts = count_adjacent_pairs(token_sequences)
            if not pair_counts:
                break

            # 首先选择频率最高的 pair。如果多个 pair 频率相同，则选择 token id
            # 二元组字典序最小的一个，避免结果依赖 dictionary 的插入顺序。
            best_pair = min(
                pair_counts,
                key=lambda pair: (-pair_counts[pair], pair),
            )
            frequency = pair_counts[best_pair]

            token_sequences = [
                merge_pair(tokens, best_pair, new_token_id)
                for tokens in token_sequences
            ]
            self._merges.append(
                Merge(
                    pair=best_pair,
                    token_id=new_token_id,
                    training_frequency=frequency,
                )
            )
            left, right = best_pair
            self._vocabulary[new_token_id] = (
                self._vocabulary[left] + self._vocabulary[right]
            )

        final_token_count = protected_token_count + sum(map(len, token_sequences))
        return TrainingReport(
            requested_vocab_size=vocab_size,
            actual_vocab_size=self.vocab_size,
            merges_learned=len(self._merges),
            original_token_count=original_token_count,
            final_token_count=final_token_count,
            split_piece_count=len(pieces),
            protected_piece_count=sum(
                not piece.merge_allowed for piece in pieces
            ),
            mergeable_piece_count=sum(piece.merge_allowed for piece in pieces),
        )

    def encode_pieces(self, text: str) -> tuple[EncodedPiece, ...]:
        """逐 piece 重放固定规则，并保留边界与中间结果供检查。"""

        encoded_pieces: list[EncodedPiece] = []
        for piece in self.split(text):
            initial_tokens = utf8_tokens(piece.text)
            tokens = initial_tokens.copy()
            if piece.merge_allowed:
                for learned_merge in self._merges:
                    tokens = merge_pair(
                        tokens,
                        learned_merge.pair,
                        learned_merge.token_id,
                    )
            encoded_pieces.append(
                EncodedPiece(
                    piece=piece,
                    initial_tokens=tuple(initial_tokens),
                    tokens=tuple(tokens),
                )
            )
        return tuple(encoded_pieces)

    def encode(self, text: str) -> list[int]:
        """使用固定 merge rules 编码每个 piece，再按原顺序展平。

        编码与训练是两个阶段：这里不会统计 ``text`` 中的 pair，也不会重新
        选择 pair。普通 piece 严格按训练顺序重放规则；protected piece 直接
        返回初始 UTF-8 byte ids，即使某条规则因其他 Unicode 字符而恰好能
        命中相同 byte pair，也不能在 protected piece 上应用。
        """

        return [
            token
            for encoded_piece in self.encode_pieces(text)
            for token in encoded_piece.tokens
        ]

    def decode(self, tokens: Sequence[int]) -> str:
        """使用当前词表把一组 token id 还原为 Python string。

        每个 BPE token 在词表中对应一段原始 bytes。这里必须先把所有 token
        的 bytes 按顺序拼接，再整体执行 UTF-8 解码；单个 token 可能只包含
        一个多字节 Unicode 字符的局部 bytes，不能逐 token 转成字符。

        ``tokens`` 中的 id 必须存在于当前词表。若 token 顺序拼出的 bytes
        不是合法 UTF-8，则按 Python ``errors="replace"`` 规则插入 Unicode
        replacement character ``�``（U+FFFD），并继续返回 string。
        """

        if isinstance(tokens, (str, bytes, bytearray)) or not isinstance(
            tokens, Sequence
        ):
            raise TypeError("tokens 必须是一个 int sequence")

        token_bytes: list[bytes] = []
        for index, token_id in enumerate(tokens):
            # bool 是 int 的子类，但 True/False 不是合法的 tokenizer id。
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError(f"tokens[{index}] 必须是 int")
            try:
                token_bytes.append(self._vocabulary[token_id])
            except KeyError as error:
                raise ValueError(
                    f"tokens[{index}]={token_id} 不在当前 BPE 词表中"
                ) from error

        # UTF-8 字符可能跨越多个 token；因此只在完整 byte stream 上解码一次。
        # 例如单独的 128 是 continuation byte，会被替换成 U+FFFD，而不是报错。
        return b"".join(token_bytes).decode("utf-8", errors="replace")

    def analyze(
        self, text: str, *, include_tokens: bool = False
    ) -> EncodingReport:
        """统计 ``text`` 编码前后的长度和压缩比例。

        ``include_tokens=False`` 时只返回统计信息；设为 True 时还会在报告中
        带上实际编码结果，适合人工检查。
        """

        byte_count = len(text.encode("utf-8"))
        tokens = self.encode(text)
        return EncodingReport(
            utf8_byte_count=byte_count,
            token_count=len(tokens),
            tokens=tuple(tokens) if include_tokens else None,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 split-aware byte-level BPE，并观察 string 的编码结果。"
    )
    parser.add_argument(
        "--train-text",
        action="append",
        required=True,
        help="训练样本；可重复传入该参数来提供多个独立样本",
    )
    parser.add_argument("--text", required=True, help="需要编码和分析的文本")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=256,
        help="目标词表大小，其中包含最初的 256 个 byte token",
    )
    parser.add_argument(
        "--protected-chars",
        default=DEFAULT_PROTECTED_CHARACTERS,
        help="这些 Unicode characters 每次出现都单独成段并完全跳过 BPE",
    )
    parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="在输出中包含编码后的 token sequence",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    encoder = SplitAwareBytePairEncoder(
        protected_characters=args.protected_chars,
    )
    training = encoder.train(args.train_text, vocab_size=args.vocab_size)
    encoding = encoder.analyze(args.text, include_tokens=args.show_tokens)
    print(
        json.dumps(
            {
                "split_policy": encoder.split_policy,
                "pieces": [
                    piece.as_dict() for piece in encoder.encode_pieces(args.text)
                ],
                "training": training.as_dict(),
                "encoding": encoding.as_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
