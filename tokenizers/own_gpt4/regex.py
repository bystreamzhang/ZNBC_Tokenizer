"""使用 GPT-4 预切分边界的可训练 byte-level BPE tokenizer。"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Set
from typing import Literal, TypeAlias

import regex as regex_module

from .base import text_to_bytes
from .basic import BasicTokenizer, _validate_training_options


GPT4_SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|"
    r"\s+(?!\S)|\s+"
)

AllowedSpecial: TypeAlias = Set[str] | Literal["all"]
DisallowedSpecial: TypeAlias = Collection[str] | Literal["all"]


def _validate_string_collection(value: object, *, name: str) -> set[str]:
    """校验 special-token 名称 collection，并返回独立 set。"""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        Collection,
    ):
        raise TypeError(f"{name} must be a collection of str or 'all'")
    names = set(value)
    if any(not isinstance(item, str) for item in names):
        raise TypeError(f"every item in {name} must be a str")
    return names


def _compile_literal_pattern(literals: Collection[str]) -> regex_module.Pattern[str]:
    """构造 longest-first 的字面 regex，正确处理有公共前缀的 token。"""

    alternatives = sorted(literals, key=lambda literal: (-len(literal), literal))
    return regex_module.compile(
        "|".join(regex_module.escape(literal) for literal in alternatives)
    )


class RegexTokenizer(BasicTokenizer):
    """先无损分块，再在每个块内独立训练/执行 BPE。

    默认 pattern 与 ``cl100k_base`` 使用的 GPT-4 规则一致。special token 在
    regex 预切分之前识别，因而既不会被普通 BPE 拆开，也不会和左右文本合并。
    """

    def __init__(self, pattern: str = GPT4_SPLIT_PATTERN) -> None:
        if not isinstance(pattern, str):
            raise TypeError("pattern must be a str")
        if not pattern:
            raise ValueError("pattern must not be empty")
        try:
            compiled_pattern = regex_module.compile(pattern)
        except regex_module.error as error:
            raise ValueError(f"invalid regex pattern: {error}") from error

        super().__init__()
        self.pattern = pattern
        self.compiled_pattern: regex_module.Pattern[str] = compiled_pattern

    def split(self, text: str) -> list[str]:
        """返回训练/编码共用的普通 regex pieces，且拼接后严格等于原文。

        GPT-4 pattern 本身能覆盖任意输入。对调用者传入的不完整自定义 pattern，
        这里也会把没有命中的间隙保留下来，避免静默丢字符。
        """

        text_to_bytes(text)
        pieces: list[str] = []
        cursor = 0
        for match in self.compiled_pattern.finditer(text):
            start, end = match.span()
            if start > cursor:
                pieces.append(text[cursor:start])
            if end > start:
                pieces.append(text[start:end])
            cursor = max(cursor, end)
        if cursor < len(text):
            pieces.append(text[cursor:])

        if "".join(pieces) != text:
            raise RuntimeError("regex split did not preserve the input text")
        return pieces

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """在 GPT-4 regex pieces 内学习 BPE，禁止跨 piece merge。"""

        requested_size, verbose_flag = _validate_training_options(
            vocab_size,
            verbose,
        )
        pieces = self.split(text)
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

        sequences = [list(piece.encode("utf-8")) for piece in pieces]
        self._train_sequences(sequences, requested_size, verbose_flag)

    def encode_ordinary(self, text: str) -> list[int]:
        """把所有 special-looking 字面量当普通文本执行 regex+BPE。"""

        pieces = self.split(text)
        return [
            token_id
            for piece in pieces
            for token_id in self._encode_chunk(piece.encode("utf-8"))
        ]

    def _normalize_special_policy(
        self,
        allowed_special: AllowedSpecial,
        disallowed_special: DisallowedSpecial,
    ) -> tuple[set[str], set[str]]:
        """把 tiktoken 风格参数归一化为 allowed/disallowed 两个集合。"""

        if allowed_special == "all":
            allowed = set(self.special_tokens)
        elif isinstance(allowed_special, Set) and not isinstance(
            allowed_special,
            (str, bytes, bytearray),
        ):
            allowed = set(allowed_special)
            if any(not isinstance(item, str) for item in allowed):
                raise TypeError("every item in allowed_special must be a str")
        else:
            raise TypeError("allowed_special must be a set of str or 'all'")

        if disallowed_special == "all":
            # allowed 显式优先；未知 allowed 名称不会凭空成为 special token。
            disallowed = set(self.special_tokens) - allowed
        else:
            disallowed = _validate_string_collection(
                disallowed_special,
                name="disallowed_special",
            )
        return allowed, disallowed

    def split_with_special_tokens(
        self,
        text: str,
        allowed_special: AllowedSpecial = frozenset(),
        disallowed_special: DisallowedSpecial = "all",
    ) -> list[tuple[str, Literal["ordinary", "special"]]]:
        """按实际 encode 路径返回 ordinary regex pieces 与 special pieces。

        special token 先于普通 regex 被识别；只有 special 左右的 ordinary chunk
        会再调用 ``split()``。前端和调试代码使用这个方法，可以避免把一个已经
        识别的 special literal 错画成若干普通 regex pieces。
        """

        text_to_bytes(text)
        allowed, disallowed = self._normalize_special_policy(
            allowed_special,
            disallowed_special,
        )

        if disallowed:
            disallowed_match = _compile_literal_pattern(disallowed).search(text)
            if disallowed_match is not None:
                literal = disallowed_match.group(0)
                raise ValueError(
                    "encountered text corresponding to disallowed special token "
                    f"{literal!r}; add it to allowed_special or set "
                    "disallowed_special=() to encode it as ordinary text"
                )

        recognized = set(self.special_tokens) & allowed
        if not recognized:
            return [(piece, "ordinary") for piece in self.split(text)]

        result: list[tuple[str, Literal["ordinary", "special"]]] = []
        special_pattern = _compile_literal_pattern(recognized)
        cursor = 0
        for match in special_pattern.finditer(text):
            if match.start() > cursor:
                result.extend(
                    (piece, "ordinary")
                    for piece in self.split(text[cursor : match.start()])
                )
            result.append((match.group(0), "special"))
            cursor = match.end()
        if cursor < len(text):
            result.extend(
                (piece, "ordinary") for piece in self.split(text[cursor:])
            )

        if "".join(piece for piece, _ in result) != text:
            raise RuntimeError("special-token split did not preserve the input text")
        return result

    def encode(
        self,
        text: str,
        allowed_special: AllowedSpecial = frozenset(),
        disallowed_special: DisallowedSpecial = "all",
    ) -> list[int]:
        """按 tiktoken 风格策略处理 special token 后编码普通片段。

        默认禁止已注册 special token；``allowed_special="all"`` 会识别全部；
        ``disallowed_special=()`` 则把所有字面量作为普通文本。显式 allowed 集合
        只识别其中已经注册的名称。
        """

        pieces = self.split_with_special_tokens(
            text,
            allowed_special,
            disallowed_special,
        )
        encoded: list[int] = []
        for piece, kind in pieces:
            if kind == "special":
                encoded.append(self.special_tokens[piece])
            else:
                # ``piece`` 已经是单个 regex piece，不能重新和相邻 piece 合并。
                encoded.extend(self._encode_chunk(piece.encode("utf-8")))
        return encoded

    def register_special_tokens(self, special_tokens: Mapping[str, int]) -> None:
        """公开该方法并沿用基类的冲突检查和替换语义。"""

        super().register_special_tokens(special_tokens)


__all__ = [
    "AllowedSpecial",
    "DisallowedSpecial",
    "GPT4_SPLIT_PATTERN",
    "RegexTokenizer",
]
