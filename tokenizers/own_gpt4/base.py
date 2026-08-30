"""byte-level BPE 的公共数据结构与基础操作。

这个模块刻意保持算法朴素：token 用 ``int`` 表示，merge rule 用
``(left_id, right_id) -> new_id`` 表示。实现优先服务于学习、测试和可视化，
因此会对公开 API 做严格校验，并用明确的异常指出输入问题。
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TypeAlias


BASE_VOCAB_SIZE = 256

Pair: TypeAlias = tuple[int, int]


def text_to_bytes(text: object) -> bytes:
    """校验 Python string，并返回严格的 UTF-8 表示。"""

    if not isinstance(text, str):
        raise TypeError("text must be a str")
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(
            "text must contain only valid Unicode scalar values"
        ) from error


def _validate_token_id(value: object, *, name: str) -> int:
    """校验一个 token id；显式排除 ``bool``（它是 ``int`` 的子类）。"""

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def validate_ids(ids: object, *, name: str = "ids") -> list[int]:
    """把严格的 int sequence 复制成 list，供 encode/decode 边界复用。"""

    if isinstance(ids, (str, bytes, bytearray)) or not isinstance(ids, Sequence):
        raise TypeError(f"{name} must be a sequence of int token ids")

    validated: list[int] = []
    for index, token_id in enumerate(ids):
        validated.append(
            _validate_token_id(token_id, name=f"{name}[{index}]")
        )
    return validated


def get_stats(
    ids: Sequence[int],
    counts: MutableMapping[Pair, int] | None = None,
) -> MutableMapping[Pair, int]:
    """统计 ``ids`` 中所有相邻 pair，可选择累加到已有 mapping。

    ``counts`` 参数让 RegexTokenizer 能分别统计多个 piece，而不会把上一个
    piece 的末尾和下一个 piece 的开头错误地组成 pair。
    """

    validated = validate_ids(ids)
    if counts is None:
        counts = {}
    elif not isinstance(counts, MutableMapping):
        raise TypeError("counts must be a mutable mapping")

    for left, right in zip(validated, validated[1:]):
        pair = (left, right)
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: Sequence[int], pair: Pair, idx: int) -> list[int]:
    """从左向右把不重叠的 ``pair`` 替换成 ``idx``。

    例如 ``merge([1, 1, 1], (1, 1), 256)`` 的结果是 ``[256, 1]``；
    中间的 1 已被第一次合并消耗，不能参与第二次重叠匹配。
    """

    validated = validate_ids(ids)
    if (
        not isinstance(pair, tuple)
        or len(pair) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) for item in pair)
    ):
        raise TypeError("pair must be a tuple of two int token ids")
    if pair[0] < 0 or pair[1] < 0:
        raise ValueError("pair token ids must be >= 0")
    new_id = _validate_token_id(idx, name="idx")

    merged: list[int] = []
    cursor = 0
    while cursor < len(validated):
        if (
            cursor + 1 < len(validated)
            and validated[cursor] == pair[0]
            and validated[cursor + 1] == pair[1]
        ):
            merged.append(new_id)
            cursor += 2
        else:
            merged.append(validated[cursor])
            cursor += 1
    return merged


class Tokenizer:
    """所有自研 tokenizer 共用的最小基类。

    ``merges`` 和 ``vocab`` 保持公开，便于练习代码、测试和前端直接观察。
    修改 ``merges`` 后应调用 ``_build_vocab()`` 重建派生词表。
    """

    pattern = ""

    def __init__(self) -> None:
        self.merges: dict[Pair, int] = {}
        self.vocab: dict[int, bytes] = {}
        self.special_tokens: dict[str, int] = {}
        self.inverse_special_tokens: dict[int, str] = {}
        self._build_vocab()

    @property
    def vocab_size(self) -> int:
        """返回普通 byte/BPE token 的数量，不把 special token 计入其中。"""

        return len(self.vocab)

    @property
    def mergeable_vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def total_vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def _build_vocab(self) -> None:
        """根据 merge DAG 重建 ``token_id -> bytes`` 词表。

        新 token 的两个父节点必须已经存在。按输出 id 排序后构建，既明确
        表达 BPE 的拓扑顺序，也能在损坏的规则上尽早报错。
        """

        vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        seen_output_ids: set[int] = set()

        for pair, token_id in sorted(self.merges.items(), key=lambda item: item[1]):
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(
                    not isinstance(item, int) or isinstance(item, bool)
                    for item in pair
                )
            ):
                raise TypeError("every merge key must be a pair of int token ids")
            _validate_token_id(token_id, name="merge output id")
            if token_id < BASE_VOCAB_SIZE:
                raise ValueError("merge output ids must be >= 256")
            if token_id in seen_output_ids:
                raise ValueError(f"duplicate merge output id: {token_id}")
            seen_output_ids.add(token_id)

            left, right = pair
            try:
                vocab[token_id] = vocab[left] + vocab[right]
            except KeyError as error:
                raise ValueError(
                    f"merge {pair!r} -> {token_id} references an unknown parent"
                ) from error

        self.vocab = vocab

    def register_special_tokens(self, special_tokens: Mapping[str, int]) -> None:
        """注册一组字面 special token，并建立反向 id mapping。

        该调用会替换旧注册表而非增量更新，因而重复调用的结果完全确定。
        special id 不能与普通词表冲突，也不能在两个字符串之间复用。
        """

        if not isinstance(special_tokens, Mapping):
            raise TypeError("special_tokens must be a mapping from str to int")

        validated: dict[str, int] = {}
        inverse: dict[int, str] = {}
        for literal, token_id_value in special_tokens.items():
            if not isinstance(literal, str):
                raise TypeError("every special token literal must be a str")
            if not literal:
                raise ValueError("special token literals must not be empty")
            try:
                literal.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError(
                    "special token literals must contain valid Unicode scalar values"
                ) from error

            token_id = _validate_token_id(
                token_id_value,
                name=f"special token id for {literal!r}",
            )
            if token_id in self.vocab:
                raise ValueError(
                    f"special token id {token_id} conflicts with the ordinary vocab"
                )
            if token_id in inverse:
                raise ValueError(
                    f"special token id {token_id} is assigned more than once"
                )
            validated[literal] = token_id
            inverse[token_id] = literal

        self.special_tokens = validated
        self.inverse_special_tokens = inverse

    def token_bytes(self, token_id: int) -> bytes:
        """返回一个普通或 special token 所代表的精确 UTF-8 bytes。"""

        validated_id = _validate_token_id(token_id, name="token_id")
        if validated_id in self.vocab:
            return self.vocab[validated_id]
        if validated_id in self.inverse_special_tokens:
            return self.inverse_special_tokens[validated_id].encode("utf-8")
        raise ValueError(f"token_id={validated_id} is not in this tokenizer vocab")

    def decode(self, ids: Sequence[int]) -> str:
        """先拼接全部 token bytes，再以 replacement 模式整体 UTF-8 解码。"""

        validated = validate_ids(ids)
        pieces: list[bytes] = []
        for index, token_id in enumerate(validated):
            try:
                pieces.append(self.token_bytes(token_id))
            except ValueError as error:
                raise ValueError(
                    f"ids[{index}]={token_id} is not in this tokenizer vocab"
                ) from error
        return b"".join(pieces).decode("utf-8", errors="replace")

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        raise NotImplementedError

    def encode(self, text: str, **kwargs: object) -> list[int]:
        raise NotImplementedError


__all__ = [
    "BASE_VOCAB_SIZE",
    "Pair",
    "Tokenizer",
    "get_stats",
    "merge",
    "text_to_bytes",
    "validate_ids",
]
