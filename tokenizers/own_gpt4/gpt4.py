"""从 cl100k_base ranks 恢复并自行执行 GPT-4 BPE。

``tiktoken`` 在这里仅作为固定词表数据源：初始化时读取
``Encoding._mergeable_ranks``，实际 encode/decode 全部由本目录代码完成，
不会调用 tiktoken 的 encode、encode_ordinary 或 decode。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from threading import Lock
from typing import Any

from .base import Pair, validate_ids
from .regex import GPT4_SPLIT_PATTERN, RegexTokenizer


ENCODING_NAME = "cl100k_base"
MODEL_NAME = "gpt-4"
DISPLAY_NAME = "own-gpt-4"

CL100K_SPECIAL_TOKENS: dict[str, int] = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}

_CL100K_DATA: dict[str, Any] | None = None
_CL100K_DATA_LOCK = Lock()


def _validated_ranks(mergeable_ranks: Mapping[bytes, int]) -> dict[bytes, int]:
    """复制并检查 bytes->rank mapping，避免恢复时接受含糊数据。"""

    if not isinstance(mergeable_ranks, Mapping):
        raise TypeError("mergeable_ranks must be a mapping from bytes to int")

    ranks: dict[bytes, int] = {}
    used_ranks: dict[int, bytes] = {}
    for token_bytes, rank in mergeable_ranks.items():
        if not isinstance(token_bytes, bytes):
            raise TypeError("every mergeable_ranks key must be bytes")
        if not token_bytes:
            raise ValueError("mergeable token bytes must not be empty")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError("every mergeable rank must be an int")
        if rank < 0:
            raise ValueError("mergeable ranks must be >= 0")
        if rank in used_ranks:
            raise ValueError(
                f"rank {rank} is assigned to both {used_ranks[rank]!r} "
                f"and {token_bytes!r}"
            )
        ranks[token_bytes] = rank
        used_ranks[rank] = token_bytes
    return ranks


def _bpe_parts(
    token_bytes: bytes,
    mergeable_ranks: Mapping[bytes, int],
    *,
    max_rank: int,
) -> list[bytes]:
    """只使用 ``rank < max_rank`` 的规则，把 bytes 归约成 BPE parts。"""

    parts = [bytes([byte]) for byte in token_bytes]
    for part in parts:
        if part not in mergeable_ranks:
            raise ValueError(
                f"missing one-byte token {part!r} required by {token_bytes!r}"
            )

    while len(parts) >= 2:
        candidate: tuple[int, int] | None = None
        for index in range(len(parts) - 1):
            combined = parts[index] + parts[index + 1]
            rank = mergeable_ranks.get(combined)
            if rank is None or rank >= max_rank:
                continue
            ranked_index = (rank, index)
            if candidate is None or ranked_index < candidate:
                candidate = ranked_index

        if candidate is None:
            break
        _, index = candidate
        parts[index : index + 2] = [parts[index] + parts[index + 1]]

    return parts


def recover_merges(
    mergeable_ranks: Mapping[bytes, int],
) -> dict[Pair, int]:
    """从 cl100k 风格的最终 token ranks 恢复直接 parent merge rules。

    对 rank 为 ``r`` 的复合 token，只允许使用更早（rank 小于 ``r``）的
    token 归约它。归约结束后应恰好剩下两个 parent；它们的 id 组成该 token
    的原始 merge pair。该方法同样可用于结构完整的小型测试词表。
    """

    ranks = _validated_ranks(mergeable_ranks)
    merges: dict[Pair, int] = {}

    for token_bytes, rank in sorted(ranks.items(), key=lambda item: item[1]):
        if len(token_bytes) == 1:
            continue
        parts = _bpe_parts(token_bytes, ranks, max_rank=rank)
        if len(parts) != 2:
            raise ValueError(
                f"cannot recover two parents for rank {rank} ({token_bytes!r}); "
                f"got {len(parts)} parts"
            )
        left_id = ranks[parts[0]]
        right_id = ranks[parts[1]]
        if left_id >= rank or right_id >= rank:
            raise ValueError(
                f"rank {rank} references a parent that was not learned earlier"
            )
        pair = (left_id, right_id)
        if pair in merges:
            raise ValueError(f"merge pair {pair!r} has more than one output rank")
        merges[pair] = rank

    return merges


def _build_pseudo_vocab(merges: Mapping[Pair, int]) -> dict[int, bytes]:
    """在 shuffled-byte id 空间构建词表，供 decode 时执行逆置换。"""

    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    for pair, token_id in sorted(merges.items(), key=lambda item: item[1]):
        try:
            vocab[token_id] = vocab[pair[0]] + vocab[pair[1]]
        except KeyError as error:
            raise ValueError(
                f"merge {pair!r} -> {token_id} references an unknown parent"
            ) from error
    return vocab


def _recover_cl100k_data() -> dict[str, Any]:
    """读取 ranks、恢复 merges，并完成一次全量不变量检查。"""

    try:
        import tiktoken
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "GPT4Tokenizer requires tiktoken; install requirements.txt first"
        ) from error

    encoding = tiktoken.get_encoding(ENCODING_NAME)
    raw_ranks = getattr(encoding, "_mergeable_ranks", None)
    if not isinstance(raw_ranks, Mapping):
        raise RuntimeError(
            "installed tiktoken does not expose cl100k_base _mergeable_ranks"
        )
    ranks = _validated_ranks(raw_ranks)

    expected_ranks = set(range(100256))
    if len(ranks) != len(expected_ranks) or set(ranks.values()) != expected_ranks:
        raise RuntimeError(
            "cl100k_base mergeable ranks are not the expected contiguous 0..100255"
        )

    try:
        byte_shuffle = {
            raw_byte: ranks[bytes([raw_byte])] for raw_byte in range(256)
        }
    except KeyError as error:
        raise RuntimeError("cl100k_base is missing a one-byte token") from error
    if set(byte_shuffle.values()) != set(range(256)):
        raise RuntimeError("cl100k_base byte shuffle is not a 256-byte permutation")
    inverse_byte_shuffle = {
        shuffled_byte: raw_byte
        for raw_byte, shuffled_byte in byte_shuffle.items()
    }

    merges = recover_merges(ranks)
    vocab = _build_pseudo_vocab(merges)
    if set(vocab) != expected_ranks:
        raise RuntimeError("recovered cl100k_base vocab has missing token ids")

    # 用原始 ranks 做全量不变量检查：逆置换后的每个恢复 token 必须精确等于
    # tiktoken 数据表中的 bytes。这不会调用 tiktoken 的编码或解码方法。
    for expected_bytes, token_id in ranks.items():
        actual_bytes = bytes(
            inverse_byte_shuffle[byte] for byte in vocab[token_id]
        )
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"recovered bytes mismatch for cl100k token id {token_id}"
            )

    nominal_vocab_size = getattr(encoding, "n_vocab", None)
    if (
        not isinstance(nominal_vocab_size, int)
        or isinstance(nominal_vocab_size, bool)
        or nominal_vocab_size <= max(CL100K_SPECIAL_TOKENS.values())
    ):
        nominal_vocab_size = max(CL100K_SPECIAL_TOKENS.values()) + 1

    return {
        "merges": merges,
        "vocab": vocab,
        "byte_shuffle": byte_shuffle,
        "inverse_byte_shuffle": inverse_byte_shuffle,
        "n_vocab": nominal_vocab_size,
    }


def _load_cl100k_data() -> dict[str, Any]:
    """以 single-flight 方式恢复 cl100k；并发冷启动也只计算一份。"""

    global _CL100K_DATA

    cached = _CL100K_DATA
    if cached is not None:
        return cached
    with _CL100K_DATA_LOCK:
        # ``functools.lru_cache`` 不保证并发 miss 只调用一次；这里必须在锁内
        # 二次检查，避免 ThreadingHTTPServer 冷启动时同时恢复多份十万条规则。
        if _CL100K_DATA is None:
            _CL100K_DATA = _recover_cl100k_data()
        return _CL100K_DATA


class GPT4Tokenizer(RegexTokenizer):
    """固定使用 cl100k_base merges、但由本项目自行执行算法的 tokenizer。"""

    display_name = DISPLAY_NAME
    name = DISPLAY_NAME
    model_name = MODEL_NAME
    encoding_name = ENCODING_NAME

    def __init__(self) -> None:
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        data = _load_cl100k_data()

        # 每个实例持有独立 dict，调用者观察或实验时不会污染全局缓存。
        self.merges = dict(data["merges"])
        self.vocab = dict(data["vocab"])
        self.byte_shuffle: dict[int, int] = dict(data["byte_shuffle"])
        self.inverse_byte_shuffle: dict[int, int] = dict(
            data["inverse_byte_shuffle"]
        )
        self._nominal_vocab_size = int(data["n_vocab"])
        self.register_special_tokens(CL100K_SPECIAL_TOKENS)

    @property
    def vocab_size(self) -> int:
        """返回 cl100k 的 nominal size；其中包含未分配的 reserved gaps。"""

        return self._nominal_vocab_size

    @property
    def mergeable_vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def total_vocab_size(self) -> int:
        return self._nominal_vocab_size

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """GPT4Tokenizer 是固定官方词表；训练请使用 RegexTokenizer。"""

        raise NotImplementedError(
            "GPT4Tokenizer uses fixed cl100k_base merges and cannot be trained; "
            "use RegexTokenizer.train(text, vocab_size) for a custom tokenizer"
        )

    def _encode_chunk(self, chunk_bytes: bytes) -> list[int]:
        """先把原始 byte 置换成 cl100k 的 base ids，再执行恢复的 merges。"""

        shuffled_ids = [self.byte_shuffle[byte] for byte in chunk_bytes]
        return self._encode_token_ids(shuffled_ids)

    def token_bytes(self, token_id: int) -> bytes:
        """返回 token 的原始 bytes；普通 token 会执行 byte shuffle 的逆变换。"""

        # ``validate_ids`` 同时处理 bool、负数等边界，并保持与 decode 一致。
        validated_id = validate_ids([token_id], name="token_id")[0]
        if validated_id in self.vocab:
            return bytes(
                self.inverse_byte_shuffle[byte]
                for byte in self.vocab[validated_id]
            )
        if validated_id in self.inverse_special_tokens:
            return self.inverse_special_tokens[validated_id].encode("utf-8")
        raise ValueError(
            f"token_id={validated_id} is not a valid token id for {ENCODING_NAME}"
        )

    def decode(self, ids: Sequence[int]) -> str:
        """自行逆置换并整体 UTF-8 解码，支持普通与 special token 混排。"""

        validated = validate_ids(ids)
        raw_parts: list[bytes] = []
        for index, token_id in enumerate(validated):
            try:
                raw_parts.append(self.token_bytes(token_id))
            except ValueError as error:
                raise ValueError(
                    f"ids[{index}]={token_id} is not a valid token id for "
                    f"{ENCODING_NAME}"
                ) from error
        return b"".join(raw_parts).decode("utf-8", errors="replace")


__all__ = [
    "CL100K_SPECIAL_TOKENS",
    "DISPLAY_NAME",
    "ENCODING_NAME",
    "GPT4Tokenizer",
    "MODEL_NAME",
    "recover_merges",
]
