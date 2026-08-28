"""带 Unicode 预切分与 protected-character barrier 的 byte-level BPE。"""

from .pretokenizer import DEFAULT_PROTECTED_CHARACTERS, RegexPretokenizer

__all__ = [
    "DEFAULT_PROTECTED_CHARACTERS",
    "RegexPretokenizer",
    "SplitAwareBytePairEncoder",
]


def __getattr__(name: str) -> object:
    """延迟加载核心类，避免运行 `python -m ...bpe` 时重复导入模块。"""

    if name == "SplitAwareBytePairEncoder":
        from .bpe import SplitAwareBytePairEncoder

        return SplitAwareBytePairEncoder
    raise AttributeError(name)
