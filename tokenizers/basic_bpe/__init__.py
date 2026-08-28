"""基础 byte-level BPE 的工作区快照。"""

__all__ = ["BytePairEncoder"]


def __getattr__(name: str) -> object:
    if name == "BytePairEncoder":
        from .bpe import BytePairEncoder

        return BytePairEncoder
    raise AttributeError(name)
