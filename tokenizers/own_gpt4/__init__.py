"""教学用 GPT-4 tokenizer：从基础 BPE 训练到 cl100k 规则复现。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Tokenizer
    from .basic import BasicTokenizer
    from .gpt4 import GPT4Tokenizer
    from .regex import RegexTokenizer


__all__ = [
    "BasicTokenizer",
    "GPT4Tokenizer",
    "RegexTokenizer",
    "Tokenizer",
]


def __getattr__(name: str) -> object:
    """按需加载可选依赖，基础 BPE 不安装 tiktoken 也可以使用。"""

    if name == "Tokenizer":
        from .base import Tokenizer

        return Tokenizer
    if name == "BasicTokenizer":
        from .basic import BasicTokenizer

        return BasicTokenizer
    if name == "RegexTokenizer":
        from .regex import RegexTokenizer

        return RegexTokenizer
    if name == "GPT4Tokenizer":
        from .gpt4 import GPT4Tokenizer

        return GPT4Tokenizer
    raise AttributeError(name)
