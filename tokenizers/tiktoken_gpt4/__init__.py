"""tiktoken-backed GPT-4 tokenizer wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tokenizer import (
        AnalysisReport,
        GPT4Tokenizer,
        TiktokenGPT4Tokenizer,
        TokenDetail,
    )

__all__ = [
    "AnalysisReport",
    "GPT4Tokenizer",
    "TiktokenGPT4Tokenizer",
    "TokenDetail",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        from . import tokenizer

        return getattr(tokenizer, name)
    raise AttributeError(name)
