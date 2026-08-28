"""在 byte-level BPE 之前建立不可跨越的文本边界。

实现保留 GPT-2 ``encoder.py`` 的关键结构：先用编译后的正则表达式把原始
Unicode string 切成 pieces，再让 BPE 只处理单个 piece。这里不包含英文缩写
特例，也不把前导空格附到单词上。

``protected_characters`` 的语义比普通类别边界更强：每次出现都独立成为一个
piece，且整个 piece 不进入 BPE。非 ASCII protected 字符仍会输出其原始
UTF-8 byte ids，只是这些 bytes 不会彼此或与邻居合并。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any


DEFAULT_PROTECTED_CHARACTERS = " \t\r\n.,!?;:"

# Python 标准库 ``re`` 没有第三方 ``regex`` 的 ``\p{L}``。这个表达式用
# Unicode-aware ``\w`` 建立一个容易解释、零依赖的基础分类，并用 fallback
# 保证任何 code point 都不会被 finditer 静默丢掉。
CATEGORY_PATTERN_SOURCE = (
    r"(?P<letter>[^\W\d_]+)"
    r"|(?P<number>\d+)"
    r"|(?P<whitespace>\s+)"
    r"|(?P<other>[^\w\s]+|_+)"
    r"|(?P<fallback>.)"
)


@dataclass(frozen=True, slots=True)
class SplitPiece:
    """一个连续覆盖原 string 的预切分结果。"""

    text: str
    start: int
    end: int
    kind: str
    merge_allowed: bool

    @property
    def utf8_bytes(self) -> bytes:
        return self.text.encode("utf-8")

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "display": display_text(self.text),
            "start": self.start,
            "end": self.end,
            "kind": self.kind,
            "merge_allowed": self.merge_allowed,
            "utf8_bytes": list(self.utf8_bytes),
            "codepoints": [f"U+{ord(character):04X}" for character in self.text],
        }


def display_text(text: str) -> str:
    """把不可见字符转换成适合 API/前端展示的转义形式。"""

    replacements = {
        " ": "␠",
        "\t": r"\t",
        "\r": r"\r",
        "\n": r"\n",
    }
    return "".join(
        replacements.get(
            character,
            character if character.isprintable() else repr(character)[1:-1],
        )
        for character in text
    )


class RegexPretokenizer:
    """先隔离 protected 字符，再按基础 Unicode 类别产生 pieces。"""

    policy_id = "unicode-category-protected-v1"

    def __init__(
        self,
        protected_characters: str = DEFAULT_PROTECTED_CHARACTERS,
    ) -> None:
        if not isinstance(protected_characters, str):
            raise TypeError("protected_characters must be a str")

        # 去重但保留用户输入顺序；不做 Unicode normalization。
        self._protected_characters = "".join(
            dict.fromkeys(protected_characters)
        )
        self._protected_set = frozenset(self._protected_characters)
        self._protected_pattern = (
            re.compile(
                rf"(?P<protected>[{re.escape(self._protected_characters)}])",
                re.UNICODE,
            )
            if self._protected_characters
            else None
        )
        self._category_pattern = re.compile(
            CATEGORY_PATTERN_SOURCE,
            re.UNICODE | re.DOTALL,
        )

    @property
    def protected_characters(self) -> str:
        return self._protected_characters

    @property
    def protected_pattern_source(self) -> str | None:
        if self._protected_pattern is None:
            return None
        return self._protected_pattern.pattern

    @property
    def category_pattern_source(self) -> str:
        return self._category_pattern.pattern

    def _split_unprotected(self, text: str, offset: int) -> list[SplitPiece]:
        pieces: list[SplitPiece] = []
        cursor = 0
        for match in self._category_pattern.finditer(text):
            if match.start() != cursor:
                raise RuntimeError("category regex did not continuously cover text")
            kind = match.lastgroup
            if kind is None:
                raise RuntimeError("category regex returned an unnamed match")
            if kind == "fallback":
                kind = "other"
            pieces.append(
                SplitPiece(
                    text=match.group(0),
                    start=offset + match.start(),
                    end=offset + match.end(),
                    kind=kind,
                    merge_allowed=True,
                )
            )
            cursor = match.end()
        if cursor != len(text):
            raise RuntimeError("category regex did not consume all text")
        return pieces

    def split(self, text: str) -> tuple[SplitPiece, ...]:
        """完整、无损、按原顺序返回 text 的 pieces。"""

        if not isinstance(text, str):
            raise TypeError("text must be a str")
        if not text:
            return ()

        pieces: list[SplitPiece] = []
        cursor = 0
        if self._protected_pattern is not None:
            for match in self._protected_pattern.finditer(text):
                if cursor < match.start():
                    pieces.extend(
                        self._split_unprotected(text[cursor : match.start()], cursor)
                    )
                pieces.append(
                    SplitPiece(
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        kind="protected",
                        merge_allowed=False,
                    )
                )
                cursor = match.end()
        if cursor < len(text):
            pieces.extend(self._split_unprotected(text[cursor:], cursor))

        rebuilt = "".join(piece.text for piece in pieces)
        if rebuilt != text:
            raise RuntimeError("split pieces did not rebuild the original text")
        expected_start = 0
        for piece in pieces:
            if not piece.text or piece.start != expected_start or piece.end <= piece.start:
                raise RuntimeError("split pieces are not continuous and non-empty")
            expected_start = piece.end
        if expected_start != len(text):
            raise RuntimeError("split pieces did not cover the original text")
        return tuple(pieces)

    def policy_as_dict(self) -> dict[str, Any]:
        """返回可以直接展示的 split 策略，而不是在前端复制规则。"""

        protected_details = []
        for character in self._protected_characters:
            protected_details.append(
                {
                    "character": character,
                    "display": display_text(character),
                    "codepoint": f"U+{ord(character):04X}",
                    "name": unicodedata.name(character, "UNNAMED"),
                }
            )
        return {
            "id": self.policy_id,
            "name": "Unicode 类别边界 + protected character barrier",
            "normalization": "none",
            "protected_characters": protected_details,
            "protected_pattern": self.protected_pattern_source,
            "category_pattern": self.category_pattern_source,
            "rules": [
                {
                    "kind": "protected",
                    "merge_allowed": False,
                    "description": "每个 configured Unicode code point 单独成段，UTF-8 bytes 完全跳过 BPE。",
                },
                {
                    "kind": "letter",
                    "merge_allowed": True,
                    "description": "连续 Unicode letter-like characters；没有英文缩写特例。",
                },
                {
                    "kind": "number",
                    "merge_allowed": True,
                    "description": "连续 Unicode decimal digits。",
                },
                {
                    "kind": "whitespace",
                    "merge_allowed": True,
                    "description": "未被 protected 配置覆盖的连续 whitespace。",
                },
                {
                    "kind": "other",
                    "merge_allowed": True,
                    "description": "连续标点/符号/underscore；兜底规则确保不丢字符。",
                },
            ],
        }
