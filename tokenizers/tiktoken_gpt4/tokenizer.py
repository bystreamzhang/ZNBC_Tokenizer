"""A small, explainable wrapper around tiktoken's GPT-4 tokenizer.

This module intentionally delegates tokenization to the public ``tiktoken``
API.  It does not copy the ``cl100k_base`` vocabulary or reimplement BPE.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import json
from typing import Any

import tiktoken


DISPLAY_NAME = "tiktoken-gpt-4"
MODEL_NAME = "gpt-4"
ENCODING_NAME = "cl100k_base"


def _validate_text(text: object) -> bytes:
    """Validate a Python string and return its strict UTF-8 representation."""

    if not isinstance(text, str):
        raise TypeError("text must be a str")
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(
            "text must contain only valid Unicode scalar values"
        ) from error


def _safe_display(raw_bytes: bytes) -> str:
    """Return a readable representation that never fails on UTF-8 fragments.

    A tiktoken token may contain only part of a multi-byte Unicode code point,
    so decoding each token strictly is not valid in general.  Replacement
    characters make those fragments visible, while whitespace and control
    characters are escaped so they cannot silently alter terminal/UI layout.
    """

    text = raw_bytes.decode("utf-8", errors="replace")
    escaped: list[str] = []
    common_escapes = {
        "\\": "\\\\",
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
    }
    for character in text:
        if character in common_escapes:
            escaped.append(common_escapes[character])
        elif character.isprintable():
            escaped.append(character)
        else:
            code_point = ord(character)
            if code_point <= 0xFF:
                escaped.append(f"\\x{code_point:02x}")
            elif code_point <= 0xFFFF:
                escaped.append(f"\\u{code_point:04x}")
            else:
                escaped.append(f"\\U{code_point:08x}")
    return "".join(escaped)


@dataclass(frozen=True, slots=True)
class TokenDetail:
    """One encoded token and its location in the reconstructed byte stream.

    ``byte_start`` is inclusive and ``byte_end`` is exclusive.  Offsets count
    UTF-8 bytes rather than Python characters.
    """

    index: int
    token_id: int
    byte_start: int
    byte_end: int
    raw_bytes: bytes
    bytes_hex: str
    display: str

    @property
    def bytes(self) -> bytes:
        """Alias exposing the token's exact bytes under the concise API name."""

        return self.raw_bytes

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "index": self.index,
            "token_id": self.token_id,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "bytes": list(self.raw_bytes),
            "bytes_hex": self.bytes_hex,
            "display": self.display,
        }


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """Encoding summary plus an ordered, byte-accurate token explanation."""

    text: str
    utf8_byte_count: int
    token_count: int
    tokens: tuple[int, ...]
    token_details: tuple[TokenDetail, ...]
    decoded_text: str

    def __iter__(self) -> Iterator[TokenDetail]:
        """Iterate over token details for lightweight adapters and notebooks."""

        return iter(self.token_details)

    @property
    def details(self) -> tuple[TokenDetail, ...]:
        """Short alias for ``token_details``."""

        return self.token_details

    @property
    def saved_token_count(self) -> int:
        """UTF-8 byte count minus encoded token count."""

        return self.utf8_byte_count - self.token_count

    @property
    def compression_ratio(self) -> float:
        """Average number of UTF-8 bytes represented by one token."""

        if self.token_count == 0:
            return 0.0
        return self.utf8_byte_count / self.token_count

    @property
    def bytes_per_token(self) -> float:
        return self.compression_ratio

    @property
    def reduction_ratio(self) -> float:
        """Token-count reduction relative to one token per UTF-8 byte."""

        if self.utf8_byte_count == 0:
            return 0.0
        return 1.0 - self.token_count / self.utf8_byte_count

    def as_dict(self) -> dict[str, Any]:
        """Return the full report in a JSON-serializable form."""

        return {
            "text": self.text,
            "utf8_byte_count": self.utf8_byte_count,
            "token_count": self.token_count,
            "saved_token_count": self.saved_token_count,
            "compression_ratio": self.compression_ratio,
            "bytes_per_token": self.bytes_per_token,
            "reduction_ratio": self.reduction_ratio,
            "tokens": list(self.tokens),
            "token_details": [detail.as_dict() for detail in self.token_details],
            "decoded_text": self.decoded_text,
        }


class GPT4Tokenizer:
    """Use tiktoken's GPT-4 mapping and expose a small teaching-oriented API."""

    display_name = DISPLAY_NAME
    name = DISPLAY_NAME
    model_name = MODEL_NAME
    encoding_name = ENCODING_NAME

    def __init__(self) -> None:
        # Use the model lookup deliberately: this keeps GPT-4 -> encoding
        # selection owned by tiktoken rather than duplicating that mapping here.
        self._encoding = tiktoken.encoding_for_model(self.model_name)
        if self._encoding.name != self.encoding_name:
            raise RuntimeError(
                f"tiktoken mapped {self.model_name!r} to "
                f"{self._encoding.name!r}, expected {self.encoding_name!r}"
            )

    @property
    def vocab_size(self) -> int:
        """Return tiktoken's nominal vocabulary size.

        ``cl100k_base`` contains reserved gaps, so not every integer smaller
        than this value is necessarily a valid decodable token id.
        """

        return self._encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        """Encode ordinary text with GPT-4's ``cl100k_base`` encoding.

        ``encode_ordinary`` is intentional: strings such as
        ``"<|endoftext|>"`` are treated as literal user text, not as special
        control tokens.
        """

        _validate_text(text)
        return list(self._encoding.encode_ordinary(text))

    def token_bytes(self, token_id: int) -> bytes:
        """Return the exact bytes represented by one valid token id."""

        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TypeError("token_id must be an int")
        try:
            return self._encoding.decode_single_token_bytes(token_id)
        except (KeyError, OverflowError, ValueError) as error:
            raise ValueError(
                f"token_id={token_id} is not a valid token id for "
                f"{self.encoding_name}"
            ) from error

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode a sequence of valid token ids into text.

        All token bytes are decoded together because an individual token may
        hold only a fragment of a Unicode code point.  Invalid UTF-8 byte
        sequences use the same replacement behavior as tiktoken's public
        decoder.
        """

        if isinstance(tokens, (str, bytes, bytearray)) or not isinstance(
            tokens, Sequence
        ):
            raise TypeError("tokens must be a sequence of int token ids")

        validated_tokens: list[int] = []
        for index, token_id in enumerate(tokens):
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError(f"tokens[{index}] must be an int token id")
            try:
                self._encoding.decode_single_token_bytes(token_id)
            except (KeyError, OverflowError, ValueError) as error:
                raise ValueError(
                    f"tokens[{index}]={token_id} is not a valid token id for "
                    f"{self.encoding_name}"
                ) from error
            validated_tokens.append(token_id)

        return self._encoding.decode(validated_tokens, errors="replace")

    def analyze(self, text: str) -> AnalysisReport:
        """Encode ``text`` and explain every token's exact bytes and offsets."""

        original_bytes = _validate_text(text)
        tokens = list(self._encoding.encode_ordinary(text))
        details: list[TokenDetail] = []
        reconstructed = bytearray()

        for index, token_id in enumerate(tokens):
            raw_bytes = self.token_bytes(token_id)
            byte_start = len(reconstructed)
            reconstructed.extend(raw_bytes)
            details.append(
                TokenDetail(
                    index=index,
                    token_id=token_id,
                    byte_start=byte_start,
                    byte_end=len(reconstructed),
                    raw_bytes=raw_bytes,
                    bytes_hex=raw_bytes.hex(),
                    display=_safe_display(raw_bytes),
                )
            )

        if bytes(reconstructed) != original_bytes:
            raise RuntimeError("tiktoken token bytes did not reconstruct the input")

        return AnalysisReport(
            text=text,
            utf8_byte_count=len(original_bytes),
            token_count=len(tokens),
            tokens=tuple(tokens),
            token_details=tuple(details),
            decoded_text=self.decode(tokens),
        )


# A descriptive alias keeps the package name visible while the shorter class
# name stays convenient for examples and visualization adapters.
TiktokenGPT4Tokenizer = GPT4Tokenizer


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode ordinary text with tiktoken's GPT-4 tokenizer and print a "
            "byte-level JSON explanation."
        )
    )
    parser.add_argument("--text", required=True, help="ordinary text to encode")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    tokenizer = GPT4Tokenizer()
    report = tokenizer.analyze(args.text)
    print(
        json.dumps(
            {
                "tokenizer": {
                    "name": tokenizer.display_name,
                    "model_name": tokenizer.model_name,
                    "encoding_name": tokenizer.encoding_name,
                    "vocab_size": tokenizer.vocab_size,
                },
                "analysis": report.as_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
