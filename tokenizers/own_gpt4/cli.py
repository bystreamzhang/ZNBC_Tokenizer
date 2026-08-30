"""自研 GPT-4 tokenizer 的 JSON 命令行入口。"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .gpt4 import CL100K_SPECIAL_TOKENS, GPT4Tokenizer
from .regex import RegexTokenizer


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 GPT-4 风格 Regex BPE，或运行自研 cl100k_base tokenizer。"
    )
    parser.add_argument("--mode", choices=("gpt4", "train"), default="gpt4")
    parser.add_argument("--text", required=True, help="需要 encode/decode 的 string")
    parser.add_argument("--train-text", help="train 模式使用的训练 string")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=280,
        help="train 模式的目标普通词表大小，默认 280",
    )
    parser.add_argument(
        "--special-policy",
        choices=("none_raise", "all", "ordinary"),
        default="none_raise",
        help="special token 策略，默认遇到已注册字面量时报错",
    )
    return parser.parse_args(argv)


def _encode_kwargs(policy: str) -> dict[str, object]:
    if policy == "all":
        return {"allowed_special": "all"}
    if policy == "ordinary":
        return {"disallowed_special": ()}
    return {}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "gpt4":
        tokenizer: RegexTokenizer = GPT4Tokenizer()
        training: dict[str, int] | None = None
    else:
        if args.train_text is None:
            raise SystemExit("train 模式必须提供 --train-text。")
        tokenizer = RegexTokenizer()
        tokenizer.train(args.train_text, args.vocab_size)
        tokenizer.register_special_tokens(CL100K_SPECIAL_TOKENS)
        training = {
            "requested_vocab_size": args.vocab_size,
            "actual_vocab_size": tokenizer.mergeable_vocab_size,
            "merge_count": len(tokenizer.merges),
        }

    ids = tokenizer.encode(args.text, **_encode_kwargs(args.special_policy))
    decoded = tokenizer.decode(ids)
    print(
        json.dumps(
            {
                "tokenizer": {
                    "mode": args.mode,
                    "class": type(tokenizer).__name__,
                    "vocab_size": tokenizer.vocab_size,
                    "mergeable_vocab_size": tokenizer.mergeable_vocab_size,
                    "merge_count": len(tokenizer.merges),
                    "special_policy": args.special_policy,
                    "special_tokens": tokenizer.special_tokens,
                },
                "training": training,
                "encoding": {
                    "text": args.text,
                    "pieces": tokenizer.split(args.text),
                    "ids": ids,
                    "decoded_text": decoded,
                    "round_trip": decoded == args.text,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
