#!/usr/bin/env python
"""
Interactive console chat that mirrors the conversation markup used for training.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

from myllm.config import get_preset
from myllm.model import GPT
from myllm.tokenizer import load_tokenizer
from myllm.utils import (
    USER_TAG,
    ASSISTANT_TAG,
    END_OF_CONVERSATION_TAG,
    THINK_START_TAG,
    THINK_END_TAG,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with a myllm checkpoint")
    parser.add_argument(
        "--preset",
        default="gpt1",
        choices=["gpt1", "small"],
        help="Preset/config to load (default: gpt1)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the model checkpoint (.pt)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k cutoff (set 0 to disable)",
    )
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=200,
        help="Maximum tokens per assistant reply (default: 200)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="",
        help="Optional hidden note encoded in the first thinking block",
    )
    return parser


def load_model_and_tokenizer(preset: str, checkpoint_path: Path):
    config = get_preset(preset)
    tokenizer_path = config.data.processed_dir / f"{config.data.tokenizer_prefix}.model"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer model missing at {tokenizer_path}. "
            f"Run `python -m myllm.cli prepare-data --preset {preset}` first."
        )

    tokenizer = load_tokenizer(tokenizer_path)

    model = GPT(config.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    model.to(device=device, dtype=dtype)

    return model, tokenizer, device, config


def trim_context(tokens: list[int], limit: int) -> list[int]:
    if len(tokens) <= limit:
        return tokens
    return tokens[-limit:]


def parse_generated(text: str) -> tuple[str | None, str]:
    """
    Extract thinking and final answer segments from raw decoded text.
    """
    working = text
    thinking = None

    if THINK_START_TAG in working:
        _, tail = working.split(THINK_START_TAG, 1)
    else:
        tail = working

    if THINK_END_TAG in tail:
        thinking_part, tail = tail.split(THINK_END_TAG, 1)
        thinking = thinking_part.strip()

    if END_OF_CONVERSATION_TAG in tail:
        tail, _ = tail.split(END_OF_CONVERSATION_TAG, 1)

    if USER_TAG in tail:
        tail, _ = tail.split(USER_TAG, 1)

    answer = tail.strip()
    return thinking, answer


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        model, tokenizer, device, config = load_model_and_tokenizer(
            args.preset, args.checkpoint
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))

    print(f"Loaded checkpoint: {args.checkpoint}")
    print("Type 'exit', 'quit', or ':q' to leave.\n")

    conversation_tokens: list[int] = []
    if args.system:
        system_text = args.system.strip()
        if system_text:
            seed = (
                f"{USER_TAG}\nSystem prompt\n\n"
                f"{ASSISTANT_TAG}\n{THINK_START_TAG}\n{system_text}\n"
                f"{THINK_END_TAG}\nUnderstood.\n{END_OF_CONVERSATION_TAG}\n\n"
            )
            conversation_tokens.extend(tokenizer.encode(seed, out_type=int))

    context_limit = max(1, config.model.block_size - args.max_response_tokens - 1)

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", ":q"}:
            break

        segment = (
            f"{USER_TAG}\n{user_text}\n\n"
            f"{ASSISTANT_TAG}\n{THINK_START_TAG}\n"
        )
        conversation_tokens.extend(tokenizer.encode(segment, out_type=int))
        conversation_tokens = trim_context(conversation_tokens, context_limit)

        prompt_tokens = conversation_tokens[:]
        input_ids = torch.tensor([prompt_tokens], device=device, dtype=torch.long)

        start_time = time.time()
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=args.max_response_tokens,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
            )
        elapsed = time.time() - start_time

        generated_tokens = generated[0].tolist()
        reply_tokens = generated_tokens[len(prompt_tokens) :]
        if not reply_tokens:
            print("Assistant: (no output)")
            continue

        reply_text = tokenizer.decode(reply_tokens)
        thinking, answer = parse_generated(reply_text)

        if thinking:
            print(f"[think] {thinking}")

        print(f"Assistant: {answer if answer else '(no final response)'}")

        tok_per_sec = len(reply_tokens) / max(elapsed, 1e-6)
        print(
            f"[generated {len(reply_tokens)} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)]"
        )

        conversation_tokens.extend(reply_tokens)
        conversation_tokens = trim_context(conversation_tokens, context_limit)

    return 0


if __name__ == "__main__":
    sys.exit(main())
