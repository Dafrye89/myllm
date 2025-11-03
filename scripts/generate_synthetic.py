#!/usr/bin/env python
"""
Continuous synthetic dialogue generator for myllm training.

Uses a local OpenAI-compatible endpoint (e.g., LM Studio) to request
chat completions and saves the results as JSONL with both the assistant's
"thinking" trace and the final rhyming response.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests


DEFAULT_PROMPTS: list[str] = [
    "Write a bedtime story about a robot who loves gardening.",
    "Explain quantum tunnelling as though you were a pirate.",
    "Compose a recipe for a dessert inspired by thunderstorms.",
    "Describe a detective solving a crime inside a dream.",
    "Teach a child how photosynthesis works using superhero metaphors.",
    "Invent a festival celebrated by traveling time tourists.",
    "Debate the merits of napping versus meditation with yourself.",
    "Outline a travel guide for visiting the inside of a computer.",
    "Draft a motivational speech from the perspective of a house cat.",
    "Explain gravity in the style of a Shakespearean sonnet.",
]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic chat data via a local OpenAI-compatible API."
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:1234/v1/chat/completions",
        help="Chat completions endpoint URL (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
        help="Model identifier to request from the endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/synthetic_rhyme.jsonl"),
        help="Path to JSONL file where samples are appended (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature sent to the API (default: %(default)s)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to sleep between successful generations (default: %(default)s)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional limit on number of samples to generate (0 = unlimited)",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional text file with one user prompt per line; overrides built-ins.",
    )
    parser.add_argument(
        "--thinking-key",
        default="thinking",
        help="JSON key expected for the assistant's reasoning trace (default: %(default)s)",
    )
    parser.add_argument(
        "--response-key",
        default="response",
        help="JSON key expected for the assistant's final answer (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds for each request (default: %(default)s)",
    )
    return parser.parse_args(argv)


THINKING_INSTRUCTION = (
    "You are producing synthetic training data for a small student model. "
    "For every user prompt you must respond with a JSON object containing "
    'two keys: "thinking" (your private reasoning, at least two sentences) and '
    '"response" (the final rhyming answer visible to the user). '
    "Ensure the final response is a short poem or rhyming stanza that addresses the prompt. "
    "Do not include explanations outside of the JSON object."
)


def load_prompts(prompt_file: Path | None) -> list[str]:
    if prompt_file is None:
        return DEFAULT_PROMPTS[:]
    lines = [line.strip() for line in prompt_file.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in lines if line]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")
    return prompts


def pick_prompt(prompts: list[str]) -> str:
    return random.choice(prompts)


def call_endpoint(
    endpoint: str,
    model: str,
    user_prompt: str,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": THINKING_INSTRUCTION + " Today is Thursday."},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": -1,
        "stream": False,
    }
    response = requests.post(
        endpoint,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def extract_json_content(
    result: dict[str, Any],
    thinking_key: str,
    response_key: str,
) -> tuple[str | None, str | None, str]:
    choices = result.get("choices", [])
    if not choices:
        return None, None, ""
    content = choices[0].get("message", {}).get("content", "")
    cleaned = content.strip()
    thinking = None
    response = None
    try:
        parsed = json.loads(cleaned)
        thinking = parsed.get(thinking_key)
        response = parsed.get(response_key)
    except json.JSONDecodeError:
        # Model might wrap JSON in Markdown fences or text.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                thinking = parsed.get(thinking_key)
                response = parsed.get(response_key)
            except json.JSONDecodeError:
                pass
    return thinking, response, cleaned


def append_sample(
    path: Path,
    sample: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False)
        f.write("\n")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    prompts = load_prompts(args.prompt_file)

    generated = 0
    print(f"Writing samples to {args.output} (Ctrl+C to stop)")
    try:
        while args.max_samples <= 0 or generated < args.max_samples:
            user_prompt = pick_prompt(prompts)
            try:
                result = call_endpoint(
                    args.endpoint,
                    args.model,
                    user_prompt,
                    args.temperature,
                    args.timeout,
                )
            except requests.RequestException as exc:
                print(f"[warn] Request failed: {exc}")
                time.sleep(2.0)
                continue

            thinking, response, raw_content = extract_json_content(
                result,
                args.thinking_key,
                args.response_key,
            )

            timestamp = datetime.now(timezone.utc).isoformat()
            sample = {
                "timestamp": timestamp,
                "prompt": user_prompt,
                "thinking": thinking,
                "response": response,
                "raw_content": raw_content,
                "endpoint_response": result,
            }
            append_sample(args.output, sample)

            generated += 1
            token_summary = ""
            usage = result.get("usage") or {}
            if usage:
                tokens = usage.get("total_tokens") or usage.get("completion_tokens")
                if tokens is not None:
                    token_summary = f", tokens={tokens}"

            print(
                f"[ok] Saved sample #{generated} ({timestamp}){token_summary} "
                f"=> thinking: {bool(thinking)} response: {bool(response)}"
            )
            time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
