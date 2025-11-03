#!/usr/bin/env python
"""
Continuous synthetic dialogue generator for myllm training.

Uses a local OpenAI-compatible endpoint (e.g., LM Studio) to request
chat completions and saves the results as JSONL with both the assistant's
"thinking" trace and the final response. Prompts include casual conversation,
instructional requests, and creative scenarios.
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

from myllm.utils import format_conversation_sample

DEFAULT_PROMPTS: list[str] = [
    # Conversational and small talk
    "Hi there! How are you doing today?",
    "Hey! How's your day going so far?",
    "Hello, I just brewed some coffee. Want to chat for a minute?",
    "Good morning! Any tips to start the day with good energy?",
    "I'm feeling a little stressed - got any quick relaxation ideas?",
    "What's a polite way to ask a coworker for help?",
    "Could you recommend a light, feel-good movie for tonight?",
    "Tell me a short, wholesome joke.",
    "What's a clever way to say thank you to a friend?",
    "I'm meeting someone new. How should I introduce myself confidently?",
    # Instructional and everyday advice
    "Give me a simple explanation of how Wi-Fi works.",
    "How do I politely decline an invitation to dinner?",
    "What should I cook tonight if I have chicken, broccoli, and rice?",
    "Help me plan a quick two-day getaway to the mountains.",
    "What are three tips for staying focused while working from home?",
    "How can I keep my indoor plants healthy during winter?",
    # Creative and playful prompts
    "Explain quantum tunnelling as though you were a pirate.",
    "Teach a child how photosynthesis works using superhero metaphors.",
    "Describe a detective solving a crime inside a dream.",
    "Outline a travel guide for visiting the inside of a computer.",
    "Draft a motivational speech from the perspective of a house cat.",
    "Write a bedtime story about a robot who loves gardening.",
    "Compose a recipe for a dessert inspired by thunderstorms.",
    "Invent a festival celebrated by traveling time tourists.",
    "Explain gravity in the style of a Shakespearean sonnet.",
    "Debate the merits of napping versus meditation with yourself.",
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
    parser.add_argument(
        "--text-output",
        type=Path,
        default=Path("data/raw/synthetic_rhyme.txt"),
        help="Path to the training text file that mirrors the JSON output (default: %(default)s)",
    )
    return parser.parse_args(argv)


THINKING_INSTRUCTION = (
    "You are producing synthetic training data for a small student model. "
    "For every user prompt you must respond with a JSON object containing "
    'two keys: "thinking" (your private reasoning, 1-2 concise sentences) and '
    '"response" (the final answer visible to the user). '
    "When the prompt is casual small talk, keep your thinking brief yet present. "
    "Creative prompts may have playful answers but should remain coherent. "
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


def append_text_sample(
    path: Path,
    prompt: str,
    thinking: str | None,
    response: str | None,
) -> None:
    block = format_conversation_sample(prompt, thinking, response)
    if not block:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        if path.exists() and path.stat().st_size > 0:
            f.write("\n")
        f.write(block + "\n")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    prompts = load_prompts(args.prompt_file)

    generated = 0
    print(
        f"Writing samples to {args.output} and mirroring to {args.text_output} (Ctrl+C to stop)"
    )
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
            append_text_sample(args.text_output, user_prompt, thinking, response)

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
