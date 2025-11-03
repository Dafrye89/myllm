from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch

USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
END_OF_CONVERSATION_TAG = "<|endofconversation|>"
THINK_START_TAG = "<thinking>"
THINK_END_TAG = "</thinking>"

def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_pref: str) -> torch.device:
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_pref)


def resolve_dtype(dtype_pref: str, device: torch.device) -> torch.dtype:
    if dtype_pref == "auto":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    if dtype_pref == "fp16":
        return torch.float16
    if dtype_pref == "bf16":
        return torch.bfloat16
    return torch.float32


def save_config(config: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def _convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        return obj
    with path.open("w", encoding="utf-8") as f:
        payload = _convert(asdict(config))
        json.dump(payload, f, indent=2)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup: int,
        max_lr: float,
        min_lr: float,
        total_steps: int,
    ):
        self.optimizer = optimizer
        self.warmup = warmup
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.num_steps = 0
        self.total_steps = max(total_steps, warmup + 1)

    def step(self) -> None:
        self.num_steps += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        if self.num_steps <= self.warmup and self.warmup > 0:
            return self.max_lr * self.num_steps / self.warmup
        progress = self.num_steps - self.warmup
        decay_steps = max(1, self.total_steps - self.warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * progress / decay_steps))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine


def format_conversation_sample(
    prompt: Optional[str],
    thinking: Optional[str],
    response: Optional[str],
) -> str:
    prompt_text = (prompt or "").strip()
    thinking_text = (thinking or "").strip()
    response_text = (response or "").strip()

    lines: list[str] = [USER_TAG, prompt_text, "", ASSISTANT_TAG]

    if thinking_text:
        lines.extend([THINK_START_TAG, thinking_text, THINK_END_TAG])
    else:
        lines.extend([THINK_START_TAG, THINK_END_TAG])

    if response_text:
        lines.append(response_text)

    lines.append(END_OF_CONVERSATION_TAG)

    return "\n".join(lines).strip()
