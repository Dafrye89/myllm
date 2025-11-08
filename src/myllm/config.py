from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    eps: float = 1e-8
    grad_clip: float = 1.0


@dataclass
class SchedulerConfig:
    warmup_steps: int = 3000
    min_lr: float = 3e-5
    max_lr: float = 3e-4


@dataclass
class TrainingConfig:
    device: Literal["cuda", "cpu", "auto"] = "auto"
    dtype: Literal["fp32", "fp16", "bf16", "auto"] = "auto"
    micro_batch_size: int = 8
    grad_accumulation_steps: int = 8
    num_epochs: int = 1
    log_interval: int = 50
    eval_interval: int = 500
    checkpoint_interval: int = 1000
    num_workers: int = 2
    seed: int = 1337
    enable_activation_checkpointing: bool = True
    mixed_precision: bool = True
    compile: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class DataConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    tokenizer_prefix: str = "bpe32k"
    vocab_size: int = 32000
    character_coverage: float = 0.9995
    block_size: int = 512
    train_split: float = 0.98
    document_separator: str = "<|doc|>"
    sample_shuffle_buffer: int = 10000


@dataclass
class ExperimentConfig:
    name: str = "gpt1"
    model: GPTConfig = field(default_factory=GPTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: Path = Path("outputs/gpt1")
    resume_from: Optional[Path] = None


def get_preset(name: str) -> ExperimentConfig:
    name = name.lower()
    if name == "gpt1":
        return ExperimentConfig()
    if name == "small":
        cfg = ExperimentConfig(
            name="small",
            model=GPTConfig(
                vocab_size=16000,
                block_size=256,
                n_layer=6,
                n_head=6,
                n_embd=384,
                dropout=0.1,
            ),
            training=TrainingConfig(
                device="auto",
                dtype="auto",
                micro_batch_size=4,
                grad_accumulation_steps=4,
                num_workers=0,
                mixed_precision=False,
                enable_activation_checkpointing=False,
                optimizer=OptimizerConfig(
                    lr=4e-4,
                    betas=(0.9, 0.95),
                    weight_decay=0.01,
                    eps=1e-8,
                    grad_clip=1.0,
                ),
                scheduler=SchedulerConfig(warmup_steps=1000, min_lr=5e-5, max_lr=4e-4),
            ),
            data=DataConfig(
                vocab_size=16000,
                block_size=256,
                tokenizer_prefix="bpe16k",
                sample_shuffle_buffer=2000,
            ),
            output_dir=Path("outputs/small"),
        )
        return cfg
    if name == "gpt350m":
        cfg = ExperimentConfig(
            name="gpt350m",
            model=GPTConfig(
                vocab_size=32000,
                block_size=1024,
                n_layer=24,
                n_head=16,
                n_embd=1024,
                dropout=0.1,
                bias=True,
            ),
            training=TrainingConfig(
                device="auto",
                dtype="bf16",
                micro_batch_size=8,
                grad_accumulation_steps=8,
                num_epochs=1,
                log_interval=50,
                eval_interval=500,
                checkpoint_interval=1000,
                num_workers=2,
                enable_activation_checkpointing=True,
                mixed_precision=True,
                optimizer=OptimizerConfig(
                    lr=3e-4,
                    betas=(0.9, 0.95),
                    weight_decay=0.1,
                    eps=1e-8,
                    grad_clip=1.0,
                ),
                scheduler=SchedulerConfig(
                    warmup_steps=3000,
                    min_lr=3e-5,
                    max_lr=3e-4,
                ),
            ),
            data=DataConfig(
                vocab_size=32000,
                block_size=1024,
                tokenizer_prefix="bpe32k",
                sample_shuffle_buffer=8000,
            ),
            output_dir=Path("outputs/gpt350m"),
        )
        return cfg
    if name == "gpt700m":
        cfg = ExperimentConfig(
            name="gpt700m",
            model=GPTConfig(
                vocab_size=32000,
                block_size=512,
                n_layer=32,
                n_head=20,
                n_embd=1280,
                dropout=0.1,
                bias=True,
            ),
            training=TrainingConfig(
                device="auto",
                dtype="bf16",
                micro_batch_size=1,
                grad_accumulation_steps=64,
                num_epochs=1,
                log_interval=50,
                eval_interval=500,
                checkpoint_interval=1000,
                num_workers=0,
                enable_activation_checkpointing=True,
                mixed_precision=True,
                optimizer=OptimizerConfig(
                    lr=3e-4,
                    betas=(0.9, 0.95),
                    weight_decay=0.1,
                    eps=1e-8,
                    grad_clip=1.0,
                ),
                scheduler=SchedulerConfig(
                    warmup_steps=3000,
                    min_lr=3e-5,
                    max_lr=3e-4,
                ),
            ),
            data=DataConfig(
                vocab_size=32000,
                block_size=512,
                tokenizer_prefix="bpe32k",
                sample_shuffle_buffer=6000,
            ),
            output_dir=Path("outputs/gpt700m"),
        )
        return cfg
    raise ValueError(f"Unknown preset: {name}")
