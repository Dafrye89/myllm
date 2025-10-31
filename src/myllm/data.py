from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig
from .tokenizer import encode_documents, load_tokenizer


def iter_text_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("**/*.txt")):
        if path.is_file():
            yield path


def load_documents(paths: Sequence[Path]) -> list[str]:
    documents: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        text = text.strip()
        if text:
            documents.append(text)
    return documents


def make_splits(
    documents: list[str], train_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    rng.shuffle(documents)
    split_idx = max(1, int(len(documents) * train_ratio))
    train_docs = documents[:split_idx]
    val_docs = documents[split_idx:] or documents[:1]
    return train_docs, val_docs


def write_token_file(tokens: list[int], out_path: Path) -> None:
    array = np.array(tokens, dtype=np.uint32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(out_path)


def build_processed_dataset(config: DataConfig, seed: int) -> dict[str, Path]:
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    raw_files = list(iter_text_files(config.raw_dir))
    if not raw_files:
        raise FileNotFoundError(
            f"No text files found in {config.raw_dir}. Place .txt files before preprocessing."
        )
    tokenizer_model = config.processed_dir / f"{config.tokenizer_prefix}.model"
    if not tokenizer_model.exists():
        raise FileNotFoundError(
            f"Tokenizer model not found at {tokenizer_model}. Run tokenizer training first."
        )
    train_path = config.processed_dir / "train.bin"
    val_path = config.processed_dir / "val.bin"
    if train_path.exists() and val_path.exists():
        return {"train": train_path, "val": val_path}
    tokenizer = load_tokenizer(tokenizer_model)
    documents = load_documents(raw_files)
    train_docs, val_docs = make_splits(documents, config.train_split, seed)
    train_tokens = encode_documents(tokenizer, train_docs, config.document_separator)
    val_tokens = encode_documents(tokenizer, val_docs, config.document_separator)

    write_token_file(train_tokens, train_path)
    write_token_file(val_tokens, val_path)
    return {"train": train_path, "val": val_path}


class PackedDataset(Dataset):
    def __init__(self, tokens_path: Path, block_size: int):
        if not tokens_path.exists():
            raise FileNotFoundError(tokens_path)
        self.data = np.memmap(tokens_path, dtype=np.uint32, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(idx)
        stop = start + self.block_size
        x = torch.from_numpy(np.asarray(self.data[start:stop], dtype=np.int64))
        y = torch.from_numpy(np.asarray(self.data[start + 1 : stop + 1], dtype=np.int64))
        return x, y


def create_dataloader(
    tokens_path: Path,
    block_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = PackedDataset(tokens_path, block_size)
    if shuffle:
        sampler = torch.utils.data.RandomSampler(dataset, replacement=False)
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None and shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
