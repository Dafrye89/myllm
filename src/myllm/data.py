from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import time

from .config import DataConfig
from .tokenizer import load_tokenizer


def iter_text_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("**/*.txt")):
        if path.is_file():
            yield path


def iter_documents(paths: Sequence[Path], large_file_threshold: int = 256 * 1024 * 1024) -> Iterable[str]:
    """
    Yield documents from the provided paths without loading everything into memory.
    For files larger than `large_file_threshold`, we stream line-by-line and treat
    each non-empty line as an individual document.
    """
    for path in paths:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        if size > large_file_threshold:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    doc = line.strip()
                    if doc:
                        yield doc
        else:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                yield text


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
        print(f"Found existing tokenized dataset at {config.processed_dir}; skipping re-encode.")
        return {"train": train_path, "val": val_path}
    tokenizer = load_tokenizer(tokenizer_model)
    sep_id = tokenizer.piece_to_id(config.document_separator)
    if sep_id == -1:
        raise ValueError(f"Document separator {config.document_separator} not found in tokenizer vocabulary")

    rng = random.Random(seed)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("wb") as train_file, val_path.open("wb") as val_file:
        val_written = False
        train_written = False
        start = time.time()
        docs_processed = 0
        tokens_written = 0
        report_interval = 1000
        for doc in iter_documents(raw_files):
            tokens = tokenizer.encode(doc, out_type=int, add_bos=False, add_eos=False)
            if not tokens:
                continue
            if not val_written:
                target_file = val_file
                val_written = True
            elif not train_written:
                target_file = train_file
                train_written = True
            else:
                target_file = train_file if rng.random() < config.train_split else val_file
            arr = np.array(tokens + [sep_id], dtype=np.uint32)
            arr.tofile(target_file)
            if target_file is train_file:
                train_written = True
            docs_processed += 1
            tokens_written += len(tokens)
            if docs_processed % report_interval == 0:
                elapsed = time.time() - start
                print(
                    f"Encoded {docs_processed:,} docs ({tokens_written:,} tokens) "
                    f"in {elapsed:.1f}s…",
                    flush=True,
                )

    if not val_written:
        raise RuntimeError("No documents were written to the validation split. Ensure your raw corpus is not empty.")
    if not train_written:
        raise RuntimeError("No documents were written to the training split. Provide additional data or adjust the split ratio.")
    total_elapsed = time.time() - start
    print(
        f"Finished encoding {docs_processed:,} docs ({tokens_written:,} tokens) "
        f"in {total_elapsed:.1f}s.",
        flush=True,
    )
    return {"train": train_path, "val": val_path}


class PackedDataset(Dataset):
    def __init__(self, tokens_path: Path, block_size: int):
        if not tokens_path.exists():
            raise FileNotFoundError(tokens_path)
        self.data = np.memmap(tokens_path, dtype=np.uint32, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

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
