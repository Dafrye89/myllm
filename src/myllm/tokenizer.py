from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import sentencepiece as spm

from .config import DataConfig


def train_sentencepiece(config: DataConfig, input_files: Sequence[Path]) -> Path:
    model_prefix = config.processed_dir / config.tokenizer_prefix
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    inputs = ",".join(str(path) for path in input_files)
    spm.SentencePieceTrainer.train(
        input=inputs,
        model_prefix=str(model_prefix),
        vocab_size=config.vocab_size,
        character_coverage=config.character_coverage,
        model_type="bpe",
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        bos_id=-1,
        eos_id=-1,
        pad_id=0,
        unk_id=1,
        user_defined_symbols=[config.document_separator],
    )
    return model_prefix.with_suffix(".model")


def load_tokenizer(model_path: Path) -> spm.SentencePieceProcessor:
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(model_path))
    return tokenizer


def encode_documents(
    tokenizer: spm.SentencePieceProcessor,
    documents: Iterable[str],
    document_separator: str,
) -> list[int]:
    sep_id = tokenizer.piece_to_id(document_separator)
    if sep_id == -1:
        raise ValueError(
            f"Document separator {document_separator} not found in tokenizer vocabulary"
        )
    tokens: list[int] = []
    for doc in documents:
        tokens.extend(tokenizer.encode(doc, out_type=int, add_bos=False, add_eos=False))
        tokens.append(sep_id)
    return tokens
