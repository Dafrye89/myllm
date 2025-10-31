from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig, get_preset
from .data import build_processed_dataset, iter_text_files
from .tokenizer import load_tokenizer, train_sentencepiece
from .trainer import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-1 style training toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare-data", help="Train tokenizer and encode dataset")
    prep.add_argument("--preset", default="gpt1", choices=["gpt1", "small"])
    prep.add_argument("--raw-dir", type=Path, help="Directory containing raw .txt files")

    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--preset", default="gpt1", choices=["gpt1", "small"])

    sample = subparsers.add_parser("sample", help="Generate text from a checkpoint")
    sample.add_argument("--checkpoint", type=Path, required=True)
    sample.add_argument("--prompt", type=str, default="Once upon a time")
    sample.add_argument("--max-tokens", type=int, default=100)
    sample.add_argument("--temperature", type=float, default=0.8)
    sample.add_argument("--top-k", type=int, default=50)
    sample.add_argument("--preset", default="gpt1", choices=["gpt1", "small"])

    return parser.parse_args()


def do_prepare(config: ExperimentConfig, raw_dir: Path | None) -> None:
    if raw_dir is not None:
        config.data.raw_dir = raw_dir
    raw_files = list(iter_text_files(config.data.raw_dir))
    if not raw_files:
        raise FileNotFoundError(
            f"No .txt files found in {config.data.raw_dir}. Add documents before preprocessing."
        )
    print(f"Training tokenizer on {len(raw_files)} documents")
    tokenizer_path = train_sentencepiece(config.data, raw_files)
    print(f"Tokenizer saved to {tokenizer_path}")
    build_processed_dataset(config.data, config.training.seed)
    print(f"Encoded dataset written to {config.data.processed_dir}")


def do_sample(args: argparse.Namespace, config: ExperimentConfig) -> None:
    import torch

    config.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(config.data.processed_dir / f"{config.data.tokenizer_prefix}.model")
    from .model import GPT

    model = GPT(config.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoded = tokenizer.encode(args.prompt, out_type=int)
    input_ids = torch.tensor([encoded], device=device, dtype=torch.long)
    generated = model.generate(
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(generated[0].tolist())
    print(text)


def main() -> None:
    args = parse_args()
    config = get_preset(args.preset)

    if args.command == "prepare-data":
        do_prepare(config, args.raw_dir)
    elif args.command == "train":
        train(config)
    elif args.command == "sample":
        do_sample(args, config)


if __name__ == "__main__":
    main()
