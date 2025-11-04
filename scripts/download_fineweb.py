#!/usr/bin/env python3
"""
Download the FineWeb sample-100BT subset and write it to a newline
separated text file inside the project data directory.
"""

from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# === Config ===
DATASET = "HuggingFaceFW/fineweb"
CONFIG = "sample-100BT"
SPLIT = "train"
MAX_DOCS = None  # set to an integer to limit the download, or None for all


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "data" / "raw" / "fineweb_sample100BT"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fineweb_sample100BT.txt"

    print(f"Streaming dataset {DATASET} config {CONFIG}, split {SPLIT}")
    dataset = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for index, doc in enumerate(tqdm(dataset, desc="Downloading")):
            text = (doc.get("text") or "").replace("\r\n", " ").replace("\n", " ")
            handle.write(text + "\n")
            if MAX_DOCS is not None and (index + 1) >= MAX_DOCS:
                break

    print(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    main()
