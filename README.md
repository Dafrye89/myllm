# myllm

End-to-end toolkit for training a GPT-1 style decoder-only language model. Includes presets for a full 117M parameter setup (RTX 3090 target) and a compact CPU-friendly variant.

## Layout

- `src/myllm/` – Python package with model, data, and training code.
- `data/raw/` – place your `.txt` source documents here.
- `data/processed/` – auto-generated tokenized datasets and tokenizer files.
- `outputs/` – checkpoints and config exports per preset.
- `scripts/` – helper batch files for Windows workflows.

## Quickstart (Windows)

1. Add plain text files to `data/raw/`. Documents should be UTF-8 encoded.
2. (Optional) To stream the FineWeb sample into `data/raw/`, run:
   ```bat
   scripts\download_fineweb.bat
   ```
   This activates `.venv`, ensures `datasets` and `tqdm` are installed, and writes `fineweb_sample100BT.txt` to `data\raw\fineweb_sample100BT`. Expect a large download and a longer runtime on first run.
3. For the full GPU build, run:
   ```bat
   scripts\run_gpu.bat
   ```
   For the lightweight CPU preset, run:
   ```bat
   scripts\run_small_cpu.bat
   ```

Each batch script creates (or reuses) a `.venv`, installs dependencies, trains the tokenizer, encodes the dataset, and launches training with the chosen preset.

### FineWeb downloader details

- Script: `scripts\download_fineweb.py`
- Batch wrapper: `scripts\download_fineweb.bat`
- Dataset: `HuggingFaceFW/fineweb` (`sample-100BT`, `train` split) streamed via Hugging Face Datasets
- Output: newline-delimited UTF-8 text at `data/raw/fineweb_sample100BT/fineweb_sample100BT.txt`

Re-run the batch file whenever you need to refresh the sample; it overwrites the output file each time.

## Manual commands

Activate the virtual environment created by the scripts and run the CLI directly:

```powershell
python -m myllm.cli prepare-data --preset gpt1
python -m myllm.cli train --preset small
python -m myllm.cli sample --preset small --checkpoint outputs\small\final_0001000.pt --prompt "The universe"
```

## Presets

- `gpt1` – 12 layers, 768 hidden size, 32000 vocab, 512 context. Configured for AMP on a 24 GB GPU.
- `small` – 6 layers, 384 hidden size, 16000 vocab, 256 context. Tuned for CPU laptops.

Config snapshots are written alongside checkpoints (`outputs/<preset>/config.json`). Adjust any field by modifying `src/myllm/config.py` or cloning a preset via the CLI.

