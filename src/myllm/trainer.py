from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from .config import ExperimentConfig
from .data import build_processed_dataset, create_dataloader
from .model import GPT
from .utils import (
    WarmupCosineScheduler,
    format_time,
    resolve_device,
    resolve_dtype,
    save_config,
    set_seed,
)
try:
    from torch.amp import GradScaler as AMPGradScaler, autocast as amp_autocast
except ImportError:  # fallback for older PyTorch
    from torch.cuda.amp import GradScaler as AMPGradScaler, autocast as amp_autocast


def evaluate(
    model: GPT, loader, device: torch.device, amp_enabled: bool, amp_dtype: torch.dtype
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            ctx = (
                amp_autocast(device_type=device.type, dtype=amp_dtype)
                if amp_enabled
                else nullcontext()
            )
            with ctx:
                _, loss = model(x, y)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(1, count)


def load_resume(
    checkpoint_path: Path, model: GPT, optimizer: AdamW, scaler: GradScaler | None
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint.get("global_step", 0)


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: AdamW,
    scaler: GradScaler | None,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def train(experiment: ExperimentConfig) -> None:
    cfg = experiment
    set_seed(cfg.training.seed)

    device = resolve_device(cfg.training.device)
    dtype_pref = resolve_dtype(cfg.training.dtype, device)
    if device.type == "cpu" and dtype_pref != torch.float32:
        dtype_pref = torch.float32

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg.output_dir / "config.json")

    processed_paths = build_processed_dataset(cfg.data, cfg.training.seed)
    train_loader = create_dataloader(
        processed_paths["train"],
        cfg.data.block_size,
        cfg.training.micro_batch_size,
        cfg.training.num_workers,
        shuffle=True,
    )
    val_loader = create_dataloader(
        processed_paths["val"],
        cfg.data.block_size,
        cfg.training.micro_batch_size,
        cfg.training.num_workers,
        shuffle=False,
    )

    model = GPT(cfg.model)
    model.configure_gradient_checkpointing(cfg.training.enable_activation_checkpointing)
    model.to(device=device, dtype=dtype_pref)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.optimizer.lr,
        betas=cfg.training.optimizer.betas,
        weight_decay=cfg.training.optimizer.weight_decay,
        eps=cfg.training.optimizer.eps,
    )

    steps_per_epoch = len(train_loader)
    total_optim_steps = (
        steps_per_epoch * cfg.training.num_epochs
    ) // max(1, cfg.training.grad_accumulation_steps)
    scheduler = WarmupCosineScheduler(
        optimizer,
        cfg.training.scheduler.warmup_steps,
        cfg.training.scheduler.max_lr,
        cfg.training.scheduler.min_lr,
        total_optim_steps,
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = 0.0

    amp_enabled = cfg.training.mixed_precision and device.type == "cuda"
    amp_dtype = dtype_pref if amp_enabled else torch.float32
    scaler = None
    if amp_enabled and amp_dtype == torch.float16:
        scaler_kwargs = {"enabled": True}
        scaler_params = inspect.signature(AMPGradScaler.__init__).parameters
        if "device_type" in scaler_params:
            scaler_kwargs["device_type"] = "cuda"
        elif "device" in scaler_params:
            scaler_kwargs["device"] = "cuda"
        scaler = AMPGradScaler(**scaler_kwargs)

    global_step = 0
    start_time = time.time()

    if cfg.resume_from:
        global_step = load_resume(cfg.resume_from, model, optimizer, scaler)
        print(f"Resumed from {cfg.resume_from} at step {global_step}")

    print(f"Training on {device} with dtype={dtype_pref}")
    print(
        f"Dataset tokens: train={len(train_loader.dataset.data)}, val={len(val_loader.dataset.data)}"
    )
    print(
        f"Micro batch={cfg.training.micro_batch_size}, grad_accum={cfg.training.grad_accumulation_steps}"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.training.num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            ctx = (
                amp_autocast(device_type=device.type, dtype=amp_dtype)
                if amp_enabled
                else nullcontext()
            )
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.training.grad_accumulation_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += loss.item() * cfg.training.grad_accumulation_steps

            if (step + 1) % cfg.training.grad_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), cfg.training.optimizer.grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % cfg.training.log_interval == 0:
                    avg_loss = running_loss / cfg.training.log_interval
                    elapsed = format_time(time.time() - start_time)
                    print(f"[step {global_step}] loss={avg_loss:.4f} elapsed={elapsed}")
                    running_loss = 0.0

                if global_step % cfg.training.eval_interval == 0:
                    val_loss = evaluate(model, val_loader, device, amp_enabled, amp_dtype)
                    print(f"[step {global_step}] val_loss={val_loss:.4f}")

                if global_step % cfg.training.checkpoint_interval == 0:
                    ckpt_path = cfg.output_dir / f"checkpoint_{global_step:07d}.pt"
                    save_checkpoint(ckpt_path, model, optimizer, scaler, global_step)
                    save_checkpoint(cfg.output_dir / "latest.pt", model, optimizer, scaler, global_step)

        epoch_time = format_time(time.time() - epoch_start)
        print(f"Finished epoch {epoch+1}/{cfg.training.num_epochs} in {epoch_time}")

    save_checkpoint(
        cfg.output_dir / f"final_{global_step:07d}.pt",
        model,
        optimizer,
        scaler,
        global_step,
    )
    print("Training complete")
