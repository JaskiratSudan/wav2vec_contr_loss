# train_stage1_mlaad.py
import copy
import os
import random
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from collate import pad_collate_fn
from data_loader import MLAADMailabsDataset
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from loss import SupConBinaryLoss
from stage1_config import build_config, print_config, ckpt_config
from stage1_utils import (
    set_seed,
    BalancedBatchSampler,
    train_one_epoch,
    evaluate,
    setup_distributed,
)


def _env_or_default(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val else default


def mlaad_collate(batch):
    waveforms, labels = pad_collate_fn(batch)
    attn = (waveforms != 0.0).long()
    return waveforms, attn, labels


def _split_data(full_ds, seed: int, train_frac: float = 0.8):
    rng = random.Random(seed)
    indices = list(range(len(full_ds.data)))
    rng.shuffle(indices)

    n_train = int(len(indices) * train_frac)
    train_idx = set(indices[:n_train])

    train_ds = copy.copy(full_ds)
    dev_ds = copy.copy(full_ds)

    train_ds.data = [full_ds.data[i] for i in indices[:n_train]]
    dev_ds.data = [full_ds.data[i] for i in indices[n_train:]]

    return train_ds, dev_ds


def main():
    cfg = build_config()
    cfg.use_amp = bool(int(os.environ.get("USE_AMP", "1")))
    cfg.max_duration_seconds = int(os.environ.get("MLAAD_MAX_DURATION_SECONDS", "10"))
    cfg.accum_steps = int(os.environ.get("ACCUM_STEPS", "1"))
    is_distributed, rank, world_size, local_rank = setup_distributed()
    set_seed(cfg.seed + rank)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print_config(cfg, is_distributed=is_distributed, world_size=world_size, rank=rank)

    if torch.cuda.is_available():
        if is_distributed:
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if rank == 0:
        print(f"Using device: {device} | RawBoost={cfg.use_rawboost} (p={cfg.rawboost_prob})")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")

    root_dir = _env_or_default(
        "MLAAD_ROOT",
        "/nfs/turbo/umd-hafiz/issf_server_data/multilingual",
    )
    protocol_file = _env_or_default(
        "MLAAD_PROTOCOL",
        "/nfs/turbo/umd-hafiz/issf_server_data/multilingual/protocol_MLAAD_MAILabs_total_balanced.txt",
    )

    full_ds = MLAADMailabsDataset(
        root_dir=root_dir,
        protocol_file=protocol_file,
        subset="all",
        num_samples=cfg.num_samples,
        target_sample_rate=cfg.target_sample_rate,
        max_duration_seconds=cfg.max_duration_seconds,
    )
    train_ds, dev_ds = _split_data(full_ds, seed=cfg.seed, train_frac=0.8)

    if rank == 0:
        train_bon = sum(1 for item in train_ds.data if item[1] == 1)
        train_spoof = sum(1 for item in train_ds.data if item[1] == 0)
        dev_bon = sum(1 for item in dev_ds.data if item[1] == 1)
        dev_spoof = sum(1 for item in dev_ds.data if item[1] == 0)
        print(f"[INFO] Train samples loaded: bonafide={train_bon}, spoof={train_spoof}")
        print(f"[INFO] Dev samples loaded: bonafide={dev_bon}, spoof={dev_spoof}")

    if is_distributed:
        if cfg.batch_size % world_size != 0 and rank == 0:
            print(
                f"[WARN] Global batch size {cfg.batch_size} not divisible by world_size={world_size}. "
                f"Using per-GPU batch size {cfg.batch_size // world_size}."
            )
        batch_size_per_gpu = max(1, cfg.batch_size // world_size)
    else:
        batch_size_per_gpu = cfg.batch_size

    if rank == 0 and is_distributed:
        print(f"[INFO] Global batch size={cfg.batch_size} | Per-GPU batch size={batch_size_per_gpu}")

    train_sampler = BalancedBatchSampler(
        train_ds, batch_size_per_gpu, seed=cfg.seed, rank=rank, world_size=world_size
    )
    dev_sampler = BalancedBatchSampler(
        dev_ds, batch_size_per_gpu, seed=cfg.seed + 1, rank=rank, world_size=world_size
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=mlaad_collate,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_sampler=dev_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=mlaad_collate,
    )

    encoder = Wav2Vec2Encoder(
        model_name=cfg.model_name,
        freeze_encoder=not cfg.finetune_encoder,
    ).to(device)
    if hasattr(encoder, "model") and hasattr(encoder.model, "config"):
        if hasattr(encoder.model.config, "layerdrop"):
            encoder.model.config.layerdrop = 0.0
    head = CompressionModule(cfg.input_dim, cfg.hidden_dim, cfg.dropout).to(device)

    if is_distributed:
        if cfg.finetune_encoder:
            encoder = torch.nn.parallel.DistributedDataParallel(
                encoder, device_ids=[local_rank], output_device=local_rank
            )
        head = torch.nn.parallel.DistributedDataParallel(
            head, device_ids=[local_rank], output_device=local_rank
        )
    elif torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder)
        head = torch.nn.DataParallel(head)

    loss_fn = SupConBinaryLoss(
        temperature=cfg.temperature,
        similarity=cfg.supcon_similarity,
        uniformity_weight=cfg.uniformity_weight,
        uniformity_t=cfg.uniformity_t,
    )

    params = [{"params": head.parameters(), "lr": cfg.head_lr}]
    if cfg.finetune_encoder:
        params.append({"params": encoder.parameters(), "lr": cfg.enc_lr})
    optim = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)

    best, best_path = float("inf"), None
    epochs_no_improve = 0
    epoch_times = []
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        if hasattr(dev_loader.batch_sampler, "set_epoch"):
            dev_loader.batch_sampler.set_epoch(epoch)

        train_loss, alpha = train_one_epoch(
            encoder, head, loss_fn, train_loader, optim, device, epoch, cfg
        )
        dev_loss = evaluate(encoder, head, loss_fn, dev_loader, device, cfg)
        print(
            f"[epoch {epoch:03d}] alpha={alpha:.2f} | "
            f"train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f}"
        )
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        remaining = max(0, cfg.epochs - epoch)
        eta_min = (avg_epoch * remaining) / 60.0
        print(f"[epoch {epoch:03d}] time={epoch_time/60.0:.2f} min | ETA={eta_min:.2f} min")

        if dev_loss < best:
            best = dev_loss
            epochs_no_improve = 0
            if rank == 0:
                best_path = os.path.join(cfg.save_dir, f"{cfg.run_tag}_stage1_head_best.pt")
                head_to_save = head.module if hasattr(head, "module") else head
                encoder_to_save = encoder.module if hasattr(encoder, "module") else encoder
                ckpt = {
                    "epoch": epoch,
                    "compression_state_dict": head_to_save.state_dict(),
                    "train_loss": train_loss,
                    "dev_loss": dev_loss,
                    "config": ckpt_config(cfg),
                }
                if cfg.finetune_encoder:
                    ckpt["encoder_state_dict"] = encoder_to_save.state_dict()
                torch.save(ckpt, best_path)
                print(f"✓ Saved best -> {best_path} (dev={best:.4f})")
        else:
            epochs_no_improve += 1

        if cfg.patience and epochs_no_improve >= cfg.patience:
            if rank == 0:
                print(
                    f"Early stopping at epoch {epoch:03d} "
                    f"(no improvement for {cfg.patience} epochs)."
                )
            break


if __name__ == "__main__":
    main()
