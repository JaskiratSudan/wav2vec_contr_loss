# train_stage1.py
import os

import torch
from torch.utils.data import DataLoader
from data_loader import ASVspoof2019Dataset, pad_collate_fn_speaker_source_multiclass
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from loss import SupConBinaryLoss
from stage1_config import build_config, print_config, ckpt_config
from stage1_utils import (
    set_seed,
    BalancedBatchSampler,
    train_one_epoch,
    evaluate,
)


def main():
    cfg = build_config()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    print_config(cfg, is_distributed=False, world_size=1, rank=0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device} | RawBoost={cfg.use_rawboost} (p={cfg.rawboost_prob})")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")

    train_ds = ASVspoof2019Dataset(
        root_dir=cfg.train_root,
        protocol_file=cfg.train_protocol,
        subset="all",
        max_duration_seconds=cfg.max_duration_seconds,
        target_sample_rate=cfg.target_sample_rate,
        num_samples=cfg.num_samples,
    )
    dev_ds = ASVspoof2019Dataset(
        root_dir=cfg.dev_root,
        protocol_file=cfg.dev_protocol,
        subset="all",
        max_duration_seconds=cfg.max_duration_seconds,
        target_sample_rate=cfg.target_sample_rate,
        num_samples=cfg.num_samples,
    )

    train_sampler = BalancedBatchSampler(
        train_ds, cfg.batch_size, seed=cfg.seed
    )
    dev_sampler = BalancedBatchSampler(
        dev_ds, cfg.batch_size, seed=cfg.seed + 1
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn_speaker_source_multiclass,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_sampler=dev_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn_speaker_source_multiclass,
    )

    encoder = Wav2Vec2Encoder(
        model_name=cfg.model_name,
        freeze_encoder=not cfg.finetune_encoder,
    ).to(device)
    if hasattr(encoder, "model") and hasattr(encoder.model, "config"):
        if hasattr(encoder.model.config, "layerdrop"):
            encoder.model.config.layerdrop = 0.0
    head = CompressionModule(cfg.input_dim, cfg.hidden_dim, cfg.dropout).to(device)

    if torch.cuda.device_count() > 1:
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
    for epoch in range(1, cfg.epochs + 1):
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

        if dev_loss < best:
            best = dev_loss
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

    if best_path:
        print(f"Best checkpoint: {best_path} (dev={best:.4f})")


if __name__ == "__main__":
    main()
