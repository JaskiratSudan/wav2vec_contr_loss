# train_stage1.py
import glob
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from data_loader import (
    ASVspoof2019Dataset,
    RAVDESSDataset,
    CommonVoiceDataset,
    pad_collate_fn_speaker_source_multiclass,
)
from asvspoof_windowed_loader import ASVspoof2019WindowedDataset
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


def main():
    cfg = build_config()
    is_distributed, rank, world_size, local_rank = setup_distributed()
    set_seed(cfg.seed + rank)
    os.makedirs(cfg.save_dir, exist_ok=True)

    cfg.bg_files = glob.glob(os.path.join(cfg.bg_noise_dir, "*.mp3"))

    print_config(cfg, is_distributed=is_distributed, world_size=world_size, rank=rank)

    if torch.cuda.is_available():
        if is_distributed:
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if rank == 0:
        print(f"Using device: {device} | RawBoost={cfg.use_rawboost} (p={cfg.rawboost_prob}) | BG_AUG={cfg.use_bg_aug} ({len(cfg.bg_files)} noise files)")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")

    # train_ds = ASVspoof2019Dataset(
    #     root_dir=cfg.train_root,
    #     protocol_file=cfg.train_protocol,
    #     subset="all",
    #     max_duration_seconds=cfg.max_duration_seconds,
    #     target_sample_rate=cfg.target_sample_rate,
    #     num_samples=cfg.num_samples,
    # )
    train_ds = ASVspoof2019WindowedDataset(
        root_dir=cfg.train_root,
        protocol_file=cfg.train_protocol,
        subset="all",
        num_samples=cfg.num_samples,
        target_sample_rate=cfg.target_sample_rate,
        window_seconds=cfg.max_duration_seconds,
        min_tail_seconds=1.0,
    )
    if cfg.use_ravdess:
        ravdess_ds = RAVDESSDataset(
            root_dir=cfg.ravdess_root,
            max_duration_seconds=cfg.max_duration_seconds,
            target_sample_rate=cfg.target_sample_rate,
            num_samples=cfg.num_samples,
        )
        for audio_path in ravdess_ds.audio_files:
            speaker_id = audio_path.parent.name
            train_ds.data.append((audio_path, 1, 0, speaker_id, audio_path.name))
        print(f"[INFO] Added RAVDESS samples: {len(ravdess_ds.audio_files)}")
    if cfg.use_commonvoice:
        commonvoice_ds = CommonVoiceDataset(
            root_dir=cfg.commonvoice_root,
            max_duration_seconds=cfg.max_duration_seconds,
            target_sample_rate=cfg.target_sample_rate,
            num_samples=cfg.num_samples,
        )
        for audio_path in commonvoice_ds.audio_files:
            speaker_id = audio_path.parent.name
            train_ds.data.append((audio_path, 1, 0, speaker_id, audio_path.name))
        print(f"[INFO] Added Common Voice samples: {len(commonvoice_ds.audio_files)}")

    bonafide_count = sum(1 for item in train_ds.data if item[1] == 1)
    spoof_count = sum(1 for item in train_ds.data if item[1] == 0)
    print(f"[INFO] Train samples loaded: bonafide={bonafide_count}, spoof={spoof_count}")
    # dev_ds = ASVspoof2019Dataset(
    #     root_dir=cfg.dev_root,
    #     protocol_file=cfg.dev_protocol,
    #     subset="all",
    #     max_duration_seconds=cfg.max_duration_seconds,
    #     target_sample_rate=cfg.target_sample_rate,
    #     num_samples=cfg.num_samples,
    # )
    dev_ds = ASVspoof2019WindowedDataset(
        root_dir=cfg.dev_root,
        protocol_file=cfg.dev_protocol,
        subset="all",
        num_samples=cfg.num_samples,
        target_sample_rate=cfg.target_sample_rate,
        window_seconds=cfg.max_duration_seconds,
        min_tail_seconds=1.0,
    )

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
    head = CompressionModule(cfg.input_dim, cfg.hidden_dim, cfg.dropout, use_bottleneck=cfg.use_bottleneck).to(device)

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
        sil_weight=cfg.sil_weight,
    )

    params = [{"params": head.parameters(), "lr": cfg.head_lr}]
    if cfg.finetune_encoder:
        params.append({"params": encoder.parameters(), "lr": cfg.enc_lr})
    optim = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)

    best, best_path = float("inf"), None
    epochs_no_improve = 0
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

    if rank == 0 and best_path:
        print(f"Best checkpoint: {best_path} (dev={best:.4f})")

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
