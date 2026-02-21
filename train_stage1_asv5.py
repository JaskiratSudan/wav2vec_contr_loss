# train_stage1_asv5.py
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from data_loader import pad_collate_fn_speaker_source_multiclass
from data_loader import ASVspoof5Dataset, pad_collate_fn_speaker_source_multiclass
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
    EmbeddingQueue,   
)
from data_loader import ASVspoof2019Dataset, MLAADMailabsDataset
import random

def stratified_subset(data, n_total, seed):
    """Return a fixed-size subset with ~50/50 bonafide/spoof."""
    rng = random.Random(seed)
    pos = [row for row in data if int(row[1]) == 1]  # bonafide
    neg = [row for row in data if int(row[1]) == 0]  # spoof
    rng.shuffle(pos); rng.shuffle(neg)

    n_pos = min(len(pos), n_total // 2)
    n_neg = min(len(neg), n_total - n_pos)
    subset = pos[:n_pos] + neg[:n_neg]
    rng.shuffle(subset)
    return subset

def _env_or_default(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val else default


def asv5_collate(batch):
    waveforms, bin_labels, *_ = pad_collate_fn_speaker_source_multiclass(batch)
    attn = (waveforms != 0.0).long()
    return waveforms, attn, bin_labels


def main():
    cfg = build_config()
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

    train_root = _env_or_default("ASV5_TRAIN_ROOT", cfg.train_root)
    train_protocol = _env_or_default("ASV5_TRAIN_PROTOCOL", cfg.train_protocol)
    dev_root = _env_or_default("ASV5_DEV_ROOT", cfg.dev_root)
    dev_protocol = _env_or_default("ASV5_DEV_PROTOCOL", cfg.dev_protocol)

    train_ds = ASVspoof5Dataset(
        root_dir=train_root,
        protocol_file=train_protocol,
        subset="all",
        num_samples=cfg.num_samples,
        sample_seed=cfg.seed,
        target_sample_rate=cfg.target_sample_rate,
        max_duration_seconds=cfg.max_duration_seconds,
    )

    def _parse_list_env(name: str, default: str):
        s = os.environ.get(name, default)
        return [x.strip().lower() for x in s.split(",") if x.strip()]

    train_sources = set(_parse_list_env("TRAIN_DATASETS", "asv5"))

    # Optional: tag ASV5 speaker IDs so speakers never collide across datasets
    train_ds.data = [
        (p, b, m, f"asv5_{spk}", name)
        for (p, b, m, spk, name) in train_ds.data
    ]

    if "asv19" in train_sources:
        asv19_root = _env_or_default("ASV19_TRAIN_ROOT", "")
        asv19_protocol = _env_or_default("ASV19_TRAIN_PROTOCOL", "")
        if not asv19_root or not asv19_protocol:
            raise ValueError("ASV19 requested in TRAIN_DATASETS but ASV19_TRAIN_ROOT/ASV19_TRAIN_PROTOCOL not set.")

        asv19_ds = ASVspoof2019Dataset(
            root_dir=asv19_root,
            protocol_file=asv19_protocol,
            subset="all",
            num_samples=cfg.num_samples,
            target_sample_rate=cfg.target_sample_rate,
            max_duration_seconds=cfg.max_duration_seconds,
        )

        # Append rows in ASV5 row format (5 fields)
        train_ds.data.extend([
            (p, b, m, f"asv19_{spk}", name)
            for (p, b, m, spk, name) in asv19_ds.data
        ])

    if "mlaad" in train_sources:
        mlaad_root = _env_or_default("MLAAD_ROOT", "")
        mlaad_protocol = _env_or_default("MLAAD_PROTOCOL", "")
        if not mlaad_root or not mlaad_protocol:
            raise ValueError("MLAAD requested in TRAIN_DATASETS but MLAAD_ROOT/MLAAD_PROTOCOL not set.")

        mlaad_ds = MLAADMailabsDataset(
            root_dir=mlaad_root,
            protocol_file=mlaad_protocol,
            subset="all",
            num_samples=cfg.num_samples,
            target_sample_rate=cfg.target_sample_rate,
            max_duration_seconds=cfg.max_duration_seconds,
        )

        # MLAAD rows are (path, bin, multi, audio_name) → convert to 5-tuple
        train_ds.data.extend([
            (p, b, m, f"mlaad_{os.path.splitext(str(name))[0]}", name)
            for (p, b, m, name) in mlaad_ds.data
        ])

    if rank == 0:
        bonafide_count = sum(1 for item in train_ds.data if item[1] == 1)
        spoof_count = sum(1 for item in train_ds.data if item[1] == 0)
        print(f"[INFO] POOLED TRAIN_DATASETS={sorted(train_sources)}")
        print(f"[INFO] Pooled train samples: bonafide={bonafide_count}, spoof={spoof_count}, total={len(train_ds.data)}")
        print(f"[INFO] Train samples loaded: bonafide={bonafide_count}, spoof={spoof_count}")

    # --- Dev ASV5 ---
    dev_ds_asv5 = ASVspoof5Dataset(
        root_dir=dev_root,
        protocol_file=dev_protocol,
        subset="all",
        num_samples=cfg.num_samples,
        sample_seed=cfg.seed + 1,
        target_sample_rate=cfg.target_sample_rate,
        max_duration_seconds=cfg.max_duration_seconds,
    )

    # --- Dev ASV19 ---
    asv19_dev_root = _env_or_default("ASV19_DEV_ROOT", "")
    asv19_dev_protocol = _env_or_default("ASV19_DEV_PROTOCOL", "")
    if not asv19_dev_root or not asv19_dev_protocol:
        raise ValueError("Need ASV19_DEV_ROOT and ASV19_DEV_PROTOCOL to validate on ASV19 dev.")

    dev_ds_asv19 = ASVspoof2019Dataset(
        root_dir=asv19_dev_root,
        protocol_file=asv19_dev_protocol,
        subset="all",
        num_samples=cfg.num_samples,
        target_sample_rate=cfg.target_sample_rate,
        max_duration_seconds=cfg.max_duration_seconds,
    )

    DEV_PER_DATASET = int(os.environ.get("DEV_PER_DATASET", "4000"))  # total per dataset
    dev_ds_asv5.data  = stratified_subset(dev_ds_asv5.data,  DEV_PER_DATASET, seed=cfg.seed + 101)
    dev_ds_asv19.data = stratified_subset(dev_ds_asv19.data, DEV_PER_DATASET, seed=cfg.seed + 202)

    if rank == 0:
        def _counts(ds):
            pos = sum(1 for r in ds.data if int(r[1]) == 1)
            neg = sum(1 for r in ds.data if int(r[1]) == 0)
            return pos, neg, len(ds.data)
        p,n,t = _counts(dev_ds_asv5);  print(f"[DEV SUBSET] ASV5  bonafide={p} spoof={n} total={t}")
        p,n,t = _counts(dev_ds_asv19); print(f"[DEV SUBSET] ASV19 bonafide={p} spoof={n} total={t}")

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
    # dev_sampler = BalancedBatchSampler(
    #     dev_ds, batch_size_per_gpu, seed=cfg.seed + 1, rank=rank, world_size=world_size
    # )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=asv5_collate,
    )

    if rank == 0:
        dev_loader_asv5 = DataLoader(dev_ds_asv5, batch_size=cfg.batch_size, shuffle=False,
                                    num_workers=cfg.num_workers, pin_memory=True, collate_fn=asv5_collate)
        dev_loader_asv19 = DataLoader(dev_ds_asv19, batch_size=cfg.batch_size, shuffle=False,
                                    num_workers=cfg.num_workers, pin_memory=True, collate_fn=asv5_collate)
    else:
        dev_loader_asv5 = dev_loader_asv19 = None

    # dev_loader = DataLoader(
    #     dev_ds,
    #     batch_sampler=dev_sampler,
    #     num_workers=cfg.num_workers,
    #     pin_memory=True,
    #     collate_fn=asv5_collate,
    # )

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

    use_queue = bool(int(os.environ.get("USE_QUEUE", "0")))
    queue_size = int(os.environ.get("QUEUE_SIZE", "0")) if use_queue else 0
    queue_start_epoch = int(os.environ.get("QUEUE_START_EPOCH", "1"))

    queue = None
    if use_queue and queue_size > 0 and queue_start_epoch <= 1:
        queue = EmbeddingQueue(queue_size, cfg.hidden_dim, device)
        if rank == 0:
            print(f"[INFO] Queue size={queue_size} (per-rank) | starts at epoch {queue_start_epoch}")
    elif use_queue and rank == 0:
        if queue_size <= 0:
            print("[WARN] USE_QUEUE=1 but QUEUE_SIZE <= 0; queue disabled.")
        else:
            print(f"[INFO] Queue will start at epoch {queue_start_epoch} (warm-start without queue)")
            print(f"[INFO] Queue size={queue_size} (per-rank)")

    params = [{"params": head.parameters(), "lr": cfg.head_lr}]
    if cfg.finetune_encoder:
        params.append({"params": encoder.parameters(), "lr": cfg.enc_lr})
    optim = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)

    best, best_path = float("inf"), None
    epochs_no_improve = 0
    for epoch in range(1, cfg.epochs + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        if use_queue and queue is None and queue_size > 0 and epoch >= queue_start_epoch:
            queue = EmbeddingQueue(queue_size, cfg.hidden_dim, device)
            if rank == 0:
                print(f"[INFO] Enabling queue at epoch {epoch} | size={queue_size} (per-rank)")

        train_loss, alpha = train_one_epoch(
            encoder, head, loss_fn, train_loader, optim, device, epoch, cfg, queue=queue
        )

        # --- Validation (rank 0 only) ---
        if rank == 0:
            dev_loss_asv5 = evaluate(encoder, head, loss_fn, dev_loader_asv5, device, cfg)
            dev_loss_asv19 = evaluate(encoder, head, loss_fn, dev_loader_asv19, device, cfg)
            dev_loss = 0.5 * (dev_loss_asv5 + dev_loss_asv19)
            print(f"[dev] asv5={dev_loss_asv5:.4f} asv19={dev_loss_asv19:.4f} avg={dev_loss:.4f}")
        else:
            dev_loss = 0.0  # placeholder

        # Broadcast dev_loss to all ranks so everyone agrees
        if is_distributed:
            dev_t = torch.tensor(dev_loss, device=device, dtype=torch.float32)
            dist.broadcast(dev_t, src=0)
            dev_loss = float(dev_t.item())

        # --- Checkpointing / early stopping (rank 0 decides) ---
        if rank == 0:
            if dev_loss < best:
                best = dev_loss
                epochs_no_improve = 0

                best_path = os.path.join(cfg.save_dir, f"{cfg.model_id}_stage1_head_best.pt")
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

            stop_training = (cfg.patience and epochs_no_improve >= cfg.patience)
        else:
            stop_training = False

        # Broadcast stop flag so all ranks exit together
        if is_distributed:
            flag = torch.tensor(int(stop_training), device=device)
            dist.broadcast(flag, src=0)
            stop_training = bool(flag.item())

        if stop_training:
            if rank == 0:
                print(f"Early stopping at epoch {epoch:03d} (no improvement for {cfg.patience} epochs).")
            break


if __name__ == "__main__":
    main()
