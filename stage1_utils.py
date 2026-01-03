import os
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
import torch.distributed as dist

from RawBoost import LnL_convolutive_noise, ISD_additive_noise, SSI_additive_noise


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size: int, seed: int = 0, rank: int = 0, world_size: int = 1):
        assert batch_size % 2 == 0
        self.batch_size = batch_size
        self.data = dataset.data
        self.real = [i for i, it in enumerate(self.data) if it[1] == 1]
        self.fake = [i for i, it in enumerate(self.data) if it[1] == 0]
        self.per_class = batch_size // 2
        self.num_batches = min(len(self.real)//self.per_class, len(self.fake)//self.per_class)
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size

    def __len__(self):
        return (self.num_batches - self.rank + self.world_size - 1) // self.world_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(self.real); rng.shuffle(self.fake)
        r = self.real[: self.num_batches*self.per_class]
        f = self.fake[: self.num_batches*self.per_class]
        for b in range(self.num_batches):
            idx = r[b*self.per_class:(b+1)*self.per_class] + f[b*self.per_class:(b+1)*self.per_class]
            rng.shuffle(idx)
            if (b % self.world_size) == self.rank:
                yield idx

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def apply_rawboost_batch(x: torch.Tensor, cfg) -> torch.Tensor:
    if not cfg.use_rawboost:
        return x
    device = x.device
    pad_mask = (x != 0.0)
    a = x.detach().cpu().numpy()
    for i in range(a.shape[0]):
        if random.random() < cfg.rawboost_prob:
            y = LnL_convolutive_noise(
                a[i], N_f=5, nBands=5,
                minF=20.0,  maxF=8000.0,
                minBW=100.0, maxBW=1000.0,
                minCoeff=10, maxCoeff=100,
                minG=0.0, maxG=0.0,
                minBiasLinNonLin=5.0, maxBiasLinNonLin=20.0,
                fs=cfg.target_sample_rate,
            )
            if random.random() < 0.5:
                y = SSI_additive_noise(
                    y, SNRmin=10.0, SNRmax=40.0, nBands=5,
                    minF=20.0, maxF=8000.0, minBW=100.0, maxBW=1000.0,
                    minCoeff=10, maxCoeff=100, minG=0.0, maxG=0.0,
                    fs=cfg.target_sample_rate,
                )
            if random.random() < 0.5:
                y = ISD_additive_noise(y, P=10.0, g_sd=2.0)
            a[i] = y
    y = torch.from_numpy(a).to(device=device, dtype=x.dtype)
    return y * pad_mask.to(device=y.device, dtype=y.dtype)


def alpha_for_epoch(epoch: int, cfg) -> float:
    if epoch <= cfg.warmup_epochs:
        return 0.0
    t = min(1.0, (epoch - cfg.warmup_epochs) / max(1, cfg.alpha_ramp_epochs))
    return t * cfg.alpha_end


def _reduce_avg(total: float, steps: int, device: torch.device) -> float:
    avg = total / max(1, steps)
    if dist.is_initialized():
        total_t = torch.tensor(total, device=device)
        steps_t = torch.tensor(steps, device=device)
        dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(steps_t, op=dist.ReduceOp.SUM)
        avg = (total_t / steps_t.clamp_min(1)).item()
    return avg


def train_one_epoch(encoder, head, loss_fn, loader, optimizer, device, epoch, cfg):
    if cfg.finetune_encoder:
        encoder.train()
    else:
        encoder.eval()
    head.train()
    total, steps = 0.0, 0
    alpha = alpha_for_epoch(epoch, cfg)
    for waveforms, labels, *_ in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device).long()
        if cfg.use_rawboost:
            waveforms = apply_rawboost_batch(waveforms, cfg)
        attn = (waveforms != 0.0).long()

        if cfg.finetune_encoder:
            hs = encoder(waveforms, attention_mask=attn)
        else:
            with torch.no_grad():
                hs = encoder(waveforms, attention_mask=attn)
        seq = head(hs)
        z = F.normalize(seq.mean(dim=-1), p=2, dim=1)

        loss = loss_fn(z, labels, topk_neg=cfg.topk_neg, alpha=alpha)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()

        total += loss.item()
        steps += 1

    return _reduce_avg(total, steps, device), alpha


@torch.no_grad()
def evaluate(encoder, head, loss_fn, loader, device, cfg):
    encoder.eval()
    head.eval()
    total, steps = 0.0, 0
    for waveforms, labels, *_ in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device).long()
        attn = (waveforms != 0.0).long()
        hs = encoder(waveforms, attention_mask=attn)
        seq = head(hs)
        z = F.normalize(seq.mean(dim=-1), p=2, dim=1)
        loss = loss_fn(z, labels, topk_neg=cfg.topk_neg, alpha=0.0)
        total += loss.item()
        steps += 1
    return _reduce_avg(total, steps, device)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    else:
        return False, 0, 1, 0

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0
