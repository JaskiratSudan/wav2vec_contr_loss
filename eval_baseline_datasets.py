import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from data_loader import (
    ASVspoof2019Dataset,
    ASVspoof5Dataset,
    InTheWildDataset,
    FamousFiguresDataset,
    FakeXposeDataset,
    MLAADMailabsDataset,
)
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from evaluation import calculate_EER


DEFAULT_ASV19_EVAL_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac"
DEFAULT_ASV19_EVAL_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_eval_protocol_with_speaker.txt"

DEFAULT_ASV5_EVAL_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_eval/flac"
DEFAULT_ASV5_EVAL_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5.eval.track_1.tsv"

DEFAULT_ITW_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild"
DEFAULT_ITW_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv"

DEFAULT_FF_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/protocol.txt"
DEFAULT_FF_ROOT = ""

DEFAULT_FAKEXPOSE_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/fakexpose"
DEFAULT_MLAAD_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/multilingual"
DEFAULT_MLAAD_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/multilingual/protocol_MLAAD_MAILabs_total_balanced.txt"


def safe_load(path: str, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_state_dict_flexible(model: torch.nn.Module, state_dict: dict) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        cleaned = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(cleaned, strict=True)


class End2EndBCEModel(nn.Module):
    """
    encoder: (B,T) -> (B,K,F,T)
    compression: (B,K,F,T) -> (B,H,T)
    mean-pool time: (B,H,T) -> (B,H)
    classifier: (B,H) -> (B,) logits
    """
    def __init__(self, encoder: nn.Module, compression: nn.Module, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.compression = compression
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor):
        with torch.no_grad():
            hs = self.encoder(waveforms, attention_mask=attention_mask)  # (B,K,F,T)
        seq = self.compression(hs)                                       # (B,H,T)
        emb = seq.mean(dim=-1)                                           # (B,H)
        logits = self.classifier(emb).squeeze(-1)                        # (B,)
        return logits


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def pad_collate_fn_generic(batch):
    waveforms = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    sources = []
    utt_ids = []
    for item in batch:
        if len(item) >= 4:
            sources.append(item[-2])
            utt_ids.append(item[-1])
        elif len(item) == 3:
            sources.append("NA")
            utt_ids.append(item[2])
        else:
            sources.append("NA")
            utt_ids.append("unknown")
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(
        waveforms, batch_first=True, padding_value=0.0
    )
    labels = torch.stack(labels)
    return padded_waveforms, labels, None, sources, utt_ids


@torch.no_grad()
def score_and_write(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: str,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.eval()

    with open(out_path, "w") as f:
        for batch in loader:
            waveforms = batch[0].to(device)
            labels = batch[1].to(device)
            sources = batch[3]
            utt_ids = batch[4]

            attn = (waveforms != 0.0).long()
            logits = model(waveforms, attn)
            scores = logits.detach().cpu().numpy()

            labels_np = labels.detach().cpu().numpy().astype(int)
            for i in range(len(scores)):
                utt_id = str(utt_ids[i]).strip().replace(" ", "_")
                source = str(sources[i]).strip().replace(" ", "_")
                key = "bonafide" if labels_np[i] == 1 else "spoof"
                f.write(f"{utt_id} {source} {key} {scores[i]:.6f}\n")

    print(f"[OK] Wrote: {out_path}")


def _merge_ranked_scores(out_path: str, world_size: int) -> None:
    with open(out_path, "w") as fout:
        for rank in range(world_size):
            part = f"{out_path}.rank{rank}"
            if not os.path.isfile(part):
                continue
            with open(part, "r") as fin:
                shutil.copyfileobj(fin, fout)
            os.remove(part)


def resolve_ckpt(model_name: str, ckpt: str, ckpt_root: str):
    run_tag = model_name.replace("/", "__")
    if not ckpt:
        ckpt = os.path.join(ckpt_root, run_tag, f"{run_tag}_baseline_bce_best.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Baseline checkpoint not found: {ckpt}")
    return ckpt


def dataset_specs(args):
    return {
        "asv19": {
            "cls": ASVspoof2019Dataset,
            "kwargs": {
                "root_dir": args.asv19_root,
                "protocol_file": args.asv19_protocol,
                "subset": args.subset,
                "num_samples": args.num_samples,
                "max_duration_seconds": args.max_duration_seconds,
                "target_sample_rate": args.target_sample_rate,
            },
            "score_rel": os.path.join("baseline", args.model_name, "score_cm_eval.txt"),
        },
        "asv5": {
            "cls": ASVspoof5Dataset,
            "kwargs": {
                "root_dir": args.asv5_root,
                "protocol_file": args.asv5_protocol,
                "subset": args.subset,
                "num_samples": args.num_samples,
                "max_duration_seconds": args.max_duration_seconds,
                "target_sample_rate": args.target_sample_rate,
            },
            "score_rel": os.path.join("baseline", args.model_name, "score_cm_eval_asv5.txt"),
        },
        "itw": {
            "cls": InTheWildDataset,
            "kwargs": {
                "root_dir": args.itw_root,
                "protocol_file": args.itw_protocol,
                "subset": args.subset,
                "num_samples": args.num_samples,
                "max_duration_seconds": args.max_duration_seconds,
            },
            "score_rel": os.path.join("baseline", args.model_name, "score_cm_itw.txt"),
        },
        "famous_figures": {
            "cls": FamousFiguresDataset,
            "kwargs": {
                "protocol_file": args.ff_protocol,
                "root_dir": args.ff_root,
                "subset": args.subset,
                "num_samples": args.num_samples,
                "max_duration_seconds": args.max_duration_seconds,
                "target_sample_rate": args.target_sample_rate,
                "include_speakers": ff_speakers if ff_speakers else None,
                "return_audio_name": True,
            },
            "score_rel": os.path.join("baseline", args.model_name, "score_cm_ff.txt"),
        },
        "fakexpose": {
            "cls": FakeXposeDataset,
            "kwargs": {
                "root_dir": args.fakexpose_root,
                "subset": args.subset,
                "num_samples": args.num_samples,
                "max_duration_seconds": args.max_duration_seconds,
                "target_sample_rate": args.target_sample_rate,
            },
            "score_rel": os.path.join("baseline", args.model_name, "score_cm_fakexpose.txt"),
        },
        "mlaad": {
            "cls": MLAADMailabsDataset,
            "kwargs": {
                "root_dir": args.mlaad_root,
                "protocol_file": args.mlaad_protocol,
                "subset": "all",
                "num_samples": args.num_samples,
                "max_duration_seconds": args.max_duration_seconds,
                "target_sample_rate": args.target_sample_rate,
            },
            "score_rel": os.path.join("baseline", args.model_name, "score_cm_mlaad.txt"),
        },
    }


def _missing_inputs(kwargs):
    missing = []
    proto = kwargs.get("protocol_file")
    if proto and not os.path.isfile(proto):
        missing.append(proto)
    root = kwargs.get("root_dir")
    if root and not os.path.isdir(root):
        missing.append(root)
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--ckpt_root", type=str, default="checkpoints_baseline/bce")
    ap.add_argument("--scores_dir", type=str, default="scores")
    ap.add_argument("--datasets", type=str, default="asv19,asv5,itw,famous_figures,fakexpose,mlaad")
    ap.add_argument("--subset", type=str, default="all", choices=["all", "bonafide", "spoof"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_duration_seconds", type=int, default=5)
    ap.add_argument("--target_sample_rate", type=int, default=16000)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--print_eer", action="store_true")
    ap.add_argument("--force_rescore", action="store_true")

    ap.add_argument("--asv19_root", type=str, default=DEFAULT_ASV19_EVAL_ROOT)
    ap.add_argument("--asv19_protocol", type=str, default=DEFAULT_ASV19_EVAL_PROTOCOL)
    ap.add_argument("--asv5_root", type=str, default=DEFAULT_ASV5_EVAL_ROOT)
    ap.add_argument("--asv5_protocol", type=str, default=DEFAULT_ASV5_EVAL_PROTOCOL)
    ap.add_argument("--itw_root", type=str, default=DEFAULT_ITW_ROOT)
    ap.add_argument("--itw_protocol", type=str, default=DEFAULT_ITW_PROTOCOL)
    ap.add_argument("--ff_protocol", type=str, default=DEFAULT_FF_PROTOCOL)
    ap.add_argument("--ff_root", type=str, default=DEFAULT_FF_ROOT)
    ap.add_argument(
        "--ff_speakers",
        type=str,
        default="",
        help="Comma-separated list of FamousFigures speaker names to include (optional).",
    )
    ap.add_argument("--fakexpose_root", type=str, default=DEFAULT_FAKEXPOSE_ROOT)
    ap.add_argument("--mlaad_root", type=str, default=DEFAULT_MLAAD_ROOT)
    ap.add_argument("--mlaad_protocol", type=str, default=DEFAULT_MLAAD_PROTOCOL)
    args = ap.parse_args()

    ff_speakers = [s.strip() for s in args.ff_speakers.split(",") if s.strip()]

    is_distributed, rank, world_size, local_rank = setup_distributed()
    if torch.cuda.is_available():
        if is_distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    ckpt_path = resolve_ckpt(args.model_name, args.ckpt, args.ckpt_root)
    ckpt = safe_load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})

    model_name = cfg.get("MODEL_NAME", args.model_name)
    input_dim = cfg.get("INPUT_DIM", 1024)
    hidden_dim = cfg.get("HIDDEN_DIM", 256)
    dropout = cfg.get("DROPOUT", 0.1)

    encoder = Wav2Vec2Encoder(model_name=model_name, freeze_encoder=True).to(device)
    head = CompressionModule(input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout).to(device)
    model = End2EndBCEModel(encoder=encoder, compression=head, hidden_dim=hidden_dim).to(device)

    sd = ckpt.get("model_state_dict", ckpt)
    load_state_dict_flexible(model, sd)
    model.eval()

    if not is_distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    spec_map = dataset_specs(args)
    dataset_list = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]

    for name in dataset_list:
        if name not in spec_map:
            print(f"[WARN] Unknown dataset '{name}', skipping.")
            continue
        spec = spec_map[name]
        missing = _missing_inputs(spec["kwargs"])
        if missing:
            print(f"[WARN] Missing inputs for '{name}': {missing}. Skipping.")
            continue

        ds = spec["cls"](**spec["kwargs"])
        if is_distributed:
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=pad_collate_fn_generic,
            )
        else:
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=pad_collate_fn_generic,
            )

        score_path = os.path.join(args.scores_dir, spec["score_rel"])
        skip = False
        if not args.force_rescore and os.path.isfile(score_path):
            skip = True

        if is_distributed:
            flag = torch.tensor(int(skip), device=device)
            dist.broadcast(flag, src=0)
            skip = bool(flag.item())

        if skip:
            if rank == 0:
                print(f"[INFO] Found existing score file for '{name}': {score_path}")
        else:
            if rank == 0:
                print(f"[INFO] Scoring dataset '{name}' -> {score_path}")
            out_path = score_path
            if is_distributed:
                out_path = f"{score_path}.rank{rank}"
            score_and_write(model, loader, device, out_path)

        if is_distributed:
            dist.barrier()
            if rank == 0 and not skip:
                _merge_ranked_scores(score_path, world_size)
            dist.barrier()

        if args.print_eer and rank == 0:
            if os.path.isfile(score_path):
                eer = calculate_EER(score_path)
                print(f"EER ({name}): {eer}")
            else:
                print(f"[WARN] Score file missing for '{name}', skipping EER.")


if __name__ == "__main__":
    main()
