"""
extract_head_embeddings.py

Reads per-sample XLSR feature files from a directory (written by
extract_xlsr_features.py), passes them through the trained projection
head (CompressionModule), and writes a single .pt cache file containing
all 256-dim l2-normalized embeddings for that split.

Input directory format (one file per sample):
    {xlsr_dir}/{AudioName}.pt
    → {"features": Tensor(1024, T) fp16, "label": int,
       "speaker": str, "source": str}

Output .pt format (consumed by train_stage2_cached.py and evaluate_stage2.py):
    {
        "embeddings":  FloatTensor [N, 256],  # l2-normalized
        "labels":      LongTensor  [N],       # 1=bonafide, 0=spoof
        "sources":     List[str]   length N,
        "audio_names": List[str]   length N,
    }

Usage
-----
    python extract_head_embeddings.py \
        --xlsr_dir   /path/to/cache/train \
        --ckpt       /path/to/stage1_cached_head_best.pt \
        --out_path   /path/to/cache/head_emb_train.pt \
        --hidden_dim 256 \
        --dropout    0.0
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from compression_module import CompressionModule


# ---------------------------------------------------------------------------
# Dataset — streams one shard at a time, only ~10MB in RAM at any point
# ---------------------------------------------------------------------------

class ShardedXLSRDataset(torch.utils.data.IterableDataset):
    """Streams shard files one at a time. No full dataset in RAM."""
    def __init__(self, cache_path: str):
        manifest          = torch.load(cache_path, map_location="cpu")
        self.shard_paths  = manifest["shard_paths"]
        self.n_samples    = manifest["n_samples"]
        print(f"  {self.n_samples} samples across {len(self.shard_paths)} shards")

    def __iter__(self):
        for sp in self.shard_paths:
            shard = torch.load(sp, map_location="cpu")
            feats        = shard["features"]
            labels       = shard["labels"].tolist()
            sources      = shard["sources"]
            audio_names  = shard["audio_names"]
            for i in range(len(labels)):
                feat = feats[i].float()
                if feat.shape[0] != 1024:
                    feat = feat.T
                yield feat, labels[i], sources[i], audio_names[i]

    def __len__(self):
        return self.n_samples


def collate_fn(batch):
    """Pad variable-length (1024, T_i) tensors along T dimension."""
    features, labels, sources, names = zip(*batch)
    max_t  = max(f.shape[1] for f in features)
    padded = torch.zeros(len(features), 1024, max_t)
    for i, f in enumerate(features):
        padded[i, :, :f.shape[1]] = f
    return padded, labels, sources, names



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsr_cache", required=True,
                        help="Path to consolidated xlsr_{split}.pt file")
    parser.add_argument("--ckpt",       required=True,
                        help="Path to the best Stage 1 head checkpoint .pt")
    parser.add_argument("--out_path",   required=True,
                        help="Output .pt file to write the embedding cache to")
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--dropout",    type=float, default=0.0)
    parser.add_argument("--batch_size", type=int,   default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")
    print(f"XLSR cache : {args.xlsr_cache}")
    print(f"Checkpoint : {args.ckpt}")
    print(f"Out path   : {args.out_path}\n")

    # ------------------------------------------------------------------
    # Load projection head
    # ------------------------------------------------------------------
    head = CompressionModule(input_dim=1024, hidden_dim=args.hidden_dim,
                             dropout_rate=args.dropout).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    head.load_state_dict(ckpt["compression_state_dict"])
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    print(f"Loaded head from epoch {ckpt.get('epoch','?')}  "
          f"(dev_loss={ckpt.get('dev_loss', float('nan')):.4f})")

    # ------------------------------------------------------------------
    # Dataset + loader
    # ------------------------------------------------------------------
    ds     = ShardedXLSRDataset(args.xlsr_cache)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True,  # IterableDataset requires num_workers=0
                        collate_fn=collate_fn)

    # ------------------------------------------------------------------
    # Extract embeddings
    # ------------------------------------------------------------------
    all_embeddings  = []
    all_labels      = []
    all_sources     = []
    all_names       = []

    t0 = time.time()
    with torch.no_grad():
        for features, labels, sources, names in loader:
            # features: (B, 1024, T) → permute to (B, T, 1024) for head
            features = features.to(device)              # (B, 1024, T)
            out      = head(features.unsqueeze(1))       # (B, 1, 1024, T) → (B, 256, T)
            embs     = out.mean(dim=-1)                  # (B, 256)
            embs     = F.normalize(embs, dim=-1).cpu()

            all_embeddings.append(embs)
            all_labels.extend(list(labels))
            all_sources.extend(list(sources))
            all_names.extend(list(names))

    embeddings_tensor = torch.cat(all_embeddings, dim=0)   # (N, 256)
    labels_tensor     = torch.tensor(all_labels, dtype=torch.long)

    elapsed = time.time() - t0
    N = len(all_labels)
    bon = int((labels_tensor == 1).sum())
    spo = int((labels_tensor == 0).sum())
    print(f"\nExtracted {N} embeddings in {elapsed:.1f}s")
    print(f"  bonafide={bon}  spoof={spo}")
    print(f"  embeddings shape: {embeddings_tensor.shape}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "embeddings":  embeddings_tensor,   # (N, 256) float32, l2-normalized
            "labels":      labels_tensor,        # (N,) long
            "sources":     all_sources,          # List[str]
            "audio_names": all_names,            # List[str]
        },
        out_path,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved to: {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()