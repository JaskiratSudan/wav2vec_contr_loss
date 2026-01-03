import os
import argparse
from types import SimpleNamespace

# =======================
#        DEFAULTS
# =======================
TRAIN_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac"
TRAIN_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt"
DEV_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac"
DEV_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt"

TARGET_SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 5

INPUT_DIM = 1024
HIDDEN_DIM = 256
DROPOUT = 0.1

EPOCHS = 100
BATCH_SIZE = 256
NUM_SAMPLES = None
HEAD_LR = 5e-3
ENC_LR = 1e-5
WEIGHT_DECAY = 3e-3
TEMPERATURE = 0.2
NUM_WORKERS = 4
SEED = 1337
SAVE_DIR = "/home/jsudan/wav2vec_contr_loss/checkpoints_stage1/supcon_geodesic_dist"
FINETUNE_ENCODER = False

UNIFORMITY_WEIGHT = 0.2
UNIFORMITY_T = 2.0
SUPCON_SIMILARITY = "cosine"

TOPK_NEG = 15
WARMUP_EPOCHS = 100
ALPHA_END = 1
ALPHA_RAMP_EPOCHS = 80

USE_RAWBOOST = True
RAWBOOST_PROB = 0.7


def build_config():
    model_name = "facebook/wav2vec2-xls-r-300m"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default=model_name,
        help="HF ID for Wav2Vec2, e.g. facebook/wav2vec2-large-960h or microsoft/wavlm-base-plus",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=SAVE_DIR,
        help="Base directory for checkpoints (run_tag subfolder will be created).",
    )
    parser.add_argument(
        "--supcon_similarity",
        type=str,
        default=SUPCON_SIMILARITY,
        choices=["cosine", "geodesic"],
        help="Similarity function for SupConBinaryLoss.",
    )
    parser.add_argument(
        "--uniformity_weight",
        type=float,
        default=UNIFORMITY_WEIGHT,
        help="Weight for uniformity regularizer (0 disables).",
    )
    parser.add_argument(
        "--uniformity_t",
        type=float,
        default=UNIFORMITY_T,
        help="Temperature t for uniformity regularizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size (must be even for BalancedBatchSampler).",
    )
    parser.add_argument(
        "--num_samples",
        type=str,
        default=str(NUM_SAMPLES),
        help="Subsample size for dataset (int) or None.",
    )
    parser.add_argument(
        "--head_lr",
        type=float,
        default=HEAD_LR,
        help="Learning rate for compression head.",
    )
    parser.add_argument(
        "--enc_lr",
        type=float,
        default=ENC_LR,
        help="Learning rate for encoder (used when finetuning).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=WEIGHT_DECAY,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="SupCon temperature.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_WORKERS,
        help="DataLoader num_workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--topk_neg",
        type=int,
        default=TOPK_NEG,
        help="Hardest negatives per anchor.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=WARMUP_EPOCHS,
        help="Warmup epochs before ramping alpha.",
    )
    parser.add_argument(
        "--alpha_end",
        type=float,
        default=ALPHA_END,
        help="Final alpha weight on mined loss.",
    )
    parser.add_argument(
        "--alpha_ramp_epochs",
        type=int,
        default=ALPHA_RAMP_EPOCHS,
        help="Epochs to ramp alpha after warmup.",
    )
    parser.add_argument(
        "--use_rawboost",
        type=int,
        default=int(USE_RAWBOOST),
        choices=[0, 1],
        help="Enable RawBoost (1) or disable (0).",
    )
    parser.add_argument(
        "--rawboost_prob",
        type=float,
        default=RAWBOOST_PROB,
        help="Probability of applying RawBoost per utterance.",
    )
    parser.add_argument(
        "--finetune_encoder",
        type=int,
        default=int(FINETUNE_ENCODER),
        choices=[0, 1],
        help="Enable encoder finetuning (1) or keep frozen (0).",
    )
    args = parser.parse_args()

    num_samples_arg = args.num_samples.strip().lower()
    if num_samples_arg in ("none", "null"):
        num_samples_arg = None
    else:
        num_samples_arg = int(num_samples_arg)

    run_tag = args.model_name.replace("/", "__")
    save_dir = os.path.join(args.save_dir, run_tag)

    return SimpleNamespace(
        model_name=args.model_name,
        run_tag=run_tag,
        save_dir=save_dir,
        train_root=TRAIN_ROOT,
        train_protocol=TRAIN_PROTOCOL,
        dev_root=DEV_ROOT,
        dev_protocol=DEV_PROTOCOL,
        target_sample_rate=TARGET_SAMPLE_RATE,
        max_duration_seconds=MAX_DURATION_SECONDS,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=num_samples_arg,
        head_lr=args.head_lr,
        enc_lr=args.enc_lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        num_workers=args.num_workers,
        seed=args.seed,
        uniformity_weight=args.uniformity_weight,
        uniformity_t=args.uniformity_t,
        supcon_similarity=args.supcon_similarity,
        topk_neg=args.topk_neg,
        warmup_epochs=args.warmup_epochs,
        alpha_end=args.alpha_end,
        alpha_ramp_epochs=args.alpha_ramp_epochs,
        use_rawboost=bool(args.use_rawboost),
        rawboost_prob=args.rawboost_prob,
        finetune_encoder=bool(args.finetune_encoder),
    )


def print_config(cfg, is_distributed=False, world_size=1, rank=0):
    if rank != 0:
        return
    print("=== CONFIG ===")
    print(f"MODEL_NAME={cfg.model_name}")
    print(f"SAVE_DIR={cfg.save_dir}")
    print(f"TRAIN_ROOT={cfg.train_root}")
    print(f"TRAIN_PROTOCOL={cfg.train_protocol}")
    print(f"DEV_ROOT={cfg.dev_root}")
    print(f"DEV_PROTOCOL={cfg.dev_protocol}")
    print(f"TARGET_SAMPLE_RATE={cfg.target_sample_rate}")
    print(f"MAX_DURATION_SECONDS={cfg.max_duration_seconds}")
    print(f"INPUT_DIM={cfg.input_dim}")
    print(f"HIDDEN_DIM={cfg.hidden_dim}")
    print(f"DROPOUT={cfg.dropout}")
    print(f"EPOCHS={cfg.epochs}")
    print(f"BATCH_SIZE={cfg.batch_size}")
    print(f"NUM_SAMPLES={cfg.num_samples}")
    print(f"HEAD_LR={cfg.head_lr}")
    print(f"ENC_LR={cfg.enc_lr}")
    print(f"WEIGHT_DECAY={cfg.weight_decay}")
    print(f"TEMPERATURE={cfg.temperature}")
    print(f"NUM_WORKERS={cfg.num_workers}")
    print(f"SEED={cfg.seed}")
    print(f"UNIFORMITY_WEIGHT={cfg.uniformity_weight}")
    print(f"UNIFORMITY_T={cfg.uniformity_t}")
    print(f"SUPCON_SIMILARITY={cfg.supcon_similarity}")
    print(f"TOPK_NEG={cfg.topk_neg}")
    print(f"WARMUP_EPOCHS={cfg.warmup_epochs}")
    print(f"ALPHA_END={cfg.alpha_end}")
    print(f"ALPHA_RAMP_EPOCHS={cfg.alpha_ramp_epochs}")
    print(f"USE_RAWBOOST={cfg.use_rawboost}")
    print(f"RAWBOOST_PROB={cfg.rawboost_prob}")
    print(f"FINETUNE_ENCODER={cfg.finetune_encoder}")
    print(f"DISTRIBUTED={is_distributed} | WORLD_SIZE={world_size} | RANK={rank}")
    print("=============")


def ckpt_config(cfg):
    return {
        "MODEL_NAME": cfg.model_name,
        "RUN_TAG": cfg.run_tag,
        "INPUT_DIM": cfg.input_dim,
        "HIDDEN_DIM": cfg.hidden_dim,
        "DROPOUT": cfg.dropout,
        "BATCH_SIZE": cfg.batch_size,
        "HEAD_LR": cfg.head_lr,
        "ENC_LR": cfg.enc_lr,
        "WEIGHT_DECAY": cfg.weight_decay,
        "TEMPERATURE": cfg.temperature,
        "TOPK_NEG": cfg.topk_neg,
        "WARMUP_EPOCHS": cfg.warmup_epochs,
        "ALPHA_END": cfg.alpha_end,
        "ALPHA_RAMP_EPOCHS": cfg.alpha_ramp_epochs,
        "USE_RAWBOOST": cfg.use_rawboost,
        "RAWBOOST_PROB": cfg.rawboost_prob,
        "UNIFORMITY_WEIGHT": cfg.uniformity_weight,
        "UNIFORMITY_T": cfg.uniformity_t,
        "SUPCON_SIMILARITY": cfg.supcon_similarity,
        "FINETUNE_ENCODER": cfg.finetune_encoder,
    }
