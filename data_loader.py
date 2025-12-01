# data_loader.py
# This version is updated to include a wrapper for applying augmentations.

import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
import soundfile
import librosa
import random

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.utils")

def pad_collate_fn(batch):
    """
    Pads audio waveforms in a batch to the length of the longest waveform.
    Uses only the first two elements of each item: (waveform, label).
    Any extra elements in the batch items are ignored.
    """
    waveforms = []
    labels = []
    for item in batch:
        waveform, label = item[0], item[1]  # ignore anything beyond (waveform, label)
        waveforms.append(waveform)
        labels.append(torch.as_tensor(label))

    padded_waveforms = torch.nn.utils.rnn.pad_sequence(
        waveforms, batch_first=True, padding_value=0.0
    )
    labels = torch.stack(labels)
    return padded_waveforms, labels

def pad_collate_fn_aug(batch):
    """
    Collate function for the DatasetWithAugmentation. It handles batches of
    (original_waveform, augmented_waveform, label) tuples.
    """
    originals, augmenteds, labels = zip(*batch)
    padded_originals = torch.nn.utils.rnn.pad_sequence(list(originals), batch_first=True, padding_value=0.0)
    padded_augmenteds = torch.nn.utils.rnn.pad_sequence(list(augmenteds), batch_first=True, padding_value=0.0)
    labels = torch.stack(list(labels))
    return padded_originals, padded_augmenteds, labels

    # Add this new function to data_loader.py

def pad_collate_fn_speaker(batch):
    """
    Pads audio waveforms and handles speaker IDs.
    Assumes each item in the batch is a (waveform, label, speaker_id) tuple.
    """
    # Unpack three items now
    waveforms, labels, speakers = zip(*batch)
    
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(list(waveforms), batch_first=True, padding_value=0.0)
    labels = torch.stack(list(labels))
    
    # Speakers are usually strings, so we return them as a tuple
    return padded_waveforms, labels, speakers

# ---------- New collate for speaker + source ----------
def pad_collate_fn_speaker_source(batch):
    """
    Pads audio waveforms and returns speaker + source strings.
    Assumes each item is (waveform, label, speaker, source).
    """
    waveforms, labels, speakers, sources = zip(*batch)
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(
        list(waveforms), batch_first=True, padding_value=0.0
    )
    labels = torch.stack(list(labels))
    # speakers/sources are strings; return as tuples (kept as-is for your encoder)
    return padded_waveforms, labels, speakers, sources

def pad_collate_fn_speaker_source_multiclass(batch):
    """
    Pads audio waveforms and returns:
      waveforms, binary_labels, multiclass_labels, speakers, sources
    Assumes each item is:
      (waveform, binary_label, multi_label, speaker, audio_name)
    """
    waveforms, bin_labels, attack_id, speakers, sources = zip(*batch)

    padded_waveforms = torch.nn.utils.rnn.pad_sequence(
        list(waveforms), batch_first=True, padding_value=0.0
    )
    bin_labels = torch.stack(list(bin_labels))
    attack_id = torch.stack(list(attack_id))

    return padded_waveforms, bin_labels, attack_id, speakers, sources

class BaseAudioDataset(Dataset):
    """
    A base class for audio datasets to handle common processing tasks.
    Tracks how many files were successfully decoded vs. failed.
    """
    loaded_count = 0
    failed_count = 0

    def __init__(self, target_sample_rate: int = 16000, max_duration_seconds: int = 5, **kwargs):
        self.target_sample_rate = target_sample_rate
        self.max_duration_seconds = max_duration_seconds

    def _process_audio(self, audio_path: Path) -> torch.Tensor:
        try:
            waveform, sample_rate = librosa.load(
                audio_path, sr=self.target_sample_rate, mono=True
            )
            waveform = torch.from_numpy(waveform).float()
            BaseAudioDataset.loaded_count += 1
        except Exception as e:
            tqdm.write(f"[WARNING] Corrupted file: {audio_path}. Error: {e}")
            BaseAudioDataset.failed_count += 1
            if self.max_duration_seconds is not None:
                return torch.zeros(self.max_duration_seconds * self.target_sample_rate)
            else:
                return torch.zeros(self.target_sample_rate)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        if self.max_duration_seconds is not None:
            target_len = self.max_duration_seconds * self.target_sample_rate
            current_len = waveform.shape[0]
            if current_len > target_len:
                waveform = waveform[:target_len]
            elif current_len < target_len:
                waveform = F.pad(waveform, (0, target_len - current_len))

        return waveform

    @classmethod
    def print_summary(cls):
        total = cls.loaded_count + cls.failed_count
        print(f"\n[DATASET SUMMARY] Loaded: {cls.loaded_count}, Failed: {cls.failed_count}, Total: {total}")

# ---------- New Dataset: FamousFigures ----------
class FamousFiguresDataset(BaseAudioDataset):
    """
    FamousFigures dataset using a protocol file with columns:
      AudioName, Speaker, Source, Label, AudioPath

    Returns: (waveform, label_tensor, speaker_str, source_str)
      - label: 1 for bonafide, 0 for spoof (case-insensitive; also normalizes 'bona-fide')
    """
    def __init__(
        self,
        protocol_file: str,
        root_dir: str = "",
        subset: str = "all",             # 'all' | 'bonafide' | 'spoof'
        include_speakers: list = None,   # optional allowlist of speakers
        include_sources: list = None,    # optional allowlist of sources
        **kwargs
    ):
        num_samples = kwargs.pop("num_samples", None)
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir) if root_dir else None

        # Load protocol; tabs or general whitespace both OK
        try:
            df = pd.read_csv(protocol_file, sep="\t")
        except Exception:
            df = pd.read_csv(protocol_file, sep=r"\s+", engine="python")

        # Normalize expected columns
        expected = {"AudioName", "Speaker", "Source", "Label", "AudioPath"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"Protocol is missing columns: {sorted(missing)}")

        # Normalize labels
        df["Label"] = df["Label"].astype(str).str.lower().str.replace("bona-fide", "bonafide")

        # Clean obviously malformed paths by trimming after first '.wav'
        def _clean_path(p):
            s = str(p)
            if ".wav" in s:
                s = s[: s.lower().find(".wav") + 4]
            return s

        df["AudioPath"] = df["AudioPath"].astype(str).map(_clean_path)

        # Make paths absolute if root_dir provided and path is relative
        if self.root_dir is not None:
            df["AudioPath"] = df["AudioPath"].apply(
                lambda p: str(Path(p)) if Path(p).is_absolute() else str(self.root_dir / p)
            )

        # Subset by label
        if subset == "bonafide":
            df = df[df["Label"] == "bonafide"]
        elif subset == "spoof":
            df = df[df["Label"] != "bonafide"]

        # Optional filters
        if include_speakers:
            keep = set(map(str, include_speakers))
            df = df[df["Speaker"].astype(str).isin(keep)]
        if include_sources:
            keep = set(map(str, include_sources))
            df = df[df["Source"].astype(str).isin(keep)]

        # Keep only rows whose files exist
        df["exists"] = df["AudioPath"].apply(lambda p: Path(p).exists())
        missing_n = int((~df["exists"]).sum())
        if missing_n > 0:
            print(f"[INFO] FamousFigures: filtered out {missing_n} missing audio files.")
        df = df[df["exists"]].copy()

        # Optional sampling (shuffle for variety)
        if num_samples is not None and len(df) > num_samples:
            df = df.sample(frac=1, random_state=42).head(num_samples)

        if len(df) == 0:
            raise RuntimeError("FamousFiguresDataset: No audio after filtering.")

        # Store compact rows for fast indexing
        # Map label to 1/0 (bonafide=1, spoof/other=0)
        def _lbl_to_int(lbl: str) -> int:
            return 1 if str(lbl).lower() == "bonafide" else 0

        self.rows = [
            (Path(row["AudioPath"]), _lbl_to_int(row["Label"]), str(row["Speaker"]), str(row["Source"]))
            for _, row in df.iterrows()
        ]

        print(f"[INFO] FamousFigures: loaded {len(self.rows)} samples (subset={subset}).")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        audio_path, label_int, speaker, source = self.rows[idx]
        waveform = self._process_audio(audio_path)
        label = torch.tensor(label_int, dtype=torch.long)
        return waveform, label, speaker, source

class ASVspoof2019Dataset(BaseAudioDataset):
    def __init__(
        self,
        protocol_file: str,
        root_dir: str = "",
        num_samples: int = None,
        subset: str = "all",
        sample_seed: int = 1337,   # <— NEW (optional)
        **kwargs
    ):
        super().__init__(**kwargs)
        self.root_dir = Path(root_dir)
        self.data = []
        self.attack_to_idx = {"bonafide": 0}  # multi-class: bonafide -> 0

        subset = (subset or "all").lower()
        if subset not in {"all", "bonafide", "spoof"}:
            raise ValueError(
                f"subset must be one of 'all', 'bonafide', or 'spoof' (got: {subset})"
            )

        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                audio_rel = parts[0]
                attack_id_raw = parts[1]          # e.g., A11 or '-' for bonafide
                label_str = parts[2].lower()      # 'bonafide' or 'spoof'
                speaker_id = parts[4]             # e.g., p240

                # subset filter
                if subset != "all" and label_str != subset:
                    continue

                # file name only
                audio_name = audio_rel.split("/").pop()
                full_path = self.root_dir / audio_name

                # Binary label for convenience
                binary_label = 1 if label_str == "bonafide" else 0

                # Multi-class label:
                if label_str == "bonafide":
                    key = "bonafide"
                else:
                    # spoof -> use attack ID (e.g., A11)
                    key = attack_id_raw

                if key not in self.attack_to_idx:
                    self.attack_to_idx[key] = len(self.attack_to_idx)

                multi_label = self.attack_to_idx[key]

                # Store both labels
                self.data.append((full_path, binary_label, multi_label, speaker_id, audio_name))

        if num_samples is not None:
            n = min(int(num_samples), len(self.data))
            rng = random.Random(sample_seed)     # reproducible
            self.data = rng.sample(self.data, n) # random subset

        if not self.data:
            raise RuntimeError(
                f"No audio files found from protocol {protocol_file} "
                f"after applying subset='{subset}'."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, binary_label, multi_label, speaker_id, audio_name = self.data[idx]
        waveform = self._process_audio(audio_path)
        return (
            waveform,
            torch.tensor(binary_label, dtype=torch.long),
            torch.tensor(multi_label, dtype=torch.long),
            speaker_id,
            audio_name,
        )

class RAVDESSDataset(BaseAudioDataset):
    """A PyTorch Dataset for loading audio from the RAVDESS dataset."""
    def __init__(self, root_dir: str, **kwargs):
        num_samples = kwargs.pop("num_samples", None)
        super().__init__(**kwargs)
        
        self.root_dir = Path(root_dir)
        self.audio_files = sorted(list(self.root_dir.glob('**/Actor_*/*.wav')))
        
        if num_samples is not None:
            self.audio_files = self.audio_files[:num_samples]
            
        if not self.audio_files:
            raise RuntimeError(f"No .wav files found in {root_dir}.")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform = self._process_audio(audio_path)
        label = torch.tensor(1, dtype=torch.long)
        return waveform, label

class CommonVoiceDataset(BaseAudioDataset):
    """A PyTorch Dataset for loading audio from the Common Voice dataset."""
    def __init__(self, root_dir: str, **kwargs):
        num_samples = kwargs.pop("num_samples", None)
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir)
        self.audio_files = sorted(list(self.root_dir.glob('**/*.wav')))
        
        if num_samples is not None:
            self.audio_files = self.audio_files[:num_samples]
            
        if not self.audio_files:
            raise RuntimeError(f"No .wav files found in {root_dir}.")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform = self._process_audio(audio_path)
        label = torch.tensor(1, dtype=torch.long)
        return waveform, label

# class ASVspoof2021Dataset(BaseAudioDataset):
#     """A PyTorch Dataset for loading audio from the ASVspoof 2021 DF eval dataset."""
#     def __init__(self, root_dir: str, protocol_file: str, subset='all', **kwargs):
#         num_samples = kwargs.pop("num_samples", None)
#         super().__init__(**kwargs)
        
#         self.root_dir = Path(root_dir)
#         self.audio_folder = self.root_dir / "flac"
        
#         if not Path(protocol_file).exists():
#             raise FileNotFoundError(f"Protocol file not found: {protocol_file}")
            
#         col_names = ['speaker_id', 'filename', 'compression', 'source', 'attack_id', 'label', 'trim', 'set', 'vocoder_type', 'col10', 'col11', 'col12', 'col13']
#         protocol_df = pd.read_csv(protocol_file, sep='\s+', header=None, engine='python', names=col_names)
        
#         protocol_df.dropna(subset=['filename'], inplace=True)

#         original_count = len(protocol_df)
#         protocol_df['exists'] = protocol_df['filename'].apply(lambda fname: (self.audio_folder / f"{fname}.flac").exists())
#         protocol_df = protocol_df[protocol_df['exists']]
#         if len(protocol_df) < original_count:
#             print(f"[INFO] ASVspoof2021: Filtered out {original_count - len(protocol_df)} missing audio files.")

#         if subset == 'bonafide':
#             self.protocol = protocol_df[protocol_df['label'] == 'bonafide'].reset_index(drop=True)
#         elif subset == 'spoof':
#             self.protocol = protocol_df[protocol_df['label'] != 'bonafide'].reset_index(drop=True)
#         else:
#             self.protocol = protocol_df
        
#         if num_samples is not None:
#             self.protocol = self.protocol.sample(frac=1, random_state=42).reset_index(drop=True).head(num_samples)

#         if len(self.protocol) == 0:
#             raise RuntimeError(f"Found 0 audio files after filtering for subset '{subset}'.")

#     def __len__(self):
#         return len(self.protocol)

#     def __getitem__(self, idx):
#         row = self.protocol.iloc[idx]
#         audio_path = self.audio_folder / f"{row['filename']}.flac"
#         waveform = self._process_audio(audio_path)
#         label = torch.tensor(1 if row['label'] == 'bonafide' else 0, dtype=torch.long)
#         return waveform, label

class ASVspoof2021Dataset(BaseAudioDataset):
    """A PyTorch Dataset for loading audio from the ASVspoof 2021 DF eval dataset using ok_files.txt."""
    def __init__(self, root_dir: str, ok_files: str, protocol_file: str, subset="all", **kwargs):
        num_samples = kwargs.pop("num_samples", None)
        super().__init__(**kwargs)
        
        self.root_dir = Path(root_dir)
        self.audio_folder = self.root_dir / "flac"

        # Load ok_files list
        with open(ok_files, "r") as f:
            ok_list = [line.strip() for line in f if line.strip()]
        ok_set = set([Path(x).stem for x in ok_list])  # keep just the stems like DF_E_2000011

        # Load protocol file for labels
        col_names = [
            "speaker_id", "filename", "compression", "source", "attack_id",
            "label", "trim", "set", "vocoder_type", "col10", "col11", "col12", "col13"
        ]
        protocol_df = pd.read_csv(protocol_file, sep="\s+", header=None, engine="python", names=col_names)

        # Filter by ok_files
        protocol_df = protocol_df[protocol_df["filename"].isin(ok_set)]

        if subset == "bonafide":
            self.protocol = protocol_df[protocol_df["label"] == "bonafide"].reset_index(drop=True)
        elif subset == "spoof":
            self.protocol = protocol_df[protocol_df["label"] != "bonafide"].reset_index(drop=True)
        else:
            self.protocol = protocol_df.reset_index(drop=True)

        if num_samples is not None:
            self.protocol = self.protocol.sample(frac=1, random_state=42).reset_index(drop=True).head(num_samples)

        if len(self.protocol) == 0:
            raise RuntimeError(f"Found 0 audio files after filtering with ok_files and subset='{subset}'.")

        print(f"[INFO] Loaded {len(self.protocol)} samples (subset={subset}).")

    def __len__(self):
        return len(self.protocol)

    def __getitem__(self, idx):
        row = self.protocol.iloc[idx]
        audio_path = self.audio_folder / f"{row['filename']}.flac"
        waveform = self._process_audio(audio_path)
        label = torch.tensor(1 if row["label"] == "bonafide" else 0, dtype=torch.long)
        return waveform, label

class InTheWildDataset(BaseAudioDataset):
    """A PyTorch Dataset for loading audio from an In-the-Wild dataset."""
    def __init__(self, root_dir: str, protocol_file: str, subset='all', **kwargs):
        num_samples = kwargs.pop("num_samples", None)
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir)
        self.audio_folder = self.root_dir  # audio files directly here (or adjust if needed)

        if not Path(protocol_file).exists():
            raise FileNotFoundError(f"Protocol file not found: {protocol_file}")

        protocol_df = pd.read_csv(protocol_file)

        # Standardize labels: 'bona-fide' -> 'bonafide'
        protocol_df['label'] = protocol_df['label'].replace('bona-fide', 'bonafide')

        # ---------- NEW: detect speaker column ----------
        self.spk_col = "speaker"
        # ------------------------------------------------

        # Keep only rows where audio file exists
        original_count = len(protocol_df)
        protocol_df['exists'] = protocol_df['file'].apply(
            lambda fname: (self.audio_folder / fname).exists()
        )
        protocol_df = protocol_df[protocol_df['exists']]
        if len(protocol_df) < original_count:
            print(f"[INFO] InTheWild: Filtered out {original_count - len(protocol_df)} missing audio files.")

        # Subset filtering
        if subset == 'bonafide':
            self.protocol = protocol_df[protocol_df['label'] == 'bonafide'].reset_index(drop=True)
        elif subset == 'spoof':
            self.protocol = protocol_df[protocol_df['label'] == 'spoof'].reset_index(drop=True)
        else:
            self.protocol = protocol_df

        # Optional sampling
        if num_samples is not None:
            self.protocol = (
                self.protocol
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
                .head(num_samples)
            )

        if len(self.protocol) == 0:
            raise RuntimeError(f"Found 0 audio files after filtering for subset '{subset}'.")

    def __len__(self):
        return len(self.protocol)

    def __getitem__(self, idx):
        row = self.protocol.iloc[idx]

        # file path + audio name
        audio_rel = str(row['file'])
        audio_path = self.audio_folder / audio_rel
        audio_name = Path(audio_rel).name

        # speaker (if column exists; else 'unknown')
        if self.spk_col is not None:
            speaker = str(row[self.spk_col])
        else:
            speaker = "unknown"

        waveform = self._process_audio(audio_path)
        label = torch.tensor(
            1 if row['label'] == 'bonafide' else 0,
            dtype=torch.long
        )

        # ---------- KEY CHANGE: now return 4 items ----------
        return waveform, label, speaker, audio_name

# Example of how to use the new InTheWildDataset
if __name__ == '__main__':

    # Path to directory with .flac files
    # ASVSPOOF_TRAIN_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac"
    # ASVSPOOF_TRAIN_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt"

    ITW_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild"
    ITW_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv"

    # train_dataset = ASVspoof2019Dataset(
    #     root_dir=ASVSPOOF_TRAIN_ROOT,
    #     protocol_file=ASVSPOOF_TRAIN_PROTOCOL,
    #     subset=None,
    #     num_samples=22000,
    #     max_duration_seconds=5
    # )

    train_dataset = InTheWildDataset(
        root_dir=ITW_ROOT,
        protocol_file=ITW_PROTOCOL,
        subset=None,
        num_samples=30,
        max_duration_seconds=5
    )

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        num_workers=4, collate_fn=pad_collate_fn_speaker_source, pin_memory=True
    )

    for wav, lab, speaker, source in train_loader:
        # print(f"Waveforms: {wav}")
        print(f"Labels: {lab}")
        # print(f"Attack: {attack}")
        print(f"Speaker: {speaker}")
        print(f"Source: {source}")