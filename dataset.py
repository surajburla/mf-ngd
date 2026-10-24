import json
import os
import time
import urllib.request
import warnings
import zipfile
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset

from src.config import Config


class RecordReadError(RuntimeError):
    """A MIT-BIH record could not be read. The pipeline never skips it silently."""


class SplitCountMismatch(RuntimeError):
    """Beat counts differ from the Bangaram specification."""


# ------------------------------------------------------------------ download
def download_mitdb_fallback(target_dir):
    zip_url = "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"
    zip_path = os.path.join(target_dir, "mitdb.zip")
    print("Falling back to direct archive download from PhysioNet...")
    os.makedirs(target_dir, exist_ok=True)
    urllib.request.urlretrieve(zip_url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    extracted = os.path.join(target_dir, "mit-bih-arrhythmia-database-1.0.0")
    if os.path.exists(extracted):
        for fname in os.listdir(extracted):
            dst = os.path.join(target_dir, fname)
            if not os.path.exists(dst):
                os.rename(os.path.join(extracted, fname), dst)
    if os.path.exists(zip_path):
        os.remove(zip_path)


def _files_present(data_dir, rec):
    return all(os.path.exists(os.path.join(data_dir, f"{rec}.{e}")) for e in ("hea", "dat", "atr"))


def _read_record_list(data_dir):
    """Offline-first. Read RECORDS from disk. Hit the network only if the file is absent."""
    path = os.path.join(data_dir, "RECORDS")
    if os.path.exists(path):
        with open(path) as f:
            return sorted(line.strip() for line in f if line.strip())
    return sorted(wfdb.get_record_list("mitdb"))


def ensure_dataset(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    records = _read_record_list(data_dir)
    if not [r for r in records if not _files_present(data_dir, r)]:
        return records

    print("Downloading MIT-BIH Database...")
    for attempt in range(3):
        try:
            wfdb.dl_database("mitdb", data_dir)
            break
        except Exception as e:  # network errors are heterogeneous; we retry then fall back
            print(f"Download attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    else:
        download_mitdb_fallback(data_dir)

    records = _read_record_list(data_dir)
    missing = [r for r in records if not _files_present(data_dir, r)]
    if missing:
        raise RecordReadError(f"Records missing after download: {missing}")
    return records


# ------------------------------------------------------------------ split
@dataclass
class ECGSplit:
    name: str
    clean: np.ndarray            # (N, L) float32, z-scored per beat
    labels: np.ndarray           # (N,)  int64, 0 normal / 1 abnormal
    groups: np.ndarray           # (N,)  int64, record id (cluster id for bootstrap)
    records: list
    stats: dict = field(default_factory=dict)


def split_records(records, cfg):
    """Reproduces np.random.seed(42); np.random.shuffle(records). Test set stays identical
    to the legacy split. Validation comes from the tail of the 38-record pool."""
    if len(records) != cfg.NUM_POOL_RECORDS + cfg.NUM_TEST_RECORDS:
        raise RuntimeError(f"Expected 48 records, found {len(records)}")
    shuffled = list(records)
    np.random.RandomState(cfg.RANDOM_SEED).shuffle(shuffled)
    pool, test = shuffled[:cfg.NUM_POOL_RECORDS], shuffled[cfg.NUM_POOL_RECORDS:]
    n_tr = cfg.NUM_POOL_RECORDS - cfg.NUM_VAL_RECORDS
    return pool[:n_tr], pool[n_tr:], test


def extract_beats(data_dir, record_list, cfg, name):
    offsets = np.arange(-cfg.WINDOW_PRE, cfg.WINDOW_POST)
    normal, abnormal = list(cfg.NORMAL_SYMBOLS), list(cfg.ABNORMAL_SYMBOLS)
    beats, labels, groups, per_record = [], [], [], {}

    for rec in record_list:
        path = os.path.join(data_dir, rec)
        try:
            signal, _ = wfdb.rdsamp(path, channels=[0])
            ann = wfdb.rdann(path, "atr")
        except Exception as exc:
            raise RecordReadError(f"Record {rec}: {type(exc).__name__}: {exc}") from exc

        sig = signal[:, 0].astype(np.float64)
        if not np.isfinite(sig).all():
            raise RecordReadError(f"Record {rec}: non-finite samples")

        samples = np.asarray(ann.sample)
        symbols = np.asarray(ann.symbol)
        is_abn = np.isin(symbols, abnormal)
        keep_sym = np.isin(symbols, normal) | is_abn
        samples, lbl = samples[keep_sym], is_abn[keep_sym].astype(np.int64)

        in_bounds = (samples - cfg.WINDOW_PRE >= 0) & (samples + cfg.WINDOW_POST <= len(sig))
        samples, lbl = samples[in_bounds], lbl[in_bounds]

        win = sig[samples[:, None] + offsets[None, :]]
        std = np.maximum(win.std(axis=1, keepdims=True), 1e-8)
        win = (win - win.mean(axis=1, keepdims=True)) / std

        beats.append(win.astype(np.float32))
        labels.append(lbl)
        groups.append(np.full(len(lbl), int(rec), dtype=np.int64))
        per_record[rec] = {
            "annotations": int(len(ann.sample)),
            "dropped_symbol": int((~keep_sym).sum()),
            "dropped_boundary": int((~in_bounds).sum()),
            "kept": int(len(lbl)),
        }

    return ECGSplit(
        name=name,
        clean=np.concatenate(beats),
        labels=np.concatenate(labels),
        groups=np.concatenate(groups),
        records=list(record_list),
        stats=per_record,
    )


def load_and_preprocess_ecg(cfg=None):
    """Returns ({'train','val','test'} -> ECGSplit, manifest dict). API changed vs legacy tuple."""
    cfg = cfg or Config()
    data_dir = os.path.abspath(cfg.DATA_DIR)
    records = ensure_dataset(data_dir)
    tr_rec, va_rec, te_rec = split_records(records, cfg)

    splits = {
        "train": extract_beats(data_dir, tr_rec, cfg, "train"),
        "val": extract_beats(data_dir, va_rec, cfg, "val"),
        "test": extract_beats(data_dir, te_rec, cfg, "test"),
    }

    n_pool = len(splits["train"].labels) + len(splits["val"].labels)
    n_test = len(splits["test"].labels)
    manifest = {
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "wfdb": wfdb.__version__},
        "config": asdict(cfg),
        "records": {k: v.records for k, v in splits.items()},
        "counts": {
            k: {"beats": int(len(v.labels)),
                "normal": int((v.labels == 0).sum()),
                "abnormal": int((v.labels == 1).sum())}
            for k, v in splits.items()
        },
        "pool_beats": int(n_pool),
        "test_beats": int(n_test),
        "expected": {"pool": cfg.EXPECTED_POOL_BEATS, "test": cfg.EXPECTED_TEST_BEATS},
        "per_record": {k: v.stats for k, v in splits.items()},
    }

    print("\n=== SPLIT MANIFEST ===")
    for k, c in manifest["counts"].items():
        print(f"{k:<6} beats={c['beats']:>7}  normal={c['normal']:>7}  abnormal={c['abnormal']:>7}")
    print(f"pool (train+val) = {n_pool}  (spec {cfg.EXPECTED_POOL_BEATS})")
    print(f"test             = {n_test}  (spec {cfg.EXPECTED_TEST_BEATS})")

    if n_pool != cfg.EXPECTED_POOL_BEATS or n_test != cfg.EXPECTED_TEST_BEATS:
        msg = (f"Beat counts deviate from spec: pool {n_pool} vs {cfg.EXPECTED_POOL_BEATS}, "
               f"test {n_test} vs {cfg.EXPECTED_TEST_BEATS}. Document this split.")
        if cfg.STRICT_SPEC_COUNTS:
            raise SplitCountMismatch(msg)
        warnings.warn(msg)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(cfg.RESULTS_DIR, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return splits, manifest


# ------------------------------------------------------------------ dataset
class ECGDataset(Dataset):
    """Holds clean beats. Generates noisy beats on demand.
    fixed=True  -> one noise realization forever (val/test).
    fixed=False -> call resample_noise(epoch) each epoch (train)."""

    def __init__(self, split, noise_factor, noise_seed, fixed):
        self.clean = torch.from_numpy(split.clean).unsqueeze(1)
        self.labels = torch.from_numpy(split.labels.astype(np.float32)).unsqueeze(1)
        self.groups = split.groups
        self.noise_factor = noise_factor
        self.noise_seed = noise_seed
        self.fixed = fixed
        self.noisy = None
        self.resample_noise(0)

    def resample_noise(self, epoch):
        if self.fixed and self.noisy is not None:
            return
        g = torch.Generator(device="cpu")
        g.manual_seed(self.noise_seed + (0 if self.fixed else int(epoch)))
        self.noisy = self.clean + self.noise_factor * torch.randn(self.clean.shape, generator=g)

    def tensors(self, device):
        return self.noisy.to(device), self.clean.to(device), self.labels.to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx], self.labels[idx]


def build_datasets(cfg=None, run_seed=0):
    cfg = cfg or Config()
    splits, manifest = load_and_preprocess_ecg(cfg)
    train_noise_seed = 1_000_003 * (run_seed + 1)
    ds = {
        "train": ECGDataset(splits["train"], cfg.NOISE_FACTOR, train_noise_seed,
                            fixed=not cfg.NOISE_RESAMPLE_PER_EPOCH),
        "val": ECGDataset(splits["val"], cfg.NOISE_FACTOR, cfg.VAL_NOISE_SEED, fixed=True),
        "test": ECGDataset(splits["test"], cfg.NOISE_FACTOR, cfg.TEST_NOISE_SEED, fixed=True),
    }
    return ds, manifest
