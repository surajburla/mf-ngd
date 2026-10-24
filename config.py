import os
from dataclasses import dataclass


@dataclass
class Config:
    # ---------------- Data / split ----------------
    DATA_DIR: str = os.path.join("data", "mitdb")
    RESULTS_DIR: str = "results"
    RANDOM_SEED: int = 42                 # SPLIT seed. Never change between runs.
    TRAIN_SEEDS: tuple = tuple(range(10)) # init / data-order seeds (Phase D)

    NUM_POOL_RECORDS: int = 38            # spec "training" pool (train + val)
    NUM_VAL_RECORDS: int = 8              # carved out of the pool
    NUM_TEST_RECORDS: int = 10

    # Spec-fidelity targets (Bangaram PDF, section 10). Applied to the PRE-carve pool.
    EXPECTED_POOL_BEATS: int = 81_301
    EXPECTED_TEST_BEATS: int = 18_732
    STRICT_SPEC_COUNTS: bool = False      # True -> raise on mismatch, False -> warn + log

    # ---------------- Beat extraction ----------------
    WINDOW_PRE: int = 108
    WINDOW_POST: int = 108
    SEQUENCE_LENGTH: int = 216
    NORMAL_SYMBOLS: tuple = ("N",)
    ABNORMAL_SYMBOLS: tuple = ("A", "V", "L", "R")

    # ---------------- Noise ----------------
    NOISE_FACTOR: float = 0.30            # spec value
    NOISE_RESAMPLE_PER_EPOCH: bool = True # DEVIATION from spec (train split only)
    VAL_NOISE_SEED: int = 10_000          # fixed noise for validation
    TEST_NOISE_SEED: int = 20_000         # fixed noise for test

    # ---------------- Models ----------------
    NORM: str = "group"                   # DEVIATION: spec uses BatchNorm. "group" | "batch" | "none"
    GN_GROUPS: int = 8
    DROPOUT: float = 0.3                  # spec value

    # ---------------- Training ----------------
    BATCH_SIZE: int = 128                 # spec value
    DAE_EPOCHS: int = 20
    CNN_EPOCHS: int = 20
    EARLY_STOP_PATIENCE: int = 3
    DAE_LR: float = 1e-3
    DEVICE: str = "cuda"

    # ---------------- Evaluation ----------------
    PRECISION_TARGET: float = 0.90        # recall@precision, threshold chosen on VAL only
    N_BOOT: int = 2000
    CI_ALPHA: float = 0.05

    # ---------------- Legacy optimizer settings (CG ablation arm) ----------------
    LEARNING_RATE: float = 1e-3
    MFNGD_CG_ITERS: int = 5
    MFNGD_INIT_DAMPING: float = 5e-2
    MFNGD_MAX_KL: float = 1e-2
    KFAC_INVERT_FREQ: int = 5
    KFAC_DAMPING: float = 5e-2

    def __post_init__(self):
        assert self.NORM in ("group", "batch", "none"), self.NORM
        assert 0 < self.NUM_VAL_RECORDS < self.NUM_POOL_RECORDS
        assert self.WINDOW_PRE + self.WINDOW_POST == self.SEQUENCE_LENGTH
