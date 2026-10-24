import statistics
import time

import torch


def _sync(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize()


class SyncTimer:
    """Synchronizes the GPU before and after the timed region."""

    def __init__(self, device="cuda"):
        self.device, self.elapsed = device, None

    def __enter__(self):
        _sync(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        _sync(self.device)
        self.elapsed = time.perf_counter() - self.t0


def time_callable(fn, warmup=3, repeat=15, device="cuda"):
    for _ in range(warmup):
        fn()
    runs = []
    for _ in range(repeat):
        with SyncTimer(device) as t:
            fn()
        runs.append(t.elapsed)
    q = statistics.quantiles(runs, n=4) if len(runs) >= 4 else [min(runs), 0, max(runs)]
    return {"median_s": statistics.median(runs), "mean_s": statistics.fmean(runs),
            "iqr_s": q[2] - q[0], "runs": runs}


def reset_peak_memory(device="cuda"):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device="cuda"):
    return torch.cuda.max_memory_allocated(device) / 2**20 if torch.cuda.is_available() else 0.0