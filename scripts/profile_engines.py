"""Engine profiler. Median + IQR over repeats, baselines included, peak VRAM per arm.
Usage (repo root): python scripts/profile_engines.py
"""
import gc
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import SpecCNN
from src.optimizers.mfngd_dual import DualNGD, ExactCGNGD
from src.profiling import peak_memory_mb, reset_peak_memory, time_callable

REPS = 9
BATCHES = [64, 128, 256, 512, 1024, 2048, 4096]


def fresh():
    gc.collect()
    torch.cuda.empty_cache()


def measure(make_fn):
    fresh()
    try:
        fn = make_fn()
        fn()                                   # warm-up: allocations, autotune
        torch.cuda.synchronize()
        reset_peak_memory()
        r = time_callable(fn, warmup=1, repeat=REPS)
        out = (r["median_s"] * 1e3, r["iqr_s"] * 1e3, peak_memory_mb())
    except torch.cuda.OutOfMemoryError:
        out = None
    fresh()
    return out


def fmt(r):
    return "OOM" if r is None else f"{r[0]:.1f}±{r[1]:.1f}"


def main():
    assert torch.cuda.is_available(), "CUDA required"
    dev = "cuda"
    pw = torch.tensor([3.3], device=dev)
    print(f"{'B':>5} | {'SGD ms':>10} | {'Adam ms':>10} | {'Dual ms':>12} | {'Dual MB':>8} | {'CG ms':>12} | {'CG MB':>8} | {'Dual/SGD':>8}")
    print("-" * 95)
    for B in BATCHES:
        xb = torch.randn(B, 1, 216, device=dev)
        yb = (torch.rand(B, device=dev) > 0.5).float()

        def base(opt_cls, **kw):
            def make():
                m = SpecCNN("group").to(dev)
                o = opt_cls(m.parameters(), **kw)

                def f():
                    o.zero_grad(set_to_none=True)
                    F.binary_cross_entropy_with_logits(m(xb).view(-1), yb, pos_weight=pw).backward()
                    o.step()
                return f
            return make

        def ngd(cls, **kw):
            def make():
                m = SpecCNN("group").to(dev)
                o = cls(m, lr=1.0, delta=1e-3, weight_decay=1e-4, **kw)
                return lambda: o.step(xb, yb, pw)
            return make

        sgd = measure(base(torch.optim.SGD, lr=1e-2, momentum=0.9, weight_decay=1e-4))
        adam = measure(base(torch.optim.Adam, lr=1e-3, weight_decay=1e-4))
        dual = measure(ngd(DualNGD))
        cg = measure(ngd(ExactCGNGD, cg_iters=5))
        ratio = "-" if (dual is None or sgd is None) else f"{dual[0] / sgd[0]:.1f}x"
        print(f"{B:>5} | {fmt(sgd):>10} | {fmt(adam):>10} | {fmt(dual):>12} | "
              f"{'OOM' if dual is None else f'{dual[2]:.0f}':>8} | {fmt(cg):>12} | "
              f"{'OOM' if cg is None else f'{cg[2]:.0f}':>8} | {ratio:>8}")


if __name__ == "__main__":
    main()