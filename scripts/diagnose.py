"""Phase B diagnostics. VALIDATION ONLY. Never touches the test set.
Tests: effective-step-size (H1), trust-region activity (H2), batch interpolation (H3).
Parity: all arms train with the same dropout and weight decay. Seeds are fixed before the DAE.
Usage (repo root):  python scripts/diagnose.py --exp lr|ngd|all [--steps 2100] [--seed 0]
"""
import argparse
import csv
import os
import sys

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.dataset import build_datasets
from src.models import SpecCNN, SpecDAE
from src.optimizers.mfngd_dual import DualNGD
from src.utils import seed_everything
from run_comparison import CUDABenchmarkLoader

TARGET = 0.90
WD = 1e-4
NAN = float("nan")


def sgd(lr):
    return lambda m: ("base", torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=WD))


def adam(lr):
    return lambda m: ("base", torch.optim.Adam(m.parameters(), lr=lr, weight_decay=WD))


def dual(**kw):
    return lambda m: ("ngd", DualNGD(m, weight_decay=WD, log_deff=True, dropout=False, **kw))


EXPS = {
    "lr": {
        **{f"SGD lr={lr:g}": sgd(lr) for lr in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)},
        **{f"Adam lr={lr:g}": adam(lr) for lr in (3e-4, 1e-3, 3e-3)},
    },
    "ngd": {
        "Dual lr=1e-3 d=1e-2 (your harness)": dual(lr=1e-3, delta=1e-2, damping=5e-2),
        "Dual TR d=1e-3": dual(lr=1.0, delta=1e-3, damping=5e-2),
        "Dual TR d=1e-4": dual(lr=1.0, delta=1e-4, damping=5e-2),
        "Dual TR d=1e-3 lam=0.5": dual(lr=1.0, delta=1e-3, damping=0.5),
        "Dual TR d=1e-3 window=4": dual(lr=1.0, delta=1e-3, damping=5e-2, window=4),
        "Dual TR d=1e-3 ggn_weighted": dual(lr=1.0, delta=1e-3, damping=5e-2, curvature="ggn_weighted"),
    },
}


@torch.no_grad()
def val_auprc(cnn, loader, dae):
    cnn.eval()
    ps, ys = [], []
    for xb, yb in loader.get_batches(0, dae):
        ps.append(torch.sigmoid(cnn(xb)).flatten())
        ys.append(yb.flatten())
    p = torch.cat(ps).cpu().numpy()
    y = torch.cat(ys).cpu().numpy().astype(int)
    cnn.train()
    return float(average_precision_score(y, p))


def run(name, make, a, cfg, device, loaders, dae, pw):
    seed_everything(a.seed)
    cnn = SpecCNN(norm=cfg.NORM, gn_groups=cfg.GN_GROUPS, dropout=a.dropout).to(device)
    kind, opt = make(cnn)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    cnn.train()
    rows, evs, stats = [], [], []
    train_ms, step, epoch = 0.0, 0, 0
    nan_t = torch.full((), NAN, device=device)
    while step < a.steps:
        for xb, yb in loaders["train"].get_batches(epoch, dae):
            s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s_ev.record()
            if kind == "ngd":
                info = opt.step(xb, yb, pw)
                stats.append(torch.stack([info["s"], info["x_over_g"], info["rho"], info["lam"],
                                          info.get("d_eff", nan_t).to(info["s"].dtype)]))
            else:
                opt.zero_grad(set_to_none=True)
                crit(cnn(xb), yb).backward()
                opt.step()
            e_ev.record()
            evs.append((s_ev, e_ev))
            step += 1
            if step % a.eval_every == 0 or step == a.steps:
                torch.cuda.synchronize()
                train_ms += sum(s.elapsed_time(e) for s, e in evs)
                evs.clear()
                m = torch.nanmean(torch.stack(stats), 0).cpu().tolist() if stats else [NAN] * 5
                stats.clear()
                rows.append({"name": name, "step": step, "train_s": train_ms / 1000.0,
                             "val_auprc": val_auprc(cnn, loaders["val"], dae),
                             "s": m[0], "x_over_g": m[1], "rho": m[2], "lam": m[3], "d_eff": m[4]})
            if step >= a.steps:
                break
        epoch += 1
    return rows


def first_hit(rows):
    for r in rows:
        if r["val_auprc"] >= TARGET:
            return r["step"], r["train_s"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["lr", "ngd", "all"])
    ap.add_argument("--steps", type=int, default=2100)
    ap.add_argument("--eval_every", type=int, default=105)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.0)
    a = ap.parse_args()

    cfg = Config()
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "diagnose.py needs CUDA (event timing)"

    seed_everything(a.seed)
    datasets, _ = build_datasets(cfg, run_seed=a.seed)
    loaders = {k: CUDABenchmarkLoader(datasets[k], cfg.BATCH_SIZE, k == "train", cfg, device)
               for k in ("train", "val")}
    y = datasets["train"].labels.numpy()
    pw = torch.tensor([(y == 0).sum() / (y == 1).sum()], dtype=torch.float32, device=device)

    seed_everything(a.seed)                       # DAE seeded: reproducible across runs
    dae = SpecDAE().to(device)
    dae_opt = torch.optim.Adam(dae.parameters(), lr=cfg.DAE_LR)
    mse = nn.MSELoss()
    dae.train()
    print("Pre-training DAE...")
    for epoch in range(cfg.DAE_EPOCHS):
        for noisy, clean in loaders["train"].get_batches(epoch, dae=None, yield_clean=True):
            dae_opt.zero_grad()
            mse(dae(noisy), clean).backward()
            dae_opt.step()
    dae.eval()

    names = ["lr", "ngd"] if a.exp == "all" else [a.exp]
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    for exp in names:
        all_rows, summary = [], []
        for name, make in EXPS[exp].items():
            print(f"[{exp}] {name}")
            rows = run(name, make, a, cfg, device, loaders, dae, pw)
            all_rows += rows
            hit = first_hit(rows)
            last = rows[-1]
            summary.append((name, max(r["val_auprc"] for r in rows), last["val_auprc"], hit,
                            last["s"], last["x_over_g"], last["d_eff"]))
        path = os.path.join(cfg.RESULTS_DIR, f"diagnose_{exp}_seed{a.seed}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n=== {exp} (val AUPRC, target {TARGET}) -> {path}")
        print(f"{'config':<38} {'best':>6} {'final':>6} {'steps@tgt':>9} {'sec@tgt':>8} {'s':>6} {'x/g':>7} {'d_eff':>7}")
        for n, b, fin, hit, s, xg, de in summary:
            st, sec = (str(hit[0]), f"{hit[1]:.2f}") if hit else ("never", "-")
            print(f"{n:<38} {b:6.4f} {fin:6.4f} {st:>9} {sec:>8} {s:6.3f} {xg:7.2f} {de:7.1f}")


if __name__ == "__main__":
    main()