import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import SpecCNN, SpecDAE, count_trainable
from src.optimizers.mfngd_dual import DualNGD, ExactCGNGD


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv1d(1, 4, 3, padding=1)
        self.n = nn.GroupNorm(2, 4)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(torch.relu(self.n(self.c(x))).mean(-1))


def _setup(B=16, L=12):
    torch.manual_seed(0)
    m = Tiny().double()
    x = torch.randn(B, 1, L, dtype=torch.double)
    y = (torch.rand(B) > 0.6).double()
    return m, x, y


def _dense(m, x, y, pw, lam):
    params = list(m.parameters())
    B = x.shape[0]
    Js, zs = [], []
    for i in range(B):
        z = m(x[i:i + 1]).reshape(())
        Js.append(torch.cat([t.reshape(-1) for t in torch.autograd.grad(z, params)]))
        zs.append(z.detach())
    J, z = torch.stack(Js), torch.stack(zs)
    p = torch.sigmoid(z)
    r = p * (1 + (pw - 1) * y) - pw * y
    g = J.t() @ r / B
    Fm = (J.t() * (p * (1 - p))) @ J / B
    xs = torch.linalg.solve(Fm + lam * torch.eye(J.shape[1], dtype=torch.double), g)
    return xs, g, Fm


def test_dual_matches_dense():
    m, x, y = _setup()
    xs, g, Fm = _dense(m, x, y, 3.0, 0.05)
    xd, gd, q, _, _ = DualNGD(m, damping=0.05, solve_dtype=torch.float64).direction(x, y, 3.0)
    assert torch.allclose(gd, g, rtol=1e-8, atol=1e-10)
    assert torch.allclose(xd, xs, rtol=1e-6, atol=1e-8)
    assert torch.allclose(q, xs @ Fm @ xs, rtol=1e-6)


def test_exact_cg_matches_dense():
    m, x, y = _setup()
    xs, _, Fm = _dense(m, x, y, 3.0, 0.05)
    xc, _, q, _, _ = ExactCGNGD(m, damping=0.05, cg_iters=25).direction(x, y, 3.0)
    assert torch.allclose(xc, xs, rtol=1e-5, atol=1e-7)
    assert torch.allclose(q, xs @ Fm @ xs, rtol=1e-5)


def test_trust_region_is_exact():
    m, x, y = _setup()
    _, _, Fm = _dense(m, x, y, 3.0, 0.05)
    opt = DualNGD(m, damping=0.05, delta=1e-6, lm_every=10 ** 9, solve_dtype=torch.float64)
    t0 = opt._flat().clone()
    info = opt.step(x, y, 3.0)
    d = opt._flat() - t0
    assert info["s"] < 1.0
    assert abs(0.5 * d @ Fm @ d - 1e-6) < 1e-10


def test_no_host_sync():
    if not torch.cuda.is_available():
        return
    m = Tiny().cuda()
    x = torch.randn(16, 1, 12, device="cuda")
    y = (torch.rand(16, device="cuda") > 0.6).float()
    opt = DualNGD(m, lm_every=1)
    opt.step(x, y, 3.0)
    torch.cuda.set_sync_debug_mode("error")
    try:
        opt.step(x, y, 3.0)
    finally:
        torch.cuda.set_sync_debug_mode("default")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)