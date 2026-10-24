"""Phase B natural-gradient engines. Never touches mfngd.py (the legacy FD-CG control arm).

DualNGD     : exact sample-space (Woodbury) solve. Per-sample Jacobian via vmap(jacrev).
ExactCGNGD  : matrix-free CG. Exact F v = J^T W J v via the double-backward (Townsend/Pearlmutter) trick.
Both      : closed-form quadratic-KL trust region, on-GPU Levenberg-Marquardt damping, no .item().
Loss      : BCEWithLogits(pos_weight=pw), mean reduction.
"""
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jacrev, vmap


def _assert_no_batchnorm(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            raise ValueError("BatchNorm couples samples. Per-sample Jacobians do not exist. Use GroupNorm.")


@contextlib.contextmanager
def _dropout_disabled(model, disable):
    flags = []
    if disable:
        flags = [(m, m.training) for m in model.modules() if isinstance(m, nn.modules.dropout._DropoutNd)]
        for m, _ in flags:
            m.eval()
    try:
        yield
    finally:
        for m, t in flags:
            m.train(t)


class _NGDBase:
    def __init__(self, model, lr=1.0, delta=1e-3, damping=5e-2, lam_min=1e-4, lam_max=1.0,
                 lm_every=5, weight_decay=0.0, curvature="fisher", dropout=False):
        assert curvature in ("fisher", "ggn_weighted")
        _assert_no_batchnorm(model)
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        ref = self.params[0]
        self.lam = torch.tensor(float(damping), dtype=ref.dtype, device=ref.device)
        self.rho = torch.zeros((), dtype=ref.dtype, device=ref.device)
        self.lr, self.delta = float(lr), float(delta)
        self.lam_min, self.lam_max = float(lam_min), float(lam_max)
        self.lm_every, self.wd = int(lm_every), float(weight_decay)
        self.curvature, self.dropout = curvature, bool(dropout)
        self.P = sum(p.numel() for p in self.params)
        self.t = 0

    # ---- flat-vector plumbing
    def _flat(self):
        return torch.cat([p.detach().reshape(-1) for p in self.params])

    @torch.no_grad()
    def _assign(self, vec):
        o = 0
        for p in self.params:
            n = p.numel()
            p.copy_(vec[o:o + n].view_as(p))
            o += n

    def _unflat(self, v):
        out, o = [], 0
        for p in self.params:
            n = p.numel()
            out.append(v[o:o + n].view_as(p))
            o += n
        return out

    # ---- loss pieces (pos_weight BCE)
    @staticmethod
    def _loss_terms(z, y, pw):
        p = torch.sigmoid(z)
        c = 1.0 + (pw - 1.0) * y          # pw on positives, 1 on negatives
        r = p * c - pw * y                # dL_i / dz_i
        return p, c, r

    def _loss(self, z, y, pw):
        return F.binary_cross_entropy_with_logits(z, y, pos_weight=pw.reshape(1))

    def _reg(self, theta):
        return 0.5 * self.wd * torch.dot(theta, theta)

    def _prep(self, xb, y, pw):
            dt = self.params[0].dtype
            yb = y.reshape(-1).to(dt, non_blocking=True)
            
            # Prevent CPU-GPU sync block by constructing on CPU, then async transfer
            if isinstance(pw, torch.Tensor):
                pw_t = pw.to(dtype=dt, device=xb.device, non_blocking=True)
            else:
                pw_t = torch.tensor(float(pw), dtype=dt).to(device=xb.device, non_blocking=True)
                
            return yb, pw_t.reshape(())

    # ---- trust region + Levenberg-Marquardt (sync-free)
    def _apply(self, x, g, q, loss0, xb, yb, pw_t):
        s = torch.clamp(torch.sqrt(2.0 * self.delta / (self.lr ** 2 * q + 1e-12)), max=1.0)
        alpha = self.lr * s
        theta0 = self._flat()
        theta1 = theta0 - alpha * x
        self._assign(theta1)
        self.t += 1
        if self.t % self.lm_every == 0:
            with torch.no_grad(), _dropout_disabled(self.model, not self.dropout):
                z1 = self.model(xb).reshape(-1)
            loss1 = self._loss(z1, yb, pw_t) + self._reg(theta1)
            actual = loss0 - loss1
            pred = alpha * torch.dot(g, x) - 0.5 * alpha ** 2 * q     # quadratic-model reduction
            self.rho = actual / pred.clamp_min(1e-12)
            self._assign(torch.where(actual > 0, theta1, theta0))     # reject on CPU-free branch
            lam = torch.where(self.rho > 0.75, self.lam * (2.0 / 3.0),
                              torch.where(self.rho < 0.25, self.lam * 1.5, self.lam))
            self.lam = lam.clamp(self.lam_min, self.lam_max)
        return {"loss": loss0.detach(), "s": s.detach(), "lam": self.lam, "rho": self.rho,
                "x_over_g": (torch.linalg.vector_norm(x) / torch.linalg.vector_norm(g).clamp_min(1e-30)),
                "q": q.detach()}

    def direction(self, xb, y, pw=1.0):
        yb, pw_t = self._prep(xb, y, pw)
        return self._direction(xb, yb, pw_t)

    def step(self, xb, y, pw=1.0):
        yb, pw_t = self._prep(xb, y, pw)
        x, g, q, loss0, extra = self._direction(xb, yb, pw_t)
        info = self._apply(x, g, q, loss0, xb, yb, pw_t)
        info.update(extra)
        return info


class DualNGD(_NGDBase):
    """x = (F + lam I)^-1 g with F = M^T M, M in R^{m x P}. Woodbury:
       x = (g - M^T (lam I + M M^T)^-1 M g) / lam.  Never forms the P x P matrix."""

    def __init__(self, model, window=1, chunk_size=None, solve_dtype=torch.float32,
                 log_deff=False, **kw):
        super().__init__(model, **kw)
        self.window, self.chunk = int(window), (chunk_size or None)
        self.solve_dtype, self.log_deff = solve_dtype, bool(log_deff)
        self._blocks = []

    def _jacobian(self, xb):
        params = {n: p.detach() for n, p in self.model.named_parameters() if p.requires_grad}
        buffers = {n: b.detach() for n, b in self.model.named_buffers()}

        def f(prm, buf, xi):
            z = functional_call(self.model, (prm, buf), (xi.unsqueeze(0),)).reshape(())
            return z, z.detach()

        jf = vmap(jacrev(f, has_aux=True), in_dims=(None, None, 0),
                  randomness="different", chunk_size=self.chunk)
        jac, z = jf(params, buffers, xb)
        B = xb.shape[0]
        J = torch.empty(B, self.P, device=xb.device, dtype=self.params[0].dtype)
        o = 0
        for n in self.names:                                   # copy then free: peak ~2x J, not 3x
            k = jac[n].numel() // B
            J[:, o:o + k] = jac[n].reshape(B, -1)
            del jac[n]
            o += k
        return J, z

    def _direction(self, xb, yb, pw_t):
        B = xb.shape[0]
        if self.window == 1:
            self._blocks.clear()                               # free the stale block before the new Jacobian
        with _dropout_disabled(self.model, not self.dropout):
            J, z = self._jacobian(xb)
        p, c, r = self._loss_terms(z, yb, pw_t)
        g = J.t() @ (r / B)
        theta = self._flat()
        if self.wd:
            g = g + self.wd * theta
        w = p * (1.0 - p)
        if self.curvature == "ggn_weighted":
            w = w * c
        J.mul_(torch.sqrt(w / B).unsqueeze(1))                 # J becomes J~ in place
        self._blocks.append(J.detach())
        del self._blocks[:-self.window]
        m = len(self._blocks)
        M = self._blocks[0] if m == 1 else torch.cat(self._blocks, 0) / (m ** 0.5)

        sd = self.solve_dtype
        n = M.shape[0]
        A = (M @ M.t()).to(sd) + self.lam.to(sd) * torch.eye(n, device=M.device, dtype=sd)
        L, info = torch.linalg.cholesky_ex(A)                  # no host sync, no exception
        u = torch.cholesky_solve((M @ g).to(sd).unsqueeze(1), L).squeeze(1).to(g.dtype)
        x = (g - M.t() @ u) / self.lam
        q = (M @ x).pow(2).sum()                               # x^T F x, closed form
        ok = info == 0
        x = torch.where(ok, x, torch.zeros_like(x))
        q = torch.where(ok, q, torch.zeros_like(q))
        extra = {}
        if self.log_deff:                                      # d_eff = tr(G (G + lam I)^-1)
            Linv = torch.linalg.solve_triangular(L, torch.eye(n, device=M.device, dtype=sd), upper=False)
            extra["d_eff"] = (n - self.lam.to(sd) * Linv.pow(2).sum()).to(g.dtype)
        loss0 = self._loss(z, yb, pw_t) + self._reg(theta)
        return x, g, q, loss0, extra


class ExactCGNGD(_NGDBase):
    """Matrix-free CG on F + lam I. F v = J^T W J v is exact: J v from double backward
    (d/du <J^T u, v>), then one VJP. No epsilon, no extra forwards, no mask mismatch."""

    def __init__(self, model, cg_iters=5, **kw):
        super().__init__(model, **kw)
        self.cg_iters = int(cg_iters)

    def _direction(self, xb, yb, pw_t):
        B = xb.shape[0]
        with _dropout_disabled(self.model, not self.dropout):
            z = self.model(xb).reshape(-1)
        p, c, r = self._loss_terms(z.detach(), yb, pw_t)
        gt_ = torch.autograd.grad(z, self.params, grad_outputs=r / B, retain_graph=True)
        g = torch.cat([t.reshape(-1) for t in gt_])
        theta = self._flat()
        if self.wd:
            g = g + self.wd * theta
        w = p * (1.0 - p)
        if self.curvature == "ggn_weighted":
            w = w * c

        u = torch.zeros_like(z, requires_grad=True)
        jtu = torch.autograd.grad(z, self.params, grad_outputs=u, create_graph=True)   # J^T u, linear in u

        def fvp(v):
            Jv = torch.autograd.grad(jtu, u, grad_outputs=self._unflat(v), retain_graph=True)[0]
            out = torch.autograd.grad(z, self.params, grad_outputs=w * Jv / B, retain_graph=True)
            return torch.cat([t.reshape(-1) for t in out])

        x = torch.zeros_like(g)
        res, pp = g.clone(), g.clone()
        rs = res @ res
        for _ in range(self.cg_iters):                         # fixed count: no sync, no early exit
            Ap = fvp(pp) + self.lam * pp
            alpha = rs / (pp @ Ap).clamp_min(1e-30)
            x = x + alpha * pp
            res = res - alpha * Ap
            rs_new = res @ res
            pp = res + (rs_new / rs.clamp_min(1e-30)) * pp
            rs = rs_new
        q = x @ fvp(x)
        loss0 = self._loss(z.detach(), yb, pw_t) + self._reg(theta)
        return x.detach(), g.detach(), q.detach(), loss0.detach(), {}