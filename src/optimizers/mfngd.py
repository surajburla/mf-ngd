import torch
from torch.optim import Optimizer

class MFNGD(Optimizer):
    r"""
    Matrix-Free Natural Gradient Descent with:
    1. Exact KL-Divergence Trust-Region Step Bounds.
    2. Adaptive Levenberg-Marquardt Damping Scheduling.
    """
    def __init__(self, params, lr=1e-3, cg_iters=5, damping=1e-2, max_kl=1e-2):
        defaults = dict(lr=lr, cg_iters=cg_iters, damping=damping, max_kl=max_kl)
        super(MFNGD, self).__init__(params, defaults)
        self.damping = damping

    def _compute_fvp(self, model, inputs, v, damping, eps=1e-4):
        params = [p for p in model.parameters() if p.requires_grad]
        batch_size = inputs.size(0)

        # 1. Finite-Difference R-Operator for J v (evaluations do not require autograd graph)
        offset = 0
        for p in params:
            numel = p.numel()
            v_p = v[offset:offset+numel].view_as(p)
            p.data.add_(v_p, alpha=eps)
            offset += numel

        with torch.no_grad():
            logits_plus = model(inputs).squeeze(-1)

        offset = 0
        for p in params:
            numel = p.numel()
            v_p = v[offset:offset+numel].view_as(p)
            p.data.add_(v_p, alpha=-2.0 * eps)
            offset += numel

        with torch.no_grad():
            logits_minus = model(inputs).squeeze(-1)

        # Restore original parameter values
        offset = 0
        for p in params:
            numel = p.numel()
            v_p = v[offset:offset+numel].view_as(p)
            p.data.add_(v_p, alpha=eps)
            offset += numel

        jv = (logits_plus - logits_minus) / (2.0 * eps)

        # 2. Vector-Jacobian Product (J^T w) with explicit enable_grad context
        with torch.enable_grad():
            logits = model(inputs).squeeze(-1)
            probs = torch.sigmoid(logits)
            d_diag = probs * (1.0 - probs) + 1e-6
            w = (d_diag.detach() * jv) / batch_size

            fvp_grads = torch.autograd.grad(torch.sum(logits * w), params, retain_graph=False)

        flat_fvp = torch.cat([g.contiguous().view(-1) for g in fvp_grads])

        return flat_fvp + damping * v

    def _conjugate_gradient(self, fvp_fn, b, cg_iters):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rs_old = torch.dot(r, r)

        if torch.sqrt(rs_old) < 1e-8:
            return x

        for _ in range(cg_iters):
            Ap = fvp_fn(p)
            p_Ap = torch.dot(p, Ap)

            if p_Ap <= 0:
                break

            alpha = rs_old / p_Ap
            x += alpha * p
            r -= alpha * Ap
            rs_new = torch.dot(r, r)

            if torch.sqrt(rs_new) < 1e-8:
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x

    def step(self, closure):
        if closure is None:
            raise RuntimeError("MFNGD requires closure containing (loss, model, inputs, targets, criterion).")

        with torch.enable_grad():
            loss_init, model, inputs, targets, criterion = closure()
            loss_init.backward()

        params = [p for p in model.parameters() if p.requires_grad]
        grads = [p.grad for p in params if p.grad is not None]

        if len(grads) == 0:
            return loss_init

        flat_g = torch.cat([g.view(-1) for g in grads])
        group = self.param_groups[0]
        lr = group['lr']
        max_kl = group['max_kl']

        def fvp_fn(v):
            return self._compute_fvp(model, inputs, v, self.damping)

        # Solve (F + damping * I) x = g
        nat_grad = self._conjugate_gradient(fvp_fn, flat_g, group['cg_iters'])

        # Compute quadratic KL distance: 0.5 * lr^2 * x^T F x
        f_x = fvp_fn(nat_grad) - self.damping * nat_grad
        kl_dist = 0.5 * (lr ** 2) * torch.dot(nat_grad, f_x).item()

        # Exact KL-Divergence Trust-Region Step Scaling
        if kl_dist > max_kl and kl_dist > 0:
            scale = (max_kl / kl_dist) ** 0.5
        else:
            scale = 1.0

        step_vec = -lr * scale * nat_grad

        # Apply candidate parameter update
        offset = 0
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    numel = p.numel()
                    p.add_(step_vec[offset:offset + numel].view_as(p))
                    offset += numel

        # Levenberg-Marquardt Reduction Ratio
        with torch.no_grad():
            logits_new = model(inputs)
            loss_new = criterion(logits_new, targets)
            actual_red = (loss_init - loss_new).item()
            pred_red = -torch.dot(flat_g, step_vec).item() - 0.5 * torch.dot(step_vec, fvp_fn(-step_vec)).item()
            
            rho = actual_red / (pred_red + 1e-8) if pred_red > 0 else 0.0

            if rho < 0.25:
                self.damping = min(self.damping * 2.0, 1.0)
                # Rollback parameter update if loss increased
                if actual_red < 0:
                    offset = 0
                    for p in params:
                        if p.grad is not None:
                            numel = p.numel()
                            p.sub_(step_vec[offset:offset + numel].view_as(p))
                            offset += numel
            elif rho > 0.75:
                self.damping = max(self.damping / 2.0, 1e-4)

        return loss_init
