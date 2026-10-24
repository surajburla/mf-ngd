import torch
import torch.nn as nn
from torch.optim import Optimizer

class KFAC(Optimizer):
    r"""
    Kronecker-Factored Approximate Curvature (K-FAC) Optimizer for Linear and Conv1d layers.
    Analytically inverts Kronecker factors (A^-1 x S^-1) in non-iterative steps.
    """
    def __init__(self, model, lr=1e-3, damping=1e-2, invert_freq=10):
        defaults = dict(lr=lr, damping=damping, invert_freq=invert_freq)
        super(KFAC, self).__init__(model.parameters(), defaults)
        self.model = model
        self.known_modules = []
        self.a_cov = {}
        self.s_cov = {}
        self.inv_a = {}
        self.inv_s = {}
        self.steps = 0
        self._register_hooks()

    def _register_hooks(self):
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                self.known_modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_full_backward_hook(self._save_grad)

    def _save_input(self, module, input):
        if module.training:
            a = input[0].data
            if isinstance(module, nn.Linear):
                if a.dim() == 2:
                    batch = a.size(0)
                    a_ext = torch.cat([a, torch.ones((batch, 1), device=a.device)], dim=1)
                    cov = (a_ext.t() @ a_ext) / batch
                    self.a_cov[module] = cov
            elif isinstance(module, nn.Conv1d):
                batch, channels, length = a.shape
                kernel_size = module.kernel_size[0]
                unfold = torch.nn.functional.unfold(a.unsqueeze(2), (1, kernel_size), padding=(0, module.padding[0]), stride=(1, module.stride[0]))
                a_flat = unfold.transpose(1, 2).reshape(-1, channels * kernel_size)
                a_ext = torch.cat([a_flat, torch.ones((a_flat.size(0), 1), device=a.device)], dim=1)
                cov = (a_ext.t() @ a_ext) / a_flat.size(0)
                self.a_cov[module] = cov

    def _save_grad(self, module, grad_input, grad_output):
        if module.training:
            s = grad_output[0].data
            if isinstance(module, nn.Linear):
                batch = s.size(0)
                cov = (s.t() @ s) * batch
                self.s_cov[module] = cov
            elif isinstance(module, nn.Conv1d):
                s_flat = s.transpose(1, 2).reshape(-1, s.size(1))
                cov = (s_flat.t() @ s_flat) * s_flat.size(0)
                self.s_cov[module] = cov

    def _update_inverses(self):
        damping = self.param_groups[0]['damping']
        for module in self.known_modules:
            if module in self.a_cov and module in self.s_cov:
                a = self.a_cov[module]
                s = self.s_cov[module]
                
                a_damp = a + damping * torch.eye(a.size(0), device=a.device)
                s_damp = s + damping * torch.eye(s.size(0), device=s.device)

                self.inv_a[module] = torch.linalg.pinv(a_damp)
                self.inv_s[module] = torch.linalg.pinv(s_damp)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        lr = group['lr']
        invert_freq = group['invert_freq']

        if self.steps % invert_freq == 0:
            self._update_inverses()

        for module in self.known_modules:
            if module in self.inv_a and module in self.inv_s:
                inv_a = self.inv_a[module]
                inv_s = self.inv_s[module]

                if isinstance(module, nn.Linear):
                    w_grad = module.weight.grad
                    b_grad = module.bias.grad if module.bias is not None else None
                    if b_grad is not None:
                        grad = torch.cat([w_grad, b_grad.unsqueeze(1)], dim=1)
                    else:
                        grad = w_grad

                    natural_grad = inv_s @ grad @ inv_a

                    if b_grad is not None:
                        module.weight.grad.copy_(natural_grad[:, :-1])
                        module.bias.grad.copy_(natural_grad[:, -1])
                    else:
                        module.weight.grad.copy_(natural_grad)

                elif isinstance(module, nn.Conv1d):
                    w_grad = module.weight.grad
                    out_c, in_c, k = w_grad.shape
                    w_grad_flat = w_grad.reshape(out_c, in_c * k)
                    
                    b_grad = module.bias.grad if module.bias is not None else None
                    if b_grad is not None:
                        grad = torch.cat([w_grad_flat, b_grad.unsqueeze(1)], dim=1)
                    else:
                        grad = w_grad_flat

                    natural_grad = inv_s @ grad @ inv_a

                    if b_grad is not None:
                        module.weight.grad.copy_(natural_grad[:, :-1].reshape(out_c, in_c, k))
                        module.bias.grad.copy_(natural_grad[:, -1])
                    else:
                        module.weight.grad.copy_(natural_grad.reshape(out_c, in_c, k))

        # Apply parameter step
        for p in self.model.parameters():
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-lr)

        self.steps += 1
        return loss
