"""Temperature scaling and Monte Carlo dropout."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .models import enable_mc_dropout


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Find T minimising NLL of softmax(logits / T) on a held-out set."""
    logits = logits.detach().float().cpu()
    labels = labels.detach().long().cpu()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


@torch.no_grad()
def mc_dropout_probs(model, x: torch.Tensor, n: int = 20):
    """Mean probabilities and predictive entropy over n stochastic passes."""
    enable_mc_dropout(model)
    probs = torch.stack([F.softmax(model(x), dim=-1) for _ in range(n)])
    model.eval()
    mean = probs.mean(0)
    entropy = -(mean * mean.clamp_min(1e-12).log()).sum(-1)
    return mean, entropy
