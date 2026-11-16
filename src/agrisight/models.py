"""Model construction and checkpoint I/O (backbones come from timm)."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


def build_model(name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.0) -> nn.Module:
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


def model_norm(model: nn.Module) -> tuple[list[float], list[float]]:
    cfg = timm.data.resolve_data_config({}, model=model)
    return list(cfg["mean"]), list(cfg["std"])


def feature_dim(model: nn.Module) -> int:
    return getattr(model, "head_hidden_size", None) or model.num_features


def features_and_logits(model: nn.Module, x: torch.Tensor):
    """Pooled features + logits for any timm classifier (used by DANN)."""
    feats = model.forward_head(model.forward_features(x), pre_logits=True)
    return feats, model.get_classifier()(feats)


def enable_mc_dropout(model: nn.Module) -> None:
    """Eval mode everywhere except dropout layers (Monte Carlo dropout)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def save_checkpoint(path, model: nn.Module, meta: dict) -> None:
    torch.save({**meta, "state_dict": model.state_dict()}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False, drop_rate=ckpt.get("drop_rate", 0.0))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
