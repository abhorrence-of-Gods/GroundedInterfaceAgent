"""Common evaluation metrics for GIA training/validation."""

import torch

def action_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean-squared error over the raw 4-dim action vector."""
    return torch.mean((pred - target) ** 2).item()

def click_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Binary accuracy of the click flag (index 2).

    Args:
        pred: (B, 4) predicted action tensor.
        target: (B, 4) ground-truth action tensor.
        threshold: value above which click is considered 1.
    """
    pred_click = (pred[:, 2] > threshold).float()
    tgt_click = (target[:, 2] > 0.5).float()
    correct = (pred_click == tgt_click).sum().item()
    return correct / pred.shape[0] 