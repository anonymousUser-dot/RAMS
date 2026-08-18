import torch
import torch.nn as nn

from utils.ExpConfigs import ExpConfigs


class Loss(nn.Module):
    """Joint squared/absolute forecasting loss for MSE-MAE aligned training."""

    def __init__(self, configs: ExpConfigs):
        super(Loss, self).__init__()
        self.mae_weight = float(getattr(configs, "mae_weight", 0.20))

    def forward(self, pred, true, mask=None, **kwargs):
        if mask is None:
            mask = torch.ones_like(true, device=true.device)

        residual = (pred - true) * mask
        num_eval = mask.sum()
        denom = num_eval if num_eval > 0 else 1
        mse = (residual ** 2).sum() / denom
        mae = torch.abs(residual).sum() / denom

        return {
            "loss": mse + self.mae_weight * mae
        }
