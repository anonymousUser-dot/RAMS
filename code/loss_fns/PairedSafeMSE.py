import torch
import torch.nn as nn

from utils.ExpConfigs import ExpConfigs


class Loss(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super(Loss, self).__init__()
        self.safe_weight = float(getattr(configs, "apn_paired_safe_weight", 0.0))
        self.safe_margin = float(getattr(configs, "apn_paired_safe_margin", 0.0))

    def forward(self, pred, true, mask=None, base_pred=None, **kwargs):
        if mask is None:
            mask = torch.ones_like(true, device=true.device)

        residual = (pred - true) * mask
        num_eval = mask.sum()
        mse = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        if base_pred is None or self.safe_weight <= 0.0:
            return {"loss": mse}

        base_residual = (base_pred.detach() - true) * mask
        pred_sample = residual.pow(2).flatten(1).sum(dim=1)
        base_sample = base_residual.pow(2).flatten(1).sum(dim=1)
        denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
        excess = pred_sample / denom - base_sample / denom + self.safe_margin
        safe_penalty = torch.relu(excess).pow(2).mean()
        return {"loss": mse + self.safe_weight * safe_penalty}
