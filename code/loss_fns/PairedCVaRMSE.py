import torch
import torch.nn as nn

from utils.ExpConfigs import ExpConfigs


class Loss(nn.Module):
    """MSE plus upper-tail paired excess penalty relative to the hard no-op."""

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.safe_weight = float(getattr(configs, "apn_paired_safe_weight", 0.0))
        self.safe_margin = float(getattr(configs, "apn_paired_safe_margin", 0.0))
        self.cvar_q = min(max(float(getattr(configs, "apn_paired_cvar_q", 1.0)), 1e-4), 1.0)

    @staticmethod
    def _sample_mse(pred, true, mask):
        residual = (pred - true) * mask
        numerator = residual.pow(2).flatten(1).sum(dim=1)
        denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
        return numerator / denom

    def forward(self, pred, true, mask=None, base_pred=None, **kwargs):
        if mask is None:
            mask = torch.ones_like(true, device=true.device)

        residual = (pred - true) * mask
        num_eval = mask.sum()
        mse = residual.pow(2).sum() / (num_eval if num_eval > 0 else 1)
        if base_pred is None or self.safe_weight <= 0.0:
            return {"loss": mse}

        pred_sample = self._sample_mse(pred, true, mask)
        base_sample = self._sample_mse(base_pred.detach(), true, mask)
        excess = (pred_sample - base_sample + self.safe_margin).clamp_min(0.0)
        if self.cvar_q >= 0.999:
            tail_penalty = excess.mean()
        else:
            k = max(1, int(torch.ceil(excess.new_tensor(excess.numel() * self.cvar_q)).item()))
            tail_penalty = torch.topk(excess, k=k, largest=True).values.mean()
        return {"loss": mse + self.safe_weight * tail_penalty}
