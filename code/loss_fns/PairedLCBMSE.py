import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ExpConfigs import ExpConfigs


class Loss(nn.Module):
    """MSE with a smooth proxy for paired EB-LCB: mean benefit, tail harm, variance."""

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.safe_weight = float(getattr(configs, "apn_paired_safe_weight", 0.0))
        self.safe_margin = float(getattr(configs, "apn_paired_safe_margin", 0.0))
        self.cvar_q = min(max(float(getattr(configs, "apn_paired_cvar_q", 1.0)), 1e-4), 1.0)
        self.benefit_weight = float(getattr(configs, "apn_paired_benefit_weight", 0.0))
        self.benefit_margin = float(getattr(configs, "apn_paired_benefit_margin", 0.0))
        self.benefit_temperature = max(float(getattr(configs, "apn_paired_benefit_temperature", 1e-3)), 1e-6)
        self.benefit_cap = max(float(getattr(configs, "apn_paired_benefit_cap", 0.005)), 1e-8)
        self.var_weight = float(getattr(configs, "apn_paired_lcb_var_weight", 0.0))
        self.var_cap = max(float(getattr(configs, "apn_paired_lcb_var_cap", 0.01)), 1e-8)

    @staticmethod
    def _sample_mse(pred, true, mask):
        residual = (pred - true) * mask
        numerator = residual.pow(2).flatten(1).sum(dim=1)
        denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
        return numerator / denom

    def _tail_harm(self, excess):
        harm = excess.clamp_min(0.0).pow(2)
        if self.cvar_q >= 0.999 or harm.numel() <= 1:
            return harm.mean()
        k = max(1, int(torch.ceil(harm.new_tensor(harm.numel() * self.cvar_q)).item()))
        return torch.topk(harm, k=k, largest=True).values.mean()

    def _smooth_reward(self, benefit):
        centered = benefit - self.benefit_margin
        reward = self.benefit_temperature * F.softplus(centered / self.benefit_temperature)
        return self.benefit_cap * torch.tanh(reward / self.benefit_cap)

    def forward(self, pred, true, mask=None, base_pred=None, **kwargs):
        if mask is None:
            mask = torch.ones_like(true, device=true.device)

        residual = (pred - true) * mask
        num_eval = mask.sum()
        mse = residual.pow(2).sum() / (num_eval if num_eval > 0 else 1)
        if base_pred is None:
            return {"loss": mse}

        pred_sample = self._sample_mse(pred, true, mask)
        base_sample = self._sample_mse(base_pred.detach(), true, mask)
        benefit = base_sample - pred_sample
        excess = -benefit + self.safe_margin

        loss = mse
        if self.safe_weight > 0.0:
            loss = loss + self.safe_weight * self._tail_harm(excess)
        if self.benefit_weight > 0.0:
            loss = loss - self.benefit_weight * self._smooth_reward(benefit).mean()
        if self.var_weight > 0.0 and benefit.numel() > 1:
            clipped = benefit.clamp(-self.var_cap, self.var_cap)
            loss = loss + self.var_weight * clipped.var(unbiased=False)
        return {"loss": loss}
