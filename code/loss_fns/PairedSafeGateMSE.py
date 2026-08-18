import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ExpConfigs import ExpConfigs


class Loss(nn.Module):
    """MSE with paired-safe candidate shaping and online selector supervision."""

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.safe_weight = float(getattr(configs, "apn_paired_safe_weight", 0.0))
        self.safe_margin = float(getattr(configs, "apn_paired_safe_margin", 0.0))
        self.gate_harm_weight = float(getattr(configs, "apn_gate_harm_weight", 0.0))
        self.gate_bce_weight = float(getattr(configs, "apn_gate_bce_weight", 0.0))
        self.gate_target_margin = float(getattr(configs, "apn_gate_target_margin", 0.0))

    @staticmethod
    def _sample_mse(pred, true, mask):
        residual = (pred - true) * mask
        numerator = residual.pow(2).flatten(1).sum(dim=1)
        denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
        return numerator / denom

    @staticmethod
    def _sample_gate(selector_gate):
        gate = selector_gate.float().flatten(1)
        return gate.max(dim=1).values.clamp(1e-5, 1.0 - 1e-5)

    def forward(self, pred, true, mask=None, base_pred=None, selector_gate=None, **kwargs):
        if mask is None:
            mask = torch.ones_like(true, device=true.device)

        residual = (pred - true) * mask
        num_eval = mask.sum()
        mse = residual.pow(2).sum() / (num_eval if num_eval > 0 else 1)
        if base_pred is None:
            return {"loss": mse}

        pred_sample = self._sample_mse(pred, true, mask)
        base_sample = self._sample_mse(base_pred.detach(), true, mask)
        excess = pred_sample - base_sample + self.safe_margin
        loss = mse

        if self.safe_weight > 0.0:
            loss = loss + self.safe_weight * F.relu(excess).pow(2).mean()

        if selector_gate is not None and (self.gate_harm_weight > 0.0 or self.gate_bce_weight > 0.0):
            gate_score = self._sample_gate(selector_gate)
            if self.gate_harm_weight > 0.0:
                loss = loss + self.gate_harm_weight * (gate_score * F.relu(excess.detach())).mean()
            if self.gate_bce_weight > 0.0:
                benefit = (base_sample - pred_sample).detach()
                target = (benefit > self.gate_target_margin).float()
                bce = -(target * gate_score.log() + (1.0 - target) * (1.0 - gate_score).log()).mean()
                loss = loss + self.gate_bce_weight * bce

        return {"loss": loss}
