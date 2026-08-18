import importlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.ExpConfigs import ExpConfigs


class MechanismSummaryEncoder(nn.Module):
    """Per-variable observation-mechanism summary used by black-box MCOR."""

    def __init__(self, hid_dim: int, dropout_rate: float = 0.1, use_integral: bool = False,
                 integral_centers: int = 6, integral_width: float = 0.18,
                 reliability_mode: str = "real"):
        super().__init__()
        self.use_integral = bool(use_integral)
        self.integral_centers = max(2, int(integral_centers))
        self.integral_width = float(integral_width)
        self.reliability_mode = str(reliability_mode)
        valid_modes = {"real", "value_only", "reliability_only", "shuffled", "random", "constant"}
        if self.reliability_mode not in valid_modes:
            raise ValueError(f"Unknown mcor_reliability_mode={self.reliability_mode!r}; expected one of {sorted(valid_modes)}")
        base_dim = 10
        integral_dim = 2 * self.integral_centers if self.use_integral else 0
        self.net = nn.Sequential(
            nn.Linear(base_dim + integral_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
        )
        self.norm = nn.LayerNorm(hid_dim)
        self.register_buffer("centers", torch.linspace(0.0, 1.0, self.integral_centers), persistent=False)

    def _time(self, x_mark: Tensor, dtype: torch.dtype) -> Tensor:
        if x_mark is None:
            raise ValueError("MCORWrapper requires x_mark or a base dataloader that supplies observation times.")
        t = x_mark.unsqueeze(-1) if x_mark.dim() == 2 else x_mark[:, :, [0]]
        return torch.nan_to_num(t.to(dtype=dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def _base_features(self, x: Tensor, x_mask: Tensor, x_mark: Tensor):
        dtype = x.dtype
        batch_size, obs_len, _ = x.shape
        t = self._time(x_mark, dtype)
        m = torch.ones_like(x, dtype=dtype) if x_mask is None else torch.nan_to_num(x_mask.to(dtype=dtype), nan=0.0).clamp(0.0, 1.0)
        v = torch.nan_to_num(x.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)

        count = m.sum(dim=1).clamp_min(1.0)
        density = m.mean(dim=1)
        recent_start = max(0, int(0.75 * obs_len))
        recent_density = m[:, recent_start:, :].mean(dim=1)
        count_feature = torch.log1p(count) / math.log1p(max(obs_len, 1))
        mean_t = (m * t).sum(dim=1) / count
        mean_t2 = (m * t.pow(2)).sum(dim=1) / count
        spread_t = (mean_t2 - mean_t.pow(2)).clamp_min(0.0).sqrt()
        mean_v = (v * m).sum(dim=1) / count
        std_v = (((v - mean_v.unsqueeze(1)).pow(2) * m).sum(dim=1) / count).clamp_min(0.0).sqrt()
        recent_w = m * torch.exp(6.0 * t)
        early_w = m * torch.exp(-6.0 * t)
        recent_den = recent_w.sum(dim=1).clamp_min(1e-6)
        early_den = early_w.sum(dim=1).clamp_min(1e-6)
        recent_t = (recent_w * t).sum(dim=1) / recent_den
        recent_v = (recent_w * v).sum(dim=1) / recent_den
        early_v = (early_w * v).sum(dim=1) / early_den
        trend_v = recent_v - early_v
        features = torch.stack([
            density,
            recent_density,
            count_feature,
            mean_t,
            spread_t,
            recent_t,
            mean_v,
            std_v,
            recent_v,
            trend_v,
        ], dim=-1)
        if self.reliability_mode == "value_only":
            mask = features.new_tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).view(1, 1, -1)
            features = features * mask
        elif self.reliability_mode == "reliability_only":
            mask = features.new_tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0]).view(1, 1, -1)
            features = features * mask
        return features, t, m, v

    def _integral_features(self, t: Tensor, m: Tensor, v: Tensor) -> Tensor:
        centers = self.centers.to(device=t.device, dtype=t.dtype).view(1, 1, self.integral_centers, 1)
        width = max(self.integral_width, 1e-3)
        weights = torch.exp(-0.5 * ((t.unsqueeze(2) - centers) / width).pow(2))
        observed_weights = weights * m.unsqueeze(2)
        mass = observed_weights.sum(dim=1).clamp_min(1e-6)
        raw_mass = weights.sum(dim=1).clamp_min(1e-6)
        value_integral = (observed_weights * v.unsqueeze(2)).sum(dim=1) / mass
        coverage_integral = (mass / raw_mass).clamp(0.0, 1.0)
        return torch.cat([
            value_integral.permute(0, 2, 1),
            coverage_integral.permute(0, 2, 1),
        ], dim=-1)

    def forward(self, x: Tensor, x_mask: Tensor, x_mark: Tensor) -> Tensor:
        base_features, t, m, v = self._base_features(x, x_mask, x_mark)
        if self.use_integral:
            integral_features = self._integral_features(t, m, v)
            if self.reliability_mode == "value_only":
                integral_features = torch.zeros_like(integral_features)
            elif self.reliability_mode == "reliability_only":
                value_width = self.integral_centers
                integral_features = torch.cat([
                    torch.zeros_like(integral_features[..., :value_width]),
                    integral_features[..., value_width:],
                ], dim=-1)
            base_features = torch.cat([base_features, integral_features], dim=-1)
        if self.reliability_mode == "constant":
            base_features = torch.zeros_like(base_features)
        elif self.reliability_mode == "random":
            base_features = torch.randn_like(base_features)
        elif self.reliability_mode == "shuffled" and base_features.size(0) > 1:
            base_features = torch.roll(base_features, shifts=1, dims=0)
        return self.norm(self.net(base_features))


class ForecastTimeEmbedding(nn.Module):
    def __init__(self, te_dim: int):
        super().__init__()
        self.te_dim = max(2, int(te_dim))
        self.scale = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, self.te_dim - 1)

    def forward(self, y_mark: Tensor) -> Tensor:
        t = y_mark.unsqueeze(-1) if y_mark.dim() == 2 else y_mark[:, :, [0]]
        t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return torch.cat([self.scale(t), torch.sin(self.periodic(t))], dim=-1)


class BlackBoxResidualBranch(nn.Module):
    def __init__(self, hid_dim: int, te_dim: int, dropout_rate: float, gate_init: float,
                 use_integral: bool, integral_centers: int, integral_width: float,
                 reliability_mode: str):
        super().__init__()
        self.summary = MechanismSummaryEncoder(
            hid_dim=hid_dim,
            dropout_rate=dropout_rate,
            use_integral=use_integral,
            integral_centers=integral_centers,
            integral_width=integral_width,
            reliability_mode=reliability_mode,
        )
        self.residual_net = nn.Sequential(
            nn.Linear(hid_dim + te_dim + 1, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim * 2, 1),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(hid_dim + te_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, float(gate_init))

    def forward(self, x: Tensor, x_mask: Tensor, x_mark: Tensor, te_pred: Tensor, base_raw: Tensor) -> Tensor:
        mechanism = self.summary(x, x_mask, x_mark)
        mechanism_expanded = mechanism.unsqueeze(2).expand(-1, -1, te_pred.size(2), -1)
        residual = self.residual_net(torch.cat([mechanism_expanded, te_pred, base_raw], dim=-1))
        gate = torch.sigmoid(self.gate_net(torch.cat([mechanism_expanded, te_pred], dim=-1)))
        return gate * residual


class BlackBoxAdaptiveResidual(nn.Module):
    def __init__(self, hid_dim: int, te_dim: int, dropout_rate: float,
                 branch_gate_init: float, switch_gate_init: float,
                 warmup_stage: int, warmup_blend: float,
                 integral_centers: int, integral_width: float,
                 force_branch: str = "none", reliability_mode: str = "real"):
        super().__init__()
        self.warmup_stage = int(warmup_stage)
        self.warmup_blend = float(warmup_blend)
        self.force_branch = str(force_branch)
        self.local_branch = BlackBoxResidualBranch(
            hid_dim, te_dim, dropout_rate, branch_gate_init, False, integral_centers, integral_width,
            reliability_mode
        )
        self.integral_branch = BlackBoxResidualBranch(
            hid_dim, te_dim, dropout_rate, branch_gate_init, True, integral_centers, integral_width,
            reliability_mode
        )
        self.switch_summary = MechanismSummaryEncoder(
            hid_dim=hid_dim,
            dropout_rate=dropout_rate,
            use_integral=True,
            integral_centers=integral_centers,
            integral_width=integral_width,
            reliability_mode=reliability_mode,
        )
        self.switch_net = nn.Sequential(
            nn.Linear(hid_dim + te_dim + 2, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, 1),
        )
        nn.init.zeros_(self.switch_net[-1].weight)
        nn.init.constant_(self.switch_net[-1].bias, float(switch_gate_init))
        self.latest_integral_gate = None
        self.latest_gate_logits = None
        self.latest_local_residual = None
        self.latest_integral_residual = None
        self.latest_boundary_penalty = None

    def forward(self, x: Tensor, x_mask: Tensor, x_mark: Tensor, te_pred: Tensor,
                base_raw: Tensor, train_stage=None) -> Tensor:
        local_residual = self.local_branch(x, x_mask, x_mark, te_pred, base_raw)
        integral_residual = self.integral_branch(x, x_mask, x_mark, te_pred, base_raw)
        mechanism = self.switch_summary(x, x_mask, x_mark)
        mechanism_expanded = mechanism.unsqueeze(2).expand(-1, -1, te_pred.size(2), -1)
        disagreement = (local_residual - integral_residual).abs()
        gate_logits = self.switch_net(torch.cat([mechanism_expanded, te_pred, base_raw, disagreement], dim=-1))
        learned_gate = torch.sigmoid(gate_logits)

        if self.force_branch == "local":
            integral_gate = torch.zeros_like(learned_gate)
            self.latest_boundary_penalty = learned_gate.new_zeros(())
        elif self.force_branch == "integral":
            integral_gate = torch.ones_like(learned_gate)
            self.latest_boundary_penalty = learned_gate.new_zeros(())
        elif self.force_branch == "mix":
            integral_gate = torch.full_like(learned_gate, self.warmup_blend)
            self.latest_boundary_penalty = learned_gate.new_zeros(())
        elif train_stage is not None and self.warmup_stage > 0 and int(train_stage) <= self.warmup_stage:
            integral_gate = torch.full_like(learned_gate, self.warmup_blend)
            self.latest_boundary_penalty = learned_gate.new_zeros(())
        else:
            integral_gate = learned_gate
            self.latest_boundary_penalty = (integral_gate * (1.0 - integral_gate)).mean()

        residual = (1.0 - integral_gate) * local_residual + integral_gate * integral_residual
        self.latest_integral_gate = integral_gate.detach()
        self.latest_gate_logits = gate_logits
        self.latest_local_residual = local_residual
        self.latest_integral_residual = integral_residual
        return residual


class Model(nn.Module):
    """Black-box MCOR wrapper: base forecast + mechanism-conditioned residual operator."""

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        base_name = getattr(configs, "mcor_base_model_name", "KAFNet")
        if base_name == "MCORWrapper":
            raise ValueError("mcor_base_model_name cannot be MCORWrapper.")
        base_module = importlib.import_module("models." + base_name)
        self.base_model = base_module.Model(configs)
        self.base_model_name = base_name
        self.freeze_base = bool(int(getattr(configs, "mcor_freeze_base", 0)))
        if self.freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad_(False)

        hid_dim = max(8, int(configs.d_model))
        te_dim = max(2, int(getattr(configs, "apn_te_dim", 8)))
        dropout_rate = float(getattr(configs, "dropout", 0.0))
        self.time_embedding = ForecastTimeEmbedding(te_dim)
        self.residual_head = BlackBoxAdaptiveResidual(
            hid_dim=hid_dim,
            te_dim=te_dim,
            dropout_rate=dropout_rate,
            branch_gate_init=float(getattr(configs, "apn_mcr_gate_init", -1.4)),
            switch_gate_init=float(getattr(configs, "apn_mcg_gate_init", -2.0)),
            warmup_stage=int(getattr(configs, "apn_mcg_warmup_stage", 0)),
            warmup_blend=float(getattr(configs, "apn_mcg_warmup_blend", 0.5)),
            integral_centers=int(getattr(configs, "apn_mcr_integral_centers", 6)),
            integral_width=float(getattr(configs, "apn_mcr_integral_width", 0.18)),
            force_branch=str(getattr(configs, "mcor_force_branch", "none")),
            reliability_mode=str(getattr(configs, "mcor_reliability_mode", "real")),
        )
        self.residual_l2 = float(getattr(configs, "apn_mcr_residual_l2", 0.0))
        self.boundary_weight = float(getattr(configs, "apn_mcg_boundary_weight", 0.0))
        self.supervised_weight = float(getattr(configs, "apn_mcg_supervised_weight", 0.0))
        self.supervised_temperature = float(getattr(configs, "apn_mcg_supervised_temperature", 0.10))
        self.supervised_margin = float(getattr(configs, "apn_mcg_supervised_margin", 0.0))
        self.supervised_min_confidence = float(getattr(configs, "apn_mcg_supervised_min_confidence", 0.0))

    def load_state_dict(self, state_dict, strict: bool = True):
        """Allow warm-starting from either MCORWrapper or plain base checkpoints."""
        if state_dict and not any(str(key).startswith("base_model.") for key in state_dict.keys()):
            base_keys = set(self.base_model.state_dict().keys())
            overlap = sum(1 for key in state_dict.keys() if key in base_keys)
            if overlap > 0:
                state_dict = {
                    (f"base_model.{key}" if key in base_keys else key): value
                    for key, value in state_dict.items()
                }
                strict = False
        return super().load_state_dict(state_dict, strict=strict)

    def _default_y_mark(self, base_pred: Tensor) -> Tensor:
        length = base_pred.size(1)
        denom = max(length - 1, 1)
        t = torch.arange(length, dtype=base_pred.dtype, device=base_pred.device) / denom
        return t.view(1, length, 1).expand(base_pred.size(0), -1, -1)

    def _align_history(self, x: Tensor, x_mask: Tensor, out_dim: int):
        if x.size(-1) == out_dim:
            return x, x_mask
        return x[:, :, -out_dim:], None if x_mask is None else x_mask[:, :, -out_dim:]

    def _restore_compressed_forecast(
        self,
        base_pred: Tensor,
        x: Tensor,
        x_mask: Tensor,
        y: Tensor,
        y_mask: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if base_pred.dim() != 2 or y is None:
            return base_pred, y, y_mask
        if not hasattr(self.base_model, "unpad_and_reshape"):
            return base_pred, y, y_mask
        x_mask_full = torch.ones_like(x) if x_mask is None else x_mask
        y_mask_full = torch.ones_like(y) if y_mask is None else y_mask
        x_padding = torch.zeros_like(y)
        original_shape = torch.cat([x, x_padding], dim=1).shape
        xy_mask = torch.cat([x_mask_full, y_mask_full], dim=1)
        restored = self.base_model.unpad_and_reshape(base_pred, xy_mask, original_shape)
        pred = restored[:, -y.shape[1]:, :]
        return pred, y, y_mask_full

    def _supervised_gate_loss(self, pred_local: Tensor, pred_integral: Tensor,
                              true: Tensor, mask: Tensor) -> Tensor | None:
        if self.supervised_weight <= 0.0 or true is None:
            return None
        gate_logits = self.residual_head.latest_gate_logits
        if gate_logits is None:
            return None
        logits = gate_logits.squeeze(-1).permute(0, 2, 1)
        if mask is None:
            mask = torch.ones_like(true)
        local_loss = (pred_local - true).pow(2)
        integral_loss = (pred_integral - true).pow(2)
        scale = (local_loss + integral_loss).detach().clamp_min(1e-6)
        target = torch.sigmoid(
            ((local_loss - integral_loss).detach() - self.supervised_margin)
            / (max(self.supervised_temperature, 1e-4) * scale + 1e-6)
        )
        confidence = (target - 0.5).abs() * 2.0
        weight = confidence * mask.to(dtype=true.dtype)
        if self.supervised_min_confidence > 0.0:
            weight = weight * (confidence >= self.supervised_min_confidence).to(dtype=weight.dtype)
        denom = weight.sum().clamp_min(1.0)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return bce.mul(weight).sum() / denom * self.supervised_weight

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor = None,
        x_mask: Tensor = None,
        y: Tensor = None,
        y_mark: Tensor = None,
        y_mask: Tensor = None,
        exp_stage: str = "train",
        train_stage=None,
        **kwargs,
    ) -> dict:
        if self.freeze_base:
            was_training = self.base_model.training
            self.base_model.eval()
            with torch.no_grad():
                base_outputs = self.base_model(
                    x=x,
                    x_mark=x_mark,
                    x_mask=x_mask,
                    y=y,
                    y_mark=y_mark,
                    y_mask=y_mask,
                    exp_stage=exp_stage,
                    train_stage=train_stage,
                    **kwargs,
                )
            if was_training:
                self.base_model.train()
        else:
            base_outputs = self.base_model(
                x=x,
                x_mark=x_mark,
                x_mask=x_mask,
                y=y,
                y_mark=y_mark,
                y_mask=y_mask,
                exp_stage=exp_stage,
                train_stage=train_stage,
                **kwargs,
            )
        base_pred = base_outputs["pred"]
        true = base_outputs.get("true", y)
        mask = base_outputs.get("mask", y_mask)
        base_pred, true, mask = self._restore_compressed_forecast(base_pred, x, x_mask, y, y_mask)
        if y_mark is None:
            y_mark = self._default_y_mark(base_pred)

        x_hist, x_mask_hist = self._align_history(x, x_mask, base_pred.size(-1))
        te = self.time_embedding(y_mark).unsqueeze(1).expand(-1, base_pred.size(-1), -1, -1)
        base_raw = base_pred.permute(0, 2, 1).unsqueeze(-1)
        residual = self.residual_head(x_hist, x_mask_hist, x_mark, te, base_raw, train_stage=train_stage)
        pred = (base_raw + residual).squeeze(-1).permute(0, 2, 1)

        outputs = dict(base_outputs)
        outputs["pred"] = pred
        outputs["true"] = true
        outputs["mask"] = mask
        outputs["base_pred"] = base_pred

        local_residual = self.residual_head.latest_local_residual
        integral_residual = self.residual_head.latest_integral_residual
        if local_residual is not None and integral_residual is not None:
            local_pred = (base_raw + local_residual).squeeze(-1).permute(0, 2, 1)
            integral_pred = (base_raw + integral_residual).squeeze(-1).permute(0, 2, 1)
            if bool(getattr(self.configs, "apn_emit_mcg_arrays", 0)):
                outputs["mcg_integral_gate"] = self.residual_head.latest_integral_gate.squeeze(-1).permute(0, 2, 1)
                outputs["mcg_local_pred"] = local_pred
                outputs["mcg_integral_pred"] = integral_pred
        else:
            local_pred = None
            integral_pred = None

        aux_loss = base_outputs.get("aux_loss")
        if self.training:
            if self.residual_l2 > 0.0:
                residual_aux = residual.pow(2).mean() * self.residual_l2
                aux_loss = residual_aux if aux_loss is None else aux_loss + residual_aux
            if self.boundary_weight > 0.0 and self.residual_head.latest_boundary_penalty is not None:
                boundary_aux = self.residual_head.latest_boundary_penalty * self.boundary_weight
                aux_loss = boundary_aux if aux_loss is None else aux_loss + boundary_aux
            warmup_stage = int(getattr(self.configs, "apn_mcg_warmup_stage", 0))
            if local_pred is not None and integral_pred is not None and (train_stage is None or int(train_stage) > warmup_stage):
                supervised_aux = self._supervised_gate_loss(local_pred, integral_pred, true, mask)
                if supervised_aux is not None:
                    aux_loss = supervised_aux if aux_loss is None else aux_loss + supervised_aux
        if aux_loss is not None:
            outputs["aux_loss"] = aux_loss
        return outputs
