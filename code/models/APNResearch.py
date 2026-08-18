# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class AttentionPatchAggregation(nn.Module):
    def __init__(self, N, P, S, te_dim, hid_dim, history, dropout_rate=0.1):
        super().__init__()
        self.N = N
        self.P = P
        self.S = max(history / P, 1e-6) if S is None else S
        self.history = history
        self.hid_dim = hid_dim
        self.te_dim = te_dim
        self.feature_dim = 1 + te_dim
        self.delta_left_params = nn.Parameter(torch.zeros(N, P))
        self.raw_log_width_params = nn.Parameter(torch.full((N, P), math.log(self.S)))
        self.tau_params = nn.Parameter(torch.zeros(N))
        self.register_buffer(
            "patch_centers_base",
            torch.linspace(self.S / 2, self.history - self.S / 2, self.P),
            persistent=False,
        )
        self.projection_layer = nn.Linear(self.feature_dim, self.hid_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim * 2, hid_dim)
        )
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, t_stacked, x_with_te, mask_stacked):
        current_device = t_stacked.device
        B_N, L_obs_pad, _ = t_stacked.shape
        B = B_N // self.N
        patch_centers = self.patch_centers_base.to(device=current_device, dtype=t_stacked.dtype)
        base_left_boundaries = (patch_centers - self.S / 2).unsqueeze(0)
        t_left_n_p = base_left_boundaries + self.delta_left_params
        width_learned_n_p = torch.exp(self.raw_log_width_params) + 1e-6
        t_right_n_p = t_left_n_p + width_learned_n_p
        current_variable_taus = F.softplus(self.tau_params).unsqueeze(-1) + 1e-6
        t_left_b_n = t_left_n_p.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        t_right_b_n = t_right_n_p.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        taus_b_n = current_variable_taus.unsqueeze(0).expand(B, -1, -1).reshape(B_N, 1).unsqueeze(-1)
        t_raw_b_n = t_stacked.transpose(-1, -2)
        weights_raw = torch.sigmoid((t_right_b_n - t_raw_b_n) / taus_b_n) * \
                      torch.sigmoid((t_raw_b_n - t_left_b_n) / taus_b_n)
        mask_b_n = mask_stacked.transpose(-1, -2)
        temporal_weights = weights_raw * mask_b_n
        sum_weights = temporal_weights.sum(dim=-1, keepdim=True) + 1e-9
        weighted_features_sum = torch.bmm(temporal_weights, x_with_te)
        h_patches_avg = weighted_features_sum / sum_weights
        h_patches_proj = self.projection_layer(h_patches_avg)
        h_patches = self.norm(h_patches_proj + self.ffn(h_patches_proj))
        return h_patches


class DensityAwarePatchAggregation(AttentionPatchAggregation):
    def __init__(self, N, P, S, te_dim, hid_dim, history, dropout_rate=0.1):
        super().__init__(N, P, S, te_dim, hid_dim, history, dropout_rate)
        self.density_gate = nn.Sequential(
            nn.Linear(1, hid_dim),
            nn.Sigmoid()
        )

    def forward(self, t_stacked, x_with_te, mask_stacked):
        current_device = t_stacked.device
        B_N, _, _ = t_stacked.shape
        B = B_N // self.N
        patch_centers = self.patch_centers_base.to(device=current_device, dtype=t_stacked.dtype)
        base_left_boundaries = (patch_centers - self.S / 2).unsqueeze(0)
        t_left_n_p = base_left_boundaries + self.delta_left_params
        width_learned_n_p = torch.exp(self.raw_log_width_params) + 1e-6
        t_right_n_p = t_left_n_p + width_learned_n_p
        current_variable_taus = F.softplus(self.tau_params).unsqueeze(-1) + 1e-6
        t_left_b_n = t_left_n_p.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        t_right_b_n = t_right_n_p.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        taus_b_n = current_variable_taus.unsqueeze(0).expand(B, -1, -1).reshape(B_N, 1).unsqueeze(-1)
        t_raw_b_n = t_stacked.transpose(-1, -2)
        weights_raw = torch.sigmoid((t_right_b_n - t_raw_b_n) / taus_b_n) * \
                      torch.sigmoid((t_raw_b_n - t_left_b_n) / taus_b_n)
        temporal_weights = weights_raw * mask_stacked.transpose(-1, -2)
        sum_weights = temporal_weights.sum(dim=-1, keepdim=True) + 1e-9
        h_patches_avg = torch.bmm(temporal_weights, x_with_te) / sum_weights
        h_patches_proj = self.projection_layer(h_patches_avg)
        density = torch.log1p(sum_weights)
        h_patches_proj = h_patches_proj * self.density_gate(density)
        return self.norm(h_patches_proj + self.ffn(h_patches_proj))


class MechanismAwarePatchAggregation(AttentionPatchAggregation):
    def __init__(self, N, P, S, te_dim, hid_dim, history, dropout_rate=0.1):
        super().__init__(N, P, S, te_dim, hid_dim, history, dropout_rate)
        self.feature_dim = 1 + te_dim + 4
        self.projection_layer = nn.Linear(self.feature_dim, self.hid_dim)
        self.mechanism_gate = nn.Sequential(
            nn.Linear(4, hid_dim),
            nn.Sigmoid()
        )

    def forward(self, t_stacked, x_with_te, mask_stacked):
        current_device = t_stacked.device
        B_N, _, _ = t_stacked.shape
        B = B_N // self.N
        patch_centers = self.patch_centers_base.to(device=current_device, dtype=t_stacked.dtype)
        base_left_boundaries = (patch_centers - self.S / 2).unsqueeze(0)
        t_left_n_p = base_left_boundaries + self.delta_left_params
        width_learned_n_p = torch.exp(self.raw_log_width_params) + 1e-6
        t_right_n_p = t_left_n_p + width_learned_n_p
        current_variable_taus = F.softplus(self.tau_params).unsqueeze(-1) + 1e-6
        t_left_b_n = t_left_n_p.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        t_right_b_n = t_right_n_p.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        taus_b_n = current_variable_taus.unsqueeze(0).expand(B, -1, -1).reshape(B_N, 1).unsqueeze(-1)
        t_raw_b_n = t_stacked.transpose(-1, -2)
        weights_raw = torch.sigmoid((t_right_b_n - t_raw_b_n) / taus_b_n) * \
                      torch.sigmoid((t_raw_b_n - t_left_b_n) / taus_b_n)
        mask_b_n = mask_stacked.transpose(-1, -2)
        temporal_weights = weights_raw * mask_b_n
        sum_weights = temporal_weights.sum(dim=-1, keepdim=True) + 1e-9

        h_patches_avg = torch.bmm(temporal_weights, x_with_te) / sum_weights
        mean_time = torch.bmm(temporal_weights, t_stacked) / sum_weights
        second_time = torch.bmm(temporal_weights, t_stacked.pow(2)) / sum_weights
        time_spread = torch.sqrt((second_time - mean_time.pow(2)).clamp_min(0.0) + 1e-9)
        raw_mass = weights_raw.sum(dim=-1, keepdim=True) + 1e-9
        coverage = (temporal_weights.sum(dim=-1, keepdim=True) / raw_mass).clamp(0.0, 1.0)
        mechanism = torch.cat([
            torch.log1p(sum_weights),
            coverage,
            mean_time.clamp(0.0, 1.0),
            time_spread.clamp(0.0, 1.0),
        ], dim=-1)

        h_patches_proj = self.projection_layer(torch.cat([h_patches_avg, mechanism], dim=-1))
        h_patches_proj = h_patches_proj * self.mechanism_gate(mechanism)
        return self.norm(h_patches_proj + self.ffn(h_patches_proj))


class MultiScalePatchAggregation(nn.Module):
    def __init__(self, N, P, S, te_dim, hid_dim, history, dropout_rate=0.1):
        super().__init__()
        self.P = P
        scale_ps = sorted({max(2, P // 2), P, max(2, P * 2)})
        self.branches = nn.ModuleList([
            AttentionPatchAggregation(N, p, S=None, te_dim=te_dim, hid_dim=hid_dim, history=history, dropout_rate=dropout_rate)
            for p in scale_ps
        ])
        self.fuse = nn.Sequential(
            nn.Linear(hid_dim * len(scale_ps), hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim)
        )
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, t_stacked, x_with_te, mask_stacked):
        outs = []
        for branch in self.branches:
            h = branch(t_stacked, x_with_te, mask_stacked)
            if h.size(1) != self.P:
                h = F.interpolate(h.transpose(1, 2), size=self.P, mode='linear', align_corners=False).transpose(1, 2)
            outs.append(h)
        h_cat = torch.cat(outs, dim=-1)
        return self.norm(self.fuse(h_cat) + outs[len(outs) // 2])


class MechanismSummaryEncoder(nn.Module):
    def __init__(self, hid_dim, dropout_rate=0.1, use_integral=False, integral_centers=6, integral_width=0.18):
        super().__init__()
        self.use_integral = use_integral
        self.integral_centers = max(2, int(integral_centers))
        self.integral_width = float(integral_width)
        base_dim = 10
        integral_dim = 2 * self.integral_centers if self.use_integral else 0
        self.net = nn.Sequential(
            nn.Linear(base_dim + integral_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
        )
        self.norm = nn.LayerNorm(hid_dim)
        centers = torch.linspace(0.0, 1.0, self.integral_centers)
        self.register_buffer("centers", centers, persistent=False)

    def _time(self, x_mark, dtype):
        if x_mark.dim() == 2:
            t = x_mark.unsqueeze(-1)
        else:
            t = x_mark[:, :, [0]]
        t = torch.nan_to_num(t.to(dtype=dtype), nan=0.0, posinf=1.0, neginf=0.0)
        return t.clamp(0.0, 1.0)

    def _base_features(self, x, x_mask, x_mark):
        dtype = x.dtype
        B, L_obs, _ = x.shape
        t = self._time(x_mark, dtype)
        m = torch.nan_to_num(x_mask.to(dtype=dtype), nan=0.0).clamp(0.0, 1.0)
        v = torch.nan_to_num(x.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
        count = m.sum(dim=1).clamp_min(1.0)
        density = m.mean(dim=1)
        recent_start = max(0, int(0.75 * L_obs))
        recent_density = m[:, recent_start:, :].mean(dim=1)
        count_feature = torch.log1p(count) / math.log1p(max(L_obs, 1))
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
        return torch.stack([
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
        ], dim=-1), t, m, v

    def _integral_features(self, t, m, v):
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

    def forward(self, x, x_mask, x_mark):
        base_features, t, m, v = self._base_features(x, x_mask, x_mark)
        if self.use_integral:
            integral_features = self._integral_features(t, m, v)
            features = torch.cat([base_features, integral_features], dim=-1)
        else:
            features = base_features
        return self.norm(self.net(features))


class MechanismConditionedResidualHead(nn.Module):
    def __init__(self, hid_dim, te_dim, dropout_rate=0.1, use_integral=False,
                 gate_init=-1.6, integral_centers=6, integral_width=0.18):
        super().__init__()
        self.summary = MechanismSummaryEncoder(
            hid_dim,
            dropout_rate=dropout_rate,
            use_integral=use_integral,
            integral_centers=integral_centers,
            integral_width=integral_width,
        )
        self.residual_net = nn.Sequential(
            nn.Linear(hid_dim * 2 + te_dim + 1, hid_dim * 2),
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

    def forward(self, h_expanded, te_pred, base_raw, x, x_mask, x_mark, train_stage=None):
        mechanism = self.summary(x, x_mask, x_mark)
        mechanism_expanded = mechanism.unsqueeze(-2).expand(-1, -1, h_expanded.size(-2), -1)
        residual_input = torch.cat([h_expanded, mechanism_expanded, te_pred, base_raw], dim=-1)
        residual = self.residual_net(residual_input)
        gate = torch.sigmoid(self.gate_net(torch.cat([mechanism_expanded, te_pred], dim=-1)))
        return gate * residual


class LevelValueSummaryEncoder(nn.Module):
    def __init__(self, hid_dim, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
        )
        self.norm = nn.LayerNorm(hid_dim)
        self.latest_features = None

    def _time(self, x_mark, dtype):
        if x_mark.dim() == 2:
            t = x_mark.unsqueeze(-1)
        else:
            t = x_mark[:, :, [0]]
        return torch.nan_to_num(t.to(dtype=dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def forward(self, x, x_mask, x_mark):
        dtype = x.dtype
        t = self._time(x_mark, dtype)
        m = torch.nan_to_num(x_mask.to(dtype=dtype), nan=0.0).clamp(0.0, 1.0)
        v = torch.nan_to_num(x.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
        count = m.sum(dim=1).clamp_min(1.0)
        mean_t = (m * t).sum(dim=1) / count
        mean_v = (v * m).sum(dim=1) / count
        std_v = (((v - mean_v.unsqueeze(1)).pow(2) * m).sum(dim=1) / count).clamp_min(0.0).sqrt()
        recent_w = m * torch.exp(6.0 * t)
        early_w = m * torch.exp(-6.0 * t)
        recent_den = recent_w.sum(dim=1).clamp_min(1e-6)
        early_den = early_w.sum(dim=1).clamp_min(1e-6)
        recent_t = (recent_w * t).sum(dim=1) / recent_den
        early_t = (early_w * t).sum(dim=1) / early_den
        recent_v = (recent_w * v).sum(dim=1) / recent_den
        early_v = (early_w * v).sum(dim=1) / early_den
        trend_v = recent_v - early_v
        features = torch.stack([
            mean_v,
            std_v,
            recent_v,
            early_v,
            trend_v,
            mean_t,
            recent_t,
            early_t,
        ], dim=-1)
        self.latest_features = features.detach()
        return self.norm(self.net(features))


class LevelValueResidualHead(nn.Module):
    def __init__(self, hid_dim, te_dim, dropout_rate=0.1, gate_init=-1.6):
        super().__init__()
        self.summary = LevelValueSummaryEncoder(hid_dim, dropout_rate=dropout_rate)
        self.residual_net = nn.Sequential(
            nn.Linear(hid_dim * 2 + te_dim + 1, hid_dim * 2),
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

    def forward(self, h_expanded, te_pred, base_raw, x, x_mask, x_mark, train_stage=None):
        level = self.summary(x, x_mask, x_mark)
        level_expanded = level.unsqueeze(-2).expand(-1, -1, h_expanded.size(-2), -1)
        residual_input = torch.cat([h_expanded, level_expanded, te_pred, base_raw], dim=-1)
        residual = self.residual_net(residual_input)
        gate = torch.sigmoid(self.gate_net(torch.cat([level_expanded, te_pred], dim=-1)))
        return gate * residual


class RawStateResidualHead(nn.Module):
    def __init__(self, hid_dim, te_dim, dropout_rate=0.1, gate_init=-1.6):
        super().__init__()
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

    def forward(self, h_expanded, te_pred, base_raw, x, x_mask, x_mark, train_stage=None):
        residual = self.residual_net(torch.cat([h_expanded, te_pred, base_raw], dim=-1))
        gate = torch.sigmoid(self.gate_net(torch.cat([h_expanded, te_pred], dim=-1)))
        return gate * residual


class LevelMechanismResidualHead(nn.Module):
    def __init__(self, hid_dim, te_dim, dropout_rate=0.1, gate_init=-1.6,
                 integral_centers=6, integral_width=0.18):
        super().__init__()
        self.level_summary = LevelValueSummaryEncoder(hid_dim, dropout_rate=dropout_rate)
        self.mechanism_summary = MechanismSummaryEncoder(
            hid_dim,
            dropout_rate=dropout_rate,
            use_integral=False,
            integral_centers=integral_centers,
            integral_width=integral_width,
        )
        self.residual_net = nn.Sequential(
            nn.Linear(hid_dim * 3 + te_dim + 1, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim * 2, 1),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(hid_dim * 2 + te_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, float(gate_init))

    def forward(self, h_expanded, te_pred, base_raw, x, x_mask, x_mark, train_stage=None):
        level = self.level_summary(x, x_mask, x_mark)
        mechanism = self.mechanism_summary(x, x_mask, x_mark)
        level_expanded = level.unsqueeze(-2).expand(-1, -1, h_expanded.size(-2), -1)
        mechanism_expanded = mechanism.unsqueeze(-2).expand(-1, -1, h_expanded.size(-2), -1)
        summary = torch.cat([level_expanded, mechanism_expanded], dim=-1)
        residual_input = torch.cat([h_expanded, summary, te_pred, base_raw], dim=-1)
        residual = self.residual_net(residual_input)
        gate = torch.sigmoid(self.gate_net(torch.cat([summary, te_pred], dim=-1)))
        return gate * residual


class LevelAnchoredMotionResidualHead(nn.Module):
    """Level-preserving motion residual for asynchronous sensor trajectories.

    The head constructs a deterministic short-horizon motion prior from the
    most recent observed level and an exponentially weighted recent-vs-early
    velocity estimate.  The learned part only gates the correction from the
    backbone prediction to this prior and adds a bounded residual.
    """

    def __init__(self, hid_dim, te_dim, dropout_rate=0.1, gate_init=-1.6,
                 motion_scale=1.0, correction_scale=0.10):
        super().__init__()
        self.level_summary = LevelValueSummaryEncoder(hid_dim, dropout_rate=dropout_rate)
        self.motion_scale = float(motion_scale)
        self.correction_scale = float(correction_scale)
        self.correction_net = nn.Sequential(
            nn.Linear(hid_dim * 2 + te_dim + 3, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim * 2, 1),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(hid_dim + te_dim + 2, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, float(gate_init))
        self.latest_motion_prior = None
        self.latest_anchor_gate = None

    def _time(self, x_mark, dtype):
        if x_mark.dim() == 2:
            t = x_mark.unsqueeze(-1)
        else:
            t = x_mark[:, :, [0]]
        return torch.nan_to_num(t.to(dtype=dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def motion_prior(self, x, x_mask, x_mark, y_time):
        dtype = x.dtype
        B, L_obs, N = x.shape
        t = self._time(x_mark, dtype)
        m = torch.nan_to_num(x_mask.to(dtype=dtype), nan=0.0).clamp(0.0, 1.0)
        v = torch.nan_to_num(x.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
        any_obs = m.sum(dim=1) > 0
        fallback = (v * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

        t_full = t.expand(B, L_obs, N)
        last_score = t_full.masked_fill(m <= 0.0, -1e9)
        last_idx = last_score.argmax(dim=1)
        last_v = torch.gather(v, 1, last_idx.unsqueeze(1)).squeeze(1)
        last_t = torch.gather(t_full, 1, last_idx.unsqueeze(1)).squeeze(1)
        last_v = torch.where(any_obs, last_v, fallback)
        last_t = torch.where(any_obs, last_t, torch.zeros_like(last_t))

        recent_w = m * torch.exp(6.0 * t)
        early_w = m * torch.exp(-6.0 * t)
        recent_den = recent_w.sum(dim=1).clamp_min(1e-6)
        early_den = early_w.sum(dim=1).clamp_min(1e-6)
        recent_v = (recent_w * v).sum(dim=1) / recent_den
        early_v = (early_w * v).sum(dim=1) / early_den
        recent_t = (recent_w * t).sum(dim=1) / recent_den
        early_t = (early_w * t).sum(dim=1) / early_den
        velocity = (recent_v - early_v) / (recent_t - early_t).abs().clamp_min(1e-3)

        if y_time is None:
            L_pred = 1
            y_time = last_t.unsqueeze(-1).unsqueeze(-1).expand(B, N, L_pred, 1)
        dt = (y_time.to(dtype=dtype).squeeze(-1) - last_t.unsqueeze(-1)).clamp_min(0.0)
        prior = last_v.unsqueeze(-1) + self.motion_scale * velocity.unsqueeze(-1) * dt
        return prior.unsqueeze(-1), dt.unsqueeze(-1)

    def forward(self, h_expanded, te_pred, base_raw, x, x_mask, x_mark, train_stage=None, y_time=None):
        level = self.level_summary(x, x_mask, x_mark)
        level_expanded = level.unsqueeze(-2).expand(-1, -1, h_expanded.size(-2), -1)
        prior, dt = self.motion_prior(x, x_mask, x_mark, y_time)
        anchor_delta = prior - base_raw
        correction_input = torch.cat([h_expanded, level_expanded, te_pred, base_raw, anchor_delta, dt], dim=-1)
        correction = torch.tanh(self.correction_net(correction_input)) * self.correction_scale
        gate_input = torch.cat([level_expanded, te_pred, anchor_delta.abs(), dt], dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        residual = gate * (anchor_delta + correction)
        self.latest_motion_prior = prior.detach()
        self.latest_anchor_gate = gate.detach()
        return residual


class AliasFactoredLevelMechanismResidualHead(nn.Module):
    """Level-aliasing residual: (a0 * alpha + a1 * beta) + bounded remainder."""

    def __init__(self, hid_dim, te_dim, dropout_rate=0.1, gate_init=-1.6,
                 integral_centers=6, integral_width=0.18, remainder_scale=0.25,
                 alias_score_gate_weight=1.0, variance_price_weight=0.0,
                 scale_price_weight=0.0):
        super().__init__()
        self.level_summary = LevelValueSummaryEncoder(hid_dim, dropout_rate=dropout_rate)
        self.mechanism_summary = MechanismSummaryEncoder(
            hid_dim,
            dropout_rate=dropout_rate,
            use_integral=False,
            integral_centers=integral_centers,
            integral_width=integral_width,
        )
        self.level_coeff = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, 2),
        )
        self.measure_coeff = nn.Sequential(
            nn.Linear(hid_dim + te_dim + 1, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, 2),
        )
        self.remainder = nn.Sequential(
            nn.Linear(hid_dim * 3 + te_dim + 1, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, 1),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(hid_dim * 2 + te_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, float(gate_init))
        self.remainder_scale = float(remainder_scale)
        self.alias_score_gate_weight = float(alias_score_gate_weight)
        self.variance_price_weight = float(variance_price_weight)
        self.scale_price_weight = float(scale_price_weight)
        self.latest_alias_residual = None
        self.latest_alias_score = None
        self.latest_alias_variance_price = None
        self.latest_remainder_residual = None
        self.latest_remainder_residual_train = None

    def forward(self, h_expanded, te_pred, base_raw, x, x_mask, x_mark, train_stage=None):
        level = self.level_summary(x, x_mask, x_mark)
        mechanism = self.mechanism_summary(x, x_mask, x_mark)
        level_expanded = level.unsqueeze(-2).expand(-1, -1, h_expanded.size(-2), -1)
        mechanism_expanded = mechanism.unsqueeze(-2).expand(-1, -1, h_expanded.size(-2), -1)
        summary = torch.cat([level_expanded, mechanism_expanded], dim=-1)

        local_coeff = self.level_coeff(level).unsqueeze(-2).expand(-1, -1, h_expanded.size(-2), -1)
        measure_coeff = self.measure_coeff(torch.cat([mechanism_expanded, te_pred, base_raw], dim=-1))
        alias = (local_coeff * measure_coeff).sum(dim=-1, keepdim=True)
        alias_score = local_coeff.norm(dim=-1, keepdim=True) * measure_coeff.norm(dim=-1, keepdim=True)
        remainder_input = torch.cat([h_expanded, summary, te_pred, base_raw], dim=-1)
        remainder = torch.tanh(self.remainder(remainder_input)) * self.remainder_scale
        residual = alias + remainder
        gate_logits = self.gate_net(torch.cat([summary, te_pred], dim=-1))
        gate_logits = gate_logits + self.alias_score_gate_weight * torch.log1p(alias_score)
        variance_price = None
        if self.variance_price_weight > 0.0 or self.scale_price_weight > 0.0:
            level_features = getattr(self.level_summary, "latest_features", None)
            if level_features is not None:
                level_scale = level_features[..., 1:2].unsqueeze(-2).expand_as(alias_score)
            else:
                level_scale = residual.detach().new_ones(residual.shape)
            level_scale = level_scale.detach().abs().clamp_min(1e-3)
            normalized_energy = residual.detach().pow(2) / (level_scale.pow(2) + 1e-4)
            variance_price = torch.log1p(normalized_energy)
            gate_logits = gate_logits - self.variance_price_weight * variance_price
            if self.scale_price_weight > 0.0:
                gate_logits = gate_logits - self.scale_price_weight * torch.log1p(level_scale)
        gate = torch.sigmoid(gate_logits)
        self.latest_alias_residual = alias.detach()
        self.latest_alias_score = alias_score.detach()
        self.latest_alias_variance_price = None if variance_price is None else variance_price.detach()
        self.latest_remainder_residual = remainder.detach()
        self.latest_remainder_residual_train = remainder
        return gate * residual


class AdaptiveMechanismResidualHead(nn.Module):
    def __init__(self, hid_dim, te_dim, dropout_rate=0.1, gate_init=-1.6,
                 switch_gate_init=-2.0, warmup_stage=0, warmup_blend=0.5,
                 integral_centers=6, integral_width=0.18):
        super().__init__()
        self.warmup_stage = int(warmup_stage)
        self.warmup_blend = float(warmup_blend)
        self.local_head = MechanismConditionedResidualHead(
            hid_dim,
            te_dim,
            dropout_rate=dropout_rate,
            use_integral=False,
            gate_init=gate_init,
            integral_centers=integral_centers,
            integral_width=integral_width,
        )
        self.integral_head = MechanismConditionedResidualHead(
            hid_dim,
            te_dim,
            dropout_rate=dropout_rate,
            use_integral=True,
            gate_init=gate_init,
            integral_centers=integral_centers,
            integral_width=integral_width,
        )
        self.switch_summary = MechanismSummaryEncoder(
            hid_dim,
            dropout_rate=dropout_rate,
            use_integral=True,
            integral_centers=integral_centers,
            integral_width=integral_width,
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
        self.latest_branch_residual_norms = None
        self.latest_boundary_penalty = None
        self.latest_gate_logits = None
        self.latest_local_residual = None
        self.latest_integral_residual = None

    def forward(self, h_expanded, te_pred, base_raw, x, x_mask, x_mark, train_stage=None):
        local_residual = self.local_head(h_expanded, te_pred, base_raw, x, x_mask, x_mark)
        integral_residual = self.integral_head(h_expanded, te_pred, base_raw, x, x_mask, x_mark)
        mechanism = self.switch_summary(x, x_mask, x_mark)
        mechanism_expanded = mechanism.unsqueeze(-2).expand(-1, -1, h_expanded.size(-2), -1)
        disagreement = (local_residual - integral_residual).abs()
        switch_input = torch.cat([mechanism_expanded, te_pred, base_raw, disagreement], dim=-1)
        gate_logits = self.switch_net(switch_input)
        learned_gate = torch.sigmoid(gate_logits)
        use_fixed_warmup = train_stage is not None and self.warmup_stage > 0 and int(train_stage) <= self.warmup_stage
        if use_fixed_warmup:
            integral_gate = torch.full_like(learned_gate, self.warmup_blend)
            self.latest_boundary_penalty = learned_gate.new_zeros(())
        else:
            integral_gate = learned_gate
            self.latest_boundary_penalty = (integral_gate * (1.0 - integral_gate)).mean()
        mixed = (1.0 - integral_gate) * local_residual + integral_gate * integral_residual
        self.latest_integral_gate = integral_gate.detach()
        self.latest_gate_logits = gate_logits
        self.latest_local_residual = local_residual
        self.latest_integral_residual = integral_residual
        self.latest_branch_residual_norms = {
            "mcr": local_residual.pow(2).mean().detach(),
            "mcir": integral_residual.pow(2).mean().detach(),
            "mixed": mixed.pow(2).mean().detach(),
            "integral_gate_mean": integral_gate.mean().detach(),
        }
        return mixed


class ReliabilityAdaptiveResidualHead(nn.Module):
    """Reliability-aware multi-branch residual state head for wearable IMTS.

    The head keeps the successful local/integral residual idea but changes the
    gating signal from mechanism-only statistics to a residual state that
    includes observed level, sampling reliability, and cross-variable context.
    It is intended as the default RAMS-Net correction head.
    """

    def __init__(self, hid_dim, te_dim, dropout_rate=0.1, gate_init=-1.8,
                 integral_centers=6, integral_width=0.18, branch_temperature=1.0):
        super().__init__()
        self.branch_temperature = max(float(branch_temperature), 1e-3)
        self.level_summary = LevelValueSummaryEncoder(hid_dim, dropout_rate=dropout_rate)
        self.local_reliability = MechanismSummaryEncoder(
            hid_dim,
            dropout_rate=dropout_rate,
            use_integral=False,
            integral_centers=integral_centers,
            integral_width=integral_width,
        )
        self.integral_reliability = MechanismSummaryEncoder(
            hid_dim,
            dropout_rate=dropout_rate,
            use_integral=True,
            integral_centers=integral_centers,
            integral_width=integral_width,
        )
        self.cross_context = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
        )
        branch_in = hid_dim * 3 + te_dim + 1
        self.local_net = nn.Sequential(
            nn.Linear(branch_in, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim * 2, 1),
        )
        self.integral_net = nn.Sequential(
            nn.Linear(branch_in, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim * 2, 1),
        )
        self.cross_net = nn.Sequential(
            nn.Linear(branch_in, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim * 2, 1),
        )
        gate_in = hid_dim * 4 + te_dim + 2
        self.branch_gate = nn.Sequential(
            nn.Linear(gate_in, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, 3),
        )
        self.open_gate = nn.Sequential(
            nn.Linear(gate_in, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, 1),
        )
        nn.init.zeros_(self.branch_gate[-1].weight)
        nn.init.zeros_(self.branch_gate[-1].bias)
        nn.init.zeros_(self.open_gate[-1].weight)
        nn.init.constant_(self.open_gate[-1].bias, float(gate_init))
        self.latest_branch_weights = None
        self.latest_open_gate = None
        self.latest_branch_residual_norms = None
        self.latest_local_residual = None
        self.latest_integral_residual = None
        self.latest_cross_residual = None

    def forward(self, h_expanded, te_pred, base_raw, x, x_mask, x_mark, train_stage=None):
        level = self.level_summary(x, x_mask, x_mark)
        local_rel = self.local_reliability(x, x_mask, x_mark)
        integral_rel = self.integral_reliability(x, x_mask, x_mark)
        global_level = level.mean(dim=1, keepdim=True).expand_as(level)
        global_rel = integral_rel.mean(dim=1, keepdim=True).expand_as(integral_rel)
        cross = self.cross_context(torch.cat([global_level, global_rel], dim=-1))

        level_e = level.unsqueeze(-2).expand(-1, -1, h_expanded.size(-2), -1)
        local_e = local_rel.unsqueeze(-2).expand_as(level_e)
        integral_e = integral_rel.unsqueeze(-2).expand_as(level_e)
        cross_e = cross.unsqueeze(-2).expand_as(level_e)

        local_in = torch.cat([h_expanded, level_e, local_e, te_pred, base_raw], dim=-1)
        integral_in = torch.cat([h_expanded, level_e, integral_e, te_pred, base_raw], dim=-1)
        cross_in = torch.cat([h_expanded, level_e, cross_e, te_pred, base_raw], dim=-1)
        local_residual = self.local_net(local_in)
        integral_residual = self.integral_net(integral_in)
        cross_residual = self.cross_net(cross_in)

        local_integral_gap = (local_residual - integral_residual).abs()
        cross_gap = (cross_residual - local_residual).abs()
        gate_in = torch.cat([level_e, local_e, integral_e, cross_e, te_pred, local_integral_gap, cross_gap], dim=-1)
        branch_weights = torch.softmax(self.branch_gate(gate_in) / self.branch_temperature, dim=-1)
        open_gate = torch.sigmoid(self.open_gate(gate_in))
        stacked = torch.cat([local_residual, integral_residual, cross_residual], dim=-1)
        mixed = (branch_weights * stacked).sum(dim=-1, keepdim=True)

        self.latest_branch_weights = branch_weights.detach()
        self.latest_open_gate = open_gate.detach()
        self.latest_local_residual = local_residual
        self.latest_integral_residual = integral_residual
        self.latest_cross_residual = cross_residual
        self.latest_branch_residual_norms = {
            "local": local_residual.pow(2).mean().detach(),
            "integral": integral_residual.pow(2).mean().detach(),
            "cross": cross_residual.pow(2).mean().detach(),
            "mixed": mixed.pow(2).mean().detach(),
            "open_gate_mean": open_gate.mean().detach(),
        }
        return open_gate * mixed


class FourierPatchModulator(nn.Module):
    def __init__(self, P, hid_dim, dropout_rate=0.1, gated=False, adaptive=False, gate_init=-2.0):
        super().__init__()
        self.gated = gated
        self.adaptive = adaptive
        self.freq_gain = nn.Parameter(torch.zeros(P // 2 + 1, hid_dim))
        self.residual_gate = nn.Parameter(torch.full((hid_dim,), float(gate_init))) if (gated or adaptive) else None
        if adaptive:
            self.context_gate = nn.Sequential(
                nn.Linear(hid_dim * 2, hid_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hid_dim, hid_dim)
            )
            nn.init.zeros_(self.context_gate[-1].weight)
            nn.init.zeros_(self.context_gate[-1].bias)
        else:
            self.context_gate = None
        self.mix = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim)
        )
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, h):
        spec = torch.fft.rfft(h, dim=2)
        gain = 1.0 + 0.1 * torch.tanh(self.freq_gain).view(1, 1, *self.freq_gain.shape)
        filtered = torch.fft.irfft(spec * gain, n=h.size(2), dim=2)
        residual = self.mix(filtered)
        if self.residual_gate is not None:
            gate_logits = self.residual_gate.view(1, 1, 1, -1)
            if self.context_gate is not None:
                patch_mean = h.mean(dim=2)
                patch_std = h.std(dim=2, unbiased=False)
                context = torch.cat([patch_mean, patch_std], dim=-1)
                gate_logits = gate_logits + self.context_gate(context).unsqueeze(2)
            residual = residual * torch.sigmoid(gate_logits)
        return self.norm(h + residual)


class PatchConvModulator(nn.Module):
    def __init__(self, hid_dim, dropout_rate=0.1, gate_init=-2.5):
        super().__init__()
        self.depthwise = nn.Conv1d(hid_dim, hid_dim, kernel_size=3, padding=1, groups=hid_dim)
        self.pointwise = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=1)
        )
        self.gate = nn.Parameter(torch.full((hid_dim,), float(gate_init)))
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, h):
        B, N, P, D = h.shape
        z = h.reshape(B * N, P, D).transpose(1, 2)
        z = self.pointwise(self.depthwise(z)).transpose(1, 2).reshape(B, N, P, D)
        return self.norm(h + torch.sigmoid(self.gate).view(1, 1, 1, D) * z)


class WaveletPatchModulator(nn.Module):
    def __init__(self, hid_dim, dropout_rate=0.1, gate_init=-2.5):
        super().__init__()
        self.high_proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim)
        )
        self.low_proj = nn.Linear(hid_dim, hid_dim)
        self.high_gate = nn.Parameter(torch.full((hid_dim,), float(gate_init)))
        self.low_gate = nn.Parameter(torch.full((hid_dim,), float(gate_init - 1.0)))
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, h):
        B, N, P, D = h.shape
        z = h.reshape(B * N, P, D).transpose(1, 2)
        z_pad = F.pad(z, (1, 1), mode='replicate')
        low = F.avg_pool1d(z_pad, kernel_size=3, stride=1).transpose(1, 2).reshape(B, N, P, D)
        high = h - low
        residual = torch.sigmoid(self.high_gate).view(1, 1, 1, D) * self.high_proj(high)
        residual = residual + torch.sigmoid(self.low_gate).view(1, 1, 1, D) * self.low_proj(low)
        return self.norm(h + residual)


class TrendPatchModulator(nn.Module):
    def __init__(self, hid_dim, dropout_rate=0.1, gate_init=-2.5):
        super().__init__()
        self.trend_proj = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim)
        )
        self.gate = nn.Parameter(torch.full((hid_dim,), float(gate_init)))
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, h):
        B, N, P, D = h.shape
        z = h.reshape(B * N, P, D).transpose(1, 2)
        z_pad = F.pad(z, (2, 2), mode='replicate')
        trend = F.avg_pool1d(z_pad, kernel_size=5, stride=1).transpose(1, 2).reshape(B, N, P, D)
        seasonal = h - trend
        residual = self.trend_proj(torch.cat([trend, seasonal], dim=-1))
        return self.norm(h + torch.sigmoid(self.gate).view(1, 1, 1, D) * residual)


class SeasonalPatchModulator(nn.Module):
    def __init__(self, P, hid_dim, dropout_rate=0.1, gate_init=-2.5):
        super().__init__()
        self.P = P
        self.season_proj = nn.Linear(4, hid_dim)
        self.mix = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim)
        )
        self.gate = nn.Parameter(torch.full((hid_dim,), float(gate_init)))
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, h):
        B, N, P, D = h.shape
        pos = torch.linspace(0.0, 1.0, P, device=h.device, dtype=h.dtype)
        harmonics = torch.stack([
            torch.sin(2.0 * math.pi * pos),
            torch.cos(2.0 * math.pi * pos),
            torch.sin(4.0 * math.pi * pos),
            torch.cos(4.0 * math.pi * pos),
        ], dim=-1)
        season = self.season_proj(harmonics).view(1, 1, P, D).expand(B, N, -1, -1)
        residual = self.mix(torch.cat([h, season], dim=-1))
        return self.norm(h + torch.sigmoid(self.gate).view(1, 1, 1, D) * residual)


class LaggedDependencyPatchModulator(nn.Module):
    def __init__(self, n_vars, hid_dim, dropout_rate=0.1, gate_init=-3.0):
        super().__init__()
        init = torch.zeros(n_vars, n_vars)
        init.fill_(-2.0)
        init.fill_diagonal_(1.0)
        self.adj_logits = nn.Parameter(init)
        self.proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim)
        )
        self.gate = nn.Parameter(torch.full((n_vars, 1, 1), float(gate_init)))
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, h):
        if h.size(2) <= 1:
            return h
        lagged = torch.cat([h[:, :, :1], h[:, :, :-1]], dim=2)
        adj = torch.softmax(self.adj_logits, dim=-1)
        mixed = torch.einsum('ij,bjpd->bipd', adj, lagged)
        residual = self.proj(mixed - h)
        return self.norm(h + torch.sigmoid(self.gate).view(1, -1, 1, 1) * residual)


class LowRankVariableMixer(nn.Module):
    def __init__(self, n_vars, hid_dim, rank=4, dropout_rate=0.1, gate_init=-3.0):
        super().__init__()
        rank = max(1, min(rank, n_vars))
        self.left = nn.Parameter(torch.randn(n_vars, rank) * 0.02)
        self.right = nn.Parameter(torch.randn(n_vars, rank) * 0.02)
        self.proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim)
        )
        self.gate = nn.Parameter(torch.full((n_vars, 1), float(gate_init)))
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, h):
        scores = torch.matmul(self.left, self.right.transpose(0, 1)) / math.sqrt(self.left.size(1))
        adj = torch.softmax(scores, dim=-1)
        mixed = torch.einsum('ij,bjd->bid', adj, h)
        return self.norm(h + torch.sigmoid(self.gate).view(1, -1, 1) * self.proj(mixed))


class ReliabilityLowRankVariableMixer(nn.Module):
    """Sample-conditioned low-rank variable mixing with observation reliability.

    The global low-rank graph learns which variables can exchange information.
    Per-sample reliability shifts the source logits and opens the residual more
    on uncertain target variables.  This keeps the APN-style mixer cheap while
    making it sensitive to IMTS-specific missingness and recency patterns.
    """

    requires_stats = True

    def __init__(
        self,
        n_vars,
        hid_dim,
        rank=4,
        dropout_rate=0.1,
        gate_init=-3.0,
        source_scale=1.0,
        target_scale=1.0,
    ):
        super().__init__()
        rank = max(1, min(rank, n_vars))
        self.left = nn.Parameter(torch.randn(n_vars, rank) * 0.02)
        self.right = nn.Parameter(torch.randn(n_vars, rank) * 0.02)
        self.proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim)
        )
        self.gate = nn.Parameter(torch.full((n_vars, 1), float(gate_init)))
        self.source_scale = float(source_scale)
        self.target_scale = float(target_scale)
        self.norm = nn.LayerNorm(hid_dim)
        self.latest_route_penalty = None

    @staticmethod
    def source_quality(stats):
        density = stats[..., 0]
        missing = stats[..., 1]
        irregularity = stats[..., 3]
        recency = stats[..., 8] if stats.size(-1) > 8 else torch.zeros_like(missing)
        observed_span = stats[..., 9] if stats.size(-1) > 9 else torch.ones_like(missing)
        return 1.5 * density + 0.5 * observed_span - 0.7 * missing - 0.6 * irregularity - 0.6 * recency

    @staticmethod
    def target_uncertainty(stats):
        missing = stats[..., 1]
        heterogeneity = stats[..., 2]
        irregularity = stats[..., 3]
        recency = stats[..., 8] if stats.size(-1) > 8 else torch.zeros_like(missing)
        observed_span = stats[..., 9] if stats.size(-1) > 9 else torch.ones_like(missing)
        return (missing + heterogeneity + irregularity + recency + (1.0 - observed_span)) / 5.0

    def forward(self, h, stats=None):
        self.latest_route_penalty = None
        scores = torch.matmul(self.left, self.right.transpose(0, 1)) / math.sqrt(self.left.size(1))
        if stats is None:
            adj = torch.softmax(scores, dim=-1).unsqueeze(0).expand(h.size(0), -1, -1)
            gate_logits = self.gate.view(1, -1, 1)
        else:
            source_bias = self.source_quality(stats).clamp(-2.0, 2.0)
            target_uncertainty = self.target_uncertainty(stats).clamp(0.0, 1.0)
            adj = torch.softmax(scores.unsqueeze(0) + self.source_scale * source_bias.unsqueeze(1), dim=-1)
            gate_logits = self.gate.view(1, -1, 1) + self.target_scale * target_uncertainty.unsqueeze(-1)
            source_reliability = torch.sigmoid(source_bias)
            self.latest_route_penalty = (adj * (1.0 - source_reliability).unsqueeze(1)).mean()
        mixed = torch.einsum('bij,bjd->bid', adj, h)
        return self.norm(h + torch.sigmoid(gate_logits) * self.proj(mixed))


class GraphVariableMixer(nn.Module):
    def __init__(self, n_vars, hid_dim, dropout_rate=0.1, gate_init=-3.0):
        super().__init__()
        self.adj_logits = nn.Parameter(torch.zeros(n_vars, n_vars))
        self.proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim)
        )
        self.gate = nn.Parameter(torch.full((n_vars, 1), float(gate_init)))
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, h):
        adj = torch.softmax(self.adj_logits, dim=-1)
        mixed = torch.einsum('ij,bjd->bid', adj, h)
        return self.norm(h + torch.sigmoid(self.gate).view(1, -1, 1) * self.proj(mixed))


class CovarianceVariableMixer(nn.Module):
    def __init__(self, hid_dim, dropout_rate=0.1, gate_init=-3.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim)
        )
        self.gate = nn.Parameter(torch.full((hid_dim,), float(gate_init)))
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, h):
        scores = torch.matmul(h, h.transpose(1, 2)) / math.sqrt(h.size(-1))
        adj = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(adj, h)
        return self.norm(h + torch.sigmoid(self.gate).view(1, 1, -1) * self.proj(mixed))


class SEVariableGate(nn.Module):
    def __init__(self, hid_dim, dropout_rate=0.1, gate_init=-2.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim)
        )
        self.gate = nn.Parameter(torch.full((hid_dim,), float(gate_init)))
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, h):
        context = torch.cat([h, h.mean(dim=1, keepdim=True).expand_as(h)], dim=-1)
        residual = self.net(context)
        return self.norm(h + torch.sigmoid(self.gate).view(1, 1, -1) * residual)


class SpectralResidualBranch(nn.Module):
    def __init__(self, P, hid_dim, dropout_rate=0.1):
        super().__init__()
        self.freq_gain = nn.Parameter(torch.zeros(P // 2 + 1, hid_dim))
        self.mix = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, h):
        spec = torch.fft.rfft(h, dim=2)
        gain = 1.0 + 0.1 * torch.tanh(self.freq_gain).view(1, 1, *self.freq_gain.shape)
        filtered = torch.fft.irfft(spec * gain, n=h.size(2), dim=2)
        return self.mix(filtered)


class TrendResidualBranch(nn.Module):
    def __init__(self, hid_dim, dropout_rate=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, h):
        B, N, P, D = h.shape
        z = h.reshape(B * N, P, D).transpose(1, 2)
        z_pad = F.pad(z, (2, 2), mode='replicate')
        trend = F.avg_pool1d(z_pad, kernel_size=5, stride=1).transpose(1, 2).reshape(B, N, P, D)
        seasonal = h - trend
        return self.proj(torch.cat([trend, seasonal], dim=-1))


class CovarianceResidualBranch(nn.Module):
    def __init__(self, hid_dim, dropout_rate=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, h):
        scores = torch.matmul(h, h.transpose(1, 2)) / math.sqrt(h.size(-1))
        adj = torch.softmax(scores, dim=-1)
        return self.proj(torch.matmul(adj, h))


class StableGraphResidualBranch(nn.Module):
    def __init__(self, n_vars, hid_dim, dropout_rate=0.1):
        super().__init__()
        init = torch.zeros(n_vars, n_vars)
        init.fill_(-2.0)
        init.fill_diagonal_(2.0)
        self.adj_logits = nn.Parameter(init)
        self.proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, h):
        adj = torch.softmax(self.adj_logits, dim=-1)
        mixed = torch.matmul(adj.unsqueeze(0), h)
        return self.proj(mixed - h)


class HybridDependencyResidualBranch(nn.Module):
    def __init__(self, n_vars, hid_dim, dropout_rate=0.1):
        super().__init__()
        self.cov = CovarianceResidualBranch(hid_dim, dropout_rate)
        self.graph = StableGraphResidualBranch(n_vars, hid_dim, dropout_rate)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, h):
        weight = torch.sigmoid(self.mix_logit)
        return weight * self.cov(h) + (1.0 - weight) * self.graph(h)


class StructuralStats(nn.Module):
    stat_dim = 10

    def __init__(self, n_vars):
        super().__init__()
        self.n_vars = n_vars

    def forward(self, h, mask_stacked, time_stacked, B, N_vars):
        eps = 1e-6
        _, _, P, _ = h.shape
        mask = mask_stacked.view(B, N_vars, -1).float()
        time = time_stacked.view(B, N_vars, -1).float()

        density = mask.mean(dim=-1).clamp(0.0, 1.0)
        missing = 1.0 - density
        missing_heterogeneity = missing.std(dim=1, unbiased=False).unsqueeze(1).expand(-1, N_vars)

        observed_time_for_max = time.masked_fill(mask <= 0.0, -1e6)
        observed_time_for_min = time.masked_fill(mask <= 0.0, 1e6)
        last_time = observed_time_for_max.max(dim=-1).values
        first_time = observed_time_for_min.min(dim=-1).values
        any_obs = (mask.sum(dim=-1) > 0.0)
        anchor_time = time.max(dim=-1).values.clamp_min(eps)
        last_time = torch.where(any_obs, last_time, torch.zeros_like(last_time))
        first_time = torch.where(any_obs, first_time, last_time)
        recency = ((anchor_time - last_time).clamp_min(0.0) / anchor_time).clamp(0.0, 1.0)
        observed_span = ((last_time - first_time).clamp_min(0.0) / anchor_time).clamp(0.0, 1.0)

        time_diff = (time[:, :, 1:] - time[:, :, :-1]).abs()
        pair_mask = mask[:, :, 1:] * mask[:, :, :-1]
        pair_count = pair_mask.sum(dim=-1).clamp_min(eps)
        mean_gap = (time_diff * pair_mask).sum(dim=-1) / pair_count
        gap_var = (((time_diff - mean_gap.unsqueeze(-1)) ** 2) * pair_mask).sum(dim=-1) / pair_count
        irregularity = (torch.sqrt(gap_var + eps) / (mean_gap + eps)).clamp(0.0, 5.0) / 5.0

        spec = torch.fft.rfft(h, dim=2)
        energy = spec.abs().pow(2).mean(dim=-1) + eps
        prob = energy / energy.sum(dim=-1, keepdim=True).clamp_min(eps)
        spectral_entropy = -(prob * torch.log(prob + eps)).sum(dim=-1)
        spectral_entropy = spectral_entropy / math.log(max(P // 2 + 1, 2))

        var_summary = F.normalize(h.mean(dim=2), dim=-1)
        cov = torch.matmul(var_summary, var_summary.transpose(1, 2)).abs()
        eye = torch.eye(N_vars, device=h.device, dtype=h.dtype).unsqueeze(0)
        cov_energy = (cov * (1.0 - eye)).sum(dim=(1, 2)) / max(N_vars * (N_vars - 1), 1)
        cov_energy = cov_energy.unsqueeze(1).expand(-1, N_vars).clamp(0.0, 1.0)

        patch_smoothness = (h[:, :, 1:] - h[:, :, :-1]).abs().mean(dim=(2, 3)) if P > 1 else h.new_zeros(B, N_vars)
        patch_smoothness = torch.tanh(patch_smoothness)
        quadrature_uncertainty = torch.tanh(missing + irregularity + 0.5 * patch_smoothness)

        return torch.stack([
            density,
            missing,
            missing_heterogeneity,
            irregularity,
            spectral_entropy.clamp(0.0, 1.0),
            cov_energy,
            patch_smoothness,
            quadrature_uncertainty,
            recency,
            observed_span
        ], dim=-1)


class StructureSelector(nn.Module):
    def __init__(self, stat_dim, hidden_dim=32, dropout_rate=0.1, gate_init=-2.0, temperature=1.0):
        super().__init__()
        self.temperature = max(float(temperature), 1e-3)
        self.net = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 4)
        )
        nn.init.constant_(self.net[-1].bias, float(gate_init))

    def forward(self, stats):
        return torch.sigmoid(self.net(stats) / self.temperature)


class OperatorCorrectionGate(nn.Module):
    """Soft training proxy for risk-controlled operator correction.

    The deployment certificate is still the selected-set LCB audit. This module
    makes the trainable gate follow the same accounting: open a correction only
    when a learned benefit score exceeds an explicit price from irregularity
    and instability statistics.
    """

    def __init__(
        self,
        stat_dim,
        hidden_dim=32,
        dropout_rate=0.1,
        gate_init=-2.0,
        temperature=1.0,
        price_weight=1.0,
        price_margin=0.0,
        price_init=0.1,
    ):
        super().__init__()
        self.temperature = max(float(temperature), 1e-3)
        self.price_weight = float(price_weight)
        self.price_margin = float(price_margin)
        self.benefit_net = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 4)
        )
        self.safety_net = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.constant_(self.benefit_net[-1].bias, float(gate_init))
        nn.init.constant_(self.safety_net[-1].bias, float(gate_init))
        raw_init = math.log(math.exp(max(float(price_init), 1e-6)) - 1.0)
        self.raw_price_coeff = nn.Parameter(torch.full((5,), raw_init))

    def price_components(self, stats):
        density = stats[..., 0]
        missing = stats[..., 1]
        heterogeneity = stats[..., 2]
        irregularity = stats[..., 3]
        spectral_entropy = stats[..., 4]
        cov_energy = stats[..., 5]
        patch_smoothness = stats[..., 6]
        quadrature_uncertainty = stats[..., 7]
        recency = stats[..., 8] if stats.size(-1) > 8 else torch.zeros_like(missing)
        observed_span = stats[..., 9] if stats.size(-1) > 9 else torch.ones_like(missing)
        interp = (0.45 * missing + 0.35 * irregularity + 0.20 * quadrature_uncertainty).clamp(0.0, 1.0)
        quadrature = quadrature_uncertainty.clamp(0.0, 1.0)
        dependency = (0.30 * (1.0 - cov_energy) + 0.30 * irregularity + 0.25 * heterogeneity + 0.15 * (1.0 - observed_span)).clamp(0.0, 1.0)
        missing_sensitivity = (0.35 * missing + 0.25 * heterogeneity + 0.20 * (1.0 - density) + 0.20 * recency).clamp(0.0, 1.0)
        smooth = (0.40 * patch_smoothness + 0.30 * irregularity + 0.30 * spectral_entropy).clamp(0.0, 1.0)
        return torch.stack([interp, quadrature, dependency, missing_sensitivity, smooth], dim=-1)

    def forward(self, stats):
        coeff = F.softplus(self.raw_price_coeff)
        components = self.price_components(stats)
        price = (components * coeff.view(*([1] * (components.ndim - 1)), -1)).sum(dim=-1)
        benefit_logits = self.benefit_net(stats)
        safe_scores = benefit_logits - self.price_weight * price.unsqueeze(-1) - self.price_margin
        branch_gates = torch.sigmoid(safe_scores / self.temperature)
        safety_scores = self.safety_net(stats).squeeze(-1) - self.price_weight * price - self.price_margin
        safety_gate = torch.sigmoid(safety_scores / self.temperature).unsqueeze(-1)
        return {
            "branch_gates": branch_gates,
            "safety_gate": safety_gate,
            "price": price,
            "safe_scores": safe_scores,
            "price_components": components,
        }


def valid_heads(d_model, requested_heads):
    for heads in range(min(requested_heads, d_model), 0, -1):
        if d_model % heads == 0:
            return heads
    return 1


class IMTS_SubModel(nn.Module):
    def __init__(self, configs):
        super(IMTS_SubModel, self).__init__()
        self.configs = configs
        self.hid_dim = configs.d_model

        self.te_dim = configs.apn_te_dim
        self.N = configs.enc_in
        self.P = configs.apn_npatch
        self.n_layer = configs.apn_nlayer
        self.attn_heads = configs.apn_attn_heads
        self.variant = getattr(configs, 'apn_research_variant', 'base')
        self.variant_tokens = set(self.variant.split('__'))
        self.operator_enabled = 'sapn_operator' in self.variant_tokens or 'operator_correction' in self.variant_tokens
        self.selector_enabled = 'sapn_selector' in self.variant_tokens or self.operator_enabled
        self.reliability_enabled = any(token in self.variant_tokens for token in [
            'reliability', 'relmix', 'reliability_lowrank', 'rel_lowrank'
        ])
        self.reliability_impute_enabled = any(token in self.variant_tokens for token in [
            'rel_impute', 'reliability_impute', 'soft_impute'
        ])
        self.residual_bias_center_enabled = any(token in self.variant_tokens for token in [
            'bias_center', 'median_center', 'spectral_bias_correct'
        ])

        self.dropout_rate = configs.dropout
        self.batch_size = None
        self.latest_aux_loss = None
        self.latest_selector_branch_gates = None
        self.latest_selector_safety_gate = None
        self.latest_selector_gates = None
        self.latest_selector_stats = None
        self.latest_selector_residual_norms = None
        self.latest_selector_uncertainty = None
        self.latest_selector_price = None
        self.latest_selector_safe_scores = None
        self.latest_selector_price_components = None
        self.latest_mechanism_residual_norm = None
        self.latest_reliability_stats = None
        self.latest_reliability_impute_weight = None
        self.latest_mcg_integral_gate = None
        self.latest_mcg_branch_residual_norms = None
        self.latest_mcg_boundary_penalty = None
        self.latest_mcg_gate_logits = None
        self.latest_mcg_local_pred = None
        self.latest_mcg_integral_pred = None
        self.selector_decoder_gate = None
        self.force_noop_selector = False
        self.reliability_impute_weight = float(getattr(configs, 'apn_reliability_impute_weight', 0.15))
        self.reliability_route_weight = float(getattr(configs, 'apn_reliability_route_weight', 0.0))

        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, self.te_dim - 1)

        patching_cls = AttentionPatchAggregation
        if 'multiscale' in self.variant_tokens:
            patching_cls = MultiScalePatchAggregation
        elif (
            'mechanism' in self.variant_tokens or
            'mnar' in self.variant_tokens or
            any(token.startswith('mechanism_') for token in self.variant_tokens)
        ):
            patching_cls = MechanismAwarePatchAggregation
        elif 'density' in self.variant_tokens:
            patching_cls = DensityAwarePatchAggregation

        self.patching = patching_cls(
            N=self.N,
            P=self.P,
            S=None,
            te_dim=self.te_dim,
            hid_dim=self.hid_dim,
            history=1.0,
            dropout_rate=self.dropout_rate
        )

        self.patch_pos_enc = PositionalEncoding(self.hid_dim, max_len=self.P)
        self.query_count = max(1, getattr(configs, 'apn_query_count', 4))
        n_queries = self.query_count if 'querymix' in self.variant_tokens else 1
        self.var_queries = nn.Parameter(torch.randn(1, self.N, n_queries, self.hid_dim))
        self.query_gate = nn.Linear(self.hid_dim, 1) if 'querymix' in self.variant_tokens else None
        uses_fourier = (
            'fourier' in self.variant_tokens or
            'gated_fourier' in self.variant_tokens or
            'adaptive_fourier' in self.variant_tokens or
            'fourier_varmix' in self.variant_tokens or
            'gated_fourier_varmix' in self.variant_tokens or
            'adaptive_fourier_varmix' in self.variant_tokens
        )
        if uses_fourier:
            self.fourier_modulator = FourierPatchModulator(
                self.P,
                self.hid_dim,
                self.dropout_rate,
                gated='gated_fourier' in self.variant_tokens or 'gated_fourier_varmix' in self.variant_tokens,
                adaptive='adaptive_fourier' in self.variant_tokens or 'adaptive_fourier_varmix' in self.variant_tokens,
                gate_init=getattr(configs, 'apn_fourier_init', -2.0)
            )
        else:
            self.fourier_modulator = None
        patch_modulators = []
        if 'patchconv' in self.variant_tokens:
            patch_modulators.append(PatchConvModulator(self.hid_dim, self.dropout_rate, getattr(configs, 'apn_fourier_init', -2.0)))
        if 'wavelet' in self.variant_tokens:
            patch_modulators.append(WaveletPatchModulator(self.hid_dim, self.dropout_rate, getattr(configs, 'apn_fourier_init', -2.0)))
        if 'trend' in self.variant_tokens:
            patch_modulators.append(TrendPatchModulator(self.hid_dim, self.dropout_rate, getattr(configs, 'apn_fourier_init', -2.0)))
        if 'seasonal' in self.variant_tokens or 'calendar' in self.variant_tokens:
            patch_modulators.append(SeasonalPatchModulator(self.P, self.hid_dim, self.dropout_rate, getattr(configs, 'apn_fourier_init', -2.0)))
        if 'lagdep' in self.variant_tokens or 'laggraph' in self.variant_tokens:
            patch_modulators.append(LaggedDependencyPatchModulator(self.N, self.hid_dim, self.dropout_rate, getattr(configs, 'apn_varmix_init', -3.0)))
        self.patch_modulators = nn.ModuleList(patch_modulators)
        heads = valid_heads(self.hid_dim, self.attn_heads)
        uses_attn_varmix = any(token in self.variant_tokens for token in ['varmix', 'gated_varmix', 'fourier_varmix', 'gated_fourier_varmix', 'adaptive_fourier_varmix'])
        self.var_mixer = nn.MultiheadAttention(self.hid_dim, heads, dropout=self.dropout_rate, batch_first=True) if uses_attn_varmix else None
        self.var_mixer_norm = nn.LayerNorm(self.hid_dim) if self.var_mixer is not None else None
        if any(token in self.variant_tokens for token in ['gated_varmix', 'fourier_varmix', 'gated_fourier_varmix', 'adaptive_fourier_varmix']):
            init_gate = getattr(configs, 'apn_varmix_init', -3.0)
            self.var_mixer_gate = nn.Parameter(torch.full((self.N, 1), float(init_gate)))
        else:
            self.var_mixer_gate = None
        variable_mixers = []
        if self.reliability_enabled:
            variable_mixers.append(ReliabilityLowRankVariableMixer(
                self.N,
                self.hid_dim,
                rank=4,
                dropout_rate=self.dropout_rate,
                gate_init=getattr(configs, 'apn_varmix_init', -3.0),
                source_scale=getattr(configs, 'apn_reliability_source_scale', 1.0),
                target_scale=getattr(configs, 'apn_reliability_target_scale', 1.0),
            ))
        if 'lowrank' in self.variant_tokens:
            variable_mixers.append(LowRankVariableMixer(self.N, self.hid_dim, rank=4, dropout_rate=self.dropout_rate, gate_init=getattr(configs, 'apn_varmix_init', -3.0)))
        if 'graph' in self.variant_tokens:
            variable_mixers.append(GraphVariableMixer(self.N, self.hid_dim, dropout_rate=self.dropout_rate, gate_init=getattr(configs, 'apn_varmix_init', -3.0)))
        if 'covmix' in self.variant_tokens:
            variable_mixers.append(CovarianceVariableMixer(self.hid_dim, dropout_rate=self.dropout_rate, gate_init=getattr(configs, 'apn_varmix_init', -3.0)))
        if 'segate' in self.variant_tokens:
            variable_mixers.append(SEVariableGate(self.hid_dim, dropout_rate=self.dropout_rate, gate_init=getattr(configs, 'apn_varmix_init', -3.0)))
        self.variable_mixers = nn.ModuleList(variable_mixers)
        self.aggregation_norm = nn.LayerNorm(self.hid_dim)
        selector_rng_state = torch.get_rng_state() if self.selector_enabled else None
        if self.selector_enabled:
            self.selector_level = getattr(configs, 'apn_selector_level', 'variable')
            self.selector_l1 = float(getattr(configs, 'apn_selector_l1', 1e-4))
            self.selector_residual = float(getattr(configs, 'apn_selector_residual', 1e-4))
            self.selector_entropy = float(getattr(configs, 'apn_selector_entropy', 0.0))
            self.selector_scale = float(getattr(configs, 'apn_selector_scale', 1.0))
            self.selector_safety_enabled = bool(getattr(configs, 'apn_selector_safety', 0))
            self.selector_branch_dropout = float(getattr(configs, 'apn_selector_branch_dropout', 0.0))
            self.selector_stat_control = str(getattr(configs, 'apn_selector_stat_control', 'real')).lower()
            self.selector_uncertainty_weight = float(getattr(configs, 'apn_selector_uncertainty_weight', 0.0))
            self.selector_uncertainty_mode = str(getattr(configs, 'apn_selector_uncertainty_mode', 'none')).lower()
            self.selector_trust_weight = float(getattr(configs, 'apn_selector_trust_weight', 0.0))
            self.selector_trust_cap = float(getattr(configs, 'apn_selector_trust_cap', 0.0))
            self.selector_mass_weight = float(getattr(configs, 'apn_selector_mass_weight', 0.0))
            self.selector_mass_min = float(getattr(configs, 'apn_selector_mass_min', 0.0))
            self.selector_mass_max = float(getattr(configs, 'apn_selector_mass_max', 1.0))
            branch_mask = torch.ones(4)
            branch_spec = str(getattr(configs, 'apn_selector_branch_mask', 'all')).lower()
            if branch_spec not in ['all', '*', '']:
                branch_mask.zero_()
                branch_names = {'spectral': 0, 'sp': 0, 'trend': 1, 'tr': 1, 'cov': 2, 'covmix': 2, 'decoder': 3, 'dec': 3, 'resdec': 3}
                for name in branch_spec.replace('+', ',').split(','):
                    name = name.strip()
                    if name in branch_names:
                        branch_mask[branch_names[name]] = 1.0
            self.register_buffer('selector_branch_mask', branch_mask.view(1, 1, 4))
            self.structural_stats = StructuralStats(self.N)
            if self.operator_enabled:
                self.structure_selector = OperatorCorrectionGate(
                    StructuralStats.stat_dim,
                    hidden_dim=max(4, getattr(configs, 'apn_selector_hidden', 32)),
                    dropout_rate=self.dropout_rate,
                    gate_init=getattr(configs, 'apn_selector_init', -2.0),
                    temperature=getattr(configs, 'apn_selector_temperature', 1.0),
                    price_weight=getattr(configs, 'apn_operator_price_weight', 1.0),
                    price_margin=getattr(configs, 'apn_operator_price_margin', 0.0),
                    price_init=getattr(configs, 'apn_operator_price_init', 0.1),
                )
            else:
                self.structure_selector = StructureSelector(
                    StructuralStats.stat_dim,
                    hidden_dim=max(4, getattr(configs, 'apn_selector_hidden', 32)),
                    dropout_rate=self.dropout_rate,
                    gate_init=getattr(configs, 'apn_selector_init', -2.0),
                    temperature=getattr(configs, 'apn_selector_temperature', 1.0)
                )
            if self.selector_safety_enabled:
                self.selector_safety = StructureSelector(
                    StructuralStats.stat_dim,
                    hidden_dim=max(4, getattr(configs, 'apn_selector_hidden', 32)),
                    dropout_rate=self.dropout_rate,
                    gate_init=getattr(configs, 'apn_selector_safety_init', -3.0),
                    temperature=getattr(configs, 'apn_selector_safety_temperature', 1.0)
                )
                self.selector_safety.net[-1] = nn.Linear(self.selector_safety.net[-1].in_features, 1)
                nn.init.zeros_(self.selector_safety.net[-1].weight)
                nn.init.constant_(self.selector_safety.net[-1].bias, float(getattr(configs, 'apn_selector_safety_init', -3.0)))
            else:
                self.selector_safety = None
            self.selector_spectral = SpectralResidualBranch(self.P, self.hid_dim, self.dropout_rate)
            self.selector_trend = TrendResidualBranch(self.hid_dim, self.dropout_rate)
            dep_branch = str(getattr(configs, 'apn_operator_dep_branch', 'cov')).lower()
            if dep_branch in ['graph', 'stable_graph']:
                self.selector_covmix = StableGraphResidualBranch(self.N, self.hid_dim, self.dropout_rate)
            elif dep_branch in ['hybrid', 'cov_graph', 'graph_cov']:
                self.selector_covmix = HybridDependencyResidualBranch(self.N, self.hid_dim, self.dropout_rate)
            else:
                self.selector_covmix = CovarianceResidualBranch(self.hid_dim, self.dropout_rate)
            self.selector_patch_norm = nn.LayerNorm(self.hid_dim)
            self.selector_var_norm = nn.LayerNorm(self.hid_dim)
            self.selector_decoder_skip = nn.Linear(self.hid_dim, 1)
        else:
            self.selector_level = 'variable'
            self.selector_l1 = 0.0
            self.selector_residual = 0.0
            self.selector_entropy = 0.0
            self.selector_scale = 0.0
            self.selector_safety_enabled = False
            self.selector_branch_dropout = 0.0
            self.selector_stat_control = 'real'
            self.selector_uncertainty_weight = 0.0
            self.selector_uncertainty_mode = 'none'
            self.selector_trust_weight = 0.0
            self.selector_trust_cap = 0.0
            self.selector_mass_weight = 0.0
            self.selector_mass_min = 0.0
            self.selector_mass_max = 1.0
            self.operator_enabled = False
            self.register_buffer('selector_branch_mask', torch.ones(1, 1, 4))
            self.structural_stats = None
            self.structure_selector = None
            self.selector_safety = None
            self.selector_spectral = None
            self.selector_trend = None
            self.selector_covmix = None
            self.selector_patch_norm = None
            self.selector_var_norm = None
            self.selector_decoder_skip = None
        self.reliability_stats = StructuralStats(self.N) if self.reliability_enabled else None
        if selector_rng_state is not None:
            torch.set_rng_state(selector_rng_state)
        self.decoder_film = nn.Sequential(
            nn.Linear(self.te_dim, self.hid_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hid_dim * 2, self.hid_dim * 2)
        ) if 'filmdec' in self.variant_tokens else None
        self.decoder_skip = nn.Linear(self.hid_dim, 1) if 'resdec' in self.variant_tokens else None
        self.kinematic_decoder_enabled = any(token in self.variant_tokens for token in [
            'kinematic_decoder', 'kinodec', 'motion_decoder', 'trajectory_decoder'
        ])
        if self.kinematic_decoder_enabled:
            self.kinematic_slope_scale = float(getattr(configs, 'apn_kinematic_slope_scale', 0.25))
            self.kinematic_accel_scale = float(getattr(configs, 'apn_kinematic_accel_scale', 0.05))
            self.kinematic_slope_clip = float(getattr(configs, 'apn_kinematic_slope_clip', 10.0))
            self.kinematic_head = nn.Sequential(
                nn.LayerNorm(self.hid_dim),
                nn.Linear(self.hid_dim, self.hid_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hid_dim, 3),
            )
            nn.init.zeros_(self.kinematic_head[-1].weight)
            nn.init.zeros_(self.kinematic_head[-1].bias)
            self.kinematic_head[-1].bias.data[2] = float(getattr(configs, 'apn_kinematic_gate_init', -2.0))
        else:
            self.kinematic_slope_scale = 0.0
            self.kinematic_accel_scale = 0.0
            self.kinematic_slope_clip = 0.0
            self.kinematic_head = None

        self.decoder = nn.Sequential(
            nn.Linear(self.hid_dim + self.te_dim, self.hid_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hid_dim * 2, 1)
        )
        self.mechanism_residual_l2 = float(getattr(configs, 'apn_mcr_residual_l2', 0.0))
        self.alias_remainder_l2 = float(getattr(configs, 'apn_alias_remainder_l2', 0.0))
        self.mcr_state_mode = str(getattr(configs, 'apn_mcr_state_mode', 'state')).lower()
        self.mcg_boundary_weight = float(getattr(configs, 'apn_mcg_boundary_weight', 0.0))
        self.mcg_supervised_weight = float(getattr(configs, 'apn_mcg_supervised_weight', 0.0))
        self.mcg_supervised_temperature = float(getattr(configs, 'apn_mcg_supervised_temperature', 0.10))
        self.mcg_supervised_margin = float(getattr(configs, 'apn_mcg_supervised_margin', 0.0))
        self.mcg_supervised_min_confidence = float(getattr(configs, 'apn_mcg_supervised_min_confidence', 0.0))
        self.mechanism_residual_head = None
        if any(token in self.variant_tokens for token in ['raw_state_residual', 'raw_residual', 'rr']):
            self.mechanism_residual_head = RawStateResidualHead(
                self.hid_dim,
                self.te_dim,
                dropout_rate=self.dropout_rate,
                gate_init=getattr(configs, 'apn_mcr_gate_init', -1.6),
            )
        elif any(token in self.variant_tokens for token in ['alias_larm', 'alias_lmr', 'af_larm', 'af_lmr']):
            self.mechanism_residual_head = AliasFactoredLevelMechanismResidualHead(
                self.hid_dim,
                self.te_dim,
                dropout_rate=self.dropout_rate,
                gate_init=getattr(configs, 'apn_mcr_gate_init', -1.6),
                integral_centers=getattr(configs, 'apn_mcr_integral_centers', 6),
                integral_width=getattr(configs, 'apn_mcr_integral_width', 0.18),
                remainder_scale=getattr(configs, 'apn_alias_remainder_scale', 0.25),
                alias_score_gate_weight=getattr(configs, 'apn_alias_score_gate_weight', 1.0),
                variance_price_weight=getattr(configs, 'apn_alias_variance_price_weight', 0.0),
                scale_price_weight=getattr(configs, 'apn_alias_scale_price_weight', 0.0),
            )
        elif any(token in self.variant_tokens for token in ['level_mechanism_residual', 'level_mcr', 'lmr']):
            self.mechanism_residual_head = LevelMechanismResidualHead(
                self.hid_dim,
                self.te_dim,
                dropout_rate=self.dropout_rate,
                gate_init=getattr(configs, 'apn_mcr_gate_init', -1.6),
                integral_centers=getattr(configs, 'apn_mcr_integral_centers', 6),
                integral_width=getattr(configs, 'apn_mcr_integral_width', 0.18),
            )
        elif any(token in self.variant_tokens for token in ['lamr', 'motion_anchor', 'level_motion_residual']):
            self.mechanism_residual_head = LevelAnchoredMotionResidualHead(
                self.hid_dim,
                self.te_dim,
                dropout_rate=self.dropout_rate,
                gate_init=getattr(configs, 'apn_mcr_gate_init', -1.6),
                motion_scale=getattr(configs, 'apn_lamr_motion_scale', 1.0),
                correction_scale=getattr(configs, 'apn_lamr_correction_scale', 0.10),
            )
        elif any(token in self.variant_tokens for token in ['level_value_residual', 'lvr']):
            self.mechanism_residual_head = LevelValueResidualHead(
                self.hid_dim,
                self.te_dim,
                dropout_rate=self.dropout_rate,
                gate_init=getattr(configs, 'apn_mcr_gate_init', -1.6),
            )
        elif any(token in self.variant_tokens for token in ['adaptive_mcr_mcir', 'mechanism_gated_residual', 'mcg']):
            self.mechanism_residual_head = AdaptiveMechanismResidualHead(
                self.hid_dim,
                self.te_dim,
                dropout_rate=self.dropout_rate,
                gate_init=getattr(configs, 'apn_mcr_gate_init', -1.6),
                switch_gate_init=getattr(configs, 'apn_mcg_gate_init', -2.0),
                warmup_stage=getattr(configs, 'apn_mcg_warmup_stage', 0),
                warmup_blend=getattr(configs, 'apn_mcg_warmup_blend', 0.5),
                integral_centers=getattr(configs, 'apn_mcr_integral_centers', 6),
                integral_width=getattr(configs, 'apn_mcr_integral_width', 0.18),
            )
        elif any(token in self.variant_tokens for token in ['rams', 'rams_residual', 'reliability_adaptive_residual']):
            self.mechanism_residual_head = ReliabilityAdaptiveResidualHead(
                self.hid_dim,
                self.te_dim,
                dropout_rate=self.dropout_rate,
                gate_init=getattr(configs, 'apn_mcr_gate_init', -1.8),
                integral_centers=getattr(configs, 'apn_mcr_integral_centers', 6),
                integral_width=getattr(configs, 'apn_mcr_integral_width', 0.18),
                branch_temperature=getattr(configs, 'apn_selector_temperature', 1.0),
            )
        elif any(token in self.variant_tokens for token in ['mechanism_residual', 'mcr']):
            self.mechanism_residual_head = MechanismConditionedResidualHead(
                self.hid_dim,
                self.te_dim,
                dropout_rate=self.dropout_rate,
                use_integral=False,
                gate_init=getattr(configs, 'apn_mcr_gate_init', -1.6),
                integral_centers=getattr(configs, 'apn_mcr_integral_centers', 6),
                integral_width=getattr(configs, 'apn_mcr_integral_width', 0.18),
            )
        if any(token in self.variant_tokens for token in ['mechanism_integral_residual', 'mcir']):
            self.mechanism_residual_head = MechanismConditionedResidualHead(
                self.hid_dim,
                self.te_dim,
                dropout_rate=self.dropout_rate,
                use_integral=True,
                gate_init=getattr(configs, 'apn_mcr_gate_init', -1.6),
                integral_centers=getattr(configs, 'apn_mcr_integral_centers', 6),
                integral_width=getattr(configs, 'apn_mcr_integral_width', 0.18),
            )

    def selector_uncertainty_score(self, stats):
        if self.selector_uncertainty_weight <= 0.0 or self.selector_uncertainty_mode in ['none', 'off', '0']:
            return None
        density = stats[..., 0]
        missing = stats[..., 1]
        heterogeneity = stats[..., 2]
        irregularity = stats[..., 3]
        spectral_entropy = stats[..., 4]
        patch_smoothness = stats[..., 6]
        quadrature_uncertainty = stats[..., 7]
        recency = stats[..., 8] if stats.size(-1) > 8 else torch.zeros_like(missing)
        observed_span = stats[..., 9] if stats.size(-1) > 9 else torch.ones_like(missing)
        mode = self.selector_uncertainty_mode
        if mode in ['interp', 'interpolation']:
            score = 0.45 * missing + 0.35 * irregularity + 0.20 * quadrature_uncertainty
        elif mode in ['quad', 'quadrature']:
            score = quadrature_uncertainty
        elif mode in ['robust', 'dro', 'distributional']:
            score = 0.20 * (missing + heterogeneity + irregularity + spectral_entropy + recency)
        elif mode in ['mechanism', 'mnar', 'sampling']:
            score = (0.30 * missing + 0.25 * heterogeneity + 0.25 * recency + 0.20 * (1.0 - observed_span))
        elif mode in ['smooth', 'local']:
            score = 0.40 * patch_smoothness + 0.30 * irregularity + 0.30 * quadrature_uncertainty
        else:
            score = (1.0 - density + heterogeneity + irregularity + spectral_entropy + patch_smoothness + quadrature_uncertainty + recency + (1.0 - observed_span)) / 8.0
        return score.clamp(0.0, 1.0)

    def apply_selector_stat_control(self, stats):
        mode = self.selector_stat_control
        if mode in ['real', 'none', 'off', '0']:
            return stats
        if mode in ['shuffle', 'shuffled']:
            flat = stats.reshape(-1, stats.shape[-1])
            if flat.shape[0] <= 1:
                return stats
            perm = torch.randperm(flat.shape[0], device=stats.device)
            return flat[perm].reshape_as(stats)
        if mode in ['random', 'rand']:
            return torch.rand_like(stats)
        if mode in ['density_only', 'density']:
            controlled = torch.zeros_like(stats)
            controlled[..., 0] = stats[..., 0]
            return controlled
        if mode in ['uncertainty_only', 'uncertainty', 'missing_irregular']:
            controlled = torch.zeros_like(stats)
            controlled[..., 1] = stats[..., 1]
            controlled[..., 2] = stats[..., 2]
            controlled[..., 3] = stats[..., 3]
            controlled[..., 7] = stats[..., 7]
            if stats.size(-1) > 8:
                controlled[..., 8] = stats[..., 8]
            if stats.size(-1) > 9:
                controlled[..., 9] = stats[..., 9]
            return controlled
        if mode in ['global', 'global_mean']:
            global_stats = stats.mean(dim=(0, 1), keepdim=True)
            return global_stats.expand_as(stats)
        return stats

    def selector_trust_penalty(self, residual_effect):
        if self.selector_trust_weight <= 0.0:
            return residual_effect.new_zeros(())
        energy = residual_effect.pow(2).mean()
        cap = residual_effect.new_tensor(max(self.selector_trust_cap, 0.0))
        return F.relu(energy - cap).pow(2) * self.selector_trust_weight

    def selector_mass_penalty(self, selector_gates):
        if self.selector_mass_weight <= 0.0:
            return selector_gates.new_zeros(())
        if self.selector_mass_max < self.selector_mass_min:
            return selector_gates.new_zeros(())
        mass = selector_gates.mean()
        lower = F.relu(selector_gates.new_tensor(self.selector_mass_min) - mass)
        upper = F.relu(mass - selector_gates.new_tensor(self.selector_mass_max))
        return (lower.pow(2) + upper.pow(2)) * self.selector_mass_weight

    def LearnableTE(self, tt):
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def kinematic_decoder_blend(self, outputs_raw, h_final, x, x_mask, x_mark, y_time):
        if self.kinematic_head is None:
            return outputs_raw
        dtype = outputs_raw.dtype
        device = outputs_raw.device
        B, L_obs, N_vars = x.shape
        mask = x_mask.to(device=device, dtype=dtype)
        x_val = x.to(device=device, dtype=dtype)
        if x_mark.dim() == 2:
            t_obs = x_mark.to(device=device, dtype=dtype).unsqueeze(-1)
        else:
            t_obs = x_mark[:, :, [0]].to(device=device, dtype=dtype)
        t_obs = t_obs.expand(-1, -1, N_vars)

        count = mask.sum(dim=1).clamp_min(1.0)
        mean_t = (t_obs * mask).sum(dim=1) / count
        mean_x = (x_val * mask).sum(dim=1) / count
        centered_t = (t_obs - mean_t.unsqueeze(1)) * mask
        centered_x = (x_val - mean_x.unsqueeze(1)) * mask
        denom = centered_t.pow(2).sum(dim=1).clamp_min(1e-4)
        slope = (centered_t * centered_x).sum(dim=1) / denom
        slope = slope.clamp(-self.kinematic_slope_clip, self.kinematic_slope_clip)

        positions = torch.arange(L_obs, device=device, dtype=dtype).view(1, L_obs, 1)
        last_idx = (mask * positions).argmax(dim=1).long()
        gather_idx = last_idx.unsqueeze(1)
        last_x = x_val.gather(1, gather_idx).squeeze(1)
        last_t = t_obs.gather(1, gather_idx).squeeze(1)
        has_obs = (x_mask.to(device=device).sum(dim=1) > 0).to(dtype=dtype)
        level = has_obs * last_x + (1.0 - has_obs) * mean_x
        last_t = has_obs * last_t + (1.0 - has_obs) * mean_t

        coeff = self.kinematic_head(h_final)
        delta_velocity = self.kinematic_slope_scale * torch.tanh(coeff[..., 0])
        acceleration = self.kinematic_accel_scale * torch.tanh(coeff[..., 1])
        blend_gate = torch.sigmoid(coeff[..., 2]).view(B, N_vars, 1, 1)

        dt = (y_time.squeeze(-1).to(dtype=dtype) - last_t.unsqueeze(-1)).clamp(-1.0, 1.0)
        prior = (
            level.unsqueeze(-1)
            + (slope + delta_velocity).unsqueeze(-1) * dt
            + 0.5 * acceleration.unsqueeze(-1) * dt.pow(2)
        ).unsqueeze(-1)
        return outputs_raw + blend_gate * (prior - outputs_raw)

    def IMTS_Model_Logic(self, x_with_te, mask_stacked, time_features_stacked):
        B = self.batch_size
        N_vars = self.N
        self.latest_aux_loss = None
        self.latest_selector_branch_gates = None
        self.latest_selector_safety_gate = None
        self.latest_selector_gates = None
        self.latest_selector_stats = None
        self.latest_selector_residual_norms = None
        self.latest_selector_uncertainty = None
        self.latest_selector_price = None
        self.latest_selector_safe_scores = None
        self.latest_selector_price_components = None
        self.latest_mechanism_residual_norm = None
        self.latest_reliability_stats = None
        self.latest_mcg_integral_gate = None
        self.latest_mcg_branch_residual_norms = None
        self.latest_mcg_boundary_penalty = None
        self.latest_mcg_gate_logits = None
        self.latest_mcg_local_pred = None
        self.latest_mcg_integral_pred = None
        self.selector_decoder_gate = None
        h_patches_stacked = self.patching(time_features_stacked, x_with_te, mask_stacked)
        h_patches_stacked_pe = self.patch_pos_enc(h_patches_stacked)
        h_patches_updated = h_patches_stacked_pe.view(B, N_vars, self.P, self.hid_dim)
        if self.fourier_modulator is not None:
            h_patches_updated = self.fourier_modulator(h_patches_updated)
        for modulator in self.patch_modulators:
            h_patches_updated = modulator(h_patches_updated)
        aux_terms = []
        selector_gates = None
        if self.selector_enabled:
            selector_stats = self.structural_stats(h_patches_updated, mask_stacked, time_features_stacked, B, N_vars)
            if self.selector_level == 'sample':
                selector_input = selector_stats.mean(dim=1, keepdim=True).expand(-1, N_vars, -1)
            else:
                selector_input = selector_stats
            selector_input = self.apply_selector_stat_control(selector_input)
            safety_gate = None
            if self.operator_enabled:
                gate_out = self.structure_selector(selector_input)
                selector_branch_gates = gate_out["branch_gates"] * self.selector_scale
                safety_gate = gate_out["safety_gate"]
                self.latest_selector_price = gate_out["price"].detach()
                self.latest_selector_safe_scores = gate_out["safe_scores"].detach()
                self.latest_selector_price_components = gate_out["price_components"].detach()
            else:
                selector_branch_gates = self.structure_selector(selector_input) * self.selector_scale
            selector_gates = selector_branch_gates
            if self.selector_safety is not None:
                extra_safety_gate = self.selector_safety(selector_input)
                safety_gate = extra_safety_gate if safety_gate is None else safety_gate * extra_safety_gate
            uncertainty_score = self.selector_uncertainty_score(selector_input)
            if uncertainty_score is not None:
                uncertainty_multiplier = torch.exp(-self.selector_uncertainty_weight * uncertainty_score).unsqueeze(-1)
                safety_gate = uncertainty_multiplier if safety_gate is None else safety_gate * uncertainty_multiplier
                self.latest_selector_uncertainty = uncertainty_score.detach()
            if safety_gate is not None:
                selector_gates = selector_gates * safety_gate
            selector_gates = selector_gates * self.selector_branch_mask.to(selector_gates.device)
            if self.force_noop_selector:
                selector_gates = torch.zeros_like(selector_gates)
                if safety_gate is not None:
                    safety_gate = torch.zeros_like(safety_gate)
            if self.training and self.selector_branch_dropout > 0.0:
                keep_prob = 1.0 - self.selector_branch_dropout
                keep = torch.bernoulli(selector_gates.new_full(selector_gates.shape, keep_prob)) / max(keep_prob, 1e-6)
                selector_gates = selector_gates * keep
            aux_terms.append(self.selector_mass_penalty(selector_gates))
            self.latest_selector_branch_gates = selector_branch_gates.detach()
            self.latest_selector_safety_gate = safety_gate.detach() if safety_gate is not None else None
            self.latest_selector_gates = selector_gates.detach()
            self.latest_selector_stats = selector_input.detach()

            spectral_gate = selector_gates[..., 0].unsqueeze(-1).unsqueeze(-1)
            trend_gate = selector_gates[..., 1].unsqueeze(-1).unsqueeze(-1)
            spectral_residual = self.selector_spectral(h_patches_updated)
            trend_residual = self.selector_trend(h_patches_updated)
            patch_residual = spectral_gate * spectral_residual + trend_gate * trend_residual
            h_patches_updated = self.selector_patch_norm(h_patches_updated + patch_residual)
            self.latest_selector_residual_norms = {
                "patch": patch_residual.pow(2).mean().detach()
            }
            aux_terms.append(selector_gates.mean() * self.selector_l1)
            aux_terms.append(patch_residual.pow(2).mean() * self.selector_residual)
            aux_terms.append(self.selector_trust_penalty(patch_residual))
            if self.selector_entropy > 0.0:
                gate_entropy = -(selector_gates * torch.log(selector_gates + 1e-6) +
                                 (1.0 - selector_gates) * torch.log(1.0 - selector_gates + 1e-6)).mean()
                aux_terms.append(gate_entropy * self.selector_entropy)
        attn_scores = torch.matmul(self.var_queries, h_patches_updated.transpose(-1, -2)) * (self.hid_dim ** -0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        h_final = torch.matmul(attn_weights, h_patches_updated)
        if self.query_gate is not None:
            query_logits = self.query_gate(h_final).squeeze(-1)
            query_weights = F.softmax(query_logits, dim=-1).unsqueeze(-1)
            h_final = (h_final * query_weights).sum(dim=-2)
        else:
            h_final = h_final.squeeze(-2)
        if self.var_mixer is not None:
            mixed, _ = self.var_mixer(h_final, h_final, h_final, need_weights=False)
            if self.var_mixer_gate is not None:
                gate = torch.sigmoid(self.var_mixer_gate).view(1, N_vars, 1)
                h_final = self.var_mixer_norm(h_final + gate * mixed)
            else:
                h_final = self.var_mixer_norm(h_final + mixed)
        reliability_stats = None
        if self.reliability_stats is not None:
            reliability_stats = self.reliability_stats(h_patches_updated, mask_stacked, time_features_stacked, B, N_vars)
            self.latest_reliability_stats = reliability_stats.detach()
        for mixer in self.variable_mixers:
            if getattr(mixer, "requires_stats", False):
                h_final = mixer(h_final, reliability_stats)
            else:
                h_final = mixer(h_final)
            route_penalty = getattr(mixer, "latest_route_penalty", None)
            if self.training and self.reliability_route_weight > 0.0 and route_penalty is not None:
                aux_terms.append(route_penalty * self.reliability_route_weight)
        if self.selector_enabled and selector_gates is not None:
            cov_gate = selector_gates[..., 2].unsqueeze(-1)
            cov_residual = self.selector_covmix(h_final)
            h_final = self.selector_var_norm(h_final + cov_gate * cov_residual)
            self.selector_decoder_gate = selector_gates[..., 3].unsqueeze(-1)
            if self.latest_selector_residual_norms is not None:
                self.latest_selector_residual_norms["covariance"] = (cov_gate * cov_residual).pow(2).mean().detach()
            aux_terms.append((cov_gate * cov_residual).pow(2).mean() * self.selector_residual)
            aux_terms.append(self.selector_trust_penalty(cov_gate * cov_residual))
            if aux_terms:
                self.latest_aux_loss = torch.stack([term.reshape(()) for term in aux_terms]).sum()
        h_final = self.aggregation_norm(h_final)
        return h_final

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor, x_mask: torch.Tensor,
                y_mark: torch.Tensor, train_stage=None) -> torch.Tensor:
        B, L_obs, N_vars_from_X = x.shape
        self.batch_size = B

        time_features = x_mark[:, :, [0]]

        x_for_patch = x
        mask_for_patch = x_mask
        self.latest_reliability_impute_weight = None
        if self.reliability_impute_enabled:
            obs_count = x_mask.sum(dim=1, keepdim=True)
            density = (obs_count / max(float(L_obs), 1.0)).clamp(0.0, 1.0)
            var_mean = (x * x_mask).sum(dim=1, keepdim=True) / obs_count.clamp_min(1.0)
            global_count = x_mask.sum(dim=(1, 2), keepdim=True)
            global_mean = (x * x_mask).sum(dim=(1, 2), keepdim=True) / global_count.clamp_min(1.0)
            fill_value = density * var_mean + (1.0 - density) * global_mean
            missing = 1.0 - x_mask
            soft_weight = (self.reliability_impute_weight * missing * (1.0 - density)).clamp(0.0, 0.5)
            x_for_patch = x * x_mask + fill_value * missing
            mask_for_patch = (x_mask + soft_weight).clamp(0.0, 1.0)
            self.latest_reliability_impute_weight = soft_weight.detach()

        X_stacked = x_for_patch.permute(0, 2, 1).reshape(B * N_vars_from_X, L_obs, 1)
        mask_stacked = mask_for_patch.permute(0, 2, 1).reshape(B * N_vars_from_X, L_obs, 1)

        time_features_stacked = time_features.unsqueeze(1).expand(
            B, N_vars_from_X, L_obs, 1
        ).reshape(B * N_vars_from_X, L_obs, 1)

        te_his = self.LearnableTE(time_features_stacked)
        X_with_te = torch.cat([X_stacked, te_his], dim=-1)

        h_final = self.IMTS_Model_Logic(X_with_te, mask_stacked, time_features_stacked)

        # 解码器部�?
        time_steps_to_predict = y_mark[:, :, [0]]
        L_pred = time_steps_to_predict.shape[1]
        h_expanded = h_final.unsqueeze(dim=-2).expand(-1, -1, L_pred, -1)
        time_steps_to_predict_exp = time_steps_to_predict.view(B, 1, L_pred, 1).expand(-1, N_vars_from_X, -1, -1)
        te_pred = self.LearnableTE(time_steps_to_predict_exp)
        if self.decoder_film is not None:
            film = self.decoder_film(te_pred)
            gamma, beta = torch.chunk(film, 2, dim=-1)
            h_expanded = h_expanded * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)
        decoder_input = torch.cat([h_expanded, te_pred], dim=-1)
        outputs_raw = self.decoder(decoder_input)
        if self.kinematic_decoder_enabled:
            outputs_raw = self.kinematic_decoder_blend(
                outputs_raw,
                h_final,
                x,
                x_mask,
                x_mark,
                time_steps_to_predict_exp,
            )
        if self.decoder_skip is not None:
            outputs_raw = outputs_raw + self.decoder_skip(h_expanded)
        if self.selector_enabled and self.selector_decoder_skip is not None and self.selector_decoder_gate is not None:
            decoder_residual = self.selector_decoder_skip(h_expanded)
            outputs_raw = outputs_raw + self.selector_decoder_gate.unsqueeze(-2) * decoder_residual
            if self.latest_selector_residual_norms is not None:
                decoder_effect = self.selector_decoder_gate.unsqueeze(-2) * decoder_residual
                self.latest_selector_residual_norms["decoder"] = decoder_effect.pow(2).mean().detach()
                decoder_aux = decoder_effect.pow(2).mean() * self.selector_residual + self.selector_trust_penalty(decoder_effect)
                if self.latest_aux_loss is None:
                    self.latest_aux_loss = decoder_aux
                else:
                    self.latest_aux_loss = self.latest_aux_loss + decoder_aux
        if self.mechanism_residual_head is not None:
            residual_state = h_expanded
            if self.mcr_state_mode in ['prediction_only', 'pred_only', 'forecast_only']:
                residual_state = torch.zeros_like(h_expanded)
            if isinstance(self.mechanism_residual_head, LevelAnchoredMotionResidualHead):
                mechanism_residual = self.mechanism_residual_head(
                    residual_state,
                    te_pred,
                    outputs_raw,
                    x,
                    x_mask,
                    x_mark,
                    train_stage=train_stage,
                    y_time=time_steps_to_predict_exp,
                )
            else:
                mechanism_residual = self.mechanism_residual_head(
                    residual_state,
                    te_pred,
                    outputs_raw,
                    x,
                    x_mask,
                    x_mark,
                    train_stage=train_stage,
                )
            if self.residual_bias_center_enabled:
                center = mechanism_residual.flatten(start_dim=1).median(dim=1).values.view(B, 1, 1, 1)
                mechanism_residual = mechanism_residual - center
            outputs_raw = outputs_raw + mechanism_residual
            self.latest_mechanism_residual_norm = mechanism_residual.pow(2).mean().detach()
            if isinstance(self.mechanism_residual_head, AdaptiveMechanismResidualHead):
                self.latest_mcg_integral_gate = self.mechanism_residual_head.latest_integral_gate
                self.latest_mcg_branch_residual_norms = self.mechanism_residual_head.latest_branch_residual_norms
                self.latest_mcg_boundary_penalty = self.mechanism_residual_head.latest_boundary_penalty
                self.latest_mcg_gate_logits = self.mechanism_residual_head.latest_gate_logits
                local_residual = self.mechanism_residual_head.latest_local_residual
                integral_residual = self.mechanism_residual_head.latest_integral_residual
                if local_residual is not None and integral_residual is not None:
                    self.latest_mcg_local_pred = (outputs_raw - mechanism_residual + local_residual).squeeze(-1).permute(0, 2, 1)
                    self.latest_mcg_integral_pred = (outputs_raw - mechanism_residual + integral_residual).squeeze(-1).permute(0, 2, 1)
            if self.training and self.mechanism_residual_l2 > 0.0:
                residual_aux = mechanism_residual.pow(2).mean() * self.mechanism_residual_l2
                if self.latest_aux_loss is None:
                    self.latest_aux_loss = residual_aux
                else:
                    self.latest_aux_loss = self.latest_aux_loss + residual_aux
            remainder = getattr(self.mechanism_residual_head, 'latest_remainder_residual_train', None)
            if self.training and self.alias_remainder_l2 > 0.0 and remainder is not None:
                remainder_aux = remainder.pow(2).mean() * self.alias_remainder_l2
                if self.latest_aux_loss is None:
                    self.latest_aux_loss = remainder_aux
                else:
                    self.latest_aux_loss = self.latest_aux_loss + remainder_aux
            if self.training and self.mcg_boundary_weight > 0.0 and self.latest_mcg_boundary_penalty is not None:
                boundary_aux = self.latest_mcg_boundary_penalty * self.mcg_boundary_weight
                if self.latest_aux_loss is None:
                    self.latest_aux_loss = boundary_aux
                else:
                    self.latest_aux_loss = self.latest_aux_loss + boundary_aux
        outputs = outputs_raw.squeeze(-1).permute(0, 2, 1)
        return outputs

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name

        self.model = IMTS_SubModel(configs)

    def mcg_supervised_gate_loss(self, y, y_mask, f_dim):
        if self.model.mcg_supervised_weight <= 0.0:
            return None
        gate_logits = self.model.latest_mcg_gate_logits
        local_pred = self.model.latest_mcg_local_pred
        integral_pred = self.model.latest_mcg_integral_pred
        if gate_logits is None or local_pred is None or integral_pred is None or y is None:
            return None
        logits = gate_logits.squeeze(-1).permute(0, 2, 1)[:, :, f_dim:]
        local_pred = local_pred[:, :, f_dim:]
        integral_pred = integral_pred[:, :, f_dim:]
        target_y = y[:, :, f_dim:]
        if y_mask is None:
            mask = torch.ones_like(target_y)
        else:
            mask = y_mask[:, :, f_dim:].to(dtype=target_y.dtype)
        local_loss = (local_pred - target_y).pow(2)
        integral_loss = (integral_pred - target_y).pow(2)
        scale = (local_loss + integral_loss).detach().clamp_min(1e-6)
        temperature = max(self.model.mcg_supervised_temperature, 1e-4)
        margin = self.model.mcg_supervised_margin
        target = torch.sigmoid(((local_loss - integral_loss).detach() - margin) / (temperature * scale + 1e-6))
        confidence = (target - 0.5).abs() * 2.0
        min_confidence = max(self.model.mcg_supervised_min_confidence, 0.0)
        if min_confidence > 0.0:
            mask = mask * (confidence >= min_confidence).to(dtype=mask.dtype)
        sample_weight = confidence * mask
        denom = sample_weight.sum().clamp_min(1.0)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        return bce.mul(sample_weight).sum() / denom * self.model.mcg_supervised_weight

    def trajectory_difference_aux_loss(self, pred, true, mask, y_mark):
        first_weight = float(getattr(self.configs, "apn_diff_loss_weight", 0.0))
        second_weight = float(getattr(self.configs, "apn_diff2_loss_weight", 0.0))
        if first_weight <= 0.0 and second_weight <= 0.0:
            return None
        if pred is None or true is None or pred.size(1) < 2:
            return None
        if mask is None:
            mask = torch.ones_like(true)
        else:
            mask = mask.to(dtype=true.dtype)
        if y_mark is None:
            dt = pred.new_ones(pred.size(0), pred.size(1) - 1, 1)
        else:
            if y_mark.dim() == 2:
                t = y_mark.unsqueeze(-1)
            else:
                t = y_mark[..., :1]
            t = t.to(device=pred.device, dtype=pred.dtype)
            dt = (t[:, 1:] - t[:, :-1]).abs().clamp_min(1e-4)
        beta = max(float(getattr(self.configs, "apn_diff_loss_beta", 0.05)), 1e-6)
        aux_terms = []
        pair_mask = mask[:, 1:] * mask[:, :-1]
        if first_weight > 0.0 and pair_mask.sum() > 0:
            pred_diff = (pred[:, 1:] - pred[:, :-1]) / dt
            true_diff = (true[:, 1:] - true[:, :-1]) / dt
            diff_loss = F.smooth_l1_loss(pred_diff, true_diff, beta=beta, reduction='none')
            aux_terms.append(first_weight * diff_loss.mul(pair_mask).sum() / pair_mask.sum().clamp_min(1.0))
        if second_weight > 0.0 and pred.size(1) >= 3:
            triplet_mask = pair_mask[:, 1:] * pair_mask[:, :-1]
            if triplet_mask.sum() > 0:
                pred_diff = (pred[:, 1:] - pred[:, :-1]) / dt
                true_diff = (true[:, 1:] - true[:, :-1]) / dt
                pred_acc = pred_diff[:, 1:] - pred_diff[:, :-1]
                true_acc = true_diff[:, 1:] - true_diff[:, :-1]
                acc_loss = F.smooth_l1_loss(pred_acc, true_acc, beta=beta, reduction='none')
                aux_terms.append(second_weight * acc_loss.mul(triplet_mask).sum() / triplet_mask.sum().clamp_min(1.0))
        if not aux_terms:
            return None
        return torch.stack([term.reshape(()) for term in aux_terms]).sum()

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor, x_mask: torch.Tensor, **kwargs) -> dict:
        y_mark = kwargs['y_mark']

        predictions = self.model(x, x_mark, x_mask, y_mark, train_stage=kwargs.get('train_stage'))
        normal_state = {
            "latest_aux_loss": self.model.latest_aux_loss,
            "latest_selector_branch_gates": self.model.latest_selector_branch_gates,
            "latest_selector_safety_gate": self.model.latest_selector_safety_gate,
            "latest_selector_gates": self.model.latest_selector_gates,
            "latest_selector_stats": self.model.latest_selector_stats,
            "latest_selector_residual_norms": self.model.latest_selector_residual_norms,
            "latest_selector_uncertainty": self.model.latest_selector_uncertainty,
            "latest_selector_price": self.model.latest_selector_price,
            "latest_selector_safe_scores": self.model.latest_selector_safe_scores,
            "latest_selector_price_components": self.model.latest_selector_price_components,
            "latest_mechanism_residual_norm": self.model.latest_mechanism_residual_norm,
            "latest_mcg_integral_gate": self.model.latest_mcg_integral_gate,
            "latest_mcg_branch_residual_norms": self.model.latest_mcg_branch_residual_norms,
            "latest_mcg_boundary_penalty": self.model.latest_mcg_boundary_penalty,
            "selector_decoder_gate": self.model.selector_decoder_gate,
        }
        base_predictions = None
        emit_noop = bool(getattr(self.configs, "apn_emit_noop_pred", 0))
        if emit_noop:
            previous_force = self.model.force_noop_selector
            self.model.force_noop_selector = True
            try:
                base_predictions = self.model(x, x_mark, x_mask, y_mark, train_stage=kwargs.get('train_stage'))
            finally:
                self.model.force_noop_selector = previous_force
                for key, value in normal_state.items():
                    setattr(self.model, key, value)

        y = kwargs.get('y')
        y_mask = kwargs.get('y_mask')
        f_dim = -1 if self.configs.features == 'MS' else 0

        outputs = {
            "pred": predictions[:, :, f_dim:],
            "true": y[:, :, f_dim:],
            "mask": y_mask[:, :, f_dim:] if y_mask is not None else None
        }
        if base_predictions is not None:
            outputs["base_pred"] = base_predictions[:, :, f_dim:]
        if bool(getattr(self.configs, "apn_emit_selector_arrays", 0)):
            gates = normal_state["latest_selector_gates"]
            safety_gate = normal_state["latest_selector_safety_gate"]
            if gates is not None:
                outputs["selector_gate"] = gates
            if safety_gate is not None:
                outputs["safety_gate"] = safety_gate
        if bool(getattr(self.configs, "apn_emit_mcg_arrays", 0)):
            mcg_gate = normal_state["latest_mcg_integral_gate"]
            if mcg_gate is not None:
                outputs["mcg_integral_gate"] = mcg_gate.squeeze(-1).permute(0, 2, 1)[:, :, f_dim:]
            if self.model.latest_mcg_local_pred is not None:
                outputs["mcg_local_pred"] = self.model.latest_mcg_local_pred[:, :, f_dim:]
            if self.model.latest_mcg_integral_pred is not None:
                outputs["mcg_integral_pred"] = self.model.latest_mcg_integral_pred[:, :, f_dim:]
        if kwargs.get('exp_stage') == 'train':
            aux_loss = self.model.latest_aux_loss
            train_stage = kwargs.get('train_stage')
            warmup_stage = int(getattr(self.configs, "apn_mcg_warmup_stage", 0))
            if train_stage is None or int(train_stage) > warmup_stage:
                gate_aux = self.mcg_supervised_gate_loss(y, y_mask, f_dim)
                if gate_aux is not None:
                    aux_loss = gate_aux if aux_loss is None else aux_loss + gate_aux
            diff_aux = self.trajectory_difference_aux_loss(
                outputs["pred"],
                outputs["true"],
                outputs["mask"],
                y_mark,
            )
            if diff_aux is not None:
                aux_loss = diff_aux if aux_loss is None else aux_loss + diff_aux
            if aux_loss is not None:
                outputs["aux_loss"] = aux_loss
        return outputs
