import math

import torch
import torch.nn as nn
from torch import Tensor
from einops import repeat

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class TTKMN(nn.Module):
    """K-Gaussian kernel pooling over time for each variable."""

    def __init__(self, k: int = 4):
        super().__init__()
        self.k = k
        self.c = nn.Parameter(torch.linspace(0, 1, k))
        self.log_alpha = nn.Parameter(torch.zeros(k))
        self.gate = nn.Parameter(torch.zeros(k))

    def forward(self, t: Tensor, x: Tensor, m: Tensor) -> Tensor:
        alpha = self.log_alpha.exp() + 1e-6
        td = t - self.c.view(1, 1, self.k)
        w = torch.exp(-0.5 * td**2 / alpha.view(1, 1, self.k) ** 2) * m
        a = w / (w.sum(1, keepdim=True) + 1e-8)
        h = torch.einsum("blk,bld->bk", a, x)
        h = h * torch.sigmoid(self.gate)
        flag = (m.sum(1) > 0).float()
        return torch.cat([h, flag], -1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


def rff(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    proj = torch.einsum("bhmd,hdR->bhmR", x, w) + b.unsqueeze(1).unsqueeze(2)
    return torch.cat([torch.cos(proj), torch.sin(proj)], -1) / math.sqrt(proj.size(-1))


class FreqLinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 1, r: int = 64):
        super().__init__()
        assert dim % heads == 0
        assert dim % 2 == 0
        self.h, self.d, self.r = heads, dim // heads, r
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        scale = 1.0 / math.sqrt(self.d)
        self.W = nn.Parameter(torch.randn(self.h, self.d, r // 2) * scale)
        self.b = nn.Parameter(2 * math.pi * torch.rand(self.h, r // 2))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_vars, dim = x.shape
        fx = torch.fft.rfft(x, norm="forward")
        fx = torch.view_as_real(fx)
        fx = torch.cat((fx[..., 0], -fx[..., 1]), -1)[..., :dim]

        def split(t: Tensor) -> Tensor:
            return t.view(batch_size, n_vars, self.h, self.d).transpose(1, 2)

        q, k, v = map(split, (self.q(fx), self.k(fx), self.v(fx)))
        phi_q, phi_k = rff(q, self.W, self.b), rff(k, self.W, self.b)
        k_sum = phi_k.sum(2)
        kv_sum = torch.einsum("bhmr,bhmd->bhrd", phi_k, v)
        denom = torch.einsum("bhmr,bhr->bhm", phi_q, k_sum).unsqueeze(-1) + 1e-6
        out = torch.einsum("bhmr,bhrd->bhmd", phi_q, kv_sum) / denom
        out = self.proj(out.transpose(1, 2).reshape(batch_size, n_vars, dim))

        real, imag = torch.chunk(out, 2, -1)
        return torch.fft.irfft(torch.complex(real, -imag), n=dim, norm="forward")


class FreqBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 1, mlp_ratio: float = 4.0, r: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FreqLinearAttention(dim, heads, r)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(True), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class KAFNetCore(nn.Module):
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.n_vars = configs.enc_in
        self.k = 4
        self.hid = configs.d_model
        self.te = configs.tpatchgnn_te_dim
        self.preconv_dim = max(4, self.hid * 2)
        self.n_heads = configs.n_heads
        self.n_layers = configs.n_layers

        self.intra = TTKMN(self.k)
        self.te_proj1d = nn.Linear(self.te, 1)

        self.pos = PositionalEncoding(self.hid, max_len=self.n_vars)
        self.blocks = nn.ModuleList(
            [FreqBlock(self.hid, heads=self.n_heads, mlp_ratio=4.0, r=64) for _ in range(self.n_layers)]
        )

        self.feat_proj = nn.Linear(self.k + 1, self.hid)
        self.var_agg = nn.Linear(self.hid, self.hid)
        self.te_scale = nn.Linear(1, 1)
        self.te_per_sin = nn.Linear(1, (self.te - 1) // 2)
        self.te_per_cos = nn.Linear(1, self.te - 1 - ((self.te - 1) // 2))
        self.pre_conv = nn.Sequential(
            nn.Conv1d(1, self.preconv_dim, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(self.preconv_dim, 1, kernel_size=1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hid + self.te, self.hid),
            nn.ReLU(True),
            nn.Linear(self.hid, self.hid),
            nn.ReLU(True),
            nn.Linear(self.hid, 1),
        )

    def time_embedding(self, t: Tensor) -> Tensor:
        return torch.cat([self.te_scale(t), torch.sin(self.te_per_sin(t)), torch.cos(self.te_per_cos(t))], -1)

    def forward(self, tp_pred: Tensor, x: Tensor, tp_true: Tensor, mask: Tensor = None) -> Tensor:
        batch_size, obs_len, n_vars = x.shape
        mask = torch.ones_like(x) if mask is None else mask

        tp_true = tp_true[..., None].repeat(1, 1, n_vars) if tp_true.dim() == 2 else tp_true
        tp_pred = tp_pred[..., None] if tp_pred.dim() == 2 else tp_pred

        xf = x.transpose(1, 2).reshape(-1, 1, obs_len)
        xf = self.pre_conv(xf).transpose(1, 2)
        tf = tp_true.permute(0, 2, 1).reshape(-1, obs_len, 1)
        tf_min = tf.min(dim=1, keepdim=True)[0]
        tf_max = tf.max(dim=1, keepdim=True)[0]
        tf_normalized = (tf - tf_min) / (tf_max - tf_min + 1e-8)
        mf = mask.permute(0, 2, 1).reshape(-1, obs_len, 1)

        te = self.time_embedding(tf)
        z = self.intra(tf_normalized, xf + self.te_proj1d(te), mf)
        z = self.feat_proj(z).view(batch_size, n_vars, self.hid)

        z = self.pos(z)
        for block in self.blocks:
            z = block(z)
        z = self.var_agg(z).transpose(1, 2)

        pred_len = tp_pred.shape[1]
        h = z.unsqueeze(2).repeat(1, 1, pred_len, 1).permute(0, 3, 2, 1)
        te_p = self.time_embedding(tp_pred).unsqueeze(1).repeat(1, n_vars, 1, 1)
        y = self.decoder(torch.cat([h, te_p], -1)).squeeze(-1)
        return y.permute(0, 2, 1)


class Model(nn.Module):
    """
    KAFNet baseline adapted to this repository's forecasting protocol.

    Core architecture follows the official KAFNet implementation:
    https://github.com/zhouziyu02/KAFNet
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len_max_irr or configs.pred_len
        self.model = KAFNetCore(configs)

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor = None,
        x_mask: Tensor = None,
        y: Tensor = None,
        y_mark: Tensor = None,
        y_mask: Tensor = None,
        **kwargs,
    ) -> dict:
        batch_size, _, enc_in = x.shape
        if x_mark is None:
            x_mark = repeat(torch.arange(x.shape[1], dtype=x.dtype, device=x.device) / x.shape[1], "l -> b l 1", b=batch_size)
        if x_mask is None:
            x_mask = torch.ones_like(x, device=x.device, dtype=x.dtype)
        if y is None:
            if self.configs.task_name in ["short_term_forecast", "long_term_forecast"]:
                logger.warning("y is missing for KAFNet input. This is only expected for FLOP tests.")
            y = torch.ones((batch_size, self.pred_len, enc_in), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(torch.arange(y.shape[1], dtype=y.dtype, device=y.device) / y.shape[1], "l -> b l 1", b=batch_size)
        if y_mask is None:
            y_mask = torch.ones_like(y, device=y.device, dtype=y.dtype)

        pred = self.model(y_mark[:, :, 0], x, x_mark[:, :, 0], x_mask)
        f_dim = -1 if self.configs.features == "MS" else 0
        return {
            "pred": pred[:, -y.shape[1] :, f_dim:],
            "true": y[:, :, f_dim:],
            "mask": y_mask[:, :, f_dim:],
        }
