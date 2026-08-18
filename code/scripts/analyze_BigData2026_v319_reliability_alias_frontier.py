#!/usr/bin/env python
"""Reliability-aliasing and residual-frontier diagnostics for wearable IMTS.

The diagnostic is deliberately model-light.  It asks whether deployment-visible
wearable reliability state explains future level residuals beyond value-only
level summaries.  This gives a real-data proxy for the reliability residual
frontier used in the RAMS-tPatch theory.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dependencies.WearableActivity import WearableActivity, Wearable_time_chunk


DATASET_SPECS = {
    "MHEALTH": dict(root="storage/datasets/MHEALTH", seq_len=250, pred_len=50, n_vars=23),
    "OPPORTUNITY": dict(root="storage/datasets/OPPORTUNITY", seq_len=300, pred_len=60, n_vars=64),
    "PAMAP2": dict(root="storage/datasets/PAMAP2", seq_len=500, pred_len=100, n_vars=37),
}


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_mean(a, mask, axis=0, fill=0.0):
    mask_sum = mask.sum(axis=axis)
    num = (a * mask).sum(axis=axis)
    out = np.divide(num, np.maximum(mask_sum, 1e-8))
    return np.where(mask_sum > 0, out, fill)


def load_wearable_chunks(dataset: str, spec: dict, split: str):
    cfg = SimpleNamespace(seq_len=spec["seq_len"], pred_len=spec["pred_len"])
    raw = WearableActivity(root=spec["root"], dataset=dataset)
    n = len(raw)
    train_end = int(0.70 * n)
    val_end = int(0.80 * n)
    if split == "train":
        records = [raw[i] for i in range(train_end)]
    elif split == "val":
        records = [raw[i] for i in range(train_end, val_end)]
    elif split == "test":
        records = [raw[i] for i in range(val_end, n)]
    else:
        records = [raw[i] for i in range(n)]
    return Wearable_time_chunk(records, cfg)


def sample_features(sample: dict, n_vars: int, integral_centers: int = 6):
    x = to_numpy(sample["x"]).astype(np.float64)
    y = to_numpy(sample["y"]).astype(np.float64)
    xm = to_numpy(sample.get("x_mask", np.isfinite(x))).astype(np.float64)
    ym = to_numpy(sample.get("y_mask", np.isfinite(y))).astype(np.float64)
    t = to_numpy(sample["x_mark"]).astype(np.float64).reshape(-1)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"expected 2D sample, got {x.shape=} {y.shape=}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    xm = np.clip(xm, 0.0, 1.0)
    ym = np.clip(ym, 0.0, 1.0)

    obs_count = xm.sum(axis=0)
    any_obs = obs_count > 0
    density = xm.mean(axis=0)
    recent_start = max(0, int(0.75 * len(t)))
    recent_density = xm[recent_start:].mean(axis=0) if len(t) else np.zeros(n_vars)
    mean_v = safe_mean(x, xm, axis=0)
    std_v = np.sqrt(safe_mean((x - mean_v.reshape(1, -1)) ** 2, xm, axis=0))
    recent_w = xm * np.exp(6.0 * t.reshape(-1, 1))
    early_w = xm * np.exp(-6.0 * t.reshape(-1, 1))
    recent_v = safe_mean(x, recent_w, axis=0)
    early_v = safe_mean(x, early_w, axis=0)
    trend_v = recent_v - early_v

    observed_t = np.where(xm > 0, t.reshape(-1, 1), np.nan)
    last_t = np.nanmax(observed_t, axis=0)
    first_t = np.nanmin(observed_t, axis=0)
    last_t = np.where(np.isfinite(last_t), last_t, 0.0)
    first_t = np.where(np.isfinite(first_t), first_t, last_t)
    recency = np.maximum(0.0, np.nanmax(t) - last_t) if len(t) else np.zeros(n_vars)
    span = np.maximum(0.0, last_t - first_t)

    gaps = np.zeros(n_vars)
    for j in range(n_vars):
        tj = t[xm[:, j] > 0]
        if len(tj) >= 2:
            d = np.diff(np.sort(tj))
            gaps[j] = d.std() / (d.mean() + 1e-8)

    centers = np.linspace(0.0, 1.0, integral_centers)
    width = 0.18
    value_integrals = []
    coverage_integrals = []
    for c in centers:
        w = np.exp(-0.5 * ((t.reshape(-1, 1) - c) / width) ** 2)
        obs_w = w * xm
        value_integrals.append(safe_mean(x, obs_w, axis=0))
        coverage_integrals.append(obs_w.sum(axis=0) / np.maximum(w.sum(axis=0), 1e-8))
    value_integrals = np.stack(value_integrals, axis=0)
    coverage_integrals = np.stack(coverage_integrals, axis=0)

    future_mean = safe_mean(y, ym, axis=0, fill=0.0)
    target_mask = (ym.sum(axis=0) > 0).astype(np.float64)
    last_idx = np.nanargmax(np.where(xm > 0, np.arange(len(x)).reshape(-1, 1), -1), axis=0)
    last_v = np.zeros(n_vars)
    for j in range(n_vars):
        if any_obs[j]:
            last_v[j] = x[last_idx[j], j]
    residual = future_mean - last_v

    level = np.concatenate([
        [
            float(np.mean(last_v)),
            float(np.std(last_v)),
            float(np.mean(mean_v)),
            float(np.mean(recent_v)),
            float(np.mean(trend_v)),
            float(np.linalg.norm(trend_v) / math.sqrt(max(n_vars, 1))),
            float(np.mean(std_v)),
        ],
        np.quantile(last_v, [0.1, 0.5, 0.9]),
        np.quantile(trend_v, [0.1, 0.5, 0.9]),
    ])
    reliability = np.concatenate([
        density,
        recent_density,
        np.log1p(obs_count) / math.log1p(max(len(t), 1)),
        recency,
        span,
        gaps,
        coverage_integrals.reshape(-1),
        value_integrals.reshape(-1),
    ])
    reliability_summary = np.concatenate([
        [
            float(np.mean(density)),
            float(np.std(density)),
            float(np.min(density)),
            float(np.max(density)),
            float(np.mean(recent_density)),
            float(np.mean(recency)),
            float(np.mean(span)),
            float(np.mean(gaps)),
            float(np.std(gaps)),
            float(np.mean(coverage_integrals)),
            float(np.std(coverage_integrals)),
            float(np.mean(value_integrals)),
            float(np.std(value_integrals)),
        ],
        np.quantile(density, [0.1, 0.5, 0.9]),
        np.quantile(recent_density, [0.1, 0.5, 0.9]),
        np.quantile(coverage_integrals.reshape(-1), [0.1, 0.5, 0.9]),
    ])
    raw_alias = float(np.std(density) * (np.linalg.norm(trend_v) / math.sqrt(max(n_vars, 1)) + 1e-8))
    return level, reliability_summary, residual, target_mask, raw_alias


def standardize_fit(x):
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return mu, sd


def ridge_fit(x, y, lam):
    x1 = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    reg = lam * np.eye(x1.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(x1.T @ x1 + reg, x1.T @ y)


def ridge_predict(x, w):
    x1 = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    return x1 @ w


def masked_mse(pred, target, mask):
    err = (pred - target) ** 2 * mask
    return float(err.sum() / max(mask.sum(), 1.0))


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x = x[ok]
    y = y[ok]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def crossfit_frontier(level, reliability, residual, mask, folds=5, ridge=1.0, seed=42):
    rng = np.random.default_rng(seed)
    n = level.shape[0]
    order = rng.permutation(n)
    fold_ids = np.arange(n) % folds
    fold_ids = fold_ids[np.argsort(order)]
    pred_l = np.zeros_like(residual)
    pred_lr = np.zeros_like(residual)
    for fold in range(folds):
        test = fold_ids == fold
        train = ~test
        x_l_train = level[train]
        x_lr_train = np.concatenate([level[train], reliability[train]], axis=1)
        mu_l, sd_l = standardize_fit(x_l_train)
        mu_lr, sd_lr = standardize_fit(x_lr_train)
        w_l = ridge_fit((x_l_train - mu_l) / sd_l, residual[train], ridge)
        w_lr = ridge_fit((x_lr_train - mu_lr) / sd_lr, residual[train], ridge)
        pred_l[test] = ridge_predict((level[test] - mu_l) / sd_l, w_l)
        x_lr_test = np.concatenate([level[test], reliability[test]], axis=1)
        pred_lr[test] = ridge_predict((x_lr_test - mu_lr) / sd_lr, w_lr)
    risk_l = masked_mse(pred_l, residual, mask)
    risk_lr = masked_mse(pred_lr, residual, mask)
    sample_gain = (((pred_l - residual) ** 2 - (pred_lr - residual) ** 2) * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
    alias_score = np.sqrt(((pred_lr - pred_l) ** 2).mean(axis=1))
    return risk_l, risk_lr, sample_gain, alias_score


def first_pc_score(x):
    x = np.asarray(x, dtype=float)
    mu, sd = standardize_fit(x)
    z = (x - mu) / sd
    if z.shape[0] < 2:
        return np.zeros(z.shape[0])
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    return z @ vt[0]


def quantile_bins(score, n_bins):
    score = np.asarray(score, dtype=float)
    edges = np.quantile(score, np.linspace(0.0, 1.0, n_bins + 1))
    # Make edges strictly usable even when ties occur.
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.digitize(score, edges[1:-1], right=False)


def vector_alias_anova(level, reliability, residual, mask, level_bins=5, reliability_bins=3):
    """Reliability residual variance inside level-matched strata.

    This estimates whether windows that look similar to a value-level summary
    still have different future residual means once grouped by reliability
    state.  It is a proxy for observation-measure aliasing, not a selector.
    """
    lbin = quantile_bins(first_pc_score(level), level_bins)
    rbin = quantile_bins(first_pc_score(reliability), reliability_bins)
    total = 0.0
    alias = 0.0
    used = 0
    for lb in range(level_bins):
        idx_l = lbin == lb
        if idx_l.sum() < reliability_bins:
            continue
        m_l = mask[idx_l]
        z_l = residual[idx_l]
        mu_l = (z_l * m_l).sum(axis=0) / np.maximum(m_l.sum(axis=0), 1.0)
        within_total = (((z_l - mu_l.reshape(1, -1)) ** 2) * m_l).sum()
        total += float(within_total)
        used += int(idx_l.sum())
        for rb in range(reliability_bins):
            idx = idx_l & (rbin == rb)
            if idx.sum() < 2:
                continue
            m = mask[idx]
            z = residual[idx]
            mu_lr = (z * m).sum(axis=0) / np.maximum(m.sum(axis=0), 1.0)
            diff = (mu_lr - mu_l) ** 2
            alias += float((diff * (m.sum(axis=0) > 0)).sum() * idx.sum())
    share = alias / max(total, 1e-12)
    return {
        "alias_anova_energy": alias / max(used, 1),
        "level_matched_residual_var": total / max(used, 1),
        "alias_anova_share_pct": 100.0 * share,
    }


def analyze_dataset(dataset, spec, splits, max_samples, folds, ridge):
    levels = []
    rels = []
    residuals = []
    masks = []
    raw_alias = []
    for split in splits:
        chunks = load_wearable_chunks(dataset, spec, split)
        if max_samples > 0:
            chunks = chunks[:max_samples]
        for sample in chunks:
            l, r, z, m, a = sample_features(sample, spec["n_vars"])
            levels.append(l)
            rels.append(r)
            residuals.append(z)
            masks.append(m)
            raw_alias.append(a)
    level = np.stack(levels)
    reliability = np.stack(rels)
    residual = np.stack(residuals)
    mask = np.stack(masks)
    risk_l, risk_lr, sample_gain, learned_alias = crossfit_frontier(
        level, reliability, residual, mask, folds=min(folds, max(2, len(level) // 3)), ridge=ridge
    )
    anova = vector_alias_anova(level, reliability, residual, mask)
    gain = risk_l - risk_lr
    gain_pct = 100.0 * gain / max(risk_l, 1e-12)
    bins = []
    q = np.quantile(learned_alias, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    labels = ["low", "mid", "high"]
    for i, label in enumerate(labels):
        if i == 2:
            idx = (learned_alias >= q[i]) & (learned_alias <= q[i + 1])
        else:
            idx = (learned_alias >= q[i]) & (learned_alias < q[i + 1])
        bins.append({
            "dataset": dataset,
            "alias_bin": label,
            "n": int(idx.sum()),
            "alias_mean": float(learned_alias[idx].mean()) if idx.any() else float("nan"),
            "frontier_gain_mean": float(sample_gain[idx].mean()) if idx.any() else float("nan"),
            "frontier_gain_positive_rate": float((sample_gain[idx] > 0).mean()) if idx.any() else float("nan"),
        })
    summary = {
        "dataset": dataset,
        "n_samples": int(len(level)),
        "n_vars": int(spec["n_vars"]),
        "risk_level": risk_l,
        "risk_level_reliability": risk_lr,
        "frontier_proxy": gain,
        "frontier_gain_pct": gain_pct,
        "learned_alias_gain_corr": corr(learned_alias, sample_gain),
        "raw_alias_gain_corr": corr(raw_alias, sample_gain),
        "alias_low_gain": bins[0]["frontier_gain_mean"],
        "alias_mid_gain": bins[1]["frontier_gain_mean"],
        "alias_high_gain": bins[2]["frontier_gain_mean"],
        **anova,
    }
    return summary, bins


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="MHEALTH,OPPORTUNITY,PAMAP2")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--out", default="storage/results_BigData2026_v319_reliability_alias_frontier")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=10.0)
    args = parser.parse_args()

    out = Path(args.out)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    summaries = []
    bins = []
    for dataset in datasets:
        summary, bin_rows = analyze_dataset(
            dataset, DATASET_SPECS[dataset], splits, args.max_samples, args.folds, args.ridge
        )
        summaries.append(summary)
        bins.extend(bin_rows)
        print(summary)
    write_csv(out / "v319_reliability_frontier_summary.csv", summaries)
    write_csv(out / "v319_alias_bin_diagnostic.csv", bins)
    print(out / "v319_reliability_frontier_summary.csv")
    print(out / "v319_alias_bin_diagnostic.csv")


if __name__ == "__main__":
    main()
