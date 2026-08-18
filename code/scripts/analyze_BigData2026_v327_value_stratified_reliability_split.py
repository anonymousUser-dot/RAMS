#!/usr/bin/env python
"""v327 theorem-aligned reliability split diagnostic.

This diagnostic estimates the concrete term in the value-only aliasing lower
bound used by the RAMS-tPatch paper.  It groups windows by value-level state
and asks whether reliability bins inside the same value stratum have different
future residual means.  The output is an empirical lower-bound proxy:

    sum_l P(L=l) sum_r p(r|l) ||mu_{l,r} - mu_l||^2.

It is deliberately model-light and uses the same wearable conversion features
as the v319 reliability-frontier diagnostic.
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
from scripts.analyze_BigData2026_v319_reliability_alias_frontier import (
    DATASET_SPECS,
    first_pc_score,
    quantile_bins,
    sample_features,
)


def csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_chunks(dataset: str, spec: dict, split: str):
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


def collect_arrays(dataset: str, splits: list[str], max_samples: int):
    spec = DATASET_SPECS[dataset]
    level_rows = []
    reliability_rows = []
    residual_rows = []
    mask_rows = []
    for split in splits:
        chunks = load_chunks(dataset, spec, split)
        limit = len(chunks) if max_samples <= 0 else min(len(chunks), max_samples)
        for idx in range(limit):
            level, reliability, residual, mask, _ = sample_features(chunks[idx], spec["n_vars"])
            level_rows.append(level)
            reliability_rows.append(reliability)
            residual_rows.append(residual)
            mask_rows.append(mask)
    return (
        np.asarray(level_rows, dtype=float),
        np.asarray(reliability_rows, dtype=float),
        np.asarray(residual_rows, dtype=float),
        np.asarray(mask_rows, dtype=float),
    )


def weighted_mean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (x * mask).sum(axis=0) / np.maximum(mask.sum(axis=0), 1.0)


def masked_sqdist(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return float((((a - b) ** 2) * mask).sum() / max(float(mask.sum()), 1.0))


def diagnostic(level, reliability, residual, mask, level_bins: int, reliability_bins: int, min_cell: int):
    lbin = quantile_bins(first_pc_score(level), level_bins)
    rbin = quantile_bins(first_pc_score(reliability), reliability_bins)
    n = len(level)
    rows = []
    proxy = 0.0
    used = 0
    max_gap = 0.0
    weighted_gap = 0.0
    valid_strata = 0
    for lb in range(level_bins):
        idx_l = lbin == lb
        n_l = int(idx_l.sum())
        if n_l < max(reliability_bins, min_cell):
            continue
        z_l = residual[idx_l]
        m_l = mask[idx_l]
        mu_l = weighted_mean(z_l, m_l)
        stratum_proxy = 0.0
        cell_rows = []
        for rb in range(reliability_bins):
            idx = idx_l & (rbin == rb)
            n_lr = int(idx.sum())
            if n_lr < min_cell:
                continue
            z_lr = residual[idx]
            m_lr = mask[idx]
            mu_lr = weighted_mean(z_lr, m_lr)
            gap = masked_sqdist(mu_lr.reshape(1, -1), mu_l.reshape(1, -1), (m_lr.sum(axis=0) > 0).reshape(1, -1))
            weight = n_lr / max(n, 1)
            stratum_proxy += weight * gap
            max_gap = max(max_gap, gap)
            weighted_gap += weight * math.sqrt(max(gap, 0.0))
            cell_rows.append({
                "level_bin": lb,
                "reliability_bin": rb,
                "n_cell": n_lr,
                "cell_prob": n_lr / max(n, 1),
                "mean_gap_sq": gap,
                "mean_gap": math.sqrt(max(gap, 0.0)),
            })
        if cell_rows:
            valid_strata += 1
            used += n_l
            proxy += stratum_proxy
            for row in cell_rows:
                row["stratum_n"] = n_l
                row["stratum_proxy"] = stratum_proxy
                rows.append(row)
    return {
        "n": n,
        "used": used,
        "valid_strata": valid_strata,
        "lower_bound_proxy": proxy,
        "lower_bound_proxy_pct_of_residual_var": 100.0 * proxy / max(float(np.var(residual[mask > 0])), 1e-12),
        "max_cell_gap": max_gap,
        "weighted_mean_gap": weighted_gap,
    }, rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(path, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="MHEALTH,OPPORTUNITY,PAMAP2")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--out", default="storage/results_BigData2026_v327_value_stratified_reliability_split")
    parser.add_argument("--level_bins", type=int, default=5)
    parser.add_argument("--reliability_bins", type=int, default=3)
    parser.add_argument("--min_cell", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    out = PROJECT_ROOT / args.out
    summary_rows = []
    cell_rows_all = []
    for dataset in csv_list(args.datasets):
        level, reliability, residual, mask = collect_arrays(dataset, csv_list(args.splits), args.max_samples)
        summary, cell_rows = diagnostic(level, reliability, residual, mask, args.level_bins, args.reliability_bins, args.min_cell)
        summary["dataset"] = dataset
        summary["level_bins"] = args.level_bins
        summary["reliability_bins"] = args.reliability_bins
        summary["min_cell"] = args.min_cell
        summary_rows.append(summary)
        for row in cell_rows:
            row["dataset"] = dataset
            cell_rows_all.append(row)
        print(dataset, summary, flush=True)
    write_csv(out / "v327_value_stratified_reliability_split_summary.csv", summary_rows)
    write_csv(out / "v327_value_stratified_reliability_split_cells.csv", cell_rows_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
