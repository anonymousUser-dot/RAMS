#!/usr/bin/env python
"""Compute the KAFNet paired aggregate CI used by Table tab:kafnet-ci."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def read_pairs(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    pending: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["seed"])
            if row["tag"] == "base":
                pending[key] = {
                    "base_mse": float(row["MSE"]),
                    "base_mae": float(row["MAE"]),
                }
            elif row["tag"] == "gate" and key in pending:
                base = pending[key]
                rows.append({
                    "mse_gain": 100.0 * (base["base_mse"] - float(row["MSE"])) / base["base_mse"],
                    "mae_gain": 100.0 * (base["base_mae"] - float(row["MAE"])) / base["base_mae"],
                })
    return rows


def bootstrap_ci(values: list[float], rounds: int, seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(rounds):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * rounds)]
    hi = means[min(rounds - 1, int(0.975 * rounds))]
    return sum(values) / n, lo, hi


def sign_p(wins: int, n: int) -> float:
    # one-sided exact sign test against p=0.5, matching the paper table
    from math import comb

    return sum(comb(n, k) for k in range(wins, n + 1)) / (2**n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detail",
        default="code/storage/results_BigData2026_v358_kafnet_rams_server/_summary/v351_kafnet_rams_detail.csv",
    )
    parser.add_argument("--rounds", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260526)
    args = parser.parse_args()

    rows = read_pairs(Path(args.detail))
    for metric in ["mse_gain", "mae_gain"]:
        values = [row[metric] for row in rows]
        mean, lo, hi = bootstrap_ci(values, args.rounds, args.seed)
        wins = sum(v > 0 for v in values)
        print(
            f"{metric}: n={len(values)} mean={mean:.3f}% "
            f"ci=[{lo:.3f},{hi:.3f}] wins={wins}/{len(values)} "
            f"p_sign={sign_p(wins, len(values)):.6g}"
        )


if __name__ == "__main__":
    main()
