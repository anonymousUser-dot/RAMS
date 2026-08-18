#!/usr/bin/env python
"""Build the W-MARS public-baseline comparison table.

The script is deliberately strict: a dataset is marked complete only when both
the validation-selected W-MARS result and the requested public baselines are
present under the same dataset/horizon protocol.  This prevents the paper from
calling a branch-frontier diagnostic a SOTA comparison.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_BASELINES = ["APN", "KAFNet", "GraFITi", "tPatchGNN", "mTAN", "CRU"]


def csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def aggregate_baselines(paths: list[Path], baselines: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_csv(path))

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        model = row.get("model", "")
        dataset = row.get("dataset", "")
        if model not in baselines or not dataset:
            continue
        if row.get("MSE", "") == "" or row.get("MAE", "") == "":
            continue
        grouped[(dataset, model)].append(row)

    out: list[dict[str, object]] = []
    for (dataset, model), vals in sorted(grouped.items()):
        mses = [float(r["MSE"]) for r in vals]
        maes = [float(r["MAE"]) for r in vals]
        seeds = sorted({r.get("seed", "") for r in vals if r.get("seed", "") != ""})
        out.append({
            "dataset": dataset,
            "model": model,
            "n": len(vals),
            "seeds": ",".join(seeds),
            "mse_mean": mean(mses),
            "mse_sd": sd(mses),
            "mae_mean": mean(maes),
            "mae_sd": sd(maes),
        })
    return out


def load_wmars_selected(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        dataset = row.get("dataset", "")
        if not dataset:
            continue
        out[dataset] = row
    return out


def build_comparison(
    baseline_agg: list[dict[str, object]],
    wmars: dict[str, dict[str, str]],
    baselines: list[str],
    min_baselines: int,
) -> list[dict[str, object]]:
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in baseline_agg:
        by_dataset[str(row["dataset"])].append(row)

    datasets = sorted(set(by_dataset) | set(wmars))
    out: list[dict[str, object]] = []
    for dataset in datasets:
        baseline_rows = by_dataset.get(dataset, [])
        best_mse = min(baseline_rows, key=lambda r: float(r["mse_mean"])) if baseline_rows else None
        best_mae = min(baseline_rows, key=lambda r: float(r["mae_mean"])) if baseline_rows else None
        w = wmars.get(dataset)
        complete_baselines = len({str(r["model"]) for r in baseline_rows if str(r["model"]) in baselines})
        row: dict[str, object] = {
            "dataset": dataset,
            "baseline_count": complete_baselines,
            "required_baseline_count": min_baselines,
            "best_mse_baseline": best_mse["model"] if best_mse else "",
            "best_mse": best_mse["mse_mean"] if best_mse else "",
            "best_mae_baseline": best_mae["model"] if best_mae else "",
            "best_mae": best_mae["mae_mean"] if best_mae else "",
            "wmars_selected": w.get("selected", "") if w else "",
            "wmars_test_mse": w.get("test_MSE", "") if w else "",
            "wmars_test_mae": w.get("test_MAE", "") if w else "",
            "status": "missing",
        }
        if w and best_mse and best_mae and complete_baselines >= min_baselines:
            wmars_mse = float(w["test_MSE"])
            wmars_mae = float(w["test_MAE"])
            row["mse_gain_vs_best_baseline_pct"] = 100.0 * (float(best_mse["mse_mean"]) - wmars_mse) / max(float(best_mse["mse_mean"]), 1e-12)
            row["mae_gain_vs_best_baseline_pct"] = 100.0 * (float(best_mae["mae_mean"]) - wmars_mae) / max(float(best_mae["mae_mean"]), 1e-12)
            row["beats_best_mse"] = wmars_mse < float(best_mse["mse_mean"])
            row["beats_best_mae"] = wmars_mae < float(best_mae["mae_mean"])
            row["status"] = "complete"
        elif w and (not best_mse or not best_mae):
            row["status"] = "missing_baseline"
        elif not w and best_mse:
            row["status"] = "missing_validation_selected_wmars"
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_summaries", default="storage/results_BigData2026_v249_strong_baselines/summary_local_complete.csv")
    parser.add_argument("--wmars_selected", default="storage/results_BigData2026_v307_wmars_validation_protocol/_summary/v307_wmars_validation_selected_test.csv")
    parser.add_argument("--baselines", default="APN,KAFNet,GraFITi,tPatchGNN,mTAN,CRU")
    parser.add_argument("--min_baselines", type=int, default=4)
    parser.add_argument("--out_dir", default="storage/results_BigData2026_v308_wmars_tables")
    args = parser.parse_args()

    baselines = csv_list(args.baselines)
    baseline_paths = [Path(p) for p in csv_list(args.baseline_summaries)]
    out_dir = Path(args.out_dir)
    baseline_agg = aggregate_baselines(baseline_paths, baselines)
    wmars = load_wmars_selected(Path(args.wmars_selected))
    comparison = build_comparison(baseline_agg, wmars, baselines, args.min_baselines)
    write_csv(out_dir / "v308_public_baseline_agg.csv", baseline_agg)
    write_csv(out_dir / "v308_wmars_vs_public_baselines.csv", comparison)

    complete = [r for r in comparison if r.get("status") == "complete"]
    beats_mse = sum(str(r.get("beats_best_mse")) == "True" for r in complete)
    beats_mae = sum(str(r.get("beats_best_mae")) == "True" for r in complete)
    print(f"baseline_agg={out_dir / 'v308_public_baseline_agg.csv'}")
    print(f"comparison={out_dir / 'v308_wmars_vs_public_baselines.csv'}")
    print(f"complete_datasets={len(complete)} beats_mse={beats_mse} beats_mae={beats_mae}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
