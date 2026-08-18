#!/usr/bin/env python
"""v322 final RAMS-tPatch experiment orchestrator.

This is the definitive experiment entry point for the RAMS-tPatch paper draft.
It separates final evidence from earlier exploration:

1. same-protocol public baselines on the fixed wearable suite;
2. fixed default RAMS-tPatch gate against matched tPatchGNN checkpoints;
3. branch ablations under a separate prefix;
4. reliability-aliasing diagnostics for the theory section;
5. clean CSV summaries for paper tables.

Example:

    python scripts/run_BigData2026_v322_ramstpatch_final.py --mode all --seeds 5301-5310

For a quick command preview:

    python scripts/run_BigData2026_v322_ramstpatch_final.py --mode all --dry_run 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path
from statistics import mean


REPO = Path(__file__).resolve().parents[1]

TARGET_DATASETS = ["MHEALTH", "OPPORTUNITY", "PAMAP2"]
PUBLIC_BASELINES = ["APN", "KAFNet", "GraFITi", "tPatchGNN"]


def csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def seed_list(value: str) -> list[int]:
    seeds: list[int] = []
    for part in csv_list(value):
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run_command(cmd: list[str], log_file: Path, dry_run: bool) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(shlex.quote(str(part)) for part in cmd), flush=True)
    if dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TQDM_DISABLE", "1")
    env.setdefault("PYTHONUTF8", "1")
    with log_file.open("w", encoding="utf-8", errors="replace") as handle:
        proc = subprocess.run(
            [str(part) for part in cmd],
            cwd=REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    print(f"exit={proc.returncode} log={log_file}", flush=True)
    return int(proc.returncode)


def metric_paths(root: Path, dataset: str, model_name: str, model_id: str) -> list[Path]:
    return sorted((root / dataset / model_name / model_id).glob("*/*/iter*/eval_*/metric.json"))


def latest_metric(root: Path, dataset: str, model_name: str, model_id: str) -> dict[str, float] | None:
    paths = metric_paths(root, dataset, model_name, model_id)
    if not paths:
        return None
    metric = json.loads(paths[-1].read_text(encoding="utf-8"))
    return {key: float(value) for key, value in metric.items() if isinstance(value, (int, float))}


def baseline_id(dataset: str, model_name: str, epochs: int, seed: int) -> str:
    return f"local_baselines_{dataset}_{model_name.lower()}_e{epochs}_s{seed}"


def ramstpatch_id(prefix: str, dataset: str, variant: str, residual_epochs: int, seed: int) -> str:
    return f"{prefix}_{dataset}_tPatchGNN_ramstpatch_{variant}_e{residual_epochs}_s{seed}"


def find_tpatch_base(args: argparse.Namespace, dataset: str, seed: int) -> tuple[dict[str, float] | None, str]:
    model_id = baseline_id(dataset, "tPatchGNN", args.base_epochs, seed)
    for root in [args.baseline_root, args.main_root]:
        metric = latest_metric(root, dataset, "tPatchGNN", model_id)
        if metric is not None:
            return metric, str(root)
    return None, ""


def run_baselines(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "scripts/run_BigData2026_local_complete.py",
        "--mode",
        "baselines",
        "--root",
        str(args.baseline_root),
        "--datasets",
        ",".join(args.datasets),
        "--baselines",
        ",".join(args.baselines),
        "--seeds",
        ",".join(str(seed) for seed in args.seeds),
        "--epochs",
        str(args.base_epochs),
        "--patience",
        str(args.base_patience),
        "--num_workers",
        str(args.num_workers),
    ]
    if args.no_skip_done:
        cmd.append("--no_skip_done")
    return run_command(cmd, args.main_root / "_logs" / "v322_public_baselines.log", bool(args.dry_run))


def run_ramstpatch(args: argparse.Namespace, *, variants: list[str], seeds: list[int], prefix: str, log_name: str) -> int:
    base_roots = [str(args.baseline_root), str(args.main_root)]
    cmd = [
        sys.executable,
        "scripts/run_BigData2026_v316_ramstpatch_strong.py",
        "--root",
        str(args.main_root),
        "--base_roots",
        ",".join(base_roots),
        "--prefix",
        prefix,
        "--datasets",
        ",".join(args.datasets),
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--variants",
        ",".join(variants),
        "--base_epochs",
        str(args.base_epochs),
        "--residual_epochs",
        str(args.residual_epochs),
        "--patience",
        str(args.residual_patience),
        "--num_workers",
        str(args.num_workers),
        "--gpu_id",
        str(args.gpu_id),
        "--learning_rate",
        str(args.learning_rate),
        "--integral_centers",
        str(args.integral_centers),
        "--integral_width",
        str(args.integral_width),
        "--save_arrays",
        "1" if args.save_arrays else "0",
        "--skip_done",
        "0" if args.no_skip_done else "1",
        "--dry_run",
        "1" if args.dry_run else "0",
    ]
    if args.batch_size > 0:
        cmd.extend(["--batch_size", str(args.batch_size)])
    return run_command(cmd, args.main_root / "_logs" / log_name, bool(args.dry_run))


def run_main(args: argparse.Namespace) -> int:
    return run_ramstpatch(
        args,
        variants=["gate"],
        seeds=args.seeds,
        prefix=args.prefix,
        log_name="v322_ramstpatch_gate.log",
    )


def run_ablation(args: argparse.Namespace) -> int:
    seeds = args.ablation_seeds or args.seeds[: min(3, len(args.seeds))]
    return run_ramstpatch(
        args,
        variants=["local", "integral", "mix", "gate", "safe_gate"],
        seeds=seeds,
        prefix=f"{args.prefix}abl",
        log_name="v322_ramstpatch_ablation.log",
    )


def run_diagnostic(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "scripts/analyze_BigData2026_v319_reliability_alias_frontier.py",
        "--datasets",
        ",".join(args.datasets),
        "--splits",
        args.diagnostic_splits,
        "--out",
        str(args.diagnostic_root),
        "--folds",
        str(args.diagnostic_folds),
        "--ridge",
        str(args.diagnostic_ridge),
        "--max-samples",
        str(args.diagnostic_max_samples),
    ]
    return run_command(cmd, args.main_root / "_logs" / "v322_reliability_alias_diagnostic.log", bool(args.dry_run))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(path, flush=True)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def bootstrap_ci(values: list[float], *, rounds: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    boot = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        boot.append(mean(sample))
    return percentile(boot, 0.025), percentile(boot, 0.975)


def summarize_baselines(args: argparse.Namespace, out_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    agg: list[dict[str, object]] = []
    for dataset in args.datasets:
        for model in args.baselines:
            metrics = []
            for seed in args.seeds:
                mid = baseline_id(dataset, model, args.base_epochs, seed)
                metric = latest_metric(args.baseline_root, dataset, model, mid)
                status = "done" if metric else "missing"
                row = {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "model_id": mid,
                    "status": status,
                }
                if metric:
                    row.update(metric)
                    metrics.append(metric)
                rows.append(row)
            if metrics:
                agg.append({
                    "dataset": dataset,
                    "model": model,
                    "n": len(metrics),
                    "MSE": mean(m["MSE"] for m in metrics),
                    "MAE": mean(m["MAE"] for m in metrics),
                })
    write_csv(out_dir / "v322_public_baseline_detail.csv", rows)
    write_csv(out_dir / "v322_public_baseline_avg.csv", agg)


def summarize_main(args: argparse.Namespace, out_dir: Path) -> None:
    detail: list[dict[str, object]] = []
    main_rows: list[dict[str, object]] = []
    ci_rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        paired = []
        for seed in args.seeds:
            base_metric, base_root = find_tpatch_base(args, dataset, seed)
            mid = ramstpatch_id(args.prefix, dataset, "gate", args.residual_epochs, seed)
            metric = latest_metric(args.main_root, dataset, "MCORWrapper", mid)
            row = {
                "dataset": dataset,
                "seed": seed,
                "base_model": "tPatchGNN",
                "model": "RAMS-tPatch",
                "variant": "gate",
                "base_root": base_root,
                "model_id": mid,
                "status": "done" if base_metric and metric else "missing",
            }
            if base_metric:
                row["base_MSE"] = base_metric["MSE"]
                row["base_MAE"] = base_metric["MAE"]
            if metric:
                row["MSE"] = metric["MSE"]
                row["MAE"] = metric["MAE"]
            if base_metric and metric:
                row["MSE_gain_pct"] = 100.0 * (base_metric["MSE"] - metric["MSE"]) / max(base_metric["MSE"], 1e-12)
                row["MAE_gain_pct"] = 100.0 * (base_metric["MAE"] - metric["MAE"]) / max(base_metric["MAE"], 1e-12)
                paired.append(row)
            detail.append(row)

        if paired:
            base_mse = mean(float(row["base_MSE"]) for row in paired)
            base_mae = mean(float(row["base_MAE"]) for row in paired)
            model_mse = mean(float(row["MSE"]) for row in paired)
            model_mae = mean(float(row["MAE"]) for row in paired)
            main_rows.append({
                "dataset": dataset,
                "n": len(paired),
                "base_MSE": base_mse,
                "MSE": model_mse,
                "MSE_gain_pct_from_means": 100.0 * (base_mse - model_mse) / max(base_mse, 1e-12),
                "MSE_gain_pct_seed_mean": mean(float(row["MSE_gain_pct"]) for row in paired),
                "MSE_wins": sum(float(row["MSE_gain_pct"]) > 0 for row in paired),
                "base_MAE": base_mae,
                "MAE": model_mae,
                "MAE_gain_pct_from_means": 100.0 * (base_mae - model_mae) / max(base_mae, 1e-12),
                "MAE_gain_pct_seed_mean": mean(float(row["MAE_gain_pct"]) for row in paired),
                "MAE_wins": sum(float(row["MAE_gain_pct"]) > 0 for row in paired),
            })
            for metric_name in ["MSE", "MAE"]:
                gains = [float(row[f"{metric_name}_gain_pct"]) for row in paired]
                lo, hi = bootstrap_ci(gains, rounds=args.bootstrap_rounds, seed=args.bootstrap_seed)
                ci_rows.append({
                    "dataset": dataset,
                    "metric": metric_name,
                    "n": len(gains),
                    "gain_pct_mean": mean(gains),
                    "ci_low": lo,
                    "ci_high": hi,
                    "wins": sum(g > 0 for g in gains),
                })
    write_csv(out_dir / "v322_ramstpatch_paired_detail.csv", detail)
    write_csv(out_dir / "v322_ramstpatch_main_table.csv", main_rows)
    write_csv(out_dir / "v322_ramstpatch_paired_ci.csv", ci_rows)


def summarize_ablation(args: argparse.Namespace, out_dir: Path) -> None:
    seeds = args.ablation_seeds or args.seeds[: min(3, len(args.seeds))]
    rows: list[dict[str, object]] = []
    agg: list[dict[str, object]] = []
    variants = ["local", "integral", "mix", "gate", "safe_gate"]
    prefix = f"{args.prefix}abl"
    for dataset in args.datasets:
        for variant in variants:
            done = []
            for seed in seeds:
                base_metric, base_root = find_tpatch_base(args, dataset, seed)
                mid = ramstpatch_id(prefix, dataset, variant, args.residual_epochs, seed)
                metric = latest_metric(args.main_root, dataset, "MCORWrapper", mid)
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "variant": variant,
                    "base_root": base_root,
                    "model_id": mid,
                    "status": "done" if base_metric and metric else "missing",
                }
                if base_metric and metric:
                    row["base_MSE"] = base_metric["MSE"]
                    row["base_MAE"] = base_metric["MAE"]
                    row["MSE"] = metric["MSE"]
                    row["MAE"] = metric["MAE"]
                    row["MSE_gain_pct"] = 100.0 * (base_metric["MSE"] - metric["MSE"]) / max(base_metric["MSE"], 1e-12)
                    row["MAE_gain_pct"] = 100.0 * (base_metric["MAE"] - metric["MAE"]) / max(base_metric["MAE"], 1e-12)
                    done.append(row)
                rows.append(row)
            if done:
                agg.append({
                    "dataset": dataset,
                    "variant": variant,
                    "n": len(done),
                    "MSE_gain_pct": mean(float(row["MSE_gain_pct"]) for row in done),
                    "MAE_gain_pct": mean(float(row["MAE_gain_pct"]) for row in done),
                    "MSE_wins": sum(float(row["MSE_gain_pct"]) > 0 for row in done),
                    "MAE_wins": sum(float(row["MAE_gain_pct"]) > 0 for row in done),
                })
    write_csv(out_dir / "v322_ramstpatch_ablation_detail.csv", rows)
    write_csv(out_dir / "v322_ramstpatch_ablation_table.csv", agg)


def write_manifest(args: argparse.Namespace, out_dir: Path) -> None:
    manifest = {
        "claim": "RAMS-tPatch targeted SOTA for group-asynchronous wearable IMTS",
        "datasets": args.datasets,
        "baselines": args.baselines,
        "seeds": args.seeds,
        "default_model": "MCORWrapper(tPatchGNN, frozen base, gate)",
        "base_epochs": args.base_epochs,
        "residual_epochs": args.residual_epochs,
        "main_root": str(args.main_root),
        "baseline_root": str(args.baseline_root),
        "diagnostic_root": str(args.diagnostic_root),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "v322_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(path, flush=True)


def summarize_all(args: argparse.Namespace) -> None:
    out_dir = args.main_root / "_summary_final"
    summarize_baselines(args, out_dir)
    summarize_main(args, out_dir)
    summarize_ablation(args, out_dir)
    write_manifest(args, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "baselines", "main", "ablation", "diagnostic", "summary"], default="summary")
    parser.add_argument("--datasets", default=",".join(TARGET_DATASETS))
    parser.add_argument("--baselines", default=",".join(PUBLIC_BASELINES))
    parser.add_argument("--seeds", default="5301-5310")
    parser.add_argument("--ablation_seeds", default="")
    parser.add_argument("--baseline_root", type=Path, default=Path("storage/results_BigData2026_v322_public_baselines_e4"))
    parser.add_argument("--main_root", type=Path, default=Path("storage/results_BigData2026_v322_ramstpatch_final"))
    parser.add_argument("--diagnostic_root", type=Path, default=Path("storage/results_BigData2026_v322_reliability_alias_frontier"))
    parser.add_argument("--prefix", default="v322")
    parser.add_argument("--base_epochs", type=int, default=4)
    parser.add_argument("--base_patience", type=int, default=2)
    parser.add_argument("--residual_epochs", type=int, default=3)
    parser.add_argument("--residual_patience", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--integral_centers", type=int, default=6)
    parser.add_argument("--integral_width", type=float, default=0.18)
    parser.add_argument("--save_arrays", type=int, default=0)
    parser.add_argument("--no_skip_done", action="store_true")
    parser.add_argument("--diagnostic_splits", default="train,val,test")
    parser.add_argument("--diagnostic_folds", type=int, default=5)
    parser.add_argument("--diagnostic_ridge", type=float, default=10.0)
    parser.add_argument("--diagnostic_max_samples", type=int, default=0)
    parser.add_argument("--bootstrap_rounds", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260524)
    parser.add_argument("--dry_run", type=int, default=0)
    args = parser.parse_args()
    args.datasets = csv_list(args.datasets)
    args.baselines = csv_list(args.baselines)
    args.seeds = seed_list(args.seeds)
    args.ablation_seeds = seed_list(args.ablation_seeds) if args.ablation_seeds else []
    args.baseline_root = (REPO / args.baseline_root).resolve()
    args.main_root = (REPO / args.main_root).resolve()
    args.diagnostic_root = (REPO / args.diagnostic_root).resolve()
    unknown = set(args.datasets) - set(TARGET_DATASETS)
    if unknown:
        raise ValueError(f"v322 target suite is fixed; unknown datasets: {sorted(unknown)}")
    return args


def main() -> int:
    args = parse_args()
    args.main_root.mkdir(parents=True, exist_ok=True)
    rc = 0
    if args.mode in {"all", "baselines"}:
        rc |= run_baselines(args)
    if args.mode in {"all", "main"}:
        rc |= run_main(args)
    if args.mode in {"all", "ablation"}:
        rc |= run_ablation(args)
    if args.mode in {"all", "diagnostic"}:
        rc |= run_diagnostic(args)
    if args.mode in {"all", "summary"}:
        summarize_all(args)
    return 1 if rc else 0


if __name__ == "__main__":
    raise SystemExit(main())
