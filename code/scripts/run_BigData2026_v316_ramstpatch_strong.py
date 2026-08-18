#!/usr/bin/env python
"""v316 RAMS-tPatch strong-backbone residual scout.

This runner tests the next accuracy-first hypothesis:

    tPatchGNN already captures the temporal patch graph well, but wearable
    datasets may still contain a deployment-visible reliability residual.

The experiment freezes a trained tPatchGNN checkpoint and trains a compact
mechanism/reliability residual head through MCORWrapper.  It is deliberately
small: OPPORTUNITY and PAMAP2 first, then MHEALTH/HumanActivity only if the
strong-backbone residual signal is positive.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    root: str
    seq_len: int
    pred_len: int
    enc_in: int
    d_model: int
    batch_size: int
    dropout: float
    apn_npatch: int
    apn_te_dim: int
    patch_len: int


DATASETS: dict[str, DatasetSpec] = {
    "HumanActivity": DatasetSpec("storage/datasets/HumanActivity", 3000, 300, 12, 16, 32, 0.0, 300, 8, 300),
    "MHEALTH": DatasetSpec("storage/datasets/MHEALTH", 250, 50, 23, 16, 4, 0.05, 50, 8, 25),
    "OPPORTUNITY": DatasetSpec("storage/datasets/OPPORTUNITY", 300, 60, 64, 24, 4, 0.05, 60, 8, 30),
    "PAMAP2": DatasetSpec("storage/datasets/PAMAP2", 500, 100, 37, 24, 4, 0.05, 50, 8, 50),
}


VARIANTS: dict[str, dict[str, str]] = {
    "local": {
        "force": "local",
        "supervised": "0.0",
        "stages": "1",
        "residual_l2": "1e-4",
        "gate_init": "-3.0",
        "switch_init": "-3.0",
        "boundary": "0.0",
    },
    "integral": {
        "force": "integral",
        "supervised": "0.0",
        "stages": "1",
        "residual_l2": "1e-4",
        "gate_init": "-3.0",
        "switch_init": "-3.0",
        "boundary": "0.0",
    },
    "mix": {
        "force": "mix",
        "supervised": "0.0",
        "stages": "1",
        "residual_l2": "1e-4",
        "gate_init": "-3.0",
        "switch_init": "-3.0",
        "boundary": "0.0",
    },
    "gate": {
        "force": "none",
        "supervised": "0.02",
        "stages": "2",
        "residual_l2": "1e-4",
        "gate_init": "-3.0",
        "switch_init": "-2.5",
        "boundary": "2e-4",
    },
    "safe_gate": {
        "force": "none",
        "supervised": "0.01",
        "stages": "2",
        "residual_l2": "3e-4",
        "gate_init": "-3.5",
        "switch_init": "-3.0",
        "boundary": "5e-4",
    },
}


def csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def run(cmd: list[str], log_file: Path, dry_run: bool = False) -> int:
    print("$", " ".join(shlex.quote(part) for part in cmd), flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TQDM_DISABLE", "1")
    with log_file.open("w", encoding="utf-8", errors="replace") as handle:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
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


def latest_checkpoint(root: Path, dataset: str, model_name: str, model_id: str) -> Path | None:
    paths = sorted((root / dataset / model_name / model_id).glob("*/*/iter*/pytorch_model.bin"))
    return paths[-1] if paths else None


def tpatch_base_id(dataset: str, seed: int, epochs: int) -> str:
    return f"local_baselines_{dataset}_tpatchgnn_e{epochs}_s{seed}"


def wrapper_id(prefix: str, dataset: str, variant: str, seed: int, epochs: int) -> str:
    return f"{prefix}_{dataset}_tPatchGNN_ramstpatch_{variant}_e{epochs}_s{seed}"


def find_base(args: argparse.Namespace, dataset: str, seed: int) -> tuple[Path | None, dict[str, float] | None, Path | None]:
    mid = tpatch_base_id(dataset, seed, args.base_epochs)
    for root_value in csv_list(args.base_roots):
        root = (REPO / root_value).resolve()
        ckpt = latest_checkpoint(root, dataset, "tPatchGNN", mid)
        metric = latest_metric(root, dataset, "tPatchGNN", mid)
        if ckpt is not None and metric is not None:
            return ckpt, metric, root
    return None, None, None


def ensure_base(args: argparse.Namespace, dataset: str, seed: int, root: Path) -> int:
    ckpt, metric, _ = find_base(args, dataset, seed)
    if ckpt is not None and metric is not None:
        print(f"[base-ready] {dataset} tPatchGNN seed={seed} {metric}", flush=True)
        return 0
    return run([
        sys.executable,
        "scripts/run_BigData2026_local_complete.py",
        "--mode", "baselines",
        "--root", str(root),
        "--datasets", dataset,
        "--baselines", "tPatchGNN",
        "--seeds", str(seed),
        "--epochs", str(args.base_epochs),
        "--patience", str(args.patience),
        "--num_workers", str(args.num_workers),
    ], root / "logs" / f"ensure_tpatchgnn_{dataset}_s{seed}.log", bool(args.dry_run))


def command_for(args: argparse.Namespace, dataset: str, seed: int, variant: str, ckpt: Path, root: Path) -> list[str]:
    spec = DATASETS[dataset]
    cfg = VARIANTS[variant]
    mid = wrapper_id(args.prefix, dataset, variant, seed, args.residual_epochs)
    cmd = [
        sys.executable, "main.py",
        "--model_id", mid,
        "--model_name", "MCORWrapper",
        "--mcor_base_model_name", "tPatchGNN",
        "--mcor_freeze_base", "1",
        "--pretrained_checkpoint_root_path", str(ckpt.parent),
        "--pretrained_checkpoint_file_name", ckpt.name,
        "--mcor_force_branch", cfg["force"],
        "--mcor_reliability_mode", cfg.get("reliability_mode", "real"),
        "--n_train_stages", cfg["stages"],
        "--apn_mcg_warmup_stage", "1" if cfg["stages"] == "2" else "0",
        "--apn_mcg_warmup_blend", "0.5",
        "--apn_mcg_supervised_weight", cfg["supervised"],
        "--apn_mcg_supervised_temperature", "0.05",
        "--apn_mcg_supervised_min_confidence", "0.15",
        "--apn_mcr_residual_l2", cfg["residual_l2"],
        "--apn_mcr_gate_init", cfg["gate_init"],
        "--apn_mcg_gate_init", cfg["switch_init"],
        "--apn_mcg_boundary_weight", cfg["boundary"],
        "--apn_mcr_integral_centers", str(args.integral_centers),
        "--apn_mcr_integral_width", str(args.integral_width),
        "--use_gpu", "1",
        "--gpu_id", str(args.gpu_id),
        "--use_multi_gpu", "0",
        "--is_training", "1",
        "--dataset_root_path", spec.root,
        "--dataset_name", dataset,
        "--features", "M",
        "--seq_len", str(spec.seq_len),
        "--pred_len", str(spec.pred_len),
        "--enc_in", str(spec.enc_in),
        "--dec_in", str(spec.enc_in),
        "--c_out", str(spec.enc_in),
        "--loss", "MSE",
        "--train_epochs", str(args.residual_epochs),
        "--patience", str(args.patience),
        "--val_interval", "1",
        "--itr", "1",
        "--seed_base", str(seed),
        "--batch_size", str(args.batch_size or spec.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--d_model", str(spec.d_model),
        "--dropout", str(spec.dropout),
        "--apn_npatch", str(spec.apn_npatch),
        "--apn_te_dim", str(spec.apn_te_dim),
        "--collate_fn", "collate_fn_patch",
        "--patch_len", str(spec.patch_len),
        "--n_layers", "1",
        "--n_heads", "1",
        "--tpatchgnn_te_dim", "10",
        "--num_workers", str(args.num_workers),
        "--pin_memory", "1" if args.num_workers > 0 else "0",
        "--persistent_workers", "1" if args.num_workers > 0 else "0",
        "--prefetch_factor", "2" if args.num_workers > 0 else "1",
        "--non_blocking_transfer", "1",
        "--tf32", "1",
        "--cudnn_benchmark", "1",
        "--amp_dtype", "off",
        "--disable_tqdm", "1",
        "--save_arrays", "1" if args.save_arrays else "0",
        "--apn_emit_mcg_arrays", "1" if args.save_arrays else "0",
        "--checkpoints", str(root),
    ]
    return cmd


def train_one(args: argparse.Namespace, root: Path, dataset: str, seed: int, variant: str) -> int:
    ckpt, base_metric, _ = find_base(args, dataset, seed)
    if ckpt is None or base_metric is None:
        print(f"[missing-base] {dataset} seed={seed}", flush=True)
        return 1
    mid = wrapper_id(args.prefix, dataset, variant, seed, args.residual_epochs)
    if args.skip_done and latest_metric(root, dataset, "MCORWrapper", mid) is not None:
        print(f"[skip] {mid}", flush=True)
        return 0
    return run(command_for(args, dataset, seed, variant, ckpt, root), root / "_logs" / f"{mid}.log", bool(args.dry_run))


def summarize(args: argparse.Namespace, root: Path, datasets: list[str], seeds: list[int], variants: list[str]) -> Path:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for seed in seeds:
            ckpt, base_metric, base_root = find_base(args, dataset, seed)
            base_mid = tpatch_base_id(dataset, seed, args.base_epochs)
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "model": "tPatchGNN",
                "tag": "base",
                "model_id": base_mid,
                "status": "done" if base_metric else "missing",
                "MSE": base_metric.get("MSE") if base_metric else "",
                "MAE": base_metric.get("MAE") if base_metric else "",
                "MSE_gain_vs_base_pct": "",
                "MAE_gain_vs_base_pct": "",
                "base_root": str(base_root) if base_root else "",
            })
            for variant in variants:
                mid = wrapper_id(args.prefix, dataset, variant, seed, args.residual_epochs)
                metric = latest_metric(root, dataset, "MCORWrapper", mid)
                row: dict[str, object] = {
                    "dataset": dataset,
                    "seed": seed,
                    "model": "RAMS-tPatch",
                    "tag": variant,
                    "model_id": mid,
                    "status": "done" if metric else "missing",
                }
                if metric:
                    row.update(metric)
                    if base_metric:
                        row["MSE_gain_vs_base_pct"] = 100.0 * (base_metric["MSE"] - metric["MSE"]) / max(base_metric["MSE"], 1e-12)
                        row["MAE_gain_vs_base_pct"] = 100.0 * (base_metric["MAE"] - metric["MAE"]) / max(base_metric["MAE"], 1e-12)
                rows.append(row)

    out_dir = root / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = out_dir / "v316_ramstpatch_detail.csv"
    fields = sorted({key for row in rows for key in row.keys()})
    with detail.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    agg_rows: list[dict[str, object]] = []
    for dataset in datasets:
        for tag in ["base"] + variants:
            done = [row for row in rows if row["dataset"] == dataset and row["tag"] == tag and row["status"] == "done"]
            if not done:
                continue
            item: dict[str, object] = {
                "dataset": dataset,
                "tag": tag,
                "n": len(done),
                "avg_MSE": sum(float(row["MSE"]) for row in done) / len(done),
                "avg_MAE": sum(float(row["MAE"]) for row in done) / len(done),
            }
            gains = [row for row in done if row.get("MSE_gain_vs_base_pct") not in {"", None}]
            if gains:
                item["avg_MSE_gain_vs_base_pct"] = sum(float(row["MSE_gain_vs_base_pct"]) for row in gains) / len(gains)
                item["avg_MAE_gain_vs_base_pct"] = sum(float(row["MAE_gain_vs_base_pct"]) for row in gains) / len(gains)
                item["positive_MSE"] = sum(float(row["MSE_gain_vs_base_pct"]) > 0 for row in gains)
                item["positive_MAE"] = sum(float(row["MAE_gain_vs_base_pct"]) > 0 for row in gains)
            agg_rows.append(item)
    agg = out_dir / "v316_ramstpatch_agg.csv"
    fields_agg = sorted({key for row in agg_rows for key in row.keys()})
    with agg.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields_agg, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"detail={detail}", flush=True)
    print(f"agg={agg}", flush=True)
    return agg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="storage/results_BigData2026_v316_ramstpatch_strong")
    parser.add_argument("--base_roots", default="storage/results_BigData2026_v313_public_baselines_e4")
    parser.add_argument("--prefix", default="v316")
    parser.add_argument("--datasets", default="OPPORTUNITY,PAMAP2")
    parser.add_argument("--seeds", default="5701,5702,5703")
    parser.add_argument("--variants", default="local,integral,mix,gate,safe_gate")
    parser.add_argument("--base_epochs", type=int, default=4)
    parser.add_argument("--residual_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--integral_centers", type=int, default=6)
    parser.add_argument("--integral_width", type=float, default=0.18)
    parser.add_argument("--save_arrays", type=int, default=0)
    parser.add_argument("--skip_done", type=int, default=1)
    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument("--summarize_only", type=int, default=0)
    args = parser.parse_args()

    root = (REPO / args.root).resolve()
    datasets = csv_list(args.datasets)
    seeds = seed_list(args.seeds)
    variants = csv_list(args.variants)
    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset: {dataset}")
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant}")

    if args.summarize_only:
        summarize(args, root, datasets, seeds, variants)
        return 0

    failures = 0
    for dataset in datasets:
        for seed in seeds:
            failures += int(ensure_base(args, dataset, seed, root) != 0)
            for variant in variants:
                failures += int(train_one(args, root, dataset, seed, variant) != 0)
                summarize(args, root, datasets, seeds, variants)
    summarize(args, root, datasets, seeds, variants)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
