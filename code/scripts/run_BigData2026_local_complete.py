#!/usr/bin/env python
"""Local experiment runner for the BigData 2026 SR-MCLI study.

The server scripts in this repository are optimized for Linux multi-GPU jobs.
This runner keeps the same protocol but is conservative enough to run on a
single local GPU/CPU workstation:

  python scripts/run_BigData2026_local_complete.py --mode smoke
  python scripts/run_BigData2026_local_complete.py --mode main --epochs 5 --seeds 3001,3002
  python scripts/run_BigData2026_local_complete.py --mode ablation --dataset P12 --epochs 5

Results are written under storage/results_BigData2026_local_complete by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    root: str
    seq_len: int
    pred_len: int
    enc_in: int
    d_model: int
    lr: float
    batch_size: int
    dropout: float
    apn_npatch: int
    apn_te_dim: int


DATASETS: dict[str, DatasetSpec] = {
    "HumanActivity": DatasetSpec("storage/datasets/HumanActivity", 3000, 300, 12, 16, 0.01, 16, 0.0, 300, 8),
    "P12": DatasetSpec("storage/datasets/P12", 36, 3, 36, 24, 0.03, 32, 0.10, 20, 8),
    "USHCN": DatasetSpec("storage/datasets/USHCN", 150, 3, 5, 6, 0.01, 32, 0.10, 100, 32),
    "MHEALTH": DatasetSpec("storage/datasets/MHEALTH", 250, 50, 23, 16, 0.001, 32, 0.05, 50, 8),
    "PAMAP2": DatasetSpec("storage/datasets/PAMAP2", 500, 100, 37, 24, 0.001, 32, 0.05, 50, 8),
    "OPPORTUNITY": DatasetSpec("storage/datasets/OPPORTUNITY", 300, 60, 64, 24, 0.001, 32, 0.05, 60, 8),
}


MODEL_EXTRAS: dict[str, list[str]] = {
    "APN": ["--loss", "MSE"],
    # KAFNet uses complex FFT tensors; PyTorch does not support complex bfloat16.
    "KAFNet": ["--loss", "MSE", "--n_layers", "1", "--n_heads", "1", "--tpatchgnn_te_dim", "8", "--amp_dtype", "off"],
    "GraFITi": ["--loss", "MSE", "--d_model", "128", "--n_layers", "2", "--n_heads", "4", "--learning_rate", "0.001", "--batch_size", "32"],
    "mTAN": ["--loss", "ModelProvidedLoss", "--learning_rate", "0.001", "--batch_size", "16"],
    "CRU": ["--loss", "MSE", "--d_model", "20", "--cru_ts", "0.2", "--learning_rate", "0.001", "--batch_size", "16"],
    "SeFT": ["--loss", "MSE", "--n_layers", "2", "--dropout", "0.1", "--learning_rate", "0.001", "--batch_size", "32"],
    "Raindrop": ["--loss", "MSE", "--d_model", "32", "--n_heads", "4", "--learning_rate", "0.001", "--batch_size", "32"],
}


def csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def model_id(prefix: str, dataset: str, tag: str, epochs: int, seed: int) -> str:
    raw = f"{prefix}_{dataset}_{tag}_e{epochs}_s{seed}"
    return raw.replace(".", "p").replace("/", "_").replace("\\", "_")


def common_args(
    *,
    dataset: str,
    spec: DatasetSpec,
    model_name: str,
    mid: str,
    root: Path,
    epochs: int,
    patience: int,
    seed: int,
    num_workers: int,
    save_arrays: bool,
    batch_override: int | None = None,
) -> list[str]:
    batch_size = batch_override or spec.batch_size
    return [
        sys.executable,
        "main.py",
        "--gpu_id",
        "0",
        "--use_multi_gpu",
        "0",
        "--is_training",
        "1",
        "--model_id",
        mid,
        "--model_name",
        model_name,
        "--dataset_root_path",
        spec.root,
        "--dataset_name",
        dataset,
        "--features",
        "M",
        "--seq_len",
        str(spec.seq_len),
        "--pred_len",
        str(spec.pred_len),
        "--enc_in",
        str(spec.enc_in),
        "--dec_in",
        str(spec.enc_in),
        "--c_out",
        str(spec.enc_in),
        "--train_epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--val_interval",
        "1",
        "--itr",
        "1",
        "--seed_base",
        str(seed),
        "--batch_size",
        str(batch_size),
        "--learning_rate",
        str(spec.lr),
        "--d_model",
        str(spec.d_model),
        "--dropout",
        str(spec.dropout),
        "--apn_npatch",
        str(spec.apn_npatch),
        "--apn_te_dim",
        str(spec.apn_te_dim),
        "--num_workers",
        str(num_workers),
        "--pin_memory",
        "1" if num_workers > 0 else "0",
        "--persistent_workers",
        "1" if num_workers > 0 else "0",
        "--prefetch_factor",
        "2" if num_workers > 0 else "1",
        "--non_blocking_transfer",
        "1",
        "--tf32",
        "1",
        "--cudnn_benchmark",
        "1",
        "--amp_dtype",
        "bf16",
        "--disable_tqdm",
        "1",
        "--save_arrays",
        "1" if save_arrays else "0",
        "--checkpoints",
        str(root),
    ]


def tpatch_extra(dataset: str) -> list[str]:
    patch_len = {
        "HumanActivity": 300,
        "P12": 6,
        "USHCN": 10,
        "MHEALTH": 25,
        "PAMAP2": 50,
        "OPPORTUNITY": 30,
    }[dataset]
    batch_size = "4" if dataset in {"MHEALTH", "PAMAP2", "OPPORTUNITY"} else "32"
    return [
        "--loss",
        "MSE",
        "--collate_fn",
        "collate_fn_patch",
        "--patch_len",
        str(patch_len),
        "--n_heads",
        "1",
        "--learning_rate",
        "0.001",
        "--batch_size",
        batch_size,
    ]


def baseline_extra(model_name: str, dataset: str) -> list[str]:
    if model_name == "tPatchGNN":
        return tpatch_extra(dataset)
    extra = list(MODEL_EXTRAS[model_name])
    if model_name == "CRU" and dataset == "HumanActivity":
        extra = replace_args(
            extra,
            {
                "--learning_rate": "0.0001",
                "--batch_size": "8",
                "--amp_dtype": "off",
            },
        )
    return extra


def sr_mcli_args(tag: str) -> list[str]:
    """Return the fixed local SR-MCLI configuration or one controlled ablation."""
    base = [
        "--loss",
        "PairedStudentizedMSE",
        "--apn_research_variant",
        "mechanism__adaptive_fourier_varmix__lagdep__sapn_operator",
        "--apn_fourier_init",
        "-2.2",
        "--apn_varmix_init",
        "-2.8",
        "--apn_selector_level",
        "variable",
        "--apn_selector_l1",
        "8e-5",
        "--apn_selector_residual",
        "4e-5",
        "--apn_selector_entropy",
        "0.0",
        "--apn_selector_init",
        "-1.1",
        "--apn_selector_scale",
        "0.85",
        "--apn_selector_safety",
        "1",
        "--apn_selector_safety_init",
        "-2.5",
        "--apn_selector_safety_temperature",
        "0.9",
        "--apn_selector_temperature",
        "0.85",
        "--apn_selector_branch_mask",
        "all",
        "--apn_selector_branch_dropout",
        "0.08",
        "--apn_selector_stat_control",
        "real",
        "--apn_selector_uncertainty_mode",
        "smooth",
        "--apn_selector_uncertainty_weight",
        "0.20",
        "--apn_selector_trust_weight",
        "5e-5",
        "--apn_selector_trust_cap",
        "0.00035",
        "--apn_selector_mass_weight",
        "0.10",
        "--apn_selector_mass_min",
        "0.02",
        "--apn_selector_mass_max",
        "0.28",
        "--apn_operator_price_weight",
        "0.04",
        "--apn_operator_price_margin",
        "0.00010",
        "--apn_operator_price_init",
        "0.02",
        "--apn_operator_dep_branch",
        "hybrid",
        "--apn_emit_noop_pred",
        "1",
        "--apn_emit_selector_arrays",
        "1",
        "--apn_paired_safe_weight",
        "5.0",
        "--apn_paired_safe_margin",
        "0.0",
        "--apn_paired_cvar_q",
        "0.30",
        "--apn_paired_benefit_weight",
        "0.018",
        "--apn_paired_benefit_margin",
        "0.0",
        "--apn_paired_benefit_temperature",
        "0.002",
        "--apn_paired_ratio_rho",
        "0.10",
        "--apn_paired_ratio_eps",
        "0.0005",
        "--apn_paired_ratio_cap",
        "0.060",
        "--apn_paired_lcb_var_weight",
        "0.0",
        "--apn_paired_lcb_var_cap",
        "0.05",
        "--apn_paired_ntr_weight",
        "0.0",
        "--apn_paired_ntr_temperature",
        "0.02",
    ]
    if tag == "sr_mcli_default":
        return base
    if tag == "ablate_raw_tail":
        return replace_args(base, {"--loss": "PairedCVaRMSE", "--apn_paired_cvar_q": "0.20", "--apn_paired_ratio_cap": "0.0"})
    if tag == "ablate_no_safety":
        return replace_args(base, {"--apn_selector_safety": "0", "--apn_paired_safe_weight": "0.0"})
    if tag == "ablate_no_scale_norm":
        return replace_args(base, {"--loss": "PairedCVaRMSE", "--apn_paired_ratio_rho": "0.0", "--apn_paired_ratio_cap": "0.0"})
    if tag == "ablate_no_mass_band":
        return replace_args(base, {"--apn_selector_mass_weight": "0.0"})
    if tag == "ablate_lcb_var":
        return replace_args(base, {"--loss": "PairedScaleRobustMSE", "--apn_paired_lcb_var_weight": "0.20"})
    if tag == "ablate_ntr":
        return replace_args(base, {"--loss": "PairedScaleRobustMSE", "--apn_paired_ntr_weight": "0.003"})
    raise ValueError(f"unknown SR-MCLI tag: {tag}")


def replace_args(args: list[str], replacements: dict[str, str]) -> list[str]:
    out = list(args)
    for key, value in replacements.items():
        try:
            idx = out.index(key)
        except ValueError:
            out.extend([key, value])
        else:
            out[idx + 1] = value
    return out


def metric_exists(root: Path, dataset: str, model_name: str, mid: str) -> bool:
    return any((root / dataset / model_name / mid).glob("*/*/iter*/eval_*/metric.json"))


def run_command(cmd: list[str], log_file: Path, env: dict[str, str], dry_run: bool) -> int:
    print("$", " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return 0
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_file.open("w", encoding="utf-8", errors="replace") as handle:
        proc = subprocess.run(
            cmd,
            cwd=REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    minutes = (time.time() - start) / 60.0
    print(f"exit={proc.returncode} time={minutes:.1f}min log={log_file}")
    return proc.returncode


def build_runs(args: argparse.Namespace) -> list[tuple[str, str, str, int, list[str], bool]]:
    datasets = csv_list(args.datasets)
    seeds = seed_list(args.seeds)
    runs: list[tuple[str, str, str, int, list[str], bool]] = []

    baseline_models = csv_list(args.baselines)
    if "tPatchGNN" in baseline_models:
        MODEL_EXTRAS["tPatchGNN"] = []

    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset}; choices={sorted(DATASETS)}")
        for seed in seeds:
            if args.mode in {"smoke", "main", "baselines"}:
                for model_name in baseline_models:
                    extra = baseline_extra(model_name, dataset)
                    tag = model_name.lower()
                    runs.append((dataset, model_name, tag, seed, extra, args.save_arrays_baselines))
            if args.mode in {"smoke", "main", "ablation"}:
                sr_tags = ["sr_mcli_default"]
                if args.mode == "ablation":
                    sr_tags = csv_list(args.sr_tags) if args.sr_tags else [
                        "sr_mcli_default",
                        "ablate_raw_tail",
                        "ablate_no_safety",
                        "ablate_no_scale_norm",
                        "ablate_no_mass_band",
                        "ablate_lcb_var",
                        "ablate_ntr",
                    ]
                for tag in sr_tags:
                    runs.append((dataset, "APNResearch", tag, seed, sr_mcli_args(tag), True))
    return runs[: args.max_runs] if args.max_runs else runs


def summarize(root: Path, out_name: str = "summary_local_complete.csv") -> Path:
    rows: list[dict[str, str]] = []
    pattern = re.compile(r"^local_(?P<mode>[^_]+)_(?P<dataset>[^_]+)_(?P<tag>.*)_e(?P<epochs>[0-9]+)_s(?P<seed>[0-9]+)$")
    for metric_path in sorted(root.glob("*/*/*/*/*/iter*/eval_*/metric.json")):
        try:
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        model_id_path = metric_path.parents[4]
        model_name = metric_path.parents[5].name
        dataset = metric_path.parents[6].name
        match = pattern.match(model_id_path.name)
        row = {
            "dataset": dataset,
            "model": model_name,
            "tag": match.group("tag") if match else model_id_path.name,
            "mode": match.group("mode") if match else "",
            "epochs": match.group("epochs") if match else "",
            "seed": match.group("seed") if match else "",
            "MAE": str(metric.get("MAE", "")),
            "MSE": str(metric.get("MSE", "")),
            "metric_path": str(metric_path),
        }
        diag_path = metric_path.parent / "selector_diagnostics.json"
        if diag_path.exists():
            try:
                diag = json.loads(diag_path.read_text(encoding="utf-8"))
                for key in [
                    "effective_gate_mean",
                    "effective_gate_active_0_01",
                    "safety_gate_mean",
                    "safety_gate_active_0_01",
                    "operator_price_mean",
                ]:
                    if key in diag:
                        row[key] = str(diag[key])
            except Exception:
                pass
        rows.append(row)

    out = root / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    if rows:
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        out.write_text("", encoding="utf-8")
    print(out)
    if rows:
        for row in rows[-10:]:
            print(f"{row['dataset']:13s} {row['model']:12s} {row['tag']:22s} seed={row.get('seed','')} MAE={row['MAE']} MSE={row['MSE']}")
    return out


def run_audits(root: Path, env: dict[str, str], dry_run: bool) -> None:
    audit_cmds = [
        [sys.executable, "scripts/audit_BigData2026_v55_paired_certificate.py", "--root", str(root), "--delta", "0.05", "--clip", "1.0"],
        [sys.executable, "scripts/audit_BigData2026_v66_crossfit_certificate.py", "--root", str(root), "--delta", "0.05", "--clip", "1.0"],
        [sys.executable, "scripts/audit_BigData2026_v70_union_family_certificate.py", "--root", str(root), "--delta", "0.05", "--clip", "1.0"],
        [sys.executable, "scripts/analyze_BigData2026_v68_scale_buckets.py", "--root", str(root)],
    ]
    for cmd in audit_cmds:
        name = Path(cmd[1]).stem
        run_command(cmd, root / "logs" / f"{name}.log", env, dry_run)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "main", "baselines", "ablation", "audit", "summary"], default="smoke")
    parser.add_argument("--root", type=Path, default=Path("storage/results_BigData2026_local_complete"))
    parser.add_argument("--datasets", default="P12,USHCN,HumanActivity")
    parser.add_argument("--baselines", default="APN,KAFNet,tPatchGNN,GraFITi,mTAN,CRU")
    parser.add_argument("--sr_tags", default="", help="comma-separated APNResearch/SR-MCLI tags for --mode ablation")
    parser.add_argument("--seeds", default="3001")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_runs", type=int, default=0)
    parser.add_argument("--skip_done", action="store_true", default=True)
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_audits", action="store_true")
    parser.add_argument("--save_arrays_baselines", action="store_true")
    parsed = parser.parse_args(list(argv) if argv is not None else None)

    if parsed.mode == "summary":
        summarize(parsed.root)
        return 0
    if parsed.mode == "audit":
        env = os.environ.copy()
        env.setdefault("TQDM_DISABLE", "1")
        run_audits(parsed.root, env, parsed.dry_run)
        summarize(parsed.root)
        return 0

    epochs = parsed.epochs if parsed.epochs is not None else (1 if parsed.mode == "smoke" else 5)
    patience = parsed.patience if parsed.patience is not None else (1 if parsed.mode == "smoke" else 3)
    root = parsed.root
    (root / "logs").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("TQDM_DISABLE", "1")
    env.setdefault("PYTHONUTF8", "1")

    failures: list[str] = []
    runs = build_runs(parsed)
    print(f"planned_runs={len(runs)} root={root} epochs={epochs} seeds={parsed.seeds}")
    for dataset, model_name, tag, seed, extra, save_arrays in runs:
        spec = DATASETS[dataset]
        mid = model_id(f"local_{parsed.mode}", dataset, tag, epochs, seed)
        if parsed.skip_done and metric_exists(root, dataset, model_name, mid):
            print(f"skip done: {mid}")
            continue
        batch_override = 16 if dataset == "HumanActivity" and model_name == "APNResearch" else None
        cmd = common_args(
            dataset=dataset,
            spec=spec,
            model_name=model_name,
            mid=mid,
            root=root,
            epochs=epochs,
            patience=patience,
            seed=seed,
            num_workers=parsed.num_workers,
            save_arrays=save_arrays,
            batch_override=batch_override,
        )
        cmd.extend(extra)
        rc = run_command(cmd, root / "logs" / f"{mid}.log", env, parsed.dry_run)
        if rc != 0:
            failures.append(mid)
        summarize(root)

    if parsed.run_audits:
        run_audits(root, env, parsed.dry_run)
        summarize(root)

    if failures:
        print("failed runs:")
        for item in failures:
            print(f"  {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
