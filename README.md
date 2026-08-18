# RAMS BigData 2026 Reproducibility Package

This folder is the paper-specific reproduction package for:

**RAMS: Mechanism-State Residual Operators for Wearable IMTS Forecasting**

It contains the source code, processed data available in this local workspace,
trained checkpoints/results that were synchronized locally, and exact commands
for regenerating every figure/table in `paper_snapshot/BigData2026_rams_multibackbone_v357.pdf`.

## 1. Folder Layout

- `code/`: runnable source code snapshot used by the paper.
- `data/processed/`: processed tensors used by the local reproduction runs.
- `results/`: synchronized experiment outputs, summaries, logs, and checkpoints.
- `models_and_checkpoints/`: checkpoint notes and manifest generated from `results/`.
- `commands/`: convenience command scripts.
- `paper_snapshot/`: the exact LaTeX/PDF snapshot linked to this package.

The most important implementation files are:

- `code/models/MCORWrapper.py`: RAMS mechanism-state residual operator.
- `code/models/KAFNet.py`: KAFNet backbone used in the second-backbone confirmation.
- `code/scripts/run_BigData2026_v316_ramstpatch_strong.py`: frozen tPatchGNN + RAMS runner.
- `code/scripts/run_BigData2026_v322_ramstpatch_final.py`: main RAMS-tPatch experiment orchestrator.
- `code/scripts/run_BigData2026_v329_mechanism_controls.py`: real/value-only/shuffled/random/constant mechanism-state controls.
- `code/scripts/run_BigData2026_v351_kafnet_rams_backbone.py`: KAFNet + RAMS runner.

## 2. Environment

From this package root, use `code/` as the runnable repository root.  The
package mirrors the necessary `storage/` tree under `code/storage/`, because the
paper scripts resolve paths relative to `code/`.

Minimal setup:

```powershell
pip install -r code/requirements.txt
$env:PYTHONUNBUFFERED="1"
$env:TQDM_DISABLE="1"
```

GPU is recommended for training. Summary-only commands can run on CPU.

## 3. Data Included

Included processed data is mirrored in both `data/processed/` for inspection and
`code/storage/datasets/` for direct execution:

- `data/processed/HumanActivity/processed/data.pt`
- `data/processed/MHEALTH/processed/data.pt`
- `data/processed/MHEALTH/processed/chunks_sl250_pl50.pt`
- `data/processed/OPPORTUNITY/processed/data.pt`
- `data/processed/OPPORTUNITY/processed/chunks_sl300_pl60.pt`

`PAMAP2` processed tensors are not present in this local snapshot; the runner
expects them under `code/storage/datasets/PAMAP2/processed/chunks_sl500_pl100.pt`.
The original experiments used the same `data/data_provider/datasets/PAMAP2.py`
loader and the synchronized result CSVs are included under `results/`.

## 4. Trained Checkpoints and Results

The synchronized result folders are mirrored in both `results/` for inspection
and `code/storage/` for direct execution.  The folders are:

- `results/results_BigData2026_v322_public_baselines_e4`: public APN/KAFNet/GraFITi/tPatchGNN 4-epoch baseline scan.
- `results/results_BigData2026_v322_ramstpatch_final`: main RAMS-tPatch rows, ablations, summaries, and many tPatchGNN/RAMS checkpoints.
- `results/results_BigData2026_v323_strong8_gate`: 8-epoch strong-backbone tPatchGNN + RAMS confirmation.
- `results/results_BigData2026_v328_soft_admission_ablation`: extra soft-admission ablation summary.
- `results/results_BigData2026_v329_mechanism_controls_server53034`: mechanism-state controls.
- `results/results_BigData2026_v327_value_stratified_reliability_split_server53034`: alias diagnostic.
- `results/results_BigData2026_v351_kafnet_rams_a`: KAFNet + RAMS summary mirror.
- `results/results_BigData2026_v358_kafnet_rams_server`: final KAFNet + RAMS summary synced from server.

To list synchronized PyTorch checkpoints:

```powershell
Get-ChildItem code/storage -Recurse -Filter pytorch_model.bin
```

The KAFNet server run was synchronized as summary CSVs only; its final
checkpoint files were not present in the local snapshot at packaging time.

## 5. Reproduction Commands

### 5.1 Main public baselines and RAMS-tPatch

```powershell
python code/scripts/run_BigData2026_v322_ramstpatch_final.py `
  --mode all `
  --datasets MHEALTH,OPPORTUNITY,PAMAP2 `
  --seeds 5301,5302,5303 `
  --base_epochs 4 `
  --residual_epochs 3 `
  --num_workers 2 `
  --gpu_id 0
```

Summary only, using synchronized outputs:

```powershell
python code/scripts/run_BigData2026_v322_ramstpatch_final.py `
  --mode summary `
  --datasets MHEALTH,OPPORTUNITY,PAMAP2 `
  --seeds 5301,5302,5303
```

### 5.2 Strong 8-epoch tPatchGNN + RAMS check

```powershell
python code/scripts/run_BigData2026_v316_ramstpatch_strong.py `
  --root storage/results_BigData2026_v323_strong8_gate `
  --base_roots storage/results_BigData2026_v323_strong8_gate,storage/results_BigData2026_v322_public_baselines_e4,storage/results_BigData2026_v322_ramstpatch_final `
  --prefix v323 `
  --datasets MHEALTH,OPPORTUNITY,PAMAP2 `
  --seeds 5301,5302,5303 `
  --variants gate `
  --base_epochs 8 `
  --residual_epochs 3 `
  --num_workers 2 `
  --gpu_id 0
```

### 5.3 KAFNet + RAMS confirmation

```powershell
python code/scripts/run_BigData2026_v351_kafnet_rams_backbone.py `
  --root storage/results_BigData2026_v351_kafnet_rams `
  --base_roots storage/results_BigData2026_v351_kafnet_rams,storage/results_BigData2026_v322_public_baselines_e4,storage/results_BigData2026_v322_ramstpatch_final `
  --prefix v351a `
  --datasets HumanActivity,MHEALTH,OPPORTUNITY,PAMAP2 `
  --seeds 3001,3002,3003 `
  --variants gate `
  --base_epochs 4 `
  --residual_epochs 3 `
  --num_workers 2 `
  --gpu_id 0
```

### 5.4 Mechanism-state controls

```powershell
python code/scripts/run_BigData2026_v329_mechanism_controls.py `
  --root storage/results_BigData2026_v329_mechanism_controls `
  --prefix v329 `
  --datasets MHEALTH,OPPORTUNITY,PAMAP2 `
  --seeds 5301,5302,5303 `
  --variants real_gate,value_only_gate,reliability_only_gate,shuffled_gate,random_gate,constant_gate `
  --base_roots storage/results_BigData2026_v322_public_baselines_e4,storage/results_BigData2026_v323_strong8_gate,storage/results_BigData2026_v322_ramstpatch_final `
  --gpu_id 0
```

### 5.5 Mechanism-state aliasing diagnostic

```powershell
python code/scripts/analyze_BigData2026_v327_value_stratified_reliability_split.py `
  --datasets MHEALTH,OPPORTUNITY,PAMAP2 `
  --splits train,val,test `
  --out storage/results_BigData2026_v327_value_stratified_reliability_split `
  --level_bins 5 `
  --reliability_bins 3 `
  --min_cell 8
```

## 6. Figure/Table Provenance

| Paper item | Source data/file | Command to regenerate |
|---|---|---|
| Fig. 1 `fig:architecture` | TikZ source inside `paper_snapshot/BigData2026_rams_multibackbone_v357.tex`; editable Draw.io sketch in submission resources | `pdflatex -interaction=nonstopmode -halt-on-error -output-directory paper/build_v357_multibackbone paper/BigData2026_rams_multibackbone_v357.tex` |
| Algorithm 1 `alg:ramstpatch` | LaTeX algorithm block in paper source; executable protocol in `code/scripts/run_BigData2026_v322_ramstpatch_final.py` and `code/scripts/run_BigData2026_v316_ramstpatch_strong.py` | Compile paper; run commands in Sec. 5.1/5.2 |
| Table 1 `tab:baseline-scan` | `results/results_BigData2026_v322_ramstpatch_final/_summary_final/v322_public_baseline_avg.csv`; `results/results_BigData2026_v322_ramstpatch_final/_summary_final/v322_ramstpatch_main_table.csv`; `results/results_BigData2026_v358_kafnet_rams_server/_summary/v351_kafnet_rams_agg.csv` | Sec. 5.1 and Sec. 5.3 |
| Table 2 `tab:protocol` | Fixed arguments in `run_BigData2026_v322_ramstpatch_final.py`, `run_BigData2026_v316_ramstpatch_strong.py`, and `run_BigData2026_v351_kafnet_rams_backbone.py` | No metric run needed; inspect parser defaults or use `--dry_run 1` |
| Table 3 `tab:main` | `results/results_BigData2026_v322_ramstpatch_final/_summary_final/v322_ramstpatch_main_table.csv` | Sec. 5.1 |
| Table 4 `tab:ci` | `results/results_BigData2026_v322_ramstpatch_final/_summary_final/v322_ramstpatch_paired_ci.csv` | `python code/scripts/run_BigData2026_v322_ramstpatch_final.py --mode summary --seeds 5301,5302,5303 --bootstrap_rounds 10000` |
| Table 5 `tab:strong` | `results/results_BigData2026_v323_strong8_gate/_summary/v323_strong8_paired_table.csv` | Sec. 5.2 |
| Table 6 `tab:kafnet-confirm` | `results/results_BigData2026_v358_kafnet_rams_server/_summary/v351_kafnet_rams_agg.csv` and `v351_kafnet_rams_detail.csv` | Sec. 5.3 |
| Table 7 `tab:kafnet-ci` | Computed from `results/results_BigData2026_v358_kafnet_rams_server/_summary/v351_kafnet_rams_detail.csv` | `python commands/compute_kafnet_ci.py --detail code/storage/results_BigData2026_v358_kafnet_rams_server/_summary/v351_kafnet_rams_detail.csv --rounds 10000` |
| Table 8 `tab:branch-ablation` | `results/results_BigData2026_v322_ramstpatch_final/_summary_final/v322_ramstpatch_ablation_table.csv` | `python code/scripts/run_BigData2026_v322_ramstpatch_final.py --mode ablation --seeds 5301,5302,5303` |
| Table 9 `tab:state-control` | `results/results_BigData2026_v329_mechanism_controls_server53034/_summary/v329_mechanism_control_table.csv` and detail CSV | Sec. 5.4 |
| Table 10 `tab:alias` | `results/results_BigData2026_v327_value_stratified_reliability_split_server53034/v327_value_stratified_reliability_split.csv` | Sec. 5.5 |

## 7. Notes on Exact Reproduction

- All reported gains are paired against the matched frozen backbone checkpoint.
- The default RAMS variant is `gate`; ablations do not redefine the default.
- The paper reports ordinary test MSE/MAE.
- If reproducing from scratch, ensure `code/storage/datasets/PAMAP2/processed/chunks_sl500_pl100.pt` exists before running PAMAP2 rows.
- Server-side KAFNet checkpoints were not fully copied back; the included KAFNet evidence is summary-level and sufficient for paper table verification.
