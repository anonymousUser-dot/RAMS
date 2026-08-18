$ErrorActionPreference = "Stop"

Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUNBUFFERED = "1"
$env:TQDM_DISABLE = "1"

python code/scripts/run_BigData2026_v322_ramstpatch_final.py `
  --mode summary `
  --datasets MHEALTH,OPPORTUNITY,PAMAP2 `
  --seeds 5301,5302,5303 `
  --bootstrap_rounds 10000

python code/scripts/run_BigData2026_v316_ramstpatch_strong.py `
  --root storage/results_BigData2026_v323_strong8_gate `
  --base_roots storage/results_BigData2026_v323_strong8_gate,storage/results_BigData2026_v322_public_baselines_e4,storage/results_BigData2026_v322_ramstpatch_final `
  --prefix v323 `
  --datasets MHEALTH,OPPORTUNITY,PAMAP2 `
  --seeds 5301,5302,5303 `
  --variants gate `
  --base_epochs 8 `
  --residual_epochs 3 `
  --summarize_only 1

python commands/compute_kafnet_ci.py `
  --detail code/storage/results_BigData2026_v358_kafnet_rams_server/_summary/v351_kafnet_rams_detail.csv `
  --rounds 10000
