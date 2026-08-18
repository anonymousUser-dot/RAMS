$ErrorActionPreference = "Stop"

Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUNBUFFERED = "1"
$env:TQDM_DISABLE = "1"

python code/scripts/run_BigData2026_v322_ramstpatch_final.py `
  --mode all `
  --datasets MHEALTH,OPPORTUNITY,PAMAP2 `
  --seeds 5301,5302,5303 `
  --base_epochs 4 `
  --residual_epochs 3 `
  --num_workers 2 `
  --gpu_id 0

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

python code/scripts/run_BigData2026_v329_mechanism_controls.py `
  --root storage/results_BigData2026_v329_mechanism_controls `
  --prefix v329 `
  --datasets MHEALTH,OPPORTUNITY,PAMAP2 `
  --seeds 5301,5302,5303 `
  --variants real_gate,value_only_gate,reliability_only_gate,shuffled_gate,random_gate,constant_gate `
  --base_roots storage/results_BigData2026_v322_public_baselines_e4,storage/results_BigData2026_v323_strong8_gate,storage/results_BigData2026_v322_ramstpatch_final `
  --gpu_id 0

python code/scripts/analyze_BigData2026_v327_value_stratified_reliability_split.py `
  --datasets MHEALTH,OPPORTUNITY,PAMAP2 `
  --splits train,val,test `
  --out storage/results_BigData2026_v327_value_stratified_reliability_split `
  --level_bins 5 `
  --reliability_bins 3 `
  --min_cell 8

if (Test-Path "code/storage/datasets/PAMAP2/processed/chunks_sl500_pl100.pt") {
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
} else {
  Write-Warning "Skipping KAFNet full rerun because PAMAP2 processed tensor is not included in this local snapshot. KAFNet summary CSVs are included under code/storage/results_BigData2026_v358_kafnet_rams_server/_summary."
}
