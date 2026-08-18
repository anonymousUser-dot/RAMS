# APNResearch Experiment Notes

## Current Backbone Decision

- Baseline: APN official implementation.
- Current working backbone: `APNResearch` spectral patch backbone.
- Preferred default for the next round: `adaptive_fourier` on HumanActivity/USHCN, and `gated_fourier` on P12.
- Reason: 5-epoch confirmation showed adaptive Fourier is best on HumanActivity and USHCN, while P12 prefers the simpler global spectral gate.

## Implemented Variants

- `base`: APN-equivalent copy inside `APNResearch`.
- `multiscale`: multiple adaptive patch banks with interpolation-based fusion.
- `density`: patch representation gated by local observation density.
- `fourier`: ungated spectral residual over patch tokens.
- `gated_fourier`: spectral residual with a small learnable global gate.
- `adaptive_fourier`: spectral residual with sample/variable-conditioned gate.
- `varmix`: dynamic variable dependency modeling with multi-head attention.
- `gated_varmix`: variable dependency residual with small per-variable gate.
- `fourier_varmix`, `gated_fourier_varmix`, `adaptive_fourier_varmix`: combined variants.

## 1-Epoch Screening on Server `connect.bjb1.seetacloud.com:25490`

| Variant | HumanActivity MSE / MAE | P12 MSE / MAE | USHCN MSE / MAE |
| --- | ---: | ---: | ---: |
| base | 0.064132 / 0.162764 | 0.321839 / 0.370667 | 0.336696 / 0.371871 |
| fourier | 0.062372 / 0.154208 | 0.322208 / 0.377251 | 0.216204 / 0.348588 |
| gated_fourier | 0.062357 / 0.156236 | 0.321543 / 0.377998 | 0.241471 / 0.388074 |
| adaptive_fourier | 0.060155 / 0.158754 | 0.321341 / 0.374613 | 0.283917 / 0.442480 |
| gated_varmix | 0.063132 / 0.163011 | 0.323544 / 0.378670 | 0.312134 / 0.405525 |
| adaptive_fourier_varmix | 0.069904 / 0.182884 | 0.331365 / 0.392997 | 0.294603 / 0.392889 |

## Adaptive Fourier Init Sweep, 1 Epoch

| Fourier gate init | HumanActivity MSE / MAE | P12 MSE / MAE | USHCN MSE / MAE |
| --- | ---: | ---: | ---: |
| -4.0 | 0.059816 / 0.157821 | 0.319951 / 0.375768 | 0.329570 / 0.484421 |
| -3.0 | 0.059572 / 0.157378 | 0.322106 / 0.377203 | 0.331125 / 0.485661 |
| -2.0 | 0.060155 / 0.158754 | 0.321341 / 0.374613 | 0.283917 / 0.442480 |
| -1.0 | 0.060036 / 0.158459 | 0.321541 / 0.373949 | 0.308809 / 0.465370 |

## Best 5-Epoch Confirmation So Far

| Dataset | base MSE / MAE | gated_fourier MSE / MAE | adaptive_fourier MSE / MAE | Best |
| --- | ---: | ---: | ---: | --- |
| HumanActivity | 0.043506 / 0.121915 | 0.043775 / 0.124047 | 0.043401 / 0.121455 | adaptive_fourier |
| P12 | 0.317591 / 0.367637 | 0.316144 / 0.373257 | 0.323472 / 0.388270 | gated_fourier |
| USHCN | 0.191998 / 0.319577 | 0.168030 / 0.268040 | 0.161798 / 0.264641 | adaptive_fourier |

## Takeaways

- Spectral patch residuals are the strongest improvement family.
- Adaptive/sample-conditioned Fourier gating is promising, but not universally better: P12 overfits or destabilizes with the adaptive gate at 5 epochs.
- Variable dependency mixing (`varmix`) is not yet useful in its current form. It should be redesigned as a constrained residual or dataset-specific low-rank dependency module rather than kept as a full dynamic attention block.
- For the next backbone, keep APN patching and add a dataset-aware spectral gate: adaptive for smooth sparse environmental/activity datasets, global gated Fourier for ICU/P12.

## Next Experiment

- Add an automatic gate regularizer/selector so P12 can stay close to global gated Fourier while HumanActivity/USHCN use adaptive Fourier.
- Test longer runs for `adaptive_fourier` with init `-3.0` on HumanActivity and `-4.0` on P12, plus the confirmed `-2.0` setting for USHCN.

## 30-Method Search, Started 2026-05-08

Server: `connect.bjb1.seetacloud.com:25490`

Script: `scripts/run_apnresearch_30_method_search.sh`

Goal: screen 30 lightweight structural innovations on top of APNResearch spectral patching. Each candidate is run for 1 epoch on HumanActivity, P12, and USHCN, then promising ones should be promoted to 5-epoch and full-length confirmation.

Method families:

- Patch token refinement: `patchconv`, `wavelet`, `trend`
- Variable dependency: `lowrank`, `graph`, `covmix`, `segate`
- Decoder adaptation: `filmdec`, `resdec`
- Spectral bases: `fourier`, `gated_fourier`, `adaptive_fourier`

First completed candidate:

| Variant | HumanActivity MSE / MAE | P12 MSE / MAE | USHCN MSE / MAE |
| --- | ---: | ---: | ---: |
| fourier__patchconv | 0.073153 / 0.195298 | 0.320993 / 0.381142 | 0.232305 / 0.384081 |

Early read: `patchconv` helps P12 relative to plain `fourier` MSE, but hurts HumanActivity and does not beat the best USHCN spectral result. Keep it as a P12-specific candidate only.
