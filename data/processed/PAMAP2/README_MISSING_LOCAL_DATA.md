# PAMAP2 Processed Tensor Note

The local workspace did not contain a non-empty `storage/datasets/PAMAP2/processed`
snapshot when this package was assembled. The paper's PAMAP2 metrics are preserved
in the synchronized result CSVs under:

- `results/results_BigData2026_v322_ramstpatch_final`
- `results/results_BigData2026_v323_strong8_gate`
- `results/results_BigData2026_v329_mechanism_controls_server53034`
- `results/results_BigData2026_v358_kafnet_rams_server`

To rerun PAMAP2 from scratch, place/build the processed tensor at:

`storage/datasets/PAMAP2/processed/chunks_sl500_pl100.pt`

The loader used by the paper is:

`code/data/data_provider/datasets/PAMAP2.py`
