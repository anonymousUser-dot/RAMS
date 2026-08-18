# Model Checkpoints

The synchronized trained model files are stored in the mirrored experiment tree:

`../code/storage/**/pytorch_model.bin`

`CHECKPOINT_MANIFEST.csv` lists 122 checkpoint files included in this package.
Most tPatchGNN, RAMS-tPatch, and public-baseline checkpoints are present because
the corresponding result folders were synchronized locally.

The final KAFNet server confirmation was synchronized as summary CSVs only, so
its checkpoint binaries are not included in this local package. The KAFNet table
values can still be verified from:

`../code/storage/results_BigData2026_v358_kafnet_rams_server/_summary/`
