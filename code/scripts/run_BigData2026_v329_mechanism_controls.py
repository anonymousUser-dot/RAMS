#!/usr/bin/env python
"""v329 mechanism-state controls for RAMS-tPatch.

This run keeps the RAMS-tPatch architecture, optimizer, seeds, and matched
tPatchGNN checkpoints fixed, and changes only the state exposed to the
residual operator:

real_gate:             full deployment-visible reliability/value state
value_only_gate:       parameter-matched value residual, reliability zeroed
reliability_only_gate: reliability residual with value moments zeroed
shuffled_gate:         batch-rolled mechanism state, preserving marginal scale
random_gate:           random Gaussian mechanism state with matching shape
constant_gate:         constant zero mechanism state

These controls directly test whether the gain comes from wearable reliability
aliasing rather than from adding a generic residual MLP.  v331 extends the
same script with random/constant controls.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent / "run_BigData2026_v316_ramstpatch_strong.py"
spec = importlib.util.spec_from_file_location("run_BigData2026_v316_ramstpatch_strong", SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load {SCRIPT_PATH}")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


BASE_GATE = dict(runner.VARIANTS["gate"])

runner.VARIANTS.update({
    "real_gate": {
        **BASE_GATE,
        "reliability_mode": "real",
    },
    "value_only_gate": {
        **BASE_GATE,
        "reliability_mode": "value_only",
    },
    "reliability_only_gate": {
        **BASE_GATE,
        "reliability_mode": "reliability_only",
    },
    "shuffled_gate": {
        **BASE_GATE,
        "reliability_mode": "shuffled",
    },
    "random_gate": {
        **BASE_GATE,
        "reliability_mode": "random",
    },
    "constant_gate": {
        **BASE_GATE,
        "reliability_mode": "constant",
    },
})


def main() -> int:
    if "--root" not in sys.argv:
        sys.argv.extend(["--root", "storage/results_BigData2026_v329_mechanism_controls"])
    if "--prefix" not in sys.argv:
        sys.argv.extend(["--prefix", "v329"])
    if "--variants" not in sys.argv:
        sys.argv.extend(["--variants", "real_gate,value_only_gate,reliability_only_gate,shuffled_gate,random_gate,constant_gate"])
    if "--datasets" not in sys.argv:
        sys.argv.extend(["--datasets", "MHEALTH,OPPORTUNITY,PAMAP2"])
    if "--seeds" not in sys.argv:
        sys.argv.extend(["--seeds", "5301,5302,5303"])
    if "--base_roots" not in sys.argv:
        sys.argv.extend([
            "--base_roots",
            "storage/results_BigData2026_v322_public_baselines_e4,storage/results_BigData2026_v323_strong8_gate,storage/results_BigData2026_v322_ramstpatch_final",
        ])
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
