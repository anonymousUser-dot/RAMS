#!/usr/bin/env python
"""v328 soft-admission ablation for RAMS-tPatch.

This runner reuses the v316 RAMS-tPatch strong-backbone trainer and adds
registered variants that isolate the method components introduced in the
paper:

default_gate:    closed branch gates + warmup + supervised switch + residual L2
no_warmup:       train the switch from the first stage
open_gate:       less conservative branch/switch initialization
no_residual_l2:  remove residual-energy penalty
no_supervised:   remove detached branch-quality switch target
no_boundary:     remove switch boundary penalty
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


runner.VARIANTS.update({
    "default_gate": dict(runner.VARIANTS["gate"]),
    "no_warmup": {
        "force": "none",
        "supervised": "0.02",
        "stages": "1",
        "residual_l2": "1e-4",
        "gate_init": "-3.0",
        "switch_init": "-2.5",
        "boundary": "2e-4",
    },
    "open_gate": {
        "force": "none",
        "supervised": "0.02",
        "stages": "2",
        "residual_l2": "1e-4",
        "gate_init": "-1.0",
        "switch_init": "0.0",
        "boundary": "2e-4",
    },
    "no_residual_l2": {
        "force": "none",
        "supervised": "0.02",
        "stages": "2",
        "residual_l2": "0.0",
        "gate_init": "-3.0",
        "switch_init": "-2.5",
        "boundary": "2e-4",
    },
    "no_supervised": {
        "force": "none",
        "supervised": "0.0",
        "stages": "2",
        "residual_l2": "1e-4",
        "gate_init": "-3.0",
        "switch_init": "-2.5",
        "boundary": "2e-4",
    },
    "no_boundary": {
        "force": "none",
        "supervised": "0.02",
        "stages": "2",
        "residual_l2": "1e-4",
        "gate_init": "-3.0",
        "switch_init": "-2.5",
        "boundary": "0.0",
    },
})


def main() -> int:
    if "--root" not in sys.argv:
        sys.argv.extend(["--root", "storage/results_BigData2026_v328_soft_admission_ablation"])
    if "--prefix" not in sys.argv:
        sys.argv.extend(["--prefix", "v328"])
    if "--variants" not in sys.argv:
        sys.argv.extend(["--variants", "default_gate,no_warmup,open_gate,no_residual_l2,no_supervised,no_boundary"])
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
