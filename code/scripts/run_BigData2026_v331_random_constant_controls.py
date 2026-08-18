#!/usr/bin/env python
"""v331 random/constant controls for RAMS-tPatch.

This run adds the two controls requested after the v330 review:

random_gate:   RAMS architecture with random Gaussian mechanism-state input.
constant_gate: RAMS architecture with zero mechanism-state input.

The control keeps the matched frozen tPatchGNN base, residual architecture,
training schedule, and seeds fixed.  It tests whether RAMS-tPatch gains are
explained by deployment-visible mechanism state rather than generic residual
capacity.
"""

from __future__ import annotations

import sys

import run_BigData2026_v329_mechanism_controls as controls


def main() -> int:
    if "--root" not in sys.argv:
        sys.argv.extend(["--root", "storage/results_BigData2026_v331_random_constant_controls"])
    if "--prefix" not in sys.argv:
        sys.argv.extend(["--prefix", "v331"])
    if "--variants" not in sys.argv:
        sys.argv.extend(["--variants", "real_gate,random_gate,constant_gate"])
    if "--datasets" not in sys.argv:
        sys.argv.extend(["--datasets", "MHEALTH,OPPORTUNITY,PAMAP2"])
    if "--seeds" not in sys.argv:
        sys.argv.extend(["--seeds", "5301,5302,5303"])
    return controls.runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
