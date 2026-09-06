#!/usr/bin/env python3
"""Entrypoint for the robust original-2D-bias arclength workflow.

Use --help for the new arguments. In particular, supply the original 2D
--kappa-dir and an ordered two-column --path-reference. A scalar projected
kappa and the former arbitrary first-1.5-Å reactant basin are not supported.
This file can also live beside GPR-Umbrella-Sampling-Analysis on DAIS.
"""
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
for source in (here.parent / "src", here / "GPR-Umbrella-Sampling-Analysis" / "src"):
    if (source / "gpr_umbrella").is_dir():
        sys.path.insert(0, str(source))
        break

from gpr_umbrella.cli_path import main


if __name__ == "__main__":
    main()
