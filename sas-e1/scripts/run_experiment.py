#!/usr/bin/env python
"""Run SAS-E1 stages (resumable): extract | fit | eval | probes | analyze | all."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sas_e1.cli import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
