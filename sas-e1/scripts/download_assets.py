#!/usr/bin/env python
"""Download models + prepare datasets for SAS-E1 (idempotent)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sas_e1.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["--stage", "download", *sys.argv[1:]])
