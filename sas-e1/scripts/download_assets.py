#!/usr/bin/env python
"""Download models + prepare datasets for SAS-E1 (idempotent)."""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sas_e1.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["--stage", "download", *sys.argv[1:]])
    # Background download threads may outlive main(); exit without interpreter
    # finalization races (all work is already flushed by the time we get here).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
