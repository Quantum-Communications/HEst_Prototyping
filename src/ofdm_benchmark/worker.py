"""Processo isolado do GNU Radio para aquisição, sincronização ou recepção."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from .config import PHY, Method, from_dict
from .io import read_json, write_json


def execute(job):
    """Dispatch an explicit job without importing a scorer into reception."""
    from gnuradio import gr

    if gr.version() != job["required_gnu_version"]:
        raise RuntimeError(
            f"GNU version mismatch: {gr.version()} != {job['required_gnu_version']}"
        )
    phy = PHY(**job["phy"])
    if job["action"] == "frontend":
        from .frontend import run

        return run(job["iq"], job["output"], phy, job["timeout"])
    if job["action"] == "receiver":
        from .frontend import verify
        from .native import run

        verify(job["frontend"])
        method = "gnu" if job["method"] == "gnu" else Method(**job["method"])
        return run(
            phy, job["frontend"], job["output"], method, job["equalizer"], job["timeout"]
        )
    if job["action"] == "capture":
        from .acquisition import capture_one

        return capture_one(from_dict(job["experiment"]), Path(job["output"]), job["burst"])
    raise ValueError("Unknown job action")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--acquire", action="store_true")
    args = parser.parse_args()
    job = read_json(args.job)
    if job["action"] == "capture" and not args.acquire:
        parser.error("a aquisição física requer --acquire")
    try:
        execute(job)
    except Exception as error:
        traceback.print_exc()
        write_json(
            args.job.with_suffix(".failure.json"), dict(error=repr(error), pid=os.getpid())
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
