"""Software and estimator checks before spending a physical RF acquisition."""

import importlib.metadata
import os
import platform
import re
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .estimators import Estimator
from .io import sha256
from .phy import sync_words


def check(cfg, *, hardware=False):
    """Validate actual imports and a finite public-preamble fit, without SDR access."""
    from gnuradio import gr

    if gr.version() != cfg.required_gnu_version:
        raise RuntimeError("GNU runtime differs from the declared experiment")
    if hardware:
        import osmosdr  # noqa: F401

        serials = (cfg.radio.tx_serial, cfg.radio.rx_serial)
        if len(set(serials)) != 2 or not all(
            re.fullmatch("[0-9a-fA-F]{32}", s) for s in serials
        ):
            raise ValueError("Declare two distinct full HackRF serials")
        if any(
            s.startswith("SET_") or not s.strip()
            for s in (cfg.session_id, cfg.channel_description)
        ):
            raise ValueError("Declare the physical session and placement before capture")
    first, second = sync_words()
    rng = np.random.default_rng(20261105)
    raw = np.stack([first, second, np.zeros(64)]).astype("c8")
    raw[:2] += 0.01 * (rng.normal(size=(2, 64)) + 1j * rng.normal(size=(2, 64)))
    checked = []
    for method in cfg.methods:
        estimator = Estimator(method, cfg.phy)
        taps, _ = estimator.estimate(raw)
        if not np.isfinite(taps).all():
            raise RuntimeError("Nonfinite preflight model")
        checked.append(method.name)
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "qiskit"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    source = Path(__file__).parent
    return dict(
        project_version=__version__,
        gnuradio_version=gr.version(),
        python=sys.version,
        platform=platform.platform(),
        machine=platform.machine(),
        logical_cpu_count=os.cpu_count(),
        parent_thread_environment={
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        worker_thread_policy="Two BLAS/OpenMP threads per isolated receiver",
        versions=versions,
        methods_checked=checked,
        radios_opened=False,
        source_sha256={
            str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob("*.py"))
        },
        scope="Import/finite-solution check; not RF performance or optimality validation",
    )
