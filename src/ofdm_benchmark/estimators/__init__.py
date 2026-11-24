"""Four estimators evaluated in the TCC; native GNU LS is a parity control."""
from __future__ import annotations
import time
import numpy as np
from ..config import PHY, Method
from ..phy import ACTIVE, ACTIVE_K, integer_cfo, ls_taps
from .quick_sklearn import QUICK_METHODS, QuickDirectEstimator
METHODS = ("ls", "lmmse", *QUICK_METHODS)

def fourier_design(length):
    """Sample-spaced delay dictionary; support is explicit and can include delay 16."""
    if not 1 <= length <= len(ACTIVE):
        raise ValueError("FIR support must be between 1 and the reference count")
    return np.exp(-2j * np.pi * ACTIVE_K[:, None] * np.arange(length) / 64)

class _ReferenceEstimator:
    """One independent estimator state per receiver run.

    estimate(raw) -> (FFT-shifted channel[64], integer CFO). failures are fatal
    to the cell and remain visible in the campaign ledger. All setup, kernel,
    prediction and fitting work is timed; acquisition time is separate.
    """

    def __init__(self, method: Method, phy: PHY):
        started = time.perf_counter()
        if method.name not in METHODS:
            raise ValueError(f"Unknown estimator: {method.name}")
        self.method, self.phy = method, phy
        self.s1 = np.asarray(phy.sync_word1, np.complex64)
        self.s2 = np.asarray(phy.sync_word2, np.complex64)
        self.calls = self.failed = 0
        self.fit_seconds = 0.0
        self.history = []
        self.params = dict(method.parameters)
        name = method.name
        allowed = {"ls": set(), "lmmse": {"length", "decay", "noise_relative", "rank"}}
        if set(self.params) - allowed[name]:
            raise ValueError(f"Unknown parameters for {name}: {self.params}")
        self.backend_import_seconds = 0.0
        self.length = int(self.params.get("length", 16))
        self.f = fourier_design(self.length)
        if name == "lmmse":
            decay = float(self.params.get("decay", 0.7))
            noise = float(self.params.get("noise_relative", 0.03))
            if not 0 < decay <= 1 or noise <= 0:
                raise ValueError("Invalid covariance or noise prior")
            power = decay ** np.arange(self.length)
            power /= power.sum()
            cross_f = np.exp(
                -2j * np.pi * (np.arange(64) - 32)[:, None] * np.arange(self.length) / 64
            )
            self.cov = (self.f * power) @ self.f.conj().T
            self.cross_cov = (cross_f * power) @ self.f.conj().T
            self.values, self.vectors = np.linalg.eigh(self.cov)
            self.values = np.maximum(self.values, 0)
        self.kernel_seconds = 0.0
        self.initialization_seconds = time.perf_counter() - started

    def _estimate(self, raw):
        """Estima H com o preâmbulo conhecido. No LMMSE, a base própria implementa
        a aplicação de R_ha (R_aa + λI)^−1 sem inverter uma matriz a cada quadro.
        """
        name, p = self.method.name, self.params
        if name == "ls":
            return (*ls_taps(raw, self.s1, self.s2), {})
        offset = integer_cfo(raw, self.s1, self.s2)
        y = np.asarray(raw, np.complex64)[1, ACTIVE + offset].astype(np.complex128)
        if name == "lmmse":
            # H_LS nas portadoras conhecidas alimenta o filtro de covariância do LMMSE.
            observed = y / self.s2[ACTIVE]
            sigma = float(p.get("noise_relative", 0.03))
            weights = (self.vectors.conj().T @ observed) / (self.values + sigma)
            rank = int(p.get("rank", len(ACTIVE)))
            if not 1 <= rank <= len(ACTIVE):
                raise ValueError("Invalid covariance rank")
            weights[: len(ACTIVE) - rank] = 0
            h = self.cross_cov @ (self.vectors @ weights)
            details = {"assumed_relative_noise": sigma, "rank": rank, "unit_covariance": True}
        return np.asarray(h, np.complex64), offset, details

    def estimate(self, raw):
        started = time.perf_counter()
        try:
            h, offset, details = self._estimate(raw)
            if h.shape != (64,) or not np.isfinite(h).all():
                raise RuntimeError("Nonfinite or malformed channel estimate")
            self.calls += 1
            self.history.append(details)
            return h, offset
        except Exception:
            self.failed += 1
            raise
        finally:
            self.fit_seconds += time.perf_counter() - started

    def diagnostics(self):
        separated = bool(self.history) and all(
            "training_seconds" in item for item in self.history
        )
        return dict(
            method=self.method.name,
            parameters=self.params,
            fit_calls=self.calls,
            fit_failures=self.failed,
            initialization_seconds=self.initialization_seconds,
            kernel_seconds=self.kernel_seconds,
            backend_import_seconds=self.backend_import_seconds,
            fit_seconds=self.fit_seconds,
            training_seconds=(
                sum(d["training_seconds"] for d in self.history) if separated else None
            ),
            prediction_seconds=(
                sum(d["prediction_seconds"] for d in self.history) if separated else None
            ),
            fits=self.history,
            scope="Public preamble only; classical/reference estimator",
        )

class Estimator(_ReferenceEstimator):
    """Despacha os quatro métodos mantendo o mesmo contrato de canal e CFO."""

    def __init__(self, method, phy):
        if method.name not in QUICK_METHODS:
            super().__init__(method, phy)
            return
        # Initialize shared receiver state through the stable LS branch;
        # this never fits LS or accesses payload/channel estimates.
        started = time.perf_counter()
        super().__init__(Method("ls", {}), phy)
        self.method = method
        self.params = dict(method.parameters)
        self.quick_model = QuickDirectEstimator(method.name, self.params, self.s2)
        self.kernel_seconds = self.quick_model.kernel_seconds
        self.initialization_seconds = time.perf_counter() - started

    def _estimate(self, raw):
        if self.method.name in QUICK_METHODS:
            return self.quick_model.estimate(raw, self.s1, self.s2)
        return super()._estimate(raw)

    def diagnostics(self):
        result = super().diagnostics()
        if self.method.name in QUICK_METHODS:
            model = self.quick_model
            if model.variant == "rbf":
                result["scope"] = "Public preamble; classical RBF sklearn baseline"
            else:
                result["scope"] = (
                    "Public preamble; Qiskit simulated finite-shot fidelity; "
                    "no physical quantum hardware or gate/readout noise"
                )
        return result
