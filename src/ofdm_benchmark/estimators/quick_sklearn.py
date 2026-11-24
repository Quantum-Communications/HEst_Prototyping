"""Final direct-Y SVR and finite-shot CX QSVR (only TCC variants)."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from ..phy import ACTIVE, ACTIVE_K, integer_cfo
from .fidelity import carrier_coordinates, gram_diagnostics
QUICK_METHODS = ("svr_skdirect_rbf", "qsvr_skdirect_cx_shots")
QUICK_PARAMS = {"c", "epsilon", "quantum_gram_path"}
DELAYS = (0.5, 1, 2, 4, 8)

def physical_circuit(carrier, variant):
    """Carrier k in 64-tone OFDM: physical delay phases with CX entanglement.

    |+>^5 with P(-2*pi*k*t_j/64), t_j in DELAYS, encodes a superposition of 32
    half-sample delay hypotheses (0..15.5 samples), compatible with CP16. The subsequent RY, CX, RX and RZ layers
    give a *data-dependent* noncommuting reupload; a final, fixed entangler
    alone would leave fidelity invariant and is explicitly avoided.

    All circuits have 5 qubits; no received IQ enters feature-map angles.
    """
    from qiskit import QuantumCircuit

    if variant != "cx":
        raise ValueError(f"Unsupported physical circuit: {variant}")
    k = float(carrier)
    phi = -2.0 * np.pi * k * np.asarray(DELAYS, float) / 64.0
    qc = QuantumCircuit(len(DELAYS), name=f"delay_{variant}")
    qc.h(range(len(DELAYS)))
    for bit, angle in enumerate(phi):
        qc.p(float(angle), bit)
    # A rotação RY depende da fase física da portadora e precede a cadeia de quatro CX.
    for bit, angle in enumerate(phi):
        qc.ry(float(0.50 * np.sin(angle)), bit)
    for bit in range(4):
        qc.cx(bit, bit + 1)
    # Noncommuting reupload ensures that entangling interactions can affect
    # pairwise fidelity as opposed to being just a removable common unitary.
    for bit, angle in enumerate(phi):
        qc.rx(float(0.35 * np.cos(angle)), bit)
        qc.rz(float(0.25 * angle), bit)
    return qc

def _digest_float64(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()

def load_custom_gram(path, variant, *, shots=4000):
    """Carrega K na ordem física de k e verifica o digest dos float64 antes do treino."""
    path = Path(path).resolve(strict=True)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"gram", "carriers", "metadata"}:
            raise ValueError("Unexpected custom Gram NPZ schema")
        g = np.asarray(archive["gram"], float)
        carriers = np.asarray(archive["carriers"], int)
        metadata = json.loads(str(archive["metadata"].item()))
    required = {"family": "ofdm_carrier_physical_v1", "variant": variant,
                "shots": shots, "backend": "Qiskit BasicSimulator shot counts",
                "read_statevector_for_training": False, "hardware_execution": False}
    if any(metadata.get(k) != v for k, v in required.items()):
        raise ValueError(f"Custom measured Gram provenance mismatch for {variant}")
    if not np.array_equal(carriers, ACTIVE_K) or g.shape != (52, 52):
        raise ValueError("Custom measured Gram has wrong carrier ordering")
    if _digest_float64(g) != metadata.get("gram_sha256"):
        raise ValueError("Custom measured Gram SHA256 mismatch")
    gram_diagnostics(g, normalized=True)
    return g, metadata

class QuickDirectEstimator:
    """Two real epsilon-insensitive regressions of the received preamble."""

    def __init__(self, name, parameters, sync2):
        if name not in QUICK_METHODS or set(parameters) - QUICK_PARAMS:
            raise ValueError("Unsupported sklearn quick method/parameter")
        self.name = name
        self.p = dict(parameters)
        self.x = np.asarray(sync2, complex)[ACTIVE]
        if not (np.allclose(self.x.imag, 0, atol=1e-7)
                and np.allclose(np.abs(self.x), 1, atol=1e-7)):
            raise ValueError("The direct-Y sklearn pilot requires a unit BPSK preamble")
        self.x = self.x.real.astype(float)
        self.kind = "sklearn"
        self.variant = self._variant(name)
        self.shots = "shots" in name
        if self.shots and "quantum_gram_path" not in self.p:
            raise ValueError("Shot-based kernel requires a frozen quantum_gram_path")
        if not self.shots and "quantum_gram_path" in self.p:
            raise ValueError("Exact/classical method cannot take a measured Gram")
        t0 = time.perf_counter()
        if self.variant == "rbf":
            features = carrier_coordinates(ACTIVE_K).reshape(52, -1)
            sqdist = np.sum((features[:, None] - features[None, :])**2, axis=-1)
            kernel = np.exp(-0.125 * sqdist)
            provenance = {"family": "classical_rbf", "gamma": 0.125}
        else:
            kernel, provenance = load_custom_gram(self.p["quantum_gram_path"], "cx", shots=4000)
        self.k = np.asarray(kernel, float)
        gram_diagnostics(self.k, normalized=True)
        # G = D_X K D_X incorpora os sinais BPSK do segundo preâmbulo.
        self.g = self.x[:, None] * self.k * self.x[None, :]
        gram_diagnostics(self.g, normalized=True)
        self.kernel_seconds = time.perf_counter() - t0
        self.provenance = provenance
        if float(self.p.get("c", 9.615)) <= 0 or float(self.p.get("epsilon", .02)) < 0:
            raise ValueError("Invalid sklearn C/epsilon")

    @staticmethod
    def _variant(name):
        return "rbf" if name == "svr_skdirect_rbf" else "cx"

    def estimate(self, raw, first, second):
        """Recebe três FFT64 centradas e usa o segundo preâmbulo nos 52 tons ativos.

        O LIBSVM resolve o dual epsilon-SVR com intercepto, separadamente para I e Q.
        Cada predict usa os coeficientes duais dos vetores de suporte e o intercepto.
        A saída do receptor contém 64 ganhos complexos e o CFO inteiro em bins.
        """
        offset = integer_cfo(raw, first, second)
        y = np.asarray(raw, np.complex64)[1, ACTIVE + offset].astype(complex)
        scale = max(float(np.sqrt(np.mean(np.abs(y)**2))), 1e-12)
        # Alvo dos dois ajustes: partes real e imaginária de Y normalizado por seu RMS.
        target = y / scale
        t0 = time.perf_counter()
        from sklearn.svm import SVR
        predictions = []
        fits = []
        for target_real in (target.real, target.imag):
            mdl = SVR(kernel="precomputed", C=float(self.p.get("c", 9.615)),
                      epsilon=float(self.p.get("epsilon", .02)))
            mdl.fit(self.g, target_real)
            predictions.append(mdl.predict(self.g))
            fits.append(dict(solver="sklearn.svm.SVR_precomputed",
                             n_support_vectors=int(len(mdl.support_)),
                             support_indices=mdl.support_.tolist(),
                             c=float(self.p.get("c",9.615)),
                             epsilon=float(self.p.get("epsilon",.02))))
        y_hat = predictions[0] + 1j * predictions[1]
        train_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        h = np.zeros(64, dtype=complex)
        # Para X = ±1, dividir a previsão por X equivale a multiplicar por X.
        # O canal nas portadoras inativas permanece zero.
        h[ACTIVE] = scale * self.x * y_hat
        predict_sec = time.perf_counter() - t1
        details = dict(
            fits=fits, kernel=dict(variant=self.variant, kind=self.kind,
                                   finite_shots=self.shots, provenance=self.provenance),
            observation_scale=scale, training_seconds=train_sec,
            prediction_seconds=predict_sec,
            normalized_training_residual=float(np.mean(abs(target-y_hat)**2)),
            phase_reference_count=0, fit_reference_count=len(ACTIVE),
            pointwise_ls_equivalent_targets=False,
            preceding_channel_estimator=None, explicit_phase_correction=False,
            payload_truth_used=False, received_iq_in_encoding=False)
        return h.astype(np.complex64), offset, details
