"""Public carrier map and Gram validation retained from original implementation."""
import hashlib
import numpy as np

def carrier_coordinates(carriers):
    """Ten public coordinates: sine/cosine of delays 1, 2, 4, 8, 16 samples.

    These resolve variation at several delay scales on the FFT64 frequency
    circle. They contain no received IQ, estimated channel or payload data.
    """
    k = np.asarray(carriers, float)
    if k.ndim != 1 or not len(k) or not np.isfinite(k).all():
        raise ValueError("Expected a finite, nonempty carrier vector")
    angles = 2 * np.pi * k[:, None] * np.array([1, 2, 4, 8, 16]) / 64
    return np.stack((np.sin(angles), np.cos(angles)), axis=-1)

def gram_diagnostics(matrix, *, normalized=False):
    """Reject indefinite/non-Hermitian Grams and report the unmodified spectrum.

    Roundoff below 1e-12 times the spectral scale is tolerated, not repaired.
    No diagonal jitter, eigenvalue clipping or inversion is used in training.
    """
    k = np.asarray(matrix, complex)
    if k.ndim != 2 or not len(k) or k.shape[0] != k.shape[1]:
        raise ValueError("Expected a nonempty square Gram")
    if not np.isfinite(k).all():
        raise ValueError("Nonfinite Gram")
    error = float(np.max(abs(k - k.conj().T)))
    if error > 1e-12 * max(1, float(np.max(abs(k)))):
        raise ValueError("Gram is not Hermitian")
    spectrum = np.linalg.eigvalsh((k + k.conj().T) / 2)
    limit = 1e-12 * max(1, float(spectrum[-1]))
    if spectrum[0] < -limit:
        raise ValueError(f"Indefinite Gram: minimum eigenvalue {spectrum[0]}")
    diagonal_error = float(np.max(abs(np.diag(k) - 1)))
    if normalized and diagonal_error > 2e-12:
        raise ValueError("Channel Gram must have unit diagonal")
    positive = spectrum[spectrum > limit]
    return dict(
        eigenvalues=spectrum.tolist(),
        min_eigenvalue=float(spectrum[0]),
        max_eigenvalue=float(spectrum[-1]),
        rank=int(len(positive)),
        rank_threshold=limit,
        participation_rank=float(spectrum.sum() ** 2 / (spectrum @ spectrum))
        if np.any(spectrum)
        else 0.0,
        positive_spectrum_condition=float(positive[-1] / positive[0])
        if len(positive)
        else None,
        full_condition_number=float(positive[-1] / positive[0])
        if len(positive) == len(k)
        else None,
        hermitian_error=error,
        unit_diagonal_error=diagonal_error,
        psd_roundoff_tolerance=limit,
        training_jitter=0.0,
        eigenvalues_clipped=False,
        gram_sha256=hashlib.sha256(np.ascontiguousarray(k, dtype="<c16").tobytes()).hexdigest(),
    )
