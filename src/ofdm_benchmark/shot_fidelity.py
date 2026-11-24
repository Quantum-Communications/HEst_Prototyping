import numpy as np
from .estimators.fidelity import gram_diagnostics

def project_correlation_psd(raw, *, policy="clip", max_relative_correction=0.20):
    """Spectral projection + unit-diagonal congruence, explicitly audited.

    No quiet jitter, and no feature/truth-driven choice of projection settings.
    If a measured kernel is indefinite and policy='reject', refuse training.
    """
    raw = np.asarray(raw, float)
    if raw.ndim != 2 or len(raw) < 2 or raw.shape[1] != len(raw):
        raise ValueError("Expected square real Gram")
    if not np.isfinite(raw).all() or not np.allclose(raw, raw.T, atol=1e-13):
        raise ValueError("Measured Gram must be finite symmetric")
    if not np.allclose(np.diag(raw), 1.0, atol=1e-13):
        raise ValueError("Measured Gram must have unit diagonal")
    if policy not in {"clip", "reject"}:
        raise ValueError("Unknown PSD policy")
    spectrum, vec = np.linalg.eigh(raw)
    min_before = float(spectrum[0])
    repaired = bool(min_before < -1e-12 * max(1, float(spectrum[-1])))
    if repaired:
        if policy == "reject":
            raise ValueError(f"Finite-shot Gram indefinite: min={min_before}")
        clipped = (vec * np.maximum(spectrum, 0)) @ vec.T
        diag = np.diag(clipped)
        if np.any(diag <= 0):
            raise ValueError("Cannot normalize PSD projection")
        repaired_gram = clipped / np.sqrt(diag[:, None] * diag[None, :])
        repaired_gram = (repaired_gram + repaired_gram.T) / 2
        np.fill_diagonal(repaired_gram, 1.0)
    else:
        repaired_gram = raw.copy()
    relative = float(np.linalg.norm(repaired_gram - raw, 'fro') /
                     max(np.linalg.norm(raw, 'fro'), 1e-15))
    if relative > max_relative_correction:
        raise ValueError(f"Excessive PSD repair {relative:.4g}")
    check = gram_diagnostics(repaired_gram, normalized=True)
    return repaired_gram, dict(
        psd_policy=policy, psd_repaired=repaired,
        min_eigenvalue_raw=min_before,
        min_eigenvalue_used=check['min_eigenvalue'],
        frobenius_relative_correction=relative,
        max_relative_correction=float(max_relative_correction),
        eigenvalues_clipped=bool(repaired),
    )
