"""Pure OFDM geometry and reference operations in FFT-shifted order."""

import numpy as np

PILOT_POLARITY = (
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    1,
    -1,
    1,
    1,
    1,
    1,
    1,
    1,
    -1,
    1,
    1,
    1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
)
PILOT_K = np.array([-21, -7, 7, 21])
ACTIVE_K = np.r_[np.arange(-26, 0), np.arange(1, 27)]
DATA_K = np.array([k for k in ACTIVE_K if k not in PILOT_K])
ACTIVE = ACTIVE_K + 32
DATA = DATA_K + 32
PILOTS = PILOT_K + 32
QPSK = np.sqrt(2.0) * np.array([-1 - 1j, 1 - 1j, -1 + 1j, 1 + 1j])


def sync_words():
    """Return complex64 public sync vectors, identical to the seed-42 GNU profile."""
    active = set(ACTIVE_K % 64)
    words = []
    for sparse in (True, False):
        rng = np.random.RandomState(42)
        x = [
            (np.sqrt(2) if sparse else 1.0) * (1 if rng.randint(2) == 0 else -1)
            if k in active and (not sparse or k % 2)
            else 0.0
            for k in range(64)
        ]
        words.append(np.fft.fftshift(x).astype(np.complex64))
    return tuple(words)


def modulate(vectors, cp=16, scale=1.0):
    """Unnormalized inverse FFT and exact cyclic extension, including CP0."""
    x = np.asarray(vectors)
    if x.shape[-1] != 64 or not 0 <= cp < 64:
        raise ValueError("Expected FFT64 and a valid CP")
    time = 64 * np.fft.ifft(np.fft.ifftshift(x, axes=-1), axis=-1)
    extended = np.concatenate((time[..., -cp:], time), axis=-1) if cp else time
    return (scale * extended).astype(np.complex64)


def demodulate(symbols, cp=16):
    """Forward FFT after CP removal; the GNU IFFT scale is deliberately retained."""
    x = np.asarray(symbols)
    if x.shape[-1] != 64 + cp:
        raise ValueError("Incorrect symbol length")
    return np.fft.fftshift(np.fft.fft(x[..., cp:], axis=-1), axes=-1)


def classes_to_bytes(classes):
    """Pack four QPSK mapper indices per byte, least-significant pair first."""
    x = np.asarray(classes, dtype=np.uint8)
    if x.size % 4 or np.any(x > 3):
        raise ValueError("Invalid QPSK indices")
    return (
        np.sum(x.reshape(-1, 4).astype(np.uint16) << (2 * np.arange(4)), axis=1)
        .astype("u1")
        .tobytes()
    )


def bytes_to_classes(data):
    """Inverse of classes_to_bytes, independent of GNU Radio."""
    x = np.frombuffer(data, dtype=np.uint8)
    return ((x[:, None] >> (2 * np.arange(4))) & 3).astype(np.uint8).ravel()


def integer_cfo(raw, first, second):
    """Match GNU's complex<float> sequential correlation and strict tie ordering."""
    a = np.asarray(raw, np.complex64)
    s1, s2 = np.asarray(first, np.complex64), np.asarray(second, np.complex64)
    if a.shape != (3, 64) or not np.isfinite(a).all():
        raise ValueError("Expected three finite FFT64 vectors")
    k = np.flatnonzero(s1)
    corr = s2[k] / s1[k]
    best, offset = 0.0, 0
    for g in (-2, 0, 2):
        acc = np.complex64(0)
        for term in a[0, k + g].conj() * corr.conj() * a[1, k + g]:
            acc = np.complex64(acc + term)
        magnitude = float(abs(acc))
        if magnitude > best:
            best, offset = magnitude, g
    return offset


def ls_taps(raw, first, second):
    """Public-preamble LS in GNU float precision, with zeros on inactive tones."""
    offset = integer_cfo(raw, first, second)
    h = np.zeros(64, np.complex64)
    k = np.flatnonzero(second)
    h[k] = np.asarray(raw, np.complex64)[1, k + offset] / np.asarray(second, np.complex64)[k]
    return h, offset


def pilot_phase(observed, predicted):
    """Weighted least-squares common phase; return zero for an unobservable phase."""
    a, b = np.asarray(observed), np.asarray(predicted)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite or incompatible pilots")
    correlation = np.vdot(b, a)
    return float(np.angle(correlation)) if abs(correlation) > 1e-20 else 0.0
