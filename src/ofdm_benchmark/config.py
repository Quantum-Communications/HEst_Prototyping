"""Small immutable configurations; unknown keys are rejected."""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PHY:
    """The audited custom FFT64/QPSK profile, not an IEEE 802.11 transmitter.

    Sample amplitudes retain GNU's unnormalized IFFT and QPSK energy of four.
    CP0 is supported only in mathematical tests, not the frozen RF receiver.
    """

    fft_length: int = 64
    cp_length: int = 16
    packet_length_bytes: int = 50
    packet_gap_samples: int = 80
    sample_rate: float = 8_000_000.0
    digital_scale: float = 0.03
    sync_threshold: float = 0.65
    gate_power: float = 0.001
    equalizer_alpha: float = 0.1

    def __post_init__(self):
        if not all(
            math.isfinite(v)
            for v in (
                self.sample_rate,
                self.digital_scale,
                self.sync_threshold,
                self.gate_power,
                self.equalizer_alpha,
            )
        ):
            raise ValueError("PHY values must be finite")
        if (self.fft_length, self.cp_length, self.packet_length_bytes) != (64, 16, 50):
            raise ValueError("The native profile requires FFT64, CP16 and 50-byte packets")
        if self.sample_rate <= 0 or self.digital_scale <= 0 or self.packet_gap_samples < 0:
            raise ValueError("Invalid waveform rate, scale or packet gap")
        if not 0 < self.sync_threshold <= 1 or self.gate_power < 0:
            raise ValueError("Invalid synchronization threshold or power gate")
        if not 0 <= self.equalizer_alpha <= 1:
            raise ValueError("DFE alpha must be in [0,1]")

    @property
    def occupied_carriers(self):
        from .phy import DATA_K

        return (tuple(map(int, DATA_K)),)

    @property
    def pilot_carriers(self):
        return ((-21, -7, 7, 21),)

    @property
    def pilot_symbols(self):
        from .phy import PILOT_POLARITY

        return tuple((p, p, p, -p) for p in PILOT_POLARITY)

    @property
    def sync_word1(self):
        from .phy import sync_words

        return tuple(sync_words()[0])

    @property
    def sync_word2(self):
        from .phy import sync_words

        return tuple(sync_words()[1])


@dataclass(frozen=True)
class Method:
    """Estimator name and explicit, JSON-serializable numerical options."""

    name: str = "ls"
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Noise:
    """Post-ADC impairment, relative to a measured activity-power proxy."""

    name: str = "natural"
    snr_db: float | None = None
    probability: float = 0.01
    variance_ratio: float = 50.0
    mean_bad_samples: float = 3.0

    def __post_init__(self):
        if not all(
            math.isfinite(v)
            for v in (self.probability, self.variance_ratio, self.mean_bad_samples)
        ) or (self.snr_db is not None and not math.isfinite(self.snr_db)):
            raise ValueError("Noise values must be finite")
        if self.name not in {"natural", "awgn", "bernoulli_gaussian", "markov_gaussian"}:
            raise ValueError("Unknown noise model")
        if (self.name == "natural") != (self.snr_db is None):
            raise ValueError("Only natural noise has no added-SNR setting")
        if not 0 < self.probability < 1 or self.variance_ratio < 1 or self.mean_bad_samples < 1:
            raise ValueError("Invalid mixture parameters")

    @property
    def label(self):
        return self.name if self.snr_db is None else f"{self.name}_{self.snr_db:g}dB"


@dataclass(frozen=True)
class Radio:
    """Two distinct HackRF devices; serials must be filled before capture."""

    tx_serial: str = "SET_TX_SERIAL"
    rx_serial: str = "SET_RX_SERIAL"
    center_frequency: float = 907_980_000.0
    bandwidth: float = 6_000_000.0
    tx_rf_gain: float = 14.0
    tx_if_gain: float = 47.0
    rx_rf_gain: float = 14.0
    rx_if_gain: float = 24.0
    rx_bb_gain: float = 12.0


@dataclass(frozen=True)
class Experiment:
    """Frozen experiment matrix and bounded post-acquisition resource budget."""

    phy: PHY = field(default_factory=PHY)
    radio: Radio = field(default_factory=Radio)
    methods: tuple[Method, ...] = field(default_factory=lambda: (Method(),))
    conditions: tuple[Noise, ...] = field(default_factory=lambda: (Noise(),))
    equalizers: tuple[str, ...] = ("static", "pilot_cpe", "simpledfe")
    workers: int = 2
    timeout_seconds: float = 120.0
    seed: int = 550010
    bursts: int = 2
    batch_size: int = 2
    measurement_packets: int = 100
    tx_warmup_seconds: float = 0.03
    rx_warmup_seconds: float = 0.20
    rx_tail_seconds: float = 0.20
    ram_root: str = "/dev/shm"
    required_gnu_version: str = "3.10.11.0"
    session_id: str = "SET_SESSION_ID"
    channel_description: str = "SET_PHYSICAL_PLACEMENT_AND_ENVIRONMENT"

    def __post_init__(self):
        if self.workers < 1 or not 1 <= self.batch_size <= self.bursts:
            raise ValueError("Invalid worker/burst/batch counts")
        if not 1 <= self.measurement_packets <= 1000 or self.timeout_seconds <= 0:
            raise ValueError("Invalid packet count or timeout")
        if min(self.tx_warmup_seconds, self.rx_warmup_seconds, self.rx_tail_seconds) < 0:
            raise ValueError("Negative capture interval")
        for values in (
            [m.name for m in self.methods],
            [c.label for c in self.conditions],
            self.equalizers,
        ):
            if not values or len(values) != len(set(values)):
                raise ValueError("Empty or duplicated experiment cell")
        if not set(self.equalizers) <= {"static", "pilot_cpe", "simpledfe"}:
            raise ValueError("Unknown equalizer")
        if "ls" not in {m.name for m in self.methods}:
            raise ValueError("The LS reference is required")

    def to_dict(self):
        return asdict(self)


def from_dict(value: dict) -> Experiment:
    """Load a record without implicit fallback or version-specific imports."""
    data = dict(value)
    data["phy"] = PHY(**data.get("phy", {}))
    data["radio"] = Radio(**data.get("radio", {}))
    data["methods"] = tuple(Method(**m) for m in data.get("methods", [{"name": "ls"}]))
    data["conditions"] = tuple(
        Noise(**n) for n in data.get("conditions", [{"name": "natural"}])
    )
    if "equalizers" in data:
        data["equalizers"] = tuple(data["equalizers"])
    return Experiment(**data)


def load(path: str | Path) -> Experiment:
    """Resolve o Gram relativo ao arquivo de configuração antes de iniciar os processos."""
    path = Path(path).resolve()
    if path.suffix == ".json":
        data = json.loads(path.read_text())
    else:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    for method in data.get("methods", []):
        params = method.get("parameters", {})
        gram = params.get("quantum_gram_path")
        if gram and not Path(gram).is_absolute():
            candidates = (path.parent / gram, path.parent.parent / gram, Path.cwd() / gram)
            params["quantum_gram_path"] = str(next((p for p in candidates if p.is_file()),
                                                   candidates[0]).resolve())
    return from_dict(data)
