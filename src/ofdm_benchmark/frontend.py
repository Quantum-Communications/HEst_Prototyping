"""Shared GNU Schmidl-Cox frontend, power-gated before plateau detection.

The sign is negative because sync1 occupies odd unshifted subcarriers.
All methods consume the same immutable corrected IQ and trigger stream.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .io import sha256, write_json
from .monitor import wait_graph_monitored


def gated_sync(phy):
    """Reproduce GNU's SC metric, adding the previously audited energy gate."""
    from gnuradio import blocks, filter, gr

    class GatedSync(gr.hier_block2):
        def __init__(self):
            super().__init__(
                "power_gated_schmidl_cox",
                gr.io_signature(1, 1, gr.sizeof_gr_complex),
                gr.io_signaturev(3, 3, [gr.sizeof_float, gr.sizeof_char, gr.sizeof_float]),
            )
            delay = blocks.delay(gr.sizeof_gr_complex, 32)
            conjugate = blocks.conjugate_cc()
            product = blocks.multiply_cc()
            # O primeiro símbolo ocupa tons ímpares: suas metades têm sinais opostos.
            correlation = filter.fir_filter_ccf(1, [-1.0] * 32)
            numerator = blocks.complex_to_mag_squared()
            power = blocks.complex_to_mag_squared()
            # R usa as duas metades. A porta de potência atua antes do detector de platô.
            energy = filter.fir_filter_fff(1, [0.5] * 64)
            denominator = blocks.multiply_ff()
            floor = blocks.add_const_ff(1e-20)
            divide = blocks.divide_ff()
            threshold = -1.0 if phy.gate_power == 0 else 32 * phy.gate_power
            gate = blocks.threshold_ff(threshold, threshold, 0.0)
            metric = blocks.multiply_ff()
            plateau = blocks.plateau_detector_fb(phy.cp_length, phy.sync_threshold)
            angle = blocks.complex_to_arg()
            hold = blocks.sample_and_hold_ff()
            self.connect(self, delay, conjugate, (product, 1))
            self.connect(self, (product, 0))
            self.connect(product, correlation, numerator, (divide, 0))
            self.connect(self, power, energy, (denominator, 0))
            self.connect(energy, (denominator, 1))
            self.connect(denominator, floor, (divide, 1))
            self.connect(divide, (metric, 0))
            self.connect(energy, gate, (metric, 1))
            self.connect(metric, plateau, (hold, 1))
            self.connect(correlation, angle, (hold, 0))
            self.connect(hold, (self, 0))
            self.connect(plateau, (self, 1))
            self.connect(energy, (self, 2))

    return GatedSync()


def run(iq_path, output, phy, timeout=120):
    """Create and seal paired complex64 IQ and uint8 triggers; no TX information."""
    from gnuradio import analog, blocks, gr

    output, iq_path = Path(output), Path(iq_path)
    output.mkdir(parents=True, exist_ok=False)
    if iq_path.stat().st_size % 8:
        raise ValueError("Malformed complex64 input")
    n = iq_path.stat().st_size // 8
    if not n:
        raise ValueError("Empty IQ input")
    flush_samples = 4096
    paths = (
        output / "corrected.c64",
        output / "triggers.u8",
        output / "phase.f32",
        output / "energy.f32",
    )

    class Sink(gr.sync_block):
        def __init__(self):
            super().__init__(
                "paired_frontend_sink",
                in_sig=[np.complex64, np.uint8, np.float32, np.float32],
                out_sig=None,
            )
            self.handles = [p.open("xb") for p in paths]
            self.samples = 0
            self.set_max_noutput_items(8192)

        def work(self, inputs, outputs):
            consumed = min(map(len, inputs))
            count = min(consumed, max(0, n - self.samples))
            for data, handle, dtype in zip(
                inputs, self.handles, ("<c8", "u1", "<f4", "<f4"), strict=True
            ):
                np.asarray(data[:count], dtype=dtype).tofile(handle)
            self.samples += count
            return consumed

        def close(self):
            for handle in self.handles:
                handle.close()

    original = blocks.file_source(gr.sizeof_gr_complex, str(iq_path), False)
    extension = blocks.vector_source_c([0j] * flush_samples, False)
    source = blocks.stream_mux(gr.sizeof_gr_complex, (n, flush_samples))
    sync = gated_sync(phy)
    delay = blocks.delay(gr.sizeof_gr_complex, 64 + phy.cp_length)
    oscillator = analog.frequency_modulator_fc(-2.0 / 64)
    multiply = blocks.multiply_vcc(1)
    sink = Sink()
    graph = gr.top_block("shared_ofdm_frontend")
    # Plateau detection requires future samples; scheduler EOF can otherwise
    # strand a short suffix. Extend only the working stream, and save exactly
    # the original sample interval. No extension enters receiver/scorer inputs.
    graph.connect(original, (source, 0))
    graph.connect(extension, (source, 1))
    graph.connect(source, sync)
    graph.connect(source, delay, (multiply, 1))
    graph.connect((sync, 0), oscillator, (multiply, 0))
    graph.connect(multiply, (sink, 0))
    graph.connect((sync, 1), (sink, 1))
    graph.connect((sync, 0), (sink, 2))
    graph.connect((sync, 2), (sink, 3))
    started = time.perf_counter()
    try:
        completed, _ = wait_graph_monitored(graph, timeout, lambda: None)
    finally:
        graph.stop()
        graph.wait()
        sink.close()
        graph.disconnect_all()
    if (
        not completed
        or original.nitems_written(0) != n
        or sink.samples != n
        or any(
            p.stat().st_size != n * size for p, size in zip(paths, (8, 1, 4, 4), strict=True)
        )
    ):
        raise RuntimeError(
            f"Incomplete or unpaired frontend stream: input={n}, sink={sink.samples}, sizes={[p.stat().st_size for p in paths]}, completed={completed}"
        )
    triggers = np.memmap(paths[1], dtype="u1", mode="r")
    positions = np.flatnonzero(triggers)
    metadata = dict(
        samples=n,
        eof_extension_samples=flush_samples,
        eof_policy="Zero-extend working stream for scheduler lookahead; retain original interval only",
        input_sha256=sha256(iq_path),
        hashes={p.name: sha256(p) for p in paths},
        raw_triggers=len(positions),
        trigger_positions=positions.tolist(),
        seconds=time.perf_counter() - started,
        gnuradio_version=gr.version(),
        gate_power=phy.gate_power,
        threshold=phy.sync_threshold,
        denominator_floor=1e-20,
    )
    write_json(output / "frontend.json", metadata)
    for path in paths:
        path.chmod(0o444)
    return metadata


def verify(folder):
    """Reject any mutation before or after processing a shared frontend."""
    from .io import read_json

    folder = Path(folder)
    record = read_json(folder / "frontend.json")
    if any(sha256(folder / name) != digest for name, digest in record["hashes"].items()):
        raise RuntimeError("Shared frontend content was changed")
    return record
