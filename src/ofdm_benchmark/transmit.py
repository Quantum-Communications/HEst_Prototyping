"""Finite native GNU OFDM waveform generation, separate from radio capture."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from .io import sha256, write_json
from .monitor import wait_graph_monitored


def protocol(phy, packets, warmup_seconds):
    """Bound header sequence IDs and declare warm-up independently of reception."""
    payload_symbols = math.ceil(phy.packet_length_bytes * 4 / 48)
    samples = (3 + payload_symbols) * (64 + phy.cp_length) + phy.packet_gap_samples
    warmup = math.ceil(warmup_seconds * phy.sample_rate / samples)
    total = warmup + packets
    if not 0 < total < 4096:
        raise ValueError("Burst would wrap the public 12-bit header sequence")
    return dict(
        warmup_packets=warmup,
        measurement_packets=packets,
        total_packets=total,
        frame_samples=samples,
        nominal_burst_seconds=total * samples / phy.sample_rate,
    )


def waveform_limits(path):
    """Measure I/Q component range before the HackRF float-to-int8 conversion."""
    components = np.memmap(path, dtype="<f4", mode="r")
    return dict(
        peak_component=float(np.max(abs(components))),
        component_fraction_above_one=float(np.mean(abs(components) > 1.0)),
        component_count=int(components.size),
        scope="Digital range before driver conversion; not RF amplifier linearity",
    )


def generate(folder, phy, packets=100, warmup_seconds=0.03, seed=1):
    """Save the exact GNU-generated DAC input and its header/payload references.

    A receiver never receives the output truth path. Known header sequence IDs
    come from the actual formatter output, rather than from payload identifiers.
    """
    import pmt
    from gnuradio import blocks, digital, fft, gr

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    plan = protocol(phy, packets, warmup_seconds)
    payloads = []
    for count, start, chosen_seed in (
        (plan["warmup_packets"], 2**31, seed ^ 0x5A17),
        (packets, 0, seed),
    ):
        rng = random.Random(chosen_seed)
        for index in range(count):
            payloads.append(
                b"OFDM"
                + (start + index).to_bytes(4, "little")
                + bytes(rng.getrandbits(8) for _ in range(phy.packet_length_bytes - 8))
            )
    payload_path = folder / "tx.bin"
    payload_path.write_bytes(b"".join(payloads))
    source = blocks.file_source(gr.sizeof_char, str(payload_path), False)
    source.set_begin_tag(pmt.PMT_NIL)
    tagged = blocks.stream_to_tagged_stream(
        gr.sizeof_char, 1, phy.packet_length_bytes, "packet_length"
    )
    header_book, payload_book = digital.constellation_bpsk(), digital.constellation_qpsk()
    fmt = digital.packet_header_ofdm(
        phy.occupied_carriers, 1, "packet_length", "frame_length", "packet_num", 1, 2, False
    )
    header = digital.packet_headergenerator_bb(fmt.base(), "packet_length")
    repack = blocks.repack_bits_bb(8, 2, "packet_length", False, gr.GR_LSB_FIRST)
    mux = blocks.tagged_stream_mux(gr.sizeof_gr_complex, "packet_length", 0)
    allocator = digital.ofdm_carrier_allocator_cvc(
        64,
        phy.occupied_carriers,
        phy.pilot_carriers,
        phy.pilot_symbols,
        (phy.sync_word1, phy.sync_word2),
        "packet_length",
        True,
    )
    ifft = fft.fft_vcc(64, False, (), True, 1)
    cp = digital.ofdm_cyclic_prefixer(64, 64 + phy.cp_length, 0, "packet_length")
    guard = (
        [digital.burst_shaper_cc([1.0], 0, phy.packet_gap_samples, False, "packet_length")]
        if phy.packet_gap_samples
        else []
    )
    scale = blocks.multiply_const_cc(phy.digital_scale)
    wave = blocks.file_sink(gr.sizeof_gr_complex, str(folder / "waveform.c64"), False)
    headers = blocks.vector_sink_b()
    payload_indices = blocks.vector_sink_b()
    graph = gr.top_block("finite_native_ofdm_transmit")
    graph.connect(source, tagged)
    graph.connect(
        tagged, header, digital.chunks_to_symbols_bc(header_book.points(), 1), (mux, 0)
    )
    graph.connect(
        tagged, repack, digital.chunks_to_symbols_bc(payload_book.points(), 1), (mux, 1)
    )
    graph.connect(mux, allocator, ifft, cp, *guard, scale, wave)
    graph.connect(header, headers)
    graph.connect(repack, payload_indices)
    try:
        completed, _ = wait_graph_monitored(graph, 30, lambda: None)
    finally:
        graph.stop()
        graph.wait()
        wave.close()
        graph.disconnect_all()
    h = np.asarray(headers.data(), dtype="u1").reshape(-1, 48)
    classes = np.asarray(payload_indices.data(), dtype="u1").reshape(
        -1, phy.packet_length_bytes * 4
    )
    if (
        not completed
        or len(h) != plan["total_packets"]
        or (folder / "waveform.c64").stat().st_size
        != plan["total_packets"] * plan["frame_samples"] * 8
    ):
        raise RuntimeError("Native transmit frame count/length mismatch")
    from .phy import classes_to_bytes

    if any(
        classes_to_bytes(row) != payload for row, payload in zip(classes, payloads, strict=True)
    ):
        raise RuntimeError("GNU payload bit-order mismatch")
    sequences = [
        int(np.sum(row[12:24].astype(np.uint64) << np.arange(12, dtype=np.uint64))) for row in h
    ]
    if len(set(sequences)) != len(sequences):
        raise ValueError("Native formatter reused a sequence ID")
    h.tofile(folder / "tx_header.u8")
    classes.tofile(folder / "tx_payload_indices.u8")
    truth = dict(
        packets=[
            dict(header_sequence=k, payload_hex=value.hex())
            for k, value in zip(sequences, payloads, strict=True)
        ],
        measurement_ids=sequences[plan["warmup_packets"] :],
    )
    write_json(folder / "truth.json", truth)
    write_json(folder / "protocol.json", plan)
    write_json(
        folder / "waveform.json",
        dict(
            gnuradio_version=gr.version(),
            waveform_sha256=sha256(folder / "waveform.c64"),
            **waveform_limits(folder / "waveform.c64"),
            payload_crc=False,
            seed=seed,
            **plan,
        ),
    )
    return plan
