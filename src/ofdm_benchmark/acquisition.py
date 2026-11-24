"""Finite HackRF captures in tmpfs; no estimator runs while either radio is active."""

from __future__ import annotations

import gc
import re
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .io import sha256, write_json
from .transmit import generate


def shortfalls(serial, *, busy_retries=4, retry_seconds=0.25):
    """Read HackRF diagnostics after release; never treat unknown as a pass.

    A newly released osmosdr device can briefly return Resource busy. Retry only
    that transient diagnostic failure. These retries do not reacquire RF or
    relax the requirement for verified zero shortfalls.
    """
    busy_outputs = []
    for attempt in range(1, busy_retries + 2):
        try:
            result = subprocess.run(
                ["hackrf_debug", "-d", serial, "-S"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return dict(
                serial=serial,
                verified=False,
                counts={},
                error=str(error),
                attempts=attempt,
                transient_busy_outputs=busy_outputs,
            )
        output = result.stdout + result.stderr
        if "resource busy" in output.lower() and attempt <= busy_retries:
            busy_outputs.append(output)
            time.sleep(retry_seconds * attempt)
            continue
        counts = {}
        for line in output.splitlines():
            low = line.lower()
            if "shortfall" not in low or any(s in low for s in ("longest", "duration")):
                continue
            values = re.findall(r"(?:[:=]\s*|\bcount\s+)(\d+)\s*$", low)
            direction = re.search(r"\b(rx|tx)\b", low)
            if values and direction:
                counts[direction.group(1)] = int(values[-1])
            elif values and re.match(r"\s*number of shortfalls\s*:", low):
                counts["total"] = int(values[-1])
        known = result.returncode == 0 and set(counts) in ({"rx", "tx"}, {"total"})
        return dict(
            serial=serial,
            verified=known,
            counts=counts,
            output=output,
            attempts=attempt,
            transient_busy_outputs=busy_outputs,
        )
    raise AssertionError("Unreachable diagnostic retry loop")


def scheduler_sample_counts(writer, file_source):
    """GNU scheduler counters; not a measurement of samples emitted by the DAC.

    osmosdr.sink is a hierarchical device API and does not expose nitems_read()
    on the user's gr-osmosdr 0.2.0 build. GNU blocks.file_source exposes the
    inherited nitems_written(0) counter for the single output stream instead.
    """
    return dict(
        rx_samples=int(writer.nitems_read(0)),
        tx_samples=int(file_source.nitems_written(0)),
    )


def inspect_iq(path, idle_samples, sample_rate):
    """Estimate power from fixed windows, independent of decoded quality.

    DC and idle-noise power use a settled interval before TX is requested.
    Active windows exceed idle power by 6 dB; exclude edge windows. This is a
    proxy, not a calibrated RF SNR. Store enough detail to audit thresholding.
    """
    data = np.memmap(path, dtype="<c8", mode="r")
    start, stop = int(0.05 * sample_rate), int(idle_samples) - 8192
    if stop - start < 8192:
        raise ValueError("Insufficient known TX-off settling interval")
    idle = np.asarray(data[start:stop], np.complex128)
    dc = complex(idle.mean())
    noise = float(np.mean(abs(idle - dc) ** 2))
    size = 4096
    power = np.array(
        [
            np.mean(abs(np.asarray(data[i : i + size], np.complex128) - dc) ** 2)
            for i in range(0, len(data) - size + 1, size)
        ]
    )
    active = np.flatnonzero(power > max(noise * 10**0.6, 1e-12))
    active = active[1:-1]
    if len(active) < 3:
        raise ValueError("Insufficient independently detected RF activity")
    selected = np.concatenate([np.asarray(data[i * size : (i + 1) * size]) for i in active])
    signal_power = max(float(np.mean(power[active]) - noise), 1e-20)
    rails = float(np.mean((abs(selected.real) >= 0.99) | (abs(selected.imag) >= 0.99)))
    return dict(
        samples=len(data),
        finite=bool(np.isfinite(data).all()),
        signal_power_proxy=signal_power,
        noise_mean_power=noise,
        dc_offset={"real": dc.real, "imag": dc.imag},
        active_rail_fraction=rails,
        known_tx_off_range=[start, stop],
        window_samples=size,
        active_windows=active.tolist(),
        window_power=power.tolist(),
        activity_threshold_db=6.0,
        power_scope="Activity-based post-ADC proxy",
    )


def capture_one(experiment, folder: Path, burst: int):
    """Record one physical burst. Partial/failed captures remain in the output.

    Called in an isolated process with folder on tmpfs. TX waveform is generated
    beforehand by GNU blocks. A C++ file source feeds the transmitter; a C++
    file sink records RX. Native stream counts and USB diagnostics bound validity.
    """
    import osmosdr
    from gnuradio import blocks, gr

    cfg, phy, radio = experiment, experiment.phy, experiment.radio
    if (
        any(
            s.startswith("SET_")
            for s in (radio.tx_serial, radio.rx_serial, cfg.session_id, cfg.channel_description)
        )
        or radio.tx_serial == radio.rx_serial
    ):
        raise ValueError(
            "Fill distinct radio serials, session ID and physical channel description"
        )
    folder.mkdir(parents=True, exist_ok=False)
    plan = generate(
        folder / "tx", phy, cfg.measurement_packets, cfg.tx_warmup_seconds, cfg.seed + burst
    )
    from .transmit import waveform_limits

    limits = waveform_limits(folder / "tx/waveform.c64")
    write_json(folder / "tx_headroom.json", limits)
    if limits["peak_component"] > 0.95:
        raise ValueError(
            "TX waveform exceeds the declared 0.95 component headroom; inspect tx_headroom.json and calibrate a fixed lower digital scale before RF"
        )
    # A margem de 350 ms acomoda a partida da cadeia além do repouso inicial e da cauda RX.
    duration = (
        cfg.rx_warmup_seconds + plan["nominal_burst_seconds"] + 0.35 + cfg.rx_tail_seconds
    )
    count = round(duration * phy.sample_rate)
    source = osmosdr.source(args="numchan=1 hackrf=" + radio.rx_serial)
    source.set_sample_rate(phy.sample_rate)
    source.set_center_freq(radio.center_frequency, 0)
    source.set_gain_mode(False, 0)
    source.set_gain(radio.rx_rf_gain, 0)
    source.set_if_gain(radio.rx_if_gain, 0)
    source.set_bb_gain(radio.rx_bb_gain, 0)
    source.set_bandwidth(radio.bandwidth, 0)
    sink = osmosdr.sink(args="numchan=1 hackrf=" + radio.tx_serial)
    sink.set_sample_rate(phy.sample_rate)
    sink.set_center_freq(radio.center_frequency, 0)
    sink.set_gain(radio.tx_rf_gain, 0)
    sink.set_if_gain(radio.tx_if_gain, 0)
    sink.set_bandwidth(radio.bandwidth, 0)
    actual = dict(
        rx_sample_rate=float(source.get_sample_rate()),
        tx_sample_rate=float(sink.get_sample_rate()),
        rx_frequency=float(source.get_center_freq()),
        tx_frequency=float(sink.get_center_freq()),
        rx_bandwidth=float(source.get_bandwidth()),
        tx_bandwidth=float(sink.get_bandwidth()),
    )
    if any(abs(actual[k] - phy.sample_rate) > 1 for k in ("rx_sample_rate", "tx_sample_rate")):
        raise RuntimeError("Driver sample rate differs from requested rate")
    head = blocks.head(gr.sizeof_gr_complex, count)
    writer = blocks.file_sink(gr.sizeof_gr_complex, str(folder / "capture.c64.partial"), False)
    writer.set_unbuffered(False)
    file_source = blocks.file_source(
        gr.sizeof_gr_complex, str(folder / "tx/waveform.c64"), False
    )
    rx, tx = gr.top_block("ram_capture"), gr.top_block("finite_rf_transmit")
    rx.connect(source, head, writer)
    tx.connect(file_source, sink)
    completed, errors, threads, started = {}, [], [], []
    status = dict(
        source_kind="physical_rf",
        burst=burst,
        hardware_sample_timestamps=False,
        actual_radio=actual,
    )

    def wait(graph, key):
        try:
            graph.wait()
            completed[key] = time.monotonic()
        except BaseException as error:
            errors.append(repr(error))

    beginning = time.monotonic()
    try:
        rx.start()
        started.append(rx)
        thread = threading.Thread(target=wait, args=(rx, "rx"), daemon=True)
        thread.start()
        threads.append(thread)
        deadline = time.monotonic() + 15
        while writer.nitems_read(0) < round(cfg.rx_warmup_seconds * phy.sample_rate):
            if errors or not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("RX warm-up failed")
            time.sleep(0.02)
        status["rx_samples_before_tx_start"] = int(writer.nitems_read(0))
        tx.start()
        started.append(tx)
        thread = threading.Thread(target=wait, args=(tx, "tx"), daemon=True)
        thread.start()
        threads.append(thread)
        deadline = time.monotonic() + duration + 20
        while any(t.is_alive() for t in threads):
            if errors or time.monotonic() > deadline:
                raise RuntimeError("Finite TX/RX completion failed: " + str(errors))
            time.sleep(0.02)
        status.update(
            **scheduler_sample_counts(writer, file_source),
            rx_end_after_tx_done_seconds=completed["rx"] - completed["tx"],
            tx_counter_source="gnu_file_source_nitems_written",
        )
    finally:
        for graph in started:
            graph.stop()
        for thread in threads:
            thread.join(5)
        if any(t.is_alive() for t in threads):
            raise RuntimeError("GNU scheduler failed to stop")
        writer.close()
        rx.disconnect_all()
        tx.disconnect_all()
        started.clear()
        if "graph" in locals():
            del graph
        del source, sink, rx, tx, writer, file_source, head
        gc.collect()
        status["capture_seconds"] = time.monotonic() - beginning
        status["usb"] = [shortfalls(s) for s in (radio.tx_serial, radio.rx_serial)]
        write_json(folder / "capture_status.json", status)
    (folder / "capture.c64.partial").rename(folder / "capture.c64")
    health = inspect_iq(
        folder / "capture.c64", status["rx_samples_before_tx_start"], phy.sample_rate
    )
    expected_tx = plan["total_packets"] * plan["frame_samples"]
    valid = (
        health["finite"]
        and health["active_rail_fraction"] < 0.01
        and status["rx_samples"] == count
        and status["tx_samples"] == expected_tx
        and status["rx_end_after_tx_done_seconds"] >= 0
        and all(x["verified"] and not any(x["counts"].values()) for x in status["usb"])
    )
    metadata = dict(
        schema=1,
        burst=burst,
        session_id=cfg.session_id,
        channel_description=cfg.channel_description,
        source_kind="physical_rf",
        physical_valid=bool(valid),
        phy=asdict(phy),
        radio=asdict(radio),
        protocol=plan,
        power=health,
        raw_sha256=sha256(folder / "capture.c64"),
        truth_sha256=sha256(folder / "tx/truth.json"),
        required_gnu_version=gr.version(),
    )
    write_json(folder / "capture.json", metadata)
    if not valid:
        raise RuntimeError(
            "Physical capture failed technical checks; preserve it, do not retry by PER"
        )
    return metadata
