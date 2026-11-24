"""GNU Radio FFT/header-parser/payload receiver with a replaceable estimator.

Derived from the audited project receiver and GNU Radio OFDM examples under
GPL-3.0-or-later. This module intentionally retains the native scheduler,
header feedback, serializers and frame equalizers. See NOTICE.md.
"""

import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .config import Method
from .estimators import Estimator
from .io import write_json as write
from .phy import classes_to_bytes


def run(cfg, frontend, output, estimator=Method(), equalizer="simpledfe", timeout=60):
    """Decode a sealed frontend without TX payload, TX timing or scorer access.

    cfg is a PHY record. estimator='gnu' selects the unmodified GNU channel
    estimator for parity checks; other values are Method records.
    """
    import pmt
    from gnuradio import blocks, digital, fft, gr

    from .monitor import wait_graph_monitored

    phy = cfg
    if equalizer not in ("simpledfe", "static", "pilot_cpe"):
        raise ValueError("Unknown equalizer")
    folder, output = Path(frontend), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = np.asarray(phy.occupied_carriers[0]) + 32
    header_book = np.asarray(digital.constellation_bpsk().points())
    payload_book = np.asarray(digital.constellation_qpsk().points())
    expected_symbols = phy.packet_length_bytes * 4
    expected_frames = int(np.ceil(expected_symbols / len(data)))
    pending, accepted, records, packets, failures = deque(), deque(), [], [], []
    trace_header, trace_payload, trace_soft_header, trace_soft_payload, trace_taps = (
        [],
        [],
        [],
        [],
        [],
    )
    lock = threading.Lock()

    class Replay(gr.sync_block):
        def __init__(self, path, dtype):
            gr.sync_block.__init__(self, name="audit_replay", in_sig=None, out_sig=[dtype])
            self.a = np.memmap(path, dtype=dtype, mode="r")
            self.pos = 0

        def work(self, ins, outs):
            if self.pos == len(self.a):
                return -1
            n = min(len(outs[0]), len(self.a) - self.pos)
            if self.pos == 0 and np.iscomplexobj(self.a):
                self.add_item_tag(
                    0,
                    self.nitems_written(0),
                    pmt.intern("rx_time"),
                    pmt.make_tuple(pmt.from_uint64(0), pmt.from_double(0.0)),
                )
            outs[0][:n] = self.a[self.pos : self.pos + n]
            self.pos += n
            return n

    class HeaderProbe(gr.sync_block):
        def __init__(self):
            gr.sync_block.__init__(
                self,
                name="audit_header_input",
                in_sig=[(np.complex64, 64)],
                out_sig=[(np.complex64, 64)],
            )
            self.parts = []
            self.sample = None
            self.serial = 0

        def work(self, ins, outs):
            n = min(len(ins[0]), len(outs[0]))
            outs[0][:n] = ins[0][:n]
            for i, row in enumerate(ins[0][:n]):
                if not self.parts:
                    self.sample = None
                    for t in self.get_tags_in_range(
                        0, self.nitems_read(0) + i, self.nitems_read(0) + i + 1
                    ):
                        if pmt.symbol_to_string(t.key) == "rx_time":
                            sec = pmt.to_uint64(pmt.tuple_ref(t.value, 0)) + pmt.to_double(
                                pmt.tuple_ref(t.value, 1)
                            )
                            self.sample = round(sec * phy.sample_rate)
                self.parts.append(row.copy())
                if len(self.parts) == 3:
                    trace_header.append(np.asarray(self.parts))
                    with lock:
                        pending.append(
                            dict(index=self.serial, rx_sample=self.sample, header=[], parsed={})
                        )
                    self.serial += 1
                    self.parts = []
            return n

    class PreambleChannelEstimator(gr.basic_block):
        def __init__(self):
            gr.basic_block.__init__(
                self,
                name="public_preamble_channel_estimator",
                in_sig=[(np.complex64, 64)],
                out_sig=[(np.complex64, 64)],
            )
            self.set_tag_propagation_policy(gr.TPP_DONT)
            self.model = Estimator(estimator, phy)
            self.failure = None

        def forecast(self, noutput, ninputs):
            return [3] * ninputs

        def general_work(self, ins, outs):
            count = min(len(ins[0]) // 3, len(outs[0]))
            for j in range(count):
                try:
                    taps, offset = self.model.estimate(ins[0][3 * j : 3 * j + 3])
                except Exception as exc:
                    # Do not raise across GNU Radio's work-thread boundary.
                    # The controller observes this flag and stops the whole flowgraph.
                    self.failure = f"{type(exc).__name__}: {exc}"
                    print("ESTIMATOR ERROR: " + self.failure, file=sys.stderr, flush=True)
                    return -1
                target = self.nitems_written(0) + j
                outs[0][j] = ins[0][3 * j + 2]
                self.add_item_tag(
                    0, target, pmt.intern("ofdm_sync_carr_offset"), pmt.from_long(offset)
                )
                self.add_item_tag(
                    0,
                    target,
                    pmt.intern("ofdm_sync_chan_taps"),
                    pmt.init_c32vector(64, taps.tolist()),
                )
                for t in self.get_tags_in_range(
                    0, self.nitems_read(0) + 3 * j, self.nitems_read(0) + 3 * j + 3
                ):
                    self.add_item_tag(0, target, t.key, t.value)
            self.consume(0, count * 3)
            return count

    class SoftHeader(gr.sync_block):
        def __init__(self):
            gr.sync_block.__init__(
                self,
                name="audit_header_soft",
                in_sig=[(np.complex64, 64)],
                out_sig=[(np.complex64, 64)],
            )

        def work(self, ins, outs):
            n = min(len(ins[0]), len(outs[0]))
            outs[0][:n] = ins[0][:n]
            for i, row in enumerate(ins[0][:n]):
                trace_soft_header.append(row.copy())
                with lock:
                    # HPD releases one header then waits for its parser feedback.
                    state = pending[0]
                    state["header"] = np.argmin(
                        abs(row[data, None] - header_book[None]) ** 2, axis=1
                    ).tolist()
                    for t in self.get_tags_in_range(
                        0, self.nitems_read(0) + i, self.nitems_read(0) + i + 1
                    ):
                        key = pmt.symbol_to_string(t.key)
                        if key == "ofdm_sync_carr_offset":
                            state["integer_cfo"] = pmt.to_long(t.value)
                        if key == "ofdm_sync_chan_taps":
                            trace_taps.append(np.asarray(pmt.c32vector_elements(t.value)))
            return n

    class Feedback(gr.basic_block):
        def __init__(self):
            gr.basic_block.__init__(
                self, name="audit_parser_feedback", in_sig=None, out_sig=None
            )
            self.message_port_register_in(pmt.intern("in"))
            self.message_port_register_out(pmt.intern("out"))
            self.set_msg_handler(pmt.intern("in"), self.handle)

        def handle(self, msg):
            with lock:
                if not pending:
                    failures.append("parser_without_header")
                    self.message_port_pub(pmt.intern("out"), pmt.PMT_F)
                    return
                state = pending.popleft()
                parsed = {}
                if pmt.is_dict(msg):
                    for k in ("packet_length", "frame_length", "packet_num"):
                        v = pmt.dict_ref(msg, pmt.intern(k), pmt.PMT_NIL)
                        if pmt.is_integer(v):
                            parsed[k] = pmt.to_long(v)
                valid = (
                    len(parsed) == 3
                    and parsed["packet_length"] == expected_symbols
                    and parsed["frame_length"] == expected_frames
                )
                state.update(
                    parsed=parsed,
                    native_parser_valid=bool(pmt.is_dict(msg)),
                    accepted=valid,
                    rejection_reason=None
                    if valid
                    else (
                        "native_parser_rejected"
                        if not pmt.is_dict(msg)
                        else "public_length_guard"
                    ),
                )
                records.append(dict(state))
                if valid:
                    accepted.append(state)
            # Preserve native channel-state and offset PMTs. No synthetic success.
            self.message_port_pub(pmt.intern("out"), msg if valid else pmt.PMT_F)

    class PayloadProbe(gr.sync_block):
        def __init__(self, soft):
            gr.sync_block.__init__(
                self,
                name="audit_payload_soft" if soft else "audit_payload_raw",
                in_sig=[(np.complex64, 64)],
                out_sig=[(np.complex64, 64)],
            )
            self.soft = soft
            self.current = None
            self.rows = []

        def work(self, ins, outs):
            n = min(len(ins[0]), len(outs[0]))
            outs[0][:n] = ins[0][:n]
            for row in ins[0][:n]:
                if not self.soft:
                    trace_payload.append(row.copy())
                    continue
                trace_soft_payload.append(row.copy())
                if self.current is None:
                    with lock:
                        if not accepted:
                            failures.append("payload_without_accepted_header")
                            return -1
                        self.current = accepted.popleft()
                    self.rows = []
                self.rows.append(row.copy())
                if len(self.rows) == self.current["parsed"]["frame_length"]:
                    soft = np.asarray(self.rows)[:, data].reshape(-1)
                    dec = np.argmin(
                        abs(soft[:, None] - payload_book[None]) ** 2, axis=1
                    ).astype("u1")[:expected_symbols]
                    start = self.current["rx_sample"]
                    packets.append(
                        dict(
                            header_index=self.current["index"],
                            rx_sample=start,
                            payload_start_sample=None
                            if start is None
                            else start + 3 * (64 + phy.cp_length),
                            payload_end_sample=None
                            if start is None
                            else start + (3 + len(self.rows)) * (64 + phy.cp_length),
                            parsed=self.current["parsed"],
                            payload_hex=classes_to_bytes(dec).hex(),
                        )
                    )
                    self.current = None
            return n

    hc, pc = digital.constellation_bpsk(), digital.constellation_qpsk()

    def equalize(book, payload):
        if equalizer in ("static", "pilot_cpe"):
            obj = digital.ofdm_equalizer_static(
                64,
                phy.occupied_carriers,
                phy.pilot_carriers,
                phy.pilot_symbols,
                int(payload),
                True,
            )
        else:
            obj = digital.ofdm_equalizer_simpledfe(
                64,
                book.base(),
                phy.occupied_carriers,
                phy.pilot_carriers,
                phy.pilot_symbols,
                int(payload),
                float(phy.equalizer_alpha),
                True,
                True,
            )
        return digital.ofdm_frame_equalizer_vcvc(
            obj.base(), phy.cp_length, "frame_length", True, 0 if payload else 1
        )

    source = Replay(folder / "corrected.c64", np.complex64)
    trigger = Replay(folder / "triggers.u8", np.uint8)
    demux = digital.header_payload_demux(
        3,
        64,
        phy.cp_length,
        "frame_length",
        "",
        True,
        gr.sizeof_gr_complex,
        "rx_time",
        phy.sample_rate,
        (),
        0,
    )
    demux.set_min_output_buffer(16384)
    hp = HeaderProbe()
    sp = SoftHeader()
    fb = Feedback()
    rp = PayloadProbe(False)
    pp = PayloadProbe(True)
    ce = (
        digital.ofdm_chanest_vcvc(phy.sync_word1, phy.sync_word2, 1, 0, 3, False)
        if estimator == "gnu"
        else PreambleChannelEstimator()
    )
    hs = digital.ofdm_serializer_vcc(64, phy.occupied_carriers, "frame_length", "", 0, "", True)
    ps = digital.ofdm_serializer_vcc(
        64, phy.occupied_carriers, "frame_length", "packet_length", 1, "", True
    )
    fmt = digital.packet_header_ofdm(
        phy.occupied_carriers, 1, "packet_length", "frame_length", "packet_num", 1, 2, False
    )
    parser = digital.packet_headerparser_b(fmt.formatter())
    sink = blocks.vector_sink_b()
    tb = gr.top_block("ofdm_benchmark_" + equalizer)
    tb.connect(source, (demux, 0))
    tb.connect(trigger, (demux, 1))
    from .cpe import PilotPhaseTracker

    # Pilot-CPE gira o símbolo pela fase comum antes da equalização estática.
    # O cabeçalho e o payload têm índices de polaridade distintos.
    use_cpe = equalizer == "pilot_cpe"
    header_cpe = [PilotPhaseTracker(phy, False)] if use_cpe else []
    payload_cpe = [PilotPhaseTracker(phy, True)] if use_cpe else []
    tb.connect(
        (demux, 0),
        fft.fft_vcc(64, True, (), True, 1),
        hp,
        ce,
        *header_cpe,
        equalize(hc, False),
        sp,
        hs,
        digital.constellation_decoder_cb(hc.base()),
        parser,
    )
    tb.msg_connect(parser, "header_data", fb, "in")
    tb.msg_connect(fb, "out", demux, "header_data")
    tb.connect(
        (demux, 1),
        fft.fft_vcc(64, True, (), True, 1),
        rp,
        *payload_cpe,
        equalize(pc, True),
        pp,
        ps,
        digital.constellation_decoder_cb(pc.base()),
        blocks.repack_bits_bb(2, 8, "packet_length", True, gr.GR_LSB_FIRST),
        sink,
    )
    start = time.perf_counter()
    done, estimator_failure = wait_graph_monitored(
        tb, timeout, lambda: getattr(ce, "failure", None) or (failures[0] if failures else None)
    )
    tb.stop()
    tb.wait()
    received = bytes(sink.data())
    expected = b"".join(bytes.fromhex(p["payload_hex"]) for p in packets)
    timestamps_ok = all(h["rx_sample"] is not None for h in records)
    technical = bool(
        done
        and source.pos == len(source.a)
        and trigger.pos == len(trigger.a)
        and received == expected
        and not failures
        and estimator_failure is None
        and not pending
        and not accepted
        and pp.current is None
        and timestamps_ok
    )
    result = dict(
        estimator="gnu" if estimator == "gnu" else estimator.name,
        equalizer=equalizer,
        impairment_post_frontend=False,
        alpha=float(phy.equalizer_alpha) if equalizer == "simpledfe" else None,
        cpe="pilot" if use_cpe else "off",
        native_parser_guard="fixed public payload length",
        headers=records,
        packets=packets,
        native_payload_hex=received.hex(),
        technical_valid=technical,
        natural_completion=done,
        errors=failures,
        estimator_block_failure=estimator_failure,
        iq_samples_read=source.pos,
        trigger_samples_read=trigger.pos,
        source_samples=len(source.a),
        input_consumption="producer delivered to GNU graph; demux consumption recorded separately",
        demux_iq_samples_consumed=int(demux.nitems_read(0)),
        demux_trigger_samples_consumed=int(demux.nitems_read(1)),
        headers_without_parser_feedback=len(pending),
        accepted_without_payload=len(accepted) + int(pp.current is not None),
        timestamps_complete=timestamps_ok,
        native_payload_byte_parity=received == expected,
        seconds=time.perf_counter() - start,
        gnuradio_version=gr.version(),
        estimator_diagnostics=ce.model.diagnostics()
        if estimator != "gnu"
        else {"method": "gnu", "fit_calls": len(records)},
    )
    write(output / "receiver.json", result)
    np.savez_compressed(
        output / "traces.npz",
        header_raw=np.asarray(trace_header, np.complex64),
        payload_raw=np.asarray(trace_payload, np.complex64),
        header_soft=np.asarray(trace_soft_header, np.complex64),
        payload_soft=np.asarray(trace_soft_payload, np.complex64),
        header_output_taps=np.asarray(trace_taps, np.complex64),
    )
    tb.disconnect_all()
    return result
