import os, time, json, argparse, traceback, struct
from collections import deque
import numpy as np
from sklearn.svm import SVR
from gnuradio import gr, blocks, digital, fft, analog, channels
import pmt
import osmosdr

SAMP_RATE_DEFAULT = 1800000
CENTER_FREQ = 907980000.0
FFT_LEN = 64
CP_LEN = 16
OCCUPIED_CARRIERS = ((-4, -3, -2, -1, 1, 2, 3, 4),)
PILOT_CARRIERS    = ((-6, -5, 5, 6),)
PILOT_SYMBOLS     = ((-1, 1, -1, 1),)
LENGTH_TAG_KEY = "packet_len"
FRAME_LEN_TAG_KEY_RX = "packet_tag"
PACKET_LENGTH_TAG_KEY_RX = "packet_len"
PACKET_LEN_BYTES = 50

# Preserve exactly the sync words from the original working runner.
SYNC_WORD1 = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.41421356, 0.0, 1.41421356,
    0.0, -1.41421356, 0.0, 1.41421356, 0.0, -1.41421356, 0.0, 1.41421356, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]
SYNC_WORD2 = [
    0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j,
    0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, (1+0j), (1+0j), (1+0j), (-1+0j),
    (1+0j), (1+0j), 0j, (1+0j), (-1+0j), (1+0j), (1+0j), (1+0j), (-1+0j), 0j,
    0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j,
    0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j,
]
assert len(SYNC_WORD1) == FFT_LEN
assert len(SYNC_WORD2) == FFT_LEN

WORKDIR = os.getcwd()
TX_PAYLOAD = os.path.join(WORKDIR, "tx_payload.bin")
RX_PAYLOAD = os.path.join(WORKDIR, "rx_payload.bin")
RX_IQ_FILE = os.path.join(WORKDIR, "rx_iq.cfile")

TX_HACKRF_DEFAULT = "hackrf=0"
RX_HACKRF_DEFAULT = "hackrf=1"

SYNC_WORD2_NP = np.array(SYNC_WORD2, dtype=np.complex128)
SYNC2_MASK = np.abs(SYNC_WORD2_NP) > 0


def ensure_payload_file(path=TX_PAYLOAD, nbytes=4096):
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(os.urandom(nbytes))
    return path


def compute_ber_bytes(tx: bytes, rx: bytes):
    n = min(len(tx), len(rx))
    if n == 0:
        return None
    bit_errors = 0
    for i in range(n):
        bit_errors += (tx[i] ^ rx[i]).bit_count()
    return bit_errors / (n * 8)


def best_offset_by_window(tx: bytes, rx: bytes, max_shift=4096, window=128):
    if len(tx) < window or len(rx) < window:
        return 0, 0
    tx_win = tx[:window]
    best_score = -1
    best_i = 0
    search_len = min(len(rx) - window, max_shift)
    for i in range(0, max(1, search_len + 1)):
        cand = rx[i:i + window]
        score = sum(a == b for a, b in zip(tx_win, cand))
        if score > best_score:
            best_score = score
            best_i = i
    return best_i, int(best_score)


def compute_ber_aligned_files(tx_path, rx_path, max_bytes=4096, search_max_shift=4096, window=128):
    if not os.path.exists(tx_path) or not os.path.exists(rx_path):
        return None
    tx = open(tx_path, "rb").read(max_bytes)
    rx = open(rx_path, "rb").read(max_bytes)
    if len(tx) == 0 or len(rx) == 0:
        return None
    ber_raw = compute_ber_bytes(tx, rx)
    off, score = best_offset_by_window(tx, rx, max_shift=search_max_shift, window=window)
    rx2 = rx[off:]
    ber_al = compute_ber_bytes(tx, rx2)
    return {"ber_raw": ber_raw, "offset": off, "score": score, "ber_aligned": ber_al}


def count_pattern_overlapping(rx: bytes, pat: bytes):
    cnt = 0
    start = 0
    while True:
        i = rx.find(pat, start)
        if i == -1:
            return cnt
        cnt += 1
        start = i + 1


def _randbytes_py(seed: int, n: int) -> bytes:
    import random
    r = random.Random(int(seed))
    return bytes(r.getrandbits(8) for _ in range(int(n)))


def generate_benchmark_payload(seed: int, num_packets: int, packet_len: int, magic4: bytes):
    if packet_len != PACKET_LEN_BYTES:
        raise ValueError(f"packet_len must be {PACKET_LEN_BYTES} to match current framing (got {packet_len})")
    if len(magic4) != 4:
        raise ValueError("magic must be exactly 4 bytes")
    per_payload = packet_len - 8
    out = bytearray()
    tx_map = {}
    for seq in range(int(num_packets)):
        rnd = _randbytes_py(seed + seq, per_payload)
        pkt = magic4 + struct.pack("<I", seq) + rnd
        out.extend(pkt)
        tx_map[seq] = rnd
    return bytes(out), tx_map


def parse_rx_packets_by_magic(rx: bytes, magic4: bytes, packet_len: int):
    if len(magic4) != 4 or packet_len < 8:
        return {}, 0
    parsed = {}
    parsed_total = 0
    i = 0
    while True:
        j = rx.find(magic4, i)
        if j == -1:
            break
        if j + packet_len <= len(rx):
            seq = struct.unpack_from("<I", rx, j + 4)[0]
            payload = rx[j + 8:j + packet_len]
            parsed_total += 1
            if seq not in parsed:
                parsed[seq] = payload
        i = j + 1
    return parsed, parsed_total


def compute_benchmark_packet_ber(tx_map, rx_bytes: bytes, magic4: bytes, packet_len: int):
    rx_map, parsed_total = parse_rx_packets_by_magic(rx_bytes, magic4, packet_len)
    matched = 0
    bit_errors = 0
    payload_bits = (packet_len - 8) * 8
    for seq, tx_payload in tx_map.items():
        rx_payload = rx_map.get(seq)
        if rx_payload is None:
            continue
        matched += 1
        n = min(len(tx_payload), len(rx_payload))
        for k in range(n):
            bit_errors += (tx_payload[k] ^ rx_payload[k]).bit_count()
    expected = len(tx_map)
    missing = expected - matched
    ber_packets = None
    denom_bits = matched * payload_bits
    if denom_bits > 0:
        ber_packets = bit_errors / denom_bits
    return {
        "rx_packets_parsed": int(parsed_total),
        "matched_packets": int(matched),
        "missing_packets": int(missing),
        "ber_packets": None if ber_packets is None else float(ber_packets),
    }


def _interp_nearest_fullband(h_partial, known_mask):
    out = np.array(h_partial, dtype=np.complex128, copy=True)
    known_idx = np.where(known_mask)[0]
    if len(known_idx) == 0:
        return out
    for i in range(len(out)):
        if not known_mask[i]:
            j = known_idx[np.argmin(np.abs(known_idx - i))]
            out[i] = out[j]
    return out


def estimate_ls_from_sync(rx_sync_freq, ref_sync_freq):
    rx_sync_freq = np.asarray(rx_sync_freq, dtype=np.complex128)
    ref_sync_freq = np.asarray(ref_sync_freq, dtype=np.complex128)
    known_mask = np.abs(ref_sync_freq) > 0
    h_est = np.zeros_like(rx_sync_freq, dtype=np.complex128)
    h_est[known_mask] = rx_sync_freq[known_mask] / ref_sync_freq[known_mask]
    return _interp_nearest_fullband(h_est, known_mask)


def estimate_svr_time_to_freq(rx_sync_freq, ref_sync_freq, fft_len, svr_C=100.0, svr_gamma="scale", svr_epsilon=1e-3):
    rx_sync_freq = np.asarray(rx_sync_freq, dtype=np.complex128)
    ref_sync_freq = np.asarray(ref_sync_freq, dtype=np.complex128)
    rx_sync_time = np.fft.ifft(rx_sync_freq, fft_len)

    used_bins = np.where(np.abs(ref_sync_freq) > 0)[0]
    n = np.arange(fft_len)
    P = len(used_bins)
    if P == 0:
        return np.ones(fft_len, dtype=np.complex128)

    X_re = np.zeros((fft_len, P), dtype=np.float64)
    X_im = np.zeros((fft_len, P), dtype=np.float64)

    for j, k in enumerate(used_bins):
        basis = ref_sync_freq[k] * np.exp(1j * 2 * np.pi * k * n / fft_len)
        X_re[:, j] = basis.real
        X_im[:, j] = basis.imag

    y_re = rx_sync_time.real
    y_im = rx_sync_time.imag

    svr_re = SVR(kernel="rbf", C=svr_C, gamma=svr_gamma, epsilon=svr_epsilon)
    svr_im = SVR(kernel="rbf", C=svr_C, gamma=svr_gamma, epsilon=svr_epsilon)
    svr_re.fit(X_re, y_re)
    svr_im.fit(X_im, y_im)

    h_est_time = svr_re.predict(X_re) + 1j * svr_im.predict(X_im)
    h_est_freq = np.fft.fft(h_est_time, fft_len)
    return np.asarray(h_est_freq, dtype=np.complex128)


def sync_reconstruction_metrics(rx_sync_freq, ref_sync_freq, h_est):
    rx_sync_freq = np.asarray(rx_sync_freq, dtype=np.complex128)
    ref_sync_freq = np.asarray(ref_sync_freq, dtype=np.complex128)
    h_est = np.asarray(h_est, dtype=np.complex128)
    mask = np.abs(ref_sync_freq) > 0
    if not np.any(mask):
        return None, None
    y_hat = h_est[mask] * ref_sync_freq[mask]
    err = rx_sync_freq[mask] - y_hat
    mse = np.mean(np.abs(err) ** 2)
    nmse = np.sum(np.abs(err) ** 2) / (np.sum(np.abs(rx_sync_freq[mask]) ** 2) + 1e-15)
    return float(mse), float(nmse)


class SyncMetricProbe(gr.sync_block):
    """
    Passive side probe on the FFT(header) stream.
    It does not feed the equalizer. It only computes sync reconstruction metrics.

    The LS path in the actual receiver remains the standard GNU Radio ofdm_chanest_vcvc.
    """
    def __init__(self, estimator_mode="ls", fft_len=FFT_LEN):
        gr.sync_block.__init__(
            self,
            name=f"sync_metric_probe_{estimator_mode}",
            in_sig=[(np.complex64, fft_len)],
            out_sig=None,
        )
        self.estimator_mode = estimator_mode.lower()
        self.fft_len = fft_len
        self.buf = deque()
        self.mse_values = []
        self.nmse_values = []

    def _estimate(self, rx_sync2):
        if self.estimator_mode == "svr":
            return estimate_svr_time_to_freq(rx_sync2, SYNC_WORD2_NP, self.fft_len)
        return estimate_ls_from_sync(rx_sync2, SYNC_WORD2_NP)

    def work(self, input_items, output_items):
        xs = input_items[0]
        for row in xs:
            self.buf.append(np.array(row, dtype=np.complex128, copy=True))
        while len(self.buf) >= 3:
            _sync1 = self.buf.popleft()
            sync2 = self.buf.popleft()
            _header = self.buf.popleft()
            h_est = self._estimate(sync2)
            mse, nmse = sync_reconstruction_metrics(sync2, SYNC_WORD2_NP, h_est)
            if mse is not None:
                self.mse_values.append(mse)
                self.nmse_values.append(nmse)
        return len(xs)

    def get_stats(self):
        return {
            "sync_mse": None if not self.mse_values else float(np.mean(self.mse_values)),
            "sync_nmse": None if not self.nmse_values else float(np.mean(self.nmse_values)),
            "sync_metric_frames": int(len(self.mse_values)),
        }


class SVRChannelEstimator(gr.basic_block):
    """
    Custom channel estimator with the same stream contract expected by the GNU Radio
    equalizer path: consume [sync1, sync2, header] and output only [header], tagging the
    first output symbol with ofdm_sync_eq_taps and ofdm_sync_carr_offset.
    """
    def __init__(self, fft_len=FFT_LEN, n_data_symbols=1, svr_C=100.0, svr_gamma="scale", svr_epsilon=1e-3):
        gr.basic_block.__init__(
            self,
            name="svr_chanest_vcvc",
            in_sig=[(np.complex64, fft_len)],
            out_sig=[(np.complex64, fft_len)],
        )
        self.fft_len = fft_len
        self.n_sync = 2
        self.n_data_symbols = int(n_data_symbols)
        self.frame_in = self.n_sync + self.n_data_symbols
        self.frame_out = self.n_data_symbols
        self.svr_C = float(svr_C)
        self.svr_gamma = svr_gamma
        self.svr_epsilon = float(svr_epsilon)
        self.key_eq = pmt.intern("ofdm_sync_eq_taps")
        self.key_off = pmt.intern("ofdm_sync_carr_offset")

    def forecast(self, noutput_items, ninput_items_required):
        need = max(self.frame_in, int(np.ceil(max(1, noutput_items) * self.frame_in / self.frame_out)))
        ninput_items_required[0] = need

    def general_work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        nin = len(inp)
        nout = len(out)
        if nin < self.frame_in or nout < self.frame_out:
            return 0

        nframes = min(nin // self.frame_in, nout // self.frame_out)
        if nframes <= 0:
            return 0

        produced = 0
        consumed = 0
        out_abs0 = self.nitems_written(0)

        for f in range(nframes):
            bi = f * self.frame_in
            bo = f * self.frame_out
            sync2 = np.array(inp[bi + 1], dtype=np.complex128)
            header_sym = np.array(inp[bi + 2], dtype=np.complex64)
            h_est = estimate_svr_time_to_freq(sync2, SYNC_WORD2_NP, self.fft_len, self.svr_C, self.svr_gamma, self.svr_epsilon)
            out[bo] = header_sym
            tag_offset = out_abs0 + bo
            self.add_item_tag(0, tag_offset, self.key_off, pmt.from_long(0))
            self.add_item_tag(
                0,
                tag_offset,
                self.key_eq,
                pmt.init_c32vector(self.fft_len, [complex(v) for v in h_est.astype(np.complex64)]),
            )
            produced += self.frame_out
            consumed += self.frame_in

        self.consume(0, consumed)
        return produced


class ChannelEstimatorWrapperLS(gr.hier_block2):
    def __init__(self):
        gr.hier_block2.__init__(
            self, "chanest_wrapper_ls",
            gr.io_signature(1, 1, gr.sizeof_gr_complex * FFT_LEN),
            gr.io_signature(1, 1, gr.sizeof_gr_complex * FFT_LEN),
        )
        self._chanest = digital.ofdm_chanest_vcvc(SYNC_WORD1, SYNC_WORD2, 1, 0, 3, False)
        self.connect(self, self._chanest, self)


class ChannelEstimatorWrapperSVR(gr.hier_block2):
    def __init__(self):
        gr.hier_block2.__init__(
            self, "chanest_wrapper_svr",
            gr.io_signature(1, 1, gr.sizeof_gr_complex * FFT_LEN),
            gr.io_signature(1, 1, gr.sizeof_gr_complex * FFT_LEN),
        )
        self._chanest = SVRChannelEstimator(fft_len=FFT_LEN, n_data_symbols=1)
        self.connect(self, self._chanest, self)


class OFDMTransmitter(gr.top_block):
    def __init__(self, hackrf_args, rf_gain, if_gain, bb_gain, scale, samp_rate, use_crc=False):
        gr.top_block.__init__(self, "ofdm_tx_headless")
        payload_mod = digital.constellation_qpsk()
        header_mod = digital.constellation_bpsk()
        hdr_format = digital.header_format_ofdm(OCCUPIED_CARRIERS, 1, LENGTH_TAG_KEY)

        self.file_source = blocks.file_source(gr.sizeof_char, TX_PAYLOAD, True)
        self.file_source.set_begin_tag(pmt.PMT_NIL)
        self.stream_to_tagged = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, PACKET_LEN_BYTES, LENGTH_TAG_KEY)
        self.use_crc = bool(use_crc)
        self.crc = digital.crc32_bb(False, LENGTH_TAG_KEY, True) if self.use_crc else None
        src_for_framing = self.crc if self.use_crc else self.stream_to_tagged
        self.protocol_formatter = digital.protocol_formatter_bb(hdr_format, LENGTH_TAG_KEY)

        self.repack_hdr = blocks.repack_bits_bb(8, header_mod.bits_per_symbol(), LENGTH_TAG_KEY, False, gr.GR_LSB_FIRST)
        self.repack_pay = blocks.repack_bits_bb(8, payload_mod.bits_per_symbol(), LENGTH_TAG_KEY, False, gr.GR_LSB_FIRST)
        self.map_hdr = digital.chunks_to_symbols_bc(header_mod.points(), 1)
        self.map_pay = digital.chunks_to_symbols_bc(payload_mod.points(), 1)
        self.mux = blocks.tagged_stream_mux(gr.sizeof_gr_complex, LENGTH_TAG_KEY, 0)

        self.carrier_alloc = digital.ofdm_carrier_allocator_cvc(
            FFT_LEN, OCCUPIED_CARRIERS, PILOT_CARRIERS, PILOT_SYMBOLS,
            (SYNC_WORD1, SYNC_WORD2), LENGTH_TAG_KEY, True
        )

        self.ifft = fft.fft_vcc(FFT_LEN, False, (), True, 1)
        self.cp = digital.ofdm_cyclic_prefixer(FFT_LEN, FFT_LEN + CP_LEN, 0, LENGTH_TAG_KEY)
        self.scale = blocks.multiply_const_cc(scale)

        self.sink = osmosdr.sink(args="numchan=1 " + hackrf_args)
        self.sink.set_sample_rate(float(samp_rate))
        self.sink.set_center_freq(CENTER_FREQ, 0)
        self.sink.set_gain(rf_gain, 0)
        self.sink.set_if_gain(if_gain, 0)
        self.sink.set_bb_gain(bb_gain, 0)

        self.connect(self.file_source, self.stream_to_tagged)
        if self.use_crc:
            self.connect(self.stream_to_tagged, self.crc)
        self.connect(src_for_framing, self.protocol_formatter, self.repack_hdr, self.map_hdr, (self.mux, 0))
        self.connect(src_for_framing, self.repack_pay, self.map_pay, (self.mux, 1))
        self.connect(self.mux, self.carrier_alloc, self.ifft, self.cp, self.scale, self.sink)


class OFDMReceiver(gr.top_block):
    def __init__(self, hackrf_args, rf_gain, if_gain, bb_gain, alpha_sync, noise_voltage, record_iq, samp_rate, use_crc=False, estimator="ls"):
        gr.top_block.__init__(self, "ofdm_rx_headless")
        payload_mod = digital.constellation_qpsk()
        header_mod = digital.constellation_bpsk()

        header_formatter = digital.packet_header_ofdm(
            OCCUPIED_CARRIERS, n_syms=1,
            len_tag_key=PACKET_LENGTH_TAG_KEY_RX,
            frame_len_tag_key=FRAME_LEN_TAG_KEY_RX,
            bits_per_header_sym=header_mod.bits_per_symbol(),
            bits_per_payload_sym=payload_mod.bits_per_symbol(),
            scramble_header=False,
        )
        self.header_parser = digital.packet_headerparser_b(header_formatter.base())

        self.src = osmosdr.source(args="numchan=1 " + hackrf_args)
        self.src.set_sample_rate(float(samp_rate))
        self.src.set_center_freq(CENTER_FREQ, 0)
        self.src.set_gain_mode(False, 0)
        self.src.set_gain(rf_gain, 0)
        self.src.set_if_gain(if_gain, 0)
        self.src.set_bb_gain(bb_gain, 0)

        self.sync = digital.ofdm_sync_sc_cfb(FFT_LEN, CP_LEN, False, alpha_sync)
        self.freq_mod = analog.frequency_modulator_fc((-2.0 / FFT_LEN))
        self.delay = blocks.delay(gr.sizeof_gr_complex, (FFT_LEN + CP_LEN))
        self.mult = blocks.multiply_vcc(1)

        self.awgn = channels.channel_model(
            noise_voltage=float(noise_voltage),
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0 + 0j],
        )

        self.iq_sink = None
        if record_iq:
            self.iq_sink = blocks.file_sink(gr.sizeof_gr_complex, RX_IQ_FILE, False)

        self.demux = digital.header_payload_demux(
            3, FFT_LEN, CP_LEN,
            FRAME_LEN_TAG_KEY_RX,
            "",
            True,
            gr.sizeof_gr_complex,
            "rx_time",
            float(samp_rate),
            (),
            0,
        )

        self.fft_hdr = fft.fft_vcc(FFT_LEN, True, (), True, 1)
        self.fft_pay = fft.fft_vcc(FFT_LEN, True, (), True, 1)

        estimator = estimator.lower()
        if estimator == "svr":
            self.chanest = ChannelEstimatorWrapperSVR()
        else:
            self.chanest = ChannelEstimatorWrapperLS()
        self.metric_probe = SyncMetricProbe(estimator_mode=estimator, fft_len=FFT_LEN)

        self.eq_hdr_obj = digital.ofdm_equalizer_simpledfe(FFT_LEN, header_mod.base(), OCCUPIED_CARRIERS, PILOT_CARRIERS, PILOT_SYMBOLS)
        self.eq_pay_obj = digital.ofdm_equalizer_simpledfe(FFT_LEN, payload_mod.base(), OCCUPIED_CARRIERS, PILOT_CARRIERS, PILOT_SYMBOLS, 1)

        self.eq_hdr = digital.ofdm_frame_equalizer_vcvc(self.eq_hdr_obj.base(), CP_LEN, FRAME_LEN_TAG_KEY_RX, True, 1)
        self.eq_pay = digital.ofdm_frame_equalizer_vcvc(self.eq_pay_obj.base(), CP_LEN, FRAME_LEN_TAG_KEY_RX, True, 0)

        self.ser_hdr = digital.ofdm_serializer_vcc(FFT_LEN, OCCUPIED_CARRIERS, FRAME_LEN_TAG_KEY_RX, '', 0, '', True)
        self.ser_pay = digital.ofdm_serializer_vcc(FFT_LEN, OCCUPIED_CARRIERS, FRAME_LEN_TAG_KEY_RX, PACKET_LENGTH_TAG_KEY_RX, 1, '', True)

        self.dec_hdr = digital.constellation_decoder_cb(header_mod.base())
        self.dec_pay = digital.constellation_decoder_cb(payload_mod.base())

        self.repack_pay = blocks.repack_bits_bb(payload_mod.bits_per_symbol(), 8, PACKET_LENGTH_TAG_KEY_RX, True, gr.GR_LSB_FIRST)
        self.use_crc = bool(use_crc)
        self.crc_check = digital.crc32_bb(True, PACKET_LENGTH_TAG_KEY_RX, True) if self.use_crc else None
        self.sink = blocks.file_sink(gr.sizeof_char, RX_PAYLOAD, False)

        self.connect(self.src, self.delay)
        self.connect(self.src, self.sync)
        self.connect((self.sync, 0), self.freq_mod)
        self.connect(self.freq_mod, (self.mult, 0))
        self.connect(self.delay, (self.mult, 1))

        if self.iq_sink is not None:
            self.connect(self.mult, self.iq_sink)

        self.connect(self.mult, self.awgn, (self.demux, 0))
        self.connect((self.sync, 1), (self.demux, 1))

        self.connect((self.demux, 0), self.fft_hdr)
        self.connect((self.demux, 1), self.fft_pay)

        # Main header path.
        self.connect(self.fft_hdr, self.chanest, self.eq_hdr, self.ser_hdr, self.dec_hdr, self.header_parser)
        # Passive metric probe in parallel with the actual estimator.
        self.connect(self.fft_hdr, self.metric_probe)

        self.msg_connect((self.header_parser, 'header_data'), (self.demux, 'header_data'))

        if self.use_crc:
            self.connect(self.fft_pay, self.eq_pay, self.ser_pay, self.dec_pay, self.repack_pay, self.crc_check, self.sink)
        else:
            self.connect(self.fft_pay, self.eq_pay, self.ser_pay, self.dec_pay, self.repack_pay, self.sink)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--noise_voltage", type=float, default=0.0)
    ap.add_argument("--record_iq", action="store_true")
    ap.add_argument("--tx_args", type=str, default=TX_HACKRF_DEFAULT)
    ap.add_argument("--rx_args", type=str, default=RX_HACKRF_DEFAULT)
    ap.add_argument("--samp_rate", type=float, default=SAMP_RATE_DEFAULT)
    ap.add_argument("--tx_rf_gain", type=float, default=14)
    ap.add_argument("--tx_if_gain", type=float, default=47)
    ap.add_argument("--tx_bb_gain", type=float, default=20)
    ap.add_argument("--tx_scale", type=float, default=0.07)
    ap.add_argument("--rx_rf_gain", type=float, default=7.7)
    ap.add_argument("--rx_if_gain", type=float, default=20)
    ap.add_argument("--rx_bb_gain", type=float, default=20)
    ap.add_argument("--alpha_sync", type=float, default=0.9)
    ap.add_argument("--use_crc", action="store_true")
    ap.add_argument("--write_known", action="store_true")
    ap.add_argument("--known_text", type=str, default="Hello World\n")
    ap.add_argument("--known_repeats", type=int, default=200)
    ap.add_argument("--use_benchmark", action="store_true")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--num_packets", type=int, default=200)
    ap.add_argument("--packet_len", type=int, default=PACKET_LEN_BYTES)
    ap.add_argument("--magic", type=str, default="OFDM")
    ap.add_argument("--estimator", type=str, default="ls", choices=["ls", "svr"])
    args = ap.parse_args()

    tx_bench_map = None
    magic4 = None
    if args.use_benchmark:
        magic4 = args.magic.encode("ascii", errors="strict")
        payload_bytes, tx_bench_map = generate_benchmark_payload(args.seed, args.num_packets, args.packet_len, magic4)
        with open(TX_PAYLOAD, "wb") as f:
            f.write(payload_bytes)
    else:
        ensure_payload_file()
        if args.write_known:
            pat = args.known_text.encode("utf-8")
            payload = pat * args.known_repeats
            with open(TX_PAYLOAD, "wb") as f:
                f.write(payload)

    if os.path.exists(RX_PAYLOAD):
        os.remove(RX_PAYLOAD)
    if args.record_iq and os.path.exists(RX_IQ_FILE):
        os.remove(RX_IQ_FILE)

    ok = True
    exc = None
    metric_stats = {"sync_mse": None, "sync_nmse": None, "sync_metric_frames": 0}
    t0 = time.time()
    try:
        rx = OFDMReceiver(
            args.rx_args, args.rx_rf_gain, args.rx_if_gain, args.rx_bb_gain,
            args.alpha_sync, args.noise_voltage, args.record_iq,
            samp_rate=args.samp_rate, use_crc=args.use_crc, estimator=args.estimator,
        )
        tx = OFDMTransmitter(
            args.tx_args, args.tx_rf_gain, args.tx_if_gain, args.tx_bb_gain,
            args.tx_scale, samp_rate=args.samp_rate, use_crc=args.use_crc,
        )
        rx.start()
        tx.start()
        time.sleep(args.seconds)
        tx.stop(); rx.stop()
        tx.wait(); rx.wait()
        metric_stats = rx.metric_probe.get_stats()
    except Exception:
        ok = False
        exc = traceback.format_exc()
    dt = time.time() - t0

    tx_bytes = open(TX_PAYLOAD, "rb").read(4096) if os.path.exists(TX_PAYLOAD) else b""
    rx_bytes = open(RX_PAYLOAD, "rb").read(4096) if os.path.exists(RX_PAYLOAD) else b""

    ber = compute_ber_bytes(tx_bytes, rx_bytes) if ok else None
    ber_al = compute_ber_aligned_files(TX_PAYLOAD, RX_PAYLOAD) if ok else None

    pat = args.known_text.encode("utf-8")
    pat_cnt = count_pattern_overlapping(rx_bytes, pat) if len(rx_bytes) > 0 else 0

    bench_metrics = None
    if ok and args.use_benchmark and tx_bench_map is not None and magic4 is not None and os.path.exists(RX_PAYLOAD):
        rx_full = open(RX_PAYLOAD, "rb").read()
        bench_metrics = compute_benchmark_packet_ber(tx_bench_map, rx_full, magic4, args.packet_len)

    out = {
        "ok": ok,
        "estimator": args.estimator,
        "seconds": args.seconds,
        "noise_voltage": args.noise_voltage,
        "samp_rate": float(args.samp_rate),
        "use_crc": bool(args.use_crc),
        "use_benchmark": bool(args.use_benchmark),
        "benchmark_seed": int(args.seed) if args.use_benchmark else None,
        "benchmark_num_packets": int(args.num_packets) if args.use_benchmark else None,
        "benchmark_packet_len": int(args.packet_len) if args.use_benchmark else None,
        "benchmark_magic": args.magic if args.use_benchmark else None,
        "dt": dt,
        "tx_size": os.path.getsize(TX_PAYLOAD) if os.path.exists(TX_PAYLOAD) else 0,
        "rx_size": os.path.getsize(RX_PAYLOAD) if os.path.exists(RX_PAYLOAD) else 0,
        "tx_head_hex": tx_bytes[:64].hex(),
        "rx_head_hex": rx_bytes[:64].hex(),
        "known_pattern_count_in_rx_head": pat_cnt,
        "ber_raw": None if ber is None else float(ber),
        "ber_aligned": None if (ber_al is None or ber_al.get("ber_aligned") is None) else float(ber_al["ber_aligned"]),
        "offset": None if ber_al is None else int(ber_al["offset"]),
        "score": None if ber_al is None else int(ber_al["score"]),
        "rx_packets_parsed": None if bench_metrics is None else int(bench_metrics["rx_packets_parsed"]),
        "matched_packets": None if bench_metrics is None else int(bench_metrics["matched_packets"]),
        "missing_packets": None if bench_metrics is None else int(bench_metrics["missing_packets"]),
        "ber_packets": None if bench_metrics is None else bench_metrics["ber_packets"],
        "sync_mse": metric_stats["sync_mse"],
        "sync_nmse": metric_stats["sync_nmse"],
        "sync_metric_frames": metric_stats["sync_metric_frames"],
        "exception": exc,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
