
# Auto-generated runner for OFDM TX/RX (HackRF)
import os, time, json, argparse, traceback
import numpy as np
from gnuradio import gr, blocks, digital, fft, analog, channels
import pmt
import osmosdr

SAMP_RATE = 1800000
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

SYNC_WORD1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.41421356, 0.0, 1.41421356, 0.0, -1.41421356, 0.0, 1.41421356, 0.0, -1.41421356, 0.0, 1.41421356, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
SYNC_WORD2 = [0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, (1+0j), (1+0j), (1+0j), (-1+0j), (1+0j), (1+0j), 0j, (1+0j), (-1+0j), (1+0j), (1+0j), (1+0j), (-1+0j), 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j]

WORKDIR = os.getcwd()
TX_PAYLOAD = os.path.join(WORKDIR, "tx_payload.bin")
RX_PAYLOAD = os.path.join(WORKDIR, "rx_payload.bin")
RX_IQ_FILE = os.path.join(WORKDIR, "rx_iq.cfile")

TX_HACKRF_DEFAULT = "hackrf=0"
RX_HACKRF_DEFAULT = "hackrf=1"

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
        cand = rx[i:i+window]
        score = sum(a == b for a, b in zip(tx_win, cand))
        if score > best_score:
            best_score = score
            best_i = i
    return best_i, int(best_score)

def compute_ber_aligned_files(tx_path, rx_path, max_bytes=4096, search_max_shift=4096, window=128):
    if not os.path.exists(tx_path) or not os.path.exists(rx_path):
        return None
    tx = open(tx_path,"rb").read(max_bytes)
    rx = open(rx_path,"rb").read(max_bytes)
    if len(tx)==0 or len(rx)==0:
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

class ChannelEstimatorWrapper(gr.hier_block2):
    def __init__(self):
        gr.hier_block2.__init__(
            self, "chanest_wrapper",
            gr.io_signature(1,1, gr.sizeof_gr_complex * FFT_LEN),
            gr.io_signature(1,1, gr.sizeof_gr_complex * FFT_LEN),
        )
        self._chanest = digital.ofdm_chanest_vcvc(SYNC_WORD1, SYNC_WORD2, 1, 0, 3, False)
        self.connect(self, self._chanest, self)

class OFDMTransmitter(gr.top_block):
    def __init__(self, hackrf_args, rf_gain, if_gain, bb_gain, scale):
        gr.top_block.__init__(self, "ofdm_tx_headless")
        payload_mod = digital.constellation_qpsk()
        header_mod  = digital.constellation_bpsk()
        hdr_format = digital.header_format_ofdm(OCCUPIED_CARRIERS, 1, LENGTH_TAG_KEY)

        self.file_source = blocks.file_source(gr.sizeof_char, TX_PAYLOAD, True)
        self.file_source.set_begin_tag(pmt.PMT_NIL)
        self.stream_to_tagged = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, PACKET_LEN_BYTES, LENGTH_TAG_KEY)
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
        self.cp   = digital.ofdm_cyclic_prefixer(FFT_LEN, FFT_LEN + CP_LEN, 0, LENGTH_TAG_KEY)
        self.scale = blocks.multiply_const_cc(scale)

        self.sink = osmosdr.sink(args="numchan=1 " + hackrf_args)
        self.sink.set_sample_rate(SAMP_RATE)
        self.sink.set_center_freq(CENTER_FREQ, 0)
        self.sink.set_gain(rf_gain, 0)
        self.sink.set_if_gain(if_gain, 0)
        self.sink.set_bb_gain(bb_gain, 0)

        self.connect(self.file_source, self.stream_to_tagged)
        self.connect(self.stream_to_tagged, self.protocol_formatter, self.repack_hdr, self.map_hdr, (self.mux, 0))
        self.connect(self.stream_to_tagged, self.repack_pay, self.map_pay, (self.mux, 1))
        self.connect(self.mux, self.carrier_alloc, self.ifft, self.cp, self.scale, self.sink)

class OFDMReceiver(gr.top_block):
    def __init__(self, hackrf_args, rf_gain, if_gain, bb_gain, alpha_sync, noise_voltage, record_iq):
        gr.top_block.__init__(self, "ofdm_rx_headless")
        payload_mod = digital.constellation_qpsk()
        header_mod  = digital.constellation_bpsk()

        header_formatter = digital.packet_header_ofdm(
            OCCUPIED_CARRIERS, n_syms=1,
            len_tag_key=PACKET_LENGTH_TAG_KEY_RX,
            frame_len_tag_key=FRAME_LEN_TAG_KEY_RX,
            bits_per_header_sym=header_mod.bits_per_symbol(),
            bits_per_payload_sym=payload_mod.bits_per_symbol(),
            scramble_header=False
        )
        self.header_parser = digital.packet_headerparser_b(header_formatter.base())

        self.src = osmosdr.source(args="numchan=1 " + hackrf_args)
        self.src.set_sample_rate(SAMP_RATE)
        self.src.set_center_freq(CENTER_FREQ, 0)
        self.src.set_gain_mode(False, 0)
        self.src.set_gain(rf_gain, 0)
        self.src.set_if_gain(if_gain, 0)
        self.src.set_bb_gain(bb_gain, 0)

        self.sync = digital.ofdm_sync_sc_cfb(FFT_LEN, CP_LEN, False, alpha_sync)
        self.freq_mod = analog.frequency_modulator_fc((-2.0/FFT_LEN))
        self.delay = blocks.delay(gr.sizeof_gr_complex, (FFT_LEN + CP_LEN))
        self.mult = blocks.multiply_vcc(1)

        self.awgn = channels.channel_model(
            noise_voltage=float(noise_voltage),
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0+0j]
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
            SAMP_RATE,
            (),
            0
        )

        self.fft_hdr = fft.fft_vcc(FFT_LEN, True, (), True, 1)
        self.fft_pay = fft.fft_vcc(FFT_LEN, True, (), True, 1)

        self.chanest = ChannelEstimatorWrapper()

        self.eq_hdr_obj = digital.ofdm_equalizer_simpledfe(FFT_LEN, header_mod.base(), OCCUPIED_CARRIERS, PILOT_CARRIERS, PILOT_SYMBOLS)
        self.eq_pay_obj = digital.ofdm_equalizer_simpledfe(FFT_LEN, payload_mod.base(), OCCUPIED_CARRIERS, PILOT_CARRIERS, PILOT_SYMBOLS, 1)

        self.eq_hdr = digital.ofdm_frame_equalizer_vcvc(self.eq_hdr_obj.base(), CP_LEN, FRAME_LEN_TAG_KEY_RX, True, 1)
        self.eq_pay = digital.ofdm_frame_equalizer_vcvc(self.eq_pay_obj.base(), CP_LEN, FRAME_LEN_TAG_KEY_RX, True, 0)

        self.ser_hdr = digital.ofdm_serializer_vcc(FFT_LEN, OCCUPIED_CARRIERS, FRAME_LEN_TAG_KEY_RX, '', 0, '', True)
        self.ser_pay = digital.ofdm_serializer_vcc(FFT_LEN, OCCUPIED_CARRIERS, FRAME_LEN_TAG_KEY_RX, PACKET_LENGTH_TAG_KEY_RX, 1, '', True)

        self.dec_hdr = digital.constellation_decoder_cb(header_mod.base())
        self.dec_pay = digital.constellation_decoder_cb(payload_mod.base())

        self.repack_pay = blocks.repack_bits_bb(payload_mod.bits_per_symbol(), 8, PACKET_LENGTH_TAG_KEY_RX, True, gr.GR_LSB_FIRST)
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

        self.connect(self.fft_hdr, self.chanest, self.eq_hdr, self.ser_hdr, self.dec_hdr, self.header_parser)
        self.msg_connect((self.header_parser, 'header_data'), (self.demux, 'header_data'))

        self.connect(self.fft_pay, self.eq_pay, self.ser_pay, self.dec_pay, self.repack_pay, self.sink)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--noise_voltage", type=float, default=0.0)
    ap.add_argument("--record_iq", action="store_true")
    ap.add_argument("--tx_args", type=str, default=TX_HACKRF_DEFAULT)
    ap.add_argument("--rx_args", type=str, default=RX_HACKRF_DEFAULT)
    ap.add_argument("--tx_rf_gain", type=float, default=14)
    ap.add_argument("--tx_if_gain", type=float, default=47)
    ap.add_argument("--tx_bb_gain", type=float, default=20)
    ap.add_argument("--tx_scale", type=float, default=0.07)
    ap.add_argument("--rx_rf_gain", type=float, default=7.7)
    ap.add_argument("--rx_if_gain", type=float, default=20)
    ap.add_argument("--rx_bb_gain", type=float, default=20)
    ap.add_argument("--alpha_sync", type=float, default=0.9)
    ap.add_argument("--write_known", action="store_true")
    ap.add_argument("--known_text", type=str, default="Hello World\\n")
    ap.add_argument("--known_repeats", type=int, default=200)
    args = ap.parse_args()

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
    t0 = time.time()
    try:
        rx = OFDMReceiver(args.rx_args, args.rx_rf_gain, args.rx_if_gain, args.rx_bb_gain,
                          args.alpha_sync, args.noise_voltage, args.record_iq)
        tx = OFDMTransmitter(args.tx_args, args.tx_rf_gain, args.tx_if_gain, args.tx_bb_gain, args.tx_scale)
        rx.start()
        tx.start()
        time.sleep(args.seconds)
        tx.stop(); rx.stop()
        tx.wait(); rx.wait()
    except Exception:
        ok = False
        exc = traceback.format_exc()
    dt = time.time() - t0

    tx_bytes = open(TX_PAYLOAD,"rb").read(4096) if os.path.exists(TX_PAYLOAD) else b""
    rx_bytes = open(RX_PAYLOAD,"rb").read(4096) if os.path.exists(RX_PAYLOAD) else b""

    ber = compute_ber_bytes(tx_bytes, rx_bytes) if ok else None
    ber_al = compute_ber_aligned_files(TX_PAYLOAD, RX_PAYLOAD) if ok else None

    pat = args.known_text.encode("utf-8")
    pat_cnt = count_pattern_overlapping(rx_bytes, pat) if len(rx_bytes)>0 else 0

    out = {
        "ok": ok,
        "seconds": args.seconds,
        "noise_voltage": args.noise_voltage,
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
        "exception": exc,
    }
    print(json.dumps(out))

if __name__ == "__main__":
    main()
