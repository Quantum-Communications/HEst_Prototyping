#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import osmosdr
import pmt
from gnuradio import analog, blocks, channels, digital, fft, gr

from ofdm_config import *
from ofdm_utils import *
from ofdm_metrics import *
from ofdm_blocks import *
from ofdm_estimators import *

# Transmitter
# =============================================================================

class OFDMTransmitter(gr.top_block):
    """
    OFDM transmitter with optional CRC and optional TX-side AWGN.

    This is the clean baseline TX:
    - it frames bytes into OFDM packets
    - optionally adds CRC
    - optionally adds AWGN right before the SDR sink
    """

    def __init__(
        self,
        payload_path: str,
        tx_args: str,
        sample_rate: float,
        center_frequency: float,
        rf_gain: float,
        if_gain: float,
        bb_gain: float,
        tx_scale: float,
        use_crc: bool,
        enable_noise: bool,
        snr_db: float,
    ) -> None:
        super().__init__("ofdm_tx_integrated")

        # Header uses BPSK because it must be as robust as possible.
        header_constellation = digital.constellation_bpsk()

        # Payload uses QPSK to provide higher throughput than BPSK.
        payload_constellation = digital.constellation_qpsk()

        # Header format must match the receiver parser exactly.
        header_format = digital.header_format_ofdm(
            OCCUPIED_CARRIERS,
            1,
            LENGTH_TAG_KEY,
            FRAME_LENGTH_TAG_KEY,
            PACKET_NUMBER_TAG_KEY,
            header_constellation.bits_per_symbol(),
            payload_constellation.bits_per_symbol(),
            False,
        )

        # Repeating byte source from file.
        # The "repeat=True" behavior makes the TX continuously loop the payload.
        self.file_source = blocks.file_source(gr.sizeof_char, payload_path, True)
        self.file_source.set_begin_tag(pmt.PMT_NIL)

        # Group raw bytes into fixed-size tagged packets.
        self.stream_to_tagged = blocks.stream_to_tagged_stream(
            gr.sizeof_char,
            1,
            PACKET_LENGTH_BYTES,
            LENGTH_TAG_KEY,
        )

        # Count how many logical packets enter the TX framing pipeline.
        self.packet_counter = TaggedPacketCounter(LENGTH_TAG_KEY)

        # Optional CRC insertion before OFDM framing.
        self.crc = digital.crc32_bb(False, LENGTH_TAG_KEY, True) if use_crc else None

        # This is the byte stream that feeds the OFDM header/payload framing.
        source_for_framing = self.crc if self.crc is not None else self.packet_counter

        # Build header bits from packet tags.
        self.protocol_formatter = digital.protocol_formatter_bb(
            header_format,
            LENGTH_TAG_KEY,
        )

        # Convert bytes to header symbol indices.
        self.repack_header = blocks.repack_bits_bb(
            8,
            header_constellation.bits_per_symbol(),
            LENGTH_TAG_KEY,
            False,
            gr.GR_LSB_FIRST,
        )

        # Convert bytes to payload symbol indices.
        self.repack_payload = blocks.repack_bits_bb(
            8,
            payload_constellation.bits_per_symbol(),
            LENGTH_TAG_KEY,
            False,
            gr.GR_LSB_FIRST,
        )

        # Map indices to constellation points.
        self.map_header = digital.chunks_to_symbols_bc(header_constellation.points(), 1)
        self.map_payload = digital.chunks_to_symbols_bc(payload_constellation.points(), 1)

        # Merge header and payload streams.
        self.tagged_mux = blocks.tagged_stream_mux(gr.sizeof_gr_complex, LENGTH_TAG_KEY, 0)

        # OFDM carrier allocation and preamble insertion.
        self.carrier_allocator = digital.ofdm_carrier_allocator_cvc(
            FFT_LENGTH,
            OCCUPIED_CARRIERS,
            PILOT_CARRIERS,
            PILOT_SYMBOLS,
            (SYNC_WORD1, SYNC_WORD2),
            LENGTH_TAG_KEY,
            True,
        )

        # Convert frequency-domain OFDM vectors to time-domain samples.
        self.ifft = fft.fft_vcc(FFT_LENGTH, False, (), True, 1)

        # Add cyclic prefix.
        self.cyclic_prefixer = digital.ofdm_cyclic_prefixer(
            FFT_LENGTH,
            FFT_LENGTH + CP_LENGTH,
            0,
            LENGTH_TAG_KEY,
        )

        # Final digital amplitude scaling before transmission.
        self.scale = blocks.multiply_const_cc(float(tx_scale))

        # Optional AWGN injected at TX output.
        self.noise_voltage = 0.0
        self.awgn = None

        if enable_noise:
            self.noise_voltage = snr_db_to_noise_voltage(snr_db)
            self.awgn = channels.channel_model(
                noise_voltage=float(self.noise_voltage),
                frequency_offset=0.0,
                epsilon=1.0,
                taps=[1.0 + 0j],
            )

        # RF sink.
        self.sink = osmosdr.sink(args="numchan=1 " + tx_args)
        self.sink.set_sample_rate(float(sample_rate))
        self.sink.set_center_freq(float(center_frequency), 0)
        self.sink.set_gain(float(rf_gain), 0)
        self.sink.set_if_gain(float(if_gain), 0)
        self.sink.set_bb_gain(float(bb_gain), 0)

        # Byte source -> packetizer -> packet counter.
        self.connect(self.file_source, self.stream_to_tagged, self.packet_counter)

        # Optional CRC insertion.
        if self.crc is not None:
            self.connect(self.packet_counter, self.crc)

        # Header path.
        self.connect(
            source_for_framing,
            self.protocol_formatter,
            self.repack_header,
            self.map_header,
            (self.tagged_mux, 0),
        )

        # Payload path.
        self.connect(
            source_for_framing,
            self.repack_payload,
            self.map_payload,
            (self.tagged_mux, 1),
        )

        # Shared OFDM modulation chain.
        self.connect(
            self.tagged_mux,
            self.carrier_allocator,
            self.ifft,
            self.cyclic_prefixer,
            self.scale,
        )

        # Final stage: direct or via AWGN.
        if self.awgn is not None:
            self.connect(self.scale, self.awgn, self.sink)
        else:
            self.connect(self.scale, self.sink)


# =============================================================================
# Receiver
# =============================================================================

class OFDMReceiver(gr.top_block):
    """
    OFDM receiver using GNU Radio native LS channel estimation.

    This is the clean baseline RX:
    - synchronize in time/frequency
    - demux header and payload
    - use native LS channel estimation
    - decode payload bytes
    - optionally verify CRC
    - write output bytes to a file
    """

    def __init__(
        self,
        rx_output_path: str,
        rx_args: str,
        sample_rate: float,
        center_frequency: float,
        rf_gain: float,
        if_gain: float,
        bb_gain: float,
        alpha_sync: float,
        use_crc: bool,
        channel_estimator_mode: str,
        freeze_after_frames: int,
        record_channel_taps: bool,
        channel_taps_output: str,
        svr_c: float,
        svr_gamma: str,
        svr_epsilon: float,
        svr_window_frames: int,
        svr_min_train_frames: int,
    ) -> None:
        super().__init__("ofdm_rx_integrated")

        # Header uses BPSK.
        header_constellation = digital.constellation_bpsk()

        # Payload uses QPSK.
        payload_constellation = digital.constellation_qpsk()

        # Header parser configuration must match TX header builder exactly.
        header_formatter = digital.packet_header_ofdm(
            OCCUPIED_CARRIERS,
            n_syms=1,
            len_tag_key=PACKET_LENGTH_TAG_KEY_RX,
            frame_len_tag_key=FRAME_LENGTH_TAG_KEY,
            bits_per_header_sym=header_constellation.bits_per_symbol(),
            bits_per_payload_sym=payload_constellation.bits_per_symbol(),
            scramble_header=False,
        )

        self.header_parser = digital.packet_headerparser_b(header_formatter.base())

        # RF source.
        self.source = osmosdr.source(args="numchan=1 " + rx_args)
        self.source.set_sample_rate(float(sample_rate))
        self.source.set_center_freq(float(center_frequency), 0)
        self.source.set_gain_mode(False, 0)
        self.source.set_gain(float(rf_gain), 0)
        self.source.set_if_gain(float(if_gain), 0)
        self.source.set_bb_gain(float(bb_gain), 0)
        self.source.set_max_output_buffer(SOURCE_MAX_BUFFER)

        # OFDM synchronizer.
        # Output 0: coarse frequency estimate
        # Output 1: trigger / timing information
        self.sync = digital.ofdm_sync_sc_cfb(
            FFT_LENGTH,
            CP_LENGTH,
            False,
            float(alpha_sync),
        )

        # Coarse CFO correction:
        # - turn estimated phase increment into a complex phasor
        # - multiply with delayed input stream
        self.frequency_modulator = analog.frequency_modulator_fc(-2.0 / FFT_LENGTH)
        self.delay = blocks.delay(gr.sizeof_gr_complex, FFT_LENGTH + CP_LENGTH)
        self.multiply = blocks.multiply_vcc(1)

        self.delay.set_min_output_buffer(GENERAL_MIN_BUFFER)
        self.multiply.set_min_output_buffer(GENERAL_MIN_BUFFER)

        # Header / payload demultiplexer.
        self.demux = digital.header_payload_demux(
            3,
            FFT_LENGTH,
            CP_LENGTH,
            FRAME_LENGTH_TAG_KEY,
            "",
            True,
            gr.sizeof_gr_complex,
            "rx_time",
            float(sample_rate),
            (),
            0,
        )
        self.demux.set_min_output_buffer(GENERAL_MIN_BUFFER)

        # FFT blocks for header and payload paths.
        self.fft_header = None
        if channel_estimator_mode not in (
            "svr_time_domain",
            "svr_frozen_async",
            "svr_frozen_async_precorrected",
        ):
            self.fft_header = fft.fft_vcc(FFT_LENGTH, True, (), True, 1)
            self.fft_header.set_min_output_buffer(GENERAL_MIN_BUFFER)

        self.fft_payload = fft.fft_vcc(FFT_LENGTH, True, (), True, 1)
        self.fft_payload.set_min_output_buffer(GENERAL_MIN_BUFFER)

        # Native GNU Radio LS channel estimator.
        if channel_estimator_mode == "native_ls":
            self.channel_estimator = digital.ofdm_chanest_vcvc(
                SYNC_WORD1,
                SYNC_WORD2,
                1,
                0,
                3,
                False,
            )
        elif channel_estimator_mode == "python_ls":
            self.channel_estimator = PythonNativeLSChanEst(
                SYNC_WORD1,
                SYNC_WORD2,
                n_data_symbols=1,
                max_carr_offset=3,
            )
        elif channel_estimator_mode == "frozen_ls_nframes":
            self.channel_estimator = FrozenAfterNFramesChanEst(
                SYNC_WORD1,
                SYNC_WORD2,
                n_data_symbols=1,
                max_carr_offset=3,
                freeze_after_frames=freeze_after_frames,
            )
        elif channel_estimator_mode == "svr_time_domain":
            self.channel_estimator = PythonSVRTimeDomainChanEst(
                SYNC_WORD1,
                SYNC_WORD2,
                n_data_symbols=1,
                max_carr_offset=3,
                svr_c=svr_c,
                svr_gamma=svr_gamma,
                svr_epsilon=svr_epsilon,
            )
        elif channel_estimator_mode == "svr_frozen_async":
            self.channel_estimator = PythonSVRFrozenAsyncChanEst(
                SYNC_WORD1,
                SYNC_WORD2,
                n_data_symbols=1,
                max_carr_offset=3,
                svr_c=svr_c,
                svr_gamma=svr_gamma,
                svr_epsilon=svr_epsilon,
                svr_window_frames=svr_window_frames,
                svr_min_train_frames=svr_min_train_frames,
            )
        elif channel_estimator_mode == "svr_frozen_async_precorrected":
            self.channel_estimator = PythonSVRFrozenAsyncPrecorrectedChanEst(
                SYNC_WORD1,
                SYNC_WORD2,
                n_data_symbols=1,
                max_carr_offset=3,
                svr_c=svr_c,
                svr_gamma=svr_gamma,
                svr_epsilon=svr_epsilon,
                svr_window_frames=svr_window_frames,
                svr_min_train_frames=svr_min_train_frames,
            )
        elif channel_estimator_mode == "svr_freq_async":
            self.channel_estimator = PythonSVRFrequencyAsyncChanEst(
                SYNC_WORD1,
                SYNC_WORD2,
                n_data_symbols=1,
                max_carr_offset=3,
                svr_c=svr_c,
                svr_gamma=svr_gamma,
                svr_epsilon=svr_epsilon,
                svr_window_frames=svr_window_frames,
                svr_min_train_frames=svr_min_train_frames,
            )
        else:
            raise ValueError(f"Unsupported channel_estimator_mode: {channel_estimator_mode}")
        
        self.tap_recorder = (
            ChannelTapRecorder(channel_taps_output) if record_channel_taps else None
        )

        # Header equalizer object.
        self.header_equalizer_object = digital.ofdm_equalizer_simpledfe(
            FFT_LENGTH,
            header_constellation.base(),
            OCCUPIED_CARRIERS,
            PILOT_CARRIERS,
            PILOT_SYMBOLS,
        )

        # Payload equalizer object.
        self.payload_equalizer_object = digital.ofdm_equalizer_simpledfe(
            FFT_LENGTH,
            payload_constellation.base(),
            OCCUPIED_CARRIERS,
            PILOT_CARRIERS,
            PILOT_SYMBOLS,
            1,
        )

        # Equalizer blocks.
        self.header_equalizer = digital.ofdm_frame_equalizer_vcvc(
            self.header_equalizer_object.base(),
            CP_LENGTH,
            FRAME_LENGTH_TAG_KEY,
            True,
            1,
        )

        self.payload_equalizer = digital.ofdm_frame_equalizer_vcvc(
            self.payload_equalizer_object.base(),
            CP_LENGTH,
            FRAME_LENGTH_TAG_KEY,
            True,
            0,
        )

        # Equalizers are among the most buffer-sensitive OFDM blocks.
        self.header_equalizer.set_min_output_buffer(CRITICAL_MIN_BUFFER)
        self.payload_equalizer.set_min_output_buffer(CRITICAL_MIN_BUFFER)

        # Serializers convert OFDM vectors to symbol streams.
        self.header_serializer = digital.ofdm_serializer_vcc(
            FFT_LENGTH,
            OCCUPIED_CARRIERS,
            FRAME_LENGTH_TAG_KEY,
            "",
            0,
            "",
            True,
        )

        self.payload_serializer = digital.ofdm_serializer_vcc(
            FFT_LENGTH,
            OCCUPIED_CARRIERS,
            FRAME_LENGTH_TAG_KEY,
            PACKET_LENGTH_TAG_KEY_RX,
            1,
            "",
            True,
        )

        self.header_serializer.set_min_output_buffer(CRITICAL_MIN_BUFFER)
        self.payload_serializer.set_min_output_buffer(CRITICAL_MIN_BUFFER)

        # Constellation decoders.
        self.header_decoder = digital.constellation_decoder_cb(header_constellation.base())
        self.payload_decoder = digital.constellation_decoder_cb(payload_constellation.base())

        # Convert payload bits back to bytes.
        self.repack_payload = blocks.repack_bits_bb(
            payload_constellation.bits_per_symbol(),
            8,
            PACKET_LENGTH_TAG_KEY_RX,
            True,
            gr.GR_LSB_FIRST,
        )

        # Optional CRC verification.
        self.crc_check = digital.crc32_bb(True, PACKET_LENGTH_TAG_KEY_RX, True) if use_crc else None

        # RX output file.
        ensure_parent_directory(rx_output_path)
        self.file_sink = blocks.file_sink(gr.sizeof_char, rx_output_path, False)

        # Coarse synchronization / CFO correction chain.
        self.connect(self.source, self.delay)
        self.connect(self.source, self.sync)
        self.connect((self.sync, 0), self.frequency_modulator)
        self.connect(self.frequency_modulator, (self.multiply, 0))
        self.connect(self.delay, (self.multiply, 1))

        # Feed corrected stream and trigger into demux.
        self.connect(self.multiply, (self.demux, 0))
        self.connect((self.sync, 1), (self.demux, 1))

        # FFT paths.
        self.connect((self.demux, 1), self.fft_payload)

        # Header path:
        # FFT -> native LS channel estimator -> equalizer -> serializer ->
        # constellation decoder -> header parser
        # SVR recebe tempo direto do demux
        if channel_estimator_mode in ("svr_time_domain", "svr_frozen_async", "svr_frozen_async_precorrected"):
            self.connect((self.demux, 0), self.channel_estimator)

        # LS/native/python/frozen recebem FFT antes
        else:
            self.connect((self.demux, 0), self.fft_header)
            self.connect(self.fft_header, self.channel_estimator)

        self.connect(self.channel_estimator, self.header_equalizer)
        self.connect(
            self.header_equalizer,
            self.header_serializer,
            self.header_decoder,
            self.header_parser,
        )

        if self.tap_recorder is not None:
            self.connect(self.channel_estimator, self.tap_recorder)

        # Feed parsed header information back into the demux.
        self.msg_connect((self.header_parser, "header_data"), (self.demux, "header_data"))

        # Payload path.
        if self.crc_check is not None:
            self.connect(
                self.fft_payload,
                self.payload_equalizer,
                self.payload_serializer,
                self.payload_decoder,
                self.repack_payload,
                self.crc_check,
                self.file_sink,
            )
        else:
            self.connect(
                self.fft_payload,
                self.payload_equalizer,
                self.payload_serializer,
                self.payload_decoder,
                self.repack_payload,
                self.file_sink,
            )


# =============================================================================
# Command-line interface
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI for the integrated experiment.

    Even though TX and RX are orchestrated together, all important parameters
    remain visible as explicit command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Integrated baseline OFDM experiment with simultaneous TX and RX orchestration."
    )

    # Payload generation / TX input file.
    parser.add_argument(
        "--payload",
        type=str,
        default="tx_payload.bin",
        help="TX payload file. If it does not exist, it will be generated automatically.",
    )
    parser.add_argument(
        "--payload_bytes",
        type=int,
        default=4096,
        help="Number of bytes to generate if the TX payload file does not exist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed used for deterministic TX payload generation.",
    )

    # Optional magic-based validation packetization.
    parser.add_argument(
        "--use_magic_validation",
        action="store_true",
        help="Generate the TX payload as fixed-size packets with a magic word and validate RX output at the end.",
    )
    parser.add_argument(
        "--magic",
        type=str,
        default="OFDM",
        help="4-character ASCII magic word used for packet validation.",
    )
    parser.add_argument(
        "--num_packets",
        type=int,
        default=200,
        help="Number of packets to generate when magic validation is enabled.",
    )
    parser.add_argument(
        "--packet_len",
        type=int,
        default=PACKET_LENGTH_BYTES,
        help="Packet length in bytes when magic validation is enabled.",
    )

    # RX output file.
    parser.add_argument(
        "--rx_output",
        type=str,
        default="rx_payload.bin",
        help="Output file for received payload bytes.",
    )

    # Shared RF timing parameters.
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=SAMPLE_RATE_DEFAULT,
        help="Baseband sample rate in samples per second.",
    )
    parser.add_argument(
        "--center_frequency",
        type=float,
        default=CENTER_FREQUENCY_DEFAULT,
        help="RF center frequency in Hz.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=6.0,
        help="Experiment runtime in seconds.",
    )

    # TX device settings.
    parser.add_argument(
        "--tx_args",
        type=str,
        default=TX_ARGS_DEFAULT,
        help='osmosdr device string for TX, for example "hackrf=0".',
    )
    parser.add_argument(
        "--tx_rf_gain",
        type=float,
        default=14.0,
        help="HackRF TX RF gain.",
    )
    parser.add_argument(
        "--tx_if_gain",
        type=float,
        default=47.0,
        help="HackRF TX IF gain.",
    )
    parser.add_argument(
        "--tx_bb_gain",
        type=float,
        default=20.0,
        help="HackRF TX baseband gain.",
    )
    parser.add_argument(
        "--tx_scale",
        type=float,
        default=0.07,
        help="Final digital scaling applied to the OFDM waveform before TX.",
    )

    # RX device settings.
    parser.add_argument(
        "--rx_args",
        type=str,
        default=RX_ARGS_DEFAULT,
        help='osmosdr device string for RX, for example "hackrf=1".',
    )
    parser.add_argument(
        "--rx_rf_gain",
        type=float,
        default=7.7,
        help="HackRF RX RF gain.",
    )
    parser.add_argument(
        "--rx_if_gain",
        type=float,
        default=20.0,
        help="HackRF RX IF gain.",
    )
    parser.add_argument(
        "--rx_bb_gain",
        type=float,
        default=20.0,
        help="HackRF RX baseband gain.",
    )
    parser.add_argument(
        "--alpha_sync",
        type=float,
        default=0.9,
        help="OFDM synchronizer smoothing parameter on RX.",
    )

    # Optional CRC support.
    parser.add_argument(
        "--use_crc",
        action="store_true",
        help="Enable CRC insertion in TX and CRC verification in RX.",
    )

    # Optional TX-side AWGN.
    parser.add_argument(
        "--enable_noise",
        action="store_true",
        help="Enable TX-side AWGN insertion before the HackRF sink.",
    )
    parser.add_argument(
        "--snrdb",
        type=float,
        default=50.0,
        help="Desired TX-side SNR in dB when noise is enabled.",
    )
    parser.add_argument(
        "--channel_estimator_mode",
        type=str,
        default="native_ls",
        choices=[
            "native_ls",
            "python_ls",
            "frozen_ls_nframes",
            "svr_time_domain",
            "svr_frozen_async",
            "svr_frozen_async_precorrected",
            "svr_freq_async",
        ],
        help="Channel estimator mode for the RX header path.",
    )

    parser.add_argument(
        "--record_channel_taps",
        action="store_true",
        help="Record channel-estimation tags (chan_taps and carr_offset) to an NPZ file.",
    )
    parser.add_argument(
        "--channel_taps_output",
        type=str,
        default="channel_taps.npz",
        help="Output NPZ path for recorded channel taps.",
    )
    parser.add_argument(
        "--freeze_after_frames",
        type=int,
        default=2,
        help="For frozen channel estimation, estimate normally until this many frames have been seen, then freeze the taps.",
    )

    parser.add_argument(
        "--svr_c",
        type=float,
        default=100.0,
        help="SVR C parameter for time-domain SVR channel estimation.",
    )
    parser.add_argument(
        "--svr_gamma",
        type=str,
        default="scale",
        help='SVR gamma parameter, e.g. "scale", "auto", or a float-like string.',
    )
    parser.add_argument(
        "--svr_epsilon",
        type=float,
        default=0.001,
        help="SVR epsilon parameter for time-domain SVR channel estimation.",
    )

    parser.add_argument(
        "--svr_window_frames",
        type=int,
        default=20,
        help="Number of recent sync frames kept in the async SVR sliding training window.",
    )
    parser.add_argument(
        "--svr_min_train_frames",
        type=int,
        default=5,
        help="Minimum number of accumulated sync frames required before async SVR training starts.",
    )

    return parser


# =============================================================================
# Main orchestration
# =============================================================================

def main() -> int:
    args = build_parser().parse_args()

    # Build TX payload.
    # Two modes are supported:
    #   1) plain deterministic byte file
    #   2) packetized file with magic-word validation
    tx_map = None

    if args.use_magic_validation:
        magic4 = args.magic.encode("ascii", errors="strict")
        if len(magic4) != 4:
            raise ValueError("--magic must be exactly 4 ASCII characters")

        tx_map = generate_magic_payload_file(
            path=args.payload,
            seed=args.seed,
            num_packets=args.num_packets,
            packet_len=args.packet_len,
            magic4=magic4,
        )
    else:
        ensure_payload_file(args.payload, args.payload_bytes, args.seed)

    # Remove previous RX output to keep each run clean.
    remove_if_exists(args.rx_output)

    # Build receiver first so it is ready before TX starts.
    rx = OFDMReceiver(
        rx_output_path=args.rx_output,
        rx_args=args.rx_args,
        sample_rate=args.sample_rate,
        center_frequency=args.center_frequency,
        rf_gain=args.rx_rf_gain,
        if_gain=args.rx_if_gain,
        bb_gain=args.rx_bb_gain,
        alpha_sync=args.alpha_sync,
        use_crc=bool(args.use_crc),
        channel_estimator_mode=args.channel_estimator_mode,
        freeze_after_frames=args.freeze_after_frames,
        record_channel_taps=bool(args.record_channel_taps),
        channel_taps_output=args.channel_taps_output,
        svr_c=args.svr_c,
        svr_gamma=args.svr_gamma,
        svr_epsilon=args.svr_epsilon,
        svr_window_frames=args.svr_window_frames,
        svr_min_train_frames=args.svr_min_train_frames,
    )

    # Build transmitter second.
    tx = OFDMTransmitter(
        payload_path=args.payload,
        tx_args=args.tx_args,
        sample_rate=args.sample_rate,
        center_frequency=args.center_frequency,
        rf_gain=args.tx_rf_gain,
        if_gain=args.tx_if_gain,
        bb_gain=args.tx_bb_gain,
        tx_scale=args.tx_scale,
        use_crc=bool(args.use_crc),
        enable_noise=bool(args.enable_noise),
        snr_db=float(args.snrdb),
    )

    # Print a fully explicit configuration summary.
    print("=== Integrated OFDM Experiment ===")
    print(f"payload: {os.path.abspath(args.payload)}")
    print(f"payload_bytes: {args.payload_bytes}")
    print(f"seed: {args.seed}")
    print(f"use_magic_validation: {args.use_magic_validation}")
    print(f"magic: {args.magic if args.use_magic_validation else None}")
    print(f"num_packets: {args.num_packets if args.use_magic_validation else None}")
    print(f"packet_len: {args.packet_len if args.use_magic_validation else None}")
    print(f"rx_output: {os.path.abspath(args.rx_output)}")
    print(f"sample_rate: {args.sample_rate}")
    print(f"center_frequency: {args.center_frequency}")
    print(f"seconds: {args.seconds}")
    print("--- TX configuration ---")
    print(f"tx_args: {args.tx_args}")
    print(f"tx_rf_gain: {args.tx_rf_gain}")
    print(f"tx_if_gain: {args.tx_if_gain}")
    print(f"tx_bb_gain: {args.tx_bb_gain}")
    print(f"tx_scale: {args.tx_scale}")
    print("--- RX configuration ---")
    print(f"rx_args: {args.rx_args}")
    print(f"rx_rf_gain: {args.rx_rf_gain}")
    print(f"rx_if_gain: {args.rx_if_gain}")
    print(f"rx_bb_gain: {args.rx_bb_gain}")
    print(f"alpha_sync: {args.alpha_sync}")
    print("--- Common options ---")
    print(f"use_crc: {args.use_crc}")
    print(f"enable_noise: {args.enable_noise}")
    print(f"snrdb: {args.snrdb if args.enable_noise else None}")
    print(f"noise_voltage: {tx.noise_voltage if args.enable_noise else 0.0}")
    print("--- RX buffers ---")
    print(f"source_max_buffer: {SOURCE_MAX_BUFFER}")
    print(f"general_min_buffer: {GENERAL_MIN_BUFFER}")
    print(f"critical_min_buffer: {CRITICAL_MIN_BUFFER}")

    # Orchestrated experiment:
    # 1) Start RX first so it is already listening.
    # 2) Start TX second.
    # 3) Run for the requested duration.
    # 4) Stop TX first.
    # 5) Stop RX second.
    rx.start()
    tx.start()

    try:
        time.sleep(args.seconds)
    finally:
        tx.stop()
        rx.stop()
        tx.wait()
        rx.wait()

    # Read how many logical packets were emitted into the TX framing pipeline.
    tx_packets_emitted = int(tx.packet_counter.packet_count)

    # Minimal post-run validation using magic-word packets.
    if args.use_magic_validation:
        rx_bytes = b""
        if os.path.exists(args.rx_output):
            with open(args.rx_output, "rb") as f:
                rx_bytes = f.read()

        magic4 = args.magic.encode("ascii", errors="strict")
        metrics = compute_magic_packet_metrics(
            tx_map=tx_map if tx_map is not None else {},
            rx_bytes=rx_bytes,
            magic4=magic4,
            packet_len=args.packet_len,
            tx_packets_emitted=tx_packets_emitted,
        )

        print("--- Magic validation ---")
        print(f"magic: {args.magic}")
        print(f"num_packets: {args.num_packets}")
        print(f"packet_len: {args.packet_len}")
        print(f"tx_packets_emitted: {metrics['tx_packets_emitted']}")
        print(f"rx_packets_parsed: {metrics['rx_packets_parsed']}")
        print(f"matched_packets: {metrics['matched_packets']}")
        print(f"missing_packets: {metrics['missing_packets']}")

        print("--- Metrics ---")
        print(f"BER (conditional): {metrics['ber_magic']}")
        print(f"BER (global): {metrics['ber_global']}")
        print(f"PER: {metrics['per']}")
        print(f"Detection rate: {metrics['detection_rate']}")
        print(f"RX parse rate: {metrics['rx_parse_rate']}")

    if args.record_channel_taps and rx.tap_recorder is not None:
        rx.tap_recorder.save()

        chan_taps_array = np.asarray(rx.tap_recorder.chan_taps, dtype=np.complex64)
        variation_metrics = compute_channel_variation_metrics(chan_taps_array)

        print("--- Channel variation ---")
        print(f"recorded_frames: {variation_metrics['n_frames']}")
        print(f"tap_power_mean: {variation_metrics['tap_power_mean']}")
        print(f"tap_power_std: {variation_metrics['tap_power_std']}")
        print(f"tap_delta_mean: {variation_metrics['tap_delta_mean']}")
        print(f"tap_delta_std: {variation_metrics['tap_delta_std']}")
        print(f"tap_delta_rel_mean: {variation_metrics['tap_delta_rel_mean']}")
        print(f"tap_delta_rel_std: {variation_metrics['tap_delta_rel_std']}")
    
    if args.channel_estimator_mode in (
        "svr_frozen_async",
        "svr_frozen_async_precorrected",
        "svr_freq_async",
    ):
        print("--- SVR async worker ---")
        print(f"worker_frames_started: {rx.channel_estimator.worker_frames_started}")
        print(f"worker_frames_completed: {rx.channel_estimator.worker_frames_completed}")
        print(f"worker_batches_skipped: {rx.channel_estimator.worker_batches_skipped}")
        print(f"worker_last_batch_size: {rx.channel_estimator.worker_last_batch_size}")
        print(f"svr_window_frames: {args.svr_window_frames}")
        print(f"svr_min_train_frames: {args.svr_min_train_frames}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
