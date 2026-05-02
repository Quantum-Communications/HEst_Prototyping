from __future__ import annotations

"""Post-run validation and channel-tap variation metrics."""

import numpy as np

from ofdm_utils import parse_packets_by_magic

def compute_magic_packet_metrics(
    tx_map: dict[int, bytes],
    rx_bytes: bytes,
    magic4: bytes,
    packet_len: int,
    tx_packets_emitted: int | None = None,
) -> dict[str, float | int | None]:
    """
    Compute packet-level validation metrics using the magic-word method.

    Metrics returned
    ----------------
    tx_packets_emitted
        Number of logical packets entering the TX framing chain during the run.
    rx_packets_parsed
        Number of packet structures detected in RX by scanning for the magic word.
    matched_packets
        Number of known TX sequence numbers recovered in RX.
    missing_packets
        Number of TX sequence numbers not recovered in RX.
    ber_magic
        BER conditional on matched packets only.
    per
        Packet error rate over the reference packet set.
    detection_rate
        Parsed packet structures relative to reference set size.
    ber_global
        BER counting missing packets as total-bit errors.
    rx_parse_rate
        Parsed packet structures relative to TX packets emitted.

    Parameters
    ----------
    tx_map : dict[int, bytes]
        Reference TX map: sequence_number -> payload_bytes.
    rx_bytes : bytes
        Raw RX output file contents.
    magic4 : bytes
        4-byte magic signature.
    packet_len : int
        Total packet length in bytes.
    tx_packets_emitted : int | None
        Number of logical packets observed entering the TX framing pipeline.

    Returns
    -------
    dict[str, float | int | None]
        Dictionary of validation metrics.
    """
    rx_map, parsed_total = parse_packets_by_magic(rx_bytes, magic4, packet_len)

    matched = 0
    bit_errors = 0

    # Compare only packets whose sequence number was recovered in RX.
    for seq, tx_payload in tx_map.items():
        rx_payload = rx_map.get(seq)
        if rx_payload is None:
            continue

        matched += 1
        n = min(len(tx_payload), len(rx_payload))

        for a, b in zip(tx_payload[:n], rx_payload[:n]):
            bit_errors += (a ^ b).bit_count()

    total_packets = len(tx_map)
    payload_bits = (packet_len - 8) * 8
    missing = total_packets - matched

    # BER conditional:
    # Only over packets that were successfully identified and matched.
    ber_conditional = None if matched == 0 else bit_errors / float(max(1, matched * payload_bits))

    # PER:
    # Packet loss relative to the known reference set of packet IDs.
    per = missing / float(total_packets) if total_packets > 0 else None

    # Detection rate:
    # Number of packet structures parsed from RX divided by reference packet count.
    # Since TX loops the payload file during the run, this value can exceed 1.0.
    detection_rate = matched / float(total_packets) if total_packets > 0 else None

    # BER global:
    # Missing packets are counted as full-bit errors.
    total_bits = total_packets * payload_bits
    lost_bits = missing * payload_bits
    ber_global = (bit_errors + lost_bits) / float(total_bits) if total_bits > 0 else None

    # RX parse rate:
    # Number of parsed packet structures relative to how many logical packets
    # the TX emitted into the framing pipeline during this run.
    rx_parse_rate = (
        parsed_total / float(tx_packets_emitted)
        if tx_packets_emitted is not None and tx_packets_emitted > 0
        else None
    )

    return {
        "tx_packets_emitted": None if tx_packets_emitted is None else int(tx_packets_emitted),
        "rx_packets_parsed": int(parsed_total),
        "matched_packets": int(matched),
        "missing_packets": int(missing),
        "ber_magic": None if ber_conditional is None else float(ber_conditional),
        "per": None if per is None else float(per),
        "detection_rate": None if detection_rate is None else float(detection_rate),
        "ber_global": None if ber_global is None else float(ber_global),
        "rx_parse_rate": None if rx_parse_rate is None else float(rx_parse_rate),
    }

def compute_channel_variation_metrics(chan_taps: np.ndarray) -> dict[str, float | int | None]:
    """
    Compute simple temporal-variation metrics from recorded channel taps.

    Parameters
    ----------
    chan_taps : np.ndarray
        Array with shape [n_frames, fft_len].

    Returns
    -------
    dict
        Summary metrics for temporal channel variation.
    """
    if chan_taps.ndim != 2 or chan_taps.shape[0] == 0:
        return {
            "n_frames": 0,
            "tap_power_mean": None,
            "tap_power_std": None,
            "tap_delta_mean": None,
            "tap_delta_std": None,
            "tap_delta_rel_mean": None,
            "tap_delta_rel_std": None,
        }

    power_per_frame = np.sum(np.abs(chan_taps) ** 2, axis=1)

    if chan_taps.shape[0] < 2:
        return {
            "n_frames": int(chan_taps.shape[0]),
            "tap_power_mean": float(np.mean(power_per_frame)),
            "tap_power_std": float(np.std(power_per_frame)),
            "tap_delta_mean": None,
            "tap_delta_std": None,
            "tap_delta_rel_mean": None,
            "tap_delta_rel_std": None,
        }

    delta = chan_taps[1:] - chan_taps[:-1]
    delta_norm = np.linalg.norm(delta, axis=1)
    prev_norm = np.linalg.norm(chan_taps[:-1], axis=1)

    eps = 1e-12
    delta_rel = delta_norm / np.maximum(prev_norm, eps)

    return {
        "n_frames": int(chan_taps.shape[0]),
        "tap_power_mean": float(np.mean(power_per_frame)),
        "tap_power_std": float(np.std(power_per_frame)),
        "tap_delta_mean": float(np.mean(delta_norm)),
        "tap_delta_std": float(np.std(delta_norm)),
        "tap_delta_rel_mean": float(np.mean(delta_rel)),
        "tap_delta_rel_std": float(np.std(delta_rel)),
    }
