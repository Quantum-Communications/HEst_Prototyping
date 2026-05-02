from __future__ import annotations

"""File, payload, SNR, and magic-packet helper functions."""

import os
import random

def ensure_parent_directory(path: str) -> None:
    """
    Ensure that the parent directory of a file path exists.

    Parameters
    ----------
    path : str
        File path whose parent directory should exist.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_payload_file(path: str, nbytes: int, seed: int) -> None:
    """
    Create a deterministic random payload file if it does not already exist.

    Parameters
    ----------
    path : str
        Output file path.
    nbytes : int
        Number of bytes to generate.
    seed : int
        Deterministic seed for reproducible experiments.
    """
    if os.path.exists(path):
        return

    ensure_parent_directory(path)
    rng = random.Random(seed)
    data = bytes(rng.getrandbits(8) for _ in range(nbytes))

    with open(path, "wb") as f:
        f.write(data)


def remove_if_exists(path: str) -> None:
    """
    Remove a file if it already exists.

    Parameters
    ----------
    path : str
        File path to remove if present.
    """
    if os.path.exists(path):
        os.remove(path)


def snr_db_to_noise_voltage(snr_db: float) -> float:
    """
    Convert SNR in dB to GNU Radio channel_model noise_voltage.

    Approximation used:
        noise_voltage = 1 / 10^(SNR_dB / 20)

    Parameters
    ----------
    snr_db : float
        Desired SNR in dB.

    Returns
    -------
    float
        Noise voltage used by channels.channel_model.
    """
    return 1.0 / (10.0 ** (float(snr_db) / 20.0))


def generate_magic_payload_file(
    path: str,
    seed: int,
    num_packets: int,
    packet_len: int,
    magic4: bytes,
) -> dict[int, bytes]:
    """
    Generate a deterministic payload file composed of fixed-size packets.

    Packet format
    -------------
    [4 bytes magic][4 bytes little-endian sequence][payload bytes]

    Parameters
    ----------
    path : str
        Output file path.
    seed : int
        Deterministic seed for reproducibility.
    num_packets : int
        Number of packets to generate.
    packet_len : int
        Total packet length in bytes.
    magic4 : bytes
        4-byte packet magic signature.

    Returns
    -------
    dict[int, bytes]
        Mapping:
            sequence_number -> payload_bytes
    """
    if len(magic4) != 4:
        raise ValueError("magic4 must be exactly 4 bytes")

    if packet_len < 8:
        raise ValueError("packet_len must be at least 8 bytes")

    ensure_parent_directory(path)

    rng = random.Random(seed)
    payload_size = packet_len - 8
    tx_map: dict[int, bytes] = {}

    with open(path, "wb") as f:
        for seq in range(num_packets):
            payload = bytes(rng.getrandbits(8) for _ in range(payload_size))
            packet = magic4 + seq.to_bytes(4, byteorder="little", signed=False) + payload
            f.write(packet)
            tx_map[seq] = payload

    return tx_map


def parse_packets_by_magic(buf: bytes, magic4: bytes, packet_len: int) -> tuple[dict[int, bytes], int]:
    """
    Parse fixed-size packets from a raw byte buffer by searching for a 4-byte magic word.

    Packet format
    -------------
    [4 bytes magic][4 bytes little-endian sequence][payload bytes]

    This function scans the RX byte stream looking for packet candidates. It
    does not assume alignment from byte 0; it searches byte by byte.

    Parameters
    ----------
    buf : bytes
        Raw RX file contents.
    magic4 : bytes
        4-byte magic signature.
    packet_len : int
        Total packet length in bytes.

    Returns
    -------
    tuple[dict[int, bytes], int]
        First element:
            mapping sequence_number -> payload_bytes
            (only first occurrence kept for each sequence number)
        Second element:
            total number of packet signatures seen in RX
    """
    parsed: dict[int, bytes] = {}
    total_hits = 0

    if len(magic4) != 4 or packet_len < 8:
        return parsed, total_hits

    i = 0
    while True:
        j = buf.find(magic4, i)
        if j < 0:
            break

        if j + packet_len <= len(buf):
            seq = int.from_bytes(buf[j + 4:j + 8], byteorder="little", signed=False)
            payload = buf[j + 8:j + packet_len]
            total_hits += 1

            # Keep only the first occurrence of each sequence number.
            # This avoids letting repeated copies of the same packet inflate the
            # set of "unique recovered packets".
            if seq not in parsed:
                parsed[seq] = payload

        i = j + 1

    return parsed, total_hits
