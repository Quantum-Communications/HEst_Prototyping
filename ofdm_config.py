from __future__ import annotations

"""Shared OFDM/RF constants and synchronization words."""

# Shared OFDM / RF parameters
# =============================================================================

# Default baseband sample rate used in the experiment.
SAMPLE_RATE_DEFAULT = 2_000_000

# Default RF center frequency.
CENTER_FREQUENCY_DEFAULT = 907_980_000.0

# OFDM FFT size.
FFT_LENGTH = 64

# Cyclic prefix length.
CP_LENGTH = 16

# Occupied subcarriers used for data.
OCCUPIED_CARRIERS = ((-4, -3, -2, -1, 1, 2, 3, 4),)

# Pilot subcarrier locations.
PILOT_CARRIERS = ((-6, -5, 5, 6),)

# Pilot symbols placed on pilot carriers.
PILOT_SYMBOLS = ((-1, 1, -1, 1),)

# Common tag keys used by GNU Radio OFDM blocks.
LENGTH_TAG_KEY = "packet_len"
FRAME_LENGTH_TAG_KEY = "frame_len"
PACKET_LENGTH_TAG_KEY_RX = "packet_len"
PACKET_NUMBER_TAG_KEY = "packet_num"

# Fixed packet length used by stream_to_tagged_stream in the TX path.
PACKET_LENGTH_BYTES = 50

# Default device strings for the two HackRF units.
TX_ARGS_DEFAULT = "hackrf=0"
RX_ARGS_DEFAULT = "hackrf=1"

# =============================================================================
# Synchronization words
# =============================================================================
# These synchronization words must remain identical between TX and RX.
# They are inserted in the TX preamble and consumed by:
# - ofdm_sync_sc_cfb
# - ofdm_chanest_vcvc

SYNC_WORD1 = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.41421356, 0.0, 1.41421356,
    0.0, -1.41421356, 0.0, 1.41421356, 0.0, -1.41421356, 0.0, 1.41421356, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

SYNC_WORD2 = [
    0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j,
    0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, (1 + 0j), (1 + 0j), (1 + 0j), (-1 + 0j),
    (1 + 0j), (1 + 0j), 0j, (1 + 0j), (-1 + 0j), (1 + 0j), (1 + 0j), (1 + 0j), (-1 + 0j), 0j,
    0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j,
    0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j,
]

assert len(SYNC_WORD1) == FFT_LENGTH
assert len(SYNC_WORD2) == FFT_LENGTH

# =============================================================================
# RX buffer sizing strategy
# =============================================================================
# HackRF source blocks often refuse very large max output buffers and stay at
# 8192. Downstream OFDM blocks, however, often behave better with larger
# internal buffers, especially equalizers and serializers.

SOURCE_MAX_BUFFER = 8192
GENERAL_MIN_BUFFER = 16384
CRITICAL_MIN_BUFFER = 32768

# =============================================================================
