from __future__ import annotations

"""Small GNU Radio helper/instrumentation blocks."""

import numpy as np
import pmt
from gnuradio import gr

from ofdm_config import FFT_LENGTH
from ofdm_utils import ensure_parent_directory

class TaggedPacketCounter(gr.sync_block):
    """
    Transparent byte-stream packet counter based on GNU Radio tags.

    Purpose
    -------
    This block sits in the TX byte stream right after `stream_to_tagged_stream`.
    It forwards the byte stream unchanged and counts how many packet-start tags
    pass through it.

    Why this works
    --------------
    `stream_to_tagged_stream` creates packet boundaries by inserting a tag with
    key `LENGTH_TAG_KEY` (in this experiment: "packet_len"). Every time that tag
    appears, a new logical packet has entered the TX framing pipeline.

    Therefore:
        packet_count = number of LENGTH_TAG_KEY tags seen by this block

    Important note
    --------------
    This counts logical packets entering the TX framing chain, not necessarily
    completed RF frames that left the antenna. However, for this experiment, it
    is a very good and much simpler proxy for "how many packets TX tried to send".
    """

    def __init__(self, tag_key: str) -> None:
        """
        Parameters
        ----------
        tag_key : str
            Name of the tag that marks packet boundaries, usually "packet_len".
        """
        super().__init__(
            name="tagged_packet_counter",
            in_sig=[np.uint8],
            out_sig=[np.uint8],
        )

        # Store the PMT version of the tag key for tag lookup.
        self.tag_key = pmt.intern(tag_key)

        # Total number of packet-start tags seen so far.
        self.packet_count = 0

    def work(self, input_items, output_items):
        """
        GNU Radio work function.

        Behavior
        --------
        1. Copy input bytes to output unchanged.
        2. Look for packet tags in the current input window.
        3. Increment the packet counter by the number of matching tags found.
        """
        inp = input_items[0]
        out = output_items[0]

        # Number of items available in this call.
        n = len(inp)

        # Transparent pass-through: preserve the original stream exactly.
        out[:n] = inp[:n]

        # Absolute item range covered by this work call.
        abs_start = self.nitems_read(0)
        abs_end = abs_start + n

        # Query tags in the current input range.
        tags = self.get_tags_in_range(0, abs_start, abs_end, self.tag_key)

        # Each matching packet-length tag corresponds to one logical packet.
        self.packet_count += len(tags)

        return n

class ChannelTapRecorder(gr.sync_block):
    """
    Passive sink that records channel-estimation tags published on its input stream.

    Expected tags on the input stream:
        - ofdm_sync_chan_taps   (c32vector)
        - ofdm_sync_carr_offset (long)

    It does not modify the main RX path because it is connected in parallel.
    """

    def __init__(self, output_path: str) -> None:
        super().__init__(
            name="channel_tap_recorder",
            in_sig=[(np.complex64, FFT_LENGTH)],
            out_sig=None,
        )

        self.output_path = output_path
        self.frame_indices: list[int] = []
        self.carr_offsets: list[int] = []
        self.chan_taps: list[np.ndarray] = []

        self._next_frame_index = 0

    def work(self, input_items, output_items):
        inp = input_items[0]
        n = len(inp)

        abs_start = self.nitems_read(0)
        abs_end = abs_start + n

        tags = self.get_tags_in_range(0, abs_start, abs_end)

        # Group tags by absolute offset, because each output vector/frame can have
        # both carr_offset and chan_taps tags at the same item offset.
        tags_by_offset: dict[int, dict[str, object]] = {}

        for tag in tags:
            key = pmt.symbol_to_string(tag.key)
            if key not in ("ofdm_sync_chan_taps", "ofdm_sync_carr_offset"):
                continue

            if tag.offset not in tags_by_offset:
                tags_by_offset[tag.offset] = {}

            if key == "ofdm_sync_carr_offset":
                tags_by_offset[tag.offset]["carr_offset"] = int(pmt.to_long(tag.value))
            elif key == "ofdm_sync_chan_taps":
                taps = np.asarray(pmt.c32vector_elements(tag.value), dtype=np.complex64)
                tags_by_offset[tag.offset]["chan_taps"] = taps

        for offset in sorted(tags_by_offset.keys()):
            entry = tags_by_offset[offset]

            # Record only entries that actually contain taps.
            # carr_offset may still be present or absent depending on the upstream block.
            if "chan_taps" not in entry:
                continue

            self.frame_indices.append(self._next_frame_index)
            self.carr_offsets.append(int(entry.get("carr_offset", 0)))
            self.chan_taps.append(entry["chan_taps"])
            self._next_frame_index += 1

        return n

    def save(self) -> None:
        ensure_parent_directory(self.output_path)

        if len(self.chan_taps) == 0:
            np.savez(
                self.output_path,
                frame_idx=np.asarray([], dtype=np.int64),
                carr_offset=np.asarray([], dtype=np.int32),
                chan_taps=np.asarray([], dtype=np.complex64).reshape(0, FFT_LENGTH),
            )
            return

        np.savez(
            self.output_path,
            frame_idx=np.asarray(self.frame_indices, dtype=np.int64),
            carr_offset=np.asarray(self.carr_offsets, dtype=np.int32),
            chan_taps=np.asarray(self.chan_taps, dtype=np.complex64),
        )
