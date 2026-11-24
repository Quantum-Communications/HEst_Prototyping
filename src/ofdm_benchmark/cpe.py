"""Native streaming adapter for causal pilot common-phase correction."""

from __future__ import annotations

import numpy as np
import pmt
from gnuradio import gr

from .phy import pilot_phase


class PilotPhaseTracker(gr.sync_block):
    """Correct common pilot phase before GNU static overwrites pilot outputs.

    Measure after accounting for integer CFO and CP phase; leave those two
    corrections to the native frame equalizer. Preserve its tags unchanged.
    """

    def __init__(self, phy, is_payload=False):
        super().__init__(
            name="pilot_common_phase",
            in_sig=[(np.complex64, phy.fft_length)],
            out_sig=[(np.complex64, phy.fft_length)],
        )
        self.phy, self.skipped = phy, int(is_payload)
        self.h = np.ones(phy.fft_length, complex)
        self.coarse = self.index = 0
        self.phases = []

    def work(self, input_items, output_items):
        x, out = input_items[0], output_items[0]
        n = min(len(x), len(out))
        first = int(self.nitems_read(0))
        tags = {}
        for tag in self.get_tags_in_range(0, first, first + n):
            tags.setdefault(int(tag.offset), {})[pmt.symbol_to_string(tag.key)] = tag.value
        for j in range(n):
            t = tags.get(first + j, {})
            if "ofdm_sync_chan_taps" in t:
                self.h = np.asarray(pmt.c32vector_elements(t["ofdm_sync_chan_taps"]), complex)
                self.index = 0
                self.coarse = 0
            if "ofdm_sync_carr_offset" in t:
                self.coarse = int(pmt.to_long(t["ofdm_sync_carr_offset"]))
            if "frame_length" in t:
                self.index = 0
            phy = self.phy
            p = (self.skipped + self.index) % len(phy.pilot_symbols)
            carriers = phy.pilot_carriers[p % len(phy.pilot_carriers)]
            ids = np.asarray(carriers, int) + phy.fft_length // 2
            raw_ids = ids + self.coarse
            valid = (raw_ids >= 0) & (raw_ids < phy.fft_length)
            cp_phase = np.exp(
                -2j * np.pi * self.coarse * phy.cp_length / phy.fft_length * (self.index + 1)
            )
            predicted = self.h[ids] * np.asarray(phy.pilot_symbols[p])
            phase = pilot_phase(x[j, raw_ids[valid]] * cp_phase, predicted[valid])
            out[j] = x[j] * np.exp(-1j * phase)
            self.phases.append(phase)
            self.index += 1
        return n
