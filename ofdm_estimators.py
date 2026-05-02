from __future__ import annotations

"""Python channel-estimator variants for the OFDM receiver."""

import threading
from collections import deque

import numpy as np
import pmt
from gnuradio import gr
from sklearn.svm import SVR

class FrozenAfterNFramesChanEst(gr.basic_block):
    """
    Channel estimator that behaves like GNU Radio native LS for the first N
    estimation frames, then freezes the channel taps from the N-th estimate.

    Behavior
    --------
    - Always computes and publishes the current carrier offset.
    - Computes channel taps normally until `freeze_after_frames` estimates.
    - After that, reuses the last estimated taps forever.
    - Copies data symbols exactly like ofdm_chanest_vcvc:
        input:  [sync symbols][data symbols]
        output: [data symbols]
    """

    def __init__(
        self,
        sync_word1,
        sync_word2,
        n_data_symbols: int,
        max_carr_offset: int = 3,
        freeze_after_frames: int = 1,
    ) -> None:
        self.fft_len = len(sync_word1)
        self.n_data_symbols = int(n_data_symbols)
        self.n_sync_symbols = 2 if len(sync_word2) > 0 else 1
        self.max_carr_offset = int(max_carr_offset)
        self.freeze_after_frames = max(1, int(freeze_after_frames))

        super().__init__(
            name="frozen_after_n_frames_chanest",
            in_sig=[(np.complex64, self.fft_len)],
            out_sig=[(np.complex64, self.fft_len)],
        )

        self.sync_word1 = np.asarray(sync_word1, dtype=np.complex64)
        self.sync_word2 = (
            np.asarray(sync_word2, dtype=np.complex64) if len(sync_word2) > 0 else None
        )
        self.ref_sym = (
            self.sync_word2
            if self.sync_word2 is not None and len(self.sync_word2) > 0
            else self.sync_word1
        )

        self.first_active_carrier = 0
        self.last_active_carrier = self.fft_len - 1

        for i in range(self.fft_len):
            if self.ref_sym[i] != 0:
                self.first_active_carrier = i
                break

        for i in range(self.fft_len - 1, -1, -1):
            if self.ref_sym[i] != 0:
                self.last_active_carrier = i
                break

        # Same carrier-offset bounds logic as native block.
        self.max_neg_carr_offset = -self.first_active_carrier
        self.max_pos_carr_offset = self.fft_len - self.last_active_carrier - 1

        if self.max_carr_offset != -1:
            self.max_neg_carr_offset = max(-self.max_carr_offset, self.max_neg_carr_offset)
            self.max_pos_carr_offset = min(self.max_carr_offset, self.max_pos_carr_offset)

        if self.max_neg_carr_offset % 2:
            self.max_neg_carr_offset += 1
        if self.max_pos_carr_offset % 2:
            self.max_pos_carr_offset -= 1

        # For the 2-sync-symbol case, replicate the Schmidl & Cox-style coarse
        # carrier-offset search used by the native block.
        self.corr_v = None
        if self.sync_word2 is not None and len(self.sync_word2) > 0:
            self.corr_v = np.zeros(self.fft_len, dtype=np.complex64)
            for i in range(self.fft_len):
                if self.sync_word1[i] == 0:
                    self.corr_v[i] = 0
                else:
                    self.corr_v[i] = self.sync_word2[i] / self.sync_word1[i]

        self.frozen_taps = None
        self.has_frozen_taps = False
        self.estimation_frame_count = 0

        self.set_output_multiple(self.n_data_symbols)
        self.set_relative_rate(
            float(self.n_data_symbols) / float(self.n_data_symbols + self.n_sync_symbols)
        )
        self.set_tag_propagation_policy(gr.TPP_DONT)

    def forecast(self, noutput_items, ninputs):
        frames = max(1, noutput_items // self.n_data_symbols)
        required = frames * (self.n_data_symbols + self.n_sync_symbols)
        return [required for _ in range(ninputs)]

    def _get_carr_offset(self, sync_sym1: np.ndarray, sync_sym2: np.ndarray) -> int:
        carr_offset = 0
        best = 0.0

        if self.corr_v is None:
            # This class is intended for the same 2-sync-symbol case as your current flow.
            # Fallback: no offset correction beyond zero.
            return 0

        for g in range(self.max_neg_carr_offset, self.max_pos_carr_offset + 1, 2):
            tmp = 0.0 + 0.0j
            for k in range(self.fft_len):
                if self.corr_v[k] != 0:
                    idx = k + g
                    if 0 <= idx < self.fft_len:
                        tmp += (
                            np.conj(sync_sym1[idx])
                            * np.conj(self.corr_v[k])
                            * sync_sym2[idx]
                        )
            score = abs(tmp)
            if score > best:
                best = score
                carr_offset = g

        return int(carr_offset)

    def _get_chan_taps(
        self,
        sync_sym1: np.ndarray,
        sync_sym2: np.ndarray,
        carr_offset: int,
    ) -> np.ndarray:
        sym = (
            sync_sym2
            if self.sync_word2 is not None and len(self.sync_word2) > 0
            else sync_sym1
        )
        taps = np.zeros(self.fft_len, dtype=np.complex64)

        loop_start = 0
        loop_end = self.fft_len
        if carr_offset > 0:
            loop_start = carr_offset
        elif carr_offset < 0:
            loop_end = self.fft_len + carr_offset

        for i in range(loop_start, loop_end):
            ref_idx = i - carr_offset
            if self.ref_sym[ref_idx] != 0:
                taps[ref_idx] = sym[i] / self.ref_sym[ref_idx]

        return taps

    def general_work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]

        framesize = self.n_sync_symbols + self.n_data_symbols
        available_in_vectors = len(inp)
        available_out_vectors = len(out)

        max_input_frames = available_in_vectors // framesize
        max_output_frames = available_out_vectors // self.n_data_symbols
        nframes = min(max_input_frames, max_output_frames)

        if nframes <= 0:
            return 0

        produced_vectors = 0
        nread0 = self.nitems_read(0)
        nwritten0 = self.nitems_written(0)

        for f in range(nframes):
            base_in = f * framesize
            base_out = f * self.n_data_symbols

            sync_sym1 = np.asarray(inp[base_in], dtype=np.complex64)
            sync_sym2 = (
                np.asarray(inp[base_in + 1], dtype=np.complex64)
                if self.n_sync_symbols == 2
                else sync_sym1
            )

            carr_offset = self._get_carr_offset(sync_sym1, sync_sym2)

            if not self.has_frozen_taps:
                current_taps = self._get_chan_taps(sync_sym1, sync_sym2, carr_offset)
                self.frozen_taps = current_taps.copy()
                self.estimation_frame_count += 1

                if self.estimation_frame_count >= self.freeze_after_frames:
                    self.has_frozen_taps = True

            # Copy only the data symbols to the output, exactly like native chanest.
            out[base_out : base_out + self.n_data_symbols] = inp[
                base_in + self.n_sync_symbols : base_in + self.n_sync_symbols + self.n_data_symbols
            ]

            write_offset = nwritten0 + base_out

            # Publish current carrier offset every frame.
            self.add_item_tag(
                0,
                write_offset,
                pmt.string_to_symbol("ofdm_sync_carr_offset"),
                pmt.from_long(int(carr_offset)),
            )

            # Publish frozen taps every frame.
            self.add_item_tag(
                0,
                write_offset,
                pmt.string_to_symbol("ofdm_sync_chan_taps"),
                pmt.init_c32vector(self.fft_len, self.frozen_taps.tolist()),
            )

            # Propagate other tags from the input frame similarly to native block.
            tags = self.get_tags_in_range(
                0,
                nread0 + base_in,
                nread0 + base_in + framesize,
            )

            for tag in tags:
                rel = int(tag.offset - (nread0 + base_in))

                if rel < self.n_sync_symbols:
                    new_rel = 0
                else:
                    new_rel = rel - self.n_sync_symbols

                # Avoid duplicating the tags we explicitly republish.
                key_str = pmt.symbol_to_string(tag.key)
                if key_str in ("ofdm_sync_carr_offset", "ofdm_sync_chan_taps"):
                    continue

                self.add_item_tag(
                    0,
                    write_offset + new_rel,
                    tag.key,
                    tag.value,
                )

            produced_vectors += self.n_data_symbols

        self.consume(0, nframes * framesize)
        return produced_vectors

class PythonNativeLSChanEst(gr.basic_block):
    """
    Python reimplementation of the GNU Radio native LS OFDM channel estimator
    for the current experiment setup.

    Behavior
    --------
    - Input is a sequence of FFT vectors arranged per frame as:
          [sync_symbol_1, sync_symbol_2, data_symbol_0, ..., data_symbol_{N-1}]
    - Output is only the data symbols:
          [data_symbol_0, ..., data_symbol_{N-1}]
    - Publishes, at the first output item of each frame:
          * ofdm_sync_carr_offset
          * ofdm_sync_chan_taps
    - Propagates all other input tags similarly to the native block.

    Notes
    -----
    This version is intended to match the current use case:
    - two sync symbols
    - no interpolation mode
    - no noise reduction
    """

    def __init__(
        self,
        sync_word1,
        sync_word2,
        n_data_symbols: int,
        max_carr_offset: int = 3,
    ) -> None:
        self.fft_len = len(sync_word1)
        self.n_data_symbols = int(n_data_symbols)
        self.n_sync_symbols = 2 if len(sync_word2) > 0 else 1
        self.max_carr_offset = int(max_carr_offset)

        super().__init__(
            name="python_native_ls_chanest",
            in_sig=[(np.complex64, self.fft_len)],
            out_sig=[(np.complex64, self.fft_len)],
        )

        self.sync_word1 = np.asarray(sync_word1, dtype=np.complex64)
        self.sync_word2 = (
            np.asarray(sync_word2, dtype=np.complex64) if len(sync_word2) > 0 else None
        )
        self.ref_sym = (
            self.sync_word2
            if self.sync_word2 is not None and len(self.sync_word2) > 0
            else self.sync_word1
        )

        self.first_active_carrier = 0
        self.last_active_carrier = self.fft_len - 1

        for i in range(self.fft_len):
            if self.ref_sym[i] != 0:
                self.first_active_carrier = i
                break

        for i in range(self.fft_len - 1, -1, -1):
            if self.ref_sym[i] != 0:
                self.last_active_carrier = i
                break

        self.max_neg_carr_offset = -self.first_active_carrier
        self.max_pos_carr_offset = self.fft_len - self.last_active_carrier - 1

        if self.max_carr_offset != -1:
            self.max_neg_carr_offset = max(-self.max_carr_offset, self.max_neg_carr_offset)
            self.max_pos_carr_offset = min(self.max_carr_offset, self.max_pos_carr_offset)

        # Carrier offsets must be even, like native block.
        if self.max_neg_carr_offset % 2:
            self.max_neg_carr_offset += 1
        if self.max_pos_carr_offset % 2:
            self.max_pos_carr_offset -= 1

        # Same correlation helper used by the native 2-sync-symbol case.
        self.corr_v = None
        if self.sync_word2 is not None and len(self.sync_word2) > 0:
            self.corr_v = np.zeros(self.fft_len, dtype=np.complex64)
            for i in range(self.fft_len):
                if self.sync_word1[i] == 0:
                    self.corr_v[i] = 0
                else:
                    self.corr_v[i] = self.sync_word2[i] / self.sync_word1[i]

        self.set_output_multiple(self.n_data_symbols)
        self.set_relative_rate(
            float(self.n_data_symbols) / float(self.n_data_symbols + self.n_sync_symbols)
        )
        self.set_tag_propagation_policy(gr.TPP_DONT)

    def forecast(self, noutput_items, ninputs):
        frames = max(1, noutput_items // self.n_data_symbols)
        required = frames * (self.n_data_symbols + self.n_sync_symbols)
        return [required for _ in range(ninputs)]

    def _get_carr_offset(self, sync_sym1: np.ndarray, sync_sym2: np.ndarray) -> int:
        """
        Replicate the native 2-sync-symbol carrier offset estimation.
        """
        if self.corr_v is None:
            return 0

        carr_offset = 0
        best_score = 0.0

        for g in range(self.max_neg_carr_offset, self.max_pos_carr_offset + 1, 2):
            tmp = 0.0 + 0.0j
            for k in range(self.fft_len):
                if self.corr_v[k] != 0:
                    idx = k + g
                    if 0 <= idx < self.fft_len:
                        tmp += (
                            np.conj(sync_sym1[idx])
                            * np.conj(self.corr_v[k])
                            * sync_sym2[idx]
                        )
            score = abs(tmp)
            if score > best_score:
                best_score = score
                carr_offset = g

        return int(carr_offset)

    def _get_chan_taps(
        self,
        sync_sym1: np.ndarray,
        sync_sym2: np.ndarray,
        carr_offset: int,
    ) -> np.ndarray:
        """
        Replicate native channel-tap estimation.
        """
        sym = (
            sync_sym2
            if self.sync_word2 is not None and len(self.sync_word2) > 0
            else sync_sym1
        )

        taps = np.zeros(self.fft_len, dtype=np.complex64)

        loop_start = 0
        loop_end = self.fft_len
        if carr_offset > 0:
            loop_start = carr_offset
        elif carr_offset < 0:
            loop_end = self.fft_len + carr_offset

        for i in range(loop_start, loop_end):
            ref_idx = i - carr_offset
            if self.ref_sym[ref_idx] != 0:
                taps[ref_idx] = sym[i] / self.ref_sym[ref_idx]

        return taps

    def general_work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]

        framesize = self.n_sync_symbols + self.n_data_symbols
        available_in_vectors = len(inp)
        available_out_vectors = len(out)

        max_input_frames = available_in_vectors // framesize
        max_output_frames = available_out_vectors // self.n_data_symbols
        nframes = min(max_input_frames, max_output_frames)

        if nframes <= 0:
            return 0

        produced_vectors = 0
        nread0 = self.nitems_read(0)
        nwritten0 = self.nitems_written(0)

        for f in range(nframes):
            base_in = f * framesize
            base_out = f * self.n_data_symbols

            sync_sym1 = np.asarray(inp[base_in], dtype=np.complex64)
            sync_sym2 = (
                np.asarray(inp[base_in + 1], dtype=np.complex64)
                if self.n_sync_symbols == 2
                else sync_sym1
            )

            carr_offset = self._get_carr_offset(sync_sym1, sync_sym2)
            chan_taps = self._get_chan_taps(sync_sym1, sync_sym2, carr_offset)

            # Copy only data symbols to output.
            out[base_out : base_out + self.n_data_symbols] = inp[
                base_in + self.n_sync_symbols : base_in + self.n_sync_symbols + self.n_data_symbols
            ]

            write_offset = nwritten0 + base_out

            # Publish the same two tags as the native block.
            self.add_item_tag(
                0,
                write_offset,
                pmt.string_to_symbol("ofdm_sync_carr_offset"),
                pmt.from_long(int(carr_offset)),
            )
            self.add_item_tag(
                0,
                write_offset,
                pmt.string_to_symbol("ofdm_sync_chan_taps"),
                pmt.init_c32vector(self.fft_len, chan_taps.tolist()),
            )

            # Propagate all other tags similarly to native block.
            tags = self.get_tags_in_range(
                0,
                nread0 + base_in,
                nread0 + base_in + framesize,
            )

            for tag in tags:
                rel = int(tag.offset - (nread0 + base_in))

                if rel < self.n_sync_symbols:
                    new_rel = 0
                else:
                    new_rel = rel - self.n_sync_symbols

                key_str = pmt.symbol_to_string(tag.key)
                if key_str in ("ofdm_sync_carr_offset", "ofdm_sync_chan_taps"):
                    continue

                self.add_item_tag(
                    0,
                    write_offset + new_rel,
                    tag.key,
                    tag.value,
                )

            produced_vectors += self.n_data_symbols

        self.consume(0, nframes * framesize)
        return produced_vectors
    
class PythonSVRTimeDomainChanEst(gr.basic_block):
    """
    Time-domain SVR channel estimator.

    Input per frame:
        [sync_symbol_1_time, sync_symbol_2_time, data_symbol_0_time, ..., data_symbol_{N-1}_time]

    Output per frame:
        [data_symbol_0_freq, ..., data_symbol_{N-1}_freq]

    Published tags at the first output vector of each frame:
        - ofdm_sync_carr_offset
        - ofdm_sync_chan_taps

    Notes
    -----
    - Channel is estimated in time domain from the second sync symbol.
    - The estimated received sync is FFT'd and divided by the known sync symbol
      to obtain channel taps in frequency domain.
    - Carrier offset is still estimated in frequency domain, like native chanest.
    - Output data symbols are FFT'd internally and emitted already in frequency domain.
    """

    def __init__(
        self,
        sync_word1,
        sync_word2,
        n_data_symbols: int,
        max_carr_offset: int = 3,
        svr_c: float = 100.0,
        svr_gamma: str = "scale",
        svr_epsilon: float = 0.001,
    ) -> None:
        self.fft_len = len(sync_word1)
        self.n_data_symbols = int(n_data_symbols)
        self.n_sync_symbols = 2 if len(sync_word2) > 0 else 1
        self.max_carr_offset = int(max_carr_offset)

        self.svr_c = float(svr_c)
        self.svr_gamma = svr_gamma
        self.svr_epsilon = float(svr_epsilon)

        super().__init__(
            name="python_svr_time_domain_chanest",
            in_sig=[(np.complex64, self.fft_len)],
            out_sig=[(np.complex64, self.fft_len)],
        )

        self.sync_word1 = np.asarray(sync_word1, dtype=np.complex64)
        self.sync_word2 = np.asarray(sync_word2, dtype=np.complex64)

        if self.n_sync_symbols != 2:
            raise ValueError("PythonSVRTimeDomainChanEst expects exactly 2 sync symbols.")

        self.ref_sym = self.sync_word2

        # Active carriers in shifted layout.
        active_shifted = np.flatnonzero(np.abs(self.sync_word2) > 0)
        self.active_shifted = active_shifted.astype(np.int32)

        self.first_active_carrier = int(active_shifted[0])
        self.last_active_carrier = int(active_shifted[-1])

        # Native-like carrier offset bounds.
        self.max_neg_carr_offset = -self.first_active_carrier
        self.max_pos_carr_offset = self.fft_len - self.last_active_carrier - 1

        if self.max_carr_offset != -1:
            self.max_neg_carr_offset = max(-self.max_carr_offset, self.max_neg_carr_offset)
            self.max_pos_carr_offset = min(self.max_carr_offset, self.max_pos_carr_offset)

        if self.max_neg_carr_offset % 2:
            self.max_neg_carr_offset += 1
        if self.max_pos_carr_offset % 2:
            self.max_pos_carr_offset -= 1

        # Native-like helper for 2-sync-symbol carrier-offset estimation.
        self.corr_v = np.zeros(self.fft_len, dtype=np.complex64)
        for i in range(self.fft_len):
            if self.sync_word1[i] == 0:
                self.corr_v[i] = 0
            else:
                self.corr_v[i] = self.sync_word2[i] / self.sync_word1[i]

        # Precompute SVR feature matrices in time domain from known sync symbol 2.
        self._svr_X_re, self._svr_X_im = self._build_time_domain_feature_matrices()

        self.set_output_multiple(self.n_data_symbols)
        self.set_relative_rate(
            float(self.n_data_symbols) / float(self.n_data_symbols + self.n_sync_symbols)
        )
        self.set_tag_propagation_policy(gr.TPP_DONT)

    def forecast(self, noutput_items, ninputs):
        frames = max(1, noutput_items // self.n_data_symbols)
        required = frames * (self.n_data_symbols + self.n_sync_symbols)
        return [required for _ in range(ninputs)]

    def _time_to_freq_shifted(self, x_time: np.ndarray) -> np.ndarray:
        return np.fft.fftshift(np.fft.fft(x_time)).astype(np.complex64)

    def _build_time_domain_feature_matrices(self):
        """
        Build feature matrices similar in spirit to the old simulation code,
        but using the actual known GNU Radio sync symbol 2.

        sync_word2 is stored in shifted frequency order.
        """
        n = self.fft_len
        t = np.arange(n, dtype=np.float64)

        sync2_unshifted = np.fft.ifftshift(self.sync_word2)
        active_unshifted = np.flatnonzero(np.abs(sync2_unshifted) > 0)

        basis = np.zeros((n, len(active_unshifted)), dtype=np.complex64)

        for col, k in enumerate(active_unshifted):
            basis[:, col] = (
                sync2_unshifted[k] * np.exp(1j * 2.0 * np.pi * k * t / n) / n
            )

        return basis.real.astype(np.float32), basis.imag.astype(np.float32)

    def _get_carr_offset(self, sync_freq1: np.ndarray, sync_freq2: np.ndarray) -> int:
        carr_offset = 0
        best_score = 0.0

        for g in range(self.max_neg_carr_offset, self.max_pos_carr_offset + 1, 2):
            tmp = 0.0 + 0.0j
            for k in range(self.fft_len):
                if self.corr_v[k] != 0:
                    idx = k + g
                    if 0 <= idx < self.fft_len:
                        tmp += (
                            np.conj(sync_freq1[idx])
                            * np.conj(self.corr_v[k])
                            * sync_freq2[idx]
                        )
            score = abs(tmp)
            if score > best_score:
                best_score = score
                carr_offset = g

        return int(carr_offset)

    def _estimate_chan_taps_svr_time(
        self,
        sync_time_2: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate the received sync symbol in time domain via SVR, then convert
        to frequency-domain channel taps by dividing by the known sync symbol.
        """
        y_re = sync_time_2.real.astype(np.float32)
        y_im = sync_time_2.imag.astype(np.float32)

        svr_re = SVR(
            kernel="rbf",
            C=self.svr_c,
            gamma=self.svr_gamma,
            epsilon=self.svr_epsilon,
        )
        svr_im = SVR(
            kernel="rbf",
            C=self.svr_c,
            gamma=self.svr_gamma,
            epsilon=self.svr_epsilon,
        )

        svr_re.fit(self._svr_X_re, y_re)
        svr_im.fit(self._svr_X_im, y_im)

        yhat_re = svr_re.predict(self._svr_X_re).astype(np.float32)
        yhat_im = svr_im.predict(self._svr_X_im).astype(np.float32)

        sync2_hat_time = (yhat_re + 1j * yhat_im).astype(np.complex64)
        sync2_hat_freq = self._time_to_freq_shifted(sync2_hat_time)

        chan_taps = np.zeros(self.fft_len, dtype=np.complex64)
        nz = np.abs(self.sync_word2) > 0
        chan_taps[nz] = sync2_hat_freq[nz] / self.sync_word2[nz]

        return chan_taps

    def general_work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]

        framesize = self.n_sync_symbols + self.n_data_symbols
        available_in_vectors = len(inp)
        available_out_vectors = len(out)

        max_input_frames = available_in_vectors // framesize
        max_output_frames = available_out_vectors // self.n_data_symbols
        nframes = min(max_input_frames, max_output_frames)

        if nframes <= 0:
            return 0

        produced_vectors = 0
        nread0 = self.nitems_read(0)
        nwritten0 = self.nitems_written(0)

        for f in range(nframes):
            base_in = f * framesize
            base_out = f * self.n_data_symbols

            sync_time_1 = np.asarray(inp[base_in + 0], dtype=np.complex64)
            sync_time_2 = np.asarray(inp[base_in + 1], dtype=np.complex64)

            # Estimate channel in time domain.
            chan_taps = self._estimate_chan_taps_svr_time(sync_time_2)

            # Convert syncs to frequency domain for carr offset estimation.
            sync_freq_1 = self._time_to_freq_shifted(sync_time_1)
            sync_freq_2 = self._time_to_freq_shifted(sync_time_2)

            carr_offset = self._get_carr_offset(sync_freq_1, sync_freq_2)

            # Convert data symbols to frequency domain and emit them.
            for i in range(self.n_data_symbols):
                data_time = np.asarray(inp[base_in + self.n_sync_symbols + i], dtype=np.complex64)
                out[base_out + i] = self._time_to_freq_shifted(data_time)

            write_offset = nwritten0 + base_out

            self.add_item_tag(
                0,
                write_offset,
                pmt.string_to_symbol("ofdm_sync_carr_offset"),
                pmt.from_long(int(carr_offset)),
            )
            self.add_item_tag(
                0,
                write_offset,
                pmt.string_to_symbol("ofdm_sync_chan_taps"),
                pmt.init_c32vector(self.fft_len, chan_taps.tolist()),
            )

            # Propagate all other tags similarly to native chanest.
            tags = self.get_tags_in_range(
                0,
                nread0 + base_in,
                nread0 + base_in + framesize,
            )

            for tag in tags:
                rel = int(tag.offset - (nread0 + base_in))

                if rel < self.n_sync_symbols:
                    new_rel = 0
                else:
                    new_rel = rel - self.n_sync_symbols

                key_str = pmt.symbol_to_string(tag.key)
                if key_str in ("ofdm_sync_carr_offset", "ofdm_sync_chan_taps"):
                    continue

                self.add_item_tag(
                    0,
                    write_offset + new_rel,
                    tag.key,
                    tag.value,
                )

            produced_vectors += self.n_data_symbols

        self.consume(0, nframes * framesize)
        return produced_vectors

class PythonSVRFrozenAsyncChanEst(gr.basic_block):
    """
    Asynchronous frozen SVR channel estimator with sliding-window training.

    Real-time path
    --------------
    The GNU Radio work thread must stay fast. Therefore, it only:
        1. Computes the current carrier offset.
        2. Publishes the latest available channel taps.
        3. Queues the newest sync snapshot for the background worker.
        4. Converts header data symbols from time domain to frequency domain.

    Background path
    ---------------
    A daemon worker thread accumulates recent sync symbols in a sliding window.
    Once enough frames are available, it trains the SVR using the accumulated
    window and updates `latest_taps`.

    Important design choice
    -----------------------
    This block does NOT use LS as initialization. It starts with a neutral flat
    channel on the active sync carriers and then lets the async SVR update the
    taps over time.
    """

    def __init__(
        self,
        sync_word1,
        sync_word2,
        n_data_symbols: int,
        max_carr_offset: int = 3,
        svr_c: float = 100.0,
        svr_gamma: str = "scale",
        svr_epsilon: float = 0.001,
        svr_window_frames: int = 20,
        svr_min_train_frames: int = 5,
    ) -> None:
        self.fft_len = len(sync_word1)
        self.n_data_symbols = int(n_data_symbols)
        self.n_sync_symbols = 2 if len(sync_word2) > 0 else 1
        self.max_carr_offset = int(max_carr_offset)

        self.svr_c = float(svr_c)
        self.svr_gamma = svr_gamma
        self.svr_epsilon = float(svr_epsilon)
        self.svr_window_frames = max(1, int(svr_window_frames))
        self.svr_min_train_frames = max(1, int(svr_min_train_frames))

        super().__init__(
            name="python_svr_frozen_async_chanest",
            in_sig=[(np.complex64, self.fft_len)],
            out_sig=[(np.complex64, self.fft_len)],
        )

        self.sync_word1 = np.asarray(sync_word1, dtype=np.complex64)
        self.sync_word2 = np.asarray(sync_word2, dtype=np.complex64)

        if self.n_sync_symbols != 2:
            raise ValueError("PythonSVRFrozenAsyncChanEst expects exactly 2 sync symbols.")

        active_shifted = np.flatnonzero(np.abs(self.sync_word2) > 0)
        if active_shifted.size == 0:
            raise ValueError("sync_word2 has no active carriers.")

        self.first_active_carrier = int(active_shifted[0])
        self.last_active_carrier = int(active_shifted[-1])

        self.max_neg_carr_offset = -self.first_active_carrier
        self.max_pos_carr_offset = self.fft_len - self.last_active_carrier - 1

        if self.max_carr_offset != -1:
            self.max_neg_carr_offset = max(-self.max_carr_offset, self.max_neg_carr_offset)
            self.max_pos_carr_offset = min(self.max_carr_offset, self.max_pos_carr_offset)

        if self.max_neg_carr_offset % 2:
            self.max_neg_carr_offset += 1
        if self.max_pos_carr_offset % 2:
            self.max_pos_carr_offset -= 1

        self.corr_v = np.zeros(self.fft_len, dtype=np.complex64)
        for i in range(self.fft_len):
            if self.sync_word1[i] == 0:
                self.corr_v[i] = 0
            else:
                self.corr_v[i] = self.sync_word2[i] / self.sync_word1[i]

        self._svr_X_re, self._svr_X_im = self._build_time_domain_feature_matrices()

        # Initial neutral channel: flat only on active sync carriers.
        self.latest_taps = np.zeros(self.fft_len, dtype=np.complex64)
        nz = np.abs(self.sync_word2) > 0
        self.latest_taps[nz] = 1.0 + 0.0j

        self._latest_taps_lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._snapshot_event = threading.Event()
        self._stop_event = threading.Event()

        self._snapshot_buffer = deque(maxlen=self.svr_window_frames)
        self._pending_batch = None

        self.worker_frames_started = 0
        self.worker_frames_completed = 0
        self.worker_batches_skipped = 0
        self.worker_last_batch_size = 0

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="svr_frozen_async_worker",
            daemon=True,
        )
        self._worker_thread.start()

        self.set_output_multiple(self.n_data_symbols)
        self.set_relative_rate(
            float(self.n_data_symbols) / float(self.n_data_symbols + self.n_sync_symbols)
        )
        self.set_tag_propagation_policy(gr.TPP_DONT)

    def stop(self):
        self._stop_event.set()
        self._snapshot_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        return True

    def forecast(self, noutput_items, ninputs):
        frames = max(1, noutput_items // self.n_data_symbols)
        required = frames * (self.n_data_symbols + self.n_sync_symbols)
        return [required for _ in range(ninputs)]

    def _time_to_freq_shifted(self, x_time: np.ndarray) -> np.ndarray:
        """
        Match the GNU Radio FFT block convention used in this receiver:
            fft.fft_vcc(FFT_LENGTH, True, (), True, 1)
        """
        return np.fft.fftshift(np.fft.fft(x_time)).astype(np.complex64)

    def _build_time_domain_feature_matrices(self):
        """
        Build deterministic time-domain features from the known second sync word.

        The old simulation used explicit pilot time-domain basis vectors. Here
        we build the same kind of basis from the actual GNU Radio sync word.
        """
        n = self.fft_len
        t = np.arange(n, dtype=np.float64)

        sync2_unshifted = np.fft.ifftshift(self.sync_word2)
        active_unshifted = np.flatnonzero(np.abs(sync2_unshifted) > 0)

        basis = np.zeros((n, len(active_unshifted)), dtype=np.complex64)

        for col, k in enumerate(active_unshifted):
            basis[:, col] = (
                sync2_unshifted[k] * np.exp(1j * 2.0 * np.pi * k * t / n) / n
            )

        return basis.real.astype(np.float32), basis.imag.astype(np.float32)

    def _get_carr_offset(self, sync_freq1: np.ndarray, sync_freq2: np.ndarray) -> int:
        """
        Native-like carrier-offset search using the two sync symbols in frequency domain.
        """
        carr_offset = 0
        best_score = 0.0

        for g in range(self.max_neg_carr_offset, self.max_pos_carr_offset + 1, 2):
            tmp = 0.0 + 0.0j
            for k in range(self.fft_len):
                if self.corr_v[k] != 0:
                    idx = k + g
                    if 0 <= idx < self.fft_len:
                        tmp += (
                            np.conj(sync_freq1[idx])
                            * np.conj(self.corr_v[k])
                            * sync_freq2[idx]
                        )

            score = abs(tmp)
            if score > best_score:
                best_score = score
                carr_offset = g

        return int(carr_offset)

    def _build_taps_from_estimated_sync(
        self,
        sync2_hat_freq: np.ndarray,
        carr_offset: int,
    ) -> np.ndarray:
        """
        Convert the estimated second sync symbol in frequency domain into the
        vector published as `ofdm_sync_chan_taps`.

        Current experimental convention:
        - Do NOT divide by sync_word2.
        - Use the estimated sync spectrum itself as the tap values.
        - Apply carrier-offset indexing so taps are published in the same
          reference convention used by the LS-like estimators.
        """
        chan_taps = np.zeros(self.fft_len, dtype=np.complex64)

        loop_start = 0
        loop_end = self.fft_len

        if carr_offset > 0:
            loop_start = carr_offset
        elif carr_offset < 0:
            loop_end = self.fft_len + carr_offset

        for i in range(loop_start, loop_end):
            ref_idx = i - carr_offset
            if self.sync_word2[ref_idx] != 0:
                chan_taps[ref_idx] = sync2_hat_freq[i] #/ self.sync_word2[ref_idx]

        return chan_taps

    def _estimate_chan_taps_svr_batch(self, batch) -> np.ndarray:
        """
        Train SVR using a sliding window of recent sync symbols.

        batch format:
            [(sync_time_2, carr_offset), ...]

        Training data:
            X = repeated deterministic time-domain basis
            y = concatenated received sync samples over the window

        Prediction:
            Predict one representative sync using the base feature matrix.
            Then convert it to frequency and publish it as channel taps.
        """
        syncs = [item[0] for item in batch]
        carr_offsets = [item[1] for item in batch]

        nframes = len(syncs)

        X_re = np.tile(self._svr_X_re, (nframes, 1))
        X_im = np.tile(self._svr_X_im, (nframes, 1))

        y_re = np.concatenate([s.real.astype(np.float32) for s in syncs])
        y_im = np.concatenate([s.imag.astype(np.float32) for s in syncs])

        svr_re = SVR(
            kernel="rbf",
            C=self.svr_c,
            gamma=self.svr_gamma,
            epsilon=self.svr_epsilon,
        )
        svr_im = SVR(
            kernel="rbf",
            C=self.svr_c,
            gamma=self.svr_gamma,
            epsilon=self.svr_epsilon,
        )

        svr_re.fit(X_re, y_re)
        svr_im.fit(X_im, y_im)

        yhat_re = svr_re.predict(self._svr_X_re).astype(np.float32)
        yhat_im = svr_im.predict(self._svr_X_im).astype(np.float32)

        sync2_hat_time = (yhat_re + 1j * yhat_im).astype(np.complex64)
        sync2_hat_freq = self._time_to_freq_shifted(sync2_hat_time)

        # Use the most recent carrier offset in the accumulated window.
        carr_offset = int(carr_offsets[-1])

        return self._build_taps_from_estimated_sync(
            sync2_hat_freq=sync2_hat_freq,
            carr_offset=carr_offset,
        )

    def _submit_snapshot(self, sync_time_2: np.ndarray, carr_offset: int) -> None:
        """
        Store the newest sync snapshot.

        The worker always receives the most recent sliding window. This avoids
        an unbounded backlog and keeps the estimator adaptive.
        """
        with self._snapshot_lock:
            self._snapshot_buffer.append(
                (
                    np.asarray(sync_time_2, dtype=np.complex64).copy(),
                    int(carr_offset),
                )
            )
            self._pending_batch = list(self._snapshot_buffer)

        self._snapshot_event.set()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            self._snapshot_event.wait()

            if self._stop_event.is_set():
                break

            with self._snapshot_lock:
                batch = self._pending_batch
                self._pending_batch = None
                self._snapshot_event.clear()

            if batch is None:
                continue

            if len(batch) < self.svr_min_train_frames:
                self.worker_batches_skipped += 1
                continue

            self.worker_frames_started += 1
            self.worker_last_batch_size = len(batch)

            try:
                new_taps = self._estimate_chan_taps_svr_batch(batch)

                with self._latest_taps_lock:
                    self.latest_taps = new_taps.astype(np.complex64)

                self.worker_frames_completed += 1

            except Exception:
                # Keep the previous taps if the worker fails.
                # This is intentionally silent to avoid breaking the real-time flow.
                pass

    def general_work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]

        framesize = self.n_sync_symbols + self.n_data_symbols
        available_in_vectors = len(inp)
        available_out_vectors = len(out)

        max_input_frames = available_in_vectors // framesize
        max_output_frames = available_out_vectors // self.n_data_symbols
        nframes = min(max_input_frames, max_output_frames)

        if nframes <= 0:
            return 0

        nread0 = self.nitems_read(0)
        nwritten0 = self.nitems_written(0)

        produced_vectors = 0

        for f in range(nframes):
            base_in = f * framesize
            base_out = f * self.n_data_symbols

            sync_time_1 = np.asarray(inp[base_in + 0], dtype=np.complex64)
            sync_time_2 = np.asarray(inp[base_in + 1], dtype=np.complex64)

            sync_freq_1 = self._time_to_freq_shifted(sync_time_1)
            sync_freq_2 = self._time_to_freq_shifted(sync_time_2)

            carr_offset = self._get_carr_offset(sync_freq_1, sync_freq_2)

            with self._latest_taps_lock:
                chan_taps = self.latest_taps.copy()

            self._submit_snapshot(sync_time_2, carr_offset)

            for i in range(self.n_data_symbols):
                data_time = np.asarray(
                    inp[base_in + self.n_sync_symbols + i],
                    dtype=np.complex64,
                )
                out[base_out + i] = self._time_to_freq_shifted(data_time)

            write_offset = nwritten0 + base_out

            self.add_item_tag(
                0,
                write_offset,
                pmt.string_to_symbol("ofdm_sync_carr_offset"),
                pmt.from_long(int(carr_offset)),
            )

            self.add_item_tag(
                0,
                write_offset,
                pmt.string_to_symbol("ofdm_sync_chan_taps"),
                pmt.init_c32vector(self.fft_len, chan_taps.tolist()),
            )

            tags = self.get_tags_in_range(
                0,
                nread0 + base_in,
                nread0 + base_in + framesize,
            )

            for tag in tags:
                rel = int(tag.offset - (nread0 + base_in))

                if rel < self.n_sync_symbols:
                    new_rel = 0
                else:
                    new_rel = rel - self.n_sync_symbols

                key_str = pmt.symbol_to_string(tag.key)
                if key_str in ("ofdm_sync_carr_offset", "ofdm_sync_chan_taps"):
                    continue

                self.add_item_tag(
                    0,
                    write_offset + new_rel,
                    tag.key,
                    tag.value,
                )

            produced_vectors += self.n_data_symbols

        self.consume(0, nframes * framesize)
        return produced_vectors
    
class PythonSVRFrozenAsyncPrecorrectedChanEst(PythonSVRFrozenAsyncChanEst):

    """
    Async frozen SVR estimator with carrier-offset pre-correction for training.

    Difference from PythonSVRFrozenAsyncChanEst
    ------------------------------------------
    This estimator applies carrier-offset correction only to the sync symbol used
    for SVR training:

        sync_time -> FFT -> carr_offset correction -> IFFT -> SVR

    The output data symbols are NOT carrier-offset corrected here. The real
    carr_offset tag is still published so the downstream GNU Radio equalizer
    can correct the data path normally.

    This creates a new experimental estimator without modifying the existing
    svr_frozen_async mode.
    """

    def __init__(
        self,
        sync_word1,
        sync_word2,
        n_data_symbols: int,
        max_carr_offset: int = 3,
        svr_c: float = 100.0,
        svr_gamma: str = "scale",
        svr_epsilon: float = 0.001,
        svr_window_frames: int = 20,
        svr_min_train_frames: int = 5,
    ) -> None:
        super().__init__(
            sync_word1=sync_word1,
            sync_word2=sync_word2,
            n_data_symbols=n_data_symbols,
            max_carr_offset=max_carr_offset,
            svr_c=svr_c,
            svr_gamma=svr_gamma,
            svr_epsilon=svr_epsilon,
            svr_window_frames=svr_window_frames,
            svr_min_train_frames=svr_min_train_frames,
        )
        self._name = "python_svr_frozen_async_precorrected_chanest"

    def _freq_to_time_shifted(self, x_freq_shifted: np.ndarray) -> np.ndarray:
        """
        Inverse of _time_to_freq_shifted().
        """
        return np.fft.ifft(np.fft.ifftshift(x_freq_shifted)).astype(np.complex64)

    def _apply_carr_offset_freq(
        self,
        x_freq: np.ndarray,
        carr_offset: int,
    ) -> np.ndarray:
        """
        Apply carrier-offset correction in frequency domain.

        Native convention:
            received index i corresponds to reference index i - carr_offset

        Therefore:
            corrected[i - carr_offset] = received[i]
        """
        corrected = np.zeros_like(x_freq, dtype=np.complex64)

        loop_start = 0
        loop_end = self.fft_len

        if carr_offset > 0:
            loop_start = carr_offset
        elif carr_offset < 0:
            loop_end = self.fft_len + carr_offset

        for i in range(loop_start, loop_end):
            ref_idx = i - carr_offset
            corrected[ref_idx] = x_freq[i]

        return corrected

    def _build_taps_from_estimated_sync(
        self,
        sync2_hat_freq: np.ndarray,
        carr_offset: int | None = None,
    ) -> np.ndarray:
        """
        Build the taps from an already carrier-offset-corrected estimated sync.

        Since the training sync was pre-corrected before the SVR, the estimated
        frequency-domain sync is already in the corrected reference frame.

        Experimental convention:
        - Do NOT divide by sync_word2.
        - Do NOT apply carr_offset again here.
        - Publish sync2_hat_freq on active sync carriers as taps.
        """
        chan_taps = np.zeros(self.fft_len, dtype=np.complex64)

        nz = np.abs(self.sync_word2) > 0
        chan_taps[nz] = sync2_hat_freq[nz] / self.sync_word2[nz]

        return chan_taps

    def _submit_snapshot(self, sync_time_2_corrected: np.ndarray) -> None:
        """
        Store the newest carrier-offset-corrected sync snapshot.
        """
        with self._snapshot_lock:
            self._snapshot_buffer.append(
                np.asarray(sync_time_2_corrected, dtype=np.complex64).copy()
            )
            self._pending_batch = list(self._snapshot_buffer)

        self._snapshot_event.set()

    def _estimate_chan_taps_svr_batch(self, batch) -> np.ndarray:
        """
        Train SVR using a sliding window of already corrected sync symbols.
        """
        syncs = batch
        nframes = len(syncs)

        X_re = np.tile(self._svr_X_re, (nframes, 1))
        X_im = np.tile(self._svr_X_im, (nframes, 1))

        y_re = np.concatenate([s.real.astype(np.float32) for s in syncs])
        y_im = np.concatenate([s.imag.astype(np.float32) for s in syncs])

        svr_re = SVR(
            kernel="rbf",
            C=self.svr_c,
            gamma=self.svr_gamma,
            epsilon=self.svr_epsilon,
        )
        svr_im = SVR(
            kernel="rbf",
            C=self.svr_c,
            gamma=self.svr_gamma,
            epsilon=self.svr_epsilon,
        )

        svr_re.fit(X_re, y_re)
        svr_im.fit(X_im, y_im)

        yhat_re = svr_re.predict(self._svr_X_re).astype(np.float32)
        yhat_im = svr_im.predict(self._svr_X_im).astype(np.float32)

        sync2_hat_time = (yhat_re + 1j * yhat_im).astype(np.complex64)
        sync2_hat_freq = self._time_to_freq_shifted(sync2_hat_time)

        return self._build_taps_from_estimated_sync(sync2_hat_freq)

    def general_work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]

        framesize = self.n_sync_symbols + self.n_data_symbols
        available_in_vectors = len(inp)
        available_out_vectors = len(out)

        max_input_frames = available_in_vectors // framesize
        max_output_frames = available_out_vectors // self.n_data_symbols
        nframes = min(max_input_frames, max_output_frames)

        if nframes <= 0:
            return 0

        nread0 = self.nitems_read(0)
        nwritten0 = self.nitems_written(0)

        produced_vectors = 0

        for f in range(nframes):
            base_in = f * framesize
            base_out = f * self.n_data_symbols

            sync_time_1 = np.asarray(inp[base_in + 0], dtype=np.complex64)
            sync_time_2 = np.asarray(inp[base_in + 1], dtype=np.complex64)

            # FFT only to estimate carrier offset.
            sync_freq_1 = self._time_to_freq_shifted(sync_time_1)
            sync_freq_2 = self._time_to_freq_shifted(sync_time_2)

            carr_offset = self._get_carr_offset(sync_freq_1, sync_freq_2)

            # Apply carrier-offset correction only to the sync used for SVR.
            sync_freq_2_corrected = self._apply_carr_offset_freq(
                sync_freq_2,
                carr_offset,
            )
            sync_time_2_corrected = self._freq_to_time_shifted(sync_freq_2_corrected)

            with self._latest_taps_lock:
                chan_taps = self.latest_taps.copy()

            # Queue corrected sync for async SVR training.
            self._submit_snapshot(sync_time_2_corrected)

            # Data path: FFT only. Do NOT apply carr_offset here.
            # The downstream equalizer will apply the published carr_offset.
            for i in range(self.n_data_symbols):
                data_time = np.asarray(
                    inp[base_in + self.n_sync_symbols + i],
                    dtype=np.complex64,
                )
                out[base_out + i] = self._time_to_freq_shifted(data_time)

            write_offset = nwritten0 + base_out

            # Publish real carr_offset, because equalizer still needs it.
            self.add_item_tag(
                0,
                write_offset,
                pmt.string_to_symbol("ofdm_sync_carr_offset"),
                pmt.from_long(int(carr_offset)),
            )

            self.add_item_tag(
                0,
                write_offset,
                pmt.string_to_symbol("ofdm_sync_chan_taps"),
                pmt.init_c32vector(self.fft_len, chan_taps.tolist()),
            )

            tags = self.get_tags_in_range(
                0,
                nread0 + base_in,
                nread0 + base_in + framesize,
            )

            for tag in tags:
                rel = int(tag.offset - (nread0 + base_in))

                if rel < self.n_sync_symbols:
                    new_rel = 0
                else:
                    new_rel = rel - self.n_sync_symbols

                key_str = pmt.symbol_to_string(tag.key)
                if key_str in ("ofdm_sync_carr_offset", "ofdm_sync_chan_taps"):
                    continue

                self.add_item_tag(
                    0,
                    write_offset + new_rel,
                    tag.key,
                    tag.value,
                )

            produced_vectors += self.n_data_symbols

        self.consume(0, nframes * framesize)
        return produced_vectors

class PythonSVRFrequencyAsyncChanEst(PythonSVRFrozenAsyncChanEst):
    """
    Async frozen SVR estimator operating directly in frequency domain.

    Input:
        [sync_freq_1, sync_freq_2, data_freq_0, ...]

    Output:
        [data_freq_0, ...]

    Behavior:
    - Receives symbols after GNU Radio FFT.
    - Computes carrier offset like LS.
    - Uses a sliding window of sync_freq_2 symbols.
    - Trains SVR in frequency domain.
    - Publishes latest estimated channel taps.
    - Publishes real carr_offset for downstream equalizer.
    """

    def __init__(
        self,
        sync_word1,
        sync_word2,
        n_data_symbols: int,
        max_carr_offset: int = 3,
        svr_c: float = 100.0,
        svr_gamma: str = "scale",
        svr_epsilon: float = 0.001,
        svr_window_frames: int = 20,
        svr_min_train_frames: int = 5,
    ) -> None:
        super().__init__(
            sync_word1=sync_word1,
            sync_word2=sync_word2,
            n_data_symbols=n_data_symbols,
            max_carr_offset=max_carr_offset,
            svr_c=svr_c,
            svr_gamma=svr_gamma,
            svr_epsilon=svr_epsilon,
            svr_window_frames=svr_window_frames,
            svr_min_train_frames=svr_min_train_frames,
        )
        self._name = "python_svr_frequency_async_chanest"

        self._svr_X_freq, self._active_idx_freq = self._build_frequency_feature_matrix()

    def _build_frequency_feature_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build features only for active sync carriers.

        Training only on active carriers avoids teaching the SVR that most
        frequency bins should be zero.
        """
        active_idx = np.flatnonzero(np.abs(self.sync_word2) > 0).astype(np.int32)

        k = active_idx.astype(np.float32)
        k_norm = (k - (self.fft_len / 2.0)) / (self.fft_len / 2.0)

        X_active = np.column_stack([k_norm]).astype(np.float32)

        return X_active, active_idx

    def _submit_snapshot(self, sync_freq_2: np.ndarray, carr_offset: int) -> None:
        """
        Store newest frequency-domain sync snapshot.
        """
        with self._snapshot_lock:
            self._snapshot_buffer.append(
                (
                    np.asarray(sync_freq_2, dtype=np.complex64).copy(),
                    int(carr_offset),
                )
            )
            self._pending_batch = list(self._snapshot_buffer)

        self._snapshot_event.set()

    def _estimate_chan_taps_svr_batch(self, batch) -> np.ndarray:
        """
        Train SVR directly in frequency domain.

        The target is the LS-like channel estimate:

            H[k] = Y[k + carr_offset] / X[k]

        where X is sync_word2 and Y is received sync_freq_2.
        """
        H_list = []

        for sync_freq_2, carr_offset in batch:
            H = np.zeros(self.fft_len, dtype=np.complex64)

            loop_start = 0
            loop_end = self.fft_len

            if carr_offset > 0:
                loop_start = carr_offset
            elif carr_offset < 0:
                loop_end = self.fft_len + carr_offset

            for i in range(loop_start, loop_end):
                ref_idx = i - carr_offset
                if self.sync_word2[ref_idx] != 0:
                    H[ref_idx] = sync_freq_2[i] / self.sync_word2[ref_idx]

            H_list.append(H)

        nframes = len(H_list)
        active = self._active_idx_freq

        X = np.tile(self._svr_X_freq, (nframes, 1))
        y_re = np.concatenate([H[active].real.astype(np.float32) for H in H_list])
        y_im = np.concatenate([H[active].imag.astype(np.float32) for H in H_list])

        svr_re = SVR(
            kernel="rbf",
            C=self.svr_c,
            gamma=self.svr_gamma,
            epsilon=self.svr_epsilon,
        )
        svr_im = SVR(
            kernel="rbf",
            C=self.svr_c,
            gamma=self.svr_gamma,
            epsilon=self.svr_epsilon,
        )

        svr_re.fit(X, y_re)
        svr_im.fit(X, y_im)

        Hhat_re = svr_re.predict(self._svr_X_freq).astype(np.float32)
        Hhat_im = svr_im.predict(self._svr_X_freq).astype(np.float32)

        Hhat_active = (Hhat_re + 1j * Hhat_im).astype(np.complex64)

        chan_taps = np.zeros(self.fft_len, dtype=np.complex64)
        chan_taps[active] = Hhat_active

        return chan_taps

    def general_work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]

        framesize = self.n_sync_symbols + self.n_data_symbols
        available_in_vectors = len(inp)
        available_out_vectors = len(out)

        max_input_frames = available_in_vectors // framesize
        max_output_frames = available_out_vectors // self.n_data_symbols
        nframes = min(max_input_frames, max_output_frames)

        if nframes <= 0:
            return 0

        nread0 = self.nitems_read(0)
        nwritten0 = self.nitems_written(0)

        produced_vectors = 0

        for f in range(nframes):
            base_in = f * framesize
            base_out = f * self.n_data_symbols

            sync_freq_1 = np.asarray(inp[base_in + 0], dtype=np.complex64)
            sync_freq_2 = np.asarray(inp[base_in + 1], dtype=np.complex64)

            carr_offset = self._get_carr_offset(sync_freq_1, sync_freq_2)

            with self._latest_taps_lock:
                chan_taps = self.latest_taps.copy()

            self._submit_snapshot(sync_freq_2, carr_offset)

            # This estimator receives frequency-domain vectors, so just copy
            # data symbols exactly like native chanest.
            out[base_out : base_out + self.n_data_symbols] = inp[
                base_in + self.n_sync_symbols : base_in + self.n_sync_symbols + self.n_data_symbols
            ]

            write_offset = nwritten0 + base_out

            self.add_item_tag(
                0,
                write_offset,
                pmt.string_to_symbol("ofdm_sync_carr_offset"),
                pmt.from_long(int(carr_offset)),
            )

            self.add_item_tag(
                0,
                write_offset,
                pmt.string_to_symbol("ofdm_sync_chan_taps"),
                pmt.init_c32vector(self.fft_len, chan_taps.tolist()),
            )

            tags = self.get_tags_in_range(
                0,
                nread0 + base_in,
                nread0 + base_in + framesize,
            )

            for tag in tags:
                rel = int(tag.offset - (nread0 + base_in))

                if rel < self.n_sync_symbols:
                    new_rel = 0
                else:
                    new_rel = rel - self.n_sync_symbols

                key_str = pmt.symbol_to_string(tag.key)
                if key_str in ("ofdm_sync_carr_offset", "ofdm_sync_chan_taps"):
                    continue

                self.add_item_tag(
                    0,
                    write_offset + new_rel,
                    tag.key,
                    tag.value,
                )

            produced_vectors += self.n_data_symbols

        self.consume(0, nframes * framesize)
        return produced_vectors    

# =============================================================================
