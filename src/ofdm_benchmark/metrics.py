"""Post-reception packet association and honest treatment of missing bits."""

from __future__ import annotations


def score(truth: dict, receiver: dict) -> dict:
    """Associate the first complete payload for each decoded public header ID.

    truth contains all sent packets and a fixed measurement ID list. This module
    is called only after receiver completion. It does not select best duplicates,
    recover IDs from payload truth, or remove corrupt packets from the denominator.
    """
    expected = {
        int(p["header_sequence"]): bytes.fromhex(p["payload_hex"]) for p in truth["packets"]
    }
    if len(expected) != len(truth["packets"]) or len(expected) > 4096:
        raise ValueError("Header IDs must not wrap or repeat within a capture")
    ids = list(map(int, truth["measurement_ids"]))
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= expected.keys():
        raise ValueError("Invalid measurement cohort")
    sizes = {len(expected[key]) for key in ids}
    if len(sizes) != 1:
        raise ValueError("Mixed payload sizes")
    seen, duplicates, foreign = {}, 0, 0
    for packet in receiver["packets"]:
        key = int(packet["parsed"]["packet_num"])
        if key not in expected:
            foreign += 1
        elif key in ids:
            if key in seen:
                duplicates += 1
            else:
                value = bytes.fromhex(packet["payload_hex"])
                if len(value) != len(expected[key]):
                    raise ValueError("Public payload-length contract violated")
                seen[key] = value
    records = []
    for key in ids:
        received = seen.get(key)
        errors = (
            None
            if received is None
            else sum((a ^ b).bit_count() for a, b in zip(expected[key], received, strict=True))
        )
        records.append(
            dict(
                header_sequence=key,
                matched=received is not None,
                delivered=errors == 0,
                bit_errors=errors,
            )
        )
    n, matched = len(ids), len(seen)
    bits_per_packet = 8 * next(iter(sizes))
    errors = sum(r["bit_errors"] or 0 for r in records)
    correct = sum(r["delivered"] for r in records)
    expected_bits, matched_bits = n * bits_per_packet, matched * bits_per_packet
    # BER condicional usa bits associados. PER inclui ausências e payloads com erro.
    missing_bits = expected_bits - matched_bits
    return dict(
        transmitted=n,
        matched=matched,
        delivered=correct,
        missing=n - matched,
        observed_bit_errors=errors,
        matched_bits=matched_bits,
        expected_bits=expected_bits,
        per=(n - correct) / n,
        erasure_rate=(n - matched) / n,
        ber_conditional=errors / matched_bits if matched_bits else None,
        ber_lower=errors / expected_bits,
        ber_upper=(errors + missing_bits) / expected_bits,
        duplicates=duplicates,
        foreign_sequences=foreign,
        packets=records,
        definition="No payload CRC: delivered is post-hoc bit-perfect recovery; missing bits are unknown",
    )


def score_header_stages(truth, receiver):
    """Separate identifiable measurement headers from all-capture attempts.

    Rejected headers have no trustworthy sequence ID. Their association with
    the measurement cohort is unknown; do not count matched payloads as raw
    synchronization acquisitions.
    """
    measurement = set(map(int, truth["measurement_ids"]))
    expected = {int(p["header_sequence"]) for p in truth["packets"]}
    accepted = [int(h["parsed"]["packet_num"]) for h in receiver["headers"] if h["accepted"]]
    selected = [key for key in accepted if key in measurement]
    return dict(
        measurement_unique_accepted_headers=len(set(selected)),
        measurement_accepted_header_events=len(selected),
        warmup_accepted_header_events=sum(key in expected - measurement for key in accepted),
        rejected_header_attempts_all_capture=sum(
            not h["accepted"] for h in receiver["headers"]
        ),
        stage_scope="Raw triggers and header attempts cover warm-up plus measurement; rejected IDs cannot be assigned reliably",
    )


def score_evm(truth, receiver, trace_path):
    """Measure post-equalizer EVM on the same first-match cohort.

    TX values are opened after decoding. No gain/phase alignment is fitted for
    scoring. Padding, warm-up, foreign IDs and later duplicates are excluded.
    Missing payloads have unknown EVM, just as they have unknown bit errors.
    """
    import numpy as np

    from .phy import DATA, QPSK, bytes_to_classes

    expected = {
        int(p["header_sequence"]): bytes.fromhex(p["payload_hex"]) for p in truth["packets"]
    }
    measurement = set(map(int, truth["measurement_ids"]))
    with np.load(trace_path, allow_pickle=False) as trace:
        soft = np.asarray(trace["payload_soft"])
    needed = sum(p["parsed"]["frame_length"] for p in receiver["packets"])
    if soft.shape != (needed, 64) and not (needed == 0 and soft.size == 0):
        raise ValueError("Payload trace does not match complete receiver packet order")
    cursor, seen, error_energy, reference_energy, symbols = 0, set(), 0.0, 0.0, 0
    for packet in receiver["packets"]:
        length = packet["parsed"]["frame_length"]
        block = soft[cursor : cursor + length]
        cursor += length
        key = int(packet["parsed"]["packet_num"])
        if key not in measurement or key in seen:
            continue
        seen.add(key)
        reference = QPSK[bytes_to_classes(expected[key])]
        predicted = block[:, DATA].ravel()[: len(reference)]
        if len(predicted) != len(reference) or not np.isfinite(predicted).all():
            raise ValueError("Nonfinite or incomplete EVM cohort")
        error_energy += float(np.sum(abs(predicted - reference) ** 2))
        reference_energy += float(np.sum(abs(reference) ** 2))
        symbols += len(reference)
    return dict(
        evm_rms=(error_energy / reference_energy) ** 0.5 if reference_energy else None,
        evm_error_energy=error_energy,
        evm_reference_energy=reference_energy,
        evm_symbols=symbols,
        evm_matched_packets=len(seen),
        evm_scope="Post-equalizer, first complete matched payloads; no post-hoc gain/phase correction",
    )
