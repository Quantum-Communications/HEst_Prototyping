"""Counts-first summaries and paired, cluster-level contrasts."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import read_json, write_json


def mean_known(rows, name):
    """Do not reconstruct timing components absent from historical records."""
    return (
        float(np.mean([r[name] for r in rows]))
        if all(r.get(name) is not None for r in rows)
        else None
    )


def summarize(rows):
    """Pool integer error counts, retaining source kind and experiment condition."""
    if not rows:
        raise ValueError("Empty experiment")
    groups = defaultdict(list)
    keys = set()
    for row in rows:
        key = tuple(
            row[k]
            for k in ("source_kind", "session_id", "burst", "condition", "equalizer", "method")
        )
        if key in keys:
            raise ValueError("Duplicated experimental cell")
        keys.add(key)
        if not 0 <= row["delivered"] <= row["matched"] <= row["transmitted"]:
            raise ValueError("Invalid packet counts")
        groups[
            tuple(row[k] for k in ("source_kind", "condition", "equalizer", "method"))
        ].append(row)
    output = []
    for key, group in sorted(groups.items()):
        n = sum(r["transmitted"] for r in group)
        matched = sum(r["matched"] for r in group)
        correct = sum(r["delivered"] for r in group)
        observed = sum(r["observed_bit_errors"] for r in group)
        bits = sum(r["matched_bits"] for r in group)
        expected = sum(r["expected_bits"] for r in group)
        evm_present = all("evm_reference_energy" in r for r in group)
        evm_error = sum(r.get("evm_error_energy", 0) for r in group)
        evm_reference = sum(r.get("evm_reference_energy", 0) for r in group)
        output.append(
            dict(zip(("source_kind", "condition", "equalizer", "method"), key, strict=True))
            | dict(
                bursts=len(group),
                sessions=len({r["session_id"] for r in group}),
                transmitted=n,
                matched=matched,
                delivered=correct,
                per=(n - correct) / n,
                observed_bit_errors=observed,
                matched_bits=bits,
                expected_bits=expected,
                ber_conditional=observed / bits if bits else None,
                ber_lower=observed / expected,
                ber_upper=(observed + expected - bits) / expected,
                evm_rms=(evm_error / evm_reference) ** 0.5
                if evm_present and evm_reference
                else None,
                evm_symbols=sum(r.get("evm_symbols", 0) for r in group)
                if evm_present
                else None,
                mean_receiver_seconds=float(np.mean([r["receiver_seconds"] for r in group])),
                mean_fit_seconds=float(np.mean([r["fit_seconds"] for r in group])),
                mean_setup_seconds=float(np.mean([r["initialization_seconds"] for r in group])),
                mean_kernel_seconds=mean_known(group, "kernel_seconds"),
                mean_backend_import_seconds=mean_known(group, "backend_import_seconds"),
                mean_training_seconds=mean_known(group, "training_seconds"),
                mean_prediction_seconds=mean_known(group, "prediction_seconds"),
                mean_worker_wall_seconds=mean_known(group, "worker_wall_seconds"),
                parallel_workers=sorted({r["parallel_workers"] for r in group}),
                timing_scope="Observed host wall time; use workers=1 for standalone method cost",
            )
        )
    return output


def paired_contrasts(rows, seed=550055, draws=5000, *, reference_method="ls"):
    """Resample physical sessions when available, otherwise bursts within one session.

    This interval quantifies the recorded cluster set only; it does not transform
    one placement into evidence of independent channel generalization.
    """
    table = {
        (
            r["source_kind"],
            r["session_id"],
            r["burst"],
            r["condition"],
            r["equalizer"],
            r["method"],
        ): r
        for r in rows
    }
    if len(table) != len(rows):
        raise ValueError("Duplicated experimental cell")
    groups = defaultdict(list)
    for key, row in table.items():
        if row["method"] == reference_method:
            continue
        reference = table.get((*key[:-1], reference_method))
        if reference is None or any(
            row[k] != reference[k]
            for k in (
                "raw_sha256",
                "replay_sha256",
                "corrected_sha256",
                "trigger_sha256",
                "transmitted",
                "expected_bits",
            )
        ):
            raise ValueError(f"Unpaired contrast against {reference_method}")
        groups[(row["source_kind"], row["condition"], row["equalizer"], row["method"])].append(
            (row, reference)
        )
    rng = np.random.default_rng(seed)
    results = []
    for key, group in sorted(groups.items()):
        clusters = defaultdict(list)
        sessions = {a["session_id"] for a, b in group}
        for a, b in group:
            unit = a["session_id"] if len(sessions) > 1 else (a["session_id"], a["burst"])
            clusters[unit].append(a["per"] - b["per"])
        values = np.array([np.mean(v) for v in clusters.values()])
        interval = None
        if len(values) > 1:
            samples = rng.integers(0, len(values), size=(draws, len(values)))
            interval = np.quantile(values[samples].mean(axis=1), [0.025, 0.975]).tolist()
        results.append(
            dict(
                source_kind=key[0],
                condition=key[1],
                equalizer=key[2],
                method=key[3],
                reference_method=reference_method,
                delta_per=float(values.mean()),
                bootstrap_95=interval,
                clusters=len(values),
                cluster_unit="session" if len(sessions) > 1 else "burst_within_one_session",
                estimand="Mean cluster PER difference; equal weights for recorded sessions, otherwise bursts",
                scope="Paired exploratory interval; no multiplicity adjustment",
            )
        )
    return results


def generate(folder, *, plots=False):
    folder = Path(folder)
    rows = read_json(folder / "rows.json")
    summary = summarize(rows)
    contrasts = paired_contrasts(rows)
    write_json(folder / "summary.json", summary, replace=True)
    write_json(folder / "contrasts.json", contrasts, replace=True)
    controls = []
    present = {r["method"] for r in rows}
    if "lmmse" in present:
        controls.extend(paired_contrasts(rows, reference_method="lmmse"))
    write_json(folder / "control_contrasts.json", controls, replace=True)
    with (folder / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for kind in sorted({r["source_kind"] for r in summary}):
            for equalizer in sorted({r["equalizer"] for r in summary}):
                selected = [
                    r
                    for r in summary
                    if r["equalizer"] == equalizer and r["source_kind"] == kind
                ]
                if not selected:
                    continue
                methods = sorted({r["method"] for r in selected}, key=lambda m: (m != "ls", m))
                conditions = sorted(
                    {r["condition"] for r in selected}, key=lambda c: (c != "natural", c)
                )
                table = {(r["method"], r["condition"]): r for r in selected}
                values = np.full((len(methods), len(conditions)), np.nan)
                for i, method in enumerate(methods):
                    for j, condition in enumerate(conditions):
                        if (method, condition) in table:
                            values[i, j] = table[(method, condition)]["per"]
                fig, ax = plt.subplots(
                    figsize=(max(6, len(conditions) * 1.4), max(4, len(methods) * 0.42 + 2))
                )
                color = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
                ax.set_xticks(np.arange(len(conditions)), conditions, rotation=35, ha="right")
                ax.set_yticks(np.arange(len(methods)), methods)
                for i in range(len(methods)):
                    for j in range(len(conditions)):
                        v = values[i, j]
                        ax.text(
                            j,
                            i,
                            f"{v:.3f}" if np.isfinite(v) else "missing",
                            ha="center",
                            va="center",
                            color="white" if v < 0.5 else "black",
                        )
                ax.set_title(f"{equalizer} — {kind}")
                fig.colorbar(color, ax=ax, label="Packet error rate (all transmitted packets)")
                fig.tight_layout()
                fig.savefig(folder / f"per_{equalizer}_{kind}.png", dpi=160)
                fig.savefig(folder / f"per_{equalizer}_{kind}.svg")
                plt.close(fig)
    return summary
