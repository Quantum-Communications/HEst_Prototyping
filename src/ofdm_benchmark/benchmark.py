"""Immutable IQ replay and bounded parallel, isolated receiver processes."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import Experiment
from .frontend import verify
from .io import read_json, sha256, write_json
from .metrics import score, score_evm, score_header_stages
from .noise import generate


def isolated(job: dict, job_path: Path, *, acquire=False) -> dict:
    """Run one fresh process, bound runtime and preserve its stdout/errors."""
    command = [sys.executable, "-m", "ofdm_benchmark.worker", "--job", str(job_path.resolve())]
    if job["action"] == "capture":
        if not acquire:
            raise ValueError("A aquisição física requer autorização explícita")
        command.append("--acquire")
    job_path = job_path.resolve()
    write_json(job_path, job)
    environment = dict(os.environ)
    environment.update(
        {
            k: "2"
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
    )
    start = time.perf_counter()
    with job_path.with_suffix(".log").open("x") as log:
        try:
            result = subprocess.run(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=job["timeout"] + 15,
                check=False,
            )
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    return dict(
        ok=code == 0,
        returncode=code,
        worker_wall_seconds=time.perf_counter() - start,
        job=str(job_path),
        pid_scope="fresh subprocess; 2 BLAS/OpenMP threads",
    )


def base_job(cfg):
    return dict(
        phy=asdict(cfg.phy),
        required_gnu_version=cfg.required_gnu_version,
        timeout=cfg.timeout_seconds,
    )


def compare_native(first, second):
    """Compare all header decisions, packets and traces, including rejected attempts."""
    a, b = read_json(Path(first) / "receiver.json"), read_json(Path(second) / "receiver.json")
    semantic = all(a[key] == b[key] for key in ("headers", "packets", "native_payload_hex"))
    arrays = {}
    with (
        np.load(Path(first) / "traces.npz", allow_pickle=False) as x,
        np.load(Path(second) / "traces.npz", allow_pickle=False) as y,
    ):
        if set(x.files) != set(y.files):
            return dict(passed=False, error="Trace schema mismatch")
        for key in x.files:
            same = x[key].shape == y[key].shape
            finite = bool(np.isfinite(x[key]).all() and np.isfinite(y[key]).all())
            # Equivalence of undefined values is reported separately from validity.
            close = same and np.allclose(x[key], y[key], rtol=2e-5, atol=2e-5, equal_nan=True)
            arrays[key] = dict(shape_equal=same, all_finite=finite, equivalent=bool(close))
    return dict(
        passed=bool(
            semantic
            and a["technical_valid"]
            and b["technical_valid"]
            and all(v["equivalent"] and v["all_finite"] for v in arrays.values())
        ),
        semantic=semantic,
        arrays=arrays,
    )


def replay_capture(
    capture: Path, output: Path, cfg: Experiment, *, working: Path | None = None
):
    """Reuse each burst/condition IQ across the entire method/equalizer matrix.

    Receiver jobs contain no truth path. Scoring starts only after every receiver
    in the condition has completed. The worker ledger retains failed cells.
    A failed matrix cannot silently become a smaller successful experiment.
    """
    capture, output = Path(capture).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    meta = read_json(capture / "capture.json")
    raw_path, truth_path = capture / "capture.c64", capture / "tx/truth.json"
    if sha256(raw_path) != meta["raw_sha256"] or sha256(truth_path) != meta["truth_sha256"]:
        raise RuntimeError("Capture or truth checksum mismatch")
    if meta["phy"] != asdict(cfg.phy):
        raise ValueError("Replay PHY differs from capture; import explicit metadata first")
    if not meta["physical_valid"]:
        raise ValueError("Technically invalid capture cannot enter performance summaries")
    from .preflight import check

    write_json(output / "preflight.json", check(cfg))
    write_json(output / "config.json", cfg.to_dict())
    write_json(output / "capture_identity.json", meta)
    rows, ledger = [], []
    raw = np.memmap(raw_path, dtype="<c8", mode="r")
    for condition in cfg.conditions:
        cell = output / condition.label
        cell.mkdir()
        with tempfile.TemporaryDirectory(prefix="ofdm_replay_", dir=working) as scratch:
            temp = Path(scratch)
            dc = meta["power"]["dc_offset"]
            natural = np.array(raw, dtype=np.complex64, copy=True)
            natural -= complex(dc["real"], dc["imag"])
            offset = {"natural": 0, "awgn": 7, "bernoulli_gaussian": 17, "markov_gaussian": 23}[
                condition.name
            ]
            noise, notes = generate(
                len(raw),
                condition,
                cfg.seed + 10007 * meta["burst"] + offset,
                meta["power"]["signal_power_proxy"],
            )
            natural += noise
            input_path = temp / "input.c64"
            natural.tofile(input_path)
            del natural, noise
            notes.update(
                input_sha256=sha256(input_path), raw_sha256=meta["raw_sha256"], dc_removed=dc
            )
            write_json(cell / "impairment.json", notes)
            front = temp / "frontend"
            frontend_job = dict(
                base_job(cfg), action="frontend", iq=str(input_path), output=str(front)
            )
            outcome = isolated(frontend_job, cell / "frontend.job.json")
            if not outcome["ok"]:
                write_json(output / "failure.json", outcome)
                raise RuntimeError("Frontend failed; see job log")
            sealed = verify(front)
            write_json(cell / "frontend.json", sealed)
            jobs = []
            for eq in cfg.equalizers:
                jobs.append(
                    (
                        "gnu",
                        eq,
                        dict(
                            base_job(cfg),
                            action="receiver",
                            method="gnu",
                            equalizer=eq,
                            frontend=str(front),
                            output=str(cell / f"gnu__{eq}"),
                        ),
                    )
                )
                for method in cfg.methods:
                    jobs.append(
                        (
                            method.name,
                            eq,
                            dict(
                                base_job(cfg),
                                action="receiver",
                                method=asdict(method),
                                equalizer=eq,
                                frontend=str(front),
                                output=str(cell / f"{method.name}__{eq}"),
                            ),
                        )
                    )
            finished = {}
            with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
                futures = {
                    executor.submit(isolated, job, cell / f"{name}__{eq}.job.json"): (name, eq)
                    for name, eq, job in jobs
                }
                for future in as_completed(futures):
                    key = futures[future]
                    finished[key] = future.result()
                    ledger.append(
                        dict(
                            condition=condition.label,
                            method=key[0],
                            equalizer=key[1],
                            **finished[key],
                        )
                    )
                    write_json(output / "worker_ledger.json", ledger, replace=True)
            verify(front)
            if not all(item["ok"] for item in finished.values()):
                write_json(
                    output / "failure.json",
                    dict(
                        reason="One or more receiver jobs failed",
                        scope="No successful-only aggregate",
                    ),
                )
                raise RuntimeError("Receiver matrix incomplete; inspect worker_ledger.json")
            parity = {
                eq: compare_native(cell / f"gnu__{eq}", cell / f"ls__{eq}")
                for eq in cfg.equalizers
            }
            write_json(cell / "native_parity.json", parity)
            if not all(item["passed"] for item in parity.values()):
                raise RuntimeError("Native LS parity/finite-domain check failed")
            # This is the first opening of TX payload truth in the replay pipeline.
            truth = read_json(truth_path)
            for method in cfg.methods:
                for eq in cfg.equalizers:
                    receiver = read_json(cell / f"{method.name}__{eq}" / "receiver.json")
                    if not receiver["technical_valid"]:
                        raise RuntimeError(f"Technically invalid receiver: {method.name}/{eq}")
                    metrics = score(truth, receiver)
                    metrics.update(score_header_stages(truth, receiver))
                    metrics.update(
                        score_evm(truth, receiver, cell / f"{method.name}__{eq}" / "traces.npz")
                    )
                    write_json(cell / f"{method.name}__{eq}" / "metrics.json", metrics)
                    brief = {
                        key: value
                        for key, value in metrics.items()
                        if key not in ("packets", "definition")
                    }
                    diag = receiver["estimator_diagnostics"]
                    rows.append(
                        dict(
                            burst=meta["burst"],
                            session_id=meta["session_id"],
                            source_kind=meta["source_kind"],
                            condition=condition.label,
                            method=method.name,
                            equalizer=eq,
                            raw_sha256=meta["raw_sha256"],
                            replay_sha256=notes["input_sha256"],
                            corrected_sha256=sealed["hashes"]["corrected.c64"],
                            trigger_sha256=sealed["hashes"]["triggers.u8"],
                            raw_triggers=sealed["raw_triggers"],
                            header_attempts=len(receiver["headers"]),
                            accepted_headers=sum(h["accepted"] for h in receiver["headers"]),
                            receiver_seconds=receiver["seconds"],
                            fit_seconds=diag["fit_seconds"],
                            initialization_seconds=diag["initialization_seconds"],
                            kernel_seconds=diag["kernel_seconds"],
                            backend_import_seconds=diag.get("backend_import_seconds"),
                            training_seconds=diag.get("training_seconds"),
                            prediction_seconds=diag.get("prediction_seconds"),
                            fit_calls=diag["fit_calls"],
                            fit_failures=diag["fit_failures"],
                            worker_wall_seconds=finished[(method.name, eq)][
                                "worker_wall_seconds"
                            ],
                            parallel_workers=cfg.workers,
                            **brief,
                        )
                    )
    if sha256(raw_path) != meta["raw_sha256"]:
        raise RuntimeError("Original capture was modified")
    write_json(output / "rows.json", rows)
    write_json(
        output / "complete.json",
        dict(
            status="complete",
            cells=len(rows),
            raw_sha256=meta["raw_sha256"],
            source_kind=meta["source_kind"],
            execution_kind="offline_replay",
        ),
    )
    return rows


def run_campaign(cfg, output, *, acquire=False):
    """Capture a whole RAM batch, stop radios, retain it, then run parallel replay."""
    if not acquire:
        raise ValueError("A aquisição física requer autorização explícita")
    from .transmit import protocol

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from .preflight import check

    write_json(output / "preflight.json", check(cfg, hardware=True))
    root = Path(cfg.ram_root)
    # Linux tmpfs validation is explicit; a fast disk is not silently called RAM.
    mounts = [line.split() for line in Path("/proc/mounts").read_text().splitlines()]
    matches = [m for m in mounts if root.resolve() == Path(m[1]).resolve()]
    if not matches or matches[0][2] != "tmpfs":
        raise ValueError("ram_root must be the actual mount point of tmpfs")
    write_json(output / "config.json", cfg.to_dict())
    plan = protocol(cfg.phy, cfg.measurement_packets, cfg.tx_warmup_seconds)
    samples = round(
        (cfg.rx_warmup_seconds + plan["nominal_burst_seconds"] + 0.35 + cfg.rx_tail_seconds)
        * cfg.phy.sample_rate
    )
    needed = samples * 8 * (cfg.batch_size + 6) + 256 * 2**20
    events, all_rows = [], []
    for start in range(1, cfg.bursts + 1, cfg.batch_size):
        if shutil.disk_usage(root).free < needed:
            raise MemoryError(f"Insufficient tmpfs space; require {needed} bytes")
        with tempfile.TemporaryDirectory(prefix="ofdm_batch_", dir=root) as tempname:
            temp = Path(tempname)
            captured = []
            try:
                for burst in range(start, min(cfg.bursts + 1, start + cfg.batch_size)):
                    events.append(
                        dict(event="capture_start", burst=burst, monotonic=time.monotonic())
                    )
                    job = dict(
                        base_job(cfg),
                        action="capture",
                        experiment=cfg.to_dict(),
                        burst=burst,
                        output=str(temp / f"burst_{burst:03d}"),
                    )
                    result = isolated(job, output / f"capture_{burst:03d}.job.json", acquire=acquire)
                    events.append(
                        dict(
                            event="capture_stop",
                            burst=burst,
                            monotonic=time.monotonic(),
                            ok=result["ok"],
                        )
                    )
                    write_json(output / "events.json", events, replace=True)
                    if not result["ok"]:
                        raise RuntimeError("Capture failed; no automatic replacement")
                    captured.append(temp / f"burst_{burst:03d}")
                # No copy, estimator, frontend or analysis process runs during capture.
                for folder in captured:
                    retained = output / "captures" / folder.name
                    shutil.copytree(folder, retained)
                    meta = read_json(retained / "capture.json")
                    if sha256(retained / "capture.c64") != meta["raw_sha256"]:
                        raise RuntimeError("Retained capture copy mismatch")
                events.append(
                    dict(
                        event="processing_start",
                        bursts=[read_json(p / "capture.json")["burst"] for p in captured],
                        monotonic=time.monotonic(),
                    )
                )
                write_json(output / "events.json", events, replace=True)
                for folder in captured:
                    all_rows.extend(
                        replay_capture(
                            folder, output / "replay" / folder.name, cfg, working=root
                        )
                    )
                events.append(dict(event="processing_stop", monotonic=time.monotonic()))
                write_json(output / "events.json", events, replace=True)
            except BaseException:
                failed = output / f"failed_batch_{start:03d}"
                shutil.copytree(temp, failed, dirs_exist_ok=True)
                raise
    write_json(output / "rows.json", all_rows)
    write_json(
        output / "complete.json",
        dict(status="complete", bursts=cfg.bursts, cells=len(all_rows)),
    )
    return all_rows
