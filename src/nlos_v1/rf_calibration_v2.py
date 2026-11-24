#!/usr/bin/env python3
"""Calibração física do enlace com busca de ganhos e validação em rajadas reservadas.

Workflow: plan (no SDR) -> optional explicit acquisition -> audit immutable trial
records. Every candidate is tested using an LS/pilot-CPE receiver, natural RF,
identical packet count and equalizer. Per-candidate repeated bursts are all
captured into RAM before any estimation. Never silently retries failed trials.
No transmitter opens without --acquire. Do not use calibration/holdout RF to
support unbiased claims about channel estimators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ofdm_benchmark.config import Method, Noise, load
from ofdm_benchmark.io import read_json

# Discrete HackRF settings; frequency and sample rate must remain frozen.
TX_RF = (0, 14)
TX_IF = (40, 44, 47)
SCALE = (0.012, 0.015, 0.018)
RX_RF = (0, 14)
RX_IF = (16, 24, 32, 40)
RX_BB = (12, 20, 26, 32)
BW = (6_000_000, 7_000_000)
AXES = (TX_RF, TX_IF, SCALE, RX_RF, RX_IF, RX_BB, BW)
FIELD_NAMES = ("tx_rf", "tx_if", "scale", "rx_rf", "rx_if", "rx_bb", "bw")

@dataclass(frozen=True)
class Gene:
    tx_rf: int
    tx_if: int
    scale: float
    rx_rf: int
    rx_if: int
    rx_bb: int
    bw: int

    @property
    def key(self):
        return (
            f"tx{self.tx_rf}_{self.tx_if}_s{self.scale:.3f}_"
            f"rx{self.rx_rf}_{self.rx_if}_{self.rx_bb}_bw{self.bw // 1000000}"
        )


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, sort_keys=True) + "\n")
    tmp.replace(path)


def nearest(values, x):
    return min(values, key=lambda v: abs(v - x))


def candidate_from_config(cfg):
    return Gene(
        *(
            nearest(TX_RF, cfg.radio.tx_rf_gain),
            nearest(TX_IF, cfg.radio.tx_if_gain),
            nearest(SCALE, cfg.phy.digital_scale),
            nearest(RX_RF, cfg.radio.rx_rf_gain),
            nearest(RX_IF, cfg.radio.rx_if_gain),
            nearest(RX_BB, cfg.radio.rx_bb_gain),
            nearest(BW, cfg.radio.bandwidth),
        )
    )


def calibrated_config(cfg, gene, *, stage, reps, packets, seed):
    # Keep sync, CFO, CP, payload, radio serials, sample rate and gate unchanged.
    return replace(
        cfg,
        phy=replace(cfg.phy, digital_scale=float(gene.scale)),
        radio=replace(
            cfg.radio,
            tx_rf_gain=float(gene.tx_rf),
            tx_if_gain=float(gene.tx_if),
            rx_rf_gain=float(gene.rx_rf),
            rx_if_gain=float(gene.rx_if),
            rx_bb_gain=float(gene.rx_bb),
            bandwidth=float(gene.bw),
        ),
        session_id=f"{cfg.session_id}_rfv2_{stage}_{gene.key}",
        seed=seed,
        bursts=reps,
        batch_size=reps,
        workers=1,
        measurement_packets=packets,
        methods=(Method("ls"),),
        conditions=(Noise(),),
        equalizers=("pilot_cpe",),
        timeout_seconds=max(240.0, cfg.timeout_seconds),
    )


def initial_population(cfg, n, rng):
    base = candidate_from_config(cfg)
    # Explicit, interpretable changes around the setting already tested in NLOS.
    edits = [
        {},
        {"rx_bb": 26},
        {"rx_if": 40},
        {"tx_if": 44},
        {"bw": 7_000_000},
        {"scale": 0.018},
        {"scale": 0.012},
        {"rx_rf": 0},
        {"tx_rf": 0},
        {"rx_bb": 32},
        {"rx_if": 24},
    ]
    candidates, seen = [], set()
    for kw in edits:
        g = replace(base, **kw)
        if g.key not in seen:
            candidates.append(g)
            seen.add(g.key)
        if len(candidates) >= n:
            return candidates
    while len(candidates) < n:
        values = [rng.choice(axis) for axis in AXES]
        g = Gene(*values)
        if g.key not in seen:
            candidates.append(g)
            seen.add(g.key)
    return candidates


def breed(rng, a, b):
    values = []
    for axis, x, y in zip(AXES, asdict(a).values(), asdict(b).values()):
        v = rng.choice((x, y))
        if rng.random() < 0.23:
            i = axis.index(v)
            v = axis[max(0, min(len(axis) - 1, i + rng.choice((-1, 1))))]
        values.append(v)
    return Gene(*values)


def power_proxy_db(row):
    p, n = row.get("signal_power_proxy"), row.get("noise_mean_power")
    if not isinstance(p, (int, float)) or not isinstance(n, (int, float)) or p <= 0 or n <= 0:
        return None
    return 10 * math.log10(p / n)


def evaluate_batch(rows, *, min_packets, max_rail_fraction, max_drift_db, reference_db=None):
    """Use worst-case packet delivery, not a lucky average, as primary rank."""
    if not rows or any(not x.get("ok", False) for x in rows):
        return dict(
            valid=False,
            reason="capture_or_replay_failed",
            rank=(-1.0, -1.0, -1.0, -1.0),
            power_drift_db=None,
            between_stage_power_drift_db=None,
        )
    per = [float(r["per"]) for r in rows]
    delivered = [int(r["delivered"]) for r in rows]
    if any(int(r["transmitted"]) != min_packets for r in rows):
        return dict(
            valid=False, reason="inconsistent_packet_count", rank=(-1.0, -1.0, -1.0, -1.0)
        )
    if any(float(r["active_rail_fraction"]) > max_rail_fraction for r in rows):
        return dict(valid=False, reason="rx_rail_exceeded", rank=(-1.0, -1.0, -1.0, -1.0))
    dbs = [power_proxy_db(r) for r in rows]
    if any(x is None for x in dbs):
        return dict(valid=False, reason="nonfinite_power_proxy", rank=(-1.0, -1.0, -1.0, -1.0))
    drift = max(dbs) - min(dbs)
    cross = abs(statistics.median(dbs) - reference_db) if reference_db is not None else None
    stable = drift <= max_drift_db and (cross is None or cross <= max_drift_db)
    rank = (
        min(delivered) / min_packets,
        sum(delivered) / (len(delivered) * min_packets),
        -max(per),
        -drift,
    )
    return dict(
        valid=stable,
        reason="ok" if stable else "power_drift_exceeded",
        rank=rank,
        power_drift_db=drift,
        median_power_db=statistics.median(dbs),
        between_stage_power_drift_db=cross,
        delivered_by_burst=delivered,
        per_by_burst=per,
    )


def ranking(summary):
    if not summary.get("valid"):
        return (-1.0, -1.0, -1.0, -1.0)
    return tuple(summary["rank"])


def validate_args(args, cfg):
    if (
        args.population < 4
        or args.generations < 1
        or args.screen < 2
        or args.confirm < 2
        or args.holdout < 3
    ):
        raise ValueError("Use population>=4, generations>=1, screen>=2, confirm>=2, holdout>=3")
    if not 1 <= args.packets <= 1000:
        raise ValueError("1..1000 measurement packets required")
    if not (
        0 <= args.max_per <= 1
        and 0 <= args.max_rail_fraction <= 0.01
        and 0 < args.max_drift_db <= 10
    ):
        raise ValueError("Invalid quality thresholds")
    if cfg.phy.cp_length != 16 or cfg.phy.sample_rate != 8e6:
        raise ValueError("This calibration is validated only for FFT64/CP16, 8 MS/s")
    if args.acquire and (cfg.radio.tx_serial == cfg.radio.rx_serial or any(
        s.startswith("SET_")
        for s in (
            cfg.radio.tx_serial,
            cfg.radio.rx_serial,
            cfg.session_id,
            cfg.channel_description,
        )
    )):
        raise ValueError("Fill real distinct HackRF serials/session description")
    if cfg.phy.sync_threshold <= 0 or cfg.phy.gate_power < 0:
        raise ValueError("Invalid frozen synchronization parameters")

def plan(args, cfg):
    rng = random.Random(args.seed)
    population = initial_population(cfg, args.population, rng)
    max_distinct = args.population + (args.generations - 1) * (args.population - 2)
    return dict(
        schema="nlos_rf_calibration_v2",
        mode="joint_TX_RX_GA",
        baseline_config=str(Path(args.config).resolve()),
        baseline_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        frozen={
            "phy_fft": cfg.phy.fft_length,
            "cp": cfg.phy.cp_length,
            "sample_rate": cfg.phy.sample_rate,
            "center_frequency": cfg.radio.center_frequency,
            "sync_threshold": cfg.phy.sync_threshold,
            "gate_power": cfg.phy.gate_power,
            "alpha": cfg.phy.equalizer_alpha,
            "rx_warmup": cfg.rx_warmup_seconds,
            "tx_warmup": cfg.tx_warmup_seconds,
            "tx_serial": cfg.radio.tx_serial,
            "rx_serial": cfg.radio.rx_serial,
        },
        ranges={key: list(values) for key, values in zip(FIELD_NAMES, AXES)},
        population=args.population,
        generations=args.generations,
        initial_population=[asdict(x) for x in population],
        screen_bursts=args.screen,
        confirm_bursts=args.confirm,
        holdout_bursts=args.holdout,
        packets_per_burst=args.packets,
        max_search_candidates=max_distinct,
        max_physical_bursts=max_distinct * args.screen + 2 * args.confirm + args.holdout,
        max_per=args.max_per,
        max_rail_fraction=args.max_rail_fraction,
        max_power_drift_db=args.max_drift_db,
        selection="Technically valid; worst burst delivery, mean delivery, max PER, power stability",
        gate_scope="Freeze gate and sync while GA tunes TX/RX; do not tune to QSVR",
        runtime="RAM capture of full candidate batch before estimator; LS+pilot_cpe only; natural noise",
        requires_explicit_acquire=True,
    )


def check_waveform(cfg, target, *, max_peak=0.90):
    """Generate real GNU TX symbols for two seeds, no SDR open; fail closed."""
    from ofdm_benchmark.transmit import generate, waveform_limits

    measured = []
    with tempfile.TemporaryDirectory(prefix="nlos_calibration_headroom_") as folder:
        for burst in (1, 2):
            out = Path(folder) / str(burst)
            generate(
                out, cfg.phy, cfg.measurement_packets, cfg.tx_warmup_seconds, cfg.seed + burst
            )
            record = waveform_limits(out / "waveform.c64")
            measured.append(dict(burst=burst, **record))
    result = {
        "passed": all(
            0 < x["peak_component"] <= max_peak and x["component_fraction_above_one"] == 0
            for x in measured
        ),
        "scope": "Pre-DAC numerical peak only; not analog PA linearity",
        "max_peak": max_peak,
        "records": measured,
        "hardware_opened": False,
    }
    save(target, result)
    return result


def analyze_result(output, count):
    source = Path(output)
    completed = read_json(source / "complete.json")
    if completed.get("status") != "complete" or int(completed.get("bursts", -1)) != count:
        raise RuntimeError("Incomplete acquisition/replay")
    rows = read_json(source / "rows.json")
    if len(rows) != count:
        raise RuntimeError(f"Expected {count} LS natural pilot-CPE rows, got {len(rows)}")
    records = []
    for item in sorted(rows, key=lambda r: r["burst"]):
        if (item.get("method"), item.get("condition"), item.get("equalizer")) != (
            "ls",
            "natural",
            "pilot_cpe",
        ):
            raise RuntimeError("Unexpected calibration matrix cell")
        p = source / "captures" / f"burst_{item['burst']:03d}" / "capture.json"
        meta = read_json(p)
        if not meta.get("physical_valid") or meta.get("raw_sha256") != item["raw_sha256"]:
            raise RuntimeError("Capture health/hash provenance failed")
        power = meta["power"]
        records.append(
            dict(
                ok=True,
                burst=item["burst"],
                delivered=item["delivered"],
                transmitted=item["transmitted"],
                per=item["per"],
                matched=item["matched"],
                raw_triggers=item["raw_triggers"],
                accepted_headers=item["accepted_headers"],
                ber_conditional=item["ber_conditional"],
                raw_sha256=item["raw_sha256"],
                signal_power_proxy=power["signal_power_proxy"],
                noise_mean_power=power["noise_mean_power"],
                active_rail_fraction=power["active_rail_fraction"],
            )
        )
    return records


def execute(args):
    from nlos_v1.prepare import emit_toml
    from ofdm_benchmark.benchmark import run_campaign

    cfg = load(args.config)
    validate_args(args, cfg)
    info = plan(args, cfg)
    if not args.acquire:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        print(
            "DRY RUN — append --acquire to open HackRFs; capture requires new output directory"
        )
        return 0
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", info)
    rng = random.Random(args.seed)
    population = initial_population(cfg, args.population, rng)
    cache = {}
    trials = []
    staged_results = []

    def trial(gene, stage, repeats):
        label = f"{stage}_{gene.key}"
        folder = output / "runs" / label
        specific = calibrated_config(
            cfg,
            gene,
            stage=stage,
            reps=repeats,
            packets=args.packets,
            seed=cfg.seed + {"screen": 1000, "confirm": 2000, "holdout": 3000}[stage],
        )
        config_file = output / "configs" / (label + ".toml")
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(emit_toml(specific))
        rec = {
            "stage": stage,
            "gene": asdict(gene),
            "name": label,
            "config": str(config_file),
            "repeats": repeats,
            "packets": args.packets,
            "run_folder": str(folder),
            "status": "pending",
        }
        trials.append(rec)
        save(output / "trials.json", trials)
        try:
            headroom = check_waveform(specific, output / "headroom" / (label + ".json"))
            if not headroom["passed"]:
                raise RuntimeError("TX waveform headroom >0.90; RF acquisition not attempted")
            rows = run_campaign(specific, folder, acquire=args.acquire)
            if len(rows) != repeats:
                raise RuntimeError("Unexpected campaign row count")
            records = analyze_result(folder, repeats)
            assessment = evaluate_batch(
                records,
                min_packets=args.packets,
                max_rail_fraction=args.max_rail_fraction,
                max_drift_db=args.max_drift_db,
            )
            rec.update(
                status="ok" if assessment["valid"] else "rejected",
                records=records,
                assessment=assessment,
            )
        except Exception as error:
            rec.update(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                assessment=evaluate_batch(
                    [],
                    min_packets=args.packets,
                    max_rail_fraction=args.max_rail_fraction,
                    max_drift_db=args.max_drift_db,
                ),
            )
        finally:
            save(output / "trials.json", trials)
        print(
            f"{stage}: {gene.key}: {rec['status']} "
            f"{rec.get('assessment', {}).get('delivered_by_burst', [])}",
            flush=True,
        )
        if rec["status"] != "failed" and stage == "screen" and not args.keep_search_iq:
            # IQ unneeded for selecting candidate. Keep capture.json, all JSONs and logs.
            for p in folder.rglob("*.c64"):
                p.unlink()
        return rec

    for generation in range(args.generations):
        for gene in population:
            if gene.key not in cache:
                cache[gene.key] = trial(gene, "screen", args.screen)
        scores = sorted(
            cache.items(), key=lambda pair: ranking(pair[1]["assessment"]), reverse=True
        )
        staged_results.append(
            dict(
                generation=generation + 1,
                ranking=[{"key": k, "rank": ranking(v["assessment"])} for k, v in scores],
            )
        )
        save(output / "generations.json", staged_results)
        if generation + 1 < args.generations:
            elite = [Gene(**v["gene"]) for _, v in scores[:2]]
            next_population = elite[:]
            seen = {p.key for p in next_population}
            tries = 0
            pool = [Gene(**v["gene"]) for _, v in scores[: min(5, len(scores))]]
            while len(next_population) < args.population and tries < 4000:
                tries += 1
                g = breed(rng, rng.choice(pool), rng.choice(pool))
                if g.key not in seen and g.key not in cache:
                    next_population.append(g)
                    seen.add(g.key)
            if len(next_population) < args.population:
                raise RuntimeError("Cannot generate enough new, diverse candidates")
            population = next_population

    shortlist = [
        Gene(**v["gene"])
        for _, v in sorted(
            cache.items(), key=lambda pair: ranking(pair[1]["assessment"]), reverse=True
        )
        if v["assessment"].get("valid")
    ][:2]
    if not shortlist:
        save(
            output / "summary.json",
            {
                "status": "no_stable_candidate",
                "reason": "Search failed technical or power stability gates",
            },
        )
        print("NO_STABLE_CANDIDATE — inspect trials.json; no holdout or preset")
        return 2
    confirmations = [trial(g, "confirm", args.confirm) for g in shortlist]
    confirmations.sort(key=lambda x: ranking(x["assessment"]), reverse=True)
    if not confirmations[0]["assessment"]["valid"]:
        save(
            output / "summary.json",
            {"status": "confirmation_failed", "shortlist": [x["gene"] for x in confirmations]},
        )
        print("CONFIRMATION_FAILED — no holdout; investigate the changing RF level")
        return 2
    winner = Gene(**confirmations[0]["gene"])
    holdout_rec = trial(winner, "holdout", args.holdout)
    h = holdout_rec["assessment"]
    c = confirmations[0]["assessment"]
    cross_drift = (
        abs(h["median_power_db"] - c["median_power_db"])
        if h.get("median_power_db") is not None
        else None
    )
    valid = (
        h["valid"]
        and cross_drift is not None
        and cross_drift <= args.max_drift_db
        and all(p <= args.max_per for p in h["per_by_burst"])
    )
    accepted = calibrated_config(
        cfg, winner, stage="calibrated", reps=3, packets=args.packets, seed=cfg.seed
    )
    candidate_file = output / (
        "VALIDATED_candidate.toml" if valid else "UNVALIDATED_candidate.toml"
    )
    candidate_file.write_text(emit_toml(accepted))
    report = dict(
        status="holdout_passed" if valid else "holdout_failed",
        winner=asdict(winner),
        winner_config=str(candidate_file),
        search_candidates=len(cache),
        confirm=[x["assessment"] for x in confirmations],
        holdout=h,
        holdout_vs_confirmation_power_drift_db=cross_drift,
        max_per=args.max_per,
        max_power_drift_db=args.max_drift_db,
        warning="Not independent estimator validation; calibrating with LS/pilot_cpe",
    )
    save(output / "summary.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if valid else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Frozen NLOS config, preferably configs/nlos_repeat_check/30a.toml",
    )
    p.add_argument("--output", type=Path, required=True, help="New directory; never overwrite")
    p.add_argument("--population", type=int, default=6)
    p.add_argument("--generations", type=int, default=2)
    p.add_argument("--screen", type=int, default=2, help="Consecutive RF bursts per candidate")
    p.add_argument("--confirm", type=int, default=3)
    p.add_argument("--holdout", type=int, default=4)
    p.add_argument("--packets", type=int, default=50, help="Same number in all stages")
    p.add_argument("--max-per", type=float, default=0.3, help="Limit in EVERY holdout burst")
    p.add_argument(
        "--max-drift-db",
        type=float,
        default=3.0,
        help="Max within-phase and between confirmation/holdout power proxy drift",
    )
    p.add_argument("--max-rail-fraction", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--keep-search-iq", action="store_true")
    p.add_argument(
        "--acquire", action="store_true", help="Required to open HackRF TX/RX devices"
    )
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    return execute(args)

if __name__ == "__main__":
    raise SystemExit(main())
