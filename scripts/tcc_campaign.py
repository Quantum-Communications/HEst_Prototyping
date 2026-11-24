#!/usr/bin/env python3
"""Planeja as células publicadas, reprocessa amostras e executa novas aquisições."""
from __future__ import annotations
import argparse
import csv
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from ofdm_benchmark.config import load, Method, Noise
from ofdm_benchmark.io import read_json, sha256, write_json
from ofdm_benchmark.research import capture_folders, check_config
from ofdm_benchmark.benchmark import replay_capture, isolated, base_job

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT/'data/campanha_final/master_cells_complete.csv'
GRAM = ROOT/'data/config/cx_4000.npz'
METHODS = ('ls','lmmse','svr_skdirect_rbf','qsvr_skdirect_cx_shots')
FIELDS = ('campaign','session_id','burst','noise_family','added_snr_db','probability','variance_ratio','mean_bad_samples','noise_replica')

def read_plan():
    with CELLS.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))

def units(rows):
    """Agrupa células que compartilham uma realização do ruído e parâmetros de ajuste."""
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in FIELDS)
        groups[key + (row['svr_c'] or '9.615', row['svr_epsilon'] or '0.02')].append(row)
    return groups


def methods(c=9.615, e=0.02):
    params = {'c': float(c), 'epsilon': float(e)}
    return (Method('ls'), Method('lmmse'), Method(METHODS[2], params),
            Method(METHODS[3], {**params, 'quantum_gram_path': str(GRAM)}))


def config_for(base, key, rows, workers):
    _, _, _, family, snr, probability, ratio, duration, replica, c, epsilon = key
    selected = {r['equalizer'] for r in rows}
    equalizers = tuple(eq for eq in ('static', 'pilot_cpe', 'simpledfe') if eq in selected)
    noise = Noise(family, float(snr) if snr else None, float(probability),
                  float(ratio), float(duration))
    recorded_workers = {int(r['parallel_workers']) for r in rows}
    if len(recorded_workers) != 1:
        raise ValueError('Concorrência histórica ambígua')
    # As campanhas de robustez espaçam réplicas antes do incremento por rajada e família.
    return replace(base, methods=methods(c, epsilon), conditions=(noise,), equalizers=equalizers,
                   workers=workers or recorded_workers.pop(), seed=base.seed + int(replica)*1000003,
                   bursts=1, batch_size=1, timeout_seconds=max(900, base.timeout_seconds))


def capture(args):
    if not args.acquire:
        raise SystemExit('A abertura dos dois rádios requer --acquire.')
    from ofdm_benchmark.preflight import check

    cfg = load(args.config)
    if args.command == 'link-test':
        cfg = replace(cfg, bursts=1, measurement_packets=100, batch_size=1,
                      methods=methods(), conditions=(Noise(),), equalizers=('pilot_cpe',))
    acquisition_cfg = replace(cfg, methods=(Method('ls'),), conditions=(Noise(),),
                              equalizers=('pilot_cpe',))
    check(cfg, hardware=True)
    ram = Path(cfg.ram_root).resolve()
    mounts = [line.split() for line in Path('/proc/mounts').read_text().splitlines()]
    if not any(Path(m[1]).resolve() == ram and m[2] == 'tmpfs' for m in mounts):
        raise ValueError('ram_root deve ser um ponto de montagem tmpfs')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'evaluation.json', cfg.to_dict())
    acq = out/'acquisition'
    acq.mkdir()
    write_json(acq/'config.json', acquisition_cfg.to_dict())
    for burst in range(1, cfg.bursts + 1):
        with tempfile.TemporaryDirectory(prefix='ofdm_capture_', dir=ram) as tmp:
            folder = Path(tmp)/f'burst_{burst:03d}'
            job = dict(base_job(acquisition_cfg), action='capture',
                       experiment=acquisition_cfg.to_dict(), burst=burst, output=str(folder))
            result = isolated(job, acq/f'capture_{burst:03d}.job.json', acquire=args.acquire)
            target = acq/'captures'/folder.name
            if folder.exists():
                target.parent.mkdir(exist_ok=True)
                shutil.copytree(folder, target)
            if not result['ok']:
                raise RuntimeError(f'Falha na aquisição {burst}. Consulte os logs da rajada.')
            meta = read_json(target/'capture.json')
            if (sha256(target/'capture.c64') != meta['raw_sha256'] or
                    sha256(target/'tx/truth.json') != meta['truth_sha256']):
                raise RuntimeError('Falha de integridade na cópia da aquisição')
    write_json(acq/'complete.json', {'status': 'complete', 'physical_bursts': cfg.bursts})
    if args.command in ('link-test', 'run-session'):
        rows = []
        for folder in capture_folders(acq):
            rows.extend(replay_capture(folder, out/'reception'/folder.name, cfg))
        write_json(out/'measurements.json', rows)
        for row in rows:
            print(row['method'], row['equalizer'], 'TX', row['transmitted'],
                  'associados', row['matched'], 'PER', row['per'], 'BER', row['ber_conditional'])


def comparison_row(old, measured, new_capture):
    """Só os campos de desenho vêm do plano. Toda medida vem desta execução."""
    design = {*FIELDS, 'case_id', 'svr_c', 'svr_epsilon'}
    row = {name: old[name] if name in design else
           measured['raw_sha256' if name == 'physical_source_sha256' else name]
           for name in old}
    exact = ('physical_source_sha256', 'replay_sha256', 'corrected_sha256',
             'trigger_sha256', 'transmitted', 'matched', 'delivered', 'observed_bit_errors',
             'matched_bits', 'expected_bits', 'evm_symbols', 'evm_matched_packets',
             'fit_calls', 'fit_failures')
    ratios = ('per', 'ber_conditional', 'evm_rms', 'evm_error_energy', 'evm_reference_energy')
    mismatches = []
    if not new_capture:
        mismatches.extend(k for k in exact if str(row[k]) != old[k])
        for key in ratios:
            observed, reference = row[key], old[key]
            if observed is None and reference == '':
                continue
            if observed is None or reference == '' or not math.isclose(
                    float(observed), float(reference), rel_tol=1e-8, abs_tol=1e-12):
                mismatches.append(key)
    row['historical_comparison'] = ('not_applicable_new_capture' if new_capture
                                    else 'mismatch' if mismatches else 'match')
    row['mismatch_fields'] = ','.join(mismatches)
    return row


def replay(args):
    rows = [r for r in read_plan()
            if (args.campaign is None or r['campaign'] == args.campaign)
            and (args.session is None or r['session_id'] == args.session)
            and (args.burst is None or int(r['burst']) == args.burst)
            and (args.noise is None or r['noise_family'] == args.noise)
            and (args.equalizer is None or r['equalizer'] == args.equalizer)]
    if not rows:
        raise ValueError('A seleção não contém células')
    groups = units(rows)
    print(json.dumps({'cells': len(rows), 'units': len(groups),
                      'campaigns': Counter(r['campaign'] for r in rows)}, indent=2))
    if not args.run:
        return
    if args.root is None or args.output is None:
        raise ValueError('--run exige --root e --output')
    captures, configs = {}, {}
    with (CELLS.parent/'fontes_iq.csv').open(newline='', encoding='utf-8') as stream:
        sources = {(r['session'], int(r['burst'])): r for r in csv.DictReader(stream)}
    for row in rows:
        session, burst = row['session_id'], int(row['burst'])
        key = (session, burst)
        if key not in captures:
            session_root = args.root/session
            if session not in configs:
                configs[session] = load(session_root/'evaluation.json')
            folder = capture_folders(session_root/'acquisition', [burst])[0]
            meta = read_json(folder/'capture.json')
            if meta['session_id'] != session or not meta['physical_valid']:
                raise ValueError('Sessão incorreta ou aquisição inválida')
            if not args.new_captures and (
                    meta['raw_sha256'] != sources[key]['raw_sha256'] or
                    meta['truth_sha256'] != sources[key]['truth_sha256']):
                raise ValueError('Os digests do I/Q ou da verdade TX divergem do protocolo histórico')
            if (sha256(folder/'capture.c64') != meta['raw_sha256'] or
                    sha256(folder/'tx/truth.json') != meta['truth_sha256']):
                raise ValueError('Falha de integridade na fonte')
            captures[key] = (folder, meta['raw_sha256'])
        if not args.new_captures and captures[key][1] != row['physical_source_sha256']:
            raise ValueError('O SHA-256 das amostras diverge do histórico')

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    all_rows = []
    for i, (key, expected) in enumerate(groups.items(), 1):
        campaign, session, burst, *_ = key
        cfg = config_for(configs[session], key, expected, args.workers)
        folder = out/f'unit_{i:05d}'
        measured = replay_capture(captures[session, int(burst)][0], folder, cfg)
        by_method = {(r['method'], r['equalizer']): r for r in measured}
        batch = [comparison_row(old, by_method[old['method'], old['equalizer']], args.new_captures)
                 for old in expected]
        all_rows.extend(batch)
        write_json(folder/'comparison.json', batch)
        print(f'{i}/{len(groups)} {campaign}/{session}/{burst}', flush=True)
    with (out/'cells.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    mismatches = sum(r['historical_comparison'] == 'mismatch' for r in all_rows)
    write_json(out/'complete.json', {'cells': len(all_rows), 'historical_mismatches': mismatches,
                                      'new_captures': args.new_captures})
    if mismatches:
        raise SystemExit('Divergências históricas em cells.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check', help='Verifica os estimadores sem SDR')
    check.add_argument('--config', type=Path, default=ROOT/'configs/tcc.example.json')
    for command in ('capture', 'run-session', 'link-test'):
        radio = sub.add_parser(command)
        radio.add_argument('--config', type=Path, required=True)
        radio.add_argument('--output', type=Path, required=True)
        radio.add_argument('--acquire', action='store_true')
    rep = sub.add_parser('replay', help='Planeja as células. --run processa I/Q existentes')
    rep.add_argument('--campaign')
    rep.add_argument('--session')
    rep.add_argument('--burst', type=int)
    rep.add_argument('--root', type=Path)
    rep.add_argument('--output', type=Path)
    rep.add_argument('--workers', type=int, choices=range(1, 7))
    rep.add_argument('--run', action='store_true')
    rep.add_argument('--new-captures', action='store_true')
    rep.add_argument('--equalizer', choices=('static', 'pilot_cpe', 'simpledfe'))
    rep.add_argument('--noise', choices=('natural', 'awgn', 'bernoulli_gaussian', 'markov_gaussian'))
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(check_config(args.config), indent=2))
    elif args.command == 'replay':
        replay(args)
    else:
        capture(args)


if __name__ == '__main__':
    main()
