"""Rebuild the TCC CX Gram using BasicSimulator; default seed from archived matrix."""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
from ofdm_benchmark.estimators.quick_sklearn import physical_circuit, load_custom_gram
from ofdm_benchmark.phy import ACTIVE_K
from ofdm_benchmark.shot_fidelity import project_correlation_psd
def _snapshot(path, **fields):
    # Store partial count results atomically, allowing interrupted generation
    # to resume without changing the already measured circuits.
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(temporary, **fields)
    os.replace(temporary, path)

def measured_4000(path, variant, *, shots=4000, seed=2901, batch_size=24):
    """Count-only 52x52 Gram, resumable per batch of circuit pairs."""
    path = Path(path).resolve()
    if path.is_relative_to(Path(__file__).resolve().parents[1] / "data"):
        raise ValueError("Selecione uma saída fora dos dados históricos, por exemplo results/gram.npz")
    if variant != "cx" or path.suffix != ".npz":
        raise ValueError("O protocolo publicado usa CX e saída .npz")
    from qiskit.providers.basic_provider import BasicSimulator
    from qiskit import transpile
    path = Path(path).resolve()
    if path.exists():
        _, meta = load_custom_gram(path, variant, shots=shots)
        if meta['shots'] != shots:
            raise ValueError('Cached Gram used a different shot count')
        print(f'[GRAM CACHED] {variant}: {path}', flush=True)
        return path
    partial = path.with_suffix('.partial.npz')
    circuits = [physical_circuit(k, variant) for k in ACTIVE_K]
    pairs = [(i, j) for i in range(52) for j in range(i + 1, 52)]
    raw = np.eye(52, dtype=float)
    done = 0
    if partial.exists():
        with np.load(partial, allow_pickle=False) as z:
            raw = np.asarray(z['raw'], float)
            done = int(z['done'])
            if (z['variant'].item() != variant or int(z['seed']) != seed
                or int(z['shots']) != shots or int(z['batch_size']) != batch_size):
                raise ValueError('Partial Gram does not match requested circuit protocol')
        if raw.shape != (52,52) or not 0 <= done <= len(pairs):
            raise ValueError('Corrupt partial Gram')
    backend = BasicSimulator()
    for start in range(done, len(pairs), batch_size):
        subset = pairs[start:start + batch_size]
        probes=[]
        for i,j in subset:
            qc = circuits[i].compose(circuits[j].inverse())
            qc.measure_all()
            probes.append(qc)
        probes = transpile(probes, backend=backend, optimization_level=0)
        result = backend.run(probes, shots=shots,
                             seed_simulator=seed + start // batch_size).result()
        for index,(i,j) in enumerate(subset):
            counts = result.get_counts(index)
            p0 = int(counts.get('0'*circuits[i].num_qubits, 0)) / shots
            raw[i,j] = raw[j,i] = p0
        done = start + len(subset)
        _snapshot(partial, raw=raw, done=np.array(done),
                  variant=np.array(variant), shots=np.array(shots),
                  seed=np.array(seed),batch_size=np.array(batch_size))
        if done % (batch_size * 5) == 0 or done == len(pairs):
            print(f'[GRAM] {variant}: {done}/{len(pairs)} pairs', flush=True)
    used, correction = project_correlation_psd(raw, policy='clip',
                                                max_relative_correction=0.20)
    meta = dict(
        schema=1,
        family='ofdm_carrier_physical_v1',
        variant=variant,
        step=None,
        shots=shots, seed=seed,
        pairs_measured=len(pairs), circuits_executed=len(pairs),
        backend='Qiskit BasicSimulator shot counts', hardware_execution=False,
        read_statevector_for_training=False,
        probability='P(0^q | U(j)^dagger U(i))',
        qubits=5, phase_information_preserved=False,
        gram_sha256=hashlib.sha256(
            np.ascontiguousarray(used,dtype='<f8').tobytes()).hexdigest(),
        **correction)
    _snapshot(path, gram=used, carriers=np.asarray(ACTIVE_K,int),
              metadata=np.array(json.dumps(meta, sort_keys=True)))
    partial.unlink(missing_ok=True)
    print(f'[GRAM OK] {variant}: PSD repaired={correction["psd_repaired"]}, '
          f'rel correction={correction["frobenius_relative_correction"]:.5g}', flush=True)
    return path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    measured_4000(a.output, "cx", shots=4000, seed=2901, batch_size=24)
