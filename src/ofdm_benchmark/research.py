"""Capture discovery and replay of immutable physical IQ."""
from pathlib import Path
import numpy as np
from .config import load
from .estimators import Estimator
from .io import read_json, sha256, write_json
from .phy import sync_words
def check_config(path):
    """Validate every configured model on public synthetic preambles; no GNU or SDR."""
    cfg = load(path)
    first, second = sync_words()
    raw = np.stack([first, second, np.zeros(64)]).astype("c8")
    records = []
    for method in cfg.methods:
        model = Estimator(method, cfg.phy)
        channel, offset = model.estimate(raw)
        if not np.isfinite(channel).all() or offset != 0:
            raise RuntimeError(f"Invalid configured method: {method.name}")
        diag = model.diagnostics()
        records.append(
            dict(
                method=method.name,
                setup_seconds=diag["initialization_seconds"],
                fit_seconds=diag["fit_seconds"],
                failures=diag["fit_failures"],
            )
        )
    return dict(
        status="passed",
        records=records,
        radios_opened=False,
        scope="Finite-solution check, not an RF performance test",
    )

def capture_folders(root, bursts=None):
    """Find native retained captures; reject ambiguous identities and missing IQ."""
    root = Path(root).resolve()
    search = root / "captures" if (root / "captures").is_dir() else root
    candidates = (
        [search / "capture.json"]
        if (search / "capture.json").is_file()
        else sorted(search.glob("*/capture.json"))
    )
    found, seen = [], set()
    for metadata in candidates:
        info = read_json(metadata)
        key = (info["session_id"], info["burst"])
        if bursts and info["burst"] not in bursts:
            continue
        if key in seen:
            raise ValueError(f"Duplicated capture identity: {key}")
        seen.add(key)
        folder = metadata.parent
        for name in ("capture.c64", "tx/truth.json"):
            if not (folder / name).is_file():
                raise FileNotFoundError(
                    f"Replay needs {folder / name}; summaries cannot recreate IQ"
                )
        found.append(folder)
    if not found or (
        bursts and set(bursts) != {read_json(p / "capture.json")["burst"] for p in found}
    ):
        raise ValueError(
            "Requested retained captures were not found; point to the original captures directory"
        )
    return found

def replay_batch(captures, config, output, *, bursts=None, check_only=False):
    """Replay selected retained IQs with one frozen matrix; never open radios."""
    cfg = load(config)
    folders = capture_folders(captures, bursts)
    identities = []
    for folder in folders:
        meta = read_json(folder / "capture.json")
        if meta["phy"] != cfg.to_dict()["phy"]:
            raise ValueError(f"Replay PHY differs from capture {folder}")
        for filename, field in (
            ("capture.c64", "raw_sha256"),
            ("tx/truth.json", "truth_sha256"),
        ):
            if sha256(folder / filename) != meta[field]:
                raise ValueError(f"Capture integrity failure: {folder / filename}")
        if not meta["physical_valid"]:
            raise ValueError(f"Invalid source acquisition: {folder}")
        identities.append(
            dict(
                session_id=meta["session_id"],
                burst=meta["burst"],
                raw_sha256=meta["raw_sha256"],
            )
        )
    if check_only:
        return dict(
            status="passed", captures=identities, radios_opened=False, decoding_run=False
        )
    from .benchmark import replay_capture
    from .report import generate

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", cfg.to_dict())
    write_json(output / "sources.json", identities)
    rows = []
    for index, folder in enumerate(folders, 1):
        rows.extend(replay_capture(folder, output / "replay" / f"capture_{index:03d}", cfg))
    write_json(output / "rows.json", rows)
    write_json(
        output / "complete.json",
        dict(
            status="complete",
            execution_kind="offline_replay",
            cells=len(rows),
            radios_opened=False,
        ),
    )
    generate(output)
    return dict(status="complete", cells=len(rows), output=str(output), radios_opened=False)
