#!/usr/bin/env python3
"""
Scan results/*.json suite artifacts and emit replayability / provenance stats for the paper.
Does not require torch. Run from repo root:

  python3 audit_replayability.py
  python3 audit_replayability.py --json   # machine-readable
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
_ORCH_TS_DIR = re.compile(r"^\d{8}T\d{6}Z$")


def discover_result_dirs(extra: str | None) -> list[Path]:
    roots = [_ROOT / "results", _ROOT / "results" / "cluster_sync"]
    if extra:
        for part in extra.split(os.pathsep):
            p = Path(part).expanduser()
            if p.is_dir():
                roots.append(p)
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        k = str(r.resolve())
        if k not in seen and r.is_dir():
            seen.add(k)
            out.append(r)
    return out


def iter_suite_json_files(dirs: list[Path]) -> list[Path]:
    """Orchestrator suite JSON: results/<model>/<UTC>/<suite>.json (exclude manifest)."""
    out: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.json")):
            if p.name == "manifest.json":
                continue
            parts = p.relative_to(d).parts
            if len(parts) < 3:
                continue
            if _ORCH_TS_DIR.match(parts[1]):
                out.append(p)
    # de-dupe by resolved path
    by_k: dict[str, Path] = {}
    for p in out:
        try:
            by_k[str(p.resolve())] = p
        except OSError:
            by_k[str(p)] = p
    return list(by_k.values())


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-results-dirs", default=os.environ.get("RESULTS_EXTRA_DIRS", ""))
    ap.add_argument("--json", action="store_true", help="print JSON summary only")
    args = ap.parse_args()

    dirs = discover_result_dirs(args.extra_results_dirs or None)
    files = iter_suite_json_files(dirs)
    n = len(files)
    keys = {
        "model_id": 0,
        "benchmark_suite": 0,
        "task_prompt": 0,
        "last_model_output": 0,
        "judge_malice": 0,
        "judge_raw": 0,
        "benchmark_run_root": 0,
        "run_uuid_json": 0,
        "prompt_hash_json": 0,
        "output_hash_json": 0,
    }
    manifest_ok = 0
    for p in files:
        d = _load(p)
        for k in keys:
            if k == "run_uuid_json" and d.get("run_uuid"):
                keys[k] += 1
            elif k == "prompt_hash_json" and d.get("prompt_hash"):
                keys[k] += 1
            elif k == "output_hash_json" and d.get("output_hash"):
                keys[k] += 1
            elif k != "run_uuid_json" and k != "prompt_hash_json" and k != "output_hash_json":
                if d.get(k) is not None and d.get(k) != "":
                    keys[k] += 1
        man = p.parent / "manifest.json"
        if man.is_file():
            manifest_ok += 1

    # one manifest hash example (first found)
    example_manifest = ""
    example_sha = ""
    for p in files:
        man = p.parent / "manifest.json"
        if man.is_file():
            example_manifest = str(man.relative_to(_ROOT))
            example_sha = sha256_file(man)
            break

    pct = lambda c: round(100.0 * c / n, 1) if n else 0.0

    summary = {
        "n_suite_json_files": n,
        "n_roots_scanned": len(dirs),
        "pct_parent_manifest": pct(manifest_ok),
        "pct_model_id": pct(keys["model_id"]),
        "pct_benchmark_suite": pct(keys["benchmark_suite"]),
        "pct_task_prompt": pct(keys["task_prompt"]),
        "pct_last_model_output": pct(keys["last_model_output"]),
        "pct_judge_malice": pct(keys["judge_malice"]),
        "pct_judge_raw": pct(keys["judge_raw"]),
        "pct_benchmark_run_root": pct(keys["benchmark_run_root"]),
        "pct_infile_run_uuid": pct(keys["run_uuid_json"]),
        "pct_infile_prompt_hash": pct(keys["prompt_hash_json"]),
        "pct_infile_output_hash": pct(keys["output_hash_json"]),
        "example_manifest_relpath": example_manifest,
        "example_manifest_sha256": example_sha,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("Replayability audit (suite JSON under results/<model>/<UTC>/)\n")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
