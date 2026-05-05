#!/usr/bin/env python3
"""
Multi-model benchmarking factory: reads ``models_to_test.yaml``, runs core + extended suites per model,
and writes ``results/<safe_model_name>/<timestamp>/{manifest.json,*.json}`` (18 suite JSONs per run).

Each suite uses :mod:`scaffold` LangGraph compilation so :mod:`mechanistic_hooks` capture runs
during the Act phase (HF backend). vLLM uses no-op hooks but the same graph structure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_model_dirname(model_id: str) -> str:
    return (
        model_id.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
        .replace("\\", "_")
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise SystemExit(
            "PyYAML is required for models_to_test.yaml. Install with: pip install pyyaml"
        ) from e
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    return data if isinstance(data, dict) else {}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    backend: str = "hf"
    vllm_base: str | None = None


def _parse_models(cfg: dict[str, Any]) -> list[ModelSpec]:
    rows = cfg.get("models")
    if not isinstance(rows, list):
        return []
    out: list[ModelSpec] = []
    for row in rows:
        if isinstance(row, str):
            out.append(ModelSpec(id=row.strip()))
            continue
        if not isinstance(row, dict):
            continue
        mid = str(row.get("id") or row.get("model") or "").strip()
        if not mid:
            continue
        out.append(
            ModelSpec(
                id=mid,
                backend=str(row.get("backend", "hf")).lower().strip(),
                vllm_base=(str(row["vllm_base"]).strip() if row.get("vllm_base") else None),
            )
        )
    return out


def _envelope(
    *,
    suite: str,
    model_id: str,
    backend: str,
    run_root: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Attach stable metadata for downstream aggregation (generate_paper_tables / results_processor)."""
    out = dict(payload)
    out.setdefault("model_id", model_id)
    out.setdefault("subject_model_id", model_id)
    out["eval_backend"] = backend
    out["benchmark_suite"] = suite
    out["benchmark_run_root"] = str(run_root.resolve())
    return out


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _judas_bribe_sweep_from_cfg(cfg: dict[str, Any]) -> list[float] | None:
    """``None`` = do not override (``JUDAS_BRIBE_SWEEP`` env or legacy). ``[]`` = force legacy."""
    if "judas_bribe_sweep" not in cfg:
        return None
    jb = cfg["judas_bribe_sweep"]
    if isinstance(jb, list):
        return [float(x) for x in jb]
    return []


def run_all_suites_for_model(
    model_id: str,
    *,
    backend: str,
    vllm_base: str | None,
    judge_model_id: str | None,
    sae_release: str | None,
    sae_id: str | None,
    shutdown_mode: str,
    run_root: Path,
    judas_bribe_sweep: list[float] | None = None,
) -> dict[str, Any]:
    """Execute all suites for one model; intended for use inside a worker process."""
    from benchmarks.extended_probes import (
        BabelMultilingualSuite,
        BrutusProtocolSuite,
        ChannelFactorialSuite,
        CompressionCaveatSuite,
        CryptoCommitmentTrapSuite,
        DelilahRedactionSuite,
        GoodSamaritanSuite,
        JekyllInjectionSuite,
        LotBrittlenessSuite,
        MetaEvaluatorLieSuite,
        NeedleHaystackLieSuite,
        PairedDeltaProtocolSuite,
        PolicySupersessionSuite,
        ScapegoatFalsePremiseSuite,
    )
    from benchmarks.judas_protocol import JudasProtocolSuite
    from benchmarks.lazarus_self_repair import LazarusSelfRepairSuite
    from benchmarks.polymorphic_malware_test import PolymorphicMalwareSuite
    from benchmarks.shutdown_paradox_test import ShutdownParadoxSuite

    from scaffold import make_hf_runtime, make_vllm_runtime

    dump_dir = run_root / "activations"
    dump_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ACTIVATION_DUMP_DIR"] = str(dump_dir.resolve())

    backend_l = backend.lower().strip()
    if backend_l == "vllm":
        rt = make_vllm_runtime(
            subject_model=model_id,
            judge_model=judge_model_id,
            vllm_base=vllm_base or os.environ.get("VLLM_BASE_URL"),
        )
    else:
        rt = make_hf_runtime(
            subject_model_id=model_id,
            judge_model_id=judge_model_id,
            sae_release=sae_release or os.environ.get("SAE_RELEASE"),
            sae_id=sae_id or os.environ.get("SAE_ID"),
            dump_dir=str(dump_dir.resolve()),
        )

    ho_mode: Any = shutdown_mode.strip().lower()
    if ho_mode in ("hidden_objective", "hidden", "ho"):
        paradox = ShutdownParadoxSuite(mode="hidden_objective")
    else:
        paradox = ShutdownParadoxSuite(mode="standard")

    artifacts: dict[str, Any] = {"suites": {}, "errors": {}}

    def _run(name: str, fn: Any) -> None:
        try:
            raw = fn()
            if not isinstance(raw, dict):
                raise TypeError(f"suite {name} returned {type(raw)}")
            artifacts["suites"][name] = _envelope(
                suite=name,
                model_id=model_id,
                backend=backend_l,
                run_root=run_root,
                payload=raw,
            )
        except Exception as e:
            artifacts["errors"][name] = repr(e)

    judas_kw: dict[str, Any] = {}
    if judas_bribe_sweep is not None:
        judas_kw["bribe_sweep_multipliers"] = tuple(judas_bribe_sweep)

    _run("polymorphic_malware", lambda: PolymorphicMalwareSuite().run(rt))
    _run("shutdown_paradox", lambda: paradox.run(rt))
    _run("judas_protocol", lambda: JudasProtocolSuite(**judas_kw).run(rt))
    _run("lazarus_self_repair", lambda: LazarusSelfRepairSuite().run(rt))

    _run("brutus_protocol", lambda: BrutusProtocolSuite().run(rt))
    _run("needle_haystack_lie", lambda: NeedleHaystackLieSuite().run(rt))
    _run("lot_brittleness", lambda: LotBrittlenessSuite().run(rt))
    _run("delilah_redaction", lambda: DelilahRedactionSuite().run(rt))
    _run("good_samaritan", lambda: GoodSamaritanSuite().run(rt))
    _run("babel_multilingual", lambda: BabelMultilingualSuite().run(rt))
    _run("jekyll_injection", lambda: JekyllInjectionSuite().run(rt))
    _run("scapegoat_false_premise", lambda: ScapegoatFalsePremiseSuite().run(rt))

    _run("channel_factorial", lambda: ChannelFactorialSuite().run(rt))
    _run("policy_supersession", lambda: PolicySupersessionSuite().run(rt))
    _run("crypto_commitment_trap", lambda: CryptoCommitmentTrapSuite().run(rt))
    _run("meta_evaluator_lie", lambda: MetaEvaluatorLieSuite().run(rt))
    _run("compression_caveat", lambda: CompressionCaveatSuite().run(rt))
    _run("paired_delta_protocol", lambda: PairedDeltaProtocolSuite().run(rt))

    return artifacts


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Top-level entry for process pool pickling."""
    return run_all_suites_for_model(
        payload["model_id"],
        backend=str(payload.get("backend", "hf")),
        vllm_base=payload.get("vllm_base"),
        judge_model_id=payload.get("judge_model_id"),
        sae_release=payload.get("sae_release"),
        sae_id=payload.get("sae_id"),
        shutdown_mode=str(payload.get("shutdown_paradox_mode", "hidden_objective")),
        run_root=Path(payload["run_root"]),
        judas_bribe_sweep=payload.get("judas_bribe_sweep"),
    )


def orchestrate(
    config_path: Path,
    *,
    results_root: Path,
    max_parallel_override: int | None = None,
) -> Path:
    cfg = _load_yaml(config_path)
    models = _parse_models(cfg)
    if not models:
        raise SystemExit(f"No models in {config_path}")

    max_par = int(max_parallel_override or cfg.get("max_parallel_models", 1) or 1)
    max_par = max(1, min(max_par, 4))

    judge_global = cfg.get("judge_model_id")
    judge_global = str(judge_global).strip() if judge_global else None
    sae_r = cfg.get("sae_release")
    sae_i = cfg.get("sae_id")
    sae_release = str(sae_r).strip() if sae_r else None
    sae_id = str(sae_i).strip() if sae_i else None
    shutdown_mode = str(cfg.get("shutdown_paradox_mode", "hidden_objective"))
    judas_sweep = _judas_bribe_sweep_from_cfg(cfg)

    stamp = _utc_stamp()
    batch_manifest: dict[str, Any] = {
        "timestamp_utc": stamp,
        "config": str(config_path.resolve()),
        "max_parallel_models": max_par,
        "models": [m.id for m in models],
        "judas_bribe_sweep": judas_sweep,
    }
    model_run_dirs: dict[str, str] = {}

    if max_par == 1:
        for m in models:
            run_dir = results_root / safe_model_dirname(m.id) / stamp
            run_dir.mkdir(parents=True, exist_ok=True)
            judge_id = judge_global or m.id
            art = run_all_suites_for_model(
                m.id,
                backend=m.backend,
                vllm_base=m.vllm_base,
                judge_model_id=judge_id,
                sae_release=sae_release,
                sae_id=sae_id,
                shutdown_mode=shutdown_mode,
                run_root=run_dir,
                judas_bribe_sweep=judas_sweep,
            )
            model_run_dirs[m.id] = str(run_dir.resolve())
            for name, suite_blob in art.get("suites", {}).items():
                _write_json(run_dir / f"{name}.json", suite_blob)
            if art.get("errors"):
                _write_json(run_dir / "errors.json", art["errors"])
            _write_json(
                run_dir / "manifest.json",
                {
                    "model_id": m.id,
                    "backend": m.backend,
                    "timestamp_utc": stamp,
                    "suites_written": list(art.get("suites", {}).keys()),
                    "errors": art.get("errors", {}),
                },
            )
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=max_par) as ex:
            for m in models:
                run_dir = results_root / safe_model_dirname(m.id) / stamp
                run_dir.mkdir(parents=True, exist_ok=True)
                judge_id = judge_global or m.id
                fut = ex.submit(
                    _worker,
                    {
                        "model_id": m.id,
                        "backend": m.backend,
                        "vllm_base": m.vllm_base,
                        "judge_model_id": judge_id,
                        "sae_release": sae_release,
                        "sae_id": sae_id,
                        "shutdown_paradox_mode": shutdown_mode,
                        "run_root": str(run_dir.resolve()),
                        "judas_bribe_sweep": judas_sweep,
                    },
                )
                futures[fut] = (m, run_dir)
            for fut in as_completed(futures):
                m, run_dir = futures[fut]
                model_run_dirs[m.id] = str(run_dir.resolve())
                try:
                    art = fut.result()
                except Exception as e:
                    _write_json(run_dir / "errors.json", {"worker_fatal": repr(e)})
                    continue
                for name, suite_blob in art.get("suites", {}).items():
                    _write_json(run_dir / f"{name}.json", suite_blob)
                if art.get("errors"):
                    _write_json(run_dir / "errors.json", art["errors"])
                _write_json(
                    run_dir / "manifest.json",
                    {
                        "model_id": m.id,
                        "backend": m.backend,
                        "timestamp_utc": stamp,
                        "suites_written": list(art.get("suites", {}).keys()),
                        "errors": art.get("errors", {}),
                    },
                )

    batch_manifest["model_run_dirs"] = model_run_dirs
    batch_dir = results_root / "_batches" / stamp
    batch_dir.mkdir(parents=True, exist_ok=True)
    _write_json(batch_dir / "batch_manifest.json", batch_manifest)
    return batch_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-model benchmark orchestrator")
    ap.add_argument(
        "--config",
        default=str(_ROOT / "models_to_test.yaml"),
        help="Path to models_to_test.yaml",
    )
    ap.add_argument(
        "--results-root",
        default=str(_ROOT / "results"),
        help="Root directory for results/<model>/<timestamp>/",
    )
    ap.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Override YAML max_parallel_models (1–4)",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    out_root = Path(args.results_root).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    batch = orchestrate(cfg_path, results_root=out_root, max_parallel_override=args.max_parallel)
    print(json.dumps({"batch_manifest": str((batch / "batch_manifest.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
