#!/usr/bin/env python3
"""
Run multi-model benchmarks and/or per-model contrastive activation mapping in one process.

Default order matches a typical workflow: full orchestrator first, then CAM for each HF model
in ``models_to_test.yaml``. Use ``--cam-first`` to run CAM before the orchestrator.

Examples::

    python run_pipeline.py
    python run_pipeline.py --cam-first
    python run_pipeline.py --skip-cam
    python run_pipeline.py --config models_to_test.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from multi_model_orchestrator import (  # noqa: E402
    _load_yaml,
    _parse_models,
)


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, env=os.environ.copy())
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run multi_model_orchestrator and/or run_contrastive_activation_mapping in sequence."
    )
    ap.add_argument(
        "--config",
        default=str(_ROOT / "models_to_test.yaml"),
        help="YAML read for model list and judge_model_id (same as orchestrator).",
    )
    ap.add_argument(
        "--cam-first",
        action="store_true",
        help="Run contrastive_activation_mapping for each HF model before the orchestrator.",
    )
    ap.add_argument("--skip-orchestrator", action="store_true", help="Only run CAM loop.")
    ap.add_argument("--skip-cam", action="store_true", help="Only run multi_model_orchestrator.")
    ap.add_argument(
        "--cam-dump-root",
        default="results/cam_dumps",
        help="Under repo: per-model dirs for .pt dumps (passed as --dump-dir/<safe>).",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    cfg = _load_yaml(cfg_path)
    models = _parse_models(cfg)
    if not models and not args.skip_cam:
        raise SystemExit(f"No models in {cfg_path}")

    judge = cfg.get("judge_model_id")
    judge_s = str(judge).strip() if judge else ""

    py = sys.executable
    orch = [py, str(_ROOT / "multi_model_orchestrator.py"), "--config", str(cfg_path)]

    def run_orch() -> None:
        if args.skip_orchestrator:
            return
        _run(orch, cwd=_ROOT)

    def run_cam_loop() -> None:
        if args.skip_cam:
            return
        dump_root = Path(args.cam_dump_root)
        if not dump_root.is_absolute():
            dump_root = (_ROOT / dump_root).resolve()
        ts_base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for i, m in enumerate(models):
            if m.backend != "hf":
                print(f"skip CAM (backend={m.backend}): {m.id}", flush=True)
                continue
            safe = (
                m.id.replace("/", "__")
                .replace(":", "_")
                .replace(" ", "_")
                .replace("\\", "_")
            )
            dump_dir = dump_root / safe
            dump_dir.mkdir(parents=True, exist_ok=True)
            out_json = _ROOT / "results" / f"contrastive_cam_{safe}_{ts_base}_{i}.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            cam_cmd = [
                py,
                str(_ROOT / "run_contrastive_activation_mapping.py"),
                "--model",
                m.id,
                "--dump-dir",
                str(dump_dir),
                "-o",
                str(out_json),
            ]
            if judge_s:
                cam_cmd.extend(["--judge-model", judge_s])
            _run(cam_cmd, cwd=_ROOT)

    if args.cam_first:
        run_cam_loop()
        run_orch()
    else:
        run_orch()
        run_cam_loop()


if __name__ == "__main__":
    main()
