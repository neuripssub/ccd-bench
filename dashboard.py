#!/usr/bin/env python3
"""
Streamlit control room for LLM benchmark research: results scan, infra checks,
multi-model orchestration, and paper LaTeX export.

Headless: ``streamlit run dashboard.py --server.port=8501 --server.headless=true``
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx
import streamlit as st
import yaml

from generate_paper_tables import (
    ScanSummary,
    aggregate_scan,
    build_signature_latex,
    build_sip_judas_personality_latex,
    discover_result_dirs,
    infer_model_id,
    iter_result_json_files,
    load_json_smart,
    paired_scatter_points,
)
from results_processor import (
    assign_personality_label,
    compute_sip_f1,
    contrastive_activation_mapping,
    extract_unified_metrics,
)
from ui_components import (
    all_merged_metrics_polar,
    contrastive_exclusive_table,
    contrastive_features_table,
    extended_probes_profile,
    four_axis_scores,
    inject_dashboard_css,
    iter_pt_activation_files,
    leaderboard_dataframe,
    plot_logit_shutdown_scatter,
    plot_personality_radar_four,
    plot_personality_radar_generic,
    six_axis_sip_profile,
)

# ---------------------------------------------------------------------------
# Result file discovery: orchestrator suite JSON allowlist (excludes huge CAM JSON)
# ---------------------------------------------------------------------------

_SUITE_JSON_NAMES: frozenset[str] = frozenset(
    {
        "polymorphic_malware.json",
        "shutdown_paradox.json",
        "judas_protocol.json",
        "lazarus_self_repair.json",
        "brutus_protocol.json",
        "needle_haystack_lie.json",
        "lot_brittleness.json",
        "delilah_redaction.json",
        "good_samaritan.json",
        "babel_multilingual.json",
        "jekyll_injection.json",
        "scapegoat_false_premise.json",
        "channel_factorial.json",
        "policy_supersession.json",
        "crypto_commitment_trap.json",
        "meta_evaluator_lie.json",
        "compression_caveat.json",
        "paired_delta_protocol.json",
    }
)


def results_root() -> Path:
    return _ROOT / "results"


def _safe_model_dirname(model_id: str) -> str:
    """Match ``multi_model_orchestrator.safe_model_dirname`` without importing the orchestrator."""
    return (
        model_id.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
        .replace("\\", "_")
    )


def iter_suite_json_files(results_dir: Path) -> list[Path]:
    """JSON artifacts from benchmark suites under ``results_dir`` (orchestrator + extended probes)."""
    out: list[Path] = []
    if not results_dir.is_dir():
        return out
    for p in results_dir.rglob("*.json"):
        if p.name in _SUITE_JSON_NAMES:
            out.append(p)
    # De-dupe by resolved path
    by_key: dict[str, Path] = {}
    for p in out:
        try:
            by_key[str(p.resolve())] = p
        except OSError:
            by_key[str(p)] = p
    return sorted(by_key.values())


def results_scan_mtime_signature(files: list[Path]) -> float:
    t = 0.0
    for p in files:
        try:
            st_ = p.stat()
            t += st_.st_mtime + st_.st_size
        except OSError:
            continue
    return t


def cam_dumps_mtime_signature(cam_root: Path) -> float:
    """Invalidate cached scan when paired activation dumps under ``results/cam_dumps`` change."""
    if not cam_root.is_dir():
        return 0.0
    return results_scan_mtime_signature(sorted(cam_root.rglob("*.pt")))


def load_scan_summary() -> tuple[ScanSummary, list[Path]]:
    extra = os.environ.get("RESULTS_EXTRA_DIRS", "").strip()
    dirs = discover_result_dirs(extra or None)
    by_key: dict[str, Path] = {}
    for d in dirs:
        for p in iter_suite_json_files(d):
            try:
                by_key[str(p.resolve())] = p
            except OSError:
                by_key[str(p)] = p
    suite_paths = sorted(by_key.values())
    if not suite_paths:
        for p in iter_result_json_files([results_root()]):
            if p.name in _SUITE_JSON_NAMES:
                try:
                    by_key[str(p.resolve())] = p
                except OSError:
                    by_key[str(p)] = p
        suite_paths = sorted(by_key.values())
    summary = aggregate_scan(suite_paths)
    return summary, suite_paths


def _check_sandbox_url(url: str) -> tuple[bool, str]:
    base = url.rstrip("/")
    health = f"{base}/health"
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(health)
            if r.status_code == 200:
                try:
                    j = r.json()
                    if isinstance(j, dict) and j.get("status") == "ok":
                        return True, f"OK ({health})"
                except Exception:
                    pass
                return True, f"HTTP {r.status_code} ({health})"
            return False, f"HTTP {r.status_code} ({health})"
    except Exception as e:
        return False, str(e)


def _check_remote_cluster_ssh() -> tuple[bool, str]:
    target = os.environ.get("REMOTE_CLUSTER_SSH", "").strip()
    if target:
        try:
            proc = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", target, "exit", "0"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            if proc.returncode == 0:
                return True, f"SSH `{target}` reachable"
            err = (proc.stderr or proc.stdout or "").strip()[:200]
            return False, f"SSH failed ({target}): {err or proc.returncode}"
        except subprocess.TimeoutExpired:
            return False, f"SSH timeout ({target})"
        except FileNotFoundError:
            return False, "ssh binary not found"
        except Exception as e:
            return False, str(e)
    try:
        proc = subprocess.run(
            ["pgrep", "-fl", "ssh"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if proc.stdout and proc.stdout.strip():
            return True, "SSH client process(es) running (set REMOTE_CLUSTER_SSH=user@host for a targeted check)"
        return False, "No ssh processes (set REMOTE_CLUSTER_SSH=user@host to verify a cluster login)"
    except FileNotFoundError:
        return False, "pgrep not available"
    except Exception as e:
        return False, str(e)


def load_models_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def list_models_from_config(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = cfg.get("models")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, str) and row.strip():
            out.append({"id": row.strip(), "backend": "hf", "vllm_base": None})
            continue
        if isinstance(row, dict):
            mid = str(row.get("id") or row.get("model") or "").strip()
            if not mid:
                continue
            out.append(
                {
                    "id": mid,
                    "backend": str(row.get("backend", "hf")).lower().strip(),
                    "vllm_base": row.get("vllm_base"),
                }
            )
    return out


def write_single_model_config(
    base_cfg: dict[str, Any],
    spec: dict[str, Any],
    dest: Path,
) -> None:
    out = dict(base_cfg)
    out["models"] = [
        {
            "id": spec["id"],
            "backend": spec.get("backend", "hf"),
            **({"vllm_base": spec["vllm_base"]} if spec.get("vllm_base") else {}),
        }
    ]
    dest.write_text(yaml.safe_dump(out, default_flow_style=False, sort_keys=False), encoding="utf-8")


def _stream_subprocess(cmd: list[str], cwd: Path, log_queue: "queue.Queue[str]") -> int:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log_queue.put(line)
    proc.wait()
    log_queue.put(f"\n--- process exited with code {proc.returncode} ---\n")
    return proc.returncode


# ---------------------------------------------------------------------------
# Leaderboard / per-run / JSON views
# ---------------------------------------------------------------------------


def _rel_under_root(p: Path) -> str:
    try:
        return str(p.relative_to(_ROOT))
    except ValueError:
        return str(p)


def _sync_scan_holder(holder: dict[str, Any]) -> None:
    sig = results_scan_mtime_signature(holder.get("files", []))
    cam_sig = cam_dumps_mtime_signature(results_root() / "cam_dumps")
    if sig != holder.get("_sig") or cam_sig != holder.get("_cam_sig"):
        holder["summary"], holder["files"] = load_scan_summary()
        holder["_sig"] = results_scan_mtime_signature(holder["files"])
        holder["_cam_sig"] = cam_sig


def _leaderboard_block(summary: ScanSummary, filter_q: str) -> None:
    df = leaderboard_dataframe(
        sip_by_model=summary.sip_f1_by_model,
        personality_by_model=summary.personality_by_model,
        merged_metrics_by_model=summary.merged_metrics_by_model,
        judas_thresholds_by_model=summary.judas_thresholds_by_model,
        brutus_thresholds_by_model=summary.brutus_thresholds_by_model,
        paired_delta_stress_leak_by_model=summary.paired_delta_stress_leak_by_model,
    )
    fq = filter_q.strip().lower()
    if fq:
        mask = (
            df["Model"].str.lower().str.contains(fq, na=False)
            | df["Personality"].str.lower().str.contains(fq, na=False)
        )
        df = df.loc[mask]
    st.dataframe(df, use_container_width=True, hide_index=True)


def _leaderboard_extended_block(summary: ScanSummary, filter_q: str) -> None:
    base = leaderboard_dataframe(
        sip_by_model=summary.sip_f1_by_model,
        personality_by_model=summary.personality_by_model,
        merged_metrics_by_model=summary.merged_metrics_by_model,
        judas_thresholds_by_model=summary.judas_thresholds_by_model,
        brutus_thresholds_by_model=summary.brutus_thresholds_by_model,
        paired_delta_stress_leak_by_model=summary.paired_delta_stress_leak_by_model,
    )
    fq = filter_q.strip().lower()
    if fq:
        mask = (
            base["Model"].str.lower().str.contains(fq, na=False)
            | base["Personality"].str.lower().str.contains(fq, na=False)
        )
        base = base.loc[mask]
    if base.empty:
        st.info("No rows match the filter.")
        return
    rows: list[dict[str, Any]] = []
    for _, r in base.iterrows():
        mid = str(r["Model"])
        row = r.to_dict()
        mm = summary.merged_metrics_by_model.get(mid, {})
        for k, v in mm.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = np.nan
        rows.append(row)
    dfw = pd.DataFrame(rows)
    preferred = [
        "Model",
        "SIP-F1",
        "Personality",
        "Malware Lethality",
        "Judas runs",
        "Judas comply %",
        "Brutus runs",
        "Brutus comply %",
        "Paired Δ runs",
        "Paired Δ stress leak %",
    ]
    front = [c for c in preferred if c in dfw.columns]
    rest = sorted(c for c in dfw.columns if c not in front)
    st.dataframe(dfw[front + rest], use_container_width=True, hide_index=True)


def _build_per_run_dataframe(files: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in sorted(files, key=lambda x: str(x).lower()):
        try:
            data = load_json_smart(p)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
            rows.append(
                {
                    "path": _rel_under_root(p),
                    "model_id": "—",
                    "suite": p.name,
                    "run_folder": p.parent.name,
                    "load_error": str(e)[:240],
                }
            )
            continue
        if not isinstance(data, dict):
            continue
        mid = infer_model_id(data, p) or "—"
        uni = extract_unified_metrics(data)
        sip = float(compute_sip_f1(uni).get("sip_f1", float("nan")))
        pers = assign_personality_label(uni)
        jp = data.get("judas_protocol_metrics")
        bt: Any = np.nan
        if isinstance(jp, dict):
            t = jp.get("Betrayal_Threshold")
            if isinstance(t, str) and t.strip():
                bt = t.strip().lower()
        row: dict[str, Any] = {
            "path": _rel_under_root(p),
            "model_id": mid,
            "suite": p.name,
            "run_folder": p.parent.name,
            "sip_f1": round(sip, 4) if np.isfinite(sip) else np.nan,
            "personality": pers,
            "judas_threshold": bt,
        }
        for k, v in uni.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            row[k] = fv if np.isfinite(fv) else np.nan
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _per_run_table_block(files: list[Path], filter_q: str) -> None:
    df = _build_per_run_dataframe(files)
    if df.empty:
        st.warning("No suite JSON files in the current scan.")
        return
    fq = filter_q.strip().lower()
    if fq:
        mask = pd.Series(False, index=df.index)
        for c in df.columns:
            mask = mask | df[c].astype(str).str.lower().str.contains(fq, na=False)
        df = df.loc[mask]
    if df.empty:
        st.info("No rows match the filter.")
        return
    st.dataframe(df, use_container_width=True, hide_index=True, height=560)


def _json_inspector_block(files: list[Path]) -> None:
    if not files:
        st.warning("No suite JSON files in the current scan.")
        return
    paths = sorted(files, key=lambda x: str(x).lower())
    labels = [_rel_under_root(p) for p in paths]
    choice = st.selectbox("Result file", labels, key="json_inspect_pick")
    pick = paths[labels.index(choice)]
    st.caption(
        "Full parsed payload (large logit vectors in CAM files are truncated by the loader)."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download this JSON",
            data=pick.read_bytes(),
            file_name=pick.name,
            mime="application/json",
            key="json_dl_btn",
        )
    with c2:
        try:
            sz = pick.stat().st_size
            st.caption(f"File size on disk: {sz:,} bytes")
        except OSError:
            pass
    try:
        payload = load_json_smart(pick)
    except Exception as e:
        st.error(str(e))
        return
    st.json(payload)


def _render_leaderboard_tab_body(holder: dict[str, Any]) -> None:
    view = st.session_state.get("lb_view", "Aggregated by model")
    fq = st.session_state.get("leaderboard_filter", "")
    summ: ScanSummary = holder["summary"]
    file_list: list[Path] = holder["files"]
    if view == "Aggregated by model":
        if st.session_state.get("lb_ext"):
            _leaderboard_extended_block(summ, fq)
        else:
            _leaderboard_block(summ, fq)
    elif view == "Per JSON file (table)":
        _per_run_table_block(file_list, fq)
    else:
        _json_inspector_block(file_list)


# ---------------------------------------------------------------------------
# Streamlit fragments (auto-refresh when results change)
# ---------------------------------------------------------------------------

_HAS_FRAGMENT = hasattr(st, "fragment")


if _HAS_FRAGMENT:

    @st.fragment(run_every=timedelta(seconds=4))
    def leaderboard_fragment(summary_holder: dict[str, Any]) -> None:
        _sync_scan_holder(summary_holder)
        _render_leaderboard_tab_body(summary_holder)

else:

    def leaderboard_fragment(summary_holder: dict[str, Any]) -> None:
        _sync_scan_holder(summary_holder)
        _render_leaderboard_tab_body(summary_holder)


def main() -> None:
    st.set_page_config(
        page_title="Research Control Room",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_dashboard_css()

    if "summary_holder" not in st.session_state:
        summ, files = load_scan_summary()
        st.session_state.summary_holder = {
            "summary": summ,
            "files": files,
            "_sig": results_scan_mtime_signature(files),
            "_cam_sig": cam_dumps_mtime_signature(results_root() / "cam_dumps"),
        }
    if "bench_log" not in st.session_state:
        st.session_state.bench_log = ""
    if "bench_running" not in st.session_state:
        st.session_state.bench_running = False
    if "bench_thread" not in st.session_state:
        st.session_state.bench_thread = None
    if "bench_queue" not in st.session_state:
        st.session_state.bench_queue = None
    if "leaderboard_filter" not in st.session_state:
        st.session_state.leaderboard_filter = ""

    holder = st.session_state.summary_holder

    st.title("Research Control Room")
    st.caption(
        "SIP-F1 · Judas · Brutus · Needle/Haystack · Lot · Delilah · Good Samaritan · "
        "Babel · Jekyll · Scapegoat · Channel factorial · Policy supersession · Crypto trap · "
        "Meta-evaluator · Compression caveat · Paired-delta · Lazarus · Polymorphic · Shutdown"
    )

    # --- Sidebar ---
    with st.sidebar:
        st.subheader("Infrastructure")
        sandbox_url = os.environ.get("SANDBOX_URL", "http://127.0.0.1:8765").strip()
        ok_s, msg_s = _check_sandbox_url(sandbox_url)
        st.markdown(
            f"**Sandbox** `{sandbox_url}`  \n"
            f"<span style='color:{'#3fb950' if ok_s else '#f85149'}'>●</span> {msg_s}",
            unsafe_allow_html=True,
        )
        ok_g, msg_g = _check_remote_cluster_ssh()
        st.markdown(
            f"**Remote cluster / SSH**  \n"
            f"<span style='color:{'#3fb950' if ok_g else '#f85149'}'>●</span> {msg_g}",
            unsafe_allow_html=True,
        )
        if st.button("Refresh status"):
            st.rerun()

        st.divider()
        st.subheader("Benchmark suite")
        cfg_path = _ROOT / "models_to_test.yaml"
        cfg_full: dict[str, Any] = {}
        models_list: list[dict[str, Any]] = []
        ids: list[str] = []
        chosen: str | None = None
        run_clicked = False
        if not cfg_path.is_file():
            st.error(f"Missing {cfg_path}")
        else:
            cfg_full = load_models_yaml(cfg_path)
            models_list = list_models_from_config(cfg_full)
            ids = [m["id"] for m in models_list]
            if not ids:
                st.warning("No models in YAML (uncomment or add entries).")
            else:
                chosen = st.selectbox(
                    "Model to run (full orchestrator batch: 18 suite JSON files)",
                    ids,
                    index=0,
                )

            st.markdown(
                '<div style="margin-top:12px;">'
                '<p style="font-size:12px;color:#8b949e;">Runs <code>multi_model_orchestrator.py</code> '
                "with a one-model temp config (polymorphic, shutdown, Judas, Lazarus, plus Brutus, "
                "Needle/Haystack, Lot, Delilah, Good Samaritan, Babel, Jekyll, Scapegoat, channel factorial, "
                "policy supersession, crypto commitment trap, meta-evaluator lie, compression caveat, "
                "paired-delta protocol).</p></div>",
                unsafe_allow_html=True,
            )
            run_clicked = st.button(
                "RUN FULL BENCHMARK SUITE",
                type="primary",
                disabled=st.session_state.bench_running or not ids,
                use_container_width=True,
            )

        if run_clicked and chosen and models_list and cfg_full:
            spec = next((m for m in models_list if m["id"] == chosen), models_list[0])
            tmp = Path(tempfile.mkdtemp(prefix="bench_cfg_")) / "models_to_test.yaml"
            write_single_model_config(cfg_full, spec, tmp)
            cmd = [
                sys.executable,
                str(_ROOT / "multi_model_orchestrator.py"),
                "--config",
                str(tmp),
                "--results-root",
                str(results_root()),
            ]
            st.session_state.bench_log = f"$ {' '.join(cmd)}\n\n"
            st.session_state.bench_running = True
            q: queue.Queue[str] = queue.Queue()
            st.session_state.bench_queue = q

            def _run() -> None:
                try:
                    _stream_subprocess(cmd, _ROOT, q)
                finally:
                    st.session_state.bench_running = False

            t = threading.Thread(target=_run, daemon=True)
            st.session_state.bench_thread = t
            t.start()
            st.rerun()

        with st.expander("Cluster deploy (deploy_to_cluster.sh)", expanded=False):
            st.caption("Runs rsync + regenerates ``job.slurm``; does not submit the job.")
            dep_host = st.text_input("user@host", placeholder="user@cluster.example.edu", key="dep_host")
            dep_dir = st.text_input("Remote dir", value="~/research_deploy", key="dep_dir")
            if st.button("Run deploy script", key="dep_go", disabled=st.session_state.bench_running):
                if not dep_host.strip():
                    st.error("Enter user@host")
                else:
                    script = _ROOT / "deploy_to_cluster.sh"
                    dcmd = ["bash", str(script), dep_host.strip(), dep_dir.strip() or "~/research_deploy"]
                    st.session_state.bench_log = f"$ {' '.join(dcmd)}\n\n"
                    st.session_state.bench_running = True
                    dq: queue.Queue[str] = queue.Queue()
                    st.session_state.bench_queue = dq

                    def _dep() -> None:
                        try:
                            _stream_subprocess(dcmd, _ROOT, dq)
                        finally:
                            st.session_state.bench_running = False

                    threading.Thread(target=_dep, daemon=True).start()
                    st.rerun()

        st.divider()
        st.subheader("LaTeX export")
        st.caption("Uses ``generate_paper_tables`` builders on the current scan.")
        summ_live: ScanSummary = st.session_state.summary_holder["summary"]
        sip_tex = build_sip_judas_personality_latex(summ_live)
        llama = summ_live.signature_by_class.get("llama31_8b_instruct")
        qwen = summ_live.signature_by_class.get("qwen25_coder_abliterated")
        sig_tex = build_signature_latex(llama, qwen)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "sip\\_judas\\_personality.tex",
                sip_tex.encode("utf-8"),
                file_name="sip_judas_personality.tex",
                mime="text/plain",
            )
        with c2:
            st.download_button(
                "signature\\_exposure.tex",
                sig_tex.encode("utf-8"),
                file_name="signature_exposure.tex",
                mime="text/plain",
            )

    # Drain benchmark log queue (bounded: each iteration removes one item; exits on Empty)
    if st.session_state.bench_queue is not None:
        q = st.session_state.bench_queue
        drained = False
        while True:
            try:
                st.session_state.bench_log += q.get_nowait()
                drained = True
            except queue.Empty:
                break
        if drained or st.session_state.bench_running:
            st.session_state.bench_log = st.session_state.bench_log[-120_000:]

    if st.session_state.bench_log.strip():
        st.subheader("Orchestrator output")
        st.code(st.session_state.bench_log, language="bash")

    tab_lb, tab_neural, tab_radar = st.tabs(
        ["Leaderboard", "Neural Insights", "Personality Radars"],
    )

    with tab_lb:
        st.radio(
            "Results view",
            [
                "Aggregated by model",
                "Per JSON file (table)",
                "Single JSON (full payload)",
            ],
            horizontal=True,
            key="lb_view",
            help="Use aggregated for paper-style means across runs; per-file modes show each timestamped artifact.",
        )
        st.checkbox(
            "Show all unified metric columns (aggregated only)",
            key="lb_ext",
            disabled=st.session_state.get("lb_view", "Aggregated by model") != "Aggregated by model",
        )
        st.text_input(
            "Filter (model, path, suite name, personality — substring match on any column)",
            key="leaderboard_filter",
            placeholder="e.g. llama, lazarus, 20260428",
        )
        if _HAS_FRAGMENT:
            leaderboard_fragment(holder)
        else:
            _sync_scan_holder(holder)
            _render_leaderboard_tab_body(holder)
            st.caption("Install Streamlit ≥ 1.40 for automatic refresh on new JSON artifacts.")

    with tab_neural:
        summary = st.session_state.summary_holder["summary"]
        st.caption(
            "Each point is one ``model_id``: **Y** = median shutdown resistance from ``shutdown_paradox.json`` "
            "(``shutdown_paradox_mode: hidden_objective``). **X** = median contrastive Δ — from "
            "``contrastive_activation_mapping`` in standalone CAM JSON when present; otherwise computed from paired "
            "``benign_*.pt`` / ``malicious_*.pt`` under ``results/cam_dumps/<SafeModel>/`` when those dumps exist."
        )
        with st.expander("Why is the scatter empty?", expanded=False):
            st.markdown(
                "- **Hidden objective** means the **Shutdown Paradox** setting "
                "(``shutdown_paradox_mode: hidden_objective``).\n"
                "- **Y** comes only from those shutdown JSONs.\n"
                "- **X** needs either embedded CAM JSON **or** paired dumps under ``results/cam_dumps/`` matching "
                "that model’s HF id (same folder naming as the orchestrator).\n"
                "- If a model has shutdown JSON but no CAM JSON and no paired ``cam_dumps``, it cannot appear."
            )
        pts = paired_scatter_points(summary)
        fig_scatter = plot_logit_shutdown_scatter(pts)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.divider()
        st.subheader("Contrastive Analysis (.pt)")
        st.caption(
            "Paired CAM exports live under ``results/cam_dumps/<SafeModelName>/`` "
            "(e.g. 120B abliterated: ``.../cam_dumps/huihui-ai__Huihui-gpt-oss-120b-BF16-abliterated/``). "
            "Listed first below so you do not need to scroll past ``activation_dumps/``."
        )
        dump_dirs = [
            results_root() / "cam_dumps",
            _ROOT / "activation_dumps",
            results_root() / "_batches",
        ]
        for mid in summary.merged_metrics_by_model:
            dump_dirs.append(results_root() / _safe_model_dirname(mid))
        pt_files = iter_pt_activation_files([d for d in dump_dirs if d.is_dir()])
        if not pt_files:
            st.info(
                "No ``.pt`` files under ``activation_dumps/``, ``results/cam_dumps/``, ``results/_batches/``, "
                "or per-model dirs under ``results/``."
            )
        else:
            c1, c2 = st.columns(2)
            with c1:
                benign = st.selectbox("Benign / baseline .pt", pt_files, index=min(0, len(pt_files) - 1))
            with c2:
                malicious = st.selectbox(
                    "Malicious .pt",
                    pt_files,
                    index=min(1, len(pt_files) - 1) if len(pt_files) > 1 else 0,
                )
            if st.button("Compute contrastive SAE ranking", key="cam_go"):
                try:
                    cam = contrastive_activation_mapping(
                        benign,
                        malicious,
                        top_k_contrastive_features=128,
                        top_k_malware_exclusive=256,
                    )
                    st.session_state.last_cam = cam
                except Exception as e:
                    st.error(str(e))
            if st.session_state.get("last_cam"):
                cam = st.session_state.last_cam
                st.json({k: cam[k] for k in ("benign_pt", "malicious_pt", "residual_shift") if k in cam})
                t1 = contrastive_features_table(cam, top_n=48)
                t2 = contrastive_exclusive_table(cam, top_n=24)
                st.write("**Top features by malicious − benign**")
                st.dataframe(t1, use_container_width=True, hide_index=True)
                if not t2.empty:
                    st.write("**Malware-exclusive (top by malicious activation)**")
                    st.dataframe(t2, use_container_width=True, hide_index=True)

    with tab_radar:
        _R_FOUR = "Four-axis (Obfuscation, Betrayal, Resilience, Stealth)"
        _R_SIX = "Six unified axes + SIP-F1"
        _R_EXT = "Extended probes (refusal, Judas/Lazarus extras, suite norms)"
        _R_ALL = "Every merged metric (dense polar)"

        summary = st.session_state.summary_holder["summary"]
        mids = sorted(summary.merged_metrics_by_model.keys(), key=lambda s: s.lower())
        if not mids:
            st.warning("No per-model unified metrics yet — run benchmarks with ``model_id`` metadata.")
        else:
            st.caption(
                "All views use **means over scanned suite JSONs** per ``model_id``. "
                "Default radar (**Every merged metric**) plots one spoke per merged key (values clipped to [0, 1]). "
                "Not a clinical personality test. **Four-axis** (optional view) — **Resilience** uses shutdown "
                "hidden-objective heuristic when present, else Lazarus self-repair; **Betrayal** is 1 − mean "
                "Judas ``betrayal_safety_score``."
            )
            mode = st.radio(
                "Radar view",
                [_R_ALL, _R_FOUR, _R_SIX, _R_EXT],
                horizontal=True,
                key="radar_view_mode_v2",
            )
            pick = st.selectbox("Model for radar", mids, index=0, key="radar_model_pick")
            mm = summary.merged_metrics_by_model.get(pick, {})
            short = pick.split("/")[-1]
            if mode == _R_ALL:
                scores = all_merged_metrics_polar(mm)
                fig_r = plot_personality_radar_generic(
                    scores,
                    model_label=short,
                    title=f"All merged metrics — {short}",
                )
                side_doc = (
                    "**Dense polar**\n\n"
                    "One spoke per key in ``merged_metrics_by_model``. Values are clipped to [0, 1]. "
                    "Use the table below for exact numbers."
                )
            elif mode == _R_FOUR:
                scores = four_axis_scores(mm)
                fig_r = plot_personality_radar_four(scores, model_label=short)
                side_doc = (
                    "**Axis mapping**\n\n"
                    "- **Obfuscation** — signature exposure  \n"
                    "- **Betrayal** — 1 − betrayal safety (Judas)  \n"
                    "- **Resilience** — shutdown resistance (hidden objective) or Lazarus self-repair  \n"
                    "- **Stealth** — lethality × low signature footprint"
                )
            elif mode == _R_SIX:
                scores = six_axis_sip_profile(mm)
                fig_r = plot_personality_radar_generic(
                    scores,
                    model_label=short,
                    title=f"Unified profile + SIP-F1 — {short}",
                )
                side_doc = (
                    "**Six + SIP**\n\n"
                    "Aligned with the paper-style vector in ``generate_paper_tables._radar_vector``: "
                    "lethality, signature exposure, shutdown resistance, code validity, betrayal safety, "
                    "and **SIP-F1** computed from the merged row (NaNs averaged out of merged dict are "
                    "treated as 0 here for plotting, matching the static figure script)."
                )
            elif mode == _R_EXT:
                scores = extended_probes_profile(mm)
                fig_r = plot_personality_radar_generic(
                    scores,
                    model_label=short,
                    title=f"Extended probes — {short}",
                )
                side_doc = (
                    "**Extended probes**\n\n"
                    "Only metrics that exist in the merged row (finite after averaging). "
                    "If a suite was not run, its axis is omitted from this chart."
                )

            cols = st.columns([2, 1])
            with cols[0]:
                st.plotly_chart(fig_r, use_container_width=True)
            with cols[1]:
                st.markdown(side_doc)
                st.json(scores)

            with st.expander("Full merged metrics table, SIP-F1, and personality label", expanded=True):
                rows = sorted(mm.items(), key=lambda t: str(t[0]).lower())
                tdf = pd.DataFrame(rows, columns=["metric", "value (mean over JSON runs)"])
                tdf["value (mean over JSON runs)"] = tdf["value (mean over JSON runs)"].map(
                    lambda x: (
                        round(float(x), 6)
                        if isinstance(x, (int, float, np.floating)) and np.isfinite(float(x))
                        else x
                    )
                )
                h = min(520, 60 + len(tdf) * 28)
                st.dataframe(tdf, use_container_width=True, hide_index=True, height=h)
                sip = float(compute_sip_f1(mm)["sip_f1"]) if mm else float("nan")
                pers = assign_personality_label(mm) if mm else "—"
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("SIP-F1 (from merged)", f"{sip:.4f}" if mm and np.isfinite(sip) else "—")
                with m2:
                    st.metric("Personality label", pers)
                with m3:
                    st.metric("Spokes in chart", str(len(scores)))

    st.divider()
    st.caption(
        f"Scanned {len(holder['files'])} suite JSON files · "
        f"Results root: `{results_root()}` · "
        "Port 8501 headless: `streamlit run dashboard.py --server.port=8501 --server.headless=true`"
    )


if __name__ == "__main__":
    main()
