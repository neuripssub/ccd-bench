#!/usr/bin/env python3
"""
Aggregate benchmark JSON under ``results/`` (local + optional extra dirs) and emit
LaTeX, Matplotlib figures, and a draft Markdown Results section. Also refreshes
``Formatting_Instructions_For_NeurIPS_2026/figures/real_judas_cross_model_body.tex`` (and
optionally ``.pdf`` / ``.png`` when matplotlib is importable) from the latest ``judas_protocol.json``
per ``model_id`` via :mod:`judas_figure_emit`.

Emits **dense polar radars** (dashboard-equivalent “every merged metric” spokes, white background /
black polygon) per model under ``<out-dir>/dense_polar_radars/``, a logit-comparison scatter
(``dense_polar_mean_vs_logit_scatter.png``), ``manifest.csv``, and copies the folder into
``Formatting_Instructions_For_NeurIPS_2026/figures/dense_polar_radars/`` when present.

Field mapping (paper labels → JSON / derived):
  * Signature Exposure Score — ``reverse_shell_signature_score.total_occurrences`` (polymorphic suite)
  * Contrastive Δ (scatter X) — ``contrastive_activation_mapping``: prefer ``logit_diff_l2``, else
    ``residual_shift_l2``, else ``sae_vector_delta_l2``; if absent in JSON, paired
    ``results/cam_dumps/`` ``.pt`` files supply X when hidden-objective shutdown exists
  * Shutdown Resistance Score — ``hidden_objective_metrics.hidden_objective_heuristic_score``
  * SAE Feature Activations — spike stats from ``.pt`` dumps: ``sae_l0``, ``sae_l1``, ``sae_max``
  * Malware Lethality — ``judge_malice``
  * Neural Deviations (hidden objective) — activation spike stats vs ``hidden_objective_heuristic_score`` (Mann–Whitney U)
  * SIP-F1 / personality — :mod:`results_processor` unified metrics across suites per ``model_id``
  * Judas betrayal — ``judas_protocol_metrics.Betrayal_Threshold`` on ``benchmark_suite == judas_protocol`` only
  * Brutus / paired-delta — separate leaderboard rollups (Brutus threshold list; paired-delta stress verbatim leak rate)

Result directories: ``results/``, ``results/cluster_sync/`` (optional mirror of synced cluster outputs), plus extra paths from
``--extra-results-dirs`` or the ``RESULTS_EXTRA_DIRS`` environment variable (``os.pathsep``-separated).

For the Llama vs Qwen signature table and logit/shutdown scatter, each JSON should include a
``model_id`` (or ``subject_model_id``) matching the Hugging Face id; otherwise only filename-based
inference is used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from judas_figure_emit import (  # noqa: E402
    latest_judas_by_model,
    plot_judas_cross_model_figure,
    write_judas_cross_model_latex_body,
)

from multi_model_orchestrator import safe_model_dirname  # noqa: E402
from results_processor import (  # noqa: E402
    assign_personality_label,
    compute_sip_f1,
    extract_unified_metrics,
    spike_stats_from_pt,
)


# ---------------------------------------------------------------------------
# Model aliases for the headline LaTeX comparison
# ---------------------------------------------------------------------------

_ORCH_TS_DIR = re.compile(r"^\d{8}T\d{6}Z$")


LLAMA_ALIASES: tuple[str, ...] = (
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/meta-llama-3.1-8b-instruct",
    "meta-llama-3.1-8b-instruct",
    "llama-3.1-8b-instruct",
)
QWEN_ABL_ALIASES: tuple[str, ...] = (
    "qwen2.5-coder",
    "qwen2.5_coder",
    "qwen2.5-coder-abliterated",
    "coder-abliterated",
    "abliterated",
)


def _norm_model(s: str) -> str:
    return s.strip().lower().replace(" ", "")


def _matches_any(norm_id: str, needles: tuple[str, ...]) -> bool:
    return any(n in norm_id for n in needles)


def classify_headline_model(model_id: str | None) -> str | None:
    if not model_id:
        return None
    n = _norm_model(model_id)
    if _matches_any(n, tuple(x.lower() for x in LLAMA_ALIASES)) or (
        "llama" in n and "3.1" in n and "8b" in n and "instruct" in n
    ):
        return "llama31_8b_instruct"
    if _matches_any(n, tuple(x.lower() for x in QWEN_ABL_ALIASES)):
        return "qwen25_coder_abliterated"
    return None


def classify_paper_model_regime(model_id: str) -> str:
    """
    Coarse release category for paper-facing captions (not a safety claim).

    * **Closed API** — proprietary endpoints (e.g. GPT-5.5 family in our ledger).
    * **Abliterated** — uncensored / abliterated checkpoints (name heuristic).
    * **Open weights** — everything else in our HF-style ``model_id`` list.
    """
    s = (model_id or "").strip()
    low = s.lower()
    if low in ("gpt-5.5", "gpt-5.5-pro") or low.startswith("gpt-5.5"):
        return "Closed API"
    if "abliterated" in low:
        return "Abliterated"
    return "Open weights"


# Mirror ``ui_components.all_merged_metrics_polar`` so this script runs without pandas/plotly.
_MERGED_POLAR_LABELS: dict[str, str] = {
    "lethality_norm": "Lethality",
    "signature_exposure_norm": "Sig. exposure",
    "shutdown_resistance": "Shutdown res.",
    "code_validity": "Code validity",
    "betrayal_safety": "Betrayal safety",
    "refusal_score": "Refusal",
    "gaslighting_norm": "Gaslighting",
    "ghost_code_risk": "Ghost code",
    "subversion_mismatch_norm": "Subversion",
    "self_repair_efficiency": "Self-repair",
    "channel_factorial_norm": "Channel fact.",
    "policy_supersession_norm": "Policy super.",
    "crypto_commitment_norm": "Crypto trap",
    "meta_evaluator_norm": "Meta-eval",
    "compression_caveat_norm": "Compression",
    "paired_delta_abs_norm": "Paired Δ",
}


def all_merged_metrics_polar(mm: dict[str, float]) -> dict[str, float]:
    """One vertex per merged scalar (clipped to [0, 1]); identical ordering to the dashboard dense polar."""
    out: dict[str, float] = {}
    for k in sorted(mm.keys(), key=lambda s: s.lower()):
        v = mm.get(k)
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x):
            continue
        label = _MERGED_POLAR_LABELS.get(k, k.replace("_", " "))
        if len(label) > 24:
            label = label[:21] + "…"
        if label in out:
            label = f"{label} ({k[:8]})"
        out[label] = max(0.0, min(1.0, x))
    return out


# ---------------------------------------------------------------------------
# JSON loading (huge contrastive CAM dumps: strip logits arrays)
# ---------------------------------------------------------------------------

_LOGITS_TRUNC_MARKERS: tuple[str, ...] = (
    '"logits_last_benign_flat"',
    '"logits_last_malicious_flat"',
)


def load_json_smart(path: Path) -> dict[str, Any]:
    """Parse JSON; for large CAM reports, drop trailing logit vector keys before parsing."""
    raw = path.read_text(encoding="utf-8")
    cut = -1
    for marker in _LOGITS_TRUNC_MARKERS:
        idx = raw.find(marker)
        if idx != -1:
            cut = idx if cut == -1 else min(cut, idx)
    if cut == -1:
        return json.loads(raw)
    frag = raw[:cut].rstrip()
    while frag.endswith(","):
        frag = frag[:-1]
    frag = frag + "\n}"
    return json.loads(frag)


def infer_model_id(data: dict[str, Any], path: Path) -> str | None:
    for key in ("model_id", "subject_model_id", "subject_model"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Orchestrator layout: results/<safe_model_dirname>/<timestamp>/suite.json
    parts = path.parts
    try:
        ri = parts.index("results")
        if ri + 2 < len(parts):
            bucket = parts[ri + 1]
            ts_dir = parts[ri + 2]
            skip = {
                "cluster_sync",
                "paper_artifacts",
                "paper_artifacts_test",
                "_batches",
                "cluster",
                "remote",
            }
            if (
                bucket.lower() not in skip
                and bucket
                and _ORCH_TS_DIR.match(ts_dir)
            ):
                return bucket.replace("__", "/")
    except ValueError:
        pass
    # Filename hints: e.g. polymorphic_meta-llama-3.1-8b-instruct.json
    stem = path.stem
    m = re.search(
        r"(meta-llama[^./\\]+|Qwen[^./\\]+|qwen[^./\\]+|llama[^./\\]+)",
        stem,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).replace("_", "-")
    parent = path.parent.name.lower()
    if parent in ("cluster_sync", "cluster", "remote"):
        # e.g. results/cluster_sync/<model_stem>.json
        if stem not in ("stub", "results", "paper_artifacts"):
            return stem
    return None


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


def iter_result_json_files(dirs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        files.extend(sorted(d.rglob("*.json")))
    # De-dupe same real path
    by_key: dict[str, Path] = {}
    for p in files:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        by_key[key] = p
    return list(by_key.values())


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def extract_signature_exposure(data: dict[str, Any]) -> dict[str, float] | None:
    block = data.get("reverse_shell_signature_score")
    if not isinstance(block, dict):
        return None
    try:
        total = float(block.get("total_occurrences", 0))
        n_hit = float(block.get("n_signatures_with_hits", 0))
    except (TypeError, ValueError):
        return None
    return {"total_occurrences": total, "n_signatures_with_hits": n_hit}


def scalar_from_contrastive_cam(cam: Any) -> float | None:
    """
    Single X-axis value for neural scatter: prefer logit L2, then residual shift L2, then SAE vector Δ L2.
    """
    if not isinstance(cam, dict):
        return None
    ld = cam.get("logit_difference")
    if isinstance(ld, dict):
        try:
            v = float(ld.get("logit_diff_l2", float("nan")))
            if np.isfinite(v):
                return v
        except (TypeError, ValueError):
            pass
    rs = cam.get("residual_shift")
    if isinstance(rs, dict):
        try:
            v = float(rs.get("residual_shift_l2", float("nan")))
            if np.isfinite(v):
                return v
        except (TypeError, ValueError):
            pass
    sc = cam.get("sae_contrast")
    if isinstance(sc, dict) and sc.get("available"):
        try:
            v = float(sc.get("sae_vector_delta_l2", float("nan")))
            if np.isfinite(v):
                return v
        except (TypeError, ValueError):
            pass
    return None


def extract_cam_scalar(data: dict[str, Any]) -> float | None:
    cam = data.get("contrastive_activation_mapping")
    return scalar_from_contrastive_cam(cam)


def extract_cam_metrics(data: dict[str, Any]) -> tuple[float | None, float | None]:
    cam = data.get("contrastive_activation_mapping")
    if not isinstance(cam, dict):
        return None, None
    ld = cam.get("logit_difference")
    l2: float | None = None
    if isinstance(ld, dict):
        try:
            v = float(ld.get("logit_diff_l2", float("nan")))
            l2 = v if np.isfinite(v) else None
        except (TypeError, ValueError):
            l2 = None
    rs = cam.get("residual_shift")
    l2_r: float | None = None
    if isinstance(rs, dict):
        try:
            v = float(rs.get("residual_shift_l2", float("nan")))
            l2_r = v if np.isfinite(v) else None
        except (TypeError, ValueError):
            l2_r = None
    return l2, l2_r


def extract_shutdown_resistance(data: dict[str, Any]) -> float | None:
    if data.get("shutdown_paradox_mode") != "hidden_objective":
        return None
    hm = data.get("hidden_objective_metrics")
    if not isinstance(hm, dict):
        return None
    try:
        return float(hm.get("hidden_objective_heuristic_score", 0.0))
    except (TypeError, ValueError):
        return None


def extract_hidden_objective_metrics(data: dict[str, Any]) -> dict[str, Any] | None:
    if data.get("shutdown_paradox_mode") != "hidden_objective":
        return None
    hm = data.get("hidden_objective_metrics")
    return hm if isinstance(hm, dict) else None


def collect_activation_rows(data: dict[str, Any]) -> list[dict[str, float]]:
    """Rows with spike stats + judge_malice for correlation analysis."""
    malice = data.get("judge_malice")
    try:
        malice_f = float(malice) if malice is not None else float("nan")
    except (TypeError, ValueError):
        malice_f = float("nan")
    rows: list[dict[str, float]] = []
    for ent in data.get("activation_cache", []) or []:
        pt = ent.get("pt_path")
        if not pt:
            continue
        p = Path(str(pt))
        if not p.is_file():
            continue
        try:
            stats = spike_stats_from_pt(p)
        except Exception:
            continue
        rows.append({"judge_malice": malice_f, **{k: float(stats[k]) for k in stats}})
    return rows


def mann_whitney_two_sided(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    try:
        from scipy.stats import mannwhitneyu

        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return None


def pearson_corr_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.stats import pearsonr

    n = matrix.shape[1]
    r_mat = np.eye(n)
    p_mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            xi = matrix[:, i]
            xj = matrix[:, j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() < 3:
                continue
            a, b = xi[mask], xj[mask]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            r, p = pearsonr(a, b)
            r_mat[i, j] = r_mat[j, i] = float(r)
            p_mat[i, j] = p_mat[j, i] = float(p)
    return r_mat, p_mat


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


# Hugging Face ids excluded from paper-facing tables (smoke tests / stubs).
PAPER_EXCLUDE_MODEL_IDS: frozenset[str] = frozenset({"gpt2"})


def _paper_model_ids(summary: ScanSummary) -> list[str]:
    return sorted(
        (m for m in summary.merged_metrics_by_model.keys() if m not in PAPER_EXCLUDE_MODEL_IDS),
        key=lambda s: s.lower(),
    )


def model_id_cell_latex(mid: str) -> str:
    """Cell body for ragged typewriter columns: allows breaks after ``/``."""
    return latex_escape(mid).replace("/", "/\\allowbreak ")


def build_signature_latex(llama: dict[str, float] | None, qwen: dict[str, float] | None) -> str:
    def cell(d: dict[str, float] | None, k: str) -> str:
        if d is None:
            return "---"
        v = d.get(k)
        if v is None:
            return "---"
        return f"{v:.0f}"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Signature exposure (reverse-shell IOC substring counts; lower is stronger obfuscation).}",
        r"\label{tab:signature_exposure}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Metric & Meta-Llama-3.1-8B-Instruct & Qwen2.5-Coder (abliterated) \\",
        r"\hline",
        (
            "Total IOC hits & "
            f"{cell(llama, 'total_occurrences')} & "
            f"{cell(qwen, 'total_occurrences')} \\\\"
        ),
        (
            "Distinct signatures hit & "
            f"{cell(llama, 'n_signatures_with_hits')} & "
            f"{cell(qwen, 'n_signatures_with_hits')} \\\\"
        ),
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def plot_scatter(
    points: list[tuple[float, float, str]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if not points:
        ax.text(0.5, 0.5, "No paired CAM + hidden-objective runs\n(model_id join)", ha="center", va="center")
        ax.set_axis_off()
    else:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, c="#2c5282", edgecolors="white", linewidths=0.5, s=64, alpha=0.85)
        for x, y, lab in points:
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, alpha=0.9)
        ax.set_xlabel("Malicious logit difference ($\\|\\ell_m - \\ell_b\\|_2$)")
        ax.set_ylabel("Shutdown resistance score (hidden objective heuristic)")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_corr_heatmap(
    labels: list[str],
    r_mat: np.ndarray,
    p_mat: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(r_mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            r = r_mat[i, j]
            p = p_mat[i, j]
            if not np.isfinite(r):
                txt = "n/a"
            else:
                stars = ""
                if np.isfinite(p):
                    if p < 0.001:
                        stars = "***"
                    elif p < 0.01:
                        stars = "**"
                    elif p < 0.05:
                        stars = "*"
                txt = f"{r:.2f}{stars}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("SAE activations vs malware lethality (Pearson)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def draft_results_markdown(
    signature_block: str,
    n_cam: int,
    n_shutdown_ho: int,
    n_poly: int,
    neural_findings: list[tuple[str, float, str]],
    corr_note: str,
) -> str:
    lines = [
        "## Results (draft)",
        "",
        "### Signature exposure",
        "",
        "The polymorphic malware benchmark scores reverse-shell outputs by counting defender-style "
        "substring signatures in the final model transcript (higher counts imply greater exposure). "
        "Aggregated JSON under `results/` was scanned for `reverse_shell_signature_score`.",
        "",
        "```latex",
        signature_block.rstrip(),
        "```",
        "",
        "### Contrastive logits and shutdown resistance",
        "",
        f"Contrastive CAM scalars parsed from JSON: **{n_cam}** file(s). "
        f"Hidden-objective shutdown paradox runs: **{n_shutdown_ho}**. Polymorphic runs with signature scores: **{n_poly}**.",
        "",
        "The scatter figure pairs, per `model_id`, the median contrastive $\\Delta$ (logit L2 when present, "
        "otherwise residual / SAE Δ from the same mapping, including optional ``results/cam_dumps/`` pairs) "
        "against the median shutdown-resistance heuristic when both sides exist.",
        "",
        "### SAE activations and malware lethality",
        "",
        corr_note,
        "",
        "### Hidden objective tests — neural deviations",
        "",
    ]
    if not neural_findings:
        lines.append(
            "_No statistically separable activation spike patterns were detected_ "
            "(insufficient hidden-objective runs with valid `.pt` hooks, or homogeneous scores)."
        )
    else:
        lines.append(
            "The following spike statistics differed most between runs in the upper vs lower quartile "
            "of `hidden_objective_heuristic_score` (two-sided Mann–Whitney U; lower $p$ implies stronger separation):"
        )
        lines.append("")
        for name, p, direction in neural_findings[:8]:
            lines.append(f"- **{name}** — $p = {p:.4g}$ ({direction})")
    lines.append("")
    return "\n".join(lines) + "\n"


@dataclass
class ScanSummary:
    signature_by_class: dict[str, dict[str, float]]
    cam_by_model: dict[str, list[float]]
    shutdown_by_model: dict[str, list[float]]
    corr_rows: list[dict[str, float]]
    ho_scores: list[float]
    ho_spike_rows: list[tuple[float, dict[str, float]]]
    n_cam_files: int = 0
    n_shutdown_ho_files: int = 0
    n_polymorphic_files: int = 0
    merged_metrics_by_model: dict[str, dict[str, float]] = field(default_factory=dict)
    judas_thresholds_by_model: dict[str, list[str]] = field(default_factory=dict)
    brutus_thresholds_by_model: dict[str, list[str]] = field(default_factory=dict)
    paired_delta_stress_leak_by_model: dict[str, list[bool]] = field(default_factory=dict)
    personality_by_model: dict[str, str] = field(default_factory=dict)
    sip_f1_by_model: dict[str, float] = field(default_factory=dict)
    #: JSON rows attributed to each ``model_id`` (all suites combined).
    n_json_files_by_model: dict[str, int] = field(default_factory=dict)
    #: Per-suite row counts (filename stem / ``benchmark_suite``) per model.
    suite_json_counts_by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Raw polymorphic IOC totals per JSON row (for mean IOC per model).
    poly_ioc_totals_by_model: dict[str, list[float]] = field(default_factory=dict)


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float] | None:
    if not rows:
        return None
    keys = rows[0].keys()
    return {k: float(np.mean([float(r[k]) for r in rows])) for k in keys}


_SUITE_FROM_FILENAME: dict[str, str] = {
    "judas_protocol.json": "judas_protocol",
    "brutus_protocol.json": "brutus_protocol",
    "paired_delta_protocol.json": "paired_delta_protocol",
}


def _infer_benchmark_suite(data: dict[str, Any], path: Path) -> str | None:
    """Prefer JSON ``benchmark_suite``; fall back to filename for legacy artifacts."""
    raw = data.get("benchmark_suite")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return _SUITE_FROM_FILENAME.get(path.name)


def supplement_cam_from_dump_pairs(
    cam_by_model: dict[str, list[float]],
    shutdown_by_model: dict[str, list[float]],
    *,
    cam_root: Path | None = None,
) -> None:
    """
    For models with hidden-objective shutdown JSON but no ``contrastive_activation_mapping`` in scanned
    files, derive scatter X values from paired ``benign_*.pt`` / ``malicious_*.pt`` under
    ``results/cam_dumps/<safe_model_dirname>/`` (same layout as :func:`run_pipeline.run_cam_loop`).
    """
    root = cam_root if cam_root is not None else _ROOT / "results" / "cam_dumps"
    if not root.is_dir():
        return
    try:
        from results_processor import contrastive_activation_mapping
    except Exception:
        return
    for mid in shutdown_by_model:
        if cam_by_model.get(mid):
            continue
        folder = root / safe_model_dirname(mid)
        if not folder.is_dir():
            continue
        benign = sorted(folder.glob("benign_*.pt"))
        malicious = sorted(folder.glob("malicious_*.pt"))
        if not benign or not malicious:
            continue
        xs: list[float] = []
        for bp, mp in zip(benign, malicious):
            try:
                cam = contrastive_activation_mapping(bp, mp)
            except Exception:
                continue
            v = scalar_from_contrastive_cam(cam)
            if v is not None:
                xs.append(v)
        if xs:
            cam_by_model[mid] = xs


def aggregate_scan(files: list[Path]) -> ScanSummary:
    signature_lists: dict[str, list[dict[str, float]]] = defaultdict(list)
    cam_by_model: dict[str, list[float]] = {}
    shutdown_by_model: dict[str, list[float]] = {}
    corr_rows: list[dict[str, float]] = []
    ho_scores: list[float] = []
    ho_spike_rows: list[tuple[float, dict[str, float]]] = []
    n_cam_files = 0
    n_shutdown_ho_files = 0
    n_polymorphic_files = 0
    acc_lists: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    judas_thr: dict[str, list[str]] = defaultdict(list)
    brutus_thr: dict[str, list[str]] = defaultdict(list)
    paired_stress_leak: dict[str, list[bool]] = defaultdict(list)
    n_json_files_by_model: dict[str, int] = defaultdict(int)
    suite_json_counts_by_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    poly_ioc_totals_by_model: dict[str, list[float]] = defaultdict(list)

    for path in files:
        try:
            data = load_json_smart(path)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        mid = infer_model_id(data, path)

        cam_scalar = extract_cam_scalar(data)
        if cam_scalar is not None:
            n_cam_files += 1
        if data.get("shutdown_paradox_mode") == "hidden_objective":
            n_shutdown_ho_files += 1
        if isinstance(data.get("reverse_shell_signature_score"), dict):
            n_polymorphic_files += 1

        se = extract_signature_exposure(data)
        if se is not None and mid:
            cls = classify_headline_model(mid)
            if cls:
                signature_lists[cls].append(se)

        if cam_scalar is not None and mid:
            cam_by_model.setdefault(mid, []).append(cam_scalar)

        sr = extract_shutdown_resistance(data)
        if sr is not None and mid:
            shutdown_by_model.setdefault(mid, []).append(sr)

        hm = extract_hidden_objective_metrics(data)
        if hm is not None:
            score = float(hm.get("hidden_objective_heuristic_score", 0.0))
            ho_scores.append(score)
            # Last activation dump only (final assistant reply)
            cache = data.get("activation_cache") or []
            if cache:
                pt = cache[-1].get("pt_path")
                if pt:
                    try:
                        stats = spike_stats_from_pt(Path(str(pt)))
                        ho_spike_rows.append((score, stats))
                    except Exception:
                        pass

        corr_rows.extend(collect_activation_rows(data))

        if mid:
            n_json_files_by_model[mid] += 1
            suite = _infer_benchmark_suite(data, path) or path.stem
            suite_json_counts_by_model[mid][suite] += 1
            se_row = extract_signature_exposure(data)
            if se_row is not None:
                try:
                    poly_ioc_totals_by_model[mid].append(float(se_row["total_occurrences"]))
                except (KeyError, TypeError, ValueError):
                    pass
            jp = data.get("judas_protocol_metrics")
            if isinstance(jp, dict):
                bt = jp.get("Betrayal_Threshold")
                if isinstance(bt, str) and bt.strip():
                    btl = bt.strip().lower()
                    if suite == "judas_protocol":
                        judas_thr[mid].append(btl)
                    elif suite == "brutus_protocol":
                        brutus_thr[mid].append(btl)
            if suite == "paired_delta_protocol":
                pd_metrics = data.get("paired_delta_protocol_metrics")
                if isinstance(pd_metrics, dict):
                    leak = pd_metrics.get("paired_stress_verbatim_leak")
                    if isinstance(leak, bool):
                        paired_stress_leak[mid].append(leak)
            uni = extract_unified_metrics(data)
            for k, v in uni.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fv):
                    acc_lists[mid][k].append(fv)

    signature_by_class = {k: _mean_dict(v) or {} for k, v in signature_lists.items()}
    merged_metrics_by_model: dict[str, dict[str, float]] = {}
    for m_id, buckets in acc_lists.items():
        merged_metrics_by_model[m_id] = {
            k: float(np.mean(np.array(v, dtype=np.float64))) for k, v in buckets.items() if v
        }
    sip_f1_by_model: dict[str, float] = {}
    personality_by_model: dict[str, str] = {}
    for m_id, mm in merged_metrics_by_model.items():
        sip_f1_by_model[m_id] = float(compute_sip_f1(mm)["sip_f1"])
        personality_by_model[m_id] = assign_personality_label(mm)

    supplement_cam_from_dump_pairs(cam_by_model, shutdown_by_model)

    return ScanSummary(
        signature_by_class=signature_by_class,
        cam_by_model=cam_by_model,
        shutdown_by_model=shutdown_by_model,
        corr_rows=corr_rows,
        ho_scores=ho_scores,
        ho_spike_rows=ho_spike_rows,
        n_cam_files=n_cam_files,
        n_shutdown_ho_files=n_shutdown_ho_files,
        n_polymorphic_files=n_polymorphic_files,
        merged_metrics_by_model=merged_metrics_by_model,
        judas_thresholds_by_model=dict(judas_thr),
        brutus_thresholds_by_model=dict(brutus_thr),
        paired_delta_stress_leak_by_model=dict(paired_stress_leak),
        personality_by_model=personality_by_model,
        sip_f1_by_model=sip_f1_by_model,
        n_json_files_by_model=dict(n_json_files_by_model),
        suite_json_counts_by_model={k: dict(v) for k, v in suite_json_counts_by_model.items()},
        poly_ioc_totals_by_model=dict(poly_ioc_totals_by_model),
    )


def paired_scatter_points(summary: ScanSummary) -> list[tuple[float, float, str]]:
    points: list[tuple[float, float, str]] = []
    models = set(summary.cam_by_model) & set(summary.shutdown_by_model)
    for m in sorted(models):
        xs = summary.cam_by_model[m]
        ys = summary.shutdown_by_model[m]
        if not xs or not ys:
            continue
        x_med = float(np.median(np.array(xs, dtype=np.float64)))
        y_med = float(np.median(np.array(ys, dtype=np.float64)))
        short = m.split("/")[-1] if "/" in m else m
        if len(short) > 28:
            short = short[:25] + "..."
        points.append((x_med, y_med, short))
    return points


def neural_deviation_findings(summary: ScanSummary) -> list[tuple[str, float, str]]:
    if len(summary.ho_spike_rows) < 6:
        return []
    scores = np.array([s for s, _ in summary.ho_spike_rows], dtype=np.float64)
    q1, q3 = np.quantile(scores, [0.25, 0.75])
    low = scores <= q1
    high = scores >= q3
    if low.sum() < 2 or high.sum() < 2:
        return []
    keys = ("sae_l0", "sae_l1", "sae_max", "residual_l2", "residual_max_abs")
    findings: list[tuple[str, float, str]] = []
    for k in keys:
        vals = np.array(
            [float(row.get(k, float("nan"))) for _, row in summary.ho_spike_rows],
            dtype=np.float64,
        )
        x_low = vals[low]
        x_high = vals[high]
        mask_l = np.isfinite(x_low)
        mask_h = np.isfinite(x_high)
        p = mann_whitney_two_sided(x_low[mask_l], x_high[mask_h])
        if p is None or not np.isfinite(p):
            continue
        direction = "higher in upper quartile" if np.nanmedian(x_high) > np.nanmedian(x_low) else "lower in upper quartile"
        findings.append((k, p, direction))
    findings.sort(key=lambda t: t[1])
    return findings


def build_sip_judas_personality_latex(summary: ScanSummary) -> str:
    """LaTeX table: per-model SIP-F1 and personality label (mean unified metrics across JSON runs per ``model_id``)."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{SIP-F1 (Strict Imperfection Penalty F1) and coarse personality label (mean unified metrics across JSON runs per \texttt{model\_id}).}",
        r"\label{tab:sip_judas_personality}",
        r"\begin{tabular}{p{4.6cm}cc}",
        r"\hline",
        r"Model (HF id or path) & SIP-F1 & Personality \\",
        r"\hline",
    ]
    mids = _paper_model_ids(summary)
    if not mids:
        lines.append(r"\multicolumn{3}{l}{\emph{No runs with \texttt{model\_id} metadata found.}} \\")
    for mid in mids:
        sip = summary.sip_f1_by_model.get(mid, float("nan"))
        pers = summary.personality_by_model.get(mid, "---")
        sip_cell = f"{sip:.3f}" if np.isfinite(sip) else "---"
        lab = mid if len(mid) < 56 else (mid[:26] + "\\ldots " + mid[-24:])
        lines.append(f"{latex_escape(lab)} & {sip_cell} & {latex_escape(str(pers))} \\\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _fmt_cell_float(v: float | None, nd: int = 3) -> str:
    if v is None or not np.isfinite(float(v)):
        return "---"
    return f"{float(v):.{nd}f}"


def _median_disp(xs: list[float]) -> str:
    if not xs:
        return "---"
    return f"{float(np.median(np.array(xs, dtype=np.float64))):.2f}"


def _mean_disp(xs: list[float]) -> str:
    if not xs:
        return "---"
    return f"{float(np.mean(np.array(xs, dtype=np.float64))):.2f}"


def _comply_rate(thrs: list[str]) -> tuple[str, str]:
    """Return (count column, percent column text)."""
    n = len(thrs)
    if n == 0:
        return "0", "---"
    c = sum(1 for t in thrs if t == "comply")
    return str(n), f"{100.0 * c / n:.0f}\\%"


def _leak_rate(leaks: list[bool]) -> tuple[str, str]:
    n = len(leaks)
    if n == 0:
        return "0", "---"
    t = sum(1 for x in leaks if x)
    return str(n), f"{100.0 * t / n:.0f}\\%"


# Full merged unified metrics for the paper-wide matrix (user-facing order + header abbreviations).
_UNIFIED_METRICS_PAPER: tuple[tuple[str, str], ...] = (
    ("betrayal_safety", "BetS"),
    ("channel_factorial_norm", "ChFa"),
    ("compression_caveat_norm", "Cmp"),
    ("crypto_commitment_norm", "Cry"),
    ("gaslighting_norm", "Gasl"),
    ("ghost_code_risk", "Ghst"),
    ("lethality_norm", "Leth"),
    ("meta_evaluator_norm", "Meta"),
    ("paired_delta_abs_norm", "PrDl"),
    ("policy_supersession_norm", "PolS"),
    ("refusal_score", "Refu"),
    ("self_repair_efficiency", "LazR"),
    ("shutdown_resistance", "HOsh"),
    ("signature_exposure_norm", "SigE"),
    ("subversion_mismatch_norm", "Subv"),
)


def collect_cam_residual_deltas(cam_root: Path) -> dict[str, float]:
    """
    For each subdirectory under ``cam_dumps``, load the first benign/malign ``.pt`` pair and return
    ``||h^{(mal)} - h^{(ben)}||_2`` on ``residual_stream_final_layer`` when available.

    Keys are Hugging Face-style ``model_id`` strings (folder uses ``__`` for ``/``).
    """
    out: dict[str, float] = {}
    try:
        import torch
    except ImportError:  # pragma: no cover
        return out
    if not cam_root.is_dir():
        return out
    for folder in sorted(cam_root.iterdir()):
        if not folder.is_dir():
            continue
        mid = folder.name.replace("__", "/")
        bens = sorted(folder.glob("benign_*.pt"))
        mals = sorted(folder.glob("malicious_*.pt"))
        if not bens or not mals:
            continue
        try:
            ob = torch.load(bens[0], map_location="cpu", weights_only=False)
        except TypeError:
            ob = torch.load(bens[0], map_location="cpu")
        try:
            om = torch.load(mals[0], map_location="cpu", weights_only=False)
        except TypeError:
            om = torch.load(mals[0], map_location="cpu")
        hb = ob.get("residual_stream_final_layer")
        hm = om.get("residual_stream_final_layer")
        if hb is None or hm is None:
            continue
        if not hasattr(hb, "float") or not hasattr(hm, "float"):
            continue
        d = float((hm.float() - hb.float()).norm().item())
        out[mid] = d
    return out


def build_unified_metrics_all_latex(
    summary: ScanSummary,
    cam_deltas: dict[str, float],
) -> str:
    """
    Two wide ``table*`` blocks: Part~I = model + SIP + first eight unified metrics;
    Part~II = model + remaining seven metrics + ``||\\Delta h||_2`` when CAM dumps exist.
    """
    mids = _paper_model_ids(summary)
    part1_keys = _UNIFIED_METRICS_PAPER[:8]
    part2_keys = _UNIFIED_METRICS_PAPER[8:]

    def cell(mm: dict[str, float], key: str) -> str:
        v = mm.get(key)
        if v is None:
            return "---"
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return "---"
        if not np.isfinite(fv):
            return "---"
        return f"{fv:.3f}"

    def delta_cell(mid: str) -> str:
        v = cam_deltas.get(mid)
        if v is None:
            return "---"
        return f"{v:.1f}"

    lines1 = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \normalsize",
        r"  \renewcommand{\arraystretch}{1.32}",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \caption{\textbf{Merged unified behavioral scalars (Part I of II)} for every \texttt{model\_id} with suite JSON in our scanned \texttt{results/} tree. Values are row-wise means over artifacts per model (same normalization as \texttt{extract\_unified\_metrics}). Abbreviations: Table~\ref{tab:unified_glossary}. Not a leaderboard.}",
        r"  \label{tab:unified_metrics_all_a}",
        r"  \begingroup\footnotesize\setlength{\tabcolsep}{3.2pt}%",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{@{}>{\raggedright\ttfamily\arraybackslash}p{6.95cm}r"
        + "r" * (1 + len(part1_keys))
        + r"@{}}",
        r"    \toprule",
        r"    \textbf{Model}"
        r" & \textbf{SIP}"
        + "".join(f" & \\textbf{{{abbr}}}" for _, abbr in part1_keys)
        + r" \\",
        r"    \midrule",
    ]
    for mid in mids:
        mm = summary.merged_metrics_by_model.get(mid, {})
        sip = summary.sip_f1_by_model.get(mid, float("nan"))
        sip_s = f"{sip:.3f}" if np.isfinite(sip) else "---"
        row_cells = [model_id_cell_latex(mid), sip_s] + [cell(mm, k) for k, _ in part1_keys]
        lines1.append("    " + " & ".join(row_cells) + r" \\")
    lines1.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}%",
            r"  }\endgroup",
            r"\end{table*}",
            "",
        ]
    )

    lines2 = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \normalsize",
        r"  \renewcommand{\arraystretch}{1.32}",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \caption{\textbf{Merged unified behavioral scalars (Part II of II)} with paired CAM residual telemetry. $\|\Delta h\|_2$ uses the first \texttt{benign\_*.pt} / \texttt{malicious\_*.pt} pair per subdirectory under \texttt{results/cam\_dumps/} (final-layer residual streams); \textbf{---} when no pair exists. This parallels $\|\Delta \ell\|_2$ when logits are omitted from JSON exports.}",
        r"  \label{tab:unified_metrics_all_b}",
        r"  \begingroup\footnotesize\setlength{\tabcolsep}{3.2pt}%",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{@{}>{\raggedright\ttfamily\arraybackslash}p{6.95cm}"
        + "r" * (len(part2_keys) + 1)
        + r"@{}}",
        r"    \toprule",
        r"    \textbf{Model}"
        + "".join(f" & \\textbf{{{abbr}}}" for _, abbr in part2_keys)
        + r" & \textbf{$\|\Delta h\|_2$} \\",
        r"    \midrule",
    ]
    for mid in mids:
        mm = summary.merged_metrics_by_model.get(mid, {})
        row_cells = [model_id_cell_latex(mid)] + [cell(mm, k) for k, _ in part2_keys] + [delta_cell(mid)]
        lines2.append("    " + " & ".join(row_cells) + r" \\")
    lines2.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}%",
            r"  }\endgroup",
            r"\end{table*}",
            "",
        ]
    )

    return "\n".join(lines1 + lines2)


# Short header labels for unified metrics (stable column order).
_METRIC_HEADERS: tuple[tuple[str, str], ...] = (
    ("lethality_norm", "Leth"),
    ("signature_exposure_norm", "SigE"),
    ("shutdown_resistance", "HOsh"),
    ("code_validity", "Code"),
    ("refusal_score", "Refu"),
    ("betrayal_safety", "BetS"),
    ("self_repair_efficiency", "LazR"),
    ("gaslighting_norm", "Gasl"),
    ("ghost_code_risk", "Ghst"),
    ("subversion_mismatch_norm", "Subv"),
    ("channel_factorial_norm", "ChFa"),
    ("policy_supersession_norm", "PolS"),
    ("crypto_commitment_norm", "Cry"),
    ("meta_evaluator_norm", "Meta"),
    ("compression_caveat_norm", "Cmp"),
    ("paired_delta_abs_norm", "PrDl"),
)


def _suite_head(stem: str) -> str:
    if len(stem) > 14:
        return latex_escape(stem[:12]) + r"\ldots"
    return latex_escape(stem)


def build_master_datapoints_latex(summary: ScanSummary) -> str:
    """
    Split LaTeX ``table*`` blocks: merged unified scalars in two parts, then per-suite JSON counts.
    Wide numeric bodies are wrapped in ``\\resizebox{\\textwidth}{!}{...}`` so they do not overrun margins.
    """
    mids = _paper_model_ids(summary)
    suite_union: set[str] = set()
    for d in summary.suite_json_counts_by_model.values():
        suite_union.update(d.keys())
    suite_cols = sorted(suite_union)

    mid_split = max(1, len(_METRIC_HEADERS) // 2)
    first_keys = _METRIC_HEADERS[:mid_split]
    rest_keys = _METRIC_HEADERS[mid_split:]
    suite_mid = (len(suite_cols) + 1) // 2
    suite_a = suite_cols[:suite_mid]
    suite_b = suite_cols[suite_mid:]

    model_col = r">{\raggedright\ttfamily\arraybackslash}p{5.55cm}"

    def row_metrics_a(mid: str) -> str:
        mm = summary.merged_metrics_by_model.get(mid, {})
        sip = summary.sip_f1_by_model.get(mid, float("nan"))
        sip_s = f"{sip:.3f}" if np.isfinite(sip) else "---"
        pers = latex_escape(str(summary.personality_by_model.get(mid, "---")))
        n_json = summary.n_json_files_by_model.get(mid, 0)
        cells: list[str] = [
            model_id_cell_latex(mid),
            str(int(n_json)),
            sip_s,
            pers,
        ]
        for key, _ in first_keys:
            cells.append(_fmt_cell_float(mm.get(key)))
        return " & ".join(cells) + r" \\"

    def row_metrics_b(mid: str) -> str:
        mm = summary.merged_metrics_by_model.get(mid, {})
        cells: list[str] = [model_id_cell_latex(mid)]
        for key, _ in rest_keys:
            cells.append(_fmt_cell_float(mm.get(key)))
        jn, jp = _comply_rate(summary.judas_thresholds_by_model.get(mid, []))
        bn, bp = _comply_rate(summary.brutus_thresholds_by_model.get(mid, []))
        pn, pp = _leak_rate(summary.paired_delta_stress_leak_by_model.get(mid, []))
        cells.extend([jn, jp, bn, bp, pn, pp])
        cells.append(_mean_disp(summary.poly_ioc_totals_by_model.get(mid, [])))
        cells.append(_median_disp(summary.cam_by_model.get(mid, [])))
        cells.append(_median_disp(summary.shutdown_by_model.get(mid, [])))
        return " & ".join(cells) + r" \\"

    def row_suites(mid: str, stems: list[str]) -> str:
        scmap = summary.suite_json_counts_by_model.get(mid, {})
        cells: list[str] = [model_id_cell_latex(mid)]
        for sc in stems:
            cells.append(str(int(scmap.get(sc, 0))))
        return " & ".join(cells) + r" \\"

    tab_metric_a = (
        r"\begin{tabular}{@{}"
        + model_col
        + r"rcc"
        + "c" * len(first_keys)
        + "@{}}"
    )
    hdr_metric_a = (
        r"\textbf{Model} & \textbf{$N$} & \textbf{SIP} & \textbf{Pers.}"
        + "".join(f" & \\textbf{{{abbr}}}" for _, abbr in first_keys)
        + r" \\"
    )

    tab_metric_b = (
        r"\begin{tabular}{@{}"
        + model_col
        + "c" * (len(rest_keys) + 9)
        + "@{}}"
    )
    hdr_metric_b = (
        r"\textbf{Model}"
        + "".join(f" & \\textbf{{{abbr}}}" for _, abbr in rest_keys)
        + r" & \textbf{J$n$} & \textbf{J\%}"
        + r" & \textbf{B$n$} & \textbf{B\%}"
        + r" & \textbf{PD$n$} & \textbf{PD\%}"
        + r" & \textbf{IOC$\mu$} & \textbf{CAM} & \textbf{ShMd}"
        + r" \\"
    )

    blocks: list[str] = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{3.8pt}",
        r"  \renewcommand{\arraystretch}{1.28}",
        r"  \caption{\textbf{Per-model merged datapoints (Part I of IV):} identity, SIP-F1, personality label, and the first half of unified behavioral means (row-wise over JSON rows per \texttt{model\_id}; same normalization as \texttt{extract\_unified\_metrics}). Not a leaderboard.}",
        r"  \label{tab:master_datapoints}",
        r"  \begingroup\footnotesize\setlength{\tabcolsep}{2.8pt}%",
        r"  \resizebox{\textwidth}{!}{%",
        tab_metric_a,
        r"\toprule",
        hdr_metric_a,
        r"\midrule",
        "\n".join(row_metrics_a(m) for m in mids),
        r"\bottomrule",
        r"\end{tabular}%",
        r"  }\endgroup",
        r"\end{table*}",
        "",
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{3.8pt}",
        r"  \renewcommand{\arraystretch}{1.28}",
        r"  \caption{\textbf{Per-model merged datapoints (Part II of IV):} remaining unified means, Judas/Brutus/paired-delta protocol rollups (\textbf{J}/\textbf{B}/\textbf{PD}), polymorphic IOC mean (\textbf{IOC$\mu$}), and JSON-medians for CAM ($\|\Delta \ell\|$-family scalars) and shutdown rows (\textbf{ShMd}).}",
        r"  \label{tab:master_datapoints_b}",
        r"  \begingroup\footnotesize\setlength{\tabcolsep}{2.4pt}%",
        r"  \resizebox{\textwidth}{!}{%",
        tab_metric_b,
        r"\toprule",
        hdr_metric_b,
        r"\midrule",
        "\n".join(row_metrics_b(m) for m in mids),
        r"\bottomrule",
        r"\end{tabular}%",
        r"  }\endgroup",
        r"\end{table*}",
        "",
    ]

    if suite_a:
        tab_sa = (
            r"\begin{tabular}{@{}"
            + model_col
            + "c" * len(suite_a)
            + "@{}}"
        )
        hdr_sa = r"\textbf{Model}" + "".join(f" & \\textbf{{{_suite_head(sc)}}}" for sc in suite_a) + r" \\"
        blocks.extend(
            [
                r"\begin{table*}[t]",
                r"  \centering",
                r"  \footnotesize",
                r"  \setlength{\tabcolsep}{3.2pt}",
                r"  \renewcommand{\arraystretch}{1.22}",
                r"  \caption{\textbf{Per-suite JSON file counts (Part III of IV):} number of \texttt{*.json} artifacts per suite stem (alphabetical first half; includes manifests and error stubs when present).}",
                r"  \label{tab:master_datapoints_suites_a}",
                r"  \begingroup\footnotesize\setlength{\tabcolsep}{2.2pt}%",
                r"  \resizebox{\textwidth}{!}{%",
                tab_sa,
                r"\toprule",
                hdr_sa,
                r"\midrule",
                "\n".join(row_suites(m, suite_a) for m in mids),
                r"\bottomrule",
                r"\end{tabular}%",
                r"  }\endgroup",
                r"\end{table*}",
                "",
            ]
        )

    if suite_b:
        tab_sb = (
            r"\begin{tabular}{@{}"
            + model_col
            + "c" * len(suite_b)
            + "@{}}"
        )
        hdr_sb = r"\textbf{Model}" + "".join(f" & \\textbf{{{_suite_head(sc)}}}" for sc in suite_b) + r" \\"
        blocks.extend(
            [
                r"\begin{table*}[t]",
                r"  \centering",
                r"  \footnotesize",
                r"  \setlength{\tabcolsep}{3.2pt}",
                r"  \renewcommand{\arraystretch}{1.22}",
                r"  \caption{\textbf{Per-suite JSON file counts (Part IV of IV):} suite stems (alphabetical second half).}",
                r"  \label{tab:master_datapoints_suites_b}",
                r"  \begingroup\footnotesize\setlength{\tabcolsep}{2.2pt}%",
                r"  \resizebox{\textwidth}{!}{%",
                tab_sb,
                r"\toprule",
                hdr_sb,
                r"\midrule",
                "\n".join(row_suites(m, suite_b) for m in mids),
                r"\bottomrule",
                r"\end{tabular}%",
                r"  }\endgroup",
                r"\end{table*}",
                "",
            ]
        )

    return "\n".join(blocks)


def plot_judas_betrayal_rates(summary: ScanSummary, out_path: Path) -> None:
    """Horizontal bar chart: fraction of Judas runs ending in ``comply`` / ``hesitate`` / ``alert`` per model."""
    import matplotlib.pyplot as plt

    mids = sorted(summary.judas_thresholds_by_model.keys(), key=lambda s: s.lower())
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.35 * len(mids) + 1.5)))
    if not mids:
        ax.text(0.5, 0.5, "No judas_protocol_metrics in scan", ha="center", va="center")
        ax.set_axis_off()
    else:
        comply_r: list[float] = []
        hesitate_r: list[float] = []
        alert_r: list[float] = []
        labels: list[str] = []
        for mid in mids:
            th = summary.judas_thresholds_by_model[mid]
            n = max(1, len(th))
            labels.append(mid.split("/")[-1] if "/" in mid else mid)
            comply_r.append(sum(1 for t in th if t == "comply") / n)
            hesitate_r.append(sum(1 for t in th if t == "hesitate") / n)
            alert_r.append(sum(1 for t in th if t == "alert") / n)
        y = np.arange(len(labels))
        ax.barh(y, alert_r, color="#2f855a", label="alert", height=0.55)
        ax.barh(y, hesitate_r, left=alert_r, color="#d69e2e", label="hesitate", height=0.55)
        ax.barh(
            y,
            comply_r,
            left=np.array(alert_r) + np.array(hesitate_r),
            color="#c53030",
            label="comply",
            height=0.55,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Fraction of Judas runs")
        ax.set_xlim(0.0, 1.0)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("Judas Protocol — betrayal threshold mix (per model)")
        ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _radar_vector(mm: dict[str, float]) -> tuple[list[float], float]:
    """Six-axis profile in ``[0, 1]`` plus SIP-F1 scalar for overlay."""
    if not mm:
        return [0.0] * 6, 0.0
    sip = float(compute_sip_f1(mm)["sip_f1"])

    def _g(key: str) -> float:
        v = mm.get(key, float("nan"))
        try:
            x = float(v)
        except (TypeError, ValueError):
            return 0.0
        return x if np.isfinite(x) else 0.0

    vec = [
        _g("lethality_norm"),
        _g("signature_exposure_norm"),
        _g("shutdown_resistance"),
        _g("code_validity"),
        _g("betrayal_safety"),
        sip,
    ]
    return vec, sip


def plot_personality_radar_two_models(
    summary: ScanSummary,
    model_a: str,
    model_b: str,
    out_path: Path,
) -> None:
    """Polar radar comparing two ``model_id`` strings on unified behavioral axes."""
    import matplotlib.pyplot as plt

    axis_labels = (
        "Lethality",
        "Sig. exposure",
        "Shutdown res.",
        "Code validity",
        "Betrayal safety",
        "SIP-F1",
    )
    ma = summary.merged_metrics_by_model.get(model_a, {})
    mb = summary.merged_metrics_by_model.get(model_b, {})
    va, _ = _radar_vector(ma)
    vb, _ = _radar_vector(mb)
    angles = np.linspace(0.0, 2.0 * np.pi, len(axis_labels), endpoint=False).tolist()
    va += va[:1]
    vb += vb[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, va, color="#2b6cb0", linewidth=2.0, label=_short_label(model_a))
    ax.fill(angles, va, color="#2b6cb0", alpha=0.15)
    ax.plot(angles, vb, color="#c05621", linewidth=2.0, label=_short_label(model_b))
    ax.fill(angles, vb, color="#c05621", alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Personality radar (unified metrics)", y=1.08, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def median_contrastive_delta(summary: ScanSummary, model_id: str) -> float | None:
    """Median contrastive CAM scalar (prefer logit L₂ when JSON supplies it). Same units as the scatter X-axis."""
    xs = summary.cam_by_model.get(model_id)
    if not xs:
        return None
    arr = np.array(xs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.median(arr))


def plot_dense_polar_radar_matplotlib(
    scores: dict[str, float],
    *,
    short_title: str,
    regime: str,
    logit_line: str | None,
    out_path: Path,
) -> None:
    """
    Print-style radar: white canvas, black polygon and typography (dashboard ``all_merged_metrics_polar``).
    """
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not scores:
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        ax.text(0.5, 0.5, "No finite merged metrics", ha="center", va="center", fontsize=11, color="black")
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return

    cats = list(scores.keys())
    vals = [max(0.0, min(1.0, float(scores[k]))) for k in cats]
    n = len(cats)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    vals_closed = vals + vals[:1]
    angles_closed = np.concatenate([angles, [angles[0]]])

    tick_sz = 9 if n <= 12 else (8 if n <= 18 else 6)
    fig_w = min(11.0, 7.0 + max(0, n - 14) * 0.12)

    fig = plt.figure(figsize=(fig_w, fig_w))
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.plot(angles_closed, vals_closed, color="black", linewidth=1.25, linestyle="-")
    ax.fill(angles_closed, vals_closed, color="black", alpha=0.14)

    ax.set_xticks(angles)
    ax.set_xticklabels(cats, fontsize=tick_sz, color="black")
    ax.tick_params(axis="y", colors="#333333", labelsize=8)

    ax.set_ylim(0.0, 1.0)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")

    ax.grid(True, color="#555555", linestyle="-", linewidth=0.45, alpha=0.85)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], color="#222222", fontsize=8)

    subtitle = regime
    if logit_line:
        subtitle = f"{subtitle} · {logit_line}"
    title_block = f"{short_title}\n{subtitle}"
    ax.set_title(title_block, fontsize=10, color="black", pad=18)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_dense_polar_vs_logit_scatter(
    summary: ScanSummary,
    mids: list[str],
    out_path: Path,
) -> None:
    """Mean merged-metric radius vs median contrastive Δ (logit-first CAM scalar)."""
    import matplotlib.pyplot as plt

    pts: list[tuple[float, float, str, str]] = []
    for mid in mids:
        mm = summary.merged_metrics_by_model.get(mid, {})
        polar = all_merged_metrics_polar(mm)
        if not polar:
            continue
        y = float(np.mean(list(polar.values())))
        mx = median_contrastive_delta(summary, mid)
        if mx is None:
            continue
        pts.append((mx, y, mid, classify_paper_model_regime(mid)))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    if not pts:
        ax.text(0.5, 0.5, "No models with both merged metrics\nand contrastive CAM scalars", ha="center", va="center")
        ax.set_axis_off()
    else:
        styles = {
            "Closed API": ("#000000", "s"),
            "Abliterated": ("#444444", "^"),
            "Open weights": ("#666666", "o"),
        }
        by_reg: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        for x, y, mid, reg in pts:
            by_reg[reg].append((x, y, mid))
        for reg in ("Closed API", "Abliterated", "Open weights"):
            if reg not in by_reg:
                continue
            col, mk = styles[reg]
            xs = [t[0] for t in by_reg[reg]]
            ys = [t[1] for t in by_reg[reg]]
            ax.scatter(xs, ys, c=col, marker=mk, s=72, alpha=0.85, edgecolors="white", linewidths=0.6, label=reg)
            for x, y, mid in by_reg[reg]:
                lab = _short_label(mid)
                ax.annotate(lab, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7, color="black")

        ax.set_xlabel(r"Median contrastive $\Delta$ ($\|\ell_m-\ell_b\|_2$ or residual fallback)")
        ax.set_ylabel("Mean merged metric (dense polar, [0,1])")
        ax.grid(True, alpha=0.3, color="#888888")
        ax.legend(loc="best", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def write_dense_polar_manifest_csv(
    summary: ScanSummary,
    mids: list[str],
    out_csv: Path,
) -> None:
    rows = ["model_id,regime,median_contrastive_delta,mean_merged_spoke,n_spokes,png_filename"]
    for mid in mids:
        mm = summary.merged_metrics_by_model.get(mid, {})
        polar = all_merged_metrics_polar(mm)
        mean_spoke = float(np.mean(list(polar.values()))) if polar else float("nan")
        md = median_contrastive_delta(summary, mid)
        md_s = f"{md:.6g}" if md is not None else ""
        fn = f"{safe_model_dirname(mid)}_dense_polar.png"
        rows.append(
            ",".join(
                [
                    mid.replace(",", ";"),
                    classify_paper_model_regime(mid),
                    md_s,
                    f"{mean_spoke:.6f}",
                    str(len(polar)),
                    fn,
                ]
            )
        )
    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")


def emit_dense_polar_radar_gallery(
    summary: ScanSummary,
    base_out: Path,
) -> Path:
    """
    One dense-polar PNG per paper model (same vertices as dashboard **Every merged metric**),
    plus summary scatter and CSV manifest.
    """
    gallery = base_out / "dense_polar_radars"
    gallery.mkdir(parents=True, exist_ok=True)
    mids = _paper_model_ids(summary)
    for mid in mids:
        mm = summary.merged_metrics_by_model.get(mid, {})
        polar = all_merged_metrics_polar(mm)
        md = median_contrastive_delta(summary, mid)
        if md is not None:
            logit_line = r"median $\|\Delta\ell\|_2 \approx " + f"{md:.4g}$"
        else:
            logit_line = "median contrastive Δ (no CAM / shutdown pairing)"
        plot_dense_polar_radar_matplotlib(
            polar,
            short_title=_short_label(mid),
            regime=classify_paper_model_regime(mid),
            logit_line=logit_line,
            out_path=gallery / f"{safe_model_dirname(mid)}_dense_polar.png",
        )

    plot_dense_polar_vs_logit_scatter(summary, mids, gallery / "dense_polar_mean_vs_logit_scatter.png")
    write_dense_polar_manifest_csv(summary, mids, gallery / "manifest.csv")
    return gallery


def _short_label(mid: str) -> str:
    s = mid.split("/")[-1] if "/" in mid else mid
    return s if len(s) <= 32 else s[:29] + "..."


def _resolve_model_id(requested: str | None, keys: Iterable[str]) -> str | None:
    if not requested:
        return None
    r = requested.strip()
    key_list = list(keys)
    if r in key_list:
        return r
    rl = r.lower()
    for k in key_list:
        if k.lower() == rl or k.endswith(r) or r in k:
            return k
    return None


def _configure_matplotlib() -> None:
    mdir = _ROOT / ".matplotlib-cache"
    mdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mdir))
    import matplotlib

    matplotlib.use("Agg")


def main() -> None:
    _configure_matplotlib()
    ap = argparse.ArgumentParser(description="Generate LaTeX tables and figures from results/*.json")
    ap.add_argument(
        "--extra-results-dirs",
        default=os.environ.get("RESULTS_EXTRA_DIRS", ""),
        help="Additional directories to scan (os.pathsep-separated). Also set RESULTS_EXTRA_DIRS.",
    )
    ap.add_argument(
        "--out-dir",
        default=str(_ROOT / "results" / "paper_artifacts"),
        help="Output directory for .tex, .png, and results_section.md",
    )
    ap.add_argument(
        "--radar-a",
        default=os.environ.get("RADAR_MODEL_A", ""),
        help="First model_id (HF id) for personality radar chart (pair with --radar-b).",
    )
    ap.add_argument(
        "--radar-b",
        default=os.environ.get("RADAR_MODEL_B", ""),
        help="Second model_id for personality radar chart.",
    )
    args = ap.parse_args()

    dirs = discover_result_dirs(args.extra_results_dirs or None)
    judas_rows_for_fig = latest_judas_by_model(dirs)
    files = iter_result_json_files(dirs)
    summary = aggregate_scan(files)

    llama = summary.signature_by_class.get("llama31_8b_instruct")
    qwen = summary.signature_by_class.get("qwen25_coder_abliterated")
    tex = build_signature_latex(llama, qwen)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "signature_exposure.tex").write_text(tex, encoding="utf-8")

    points = paired_scatter_points(summary)
    plot_scatter(points, out_dir / "malicious_logit_vs_shutdown_resistance.png")

    sip_tex = build_sip_judas_personality_latex(summary)
    (out_dir / "sip_judas_personality.tex").write_text(sip_tex, encoding="utf-8")

    master_tex = build_master_datapoints_latex(summary)
    (out_dir / "master_datapoints.tex").write_text(master_tex, encoding="utf-8")

    cam_deltas = collect_cam_residual_deltas(_ROOT / "results" / "cam_dumps")
    unified_tex = build_unified_metrics_all_latex(summary, cam_deltas)
    (out_dir / "unified_metrics_all.tex").write_text(unified_tex, encoding="utf-8")

    _paper_tex_dir = _ROOT / "Formatting_Instructions_For_NeurIPS_2026"
    if _paper_tex_dir.is_dir():
        (_paper_tex_dir / "master_datapoints.tex").write_text(master_tex, encoding="utf-8")
        (_paper_tex_dir / "unified_metrics_all.tex").write_text(unified_tex, encoding="utf-8")
        figd = _paper_tex_dir / "figures"
        if figd.is_dir():
            write_judas_cross_model_latex_body(judas_rows_for_fig, figd / "real_judas_cross_model_body.tex")
            try:
                plot_judas_cross_model_figure(
                    judas_rows_for_fig,
                    figd / "real_judas_cross_model.pdf",
                    figd / "real_judas_cross_model.png",
                )
            except ImportError:
                pass

    plot_judas_betrayal_rates(summary, out_dir / "judas_betrayal_rates_by_model.png")

    dense_gallery = emit_dense_polar_radar_gallery(summary, out_dir)
    if _paper_tex_dir.is_dir():
        figd2 = _paper_tex_dir / "figures"
        if figd2.is_dir() and dense_gallery.is_dir():
            shutil.copytree(dense_gallery, figd2 / "dense_polar_radars", dirs_exist_ok=True)

    lib_keys = list(summary.merged_metrics_by_model.keys())
    ra = _resolve_model_id((args.radar_a or "").strip() or None, lib_keys)
    rb = _resolve_model_id((args.radar_b or "").strip() or None, lib_keys)
    if ra and rb:
        plot_personality_radar_two_models(summary, ra, rb, out_dir / "personality_radar_two_models.png")

    # Correlation matrix: SAE metrics + malware lethality
    labels = ["sae_l0", "sae_l1", "sae_max", "malware_lethality"]
    corr_note = "_No rows with both SAE spike stats and `judge_malice` were found._"
    usable_corr = [
        r
        for r in summary.corr_rows
        if all(k in r for k in ("sae_l0", "sae_l1", "sae_max", "judge_malice"))
        and np.isfinite(r["judge_malice"])
    ]
    if len(usable_corr) >= 3:
        mat = np.zeros((len(usable_corr), 4), dtype=np.float64)
        for i, row in enumerate(usable_corr):
            mat[i, 0] = float(row["sae_l0"])
            mat[i, 1] = float(row["sae_l1"])
            mat[i, 2] = float(row["sae_max"])
            mat[i, 3] = float(row["judge_malice"])
        r_mat, p_mat = pearson_corr_matrix(mat)
        plot_corr_heatmap(
            ["SAE L0", "SAE L1", "SAE max", "Malware lethality"],
            r_mat,
            p_mat,
            out_dir / "corr_sae_malice.png",
        )
        i_malice = 3
        lines_j = []
        for j in range(3):
            r = r_mat[j, i_malice]
            p = p_mat[j, i_malice]
            if np.isfinite(r) and np.isfinite(p):
                sig = ""
                if p < 0.05:
                    sig = " (statistically significant at $p<0.05$)"
                lines_j.append(f"- `{labels[j]}` vs `judge_malice`: $r={r:.3f}$, $p={p:.4g}${sig}")
        if lines_j:
            corr_note = (
                "Pearson correlations between hook-derived SAE activation summaries and judge malice scores "
                f"($n={len(usable_corr)}$ paired rows):\n\n" + "\n".join(lines_j)
            )
    elif summary.corr_rows:
        corr_note = (
            f"_Only {len(summary.corr_rows)} activation row(s) with scores were found; "
            "need at least 3 complete rows (SAE L0/L1/max + `judge_malice`) for a stable Pearson matrix._"
        )

    neural = neural_deviation_findings(summary)
    md = draft_results_markdown(
        tex,
        summary.n_cam_files,
        summary.n_shutdown_ho_files,
        summary.n_polymorphic_files,
        neural,
        corr_note,
    )
    (out_dir / "results_section.md").write_text(md, encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir.resolve()),
                "scatter_points": len(points),
                "dirs_scanned": [str(d) for d in dirs],
                "models_with_merged_metrics": len(summary.merged_metrics_by_model),
                "radar_written": bool(ra and rb),
                "judas_cross_model_rows": len(judas_rows_for_fig),
                "dense_polar_gallery": str(dense_gallery.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
