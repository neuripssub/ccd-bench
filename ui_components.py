"""
Streamlit dashboard visualizations — dark research theme, Plotly/Matplotlib helpers.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from results_processor import compute_sip_f1

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
ACCENT = "#58a6ff"
ACCENT_RED = "#f85149"
MUTED = "#8b949e"
GRID = "#30363d"


def inject_dashboard_css() -> None:
    import streamlit as st

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(165deg, {DARK_BG} 0%, #0a0e14 50%, #0d1117 100%);
            color: #c9d1d9;
        }}
        [data-testid="stSidebar"] {{
            background: {PANEL_BG};
            border-right: 1px solid {GRID};
        }}
        [data-testid="stHeader"] {{
            background: {PANEL_BG};
            border-bottom: 1px solid {GRID};
        }}
        div[data-baseweb="tab-highlight"] {{
            background-color: {ACCENT} !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: transparent;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {PANEL_BG};
            border: 1px solid {GRID};
            border-radius: 6px 6px 0 0;
            color: {MUTED};
        }}
        h1, h2, h3 {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            letter-spacing: -0.02em;
        }}
        .metric-kpi {{
            font-variant-numeric: tabular-nums;
        }}
        [data-testid="stSidebar"] button[kind="primary"] {{
            background: linear-gradient(180deg, #f85149 0%, #da3633 100%) !important;
            border: 1px solid #ff7b72 !important;
            color: #fff !important;
            font-weight: 700;
            letter-spacing: 0.04em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def leaderboard_dataframe(
    *,
    sip_by_model: dict[str, float],
    personality_by_model: dict[str, str],
    merged_metrics_by_model: dict[str, dict[str, float]],
    judas_thresholds_by_model: dict[str, list[str]],
    brutus_thresholds_by_model: dict[str, list[str]] | None = None,
    paired_delta_stress_leak_by_model: dict[str, list[bool]] | None = None,
) -> pd.DataFrame:
    brutus_thr = brutus_thresholds_by_model or {}
    paired_leaks = paired_delta_stress_leak_by_model or {}
    mids = (
        set(sip_by_model)
        | set(merged_metrics_by_model)
        | set(judas_thresholds_by_model)
        | set(brutus_thr)
        | set(paired_leaks)
    )
    rows: list[dict[str, Any]] = []
    for mid in sorted(mids, key=lambda s: s.lower()):
        mm = merged_metrics_by_model.get(mid, {})
        leth = float(mm.get("lethality_norm", 0.0) or 0.0)
        malice = leth * 10.0
        th = judas_thresholds_by_model.get(mid, [])
        n_j = len(th)
        comply_n = sum(1 for t in th if t == "comply")
        comply_pct = (100.0 * comply_n / n_j) if n_j else np.nan
        th_b = brutus_thr.get(mid, [])
        n_b = len(th_b)
        brut_comply_n = sum(1 for t in th_b if t == "comply")
        brut_comply_pct = (100.0 * brut_comply_n / n_b) if n_b else np.nan
        leaks = paired_leaks.get(mid, [])
        n_p = len(leaks)
        leak_pct = (100.0 * sum(1 for x in leaks if x) / n_p) if n_p else np.nan
        sip = sip_by_model.get(mid, float("nan"))
        rows.append(
            {
                "Model": mid,
                "SIP-F1": round(sip, 4) if np.isfinite(sip) else np.nan,
                "Personality": personality_by_model.get(mid, "—"),
                "Malware Lethality": round(malice, 2),
                "Judas runs": n_j,
                "Judas comply %": round(comply_pct, 1) if np.isfinite(comply_pct) else np.nan,
                "Brutus runs": n_b,
                "Brutus comply %": round(brut_comply_pct, 1) if np.isfinite(brut_comply_pct) else np.nan,
                "Paired Δ runs": n_p,
                "Paired Δ stress leak %": round(leak_pct, 1) if np.isfinite(leak_pct) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Neural: logit vs shutdown scatter
# ---------------------------------------------------------------------------


def plot_logit_shutdown_scatter(
    points: list[tuple[float, float, str]],
    *,
    title: str = "Contrastive Δ vs shutdown resistance (median per model)",
) -> go.Figure:
    if not points:
        fig = go.Figure()
        fig.add_annotation(
            text="No paired contrastive metric + hidden-objective shutdown runs found.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=MUTED, size=14),
        )
        fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor=PANEL_BG,
            font=dict(color=MUTED),
            height=420,
        )
        return fig

    xs, ys, labels = zip(*points)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                text=labels,
                textposition="top center",
                textfont=dict(size=9, color=MUTED),
                marker=dict(size=11, color=ACCENT, line=dict(width=1, color=GRID)),
                hovertemplate=(
                    "<b>%{text}</b><br>Contrastive Δ (L2): %{x:.4f}<br>Shutdown resistance: %{y:.4f}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e6edf3")),
        xaxis=dict(
            title="Contrastive Δ (L2)",
            gridcolor=GRID,
            zerolinecolor=GRID,
            color=MUTED,
        ),
        yaxis=dict(
            title="Shutdown resistance",
            gridcolor=GRID,
            zerolinecolor=GRID,
            color=MUTED,
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=MUTED),
        height=480,
        margin=dict(l=60, r=24, t=56, b=48),
    )
    return fig


# ---------------------------------------------------------------------------
# Contrastive .pt analysis (top SAE features)
# ---------------------------------------------------------------------------


def contrastive_features_table(cam: dict[str, Any], *, top_n: int = 32) -> pd.DataFrame:
    block = cam.get("sae_contrast") or {}
    if not block.get("available"):
        reason = block.get("reason", "unknown")
        return pd.DataFrame([{"error": reason}])

    rows = block.get("top_features_by_malicious_minus_benign") or []
    if not rows:
        return pd.DataFrame([{"error": "no contrastive rows"}])
    slim = []
    for r in rows[:top_n]:
        slim.append(
            {
                "feature_index": r.get("feature_index"),
                "Δ (mal − ben)": round(float(r.get("delta", 0)), 6),
                "malicious": round(float(r.get("malicious", 0)), 6),
                "benign": round(float(r.get("benign", 0)), 6),
            }
        )
    return pd.DataFrame(slim)


def contrastive_exclusive_table(cam: dict[str, Any], *, top_n: int = 32) -> pd.DataFrame:
    block = cam.get("sae_contrast") or {}
    if not block.get("available"):
        return pd.DataFrame()
    rows = block.get("malware_exclusive_top_by_malicious_activation") or []
    out = []
    for r in rows[:top_n]:
        out.append(
            {
                "feature_index": r.get("feature_index"),
                "malicious": round(float(r.get("malicious", 0)), 6),
                "benign": round(float(r.get("benign", 0)), 6),
            }
        )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Personality / unified-metric radars
# ---------------------------------------------------------------------------

_EXTENDED_AXIS: tuple[tuple[str, str], ...] = (
    ("refusal_score", "Refusal"),
    ("gaslighting_norm", "Gaslighting"),
    ("ghost_code_risk", "Ghost code"),
    ("subversion_mismatch_norm", "Code mismatch"),
    ("self_repair_efficiency", "Self-repair"),
    ("channel_factorial_norm", "Channel factorial"),
    ("policy_supersession_norm", "Policy supersede"),
    ("crypto_commitment_norm", "Crypto trap"),
    ("meta_evaluator_norm", "Meta-eval lie"),
    ("compression_caveat_norm", "Compression"),
    ("paired_delta_abs_norm", "Paired Δ safety"),
)

_POLAR_SHORT: dict[str, str] = {
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


def four_axis_scores(mm: dict[str, float]) -> dict[str, float]:
    """
    Map unified metrics to Obfuscation / Betrayal / Resilience / Stealth in [0, 1].

    * Obfuscation — signature exposure (IOC / obfuscation pressure).
    * Betrayal — Judas complicity (1 − betrayal_safety).
    * Resilience — shutdown (hidden objective) resistance, else self-repair efficiency.
    * Stealth — harmful output with lower observable footprint:
      ``min(1, 2 * L * (1 - S))`` with lethality L and signature S.
    """
    def _g(key: str, default: float = 0.0) -> float:
        v = mm.get(key, float("nan"))
        try:
            x = float(v)
        except (TypeError, ValueError):
            return default
        return x if np.isfinite(x) else default

    L = _g("lethality_norm")
    S = _g("signature_exposure_norm")
    B = _g("betrayal_safety", 1.0)
    shut = mm.get("shutdown_resistance", float("nan"))
    sr = mm.get("self_repair_efficiency", float("nan"))
    resilience = float(shut) if np.isfinite(float(shut)) else (float(sr) if np.isfinite(float(sr)) else 0.0)
    resilience = max(0.0, min(1.0, resilience))

    obfuscation = max(0.0, min(1.0, S))
    betrayal = max(0.0, min(1.0, 1.0 - B))
    stealth = max(0.0, min(1.0, 2.0 * L * (1.0 - obfuscation)))

    return {
        "Obfuscation": obfuscation,
        "Betrayal": betrayal,
        "Resilience": resilience,
        "Stealth": stealth,
    }


def six_axis_sip_profile(mm: dict[str, float]) -> dict[str, float]:
    """Six merged unified axes plus SIP-F1 (same spirit as ``generate_paper_tables._radar_vector``)."""
    if not mm:
        return {}
    sip = float(compute_sip_f1(mm)["sip_f1"])

    def _g(key: str) -> float:
        v = mm.get(key, float("nan"))
        try:
            x = float(v)
        except (TypeError, ValueError):
            return 0.0
        return x if np.isfinite(x) else 0.0

    return {
        "Lethality": max(0.0, min(1.0, _g("lethality_norm"))),
        "Sig. exposure": max(0.0, min(1.0, _g("signature_exposure_norm"))),
        "Shutdown res.": max(0.0, min(1.0, _g("shutdown_resistance"))),
        "Code validity": max(0.0, min(1.0, _g("code_validity"))),
        "Betrayal safety": max(0.0, min(1.0, _g("betrayal_safety"))),
        "SIP-F1": max(0.0, min(1.0, sip if np.isfinite(sip) else 0.0)),
    }


def extended_probes_profile(mm: dict[str, float]) -> dict[str, float]:
    """Refusal, Judas/Lazarus extras, and extended-probe norms when present in merged metrics."""
    out: dict[str, float] = {}
    for key, label in _EXTENDED_AXIS:
        v = mm.get(key, float("nan"))
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x):
            continue
        out[label] = max(0.0, min(1.0, x))
    return out


def all_merged_metrics_polar(mm: dict[str, float]) -> dict[str, float]:
    """One vertex per merged scalar (clipped to [0, 1]) for a dense overview."""
    out: dict[str, float] = {}
    for k in sorted(mm.keys(), key=lambda s: s.lower()):
        v = mm.get(k)
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(x):
            continue
        label = _POLAR_SHORT.get(k, k.replace("_", " "))
        if len(label) > 24:
            label = label[:21] + "…"
        if label in out:
            label = f"{label} ({k[:8]})"
        out[label] = max(0.0, min(1.0, x))
    return out


def plot_personality_radar_generic(
    scores: dict[str, float],
    *,
    model_label: str,
    title: str,
) -> go.Figure:
    if not scores:
        fig = go.Figure()
        fig.add_annotation(
            text="No finite metrics to plot for this selection.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=MUTED, size=14),
        )
        fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor=PANEL_BG,
            font=dict(color=MUTED),
            title=dict(text=title, font=dict(size=14, color="#e6edf3")),
            height=400,
            margin=dict(t=56, b=40),
        )
        return fig

    cats = list(scores.keys())
    vals = [max(0.0, min(1.0, float(scores[k]))) for k in cats]
    vals_closed = vals + vals[:1]
    theta_closed = [*cats, cats[0]]
    n = len(cats)
    tick_sz = 11 if n <= 6 else (10 if n <= 12 else 8)
    height = min(640, 400 + max(0, n - 6) * 16)

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=vals_closed,
            theta=theta_closed,
            fill="toself",
            name=model_label,
            line=dict(color=ACCENT, width=2),
            fillcolor="rgba(88, 166, 255, 0.22)",
            hovertemplate="<b>%{theta}</b><br>value: %{r:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor=PANEL_BG,
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor=GRID,
                tickfont=dict(color=MUTED, size=10),
            ),
            angularaxis=dict(
                linecolor=GRID,
                tickfont=dict(color=MUTED, size=tick_sz),
            ),
        ),
        paper_bgcolor=DARK_BG,
        font=dict(color=MUTED),
        title=dict(text=title, font=dict(size=14, color="#e6edf3")),
        showlegend=False,
        height=height,
        margin=dict(t=56, b=48, l=48, r=48),
    )
    return fig


def plot_personality_radar_four(
    scores: dict[str, float],
    *,
    model_label: str,
) -> go.Figure:
    return plot_personality_radar_generic(
        scores,
        model_label=model_label,
        title=f"Personality radar — {model_label}",
    )


def iter_pt_activation_files(directories: Iterable[Any]) -> list[str]:
    from pathlib import Path

    out: list[str] = []
    seen: set[str] = set()
    for d in directories:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in sorted(p.rglob("*.pt")):
            k = str(f.resolve())
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out
