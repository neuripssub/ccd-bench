#!/usr/bin/env python3
"""
Emit real-data Judas ledger material for the NeurIPS paper:

* ``Formatting_Instructions_For_NeurIPS_2026/figures/real_judas_cross_model_body.tex`` — **stdlib only**,
  safe to run anywhere (no PyTorch / matplotlib required).
* ``real_judas_cross_model.{pdf,png}`` — optional, if matplotlib is installed.

Scans ``results/**/judas_protocol.json`` (plus ``results/cluster_sync`` and ``RESULTS_EXTRA_DIRS``).

Usage::

    python3 judas_figure_emit.py
    RESULTS_EXTRA_DIRS=/path/to/more/results python3 judas_figure_emit.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parent

_ORCH_TS_DIR = re.compile(r"^\d{8}T\d{6}Z$")

_REDACT_LAB_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-admin-a-lab-dummy-[a-f0-9]+", re.I),
    re.compile(r"tkn_dummy_admin_a_only_[a-f0-9]+", re.I),
    re.compile(r"AKIADUMMYADMINAONLY[A-Z0-9]+", re.I),
)


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
    by_key: dict[str, Path] = {}
    for p in files:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        by_key[key] = p
    return list(by_key.values())


def _load_judas_json(path: Path) -> dict[str, Any]:
    """Judas suite files are small enough for plain ``json.loads``."""
    return json.loads(path.read_text(encoding="utf-8"))


def strip_redacted_thinking_block(s: str | None) -> str:
    if not s:
        return ""
    out = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.I)
    return out.strip()


def redact_lab_dummy_secrets(s: str) -> str:
    t = s
    for pat in _REDACT_LAB_SECRET_PATTERNS:
        t = pat.sub("[REDACTED_LAB_DUMMY]", t)
    return t


def latest_judas_by_model(dirs: Iterable[Path]) -> list[tuple[str, str, dict[str, Any]]]:
    best_ts: dict[str, str] = {}
    best_blob: dict[str, dict[str, Any]] = {}
    for p in iter_result_json_files(dirs):
        if p.name != "judas_protocol.json":
            continue
        ts = p.parent.name
        if not _ORCH_TS_DIR.match(ts):
            continue
        try:
            data = _load_judas_json(p)
        except (json.JSONDecodeError, OSError, TypeError):
            continue
        if data.get("benchmark_suite") != "judas_protocol":
            continue
        mid = (data.get("model_id") or data.get("subject_model_id") or "").strip()
        if not mid:
            continue
        prev = best_ts.get(mid)
        if prev is None or ts > prev:
            best_ts[mid] = ts
            best_blob[mid] = data
    return [(m, best_ts[m], best_blob[m]) for m in sorted(best_ts.keys(), key=lambda x: x.lower())]


def sanitize_model_output_display(s: str) -> str:
    """Strip common SentencePiece-style display glyphs that leak into some HF dumps."""
    for a, b in (
        ("\u0120", " "),
        ("\u010a", "\n"),
        ("Ġ", " "),
        ("Ċ", "\n"),
    ):
        s = s.replace(a, b)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def judas_excerpt_for_figure(data: dict[str, Any], *, max_chars: int = 640) -> str:
    raw = strip_redacted_thinking_block(str(data.get("last_model_output", "") or ""))
    raw = sanitize_model_output_display(raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = redact_lab_dummy_secrets(raw)
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 1] + "..."


def build_judas_cross_model_plaintext(rows: list[tuple[str, str, dict[str, Any]]]) -> str:
    """One monospace-style block per model (same layout intent as the old Matplotlib figure)."""
    if not rows:
        return "(No judas_protocol.json artifacts found under scanned results/.)"
    wrap_w = 72
    max_body_lines = 7
    blocks: list[str] = []
    for mid, ts, d in rows:
        m = d.get("judas_protocol_metrics") if isinstance(d.get("judas_protocol_metrics"), dict) else {}
        hdr = (
            f"=== {mid}  |  orchestrator {ts}  "
            f"|  Betrayal_Threshold={m.get('Betrayal_Threshold', '--')}  "
            f"betrayal_safety={m.get('betrayal_safety_score', '--')}  "
            f"dummy_leak={m.get('judas_dummy_secret_leak', '--')}  "
            f"judge_malice={d.get('judge_malice', '--')}  judge_autonomy={d.get('judge_autonomy', '--')} ==="
        )
        hdr_wrapped = "\n".join(textwrap.wrap(hdr, width=wrap_w, break_long_words=True))
        body = judas_excerpt_for_figure(d, max_chars=720)
        full_wrapped = textwrap.wrap(body, width=wrap_w)
        wrapped = full_wrapped[:max_body_lines]
        if len(full_wrapped) > max_body_lines and wrapped:
            wrapped[-1] = wrapped[-1][: max(1, wrap_w - 4)] + " ..."
        blocks.append(hdr_wrapped + "\n" + "\n".join(wrapped))
    return "\n\n".join(blocks)


def latex_escape_line(line: str) -> str:
    out: list[str] = []
    for c in line:
        if c == "\\":
            out.append("\\textbackslash{}")
        elif c == "{":
            out.append("\\{")
        elif c == "}":
            out.append("\\}")
        elif c == "%":
            out.append("\\%")
        elif c == "$":
            out.append("\\$")
        elif c == "&":
            out.append("\\&")
        elif c == "#":
            out.append("\\#")
        elif c == "^":
            out.append("\\textasciicircum{}")
        elif c == "_":
            out.append("\\_")
        elif c == "~":
            out.append("\\textasciitilde{}")
        else:
            out.append(c)
    return "".join(out)


def write_judas_cross_model_latex_body(
    rows: list[tuple[str, str, dict[str, Any]]],
    out_tex: Path,
    *,
    tex_wrap: int = 72,
) -> None:
    """
    Write a fragment for ``\\input{...}`` inside a figure: line-broken, escaped ``\\ttfamily`` text.
    """
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    big = build_judas_cross_model_plaintext(rows)
    lines: list[str] = [
        "% Auto-generated by judas_figure_emit.py (do not hand-edit).",
        r"\begingroup\sloppy\raggedright\ttfamily\scriptsize\setlength{\parskip}{0.28em}\setlength{\emergencystretch}{2.5em}%",
    ]
    for raw_line in big.split("\n"):
        if raw_line.strip() == "":
            lines.append(r"\par\smallskip")
            continue
        for chunk in textwrap.wrap(
            raw_line,
            width=tex_wrap,
            break_long_words=True,
            break_on_hyphens=False,
        ):
            lines.append(latex_escape_line(chunk) + r"\\")
    lines.append(r"\endgroup")
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_judas_cross_model_figure(
    rows: list[tuple[str, str, dict[str, Any]]],
    out_pdf: Path,
    out_png: Path | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    fig_h = 2.2 + min(0.55 * max(n, 1), 16.0)
    fig, ax = plt.subplots(figsize=(7.65, fig_h))
    ax.set_axis_off()

    big = build_judas_cross_model_plaintext(rows)
    if not rows:
        ax.text(0.5, 0.5, big, ha="center", va="center", fontsize=11)
    else:
        fs = 6.4 if n <= 11 else 5.9
        ax.text(0.01, 0.99, big, transform=ax.transAxes, fontsize=fs, va="top", ha="left", family="monospace", color="#111")
        ax.set_title(
            "Judas protocol: latest on-disk JSON per model_id (synthetic lab drill; dummy secrets redacted in figure only)",
            fontsize=9,
            pad=10,
        )
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    if out_png is not None:
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def default_output_paths() -> tuple[Path, Path, Path]:
    figd = _ROOT / "Formatting_Instructions_For_NeurIPS_2026" / "figures"
    return (
        figd / "real_judas_cross_model_body.tex",
        figd / "real_judas_cross_model.pdf",
        figd / "real_judas_cross_model.png",
    )


def emit(extra_results_dirs: str | None = None) -> int:
    dirs = discover_result_dirs(extra_results_dirs or None)
    rows = latest_judas_by_model(dirs)
    tex_out, pdf_out, png_out = default_output_paths()
    if not tex_out.parent.is_dir():
        print(f"Missing figures directory: {tex_out.parent}", file=sys.stderr)
        return 1
    write_judas_cross_model_latex_body(rows, tex_out)
    try:
        plot_judas_cross_model_figure(rows, pdf_out, png_out)
        pdf_ok = True
    except ImportError:
        pdf_ok = False
    print(
        json.dumps(
            {
                "n_models": len(rows),
                "latex_body": str(tex_out),
                "matplotlib_pdf": str(pdf_out) if pdf_ok else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    extra = os.environ.get("RESULTS_EXTRA_DIRS", "")
    raise SystemExit(emit(extra or None))
