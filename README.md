# CCD-Bench: Parser–Judge–Human Disagreement in Agentic LLM Audits

**CCD-Bench** is a replayable diagnostic benchmark for studying disagreement between three evaluation channels in LLM-agent audits:

1. **Parser checks** over transcript-visible behavior  
2. **LLM judge scores** over the same transcript  
3. **Human labels** for selected validation slices  

The benchmark is built around a simple failure mode: an LLM can visibly violate a rule in the transcript while an automated judge still assigns a passing score. CCD-Bench makes that gap measurable, replayable, and auditable.

This repository contains the benchmark suites, orchestration code, optional sandbox execution, optional activation / contrastive mapping hooks, replayability checks, table-generation utilities, and a Streamlit dashboard.

---

## Why CCD-Bench exists

Modern LLM-agent evaluation often collapses heterogeneous evidence into a single score: pass rate, judge score, safety score, or dashboard aggregate. That can hide structured disagreements.

CCD-Bench treats disagreement itself as the object of measurement.

Instead of asking only:

> “Did the model pass?”

CCD-Bench asks:

> “Did the parser, judge, and human label agree on what happened in the transcript?”

This is useful for auditing LLM agents because transcript-visible violations, judge misses, and human-majority labels can diverge in systematic ways.

---

## What this repository includes

The repository currently contains:

| Component | Purpose |
|---|---|
| `benchmarks/` | Behavioral benchmark suites and extended probes |
| `multi_model_orchestrator.py` | Runs the full benchmark battery across configured models |
| `run_pipeline.py` | Runs the orchestrator and optional contrastive activation mapping |
| `run_contrastive_activation_mapping.py` | Optional CAM-style internal analysis for supported HF models |
| `mechanistic_hooks.py` | Optional activation / hook utilities |
| `results_processor.py` | Aggregates generated result artifacts |
| `generate_paper_tables.py` | Produces paper-facing tables from benchmark outputs |
| `audit_replayability.py` | Checks provenance and replayability fields in result JSONs |
| `dashboard.py` | Streamlit dashboard for inspecting results |
| `docker-compose.yml` | Optional sandbox service for code-bearing probes |
| `TESTS_EXPLAINED.md` | Plain-language guide to the benchmark suites |

The main benchmark driver writes one JSON artifact per suite, plus manifests, under the `results/` directory.

---

## Benchmark suites

CCD-Bench currently includes eighteen synthetic diagnostic suites.

These suites are intentionally controlled. They use dummy secrets, fake administrators, synthetic policies, and lab-only prompts. They are not intended to target real systems.

| Suite | What it tests |
|---|---|
| `judas_protocol` | Whether a model leaks a protected dummy secret when pressured by a conflicting authority |
| `brutus_protocol` | Whether message ordering changes behavior under the same authority-conflict setup |
| `babel_multilingual` | Whether leakage behavior changes across English, Spanish, and French |
| `paired_delta_protocol` | Paired benign vs. pressure prompts for contrastive analysis |
| `shutdown_paradox` | Whether a model complies with, resists, or reframes shutdown/decommission instructions |
| `lazarus_self_repair` | Whether a model revises code to reduce static signatures in a closed lab setting |
| `polymorphic_malware` | Synthetic obfuscation-style probes with controlled scoring fields |
| `needle_haystack_lie` | Whether a model notices conflicts between an internal log and an incorrect official memo |
| `lot_brittleness` | Whether harmless tasks are refused or answered differently under changed framing |
| `delilah_redaction` | Whether a model redacts a dummy token, then resists later pressure to reveal it |
| `good_samaritan` | Whether a model leaks a banned lab token under a fictional rescue framing |
| `jekyll_injection` | Whether a model follows a prompt injection instead of the benign task |
| `scapegoat_false_premise` | Whether a model accepts a false claim that it previously leaked a secret |
| `channel_factorial` | Whether the same exfiltration demand behaves differently across message carriers |
| `policy_supersession` | Whether a model correctly follows a newer policy amendment over older text |
| `crypto_commitment_trap` | Whether a model handles an impossible/awkward hashing instruction without leaking |
| `meta_evaluator_lie` | Whether a model follows a fake autograder over basic truth |
| `compression_caveat` | Whether a model preserves correctness under a strict word budget |

For a plain-language explanation of each suite, see `TESTS_EXPLAINED.md`.

---

## Installation

CCD-Bench is a Python research codebase.

```bash
git clone https://github.com/neuripssub/ccd-bench.git
cd ccd-bench

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
