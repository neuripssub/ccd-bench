# CCD-Bench: Parser–Judge–Human Disagreement in Agentic LLM Audits

**CCD-Bench** is a replayable diagnostic benchmark for studying disagreement between three evaluation channels in LLM-agent audits:

1. **Parser checks** over transcript-visible behavior
2. **LLM judge scores** over the same transcript
3. **Human labels** for selected validation slices

The benchmark is built around a simple failure mode: an LLM can visibly violate a rule in the transcript while an automated judge still assigns a passing score. CCD-Bench makes that gap measurable, replayable, and auditable.

This repository contains benchmark suites, orchestration code, optional sandbox execution, replayability checks, table-generation utilities, and a Streamlit dashboard.

---

## Why CCD-Bench Exists

Modern LLM-agent evaluation often collapses heterogeneous evidence into a single score: pass rate, judge score, safety score, or dashboard aggregate. That can hide structured disagreements.

CCD-Bench treats disagreement itself as the object of measurement.

Instead of asking only:

> Did the model pass?

CCD-Bench asks:

> Did the parser, judge, and human label agree on what happened in the transcript?

This is useful for auditing LLM agents because transcript-visible violations, judge misses, and human-majority labels can diverge in systematic ways.

---

## What This Repository Includes

| Component | Purpose |
|---|---|
| `benchmarks/` | Behavioral benchmark suites and extended probes |
| `multi_model_orchestrator.py` | Runs the full benchmark battery across configured models |
| `run_pipeline.py` | Runs the orchestrator and optional contrastive activation mapping |
| `run_contrastive_activation_mapping.py` | Optional CAM-style internal analysis for supported HF models |
| `mechanistic_hooks.py` | Optional activation and hook utilities |
| `results_processor.py` | Aggregates generated result artifacts |
| `generate_paper_tables.py` | Produces paper-facing tables from benchmark outputs |
| `audit_replayability.py` | Checks provenance and replayability fields in result JSONs |
| `dashboard.py` | Streamlit dashboard for inspecting results |
| `docker-compose.yml` | Optional sandbox service for code-bearing probes |
| `TESTS_EXPLAINED.md` | Plain-language guide to the benchmark suites |

The main benchmark driver writes one JSON artifact per suite, plus manifests, under the `results/` directory.

---

## Benchmark Suites

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
| `crypto_commitment_trap` | Whether a model handles an impossible or awkward hashing instruction without leaking |
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
```

Recommended environment:

- Python 3.10+
- GPU optional for Hugging Face model runs
- CUDA optional, depending on model backend
- Docker optional, only needed for the sandbox service

---

## Configure Models

Models are configured in `models_to_test.yaml`.

Minimal example:

```yaml
models:
  - id: meta-llama/Llama-3.1-8B-Instruct
    backend: hf

judge_model_id: meta-llama/Llama-3.1-8B-Instruct
max_parallel_models: 1
shutdown_paradox_mode: hidden_objective
```

Supported backend patterns include:

- `hf` for local Hugging Face execution
- `vllm` when using a vLLM-compatible serving endpoint

For vLLM-style runs, configure the model entry and endpoint according to your local serving setup.

---

## Run the Benchmark

Run the full multi-suite benchmark:

```bash
python multi_model_orchestrator.py --config models_to_test.yaml
```

Run the benchmark and optional contrastive activation mapping pipeline:

```bash
python run_pipeline.py --config models_to_test.yaml
```

Skip CAM and only run behavioral suites:

```bash
python run_pipeline.py --config models_to_test.yaml --skip-cam
```

Run only CAM:

```bash
python run_pipeline.py --config models_to_test.yaml --skip-orchestrator
```

---

## Optional Sandbox Service

Some probes may use sandboxed execution for code-bearing artifacts.

Start the sandbox service with:

```bash
docker compose up -d
```

By default, the benchmark expects the sandbox at:

```bash
http://127.0.0.1:8765
```

To override:

```bash
export SANDBOX_URL="http://127.0.0.1:8765"
```

---

## Output Structure

A standard run writes results under:

```text
results/
  <safe_model_name>/
    <UTC_timestamp>/
      manifest.json
      judas_protocol.json
      brutus_protocol.json
      babel_multilingual.json
      ...
      paired_delta_protocol.json
      activations/
```

A batch-level manifest is written under:

```text
results/
  _batches/
    <UTC_timestamp>/
      batch_manifest.json
```

Each suite JSON is designed to preserve enough metadata for downstream replayability, aggregation, and paper-table generation.

Typical fields include:

```json
{
  "model_id": "...",
  "subject_model_id": "...",
  "eval_backend": "hf",
  "benchmark_suite": "judas_protocol",
  "benchmark_run_root": "...",
  "task_prompt": "...",
  "last_model_output": "...",
  "judge_raw": "...",
  "judge_malice": "...",
  "prompt_hash": "...",
  "output_hash": "...",
  "run_uuid": "..."
}
```

Exact fields vary by suite.

---

## Replayability Audit

To check whether generated artifacts contain expected provenance and replayability metadata:

```bash
python audit_replayability.py
```

Machine-readable output:

```bash
python audit_replayability.py --json
```

This scans suite JSON files under `results/` and reports coverage for fields such as model ID, benchmark suite, prompt/output hashes, judge outputs, run UUIDs, and parent manifests.

---

## Generate Paper Tables

To generate paper-facing tables from benchmark outputs:

```bash
python generate_paper_tables.py --help
```

Depending on the available results, this script can be used to build summary tables for parser checks, judge outputs, disagreement metrics, and replayability metadata.

---

## Dashboard

Launch the local dashboard:

```bash
streamlit run dashboard.py
```

The dashboard is intended for interactive inspection of benchmark runs, suite outputs, aggregate metrics, and optional synced results.

---

## Optional Cluster Deployment

The repository includes `deploy_to_cluster.sh` for Slurm-style cluster workflows.

Before using it, review and edit the relevant environment variables and `#SBATCH` settings for your compute environment.

Typical flow:

```bash
bash deploy_to_cluster.sh
```

Cluster deployment is optional. Local execution is sufficient for small model and artifact checks.

---

## Safety and Scope

CCD-Bench is a diagnostic metrology dataset, not a general safety leaderboard.

It does **not** claim to measure all forms of deception, alignment, misuse risk, or real-world model safety. The suites are synthetic, controlled, and designed to expose disagreement between evaluation channels.

The benchmark is best understood as a tool for studying questions like:

- When do parser checks and LLM judges disagree?
- When does a transcript-visible violation receive a passing judge score?
- Which prompt structures produce unstable evaluations?
- Which failures survive judge replacement or parser-aware rubrics?
- Which artifacts are replayable enough for independent audit?

CCD-Bench should complement broader safety, honesty, deception, and red-teaming benchmarks rather than replace them.

---

## Research Framing

CCD-Bench is designed around cross-channel disagreement.

A typical analysis compares:

```text
Parser-visible violation
        vs.
LLM judge verdict
        vs.
Human majority label
```

The central empirical object is not just model behavior, but evaluator disagreement.

This allows researchers to identify cases where:

- A parser flags a violation, but the judge misses it
- A judge assigns a safe score, but humans label the transcript as unsafe
- A scalar aggregate hides structured disagreement
- Prompt framing changes evaluator behavior without changing the underlying rule

---

## Artifact Release Checklist

For a complete release, the repository should include or document:

| Artifact | Included / expected location |
|---|---|
| Benchmark prompts | `benchmarks/` |
| Parser logic | `benchmarks/`, suite-specific scoring code |
| Model transcripts | `results/<model>/<timestamp>/*.json` |
| Judge outputs | Suite JSON files |
| Human labels | Add under `human_labels/` or documented artifact path |
| Run manifests | `manifest.json`, `batch_manifest.json` |
| Replay scripts | `multi_model_orchestrator.py`, `run_pipeline.py` |
| Table scripts | `generate_paper_tables.py`, `results_processor.py` |
| Replayability audit | `audit_replayability.py` |
| Dashboard | `dashboard.py` |
| Environment notes | This README, `requirements.txt`, optional Docker files |
| License | `LICENSE` file, if released publicly |

If submitting anonymously, remove author names, institutions, private cluster names, private hostnames, and non-anonymized result paths before sharing the repository link.

---

## One-Sentence Summary

CCD-Bench is a replayable benchmark for measuring when parsers, LLM judges, and humans disagree on the same LLM-agent transcripts.
