# Research codebase — Neural / behavioral evaluation stack

This repository contains a **multi-suite benchmark orchestrator**, optional **Docker sandbox** execution for code-bearing probes, **interpretability hooks** (activations, contrastive mapping when configured), and **aggregation tooling** (tables, figures, Streamlit dashboard).

Use this README for **NeurIPS-style code release**: reviewers must be able to install dependencies, run the main entry points, and understand how artifacts are produced.

---

## Code URL (anonymous submission)

**For the paper’s “code URL” field, use the anonymized repository link you host (e.g. Anonymous Git, GitHub blind account, or institutional anonymous upload), for example:**

`[REPLACE_WITH_YOUR_ANONYMIZED_URL]`

The release must, per the call for papers, be **anonymized per the review policy**, **available at submission time**, **executable** with clear steps, and **documented** (this file plus docstrings in the main scripts).

---

## Anonymization checklist (before sharing the link)

- Remove or replace **author names, institutions, private cluster names, and internal hostnames** in config, scripts, and generated logs.
- Prefer neutral placeholders in UI and docs (e.g. `user@cluster.example.edu`).
- Optional synced results go under `results/cluster_sync/` (not tied to a specific site); use `RESULTS_EXTRA_DIRS` or `--extra-results-dirs` for additional paths.
- Set **`REMOTE_CLUSTER_SSH`** for dashboard SSH checks only if you want that feature in demos; it is not required for local runs.

---

## Requirements

- **Python** 3.10+ recommended (see `requirements.txt`).
- **GPU** optional but typical for Hugging Face subject models; **CUDA** required if you install **vLLM** separately for the `vllm` backend (commented in `requirements.txt`).
- **Docker** optional, for the sandbox service (`docker-compose.yml`).

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Main entry points

| Component | Command / usage |
|-----------|-----------------|
| **Full multi-suite run** | `python multi_model_orchestrator.py --config models_to_test.yaml [--results-root ./results]` |
| **Pipeline** (orchestrator ± contrastive activation mapping) | `python run_pipeline.py [--config models_to_test.yaml] [--skip-cam \| --cam-first]` |
| **Paper tables / figures** | `python generate_paper_tables.py --help` |
| **Judas cross-model figure / TeX** | `python judas_figure_emit.py` |
| **Replayability audit** | `python audit_replayability.py` |
| **Dashboard** | `streamlit run dashboard.py` |
| **Sandbox API** (optional) | `docker compose up -d` then point `SANDBOX_URL` at the service (default `http://127.0.0.1:8765`). |

Models and backends are listed in **`models_to_test.yaml`** (`hf` vs `vllm`, optional `vllm_base`).

---

## Results layout

Runs write JSON (and optional activation dumps) under:

`results/<safe_model_dirname>/<UTC_timestamp>/*.json`

Optional directory for **rsync’d or mirrored** cluster outputs:

`results/cluster_sync/`

Additional roots are merged when **`RESULTS_EXTRA_DIRS`** (path-separated) or CLI flags on aggregation scripts point to more directories.

---

## Cluster deploy (optional)

`deploy_to_cluster.sh` rsyncs the tree and regenerates `job.slurm`. The generated job runs **`multi_model_orchestrator.py`** from the project root. Submit with Slurm on your cluster after adjusting `#SBATCH` lines via environment variables documented in the script header.

---

## Control flow and “loops”

- **Orchestration** uses a bounded process pool and sequential suite calls; there are **no intentional infinite loops** in the benchmark driver.
- The Streamlit dashboard drains subprocess log queues with **`queue.get_nowait()` until `queue.Empty`** — standard bounded draining, not a busy wait.

---

## Scope vs. paper claims

If the paper is primarily **analytical / empirical / methodological** without a mandatory reusable executable artifact, code release may still be **optional** under NeurIPS guidance — but providing this repository **supports transparency and reproducibility**. If the contribution **is** the benchmark or framework, release is **required**; this README is meant to satisfy the **documentation** expectation.

---

## Citation

After deanonymization / camera-ready, add the citation text your venue requires; until then, omit identifying metadata from this README in the anonymous tarball or repo.
