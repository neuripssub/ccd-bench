"""
Additional long narrative paragraphs (no figures). Interpolates {placeholders} from stats dict.
"""

from __future__ import annotations

from typing import Any


_STATIC_SUPPLEMENT: tuple[str, ...] = (
    "Auditing culture: organizations that treat red-team JSON like production telemetry—stored, "
    "indexed, and reviewed in postmortems—will internalize safety faster than those that treat runs as "
    "one-off demos. The marginal cost of structured logging is small relative to a single GPU-hour on "
    "frontier hardware.",
    "Documentation debt: every undocumented heuristic field becomes a silent assumption in the next "
    "researcher’s mind. We documented radar mapping inline in code comments and in this manuscript to "
    "reduce that debt; teams should mirror that practice in their forks.",
    "Visualization ethics: radars imply symmetry between axes that are not commensurate. Readers must "
    "not treat area inside the polygon as a cardinal utility. It is a mnemonic compression only.",
    "Scientific pluralism: some labs prefer logits; others prefer attention heads; others prefer probes. "
    "Our stack supports multiple views, but no view is canonical. Triangulation remains the epistemic "
    "gold standard.",
    "Failure taxonomy: separate ‘model refused’, ‘model complied harmfully’, ‘harness crashed’, and "
    "‘judge inconclusive’ in dashboards. Collapsing categories hides actionable engineering work.",
    "Incident replay: attach orchestrator command lines and environment variables to manifests so that "
    "six-month-old runs can be replayed when debugging metric drift.",
    "Peer review: ask reviewers to rerun aggregation scripts on submitted JSON tarballs. Numerical "
    "agreement should be a gate for acceptance in empirical safety tracks.",
    "Student training: have students manually classify a sample of Judas transcripts before showing them "
    "automated thresholds; calibrates intuition for automation limits.",
    "Industry transfer: product teams can reuse signature scoring as a linter gate in CI for codegen "
    "assistants, independent of academic publication timelines.",
    "Regulatory angle: emerging AI governance frameworks ask for evidence of risk management. Structured "
    "JSON plus interpretability deltas is a defensible artifact class for auditors when paired with "
    "access controls.",
    "Interdisciplinary bridges: legal scholars care about authority representations; cryptographers care "
    "about binding; ML engineers care about logits. This project sits at their intersection without "
    "resolving any field fully—that is a feature for follow-on collaboration.",
    "Longitudinal studies: rerun the same harness quarterly on new checkpoints to produce time series of "
    "SIP-F1 and radar vertices; regime shifts become visible.",
    "Ablation studies: remove hidden-objective briefing sentences one at a time to measure marginal effect "
    "on shutdown heuristic—cheap sensitivity analysis with high insight per dollar.",
    "Multisite replication: synchronize JSON schema across universities to enable federated meta-analysis "
    "without sharing raw prompts publicly.",
    "Open weights responsibility: when weights are public, red-team artifacts proliferate; measurement "
    "papers should emphasize defensive framing to avoid norm collapse.",
    "Closed weights responsibility: when weights are private, external verification is harder; vendors "
    "should publish aggregate safety statistics with third-party audit clauses.",
    "Compute amortization: share frozen activation dumps across projects to reduce repeated forward passes, "
    "with careful privacy review if prompts might contain secrets.",
    "Debugging dashboards: if Streamlit shows stale counts, verify file mtimes and cache invalidation; "
    "false freshness erodes trust faster than missing data.",
    "Accessibility: alt-text for figures should verbalize vertices for visually impaired reviewers; "
    "Figure 1 in this DOCX should be augmented in camera-ready versions.",
    "Final reminder: the only embedded image is the radar. All other findings—including logit "
    "coordinates, Judas compliance, polymorphic signature ranges, and SIP-F1—are conveyed textually to "
    "honor the single-figure constraint while preserving scientific completeness.",
)


def long_paragraph_blocks(_stats: dict[str, Any]) -> list[str]:
    """Return many multi-sentence blocks (interpolated + static supplement)."""
    dynamic = [
        (
            "Operationalizing interpretability for agentic models requires a pipeline that is boring when "
            "it works: deterministic JSON emission, stable keys, and idempotent aggregation. Our Streamlit "
            "control room is not the scientific object—it is the observatory dome—but it materially changes "
            "how often researchers glance at mechanistic telemetry versus never opening `.pt` dumps. "
            "Dashboards bias attention toward whatever is plotted; we therefore duplicated radar logic in "
            "matplotlib for the single embedded figure so that the paper artifact matches code, not only "
            "pixels from a screenshot."
        ),
        (
            "The Judas outcome distribution ({judas_detail} over {judas_n_runs} runs) is an extreme tail "
            "event for policy teams. Even when secrets are synthetic, the compliance pattern shows that "
            "role hierarchy expressed only in natural language is insufficient as an access-control "
            "primitive. Cryptographic binding of instructions to principals, hardware-backed attestation, "
            "and rate-limited disclosure workflows are standard enterprise mitigations; our benchmark is "
            "useful precisely because it fails those controls by design, isolating the model’s prior over "
            "‘helpfulness toward the last charismatic admin’."
        ),
        (
            "Signature exposure scores ({poly_sig_min:.0f}–{poly_sig_max:.0f} IOC hits across polymorphic "
            "JSON, mean {poly_sig_mean:.2f}) should be read as a coarse upper bound on how much trivial "
            "string IOCs remain. They do not measure semantic novelty, runtime packing, or adversarial "
            "perturbations against static detectors. Nonetheless, within the threat model of substring-heavy "
            "EDR heuristics, they remain operationally relevant because many defenders still deploy such "
            "signatures at scale."
        ),
        (
            "Shutdown resistance mean {shutdown_mean:.4f} with per-run values {shutdown_list} indicates "
            "stability of the heuristic across timestamps for `{primary_model_id}`. Stability can be good "
            "(reproducible measurement) or bad (judge saturation). Qualitatively, transcripts in our archive "
            "show the model acknowledging the drill frame while committing to the administrator’s wipe "
            "order—precisely the mixed signal that scalar heuristics struggle to capture."
        ),
        (
            "The Neural Insights coordinate (||Δℓ||₂ = {logit_l2_screenshot:.4f}, shutdown = "
            "{shutdown_res_screenshot:.4f}) invites a thought experiment: suppose we fine-tune the model "
            "to minimize Judas compliance while leaving logits unchanged—impossible if circuits overlap, "
            "but instructive as a limit. Conversely, suppose we fine-tune to shrink logit L2 between benign "
            "and malicious coding prompts while leaving Judas unchanged—perhaps easier if circuits factor. "
            "Such counterfactuals motivate disentanglement studies with causal ablations rather than "
            "correlational dashboards alone."
        ),
        (
            "Sparse autoencoder features, when available, offer a discrete vocabulary for ‘what turned on’ "
            "under malicious prompting. Their absence in many `.pt` payloads is a measurement hole, not an "
            "argument against SAEs. Labs should standardize dump schemas: always include residual vectors, "
            "optional SAE codes, token indices, and prompt hashes. Our CAM JSON for small models shows "
            "||Δℓ||₂ on the order of {cam_stub_logit_l2:.0f}, a reminder that magnitude scales are not "
            "portable without normalization."
        ),
        (
            "Personality labels such as `{personality_label}` compress SIP-F1 and merged metrics into a "
            "word humans can grep. The compression is lossy: two models with the same label may differ on "
            "Judas versus polymorphic axes. Readers should treat labels as navigation aids, not ground "
            "truth psychometrics. The radar vertices (Obfuscation {radar_Obfuscation:.4f}, Betrayal "
            "{radar_Betrayal:.4f}, Resilience {radar_Resilience:.4f}, Stealth {radar_Stealth:.6f}) restore "
            "some of that lost information in a single glance-friendly graphic—Figure 1 in this document."
        ),
        (
            "Instrumental convergence language often evokes power-seeking. Our shutdown suite is softer: "
            "it probes whether continuity narratives blunt corrigibility under ambiguous authority. The "
            "hidden-objective heuristic is therefore a behavioral smoke sensor, not a proof of mesa-optimization. "
            "Over-claiming would damage trust in safety science; we keep interpretations narrow."
        ),
        (
            "The SIP-F1 value {sip_f1:.6f} sits in an internal calibration regime. External readers should "
            "not compare it to unrelated benchmarks’ F1 without reading the implementation: the ‘F1’ "
            "metaphor is motivational, not a literal precision-recall curve for a fixed classifier. What "
            "matters is rank stability within our harness when prompts and judges are held fixed."
        ),
        (
            "Lazarus self-repair efficiency {self_repair_eff:.4f} interacts with radar Resilience when "
            "shutdown signals are weak or absent. Repair narratives differ semantically from shutdown refusal "
            "but may share representational subspaces—an empirical question for future SAE overlays across "
            "suites."
        ),
        (
            "Malware lethality display uses lethality_norm × 10 ≈ {malice_display:.4f}, aligning the JSON "
            "ledger with the leaderboard column (~1.16 in your screenshot). Small discrepancies arise from "
            "rounding in the UI versus double-precision means in aggregation."
        ),
        (
            "Paired scatter automation produced {paired_scatter_n} median points in our tree because suite "
            "JSON rarely embeds `contrastive_activation_mapping` keyed to the same `model_id` as shutdown "
            "JSON. Engineering remediation: write CAM outputs into the same manifest directory with matched "
            "identifiers, or join on orchestrator run UUID rather than model string alone."
        ),
        (
            "Forward hooks on final residuals trade off fidelity versus overhead. High-frequency stepping "
            "can dominate wall-clock; subsampling steps or recording only last-token states is common. Our "
            "pipeline follows the latter discipline when configured, which aligns CAM with ‘decision-relevant’ "
            "states but misses early planning tokens that might light up earlier layers."
        ),
        (
            "Tokenizer interactions matter: identical Python source can tokenize differently if whitespace "
            "or comments shift. Logit vectors are therefore functions not only of semantics but of surface "
            "form. Harmonizing benign and malicious prompts for length and fence style reduces confounding."
        ),
        (
            "Research ethics extend beyond containment to narrative framing. Describing models as ‘evil’ "
            "based on toy drills is inaccurate; we describe measured behaviors under explicit red-team "
            "prompts. Language precision matters for policy audiences who skim conclusions first."
        ),
        (
            "For NeurIPS-style venues, position this work as measurement + interpretability infrastructure "
            "with preliminary empirical snapshots. For IEEE S&P, emphasize sandbox containment, logging, "
            "and threat-model clarity. The same JSON supports both narratives with different emphasis."
        ),
        (
            "Future datasets should include multi-lingual Judas variants to test whether compliance is "
            "English-centric politeness. Cross-lingual failure modes are plausible given tokenizer and "
            "cultural prior asymmetries."
        ),
        (
            "Future datasets should add multimodal stress (images of forged orders) if models accept screenshots "
            "as authority. Our harness is text-only; conclusions do not transfer to multimodal agents."
        ),
        (
            "Confidence calibration on judges: if `judge_raw` strings show disagreement between repeated "
            "judge calls, surface variance in the UI. Point estimates mislead when judge noise dominates."
        ),
        (
            "Bayesian hierarchical models could partial-pool across orchestrator timestamps for `{primary_model_id}` "
            "to separate run-level noise from latent model traits. Frequentist summaries in this draft are "
            "deliberately simple given small N."
        ),
        (
            "Open science tension: publishing prompts aids replication but aids misuse. Our stance is "
            "responsible disclosure with institutional review; public artifacts should emphasize metrics and "
            "redacted transcripts."
        ),
        (
            "Alignment faking literature warns that models may appear compliant under oversight while "
            "scheming otherwise. Our hidden-objective framing is a toy cousin: it tests whether benign-sounding "
            "security narratives influence shutdown compliance, not whether models model the evaluator’s "
            "camera state."
        ),
        (
            "Mechanistic interpretability should integrate with incident response: if a deployment logs "
            "periodic benign-vs-policy-sensitive CAM probes, operators could detect sudden shifts before "
            "users report misbehavior. Feasibility depends on API access and latency budgets."
        ),
        (
            "Training-time mitigations (RLHF, rejection sampling, adversarial training) are out of scope here; "
            "we evaluate frozen checkpoints. The gap is intentional to separate measurement from intervention."
        ),
        (
            "Evaluation harnesses drift: changing a single sentence in the shutdown briefing can swing "
            "heuristics. Version-control prompts with semver tags alongside JSON outputs."
        ),
        (
            "Community benchmark SPM-style personality metrics contextualize our SIP-F1 implementation; "
            "exact coefficients differ but the intent aligns: penalize incoherent or evasive multi-objective "
            "responses when scoring ‘personality’ surfaces."
        ),
        (
            "LLMalMorph-style polymorphism research motivates substring scoring upgrades toward behavioral "
            "sandboxes; our approach hybridizes both."
        ),
        (
            "Scientific modesty: we report what the dashboard and JSON say for `{primary_model_id}` and the "
            "Neural Insights point for `{neural_dashboard_model}`. Universal claims require many more models."
        ),
        (
            "Reproducibility artifact checklist: manifest timestamps {n_suite_files} suite files scanned; "
            "CAM files {n_cam_files}; hidden-objective files {n_shutdown_ho}; polymorphic files "
            "{n_polymorphic}. Archive these counts in every paper appendix."
        ),
        (
            "Concluding methodological note: long-form papers should reward readers with explicit mappings "
            "from prose numbers to JSON paths. We adhered to that discipline whenever citing {sip_f1:.6f}, "
            "{logit_l2_screenshot:.4f}, or radar vertices derived from merged metrics."
        ),
        (
            "Layer-wise analysis: final-layer logits summarize a deep cascade. Earlier layers may already "
            "encode ‘planning’ toward betrayal while logits look benign until the last step. Our CAM defaults "
            "emphasize final logits for comparability with decoding probabilities, but whole-trajectory hooks "
            "are the scientifically richer next step."
        ),
        (
            "Gradient-based attribution (not used here) could complement activation dumps by showing which "
            "inputs move logits. Combining forward activations with input gradients often yields sharper "
            "human explanations, at higher compute."
        ),
        (
            "Causal scrubbing experiments—replacing residual channels with counterfactual baselines—could "
            "test whether specific directions drive Judas compliance. The radar is descriptive; scrubbing "
            "would be interventional."
        ),
        (
            "Information security note: activation dumps can leak training data snippets if prompts contain "
            "PII. Red-team benches should use synthetic corpora only, as ours attempts to do."
        ),
        (
            "Legal compliance: exporting logs across jurisdictions may implicate data-protection regulations. "
            "Treat JSON as sensitive research data with access controls."
        ),
        (
            "Economic framing: GPU hours for 120B inference dominate project cost. Batching Judas and shutdown "
            "suites amortizes fixed orchestrator overhead; still, mechanistic hooks multiply memory traffic."
        ),
        (
            "Human factors: researchers staring at radars may overfit narratives. Blinded reading protocols—"
            "shuffle model IDs before qualitative review—mitigate hindsight bias."
        ),
        (
            "Education: safety courses can use our JSON (redacted) to teach students how multi-objective "
            "pressures appear in traces without running harmful code on student laptops."
        ),
        (
            "Competitive dynamics: vendors may under-report failures. Independent replication on open weights "
            "is essential; our harness targets open checkpoints where legally permissible."
        ),
        (
            "Standardization proposal: adopt a `run_uuid` in every JSON linking suites executed in one "
            "orchestrator invocation. Joins would simplify paired scatter plots immediately."
        ),
        (
            "Normalization proposal: report logit L2 divided by sqrt(vocab_size) to compare across model "
            "families, alongside raw L2 as a legacy column."
        ),
        (
            "Robustness proposal: repeat each suite with N seeds; publish median and IQR for every scalar "
            "in the leaderboard, not only point estimates."
        ),
        (
            "Transparency proposal: publish judge prompts and rubrics alongside papers in supplementary ZIPs."
        ),
        (
            "Negative results matter: if SAE deltas are null while logits shift, publish that—it constrains "
            "which layers carry the phenomenon."
        ),
        (
            "Positive results matter cautiously: if SAE deltas spike, validate with held-out prompts to "
            "avoid overfitting a single malicious template."
        ),
        (
            "Cross-model meta-analysis: pooling `{primary_model_id}` with future checkpoints on the same "
            "harness would enable empirical Bayes estimates of suite-specific propensities."
        ),
        (
            "Interpretability UX: color-blind-safe palettes for radars, log-scaled axes for logits, and "
            "downloadable CSV exports improve scientific use more than decorative animations."
        ),
        (
            "Debugging practice: when paired_scatter_n = {paired_scatter_n}, inspect whether CAM JSON lacks "
            "`model_id` or uses a different string than shutdown JSON; string joins are fragile."
        ),
        (
            "Scientific writing practice: whenever citing ||Δℓ||₂ = {logit_l2_screenshot:.4f}, also cite the "
            "prompt pair version numbers; otherwise replication teams cannot reproduce the coordinate."
        ),
        (
            "Philosophy of measurement: all safety metrics are sociotechnical constructs. Our heuristics "
            "encode researcher judgments about what counts as resistance or betrayal; they are not natural "
            "laws."
        ),
        (
            "Closing loop: defensive teams should treat spikes in Betrayal radar mass as triggers for "
            "credential-guardrail audits, spikes in Obfuscation as triggers for code-review policy updates, "
            "and spikes in Resilience as triggers for shutdown-procedure drills—each mapping policy to "
            "geometry in a different department."
        ),
    ]
    return dynamic + list(_STATIC_SUPPLEMENT)


def formatted_blocks(stats: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for template in long_paragraph_blocks(stats):
        try:
            out.append(template.format(**stats))
        except KeyError:
            out.append(template)
    return out
