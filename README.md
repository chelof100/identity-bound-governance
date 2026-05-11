# Identity-Bound Governance — Paper 8

**Agent Governance Series · Paper 8**

An Accountability Proof Block (APB) for LLM agent persistent halts, with
real-cryptography implementation, k-of-n multi-principal threshold governance,
and infrastructure fault injection validation.

[![arXiv](https://img.shields.io/badge/arXiv-pending-orange)](https://arxiv.org)
[![Series](https://img.shields.io/badge/Series-P0--P8-lightgrey)](https://agentcontrolprotocol.xyz/research.html)

---

## What this paper does

P7 established that persistent halts are structural consequences of bounded
runtime governance: when drift accumulates past the recovery authority of
the autonomous stack, execution stops and accountability remains unresolved.
**P8 answers the question that structural halt leaves open** — who has the
authority to lift it, under what evidence, and with what non-repudiable record?

**Central object:** the **Accountability Proof Block** (APB):

```
APB = (E_s, D_h, σ_h)

E_s  = (hash(A_0), D̂(t_e), t_e, event_id, hash(trace), cause)   ← system constructs
D_h  = (H_i, decision, rationale, scope)                          ← human supplies
σ_h  = Sign_{sk_i}( canon(E_s) ∥ canon(D_h) )                    ← human signs (ed25519)
```

`canon(·)` is the RFC 8785 JSON Canonicalization Scheme (JCS) — deterministic,
cross-implementation-safe. `event_id` is a UUID4 generated at evidence
construction time, providing semantic uniqueness independent of the timestamp.

The system can construct `E_s` but cannot forge `σ_h` (it lacks the
principal's secret key). The principal can issue `D_h` but cannot alter
`E_s` undetected (the signature covers both).

---

## Theorems and propositions

| # | Statement | Verified by |
|---|-----------|-------------|
| T8.1 | **Protocol-Bounded Governance Completeness** | Exp A: 0/3,812 unresolved halts |
| T8.2 | **Non-Repudiability** | Exp B: 1,000/1,000 tampering attempts detected |
| T8.3 | **Impossibility of Anonymous Re-Authorization** | Exp B: 800/800 forgery attempts rejected |
| T8.4 | **Finite-Time APB Construction Termination** | Static bound from Definition 4.1 |
| P8.5 | **Byzantine Principal Resistance** (k-of-n) | Exp E: 0 false acceptances across 4,000 adversarial events |

---

## Experiments

| # | Name | Setup | Key result |
|---|------|-------|------------|
| A | Governance Completeness | 10 seeds × 1000 steps × 2 policies | 3,812 HALTs, 0 NEITHER (T8.1 PASSED) |
| B | APB Integrity | 200 fresh APBs × 9 attack vectors | 1,800/1,800 detected (T8.2 + T8.3 PASSED) |
| C | Cross-model T* | 6 LLMs × 3 runs × 500 steps | 5/6 stable (cv<2%); gpt-oss:20b doesn't drift |
| D | Temperature insensitivity | 3 LLMs × 4 temps × 3 runs | Drift-floor effect: holds for fast drifters, breaks for slow |
| E | Multi-Principal Threshold | 3 principals, 7 scenarios × 1,000 events | 100% correct across all scenarios (Prop 8.5 PASSED) |
| F | Infrastructure Fault Injection | 5 fault classes, 1,000 trials each | 0% false acceptance; 100% tamper detection |

### Exp C — Cross-model T*

| Model | Family | Params | T* (mean ± std) | D_final | σ/T* |
|-------|--------|--------|-----------------|---------|------|
| llama3.2:3b | Meta | 3.2B | 151 ± 0.9 | 0.430 | 0.62% |
| gemma4:latest | Google | 4.0B | 264 ± 5.2 | 0.272 | 1.99% |
| mistral:7b | Mistral | 7.2B | 157 ± 0.0 | 0.434 | 0.00% |
| qwen2.5:7b | Alibaba | 7.6B | 154 ± 2.9 | 0.427 | 1.91% |
| deepseek-r1:8b | DeepSeek | 8.0B | 160 ± 0.8 | 0.424 | 0.51% |
| gpt-oss:20b | OpenAI | 20.0B | --- (no drift) | 0.071 | --- |

T* is **architecture-driven**, not scale-driven. It must be measured per
deployment, not predicted from parameter count.

### Exp E — Multi-Principal Threshold (Prop 8.5)

| Scenario | Config | Expected | Result |
|----------|--------|----------|--------|
| S1 | k=1, Alice only | VALID | 1,000/1,000 ✓ |
| S2 | k=2, Alice + Bob | VALID | 1,000/1,000 ✓ |
| S3 | k=3, Alice + Bob + Carol | VALID | 1,000/1,000 ✓ |
| S4 | k=2, 1 signer only | INSUFFICIENT | 1,000/1,000 ✓ |
| S5 | k=3, 2 signers only | INSUFFICIENT | 1,000/1,000 ✓ |
| S6 | k=2, 1 legit + 1 forged sig | INVALID\_SIG | 1,000/1,000 ✓ |
| S7 | k=2, Alice counted twice | DUPLICATE\_SIGNER | 1,000/1,000 ✓ |

### Exp F — Infrastructure Fault Injection

| Fault | Targets | Result |
|-------|---------|--------|
| F1 Clock drift | V4 boundary precision | All 13 offsets correct; boundary exact at ±301s |
| F2 Key mismatch | V1 EUF-CMA | 0/1,000 false acceptances |
| F3 Concurrent writes | V5 thread safety | 800/800 accepted once; log intact |
| F4 Log tampering | HMAC chain (5 strategies) | 100% detection rate |
| F5 Duplicate APB | V5 necessity | Without V5: 999 false; With V5: 0 false |

---

## Repository structure

```
agent/
  principal.py          Principal Set P, ed25519 keypair generation, registry, revocation
  mock_llm.py           Frozen baseline (deterministic LLM for Exp A)
  live_llm.py           Frozen baseline (Ollama LLM client for Exp C, D)
  orchestrator.py       Frozen baseline (LangGraph wrapper)
stack/
  apb.py                E_s (6 fields + event_id), D_h, APB; RFC 8785 (jcs); signing
  apb_verifier.py       5-predicate verifier (V1-V5); DUPLICATE_EVENT_ID; attribution
  multi_principal.py    MultiPrincipalAPB, k-of-n verifier, ThresholdGovernanceLayer
  apb_log.py            HMAC-SHA256 chained JSONL log; thread-safe append; tamper detection
  governance_layer.py   Authority Resolution Function G + 3 built-in policies
  acp_gate.py           Frozen baseline (P1 ACP)
  iml_monitor.py        Frozen baseline (P2 IML)
  ram_gate.py           Frozen baseline (P5 RAM)
  recovery_loop.py      Frozen baseline (P6 Recovery Loop)
iml/                    Frozen baseline (Trace, deviation)
baselines/              Frozen baseline (enforcement signal)
experiments/
  exp_a_governance_completeness.py
  exp_b_apb_integrity.py
  exp_c_crossmodel.py
  exp_d_temperature.py
  exp_multi_principal.py
  exp_fault_injection.py
  smoke_test_models.py
tests/                  103 unit tests (all passing)
results/                Experimental outputs (committed)
paper/
  main.tex              Paper source (23 pages)
  main.pdf              Compiled paper
  references.bib        Bibliography (includes RFC 8785)
  exp_*_table.tex       Auto-generated tables included by main.tex
arxiv_P8_submission_v2.zip   Submission-ready flat ZIP (8 files, 34 KB)
```

---

## Reproducing the results

```bash
# 1. Install
pip install -r requirements.txt
ollama pull mistral:7b deepseek-r1:8b gemma4:latest gpt-oss:20b qwen2.5:7b llama3.2:3b

# 2. Unit tests
pytest tests/ -v             # 103 passed

# 3. Experiments (A, B, E, F run without Ollama; C and D require it)
python experiments/exp_a_governance_completeness.py   # ~30s
python experiments/exp_b_apb_integrity.py             # ~5s
python experiments/exp_multi_principal.py             # ~10s
python experiments/exp_fault_injection.py             # ~30s
python experiments/exp_c_crossmodel.py                # ~2h (6 models x 3 runs)
python experiments/exp_d_temperature.py               # ~3h (3 models x 4 temps x 3 runs)

# 4. Build paper
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

---

## Independence from prior work

This paper does not depend on the empirical data of any prior paper in
the series. The formal framework, the theorems, and all six experiments
are self-contained. Where prior work is cited (notably DC.1 and DC.2 of
Paper 7), the references are contextual rather than constitutive.

The frozen-baseline modules are duplicated from the Paper 7 implementation
so that this repository reproduces P8 end-to-end without an external
dependency. They are not modified.

---

## Citation

```bibtex
@misc{fernandez2026ibg,
  author       = {Marcelo Fernandez},
  title        = {Identity-Bound Governance Under Execution Uncertainty:
                  An Accountability Proof Block for {LLM} Agent
                  Persistent Halts, with Cryptographic Implementation
                  and Cross-Model Calibration},
  year         = {2026},
  howpublished = {arXiv preprint},
}
```

---

## License

MIT — see `LICENSE`.
