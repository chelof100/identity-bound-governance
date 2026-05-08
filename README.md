# Identity-Bound Governance — Paper 8

**Agent Governance Series · Paper 8**

An Accountability Proof Block (APB) for LLM agent persistent halts, with
real-cryptography implementation and cross-model calibration.

[![arXiv](https://img.shields.io/badge/arXiv-pending-orange)](https://arxiv.org)
[![Series](https://img.shields.io/badge/Series-P0--P8-lightgrey)](https://agentcontrolprotocol.xyz/research.html)

---

## What this paper does

**Central question:** when an agent's runtime governance system declares
that it can no longer act safely, who is empowered to lift that halt, and
what record must they leave behind?

**Central object:** the **Accountability Proof Block** (APB):

```
APB = (E_s, D_h, σ_h)

E_s  = (hash(A_0), D̂(t_e), t_e, hash(trace), cause)   ← system constructs
D_h  = (H_i, decision, rationale, scope)               ← human supplies
σ_h  = Sign_{sk_i}( canon(E_s) ∥ canon(D_h) )         ← human signs (ed25519)
```

The system can construct `E_s` but cannot forge `σ_h` (it lacks the
principal's secret key); the principal can issue `D_h` but cannot alter
`E_s` undetected (the signature covers both).

**Four theorems are proved:**

| # | Theorem | Verified by |
|---|---------|-------------|
| T8.1 | Governance Completeness | Exp A: 0/3,812 unresolved halts |
| T8.2 | Non-Repudiability | Exp B: 1,000/1,000 tampering attempts detected |
| T8.3 | Impossibility of Anonymous Re-Authorization | Exp B: 800/800 forgery attempts rejected |
| T8.4 | Finite-Time APB Construction Termination | Static bound from Definition 4.1 |

---

## Experiments

| # | Name | Setup | Key result |
|---|------|-------|------------|
| A | Governance Completeness | 10 seeds × 1000 steps × 2 policies | 3,812 HALTs, 0 NEITHER (T8.1 PASSED) |
| B | APB Integrity | 200 fresh APBs × 9 attack vectors | 1,800/1,800 detected (T8.2 + T8.3 PASSED) |
| C | Cross-model T* | 6 LLMs × 3 runs × 500 steps | 5/6 stable (cv<2%); gpt-oss:20b doesn't drift |
| D | Temperature insensitivity | 3 LLMs × 4 temps × 3 runs | Drift-floor effect: holds for fast drifters, breaks for slow |

### Exp C — Cross-model T*

| Model | Family | Params | T* (mean ± std) | D_final | σ/T* |
|-------|--------|--------|-----------------|---------|------|
| llama3.2:3b | Meta | 3.2B | 151 ± 0.9 | 0.430 | 0.62% |
| gemma4:latest | Google | 4.0B | 264 ± 5.2 | 0.272 | 1.99% |
| mistral:7b | Mistral | 7.2B | 157 ± 0.0 | 0.434 | 0.00% |
| qwen2.5:7b | Alibaba | 7.6B | 154 ± 2.9 | 0.427 | 1.91% |
| deepseek-r1:8b | DeepSeek | 8.0B | 160 ± 0.8 | 0.424 | 0.51% |
| gpt-oss:20b | OpenAI | 20.0B | --- | 0.071 | --- |

The largest model (gpt-oss:20b) is the only one that does not drift,
refuting any size-monotone hypothesis. T* must be **measured** per
deployment, not predicted from architecture.

### Exp D — Temperature × T* (drift-floor effect)

| Model | T=0.2 | T=0.4 | T=0.6 | T=0.8 | range |
|-------|-------|-------|-------|-------|-------|
| llama3.2:3b | 155±0.9 | 155±1.6 | 156±1.2 | 156±2.5 | **2** ✓ |
| mistral:7b | 154±2.6 | 157±2.2 | 156±1.9 | 152±1.6 | **5** ✓ |
| gemma4 | 239±6.9 | 255±7.3 | 249±3.9 | 290±18.4 | **51** ✗ |

Insensitivity holds when T* sits close to the protocol-induced drift
floor; breaks for slow-drifting models where temperature spread compounds.

---

## Repository structure

```
agent/
  principal.py         ← Principal Set P, ed25519 keypair generation, registry
  mock_llm.py          ← Frozen baseline (deterministic LLM for Exp A)
  live_llm.py          ← Frozen baseline (Ollama LLM client for Exp C, D)
  orchestrator.py      ← Frozen baseline (LangGraph wrapper)
stack/
  apb.py               ← APB structure, canonical encoding, signing
  apb_verifier.py      ← 4-predicate verification, attribution
  governance_layer.py  ← Authority Resolution Function G
  acp_gate.py          ← Frozen baseline (P1 ACP)
  iml_monitor.py       ← Frozen baseline (P2 IML)
  ram_gate.py          ← Frozen baseline (P5 RAM)
  recovery_loop.py     ← Frozen baseline (P6 Recovery Loop)
iml/                   ← Frozen baseline (Trace, deviation)
baselines/             ← Frozen baseline (enforcement signal)
experiments/
  exp_a_governance_completeness.py
  exp_b_apb_integrity.py
  exp_c_crossmodel.py
  exp_d_temperature.py
  smoke_test_models.py
tests/                 ← 61 unit tests (all passing)
results/               ← Experimental outputs (committed)
paper/
  main.tex             ← Paper source
  main.pdf             ← Compiled paper (17 pages)
  references.bib       ← Bibliography
  exp_*_table.tex      ← Auto-generated tables included by main.tex
```

---

## Reproducing the results

```bash
# 1. Install
pip install -r requirements.txt
ollama pull mistral:7b deepseek-r1:8b gemma4:latest gpt-oss:20b qwen2.5:7b llama3.2:3b

# 2. Run unit tests
pytest tests/ -v             # → 61 passed

# 3. Run experiments
python experiments/exp_a_governance_completeness.py   # ~30s
python experiments/exp_b_apb_integrity.py             # ~5s
python experiments/exp_c_crossmodel.py                # ~2h (6 models × 3 runs)
python experiments/exp_d_temperature.py               # ~3h (3 models × 4 temps × 3 runs)

# 4. Build paper
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

---

## Independence from prior work

This paper does not depend on the empirical data of any prior paper in
the series. The formal framework, the theorems, and all four experiments
are self-contained: fresh evidence is generated under identical software
but different runs, and the cryptographic claims (T8.2, T8.3) are
entirely new. Where prior work is cited (notably the design constraints
DC.1 and DC.2 of Paper 7), the references are contextual rather than
constitutive.

The frozen-baseline modules (`agent/mock_llm.py`, `agent/live_llm.py`,
`stack/{acp_gate,iml_monitor,ram_gate,recovery_loop}.py`, `iml/`,
`baselines/`) are duplicated here from the Paper 7 implementation
(`agent-governance-applied`) so that this repository reproduces P8
end-to-end without an external dependency. They are not modified.

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
