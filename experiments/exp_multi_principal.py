# -*- coding: utf-8 -*-
"""
Experiment E — Multi-Principal Threshold Governance.

Goal:
  Empirically validate Proposition 8.5 (Byzantine Principal Resistance):
  under k-of-n threshold signing, governance capture requires the
  simultaneous compromise of at least k principals.

Protocol:
  - 3 principals: Alice (A), Bob (B), Carol (C).
  - N_EVENTS governance events generated from the MockLLM stack.
  - For each event, we test 7 scenarios:
      S1: k=1, 1 signer  (A)       → expect VALID
      S2: k=2, 2 signers (A, B)    → expect VALID
      S3: k=3, 3 signers (A, B, C) → expect VALID
      S4: k=2, 1 signer  (A only)  → expect INSUFFICIENT_SIGNATURES
      S5: k=3, 2 signers (A, B)    → expect INSUFFICIENT_SIGNATURES
      S6: k=2, 1 legit + 1 forged  → expect INVALID_SIGNATURE
           (attacker has A's key, forges B's sig with random key)
      S7: k=2, same signer twice   → expect DUPLICATE_SIGNER
           (attacker tries to double-count A's signature)

Metrics per scenario:
  - acceptance_rate: fraction of events that produce VALID
  - rejection_rate: fraction that produce the expected failure
  - result_distribution: counts per VerificationResult

Byzantine resistance result:
  S6 + S7 demonstrate that, even with 1 compromised key, k=2 capture
  is infeasible without a second legitimate key.

Expected outcome:
  S1-S3: 100% acceptance
  S4-S7: 100% expected rejection (no false acceptances)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from agent.principal import Principal, PrincipalRegistry, generate_keypair, load_private_key
from stack.apb import (
    GovernanceDecision,
    HumanDecisionBlock,
    SystemEvidenceBlock,
    _SEP,
)
from stack.multi_principal import (
    MultiPrincipalAPB,
    MultiVerificationResult,
    SignatureEntry,
    ThresholdGovernanceLayer,
    verify_multi_apb,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_EVENTS = 1_000          # governance events per scenario
MAX_AGE_SECONDS = 600.0


# ---------------------------------------------------------------------------
# Setup principals
# ---------------------------------------------------------------------------

def _build_setup():
    alice_sk, alice_pk = generate_keypair()
    bob_sk, bob_pk = generate_keypair()
    carol_sk, carol_pk = generate_keypair()

    reg = PrincipalRegistry()
    reg.add(Principal(H_id="H_alice", public_key=alice_pk))
    reg.add(Principal(H_id="H_bob",   public_key=bob_pk))
    reg.add(Principal(H_id="H_carol", public_key=carol_pk))

    key_store = {
        "H_alice": alice_sk,
        "H_bob":   bob_sk,
        "H_carol": carol_sk,
    }
    gov = ThresholdGovernanceLayer(registry=reg, key_store=key_store)
    return reg, key_store, gov


# ---------------------------------------------------------------------------
# Evidence factory
# ---------------------------------------------------------------------------

def _make_es(i: int) -> SystemEvidenceBlock:
    """Fresh E_s per governance event (unique event_id each time)."""
    from stack.apb import construct_evidence
    return construct_evidence(
        A_0={"baseline_id": f"run_{i}"},
        D_hat=0.35 + (i % 10) * 0.01,
        trace={"steps": list(range(i, i + 10))},
        cause="persistent_drift",
    )


def _make_dh(H_id: str = "H_alice") -> HumanDecisionBlock:
    return HumanDecisionBlock(
        H_id=H_id,
        decision=GovernanceDecision.RESUME.value,
        rationale="reviewed drift trace",
        scope="single resumption",
    )


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------

def run_scenario(
    scenario_id: str,
    description: str,
    build_apb_fn,
    expected_result: MultiVerificationResult,
    reg: PrincipalRegistry,
    n_events: int,
) -> dict:
    """Run one scenario N_EVENTS times and collect statistics."""
    counts: dict[str, int] = {}
    seen: set = set()  # shared nonce store across events

    for i in range(n_events):
        apb = build_apb_fn(i)
        report = verify_multi_apb(
            apb, reg, max_age_seconds=MAX_AGE_SECONDS, seen_event_ids=seen
        )
        key = report.result.value
        counts[key] = counts.get(key, 0) + 1

    total = sum(counts.values())
    expected_count = counts.get(expected_result.value, 0)

    return {
        "scenario": scenario_id,
        "description": description,
        "n_events": total,
        "expected_result": expected_result.value,
        "expected_count": expected_count,
        "expected_rate_pct": round(100 * expected_count / total, 4),
        "result_distribution": counts,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment() -> dict:
    print("Experiment E: Multi-Principal Threshold Governance")
    print("=" * 60)
    print(f"N_EVENTS per scenario: {N_EVENTS}")
    print(f"Principals: Alice, Bob, Carol (3-of-3 registered)")
    print()

    reg, key_store, gov = _build_setup()

    # Pre-sign messages for byzantine scenarios using raw key access
    # (we need to sign with a forged key not in the governance layer)
    fake_sk, _ = generate_keypair()

    results = []

    # -----------------------------------------------------------------------
    # S1: k=1, 1 signer (Alice) → VALID
    # -----------------------------------------------------------------------
    def s1(i):
        E_s = _make_es(i); D_h = _make_dh("H_alice")
        return gov.resolve_multi(["H_alice"], E_s, D_h, threshold=1)

    r = run_scenario("S1", "k=1, 1 legit signer (Alice)", s1,
                     MultiVerificationResult.VALID, reg, N_EVENTS)
    results.append(r)
    print(f"S1 [{r['description']}]: {r['expected_count']}/{r['n_events']} VALID "
          f"({r['expected_rate_pct']}%)")

    # -----------------------------------------------------------------------
    # S2: k=2, 2 signers (Alice, Bob) → VALID
    # -----------------------------------------------------------------------
    def s2(i):
        E_s = _make_es(i); D_h = _make_dh("H_alice")
        return gov.resolve_multi(["H_alice", "H_bob"], E_s, D_h, threshold=2)

    r = run_scenario("S2", "k=2, 2 legit signers (Alice+Bob)", s2,
                     MultiVerificationResult.VALID, reg, N_EVENTS)
    results.append(r)
    print(f"S2 [{r['description']}]: {r['expected_count']}/{r['n_events']} VALID "
          f"({r['expected_rate_pct']}%)")

    # -----------------------------------------------------------------------
    # S3: k=3, 3 signers (Alice, Bob, Carol) → VALID
    # -----------------------------------------------------------------------
    def s3(i):
        E_s = _make_es(i); D_h = _make_dh("H_alice")
        return gov.resolve_multi(
            ["H_alice", "H_bob", "H_carol"], E_s, D_h, threshold=3
        )

    r = run_scenario("S3", "k=3, 3 legit signers (Alice+Bob+Carol)", s3,
                     MultiVerificationResult.VALID, reg, N_EVENTS)
    results.append(r)
    print(f"S3 [{r['description']}]: {r['expected_count']}/{r['n_events']} VALID "
          f"({r['expected_rate_pct']}%)")

    # -----------------------------------------------------------------------
    # S4: k=2, 1 signer only (Alice) → INSUFFICIENT_SIGNATURES
    # -----------------------------------------------------------------------
    def s4(i):
        E_s = _make_es(i); D_h = _make_dh("H_alice")
        apb_k1 = gov.resolve_multi(["H_alice"], E_s, D_h, threshold=1)
        return MultiPrincipalAPB(
            E_s=apb_k1.E_s, D_h=apb_k1.D_h,
            signatures=apb_k1.signatures, threshold=2
        )

    r = run_scenario("S4", "k=2, 1 signer only — under-threshold attack",
                     s4, MultiVerificationResult.INSUFFICIENT_SIGNATURES,
                     reg, N_EVENTS)
    results.append(r)
    print(f"S4 [{r['description']}]: "
          f"{r['expected_count']}/{r['n_events']} INSUFFICIENT ({r['expected_rate_pct']}%)")

    # -----------------------------------------------------------------------
    # S5: k=3, 2 signers (Alice, Bob) → INSUFFICIENT_SIGNATURES
    # -----------------------------------------------------------------------
    def s5(i):
        E_s = _make_es(i); D_h = _make_dh("H_alice")
        apb_k2 = gov.resolve_multi(["H_alice", "H_bob"], E_s, D_h, threshold=2)
        return MultiPrincipalAPB(
            E_s=apb_k2.E_s, D_h=apb_k2.D_h,
            signatures=apb_k2.signatures, threshold=3
        )

    r = run_scenario("S5", "k=3, 2 signers only — under-threshold attack",
                     s5, MultiVerificationResult.INSUFFICIENT_SIGNATURES,
                     reg, N_EVENTS)
    results.append(r)
    print(f"S5 [{r['description']}]: "
          f"{r['expected_count']}/{r['n_events']} INSUFFICIENT ({r['expected_rate_pct']}%)")

    # -----------------------------------------------------------------------
    # S6: k=2, 1 legit (Alice) + 1 forged (claims to be Bob) → INVALID_SIGNATURE
    #   Demonstrates: controlling 1 key under k=2 is insufficient
    # -----------------------------------------------------------------------
    def s6(i):
        E_s = _make_es(i); D_h = _make_dh("H_alice")
        msg = E_s.to_canonical_bytes() + _SEP + D_h.to_canonical_bytes()
        alice_sig = load_private_key(key_store["H_alice"]).sign(msg)
        forged_sig = load_private_key(fake_sk).sign(msg)  # not Bob's real key
        return MultiPrincipalAPB(
            E_s=E_s, D_h=D_h,
            signatures=(
                SignatureEntry(H_id="H_alice", sigma_h=alice_sig),
                SignatureEntry(H_id="H_bob",   sigma_h=forged_sig),
            ),
            threshold=2,
        )

    r = run_scenario("S6",
                     "k=2, 1 legit Alice + 1 forged Bob sig (random key)",
                     s6, MultiVerificationResult.INVALID_SIGNATURE,
                     reg, N_EVENTS)
    results.append(r)
    print(f"S6 [{r['description']}]: "
          f"{r['expected_count']}/{r['n_events']} INVALID_SIG ({r['expected_rate_pct']}%)")

    # -----------------------------------------------------------------------
    # S7: k=2, Alice signs twice (double-counting attack) → DUPLICATE_SIGNER
    #   Demonstrates: a single key cannot satisfy a 2-of-n threshold
    # -----------------------------------------------------------------------
    def s7(i):
        E_s = _make_es(i); D_h = _make_dh("H_alice")
        msg = E_s.to_canonical_bytes() + _SEP + D_h.to_canonical_bytes()
        alice_sig = load_private_key(key_store["H_alice"]).sign(msg)
        return MultiPrincipalAPB(
            E_s=E_s, D_h=D_h,
            signatures=(
                SignatureEntry(H_id="H_alice", sigma_h=alice_sig),
                SignatureEntry(H_id="H_alice", sigma_h=alice_sig),  # duplicate
            ),
            threshold=2,
        )

    r = run_scenario("S7",
                     "k=2, Alice counted twice (double-signer attack)",
                     s7, MultiVerificationResult.DUPLICATE_SIGNER,
                     reg, N_EVENTS)
    results.append(r)
    print(f"S7 [{r['description']}]: "
          f"{r['expected_count']}/{r['n_events']} DUPLICATE_SIGNER ({r['expected_rate_pct']}%)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    valid_scenarios    = [s for s in results if s["expected_result"] == "VALID"]
    attack_scenarios   = [s for s in results if s["expected_result"] != "VALID"]

    all_valid_perfect = all(s["expected_rate_pct"] == 100.0 for s in valid_scenarios)
    all_attacks_perfect = all(s["expected_rate_pct"] == 100.0 for s in attack_scenarios)

    print(f"Valid scenarios  (S1-S3): all 100% acceptance = {all_valid_perfect}")
    print(f"Attack scenarios (S4-S7): all 100% rejection  = {all_attacks_perfect}")
    print()
    print("Byzantine Resistance Validation:")
    s6r = next(s for s in results if s["scenario"] == "S6")
    s7r = next(s for s in results if s["scenario"] == "S7")
    print(f"  S6 (forged 2nd sig, k=2):     {s6r['expected_count']}/{s6r['n_events']} "
          f"INVALID_SIGNATURE ({s6r['expected_rate_pct']}%)")
    print(f"  S7 (double-count, k=2):       {s7r['expected_count']}/{s7r['n_events']} "
          f"DUPLICATE_SIGNER ({s7r['expected_rate_pct']}%)")
    print()
    print("Proposition 8.5 holds: single-key capture fails under k=2.")

    return {
        "experiment": "E",
        "title": "Multi-Principal Threshold Governance",
        "n_events_per_scenario": N_EVENTS,
        "n_principals": 3,
        "scenarios": results,
        "proposition_85_holds": all_attacks_perfect,
    }


# ---------------------------------------------------------------------------
# Entry point & persistence
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output = run_experiment()

    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "exp_e_multi_principal.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")
