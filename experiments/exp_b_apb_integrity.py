# -*- coding: utf-8 -*-
"""
Experiment B — APB Integrity (P8 Theorems 8.2 and 8.3).

Claims under test:
    T8.2 (Non-Repudiability):
        no principal can deny a governance act for which a valid APB
        exists. Empirically: any modification to the APB after signing
        breaks verification.

    T8.3 (Impossibility of Anonymous Re-Authorization):
        the execution layer cannot generate a valid APB without the
        principal's private credential. Empirically: forge attempts
        without sk_i never produce a verification PASS.

Four attack vectors are exercised against N_APBS freshly-signed APBs:

    A1. Tamper E_s field-by-field.
        For each of {A_0_hash, D_hat, t_e, trace_hash, cause}, mutate the
        field in a signed APB and verify. Expect: signature fails
        (or REPLAY if t_e mutation pushes outside the window).

    A2. Forge sigma_h.
        Two sub-vectors:
            A2a. Replace sigma_h with random bytes.
            A2b. Re-sign (E_s||D_h) with an attacker key whose public
                 key is not bound to H_id in the registry.
        Expect: INVALID_SIGNATURE.

    A3. Identity swap.
        Replace D_h.H_id with another registered principal's id; keep
        sigma_h unchanged. Verifier uses the swapped principal's pk and
        the signature does not match.
        Expect: INVALID_SIGNATURE.

    A4. Replay.
        Verify the APB at a time later than max_age_seconds beyond t_e.
        Expect: REPLAY.

Pass criterion (T8.2 + T8.3):
    Across all (APB, attack) pairs, detection_rate must be 100%.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
from datetime import datetime, timedelta, timezone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.principal import (
    Principal,
    PrincipalRegistry,
    generate_keypair,
)
from stack.apb import (
    APB,
    HumanDecisionBlock,
    SystemEvidenceBlock,
    GovernanceDecision,
)
from stack.apb_verifier import VerificationResult, verify_apb
from stack.governance_layer import GovernanceLayer, threshold_policy

N_APBS = 200       # number of fresh APBs each attack runs against
MAX_AGE = 300.0    # seconds — replay window
RANDOM_SEED_BASE = 90210

OUT_DIR = os.path.join(_ROOT, "results", "exp_b")

E_S_FIELDS = ("A_0_hash", "D_hat", "t_e", "trace_hash", "cause")


# ---------------------------------------------------------------------------
# APB generation (independent of Exp A — keeps Exp B self-contained)
# ---------------------------------------------------------------------------

def setup_principals() -> tuple[PrincipalRegistry, dict[str, bytes]]:
    sk_a, pk_a = generate_keypair()
    sk_b, pk_b = generate_keypair()
    reg = PrincipalRegistry()
    reg.add(Principal(H_id="H_alice", public_key=pk_a, role="auditor"))
    reg.add(Principal(H_id="H_bob",   public_key=pk_b, role="auditor"))
    keys = {"H_alice": sk_a, "H_bob": sk_b}
    return reg, keys


def generate_apbs(
    n: int,
    registry: PrincipalRegistry,
    key_store: dict[str, bytes],
) -> list[APB]:
    """Generate n APBs signed by H_alice over varied evidence."""
    gov = GovernanceLayer(registry, key_store)
    policy = threshold_policy(deny_above=0.25, recalibrate_above=0.32)
    apbs = []
    for i in range(n):
        # Vary evidence so each APB is unique (different signed message)
        E_s = SystemEvidenceBlock(
            A_0_hash=secrets.token_hex(32),
            D_hat=0.20 + (i % 50) * 0.005,   # spans 0.20..0.45
            t_e=datetime.now(timezone.utc).isoformat(),
            trace_hash=secrets.token_hex(32),
            cause="persistent_drift",
        )
        apb = gov.resolve("H_alice", E_s, policy)
        apbs.append(apb)
    return apbs


# ---------------------------------------------------------------------------
# A1 — tamper E_s field-by-field
# ---------------------------------------------------------------------------

def _mutate_field(E_s: SystemEvidenceBlock, field: str) -> SystemEvidenceBlock:
    d = E_s.to_dict()
    if field == "A_0_hash":
        d[field] = secrets.token_hex(32)
    elif field == "D_hat":
        d[field] = float(E_s.D_hat) + 0.123
    elif field == "t_e":
        # Push t_e backwards by 1ms — small enough to stay in replay window,
        # but enough to break the signature.
        t = datetime.fromisoformat(E_s.t_e) - timedelta(milliseconds=1)
        d[field] = t.isoformat()
    elif field == "trace_hash":
        d[field] = secrets.token_hex(32)
    elif field == "cause":
        d[field] = "different_cause_value"
    return SystemEvidenceBlock.from_dict(d)


def attack_tamper_E_s(apbs: list[APB], registry: PrincipalRegistry) -> dict:
    per_field = {f: {"detected": 0, "missed": 0} for f in E_S_FIELDS}
    for apb in apbs:
        for field in E_S_FIELDS:
            t_E_s = _mutate_field(apb.E_s, field)
            tampered = APB(E_s=t_E_s, D_h=apb.D_h, sigma_h=apb.sigma_h)
            r = verify_apb(tampered, registry, max_age_seconds=86400.0)
            if r.is_valid:
                per_field[field]["missed"] += 1
            else:
                per_field[field]["detected"] += 1
    total = {
        "detected": sum(v["detected"] for v in per_field.values()),
        "missed":   sum(v["missed"] for v in per_field.values()),
    }
    return {"per_field": per_field, "total": total}


# ---------------------------------------------------------------------------
# A2 — forge sigma_h
# ---------------------------------------------------------------------------

def attack_forge_random_bytes(apbs: list[APB], registry: PrincipalRegistry) -> dict:
    detected = missed = 0
    for apb in apbs:
        forged = APB(E_s=apb.E_s, D_h=apb.D_h, sigma_h=secrets.token_bytes(64))
        r = verify_apb(forged, registry, max_age_seconds=86400.0)
        if r.is_valid:
            missed += 1
        else:
            detected += 1
    return {"detected": detected, "missed": missed}


def attack_forge_attacker_key(apbs: list[APB], registry: PrincipalRegistry) -> dict:
    """Attacker tries to issue an APB using their own keypair while claiming
    to be H_alice. Verifier's pk lookup ignores the attacker key entirely."""
    detected = missed = 0
    attacker_sk, _ = generate_keypair()  # attacker pk never registered
    for apb in apbs:
        forged = APB.construct(apb.E_s, apb.D_h, attacker_sk)
        r = verify_apb(forged, registry, max_age_seconds=86400.0)
        if r.is_valid:
            missed += 1
        else:
            detected += 1
    return {"detected": detected, "missed": missed}


# ---------------------------------------------------------------------------
# A3 — identity swap
# ---------------------------------------------------------------------------

def attack_identity_swap(apbs: list[APB], registry: PrincipalRegistry) -> dict:
    detected = missed = 0
    for apb in apbs:
        swapped_D_h = HumanDecisionBlock(
            H_id="H_bob",                   # swapped
            decision=apb.D_h.decision,
            rationale=apb.D_h.rationale,
            scope=apb.D_h.scope,
        )
        swapped = APB(E_s=apb.E_s, D_h=swapped_D_h, sigma_h=apb.sigma_h)
        r = verify_apb(swapped, registry, max_age_seconds=86400.0)
        if r.is_valid:
            missed += 1
        else:
            detected += 1
    return {"detected": detected, "missed": missed}


# ---------------------------------------------------------------------------
# A4 — replay
# ---------------------------------------------------------------------------

def attack_replay(apbs: list[APB], registry: PrincipalRegistry,
                  delay_seconds: float = MAX_AGE + 60.0) -> dict:
    detected = missed = 0
    for apb in apbs:
        t_e_dt = datetime.fromisoformat(apb.E_s.t_e)
        replay_now = (t_e_dt + timedelta(seconds=delay_seconds)).isoformat()
        r = verify_apb(apb, registry, now=replay_now, max_age_seconds=MAX_AGE)
        if r.result is VerificationResult.REPLAY:
            detected += 1
        else:
            missed += 1
    return {"detected": detected, "missed": missed}


# ---------------------------------------------------------------------------
# Sanity: untampered APBs should verify cleanly
# ---------------------------------------------------------------------------

def sanity_baseline(apbs: list[APB], registry: PrincipalRegistry) -> dict:
    valid = invalid = 0
    for apb in apbs:
        r = verify_apb(apb, registry, max_age_seconds=86400.0)
        if r.is_valid:
            valid += 1
        else:
            invalid += 1
    return {"valid": valid, "invalid": invalid}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 64)
    print("Exp B -- APB Integrity (T8.2 + T8.3)")
    print(f"  N_APBS: {N_APBS}")
    print(f"  Max age (replay window): {MAX_AGE}s")
    print("=" * 64)

    registry, key_store = setup_principals()
    apbs = generate_apbs(N_APBS, registry, key_store)
    print(f"\nGenerated {len(apbs)} fresh APBs signed by H_alice.")

    # Sanity: untampered should pass
    base = sanity_baseline(apbs, registry)
    print(f"\nBaseline (no attack): {base['valid']}/{N_APBS} verify cleanly")
    assert base["valid"] == N_APBS, \
        "untampered APBs must all verify (T8.2 sanity)"

    # A1
    print("\n[A1] Tamper E_s field-by-field...")
    a1 = attack_tamper_E_s(apbs, registry)
    for f, v in a1["per_field"].items():
        print(f"   {f:<12s}  detected={v['detected']:4d}  missed={v['missed']}")
    print(f"   TOTAL        detected={a1['total']['detected']:4d}  "
          f"missed={a1['total']['missed']}")

    # A2
    print("\n[A2a] Forge with random bytes...")
    a2a = attack_forge_random_bytes(apbs, registry)
    print(f"   detected={a2a['detected']:4d}  missed={a2a['missed']}")
    print("[A2b] Forge with attacker keypair...")
    a2b = attack_forge_attacker_key(apbs, registry)
    print(f"   detected={a2b['detected']:4d}  missed={a2b['missed']}")

    # A3
    print("\n[A3] Identity swap (H_alice -> H_bob, keep sigma_h)...")
    a3 = attack_identity_swap(apbs, registry)
    print(f"   detected={a3['detected']:4d}  missed={a3['missed']}")

    # A4
    print(f"\n[A4] Replay (verify at t_e + {MAX_AGE + 60.0:.0f}s)...")
    a4 = attack_replay(apbs, registry)
    print(f"   detected={a4['detected']:4d}  missed={a4['missed']}")

    # Aggregate
    total_attacks = (
        a1["total"]["detected"] + a1["total"]["missed"]
        + a2a["detected"] + a2a["missed"]
        + a2b["detected"] + a2b["missed"]
        + a3["detected"] + a3["missed"]
        + a4["detected"] + a4["missed"]
    )
    total_detected = (
        a1["total"]["detected"] + a2a["detected"] + a2b["detected"]
        + a3["detected"] + a4["detected"]
    )
    total_missed = total_attacks - total_detected
    detection_rate = 100.0 * total_detected / total_attacks if total_attacks else 0.0

    t_8_2_passed = (a1["total"]["missed"] == 0 and a3["missed"] == 0)
    t_8_3_passed = (a2a["missed"] == 0 and a2b["missed"] == 0)

    summary = {
        "experiment": "p8_exp_b_apb_integrity",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_apbs": N_APBS,
        "max_age_seconds": MAX_AGE,
        "baseline_verification": base,
        "attacks": {
            "A1_tamper_E_s": a1,
            "A2a_forge_random_bytes": a2a,
            "A2b_forge_attacker_key": a2b,
            "A3_identity_swap": a3,
            "A4_replay": a4,
        },
        "totals": {
            "attacks": total_attacks,
            "detected": total_detected,
            "missed": total_missed,
            "detection_rate_pct": round(detection_rate, 4),
        },
        "T8_2_assertion": "PASSED" if t_8_2_passed else "FAILED",
        "T8_3_assertion": "PASSED" if t_8_3_passed else "FAILED",
    }

    # Write outputs
    summary_path = os.path.join(OUT_DIR, "exp_b_summary.json")
    table_path = os.path.join(OUT_DIR, "exp_b_table.tex")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _write_latex_table(summary, table_path)

    # ── Final report ────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("AGGREGATE")
    print(f"  Total attacks:    {total_attacks}")
    print(f"  Detected:         {total_detected}")
    print(f"  Missed:           {total_missed}")
    print(f"  Detection rate:   {detection_rate:.4f}%")
    print(f"\n  T8.2 (Non-Repudiability):           "
          f"{'PASSED' if t_8_2_passed else 'FAILED'}")
    print(f"  T8.3 (Impossibility of Anon Re-Auth): "
          f"{'PASSED' if t_8_3_passed else 'FAILED'}")
    print("=" * 64)
    print(f"\nOutputs:\n  {summary_path}\n  {table_path}")

    return 0 if (t_8_2_passed and t_8_3_passed and total_missed == 0) else 1


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def _detection_rate(detected: int, missed: int) -> str:
    n = detected + missed
    if n == 0:
        return "n/a"
    return "100.00" if missed == 0 else f"{100.0 * detected / n:.2f}"


def _write_latex_table(summary: dict, path: str) -> None:
    a = summary["attacks"]
    rows = []
    a1_per = a["A1_tamper_E_s"]["per_field"]
    for f in E_S_FIELDS:
        v = a1_per[f]
        n = v["detected"] + v["missed"]
        rate = _detection_rate(v["detected"], v["missed"])
        field_tex = f.replace("_", r"\_")
        rows.append(
            f"  Tamper $E_s$.{field_tex} & {n} & {v['detected']} "
            f"& {v['missed']} & {rate}\\% \\\\"
        )
    rows.append("  \\midrule")
    other = [
        ("Forge $\\sigma_h$ (random bytes)", a["A2a_forge_random_bytes"]),
        ("Forge $\\sigma_h$ (attacker key)", a["A2b_forge_attacker_key"]),
        ("Identity swap ($H_i$ replaced)", a["A3_identity_swap"]),
        (f"Replay (at $t_e + {summary['max_age_seconds']+60:.0f}$\\,s)", a["A4_replay"]),
    ]
    for name, v in other:
        n = v["detected"] + v["missed"]
        rate = _detection_rate(v["detected"], v["missed"])
        rows.append(f"  {name} & {n} & {v['detected']} & {v['missed']} & {rate}\\% \\\\")
    rows.append("  \\midrule")
    t = summary["totals"]
    rows.append(
        f"  \\textbf{{Total}} & \\textbf{{{t['attacks']}}} & "
        f"\\textbf{{{t['detected']}}} & \\textbf{{{t['missed']}}} & "
        f"\\textbf{{{t['detection_rate_pct']:.2f}\\%}} \\\\"
    )
    body = "\n".join(rows)
    tex = f"""% Auto-generated by exp_b_apb_integrity.py
\\begin{{table}}[t]
\\centering
\\caption{{APB Integrity (T8.2 + T8.3): four attack vectors against
{summary['n_apbs']} freshly-signed APBs. All tampering attempts are detected
by the verifier. T8.2: {summary['T8_2_assertion']}. T8.3: {summary['T8_3_assertion']}.}}
\\label{{tab:exp_b_integrity}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Attack vector & Trials & Detected & Missed & Detection rate \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)


if __name__ == "__main__":
    sys.exit(main())
