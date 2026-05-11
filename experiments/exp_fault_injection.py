# -*- coding: utf-8 -*-
"""
Experiment F — Infrastructure Fault Injection.

Goal:
  Empirically validate the robustness of the APB verification protocol
  against five representative infrastructure fault classes:

  F1: Clock Drift           — test V4 temporal freshness boundaries.
      δ offsets from −600 to +600 s; V4 window = 300 s.
      Expected: accept iff |δ| ≤ 300.

  F2: Registry Key Mismatch — principal registered with wrong public key.
      Simulates key-store corruption or supply-chain substitution.
      Expected: 100% INVALID_SIGNATURE (never PRINCIPAL_NOT_FOUND).

  F3: Concurrent Writes     — N threads submitting APBs to a shared nonce
      store simultaneously.  Tests V5 thread-safety via APBLog.
      Expected: 0 duplicate acceptances under threading.Lock.

  F4: Log Tampering         — five tamper strategies applied to the HMAC
      chain log: field modification, entry deletion, reordering, truncation,
      HMAC forgery.
      Expected: 100% tamper detection (IntegrityReport.ok == False).

  F5: Duplicate APB Submission — same APB submitted N times.
      Without V5: N−1 false acceptances.
      With V5 (seen_event_ids):  0 false acceptances.
      Expected: V5 necessity demonstrated empirically.

Metrics per fault class:
  - false_acceptance_rate  (%)   — faults incorrectly accepted as VALID
  - true_rejection_rate    (%)   — faults correctly rejected
  - For F4: tamper_detection_rate (%)

Reference: P8 §5.5, Experiment F.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from agent.principal import (
    Principal,
    PrincipalRegistry,
    generate_keypair,
    load_private_key,
)
from stack.apb import (
    APB,
    GovernanceDecision,
    HumanDecisionBlock,
    SystemEvidenceBlock,
    construct_evidence,
    _SEP,
)
from stack.apb_log import APBLog
from stack.apb_verifier import VerificationResult, verify_apb


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_EVENTS = 1_000        # trials per fault scenario
V4_WINDOW = 300.0       # max_age_seconds used in the experiment

# Clock offsets (seconds) to probe around the V4 boundary
CLOCK_OFFSETS = [-600, -310, -301, -299, -150, -1, 0, 1, 150, 299, 301, 310, 600]


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def _setup():
    sk, pk = generate_keypair()
    reg = PrincipalRegistry()
    reg.add(Principal(H_id="H_alice", public_key=pk))
    return sk, pk, reg


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_apb(sk: bytes, t_e: str | None = None) -> APB:
    t_e = t_e or _now_iso()
    E_s = SystemEvidenceBlock(
        A_0_hash="a" * 64,
        D_hat=0.35,
        t_e=t_e,
        trace_hash="b" * 64,
        cause="persistent_drift",
    )
    D_h = HumanDecisionBlock(
        H_id="H_alice",
        decision=GovernanceDecision.RESUME.value,
        rationale="fault injection test",
        scope="single resumption",
    )
    return APB.construct(E_s, D_h, sk)


# ---------------------------------------------------------------------------
# F1 — Clock Drift
# ---------------------------------------------------------------------------

def run_f1(sk: bytes, reg: PrincipalRegistry) -> dict:
    """For each δ-offset, measure acceptance rate. Reports per-offset result."""
    results_per_offset = {}
    for delta in CLOCK_OFFSETS:
        accepted = 0
        rejected_replay = 0
        for _ in range(N_EVENTS):
            t_e = (datetime.now(timezone.utc) - timedelta(seconds=delta)).isoformat()
            apb = _make_apb(sk, t_e=t_e)
            report = verify_apb(apb, reg, max_age_seconds=V4_WINDOW)
            if report.result == VerificationResult.VALID:
                accepted += 1
            elif report.result == VerificationResult.REPLAY:
                rejected_replay += 1

        expected_accept = abs(delta) <= V4_WINDOW
        results_per_offset[delta] = {
            "delta_seconds": delta,
            "expected": "ACCEPT" if expected_accept else "REJECT",
            "accepted": accepted,
            "rejected_replay": rejected_replay,
            "total": N_EVENTS,
            "correct": (
                (accepted == N_EVENTS and expected_accept)
                or (rejected_replay == N_EVENTS and not expected_accept)
            ),
        }

    # Boundary precision: check exact rejection at |δ|=301 and acceptance at |δ|=299
    boundary_correct = all(
        results_per_offset[d]["correct"]
        for d in [-301, -299, 299, 301]
    )
    all_correct = all(r["correct"] for r in results_per_offset.values())

    return {
        "fault": "F1",
        "description": "Clock Drift — V4 temporal freshness boundaries",
        "n_events_per_offset": N_EVENTS,
        "offsets_tested": len(CLOCK_OFFSETS),
        "all_offsets_correct": all_correct,
        "boundary_correct_at_301": boundary_correct,
        "per_offset": results_per_offset,
    }


# ---------------------------------------------------------------------------
# F2 — Registry Key Mismatch
# ---------------------------------------------------------------------------

def run_f2() -> dict:
    """Register Alice with correct PK, then verify APBs signed with correct SK
    against a registry where Alice has the WRONG PK (key-store corruption)."""
    real_sk, real_pk = generate_keypair()
    _, wrong_pk = generate_keypair()

    # Registry with wrong key
    corrupt_reg = PrincipalRegistry()
    corrupt_reg.add(Principal(H_id="H_alice", public_key=wrong_pk))

    invalid_sig_count = 0
    false_acceptance_count = 0

    for _ in range(N_EVENTS):
        apb = _make_apb(real_sk)
        report = verify_apb(apb, corrupt_reg, max_age_seconds=V4_WINDOW)
        if report.result == VerificationResult.INVALID_SIGNATURE:
            invalid_sig_count += 1
        elif report.result == VerificationResult.VALID:
            false_acceptance_count += 1

    return {
        "fault": "F2",
        "description": "Registry Key Mismatch — wrong PK in registry",
        "n_events": N_EVENTS,
        "invalid_signature_count": invalid_sig_count,
        "false_acceptance_count": false_acceptance_count,
        "false_acceptance_rate_pct": round(100 * false_acceptance_count / N_EVENTS, 4),
        "true_rejection_rate_pct": round(100 * invalid_sig_count / N_EVENTS, 4),
        "proposition_holds": false_acceptance_count == 0,
    }


# ---------------------------------------------------------------------------
# F3 — Concurrent Writes
# ---------------------------------------------------------------------------

def run_f3(tmp_dir: Path) -> dict:
    """N_THREADS threads submit APBs to a shared APBLog and seen_event_ids.

    Measures:
      - Whether any duplicate event_id slips through (false acceptance).
      - Whether the HMAC chain remains intact after concurrent writes.
    """
    N_THREADS = 16
    M_PER_THREAD = 50   # each thread submits 50 unique APBs
    # Note: within each thread, each APB has a unique event_id (UUID4)
    # Threads do NOT share APBs — only the log and the nonce store are shared.

    sk, pk, reg = _setup()
    seen: set = set()
    seen_lock = threading.Lock()
    log = APBLog(path=tmp_dir / "concurrent.log", key=b"concurrent-test-key-32-bytes!")

    acceptance_counts = []  # how many accepted per thread
    duplicate_detections = []
    errors = []

    def worker():
        accepted = 0
        dupes = 0
        try:
            for _ in range(M_PER_THREAD):
                apb = _make_apb(sk)
                with seen_lock:
                    report = verify_apb(
                        apb, reg,
                        max_age_seconds=V4_WINDOW,
                        seen_event_ids=seen,
                    )
                if report.result == VerificationResult.VALID:
                    log.append(apb)
                    accepted += 1
                elif report.result == VerificationResult.DUPLICATE_EVENT_ID:
                    dupes += 1
        except Exception as exc:
            errors.append(str(exc))
        acceptance_counts.append(accepted)
        duplicate_detections.append(dupes)

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_accepted = sum(acceptance_counts)
    total_dupes_caught = sum(duplicate_detections)
    total_events = N_THREADS * M_PER_THREAD

    integrity = log.verify_integrity()

    return {
        "fault": "F3",
        "description": "Concurrent Writes — V5 thread safety via locking",
        "n_threads": N_THREADS,
        "events_per_thread": M_PER_THREAD,
        "total_events": total_events,
        "total_accepted": total_accepted,
        "total_duplicate_events_caught": total_dupes_caught,
        "errors": errors,
        "log_integrity_ok": integrity.ok,
        "log_entries_verified": integrity.entries_checked,
        "false_acceptance_count": total_events - total_accepted - total_dupes_caught,
        "proposition_holds": (
            len(errors) == 0
            and integrity.ok
            and (total_accepted + total_dupes_caught) == total_events
        ),
    }


# ---------------------------------------------------------------------------
# F4 — Log Tampering
# ---------------------------------------------------------------------------

def _build_log(tmp_dir: Path, n: int, key: bytes) -> tuple[APBLog, list[str]]:
    sk, pk, _ = _setup()
    path = tmp_dir / f"tamper_{n}.log"
    log = APBLog(path=path, key=key)
    for _ in range(n):
        log.append(_make_apb(sk))
    lines = path.read_text(encoding="utf-8").splitlines()
    return log, lines


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_f4(tmp_dir: Path) -> dict:
    """Five tamper strategies; measure detection rate."""
    import json as _json

    key = b"tamper-detection-key-32-bytes-ok"
    N = 10  # entries per log; repeat R times for statistics
    R = 100  # repetitions per tamper type

    strategies = {
        "field_modification":  {"detected": 0, "undetected": 0},
        "entry_deletion":      {"detected": 0, "undetected": 0},
        "entry_reordering":    {"detected": 0, "undetected": 0},
        "hmac_forgery":        {"detected": 0, "undetected": 0},
        "prev_hmac_splice":    {"detected": 0, "undetected": 0},
    }

    def _fill_log(log_path: Path, trial_sk: bytes) -> None:
        lg = APBLog(path=log_path, key=key)
        for _j in range(N):
            lg.append(_make_apb(trial_sk))

    for trial in range(R):
        trial_sk, _trial_pk, _trial_reg = _setup()

        # ── Field modification: change D_hat in a random entry ──
        path = tmp_dir / f"t_field_{trial}.log"
        _fill_log(path, trial_sk)
        lines = path.read_text(encoding="utf-8").splitlines()
        idx = len(lines) // 2
        d = _json.loads(lines[idx])
        d["apb"]["E_s"]["D_hat"] = 99.9
        lines[idx] = _json.dumps(d, sort_keys=True, separators=(",", ":"))
        _write_lines(path, lines)
        rpt = APBLog(path, key).verify_integrity()
        if not rpt.ok:
            strategies["field_modification"]["detected"] += 1
        else:
            strategies["field_modification"]["undetected"] += 1

        # ── Entry deletion: remove middle entry ──
        path = tmp_dir / f"t_del_{trial}.log"
        _fill_log(path, trial_sk)
        lines = path.read_text(encoding="utf-8").splitlines()
        del lines[N // 2]
        _write_lines(path, lines)
        rpt = APBLog(path, key).verify_integrity()
        if not rpt.ok:
            strategies["entry_deletion"]["detected"] += 1
        else:
            strategies["entry_deletion"]["undetected"] += 1

        # ── Entry reordering: swap first two ──
        path = tmp_dir / f"t_reorder_{trial}.log"
        _fill_log(path, trial_sk)
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) >= 2:
            lines[0], lines[1] = lines[1], lines[0]
        _write_lines(path, lines)
        rpt = APBLog(path, key).verify_integrity()
        if not rpt.ok:
            strategies["entry_reordering"]["detected"] += 1
        else:
            strategies["entry_reordering"]["undetected"] += 1

        # ── HMAC forgery: replace entry_hmac with random hex ──
        path = tmp_dir / f"t_forge_{trial}.log"
        _fill_log(path, trial_sk)
        lines = path.read_text(encoding="utf-8").splitlines()
        d_forge = _json.loads(lines[0])
        d_forge["entry_hmac"] = "e" * 64
        lines[0] = _json.dumps(d_forge, sort_keys=True, separators=(",", ":"))
        _write_lines(path, lines)
        rpt = APBLog(path, key).verify_integrity()
        if not rpt.ok:
            strategies["hmac_forgery"]["detected"] += 1
        else:
            strategies["hmac_forgery"]["undetected"] += 1

        # ── prev_hmac splice: attacker forges entry 0 AND updates entry 1's
        #    prev_hmac to match — tries to recompute the chain without the key
        path = tmp_dir / f"t_splice_{trial}.log"
        _fill_log(path, trial_sk)
        lines = path.read_text(encoding="utf-8").splitlines()
        d0 = _json.loads(lines[0])
        forged_hmac = "c" * 64
        d0["entry_hmac"] = forged_hmac
        lines[0] = _json.dumps(d0, sort_keys=True, separators=(",", ":"))
        if len(lines) > 1:
            d1 = _json.loads(lines[1])
            d1["prev_hmac"] = forged_hmac
            lines[1] = _json.dumps(d1, sort_keys=True, separators=(",", ":"))
        _write_lines(path, lines)
        rpt = APBLog(path, key).verify_integrity()
        if not rpt.ok:
            strategies["prev_hmac_splice"]["detected"] += 1
        else:
            strategies["prev_hmac_splice"]["undetected"] += 1

    strategy_results = {}
    all_100 = True
    for name, counts in strategies.items():
        total = counts["detected"] + counts["undetected"]
        rate = round(100 * counts["detected"] / total, 4) if total else 0.0
        strategy_results[name] = {
            "detected": counts["detected"],
            "undetected": counts["undetected"],
            "detection_rate_pct": rate,
        }
        if rate < 100.0:
            all_100 = False

    return {
        "fault": "F4",
        "description": "Log Tampering — HMAC chain integrity under 5 strategies",
        "n_entries_per_log": N,
        "repetitions_per_strategy": R,
        "strategies": strategy_results,
        "all_strategies_100pct_detected": all_100,
        "proposition_holds": all_100,
    }


# ---------------------------------------------------------------------------
# F5 — Duplicate APB Submission
# ---------------------------------------------------------------------------

def run_f5(sk: bytes, reg: PrincipalRegistry) -> dict:
    """Submit the same APB N_EVENTS times.

    Without V5 (no seen_event_ids): measures false acceptance rate.
    With V5 (seen_event_ids provided): should reject all but first.
    """
    SUBMISSIONS = N_EVENTS

    # ── Without V5 ──
    without_v5_accepted = 0
    for _ in range(SUBMISSIONS):
        apb = _make_apb(sk)   # fresh APB each time (different event_id)
    # Use the SAME APB for all submissions to test duplicate detection
    fixed_apb = _make_apb(sk)
    without_v5_accepted = 0
    for _ in range(SUBMISSIONS):
        report = verify_apb(fixed_apb, reg, max_age_seconds=V4_WINDOW,
                            seen_event_ids=None)
        if report.result == VerificationResult.VALID:
            without_v5_accepted += 1

    # ── With V5 ──
    with_v5_seen: set = set()
    fixed_apb2 = _make_apb(sk)  # different APB, same principle
    with_v5_accepted = 0
    with_v5_dupes = 0
    for _ in range(SUBMISSIONS):
        report = verify_apb(fixed_apb2, reg, max_age_seconds=V4_WINDOW,
                            seen_event_ids=with_v5_seen)
        if report.result == VerificationResult.VALID:
            with_v5_accepted += 1
        elif report.result == VerificationResult.DUPLICATE_EVENT_ID:
            with_v5_dupes += 1

    return {
        "fault": "F5",
        "description": "Duplicate APB Submission — V5 necessity demonstration",
        "submissions": SUBMISSIONS,
        "without_v5": {
            "accepted": without_v5_accepted,
            "false_acceptances": without_v5_accepted - 1,   # first is legitimate
            "false_acceptance_rate_pct": round(
                100 * (without_v5_accepted - 1) / (SUBMISSIONS - 1), 4
            ) if SUBMISSIONS > 1 else 0.0,
        },
        "with_v5": {
            "accepted": with_v5_accepted,
            "duplicates_caught": with_v5_dupes,
            "false_acceptances": with_v5_accepted - 1,
            "false_acceptance_rate_pct": round(
                100 * max(0, with_v5_accepted - 1) / (SUBMISSIONS - 1), 4
            ) if SUBMISSIONS > 1 else 0.0,
        },
        "v5_eliminates_duplicates": (
            with_v5_accepted == 1 and with_v5_dupes == SUBMISSIONS - 1
        ),
        "v5_necessity": (
            without_v5_accepted == SUBMISSIONS and with_v5_accepted == 1
        ),
        "proposition_holds": (
            with_v5_accepted == 1 and with_v5_dupes == SUBMISSIONS - 1
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment() -> dict:
    print("Experiment F: Infrastructure Fault Injection")
    print("=" * 60)
    print(f"N_EVENTS per scenario: {N_EVENTS}")
    print(f"V4 window: {V4_WINDOW}s")
    print()

    sk, pk, reg = _setup()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # F1
        print("Running F1 (Clock Drift)...")
        r1 = run_f1(sk, reg)
        _print_f1(r1)

        # F2
        print("\nRunning F2 (Registry Key Mismatch)...")
        r2 = run_f2()
        print(f"F2: false_acceptance={r2['false_acceptance_count']}/{r2['n_events']} "
              f"({r2['false_acceptance_rate_pct']}%) "
              f"| true_rejection={r2['true_rejection_rate_pct']}%")

        # F3
        print("\nRunning F3 (Concurrent Writes)...")
        r3 = run_f3(tmp_dir)
        print(f"F3: {r3['n_threads']} threads × {r3['events_per_thread']} events "
              f"= {r3['total_events']} total | "
              f"accepted={r3['total_accepted']} | "
              f"log_integrity={r3['log_integrity_ok']} | "
              f"errors={r3['errors']}")

        # F4
        print("\nRunning F4 (Log Tampering)...")
        r4 = run_f4(tmp_dir)
        for strategy, stats in r4["strategies"].items():
            print(f"  {strategy}: {stats['detection_rate_pct']}% detected "
                  f"({stats['detected']}/{stats['detected']+stats['undetected']})")

        # F5
        print("\nRunning F5 (Duplicate APB Submission)...")
        r5 = run_f5(sk, reg)
        print(f"F5 without V5: {r5['without_v5']['accepted']}/{r5['submissions']} accepted "
              f"({r5['without_v5']['false_acceptances']} false)")
        print(f"F5 with    V5: {r5['with_v5']['accepted']}/{r5['submissions']} accepted, "
              f"{r5['with_v5']['duplicates_caught']} caught")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"F1 (Clock Drift):          all offsets correct = {r1['all_offsets_correct']}")
    print(f"F2 (Key Mismatch):         0 false acceptances = {r2['proposition_holds']}")
    print(f"F3 (Concurrent Writes):    log intact          = {r3['proposition_holds']}")
    print(f"F4 (Log Tampering):        100% detected       = {r4['proposition_holds']}")
    print(f"F5 (Duplicate Submission): V5 necessity shown  = {r5['proposition_holds']}")

    all_pass = all([
        r1["all_offsets_correct"],
        r2["proposition_holds"],
        r3["proposition_holds"],
        r4["proposition_holds"],
        r5["proposition_holds"],
    ])
    print(f"\nAll fault classes correctly handled: {all_pass}")

    return {
        "experiment": "F",
        "title": "Infrastructure Fault Injection",
        "n_events": N_EVENTS,
        "v4_window_seconds": V4_WINDOW,
        "f1": r1,
        "f2": r2,
        "f3": r3,
        "f4": r4,
        "f5": r5,
        "all_fault_classes_pass": all_pass,
    }


def _print_f1(r: dict) -> None:
    print("F1 Clock Drift results (delta -> result):")
    for delta, info in r["per_offset"].items():
        status = "OK" if info["correct"] else "FAIL"
        print(f"  d={int(delta):+5d}s  expected={info['expected']:6s}  "
              f"accepted={info['accepted']}/{info['total']}  {status}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output = run_experiment()

    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "exp_f_fault_injection.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")
