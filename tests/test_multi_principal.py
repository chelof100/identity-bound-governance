# -*- coding: utf-8 -*-
"""Tests for stack/multi_principal.py — k-of-n threshold governance."""
import pytest
from datetime import datetime, timedelta, timezone

from agent.principal import Principal, PrincipalRegistry, generate_keypair
from stack.apb import (
    HumanDecisionBlock,
    SystemEvidenceBlock,
    GovernanceDecision,
)
from stack.multi_principal import (
    MultiPrincipalAPB,
    MultiVerificationResult,
    SignatureEntry,
    ThresholdGovernanceLayer,
    verify_multi_apb,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def alice():
    sk, pk = generate_keypair()
    return {"H_id": "H_alice", "sk": sk, "pk": pk}


@pytest.fixture
def bob():
    sk, pk = generate_keypair()
    return {"H_id": "H_bob", "sk": sk, "pk": pk}


@pytest.fixture
def carol():
    sk, pk = generate_keypair()
    return {"H_id": "H_carol", "sk": sk, "pk": pk}


@pytest.fixture
def registry(alice, bob, carol):
    reg = PrincipalRegistry()
    reg.add(Principal(H_id=alice["H_id"], public_key=alice["pk"]))
    reg.add(Principal(H_id=bob["H_id"], public_key=bob["pk"]))
    reg.add(Principal(H_id=carol["H_id"], public_key=carol["pk"]))
    return reg


@pytest.fixture
def key_store(alice, bob, carol):
    return {
        alice["H_id"]: alice["sk"],
        bob["H_id"]: bob["sk"],
        carol["H_id"]: carol["sk"],
    }


@pytest.fixture
def gov(registry, key_store):
    return ThresholdGovernanceLayer(registry=registry, key_store=key_store)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mk_E_s() -> SystemEvidenceBlock:
    return SystemEvidenceBlock(
        A_0_hash="a" * 64,
        D_hat=0.35,
        t_e=_now_iso(),
        trace_hash="b" * 64,
        cause="persistent_drift",
        # event_id auto-generated (UUID4)
    )


def _mk_D_h(H_id: str = "H_alice") -> HumanDecisionBlock:
    return HumanDecisionBlock(
        H_id=H_id,
        decision=GovernanceDecision.RESUME.value,
        rationale="manual review",
        scope="next 100 steps",
    )


# ---------------------------------------------------------------------------
# SignatureEntry validation
# ---------------------------------------------------------------------------

def test_signature_entry_rejects_short_sigma():
    with pytest.raises(ValueError, match="64 bytes"):
        SignatureEntry(H_id="H_alice", sigma_h=b"\x00" * 32)


def test_signature_entry_accepts_64_bytes():
    entry = SignatureEntry(H_id="H_alice", sigma_h=b"\x00" * 64)
    assert entry.H_id == "H_alice"


# ---------------------------------------------------------------------------
# MultiPrincipalAPB construction
# ---------------------------------------------------------------------------

def test_threshold_layer_k1_of_3(gov, alice):
    E_s = _mk_E_s()
    D_h = _mk_D_h(alice["H_id"])
    apb = gov.resolve_multi([alice["H_id"]], E_s, D_h, threshold=1)
    assert len(apb.signatures) == 1
    assert apb.threshold == 1


def test_threshold_layer_k2_of_3(gov, alice, bob):
    E_s = _mk_E_s()
    D_h = _mk_D_h(alice["H_id"])
    apb = gov.resolve_multi([alice["H_id"], bob["H_id"]], E_s, D_h, threshold=2)
    assert len(apb.signatures) == 2
    assert apb.threshold == 2


def test_threshold_layer_k3_of_3(gov, alice, bob, carol):
    E_s = _mk_E_s()
    D_h = _mk_D_h(alice["H_id"])
    apb = gov.resolve_multi(
        [alice["H_id"], bob["H_id"], carol["H_id"]], E_s, D_h, threshold=3
    )
    assert len(apb.signatures) == 3
    assert apb.threshold == 3


def test_threshold_layer_rejects_insufficient_signers(gov, alice):
    E_s = _mk_E_s()
    D_h = _mk_D_h(alice["H_id"])
    with pytest.raises(ValueError, match="at least 2 signers"):
        gov.resolve_multi([alice["H_id"]], E_s, D_h, threshold=2)


def test_threshold_layer_rejects_duplicate_hids(gov, alice):
    E_s = _mk_E_s()
    D_h = _mk_D_h(alice["H_id"])
    with pytest.raises(ValueError, match="distinct"):
        gov.resolve_multi(
            [alice["H_id"], alice["H_id"]], E_s, D_h, threshold=2
        )


# ---------------------------------------------------------------------------
# Happy-path verification
# ---------------------------------------------------------------------------

def test_verify_k1_valid(gov, alice, registry):
    E_s = _mk_E_s()
    D_h = _mk_D_h(alice["H_id"])
    apb = gov.resolve_multi([alice["H_id"]], E_s, D_h, threshold=1)
    r = verify_multi_apb(apb, registry, max_age_seconds=600.0)
    assert r.is_valid
    assert r.valid_count == 1


def test_verify_k2_valid(gov, alice, bob, registry):
    E_s = _mk_E_s()
    D_h = _mk_D_h(alice["H_id"])
    apb = gov.resolve_multi([alice["H_id"], bob["H_id"]], E_s, D_h, threshold=2)
    r = verify_multi_apb(apb, registry, max_age_seconds=600.0)
    assert r.is_valid
    assert r.valid_count == 2


def test_verify_k3_valid(gov, alice, bob, carol, registry):
    E_s = _mk_E_s()
    D_h = _mk_D_h(alice["H_id"])
    apb = gov.resolve_multi(
        [alice["H_id"], bob["H_id"], carol["H_id"]], E_s, D_h, threshold=3
    )
    r = verify_multi_apb(apb, registry, max_age_seconds=600.0)
    assert r.is_valid
    assert r.valid_count == 3


# ---------------------------------------------------------------------------
# Threshold enforcement
# ---------------------------------------------------------------------------

def test_k2_with_only_1_sig_is_insufficient(gov, alice, registry):
    """1 valid sig but threshold=2 → INSUFFICIENT_SIGNATURES."""
    E_s = _mk_E_s()
    D_h = _mk_D_h(alice["H_id"])
    # Build k=1 APB but set threshold=2 manually to simulate under-provision
    apb_k1 = gov.resolve_multi([alice["H_id"]], E_s, D_h, threshold=1)
    # Reconstruct with threshold=2 but only 1 signature
    under_provisioned = MultiPrincipalAPB(
        E_s=apb_k1.E_s,
        D_h=apb_k1.D_h,
        signatures=apb_k1.signatures,  # only 1 entry
        threshold=2,
    )
    r = verify_multi_apb(under_provisioned, registry, max_age_seconds=600.0)
    assert r.result is MultiVerificationResult.INSUFFICIENT_SIGNATURES
    assert r.valid_count == 1
    assert r.required == 2


def test_k3_with_2_sigs_is_insufficient(gov, alice, bob, registry):
    """2 valid sigs but threshold=3 → INSUFFICIENT_SIGNATURES."""
    E_s = _mk_E_s()
    D_h = _mk_D_h(alice["H_id"])
    apb_k2 = gov.resolve_multi([alice["H_id"], bob["H_id"]], E_s, D_h, threshold=2)
    under_provisioned = MultiPrincipalAPB(
        E_s=apb_k2.E_s,
        D_h=apb_k2.D_h,
        signatures=apb_k2.signatures,
        threshold=3,
    )
    r = verify_multi_apb(under_provisioned, registry, max_age_seconds=600.0)
    assert r.result is MultiVerificationResult.INSUFFICIENT_SIGNATURES
    assert r.valid_count == 2
    assert r.required == 3


# ---------------------------------------------------------------------------
# Byzantine resistance: single-key capture fails under k>=2
# ---------------------------------------------------------------------------

def test_byzantine_single_forged_sig_fails_k2(gov, alice, bob, registry):
    """Attacker controls 1 key (alice). Under k=2, single-principal APB rejected."""
    E_s = _mk_E_s()
    D_h = _mk_D_h(alice["H_id"])
    # Attacker signs only with alice's key (the compromised one)
    apb_single = gov.resolve_multi([alice["H_id"]], E_s, D_h, threshold=1)
    # But the policy requires k=2
    under_threshold = MultiPrincipalAPB(
        E_s=apb_single.E_s,
        D_h=apb_single.D_h,
        signatures=apb_single.signatures,
        threshold=2,
    )
    r = verify_multi_apb(under_threshold, registry, max_age_seconds=600.0)
    assert r.result is MultiVerificationResult.INSUFFICIENT_SIGNATURES


def test_byzantine_forged_second_sig_fails(gov, alice, bob, registry):
    """Attacker forges the second signature with a non-registered key.

    Under k=2: alice signs legitimately, attacker forges bob's sig with
    a random key. The forged entry fails V1 → INVALID_SIGNATURE.
    """
    E_s = _mk_E_s()
    D_h = _mk_D_h(alice["H_id"])
    # Alice's legitimate signature
    message = E_s.to_canonical_bytes()
    from stack.apb import _SEP
    message = E_s.to_canonical_bytes() + _SEP + D_h.to_canonical_bytes()
    from agent.principal import load_private_key
    alice_sig = load_private_key(gov._key_store[alice["H_id"]]).sign(message)

    # Attacker signs with a random key but claims it's bob
    fake_sk, _ = generate_keypair()
    fake_sig = load_private_key(fake_sk).sign(message)

    apb = MultiPrincipalAPB(
        E_s=E_s,
        D_h=D_h,
        signatures=(
            SignatureEntry(H_id=alice["H_id"], sigma_h=alice_sig),
            SignatureEntry(H_id=bob["H_id"], sigma_h=fake_sig),  # forged
        ),
        threshold=2,
    )
    r = verify_multi_apb(apb, registry, max_age_seconds=600.0)
    assert r.result is MultiVerificationResult.INVALID_SIGNATURE


def test_byzantine_duplicate_signer_fails(gov, alice, registry):
    """Attacker tries to count alice's signature twice to meet threshold=2."""
    E_s = _mk_E_s()
    D_h = _mk_D_h(alice["H_id"])
    message = E_s.to_canonical_bytes()
    from stack.apb import _SEP
    from agent.principal import load_private_key
    message = E_s.to_canonical_bytes() + _SEP + D_h.to_canonical_bytes()
    alice_sig = load_private_key(gov._key_store[alice["H_id"]]).sign(message)

    apb = MultiPrincipalAPB(
        E_s=E_s,
        D_h=D_h,
        signatures=(
            SignatureEntry(H_id=alice["H_id"], sigma_h=alice_sig),
            SignatureEntry(H_id=alice["H_id"], sigma_h=alice_sig),  # duplicate
        ),
        threshold=2,
    )
    r = verify_multi_apb(apb, registry, max_age_seconds=600.0)
    assert r.result is MultiVerificationResult.DUPLICATE_SIGNER


# ---------------------------------------------------------------------------
# V4: temporal replay protection
# ---------------------------------------------------------------------------

def test_multi_apb_replay_old_rejected(gov, alice, registry):
    old_t_e = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    E_s = SystemEvidenceBlock(
        A_0_hash="a" * 64, D_hat=0.35, t_e=old_t_e,
        trace_hash="b" * 64, cause="persistent_drift",
    )
    D_h = _mk_D_h(alice["H_id"])
    apb = gov.resolve_multi([alice["H_id"]], E_s, D_h, threshold=1)
    r = verify_multi_apb(apb, registry, max_age_seconds=300.0)
    assert r.result is MultiVerificationResult.REPLAY


# ---------------------------------------------------------------------------
# V5: semantic uniqueness (event_id nonce store)
# ---------------------------------------------------------------------------

def test_multi_apb_duplicate_event_id_rejected(gov, alice, registry):
    E_s = _mk_E_s()
    D_h = _mk_D_h(alice["H_id"])
    apb = gov.resolve_multi([alice["H_id"]], E_s, D_h, threshold=1)
    seen: set = set()
    r1 = verify_multi_apb(apb, registry, max_age_seconds=600.0, seen_event_ids=seen)
    assert r1.is_valid
    r2 = verify_multi_apb(apb, registry, max_age_seconds=600.0, seen_event_ids=seen)
    assert r2.result is MultiVerificationResult.DUPLICATE_EVENT_ID


# ---------------------------------------------------------------------------
# Principal lifecycle
# ---------------------------------------------------------------------------

def test_multi_apb_unknown_principal_rejected(alice):
    reg = PrincipalRegistry()  # empty
    sk, pk = generate_keypair()
    reg.add(Principal(H_id="H_other", public_key=pk))
    ks = {"H_other": sk}
    gov_empty = ThresholdGovernanceLayer(registry=reg, key_store=ks)
    E_s = _mk_E_s()
    D_h = _mk_D_h("H_other")
    apb = gov_empty.resolve_multi(["H_other"], E_s, D_h, threshold=1)
    # Verify against a registry that doesn't have H_other
    empty_reg = PrincipalRegistry()
    r = verify_multi_apb(apb, empty_reg, max_age_seconds=600.0)
    assert r.result is MultiVerificationResult.PRINCIPAL_NOT_FOUND
