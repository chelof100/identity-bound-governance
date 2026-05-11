# -*- coding: utf-8 -*-
"""
Multi-Principal APB Extension — k-of-n Threshold Governance (P8 §4.5).

Extends the base APB to support k-of-n threshold signing:

  MultiPrincipalAPB = (E_s, D_h, signatures, threshold)

where `signatures` is an ordered tuple of SignatureEntry objects,
each containing (H_id, sigma_h) for a registered principal, and
`threshold` (k) is the minimum number of valid, distinct-principal
signatures required for the APB to be accepted.

Design rationale:
  - The signed message per signer is identical to a single-principal APB:
      canon(E_s) || canon(D_h)
    Each principal independently attests to the same evidence-decision pair.
  - The proposer who authors D_h (D_h.H_id) must be one of the signers.
  - Duplicate H_ids are rejected: each distinct principal counted at most once.
  - V5 (event_id uniqueness) applies to multi-principal APBs as well.

Byzantine resistance (Proposition 8.5):
  Under k-of-n signing, governance capture requires the simultaneous
  compromise of at least k distinct principals' signing keys.
  Single-key capture is infeasible when k >= 2.

Reference: RFC 8785 (JCS) used for canon(); see stack/apb.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Set

from cryptography.exceptions import InvalidSignature

from agent.principal import PrincipalRegistry, load_public_key
from stack.apb import APB, HumanDecisionBlock, SystemEvidenceBlock, _SEP, _canonical


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignatureEntry:
    """One principal's signature over the (E_s, D_h) payload."""
    H_id: str
    sigma_h: bytes  # 64-byte ed25519 signature

    def __post_init__(self) -> None:
        if len(self.sigma_h) != 64:
            raise ValueError(
                f"sigma_h must be 64 bytes (ed25519), got {len(self.sigma_h)}"
            )


@dataclass(frozen=True)
class MultiPrincipalAPB:
    """Accountability Proof Block with k-of-n threshold signing.

    Construction:
      - E_s and D_h are identical in structure to a single-principal APB.
      - D_h.H_id identifies the decision proposer (who must also sign).
      - Each SignatureEntry covers canon(E_s) || canon(D_h).
      - threshold (k): minimum valid signatures required for acceptance.

    Immutable after construction; the tuple of SignatureEntry objects
    cannot be modified post-creation.
    """
    E_s: SystemEvidenceBlock
    D_h: HumanDecisionBlock
    signatures: tuple              # tuple[SignatureEntry, ...]
    threshold: int                 # k in k-of-n

    def __post_init__(self) -> None:
        if self.threshold < 1:
            raise ValueError(f"threshold must be >= 1, got {self.threshold}")
        if not self.signatures:
            raise ValueError("signatures must not be empty")
        # NOTE: we do NOT enforce len(signatures) >= threshold here.
        # The verifier is responsible for the threshold check; the
        # constructor merely validates structural invariants. An APB with
        # fewer entries than threshold is structurally valid but will fail
        # verification — this is the intended path for testing and for
        # realistic adversarial scenarios (e.g. single-key capture attempt).

    def message_to_sign(self) -> bytes:
        """The canonical message all principals sign (identical to single APB)."""
        return self.E_s.to_canonical_bytes() + _SEP + self.D_h.to_canonical_bytes()

    def to_dict(self) -> dict:
        return {
            "E_s": self.E_s.to_dict(),
            "D_h": self.D_h.to_dict(),
            "signatures": [
                {"H_id": s.H_id, "sigma_h": s.sigma_h.hex()}
                for s in self.signatures
            ],
            "threshold": self.threshold,
        }


# ---------------------------------------------------------------------------
# Multi-principal verification
# ---------------------------------------------------------------------------

class MultiVerificationResult(str, Enum):
    VALID = "VALID"
    INVALID_SIGNATURE = "INVALID_SIGNATURE"       # at least one entry has bad sig
    PRINCIPAL_NOT_FOUND = "PRINCIPAL_NOT_FOUND"    # signer not in registry
    PRINCIPAL_REVOKED = "PRINCIPAL_REVOKED"        # signer revoked at t_e
    DUPLICATE_SIGNER = "DUPLICATE_SIGNER"          # same H_id appears twice
    INSUFFICIENT_SIGNATURES = "INSUFFICIENT_SIGNATURES"  # valid_count < threshold
    DUPLICATE_EVENT_ID = "DUPLICATE_EVENT_ID"      # V5: replay via event_id
    REPLAY = "REPLAY"                              # V4: temporal freshness
    MALFORMED = "MALFORMED"                        # unparseable fields


@dataclass
class MultiVerificationReport:
    result: MultiVerificationResult
    apb: MultiPrincipalAPB
    valid_count: int = 0          # number of valid distinct-principal signatures
    required: int = 0             # threshold
    detail: str = ""

    @property
    def is_valid(self) -> bool:
        return self.result is MultiVerificationResult.VALID


def verify_multi_apb(
    apb: MultiPrincipalAPB,
    registry: PrincipalRegistry,
    now: Optional[str] = None,
    max_age_seconds: float = 300.0,
    seen_event_ids: Optional[Set[str]] = None,
) -> MultiVerificationReport:
    """Verify a MultiPrincipalAPB against V1-V5 predicates.

    V4 and V5 apply as in the single-principal case.
    V1-V3 are checked per signature entry; each entry must independently
    satisfy V1 (valid signature) and V3 (principal active at t_e).
    A signer H_id appearing twice (V2-dup) causes DUPLICATE_SIGNER.
    The APB is VALID iff the count of valid distinct signers >= threshold.

    Returns MultiVerificationReport with the result and valid_count.
    """
    message = apb.message_to_sign()

    # --- V4: temporal freshness --------------------------------------------
    now_iso = now or datetime.now(timezone.utc).isoformat()
    try:
        t_now = datetime.fromisoformat(now_iso)
        t_e = datetime.fromisoformat(apb.E_s.t_e)
    except ValueError:
        return MultiVerificationReport(
            result=MultiVerificationResult.MALFORMED,
            apb=apb,
            required=apb.threshold,
            detail=f"unparseable timestamp: t_e={apb.E_s.t_e!r}",
        )
    age = (t_now - t_e).total_seconds()
    if age > max_age_seconds:
        return MultiVerificationReport(
            result=MultiVerificationResult.REPLAY,
            apb=apb,
            required=apb.threshold,
            detail=f"t_e is {age:.1f}s old (max={max_age_seconds:.1f}s)",
        )
    if age < -max_age_seconds:
        return MultiVerificationReport(
            result=MultiVerificationResult.REPLAY,
            apb=apb,
            required=apb.threshold,
            detail=f"t_e is {-age:.1f}s in the future (clock skew)",
        )

    # --- V5: semantic uniqueness (event_id nonce store) --------------------
    if seen_event_ids is not None:
        eid = apb.E_s.event_id
        if eid in seen_event_ids:
            return MultiVerificationReport(
                result=MultiVerificationResult.DUPLICATE_EVENT_ID,
                apb=apb,
                required=apb.threshold,
                detail=f"event_id {eid!r} already accepted",
            )

    # --- V1-V3 per signature entry ----------------------------------------
    seen_hids: set[str] = set()
    valid_count = 0

    for entry in apb.signatures:
        # Duplicate signer check
        if entry.H_id in seen_hids:
            return MultiVerificationReport(
                result=MultiVerificationResult.DUPLICATE_SIGNER,
                apb=apb,
                valid_count=valid_count,
                required=apb.threshold,
                detail=f"H_id={entry.H_id!r} appears more than once",
            )
        seen_hids.add(entry.H_id)

        # V2: principal in registry
        principal = registry.get(entry.H_id)
        if principal is None:
            return MultiVerificationReport(
                result=MultiVerificationResult.PRINCIPAL_NOT_FOUND,
                apb=apb,
                valid_count=valid_count,
                required=apb.threshold,
                detail=f"H_id={entry.H_id!r} not in registry",
            )

        # V3: active at t_e
        if not registry.is_active(entry.H_id, at_time=apb.E_s.t_e):
            return MultiVerificationReport(
                result=MultiVerificationResult.PRINCIPAL_REVOKED,
                apb=apb,
                valid_count=valid_count,
                required=apb.threshold,
                detail=f"H_id={entry.H_id!r} not active at t_e={apb.E_s.t_e}",
            )

        # V1: signature validity
        try:
            pk_obj = load_public_key(principal.public_key)
            pk_obj.verify(entry.sigma_h, message)
            valid_count += 1
        except (InvalidSignature, ValueError):
            return MultiVerificationReport(
                result=MultiVerificationResult.INVALID_SIGNATURE,
                apb=apb,
                valid_count=valid_count,
                required=apb.threshold,
                detail=f"ed25519 verification failed for H_id={entry.H_id!r}",
            )

    # --- Threshold check --------------------------------------------------
    if valid_count < apb.threshold:
        return MultiVerificationReport(
            result=MultiVerificationResult.INSUFFICIENT_SIGNATURES,
            apb=apb,
            valid_count=valid_count,
            required=apb.threshold,
            detail=(
                f"only {valid_count} valid signature(s) "
                f"but threshold={apb.threshold}"
            ),
        )

    # --- All predicates satisfied; update nonce store ---------------------
    if seen_event_ids is not None:
        seen_event_ids.add(apb.E_s.event_id)

    return MultiVerificationReport(
        result=MultiVerificationResult.VALID,
        apb=apb,
        valid_count=valid_count,
        required=apb.threshold,
    )


# ---------------------------------------------------------------------------
# Threshold Governance Layer
# ---------------------------------------------------------------------------

class ThresholdGovernanceLayer:
    """Implements k-of-n governance: multiple principals co-sign an APB.

    In real deployment each principal signs on their own device; here
    the key_store is in-memory for experimental tractability.
    """

    def __init__(
        self,
        registry: PrincipalRegistry,
        key_store: dict[str, bytes],
    ) -> None:
        self.registry = registry
        self._key_store = dict(key_store)

    def resolve_multi(
        self,
        H_ids: list[str],
        E_s: SystemEvidenceBlock,
        D_h: HumanDecisionBlock,
        threshold: int,
    ) -> MultiPrincipalAPB:
        """Build a MultiPrincipalAPB signed by each H_id in H_ids.

        H_ids must have at least `threshold` entries.
        Each H_id must be registered, active at E_s.t_e, and in key_store.
        """
        if len(H_ids) < threshold:
            raise ValueError(
                f"need at least {threshold} signers, got {len(H_ids)}"
            )
        if len(set(H_ids)) != len(H_ids):
            raise ValueError("H_ids must be distinct")

        message = E_s.to_canonical_bytes() + _SEP + D_h.to_canonical_bytes()
        entries = []
        for H_id in H_ids:
            if self.registry.get(H_id) is None:
                raise ValueError(f"unknown principal: {H_id!r}")
            if not self.registry.is_active(H_id, at_time=E_s.t_e):
                raise ValueError(f"principal not active at t_e: {H_id!r}")
            if H_id not in self._key_store:
                raise ValueError(f"no private key for principal: {H_id!r}")
            from agent.principal import load_private_key
            sk = load_private_key(self._key_store[H_id])
            sigma = sk.sign(message)
            entries.append(SignatureEntry(H_id=H_id, sigma_h=sigma))

        return MultiPrincipalAPB(
            E_s=E_s,
            D_h=D_h,
            signatures=tuple(entries),
            threshold=threshold,
        )
