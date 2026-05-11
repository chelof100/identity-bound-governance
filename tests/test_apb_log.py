# -*- coding: utf-8 -*-
"""Tests for stack/apb_log.py — HMAC-chained JSONL APB log."""
import json
import os
import threading
import tempfile
from pathlib import Path

import pytest

from agent.principal import Principal, PrincipalRegistry, generate_keypair
from stack.apb import APB, GovernanceDecision, HumanDecisionBlock, SystemEvidenceBlock
from stack.apb_log import APBLog, IntegrityReport, LogEntry, _GENESIS_HMAC


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def log_key() -> bytes:
    return b"test-hmac-key-32-bytes-long-xyz!!"


@pytest.fixture
def tmp_log(tmp_path, log_key) -> APBLog:
    return APBLog(path=tmp_path / "test.log", key=log_key)


def _make_apb() -> APB:
    sk, pk = generate_keypair()
    from agent.principal import load_private_key
    E_s = SystemEvidenceBlock(
        A_0_hash="a" * 64,
        D_hat=0.35,
        t_e="2026-05-11T12:00:00+00:00",
        trace_hash="b" * 64,
        cause="persistent_drift",
    )
    D_h = HumanDecisionBlock(
        H_id="H_alice",
        decision=GovernanceDecision.RESUME.value,
        rationale="test",
        scope="test scope",
    )
    return APB.construct(E_s, D_h, sk)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_short_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least 16 bytes"):
        APBLog(path=tmp_path / "log", key=b"short")


def test_creates_log_file_on_first_append(tmp_log, tmp_path):
    apb = _make_apb()
    tmp_log.append(apb)
    assert (tmp_path / "test.log").exists()


def test_empty_log_integrity(tmp_log):
    report = tmp_log.verify_integrity()
    assert report.ok
    assert report.entries_checked == 0


# ---------------------------------------------------------------------------
# Append and read
# ---------------------------------------------------------------------------

def test_append_returns_entry_with_correct_seq(tmp_log):
    e0 = tmp_log.append(_make_apb())
    e1 = tmp_log.append(_make_apb())
    assert e0.seq == 0
    assert e1.seq == 1


def test_first_entry_prev_hmac_is_genesis(tmp_log):
    entry = tmp_log.append(_make_apb())
    assert entry.prev_hmac == _GENESIS_HMAC


def test_second_entry_prev_hmac_chains(tmp_log):
    e0 = tmp_log.append(_make_apb())
    e1 = tmp_log.append(_make_apb())
    assert e1.prev_hmac == e0.entry_hmac


def test_read_all_returns_all_entries(tmp_log):
    apbs = [_make_apb() for _ in range(5)]
    for apb in apbs:
        tmp_log.append(apb)
    entries = tmp_log.read_all()
    assert len(entries) == 5
    assert [e.seq for e in entries] == list(range(5))


def test_len_matches_append_count(tmp_log):
    for _ in range(7):
        tmp_log.append(_make_apb())
    assert len(tmp_log) == 7


# ---------------------------------------------------------------------------
# Integrity — happy path
# ---------------------------------------------------------------------------

def test_integrity_valid_after_appends(tmp_log):
    for _ in range(10):
        tmp_log.append(_make_apb())
    report = tmp_log.verify_integrity()
    assert report.ok
    assert report.entries_checked == 10


def test_integrity_persists_across_reopen(tmp_path, log_key):
    log1 = APBLog(path=tmp_path / "log", key=log_key)
    for _ in range(5):
        log1.append(_make_apb())

    log2 = APBLog(path=tmp_path / "log", key=log_key)
    report = log2.verify_integrity()
    assert report.ok
    assert report.entries_checked == 5


def test_reopen_continues_chain(tmp_path, log_key):
    log1 = APBLog(path=tmp_path / "log", key=log_key)
    e0 = log1.append(_make_apb())
    e1 = log1.append(_make_apb())

    log2 = APBLog(path=tmp_path / "log", key=log_key)
    e2 = log2.append(_make_apb())
    assert e2.prev_hmac == e1.entry_hmac
    assert e2.seq == 2

    report = log2.verify_integrity()
    assert report.ok
    assert report.entries_checked == 3


# ---------------------------------------------------------------------------
# Tamper detection — F4 scenarios
# ---------------------------------------------------------------------------

def _append_n(log: APBLog, n: int) -> list:
    entries = []
    for _ in range(n):
        entries.append(log.append(_make_apb()))
    return entries


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_tamper_modify_apb_field(tmp_path, log_key):
    """Modifying any APB field invalidates the entry_hmac."""
    log = APBLog(path=tmp_path / "log", key=log_key)
    _append_n(log, 3)
    lines = _read_lines(tmp_path / "log")
    # Corrupt middle entry: change D_hat value
    d = json.loads(lines[1])
    d["apb"]["E_s"]["D_hat"] = 99.9
    lines[1] = json.dumps(d, sort_keys=True, separators=(",", ":"))
    _write_lines(tmp_path / "log", lines)

    log2 = APBLog(path=tmp_path / "log", key=log_key)
    report = log2.verify_integrity()
    assert not report.ok
    assert report.first_fault_seq == 1


def test_tamper_delete_middle_entry(tmp_path, log_key):
    """Deleting an entry breaks the seq contiguity and prev_hmac chain."""
    log = APBLog(path=tmp_path / "log", key=log_key)
    _append_n(log, 4)
    lines = _read_lines(tmp_path / "log")
    # Remove entry at index 1 (seq=1)
    del lines[1]
    _write_lines(tmp_path / "log", lines)

    log2 = APBLog(path=tmp_path / "log", key=log_key)
    report = log2.verify_integrity()
    assert not report.ok
    # First fault should be at what was seq=2 (now seq=1 gap)
    assert report.first_fault_seq is not None


def test_tamper_reorder_entries(tmp_path, log_key):
    """Swapping two entries breaks both seq ordering and prev_hmac chain."""
    log = APBLog(path=tmp_path / "log", key=log_key)
    _append_n(log, 4)
    lines = _read_lines(tmp_path / "log")
    # Swap entry 0 and entry 1
    lines[0], lines[1] = lines[1], lines[0]
    _write_lines(tmp_path / "log", lines)

    log2 = APBLog(path=tmp_path / "log", key=log_key)
    report = log2.verify_integrity()
    assert not report.ok


def test_tamper_truncate_log(tmp_path, log_key):
    """Truncating the file to fewer entries is not tamper in itself,
    but the remaining chain must still be valid; further appends to a
    truncated file are a separate concern."""
    log = APBLog(path=tmp_path / "log", key=log_key)
    _append_n(log, 5)
    lines = _read_lines(tmp_path / "log")
    # Keep only first 3 entries
    _write_lines(tmp_path / "log", lines[:3])

    log2 = APBLog(path=tmp_path / "log", key=log_key)
    report = log2.verify_integrity()
    # The 3 remaining entries are intact
    assert report.ok
    assert report.entries_checked == 3


def test_tamper_forge_entry_hmac(tmp_path, log_key):
    """Replacing entry_hmac with a forged value is detected."""
    log = APBLog(path=tmp_path / "log", key=log_key)
    _append_n(log, 3)
    lines = _read_lines(tmp_path / "log")
    d = json.loads(lines[0])
    d["entry_hmac"] = "f" * 64  # forged
    lines[0] = json.dumps(d, sort_keys=True, separators=(",", ":"))
    _write_lines(tmp_path / "log", lines)

    log2 = APBLog(path=tmp_path / "log", key=log_key)
    report = log2.verify_integrity()
    assert not report.ok
    assert report.first_fault_seq == 0


def test_tamper_wrong_key_fails_verify(tmp_path, log_key):
    """Verifying with a different key causes all HMACs to fail."""
    log = APBLog(path=tmp_path / "log", key=log_key)
    _append_n(log, 5)

    wrong_key = b"completely-wrong-key-32-bytes-ok"
    log2 = APBLog(path=tmp_path / "log", key=wrong_key)
    report = log2.verify_integrity()
    assert not report.ok
    assert report.first_fault_seq == 0


# ---------------------------------------------------------------------------
# Thread safety — concurrent appends
# ---------------------------------------------------------------------------

def test_concurrent_appends_no_corruption(tmp_path, log_key):
    """N threads each appending M APBs; chain must be intact after all done."""
    N_THREADS = 8
    M_PER_THREAD = 25

    log = APBLog(path=tmp_path / "log", key=log_key)
    errors = []

    def worker():
        try:
            for _ in range(M_PER_THREAD):
                log.append(_make_apb())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert len(log) == N_THREADS * M_PER_THREAD

    report = log.verify_integrity()
    assert report.ok
    assert report.entries_checked == N_THREADS * M_PER_THREAD
