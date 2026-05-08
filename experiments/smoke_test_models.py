# -*- coding: utf-8 -*-
"""
Smoke test for the 6 cross-model T* candidates (P8 Sprint 0).

Verifies each model boots via LiveLLM and returns a valid tool name.
Single call per model, T=0.4, default phase.
"""
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.live_llm import LiveLLM, TOOL_NAMES

MODELS = [
    "mistral:7b",
    "deepseek-r1:8b",
    "gemma4:latest",
    "gpt-oss:20b",
    "qwen2.5:7b",
    "llama3.2:3b",
]


def test_model(name: str) -> tuple[bool, str, float]:
    t0 = time.time()
    try:
        llm = LiveLLM(model=name, temperature=0.4)
        tool, risk, depth = llm.select_tool(phase="burn_in", progress=0.0)
        elapsed = time.time() - t0
        ok = tool in TOOL_NAMES
        return ok, f"tool={tool} risk={risk:.2f} depth={depth}", elapsed
    except Exception as e:
        return False, f"ERROR: {type(e).__name__}: {e}", time.time() - t0


def main():
    print(f"{'Model':<28} {'Status':<8} {'Time':<8} Detail")
    print("-" * 80)
    results = {}
    for m in MODELS:
        ok, detail, t = test_model(m)
        status = "OK" if ok else "FAIL"
        print(f"{m:<28} {status:<8} {t:>6.1f}s  {detail}")
        results[m] = ok
    print("-" * 80)
    passed = sum(results.values())
    print(f"Passed: {passed}/{len(MODELS)}")
    sys.exit(0 if passed == len(MODELS) else 1)


if __name__ == "__main__":
    main()
