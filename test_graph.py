"""
Phase 3 — End-to-End Test
===========================
Runs a single user query through the LangGraph pipeline twice:
  1. Protocol A (Free-Text)
  2. Protocol B (Strict JSON)

Prints the full intermediate states to verify data flow.

Usage:
    python test_graph.py
"""

import json
import time

# ── Colour helpers ──────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def banner(title: str, color: str = CYAN):
    width = 70
    print(f"\n{BOLD}{color}{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}{RESET}\n")


def section(title: str):
    print(f"  {BOLD}{YELLOW}── {title} ──{RESET}")


def kv(key: str, value: str, indent: int = 4):
    prefix = " " * indent
    # For multiline values, indent continuation lines
    lines = str(value).split("\n")
    print(f"{prefix}{BOLD}{key}:{RESET} {lines[0]}")
    for line in lines[1:]:
        print(f"{prefix}  {DIM}{line}{RESET}")


def run_test():
    # Import here so we get nice error messages if something is missing
    from graph import build_graph, run_pipeline

    # ── Test query ─────────────────────────────────────────────────────
    test_query = (
        "I'm 30, non-smoker, and I want comprehensive coverage for diabetes and mental health. Budget is under $600."
    )
    expected_policy = "POL-003"

    banner("PHASE 3 — END-TO-END PIPELINE TEST")
    print(f"  {BOLD}Test Query:{RESET}")
    print(f"    \"{test_query}\"")
    print(f"  {BOLD}Expected Policy:{RESET} {expected_policy}")
    print()

    # ── Build graph ────────────────────────────────────────────────────
    section("Building LangGraph pipeline...")
    graph = build_graph()
    print(f"    {GREEN}✔ Graph compiled successfully{RESET}\n")

    # ══════════════════════════════════════════════════════════════════
    #  RUN 1: PROTOCOL A (Free-Text)
    # ══════════════════════════════════════════════════════════════════
    banner("RUN 1 — PROTOCOL A (Free-Text Summaries)", MAGENTA)
    start_time = time.time()
    result_a = run_pipeline(graph, user_input=test_query, protocol="text")
    elapsed_a = time.time() - start_time

    section("Agent 1 — Intake Output (Free-Text Paragraph)")
    kv("Payload", result_a["intake_payload"])
    print()

    section("Agent 2 — Recommender Output")
    kv("Policy ID", result_a["recommended_policy_id"])
    kv("Reasoning", result_a["recommendation_reasoning"])
    print()

    section("Agent 3 — Auditor Output")
    status_color = GREEN if result_a["audit_result"] == "Approved" else RED
    print(f"    {BOLD}Result: {status_color}{result_a['audit_result']}{RESET}")
    kv("Reasoning", result_a["audit_reasoning"])
    print()

    if result_a.get("error"):
        print(f"    {RED}⚠ Errors: {result_a['error']}{RESET}")

    match_a = result_a["recommended_policy_id"] == expected_policy
    match_color = GREEN if match_a else RED
    print(f"    {BOLD}Match Expected:{RESET} {match_color}{'✔ YES' if match_a else '✘ NO'}{RESET} "
          f"(got {result_a['recommended_policy_id']}, expected {expected_policy})")
    print(f"    {DIM}Time: {elapsed_a:.2f}s{RESET}")

    # ══════════════════════════════════════════════════════════════════
    #  RUN 2: PROTOCOL B (Strict JSON)
    # ══════════════════════════════════════════════════════════════════
    banner("RUN 2 — PROTOCOL B (Strict JSON Schema)", MAGENTA)
    start_time = time.time()
    result_b = run_pipeline(graph, user_input=test_query, protocol="json")
    elapsed_b = time.time() - start_time

    section("Agent 1 — Intake Output (JSON)")
    # Pretty-print the JSON if valid
    try:
        parsed = json.loads(result_b["intake_payload"])
        kv("Payload", json.dumps(parsed, indent=2))
    except (json.JSONDecodeError, TypeError):
        kv("Payload (raw)", result_b["intake_payload"])
    print()

    section("Agent 2 — Recommender Output")
    kv("Policy ID", result_b["recommended_policy_id"])
    kv("Reasoning", result_b["recommendation_reasoning"])
    print()

    section("Agent 3 — Auditor Output")
    status_color = GREEN if result_b["audit_result"] == "Approved" else RED
    print(f"    {BOLD}Result: {status_color}{result_b['audit_result']}{RESET}")
    kv("Reasoning", result_b["audit_reasoning"])
    print()

    if result_b.get("error"):
        print(f"    {RED}⚠ Errors: {result_b['error']}{RESET}")

    match_b = result_b["recommended_policy_id"] == expected_policy
    match_color = GREEN if match_b else RED
    print(f"    {BOLD}Match Expected:{RESET} {match_color}{'✔ YES' if match_b else '✘ NO'}{RESET} "
          f"(got {result_b['recommended_policy_id']}, expected {expected_policy})")
    print(f"    {DIM}Time: {elapsed_b:.2f}s{RESET}")

    # ══════════════════════════════════════════════════════════════════
    #  COMPARISON SUMMARY
    # ══════════════════════════════════════════════════════════════════
    banner("COMPARISON SUMMARY", GREEN)
    print(f"  {'Metric':<30} {'Protocol A (Text)':<25} {'Protocol B (JSON)':<25}")
    print(f"  {'─' * 80}")
    print(f"  {'Recommended Policy':<30} {result_a['recommended_policy_id']:<25} {result_b['recommended_policy_id']:<25}")
    print(f"  {'Audit Result':<30} {result_a['audit_result']:<25} {result_b['audit_result']:<25}")
    print(f"  {'Matched Expected':<30} {'✔ YES' if match_a else '✘ NO':<25} {'✔ YES' if match_b else '✘ NO':<25}")
    print(f"  {'Errors':<30} {result_a.get('error') or 'None':<25} {result_b.get('error') or 'None':<25}")
    print(f"  {'Latency':<30} {elapsed_a:.2f}s{'':<20} {elapsed_b:.2f}s")
    print()

    banner("PHASE 3 TEST COMPLETE", GREEN)


if __name__ == "__main__":
    run_test()
