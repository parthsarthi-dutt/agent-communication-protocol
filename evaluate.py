"""
Phase 4 — Evaluation Engine (with Checkpointing)
==================================================
Loops through all mock users and runs the pipeline for three communication
protocols: text, json, and markdown_json.

Features:
- **Checkpointing**: Saves progress after every run to `checkpoint.json`
  and appends results to `evaluation_log.csv`. On re-run, automatically
  resumes from where it left off.
- **Rate-Limit Handling**: Catches 429/rate-limit errors, saves state,
  and exits gracefully so the user can swap API keys and continue.
- **Three Protocols**: Evaluates Protocol A (text), B (json), C (markdown_json).
- **Metrics Export**: On completion, writes `evaluation_summary.json`.

Usage:
    python evaluate.py
"""

import csv
import json
import os
import sys
import time
from datetime import datetime

from graph import build_graph, run_pipeline

# ── Configuration ────────────────────────────────────────────────────────
# Using Round-Robin with 3 API keys gives us 90 RPM / 90k TPM.
# 1 run every 3 seconds = 20 runs/minute = ~90k tokens/minute. Safe.
API_DELAY_SECONDS = 3.0

PROTOCOLS = ["text", "json", "markdown_json"]

# File paths (in project root)
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(_PROJECT_DIR, "checkpoint.json")
CSV_LOG_FILE = os.path.join(_PROJECT_DIR, "evaluation_log.csv")
SUMMARY_FILE = os.path.join(_PROJECT_DIR, "evaluation_summary.json")

CSV_COLUMNS = [
    "timestamp",
    "user_id",
    "protocol",
    "expected_policy",
    "recommended_policy",
    "audit_result",
    "intake_payload_chars",
    "latency_seconds",
    "error_flags",
]


# ── Checkpoint Helpers ───────────────────────────────────────────────────

def _load_checkpoint() -> dict | None:
    """Load checkpoint if it exists, else return None."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_checkpoint(user_index: int, protocol_index: int, metrics: dict):
    """Save current progress to checkpoint.json."""
    checkpoint = {
        "last_user_index": user_index,
        "last_protocol_index": protocol_index,
        "metrics": metrics,
        "saved_at": datetime.now().isoformat(),
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def _delete_checkpoint():
    """Remove checkpoint file on successful completion."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


def _init_csv():
    """Create CSV file with headers if it doesn't exist."""
    if not os.path.exists(CSV_LOG_FILE):
        with open(CSV_LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)


def _append_csv_row(row: list):
    """Append a single result row to the CSV log."""
    with open(CSV_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _init_metrics() -> dict:
    """Create a fresh metrics dictionary for all protocols."""
    return {
        proto: {
            "total_runs": 0,
            "successes": 0,
            "auditor_rejections": 0,
            "parsing_errors": 0,
            "total_latency": 0.0,
            "total_payload_chars": 0,
        }
        for proto in PROTOCOLS
    }


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is a rate-limit / 429 error."""
    err_str = str(error).lower()
    return any(kw in err_str for kw in ["429", "resource_exhausted", "rate", "quota", "too many requests"])


# ── Main Evaluation Loop ────────────────────────────────────────────────

def evaluate():
    from prompts import _DATA_DIR

    users_path = os.path.join(_DATA_DIR, "mock_users.json")
    with open(users_path, "r", encoding="utf-8") as f:
        users = json.load(f)

    total_users = len(users)
    total_runs = total_users * len(PROTOCOLS)

    # ── Load checkpoint or start fresh ──
    checkpoint = _load_checkpoint()
    if checkpoint:
        start_user = checkpoint["last_user_index"]
        start_proto = checkpoint["last_protocol_index"] + 1
        metrics = checkpoint["metrics"]

        # If we finished all protocols for that user, move to next
        if start_proto >= len(PROTOCOLS):
            start_user += 1
            start_proto = 0

        completed = start_user * len(PROTOCOLS) + start_proto
        print(f"Resuming from checkpoint: User {start_user + 1}/{total_users}, "
              f"Protocol {PROTOCOLS[start_proto]} ({completed}/{total_runs} runs completed)")
        print(f"Checkpoint saved at: {checkpoint['saved_at']}\n")
    else:
        start_user = 0
        start_proto = 0
        metrics = _init_metrics()
        print(f"Starting fresh evaluation: {total_users} users × {len(PROTOCOLS)} protocols = {total_runs} total runs\n")

    # ── Initialize CSV ──
    _init_csv()

    # ── Build graph once ──
    graph = build_graph()

    # ── Evaluation loop ──
    for user_idx in range(start_user, total_users):
        user = users[user_idx]
        user_id = user["user_id"]
        user_input = user["query"]
        expected_policy = user["expected_policy_id"]

        # Determine which protocol to start from for this user
        proto_start = start_proto if user_idx == start_user else 0

        print(f"\n[{user_idx + 1}/{total_users}] {user_id} | Expected: {expected_policy}")

        for proto_idx in range(proto_start, len(PROTOCOLS)):
            protocol = PROTOCOLS[proto_idx]
            run_num = user_idx * len(PROTOCOLS) + proto_idx + 1

            start_time = time.time()
            try:
                result = run_pipeline(graph, user_input=user_input, protocol=protocol)
                elapsed = time.time() - start_time

                # ── Compute metrics ──
                rec_policy = result.get("recommended_policy_id", "UNKNOWN")
                audit = result.get("audit_result", "Rejected")
                has_error = result.get("error") is not None
                is_success = (rec_policy == expected_policy and audit == "Approved")
                is_rejected = (audit == "Rejected")
                is_parse_error = has_error or rec_policy in ["UNKNOWN", "ERROR", "PARSE_ERROR"]
                payload_chars = len(result.get("intake_payload", ""))

                # ── Update metrics ──
                m = metrics[protocol]
                m["total_runs"] += 1
                m["total_latency"] += elapsed
                m["total_payload_chars"] += payload_chars
                if is_success and not has_error:
                    m["successes"] += 1
                if is_rejected and not has_error:
                    m["auditor_rejections"] += 1
                if is_parse_error:
                    m["parsing_errors"] += 1

                # ── Build error flags string ──
                error_flags = ""
                if has_error:
                    error_flags = str(result["error"])[:200]  # Truncate for CSV

                # ── Log to CSV ──
                _append_csv_row([
                    datetime.now().isoformat(),
                    user_id,
                    protocol,
                    expected_policy,
                    rec_policy,
                    audit,
                    payload_chars,
                    f"{elapsed:.2f}",
                    error_flags,
                ])

                # ── Print progress ──
                match_str = "SUCCESS" if is_success else "FAIL"
                err_str = f" [ERR: {error_flags[:80]}]" if has_error else ""
                print(f"  [{protocol.upper():>13}] -> {rec_policy:<12} | {audit:<10} | {match_str}{err_str}  ({run_num}/{total_runs})")

            except Exception as e:
                elapsed = time.time() - start_time

                if _is_rate_limit_error(e):
                    # ── Rate limit hit — save and exit gracefully ──
                    print(f"\n[WARNING] API Limit Reached. Saving state and pausing...")
                    print(f"   Error: {str(e)[:150]}")
                    _save_checkpoint(user_idx, proto_idx - 1 if proto_idx > 0 else 0, metrics)
                    print(f"   Checkpoint saved. Re-run `python evaluate.py` to resume.")
                    print(f"   You can swap your API key in .env before restarting.\n")
                    sys.exit(0)
                else:
                    # ── Non-rate-limit error — log it and continue ──
                    print(f"  [{protocol.upper():>13}] -> EXCEPTION: {str(e)[:100]}  ({run_num}/{total_runs})")
                    metrics[protocol]["total_runs"] += 1
                    metrics[protocol]["parsing_errors"] += 1
                    metrics[protocol]["total_latency"] += elapsed

                    _append_csv_row([
                        datetime.now().isoformat(),
                        user_id,
                        protocol,
                        expected_policy,
                        "EXCEPTION",
                        "N/A",
                        0,
                        f"{elapsed:.2f}",
                        str(e)[:200],
                    ])

            # ── Save checkpoint after each run ──
            _save_checkpoint(user_idx, proto_idx, metrics)

            # ── Rate-limit delay ──
            time.sleep(API_DELAY_SECONDS)

    # ═══════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print(" EVALUATION COMPLETE ".center(60, "═"))
    print("═" * 60)

    summary = {}
    for protocol in PROTOCOLS:
        m = metrics[protocol]
        total = max(m["total_runs"], 1)
        success_rate = (m["successes"] / total) * 100
        rejection_rate = (m["auditor_rejections"] / total) * 100
        error_rate = (m["parsing_errors"] / total) * 100
        avg_latency = m["total_latency"] / total
        avg_payload = m["total_payload_chars"] / total

        summary[protocol] = {
            "total_runs": m["total_runs"],
            "successes": m["successes"],
            "task_success_rate_pct": round(success_rate, 2),
            "auditor_rejections": m["auditor_rejections"],
            "auditor_rejection_rate_pct": round(rejection_rate, 2),
            "parsing_errors": m["parsing_errors"],
            "parsing_error_rate_pct": round(error_rate, 2),
            "average_latency_seconds": round(avg_latency, 2),
            "average_payload_chars": round(avg_payload, 1),
        }

        print(f"\n{'─' * 50}")
        print(f"  Protocol: {protocol.upper()}")
        print(f"{'─' * 50}")
        print(f"  Total Runs            : {m['total_runs']}")
        print(f"  Task Success Rate     : {success_rate:.1f}% ({m['successes']}/{total})")
        print(f"  Auditor Rejection Rate: {rejection_rate:.1f}% ({m['auditor_rejections']}/{total})")
        print(f"  Parsing Error Rate    : {error_rate:.1f}% ({m['parsing_errors']}/{total})")
        print(f"  Average Latency       : {avg_latency:.2f}s")
        print(f"  Avg Payload Chars     : {avg_payload:.0f} chars")

    print(f"\n{'═' * 60}")

    # ── Export summary ──
    summary_export = {
        "evaluation_completed_at": datetime.now().isoformat(),
        "total_users": total_users,
        "protocols_tested": PROTOCOLS,
        "results": summary,
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)
    print(f"\nSummary exported to: {SUMMARY_FILE}")
    print(f"Full log available at: {CSV_LOG_FILE}")

    # ── Clean up checkpoint on success ──
    _delete_checkpoint()
    print("Checkpoint cleared (evaluation complete).\n")


if __name__ == "__main__":
    evaluate()
