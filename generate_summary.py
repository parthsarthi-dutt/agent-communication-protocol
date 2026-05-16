import json
import os
from datetime import datetime

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(_PROJECT_DIR, "checkpoint.json")
SUMMARY_FILE = os.path.join(_PROJECT_DIR, "evaluation_summary.json")
PROTOCOLS = ["text", "json", "markdown_json"]

def generate_summary():
    if not os.path.exists(CHECKPOINT_FILE):
        print("❌ No checkpoint.json found. The evaluation has not started or has already completed and cleaned up.")
        return

    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    metrics = checkpoint.get("metrics", {})
    last_user_index = checkpoint.get("last_user_index", 0)
    users_processed = last_user_index + 1

    print("\n" + "═" * 60)
    print(f" GENERATING SUMMARY (Based on {users_processed} users processed) ".center(60, "═"))
    print("═" * 60)

    summary = {}
    for protocol in PROTOCOLS:
        if protocol not in metrics:
            continue
            
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

    summary_export = {
        "evaluation_generated_at": datetime.now().isoformat(),
        "users_processed_so_far": users_processed,
        "protocols_tested": PROTOCOLS,
        "results": summary,
        "note": "This summary was generated early before the full evaluation completed."
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)
    print(f"\n📊 Summary successfully exported to: {SUMMARY_FILE}\n")

if __name__ == "__main__":
    generate_summary()
