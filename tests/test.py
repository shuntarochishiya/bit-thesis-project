
"""
PERFORMANCE BENCHMARK — REAL FASTAPI + REAL LLM

Project: DynAgentGame

Tests:
    - HTTP health
    - Real /interact requests
    - End-to-end latency
    - NPC dialogue latency
    - Repeated NPC dialogue
    - Error rate
    - Response consistency
    - p50 / p95 latency
    - CSV and JSON results

Requirements:
    - FastAPI server must already be running
    - Real LLM configuration must be active

No mocks.
No fake delays.
No reset endpoints.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

API_URL = os.getenv(
    "RPG_BENCHMARK_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

REQUEST_TIMEOUT = 180

# Number of repetitions per scenario.
# Start small because every /interact request uses the real LLM.
REPEATS = 3

RESULTS_DIR = ROOT / "tests" / "benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now(timezone.utc).strftime(
    "%Y%m%d_%H%M%S"
)

CSV_PATH = RESULTS_DIR / f"performance_{RUN_ID}.csv"
JSON_PATH = RESULTS_DIR / f"performance_{RUN_ID}.json"


SCENARIOS = [
    {
        "id": "bartender_first",
        "description": "Bartender dialogue",
        "text": "Hello, bartender. How are you today?",
        "target": "bartender",
    },
    {
        "id": "bartender_repeat",
        "description": "Repeated bartender dialogue",
        "text": "Hello again, bartender. Do you remember me?",
        "target": "bartender",
    },
    {
        "id": "merchant_first",
        "description": "Merchant dialogue",
        "text": "Hello, merchant. What goods do you sell?",
        "target": "merchant",
    },
    {
        "id": "merchant_repeat",
        "description": "Repeated merchant dialogue",
        "text": "Hello again, merchant. What do you recommend?",
        "target": "merchant",
    },
]


def http_request(method, path, payload=None, timeout=30):
    data = None

    headers = {
        "Accept": "application/json",
    }

    if payload is not None:
        data = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        API_URL + path,
        data=data,
        headers=headers,
        method=method,
    )

    started = time.perf_counter()

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:

            raw = response.read()
            elapsed = time.perf_counter() - started

            body = json.loads(raw.decode("utf-8"))

            return {
                "status": response.status,
                "elapsed": elapsed,
                "body": body,
                "error": None,
            }

    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started

        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)

        return {
            "status": exc.code,
            "elapsed": elapsed,
            "body": None,
            "error": detail,
        }

    except Exception as exc:
        elapsed = time.perf_counter() - started

        return {
            "status": None,
            "elapsed": elapsed,
            "body": None,
            "error": repr(exc),
        }


def percentile(values, percentile_value):
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        len(ordered) - 1
    ) * percentile_value / 100

    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)

    fraction = position - lower

    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def rounded(value):
    return round(value, 4) if value is not None else None


def run_scenario(scenario, repetition):
    payload = {
        "text": scenario["text"],
        "target": scenario["target"],
    }

    result = http_request(
        "POST",
        "/interact",
        payload=payload,
        timeout=REQUEST_TIMEOUT,
    )

    body = result["body"]

    valid_response = (
        result["status"] == 200
        and isinstance(body, dict)
        and body.get("success") is True
        and isinstance(body.get("response"), str)
        and bool(body["response"].strip())
        and isinstance(body.get("state"), dict)
        and isinstance(body.get("context"), dict)
    )

    response_text = (
        body.get("response", "")
        if isinstance(body, dict)
        else ""
    )

    row = {
        "scenario": scenario["id"],
        "repetition": repetition,
        "target": scenario["target"],
        "http_status": result["status"],
        "latency_seconds": rounded(result["elapsed"]),
        "success": valid_response,
        "response_length": len(response_text),
        "response_preview": response_text,
        "error": result["error"],
    }

    status_label = (
        "PASS" if valid_response else "FAIL"
    )

    print(
        f"[{status_label}] "
        f"{scenario['id']:<22} "
        f"run={repetition} "
        f"HTTP={result['status']} "
        f"time={result['elapsed']:.3f}s"
    )

    if result["error"]:
        print("       Error:", result["error"][:500])

    if valid_response:
        print(
            "       Response:",
            response_text.replace("\n", " "),
        )

    return row


def build_summary(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["scenario"]].append(row)

    summary = {}

    for scenario, group in grouped.items():
        successful = [
            item["latency_seconds"]
            for item in group
            if item["success"]
        ]

        failures = sum(
            not item["success"]
            for item in group
        )

        summary[scenario] = {
            "requests": len(group),
            "successful": len(successful),
            "failed": failures,
            "error_rate": rounded(
                failures / len(group)
            ),
            "mean_seconds": rounded(
                statistics.mean(successful)
            ) if successful else None,
            "median_seconds": rounded(
                statistics.median(successful)
            ) if successful else None,
            "p95_seconds": rounded(
                percentile(successful, 95)
            ),
            "min_seconds": rounded(
                min(successful)
            ) if successful else None,
            "max_seconds": rounded(
                max(successful)
            ) if successful else None,
        }

    return summary


def save_results(rows, summary, metadata):
    fields = [
        "scenario",
        "repetition",
        "target",
        "http_status",
        "latency_seconds",
        "success",
        "response_length",
        "response_preview",
        "error",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    report = {
        "metadata": metadata,
        "summary": summary,
        "requests": rows,
    }

    with JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main():
    print("=" * 76)
    print("PERFORMANCE BENCHMARK — REAL FASTAPI + REAL LLM")
    print("PROJECT:", ROOT)
    print("API:", API_URL)
    print("REPEATS:", REPEATS)
    print("=" * 76)

    print("\n--- A. SERVER HEALTH ---")

    health = http_request(
        "GET",
        "/health",
        timeout=10,
    )

    if (
        health["status"] != 200
        or not isinstance(health["body"], dict)
        or health["body"].get("status") != "ok"
    ):
        print("[FAIL] Backend is unavailable")
        print("Details:", health["error"])
        return 1

    print(
        f"[PASS] Backend is available "
        f"({health['elapsed']:.3f}s)"
    )

    print("\n--- B. INITIAL STATE ---")

    state_result = http_request(
        "GET",
        "/state",
        timeout=10,
    )

    if state_result["status"] != 200:
        print("[FAIL] Cannot retrieve initial state")
        return 1

    initial_state = state_result["body"]["state"]

    print("[PASS] Initial state retrieved")
    print("Location:", initial_state.get("location"))
    print("Gold:", initial_state.get("gold"))

    print("\n--- C. REAL LLM INTERACTIONS ---")

    location_result = http_request(
        "POST",
        "/location",
        payload={"location": "prototype_hub"},
        timeout=10,
    )

    if (
        location_result["status"] != 200
        or not isinstance(location_result["body"], dict)
        or location_result["body"].get("location") != "prototype_hub"
    ):
        print("[FAIL] Cannot synchronize prototype_hub")
        print("Details:", location_result)
        return 1

    print("[PASS] Location synchronized: prototype_hub")

    rows = []

    benchmark_started = time.perf_counter()

    # Preserve scenario order.
    # The repeated dialogues are intentionally executed
    # after the first dialogues.
    for scenario in SCENARIOS:
        print(
            f"\nScenario: {scenario['description']}"
        )

        for repetition in range(1, REPEATS + 1):
            row = run_scenario(
                scenario,
                repetition,
            )

            rows.append(row)

    benchmark_duration = (
        time.perf_counter() - benchmark_started
    )

    print("\n--- D. FINAL STATE ---")

    final_state_result = http_request(
        "GET",
        "/state",
        timeout=10,
    )

    final_state = None

    if final_state_result["status"] == 200:
        final_state = final_state_result["body"]["state"]
        print("[PASS] Final state retrieved")
        print("Location:", final_state.get("location"))
        print("Gold:", final_state.get("gold"))
    else:
        print("[FAIL] Final state unavailable")

    print("\n--- E. PERFORMANCE SUMMARY ---")

    summary = build_summary(rows)

    print(
        f"{'Scenario':<24}"
        f"{'Mean':>10}"
        f"{'Median':>10}"
        f"{'p95':>10}"
        f"{'Errors':>10}"
    )

    print("-" * 64)

    for scenario, metrics in summary.items():

        def fmt(value):
            return (
                f"{value:.3f}s"
                if value is not None
                else "N/A"
            )

        print(
            f"{scenario:<24}"
            f"{fmt(metrics['mean_seconds']):>10}"
            f"{fmt(metrics['median_seconds']):>10}"
            f"{fmt(metrics['p95_seconds']):>10}"
            f"{metrics['failed']:>10}"
        )

    successful_rows = [
        row for row in rows
        if row["success"]
    ]

    total_requests = len(rows)
    total_success = len(successful_rows)
    total_fail = total_requests - total_success

    all_latencies = [
        row["latency_seconds"]
        for row in successful_rows
    ]

    print("\n--- F. GLOBAL METRICS ---")

    print("Requests:", total_requests)
    print("Successful:", total_success)
    print("Failed:", total_fail)

    print(
        "Error rate:",
        f"{100 * total_fail / total_requests:.2f}%"
    )

    if all_latencies:
        print(
            "Median latency:",
            f"{statistics.median(all_latencies):.3f}s",
        )

        print(
            "p95 latency:",
            f"{percentile(all_latencies, 95):.3f}s",
        )

    print(
        "Benchmark duration:",
        f"{benchmark_duration:.3f}s",
    )

    metadata = {
        "run_id": RUN_ID,
        "api_url": API_URL,
        "repeats": REPEATS,
        "total_requests": total_requests,
        "successful": total_success,
        "failed": total_fail,
        "benchmark_duration_seconds": rounded(
            benchmark_duration
        ),
        "initial_location": initial_state.get("location"),
        "final_location": (
            final_state.get("location")
            if final_state else None
        ),
        "measurement_scope": "HTTP end-to-end latency",
        "llm_calls": None,
        "input_tokens": None,
        "output_tokens": None,
        "note": (
            "LLM call counts and token usage are not exposed "
            "by the current HTTP response schema. "
            "No estimates are substituted."
        ),
    }

    save_results(
        rows,
        summary,
        metadata,
    )

    print("\n--- G. OUTPUT FILES ---")

    print("CSV:", CSV_PATH)
    print("JSON:", JSON_PATH)

    print("\n" + "=" * 76)

    if total_fail == 0:
        print(
            f"PERFORMANCE BENCHMARK: "
            f"{total_success} PASS / 0 FAIL"
        )
    else:
        print(
            f"PERFORMANCE BENCHMARK: "
            f"{total_success} PASS / {total_fail} FAIL"
        )

    print("=" * 76)

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
