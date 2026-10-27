"""Benchmark the live API with identical repeated/unique requests, cache on/off."""
from __future__ import annotations
import argparse, concurrent.futures, json, statistics, time, urllib.error, urllib.request, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.metrics import summarize_latencies


def request(url, question, bypass):
    body = json.dumps({"question": question}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json", "X-Cache-Bypass": "true" if bypass else "false"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            data = json.loads(response.read())
        return {"latency_ms": (time.perf_counter() - start) * 1000, "generation_latency_ms": data.get("generation_latency_ms"),
                "cache_hit": data.get("cache_hit", False), "cache_error": data.get("cache_error", False), "error": None}
    except Exception as exc:
        return {"latency_ms": (time.perf_counter() - start) * 1000, "generation_latency_ms": None,
                "cache_hit": False, "cache_error": False, "error": type(exc).__name__}


def run(url, questions, bypass, concurrency):
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        values = list(pool.map(lambda q: request(url, q, bypass), questions))
    hits = sum(v["cache_hit"] for v in values)
    misses = len(values) - hits
    gens = [v["generation_latency_ms"] for v in values if v["generation_latency_ms"] is not None]
    return {"requests": len(values), "cache_hits": hits, "cache_misses": misses,
            "hit_rate": hits / len(values) if values else 0,
            "end_to_end_latency": summarize_latencies([v["latency_ms"] for v in values]),
            "cache_hit_latency": summarize_latencies([v["latency_ms"] for v in values if v["cache_hit"]]),
            "cache_miss_latency": summarize_latencies([v["latency_ms"] for v in values if not v["cache_hit"]]),
            "model_generation_latency_on_misses": summarize_latencies(gens),
            "inference_calls_avoided": hits, "cache_errors": sum(v["cache_error"] for v in values),
            "throughput_requests_per_second": len(values) / (time.perf_counter() - started),
            "http_errors": sum(v["error"] is not None for v in values),
            "errors_by_type": {kind: sum(v["error"] == kind for v in values) for kind in sorted({v["error"] for v in values if v["error"]})}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000/generate-sql")
    p.add_argument("--workload", default=None, help="JSON file containing question strings")
    p.add_argument("--requests", type=int, default=30)
    p.add_argument("--unique-ratio", type=float, default=0.3)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--output", default="reports/cache-benchmark.json")
    a = p.parse_args()
    if a.workload:
        raw = json.loads(Path(a.workload).read_text(encoding="utf-8"))
        base = [r["question"] if isinstance(r, dict) else r for r in raw]
    else:
        base = ["Show total expenses by category", "What is the average account balance?", "List the five largest expenses"]
    unique_count = max(1, int(a.requests * a.unique_ratio))
    questions = [base[i % len(base)] for i in range(a.requests - unique_count)] + [f"{base[i % len(base)]} for benchmark case {i}" for i in range(unique_count)]
    # Cache-disabled baseline then the same workload with caching enabled.
    disabled = run(a.url, questions, True, a.concurrency)
    enabled = run(a.url, questions, False, a.concurrency)
    result = {"configuration": {"url": a.url, "concurrency": a.concurrency, "requests": len(questions), "unique_ratio": a.unique_ratio},
              "cache_disabled": disabled, "cache_enabled": enabled}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
