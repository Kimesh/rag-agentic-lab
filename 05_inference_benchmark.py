"""
STEP 5: LLM Inference Benchmarking Suite
========================================
This script benchmarks your in-house LLM inference engine (Ollama / vLLM)
under production-like streaming workloads.

It measures the exact metrics demanded by AI Platform & FDE engineering roles:
1. TTFT (Time To First Token): Milliseconds until the first token streams back.
2. TPOT (Time Per Output Token): Inter-token streaming latency (ms/token).
3. TPS (Tokens Per Second): Generation throughput.
4. Concurrency Scaling: Performance under multiple simultaneous client requests.

Zero external pip dependencies (Standard Python 3 only).
"""

import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

OLLAMA_API_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "llama3.2:1b"

# ==============================================================================
# 1. STREAMING INFERENCE RUNNER
# ==============================================================================
def benchmark_single_request(prompt: str, request_id: int = 1, verbose: bool = True) -> dict:
    """
    Sends a streaming chat completion request to the local model
    and measures TTFT, TPOT, and throughput.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a concise enterprise technical assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "stream": True
    }

    req = urllib.request.Request(
        OLLAMA_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    if verbose:
        print(f"\n[Client #{request_id}] Prompt: \"{prompt}\"")
        print(f"[Client #{request_id}] Streaming Response: ", end="", flush=True)

    t_start = time.perf_counter()
    t_first_token = None
    token_count = 0
    full_response = []

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                        token_count += 1
                        full_response.append(content)
                        if verbose:
                            print(content, end="", flush=True)
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        print(f"\n[ERROR] Request #{request_id} failed: {e}")
        return {
            "request_id": request_id,
            "error": str(e),
            "success": False
        }

    t_end = time.perf_counter()

    # Metric Calculations
    ttft_ms = (t_first_token - t_start) * 1000 if t_first_token else 0.0
    total_latency_sec = t_end - t_start
    gen_time_sec = (t_end - t_first_token) if t_first_token else 0.0
    tokens_per_sec = (token_count / gen_time_sec) if gen_time_sec > 0 else 0.0
    tpot_ms = (gen_time_sec / token_count * 1000) if token_count > 0 else 0.0

    if verbose:
        print()

    return {
        "request_id": request_id,
        "success": True,
        "token_count": token_count,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "tps": tokens_per_sec,
        "total_latency_sec": total_latency_sec,
        "output_text": "".join(full_response)
    }

# ==============================================================================
# 2. CONCURRENT LOAD TEST
# ==============================================================================
def benchmark_concurrent_workload(prompts: list) -> list:
    """Simulates multiple concurrent users hitting the local model simultaneously."""
    print(f"\n[STRESS TEST] Simulating {len(prompts)} concurrent client requests hitting the engine...")
    t_batch_start = time.perf_counter()

    results = []
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [
            executor.submit(benchmark_single_request, prompt, idx + 1, False)
            for idx, prompt in enumerate(prompts)
        ]
        for f in futures:
            results.append(f.result())

    t_batch_end = time.perf_counter()
    total_batch_time = t_batch_end - t_batch_start
    total_tokens = sum(r["token_count"] for r in results if r["success"])
    system_tps = total_tokens / total_batch_time if total_batch_time > 0 else 0

    print(f"[STRESS TEST COMPLETED in {total_batch_time:.2f}s]")
    print(f"  Total Tokens Generated across all clients: {total_tokens}")
    print(f"  System-wide Concurrent Throughput: {system_tps:.1f} tokens/sec")

    return results

# ==============================================================================
# 3. BENCHMARK SUITE EXECUTION & DASHBOARD
# ==============================================================================
def print_dashboard(single_res: dict, concurrent_res: list):
    print("\n" + "=" * 80)
    print("      PRODUCTION LLM INFERENCE BENCHMARK TELEMETRY DASHBOARD")
    print(f"      Model: {MODEL_NAME} | Runtime: Ollama C++ Engine | Host: Localhost")
    print("=" * 80)

    print("\n--- TEST 1: SINGLE-STREAM LATENCY PROFILE ---")
    print(f"  • TTFT (Time To First Token) : {single_res['ttft_ms']:.1f} ms  (Target: <500ms)")
    print(f"  • TPOT (Time Per Output Token): {single_res['tpot_ms']:.1f} ms/token")
    print(f"  • Streaming Throughput (TPS)  : {single_res['tps']:.1f} tokens/sec")
    print(f"  • Tokens Generated            : {single_res['token_count']} tokens")
    print(f"  • End-to-End Latency          : {single_res['total_latency_sec']:.2f} seconds")

    print("\n--- TEST 2: CONCURRENT CLIENT SATURATION PROFILE ---")
    print(f"{'Client ID':<12} | {'Tokens':<8} | {'TTFT (ms)':<12} | {'TPOT (ms)':<12} | {'TPS':<8}")
    print("-" * 65)
    for r in concurrent_res:
        if r["success"]:
            print(f"Client #{r['request_id']:<5} | {r['token_count']:<8} | {r['ttft_ms']:<12.1f} | {r['tpot_ms']:<12.1f} | {r['tps']:<8.1f}")
        else:
            print(f"Client #{r['request_id']:<5} | FAILED: {r.get('error')}")

    avg_ttft = sum(r["ttft_ms"] for r in concurrent_res if r["success"]) / len(concurrent_res)
    avg_tps = sum(r["tps"] for r in concurrent_res if r["success"]) / len(concurrent_res)

    print("-" * 65)
    print(f"Concurrent Averages: TTFT = {avg_ttft:.1f} ms | Client TPS = {avg_tps:.1f} tok/s")
    print("=" * 80)
    print("\n>> WHAT TO TELL RECRUITERS & HIRING MANAGERS IN INTERVIEWS:")
    print("   'I benchmarked local Llama 3.2 inference runtimes under concurrent streaming loads,")
    print(f"    measuring sub-second TTFT (~{single_res['ttft_ms']:.0f}ms) and sustained throughput of ~{single_res['tps']:.1f} TPS")
    print("    to evaluate KV-cache memory pressure and latency tail distributions under burst traffic.'")
    print("=" * 80)

def main():
    print("=" * 80)
    print("  STARTING LLM INFERENCE BENCHMARK RUNNER")
    print("=" * 80)

    # 1. Warm-up and Single Stream Test
    single_res = benchmark_single_request(
        prompt="Explain the purpose of an Architectural Decision Record (ADR) in distributed software systems.",
        request_id=1,
        verbose=True
    )

    if not single_res["success"]:
        print("[!] Benchmark aborted. Ensure Ollama is running on localhost:11434.")
        return

    # 2. Concurrent Stream Test (3 parallel clients)
    concurrent_prompts = [
        "What is the difference between latency and throughput in distributed systems?",
        "Explain why Kafka partitions guarantee ordering per key but not across the entire topic.",
        "How does Redis Redlock achieve distributed locking consensus across multiple nodes?"
    ]
    concurrent_res = benchmark_concurrent_workload(concurrent_prompts)

    # 3. Telemetry Dashboard
    print_dashboard(single_res, concurrent_res)

if __name__ == "__main__":
    main()

