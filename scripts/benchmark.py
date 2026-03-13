#!/usr/bin/env python3
"""CrabInfer vs vLLM benchmark script.

Usage:
    python benchmark.py --endpoint http://localhost:8080/v1 --model local \
        --num-requests 100 --concurrency 10 --max-tokens 200
"""

import argparse
import json
import time
import statistics
import concurrent.futures
from urllib.request import Request, urlopen
from urllib.error import URLError


PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to sort a list of dictionaries by a key.",
    "What are the main differences between TCP and UDP?",
    "Describe the process of photosynthesis step by step.",
    "Write a haiku about machine learning.",
    "Explain how a transformer neural network works.",
    "What is the difference between a stack and a queue?",
    "Describe the water cycle in detail.",
    "Write a short story about a robot learning to paint.",
    "Explain quantum entanglement to a 10 year old.",
    "What are the SOLID principles in software engineering?",
    "Describe how a compiler works in three paragraphs.",
    "Write a recipe for chocolate chip cookies.",
    "Explain the CAP theorem in distributed systems.",
    "What is the significance of the Turing test?",
    "Describe the differences between REST and GraphQL APIs.",
    "Write a limerick about programming in Rust.",
    "Explain how garbage collection works in modern languages.",
    "What are the key features of the Rust programming language?",
    "Describe the architecture of a modern web browser.",
]


def send_request(endpoint: str, model: str, prompt: str, max_tokens: int) -> dict:
    """Send a chat completion request and measure timing."""
    url = f"{endpoint}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }).encode("utf-8")

    req = Request(url, data=payload, headers={"Content-Type": "application/json"})

    start = time.monotonic()
    try:
        with urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        return {"error": str(e), "elapsed_s": time.monotonic() - start}

    elapsed = time.monotonic() - start

    usage = body.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    return {
        "elapsed_s": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_sec": completion_tokens / elapsed if elapsed > 0 else 0,
    }


def send_streaming_request(endpoint: str, model: str, prompt: str, max_tokens: int) -> dict:
    """Send a streaming request and measure TTFT + ITL."""
    url = f"{endpoint}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }).encode("utf-8")

    req = Request(url, data=payload, headers={"Content-Type": "application/json"})

    start = time.monotonic()
    ttft = None
    token_times = []
    total_tokens = 0

    try:
        with urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content"):
                        now = time.monotonic()
                        if ttft is None:
                            ttft = now - start
                        token_times.append(now)
                        total_tokens += 1
                except json.JSONDecodeError:
                    continue
    except URLError as e:
        return {"error": str(e)}

    elapsed = time.monotonic() - start

    # Inter-token latencies
    itls = []
    for i in range(1, len(token_times)):
        itls.append(token_times[i] - token_times[i - 1])

    return {
        "elapsed_s": elapsed,
        "ttft_s": ttft or elapsed,
        "completion_tokens": total_tokens,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else 0,
        "itl_mean_s": statistics.mean(itls) if itls else 0,
        "itl_p50_s": statistics.median(itls) if itls else 0,
        "itl_p99_s": sorted(itls)[int(len(itls) * 0.99)] if itls else 0,
    }


def run_benchmark(endpoint: str, model: str, num_requests: int, concurrency: int,
                  max_tokens: int, streaming: bool = False) -> dict:
    """Run the benchmark with the given concurrency."""
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]
    results = []

    func = send_streaming_request if streaming else send_request
    overall_start = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(func, endpoint, model, p, max_tokens) for p in prompts]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    overall_elapsed = time.monotonic() - overall_start

    # Filter errors
    successes = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not successes:
        return {"error": "All requests failed", "errors": errors}

    elapsed_list = [r["elapsed_s"] for r in successes]
    tps_list = [r["tokens_per_sec"] for r in successes]
    total_tokens = sum(r["completion_tokens"] for r in successes)

    summary = {
        "total_requests": num_requests,
        "successful": len(successes),
        "failed": len(errors),
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "streaming": streaming,
        "overall_elapsed_s": round(overall_elapsed, 3),
        "total_tokens_generated": total_tokens,
        "overall_tokens_per_sec": round(total_tokens / overall_elapsed, 2),
        "latency_p50_s": round(statistics.median(elapsed_list), 3),
        "latency_p95_s": round(sorted(elapsed_list)[int(len(elapsed_list) * 0.95)], 3),
        "latency_p99_s": round(sorted(elapsed_list)[int(len(elapsed_list) * 0.99)], 3),
        "per_request_tps_mean": round(statistics.mean(tps_list), 2),
        "per_request_tps_p50": round(statistics.median(tps_list), 2),
    }

    if streaming and successes:
        ttfts = [r.get("ttft_s", 0) for r in successes if "ttft_s" in r]
        itls = [r.get("itl_p50_s", 0) for r in successes if "itl_p50_s" in r]
        if ttfts:
            summary["ttft_p50_s"] = round(statistics.median(ttfts), 4)
            summary["ttft_p95_s"] = round(sorted(ttfts)[int(len(ttfts) * 0.95)], 4)
            summary["ttft_p99_s"] = round(sorted(ttfts)[int(len(ttfts) * 0.99)], 4)
        if itls:
            summary["itl_p50_s"] = round(statistics.median(itls), 4)

    return summary


def main():
    parser = argparse.ArgumentParser(description="CrabInfer benchmark")
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL (e.g. http://localhost:8080/v1)")
    parser.add_argument("--model", default="local", help="Model name to use in requests")
    parser.add_argument("--num-requests", type=int, default=50, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens per response")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode (measures TTFT + ITL)")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    parser.add_argument("--label", default="benchmark", help="Label for this run")
    args = parser.parse_args()

    print(f"=== CrabInfer Benchmark: {args.label} ===")
    print(f"Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"Requests: {args.num_requests}, Concurrency: {args.concurrency}, Max tokens: {args.max_tokens}")
    print(f"Streaming: {args.streaming}")
    print()

    # Warmup
    print("Warming up (3 requests)...")
    for i in range(3):
        send_request(args.endpoint, args.model, PROMPTS[i], 50)
    print()

    # Non-streaming benchmark
    print("Running non-streaming benchmark...")
    results = run_benchmark(args.endpoint, args.model, args.num_requests, args.concurrency, args.max_tokens)
    results["label"] = args.label
    print(json.dumps(results, indent=2))
    print()

    # Streaming benchmark (for TTFT/ITL)
    if args.streaming:
        print("Running streaming benchmark...")
        stream_results = run_benchmark(args.endpoint, args.model, min(args.num_requests, 20),
                                        args.concurrency, args.max_tokens, streaming=True)
        stream_results["label"] = f"{args.label}_streaming"
        print(json.dumps(stream_results, indent=2))
        results["streaming_results"] = stream_results

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
