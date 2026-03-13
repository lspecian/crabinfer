#!/bin/bash
set -euo pipefail

# CrabInfer GPU Model Testing — run after gpu-test-setup.sh
# Usage: bash gpu-test-models.sh [phase]
# Phases: 1=single-gpu, 2=quantization, 3=flash-attn, 4=tensor-parallel, all

REPO_DIR="$HOME/crabinfer"
cd "$REPO_DIR"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

PHASE="${1:-all}"
PORT=8080
RESULTS_DIR="/tmp/crabinfer-results"
mkdir -p "$RESULTS_DIR"

wait_for_server() {
    echo "Waiting for server to be ready..."
    for i in $(seq 1 60); do
        if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
            echo "Server ready!"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Server did not start within 120s"
    return 1
}

stop_server() {
    pkill -f crabinfer-server 2>/dev/null || true
    sleep 2
}

test_generation() {
    local label="$1"
    local prompt="${2:-Write a haiku about Rust programming}"
    local max_tokens="${3:-100}"

    echo "--- Testing: $label ---"

    # Health check
    curl -s "http://localhost:$PORT/health" | python3 -m json.tool 2>/dev/null || curl -s "http://localhost:$PORT/health"
    echo ""

    # Generation
    local start_time=$(date +%s%N)
    local response=$(curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$max_tokens,\"temperature\":0.7}")
    local end_time=$(date +%s%N)
    local elapsed=$(( (end_time - start_time) / 1000000 ))

    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    echo ""
    echo "Total request time: ${elapsed}ms"
    echo ""

    # Metrics
    echo "--- Metrics ---"
    curl -s "http://localhost:$PORT/metrics" | grep -E "tokens_per_second|time_to_first|kv_cache|tokens_generated"
    echo ""

    # GPU memory
    echo "--- GPU Memory ---"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo ""

    # Save results
    echo "{\"label\":\"$label\",\"elapsed_ms\":$elapsed,\"response\":$response}" >> "$RESULTS_DIR/generations.jsonl"
}

# ============================================================
# Phase 1: Single-GPU Inference (Qwen3-8B FP16)
# ============================================================
if [[ "$PHASE" == "1" || "$PHASE" == "all" ]]; then
    echo "=========================================="
    echo "Phase 1: Single-GPU Inference (Qwen3-8B)"
    echo "=========================================="
    stop_server

    cargo run --release -p crabinfer-server --no-default-features --features "cuda,providers" -- \
        --model "Qwen/Qwen3-8B" \
        --serving \
        --host 0.0.0.0 --port $PORT \
        --gpu_memory_utilization 0.85 \
        --context_length 4096 &

    wait_for_server
    test_generation "qwen3-8b-fp16" "Write a haiku about Rust programming" 50
    test_generation "qwen3-8b-fp16-long" "Explain the transformer architecture in detail, covering attention mechanisms, positional encoding, and the encoder-decoder structure" 300

    # Streaming test
    echo "--- Streaming test ---"
    curl -N -s -X POST "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"local","messages":[{"role":"user","content":"Count from 1 to 10"}],"max_tokens":100,"stream":true}' | head -20
    echo ""

    stop_server
    echo "Phase 1 complete!" | tee -a "$RESULTS_DIR/summary.txt"
fi

# ============================================================
# Phase 2: Quantization (GPTQ INT4 + FP8)
# ============================================================
if [[ "$PHASE" == "2" || "$PHASE" == "all" ]]; then
    echo "=========================================="
    echo "Phase 2: Quantization Testing"
    echo "=========================================="

    # 2a: GPTQ INT4
    echo "--- 2a: GPTQ INT4 ---"
    stop_server
    cargo run --release -p crabinfer-server --no-default-features --features "cuda,providers" -- \
        --model "Qwen/Qwen3-8B-GPTQ-Int4" \
        --serving --quantization gptq \
        --host 0.0.0.0 --port $PORT \
        --gpu_memory_utilization 0.85 &

    wait_for_server
    test_generation "qwen3-8b-gptq-int4" "Explain quantum computing in simple terms" 200
    stop_server

    # 2b: FP8 quantization
    echo "--- 2b: FP8 Quantization ---"
    cargo run --release -p crabinfer-server --no-default-features --features "cuda,providers" -- \
        --model "Qwen/Qwen3-8B" \
        --serving --quantization fp8 \
        --host 0.0.0.0 --port $PORT \
        --gpu_memory_utilization 0.85 &

    wait_for_server
    test_generation "qwen3-8b-fp8" "What is the meaning of life?" 200
    stop_server

    # 2c: FP8 KV cache
    echo "--- 2c: FP8 KV Cache ---"
    cargo run --release -p crabinfer-server --no-default-features --features "cuda,providers" -- \
        --model "Qwen/Qwen3-8B" \
        --serving --kv_cache_dtype fp8 \
        --host 0.0.0.0 --port $PORT \
        --gpu_memory_utilization 0.85 &

    wait_for_server
    test_generation "qwen3-8b-fp8-kv" "Describe the solar system" 200
    stop_server

    echo "Phase 2 complete!" | tee -a "$RESULTS_DIR/summary.txt"
fi

# ============================================================
# Phase 3: FlashAttention-2
# ============================================================
if [[ "$PHASE" == "3" || "$PHASE" == "all" ]]; then
    echo "=========================================="
    echo "Phase 3: FlashAttention-2"
    echo "=========================================="
    stop_server

    cargo run --release -p crabinfer-server --no-default-features --features "cuda,flash-attn,providers" -- \
        --model "Qwen/Qwen3-8B" \
        --serving \
        --host 0.0.0.0 --port $PORT \
        --gpu_memory_utilization 0.85 \
        --context_length 8192 &

    wait_for_server

    # Long prompt to stress prefill
    LONG_PROMPT=$(python3 -c "print('Summarize the following text: ' + 'The quick brown fox jumps over the lazy dog. ' * 200)")
    test_generation "qwen3-8b-flash-attn-long-prefill" "$LONG_PROMPT" 100
    test_generation "qwen3-8b-flash-attn-short" "Hello world" 50

    stop_server
    echo "Phase 3 complete!" | tee -a "$RESULTS_DIR/summary.txt"
fi

# ============================================================
# Phase 4: Tensor Parallelism (2x GPU, 72B model)
# ============================================================
if [[ "$PHASE" == "4" || "$PHASE" == "all" ]]; then
    echo "=========================================="
    echo "Phase 4: Tensor Parallelism (72B, TP=2)"
    echo "=========================================="

    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$NUM_GPUS" -lt 2 ]; then
        echo "SKIP: Only $NUM_GPUS GPU(s) available, need 2 for TP=2"
    else
        stop_server
        cargo run --release -p crabinfer-server --no-default-features --features "cuda,providers" -- \
            --model "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4" \
            --serving \
            --tensor_parallel_size 2 \
            --quantization gptq \
            --gpu_memory_utilization 0.90 \
            --host 0.0.0.0 --port $PORT &

        wait_for_server
        test_generation "qwen25-72b-tp2-gptq" "Write a detailed essay about the history of artificial intelligence" 500

        echo "--- Multi-GPU Memory ---"
        nvidia-smi

        stop_server
    fi
    echo "Phase 4 complete!" | tee -a "$RESULTS_DIR/summary.txt"
fi

echo ""
echo "=========================================="
echo "All phases complete!"
echo "Results saved to $RESULTS_DIR/"
echo "=========================================="
cat "$RESULTS_DIR/summary.txt" 2>/dev/null || true
