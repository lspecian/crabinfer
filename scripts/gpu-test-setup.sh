#!/bin/bash
set -euo pipefail

# CrabInfer GPU Test Setup — run on Vast.ai instance
# Usage: bash gpu-test-setup.sh

echo "=== CrabInfer GPU Test Setup ==="
echo ""

# 1. System info
echo "--- GPU Info ---"
nvidia-smi
echo ""
nvcc --version 2>/dev/null || echo "nvcc not found, will use NVRTC runtime compilation"
echo ""

# 2. Install Rust if not present
if ! command -v cargo &>/dev/null; then
    echo "--- Installing Rust ---"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"
echo ""

# 3. Clone repo
REPO_DIR="$HOME/crabinfer"
if [ ! -d "$REPO_DIR" ]; then
    echo "--- Cloning CrabInfer ---"
    git clone https://github.com/lspecian/crabinfer.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull origin main
echo "Commit: $(git log --oneline -1)"
echo ""

# 4. Set CUDA paths
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
echo "CUDA_HOME=$CUDA_HOME"
echo ""

# 5. Build with CUDA features (release mode)
echo "=== Building with CUDA features ==="
time cargo build --release -p crabinfer-core --no-default-features --features "cuda,providers" 2>&1
time cargo build --release -p crabinfer-server --no-default-features --features "cuda,providers" 2>&1
time cargo build --release -p crabinfer-cli --no-default-features --features "cuda,providers" 2>&1
echo ""
echo "=== Build complete ==="

# 6. Run CUDA test suite
echo ""
echo "=== Running CUDA test suite ==="
cargo test -p crabinfer-core --no-default-features --features "cuda" --lib 2>&1 | tee /tmp/cuda-test-results.txt
echo ""
cargo test -p crabinfer-server --no-default-features 2>&1 | tee -a /tmp/cuda-test-results.txt
echo ""
echo "Test results saved to /tmp/cuda-test-results.txt"

# 7. Build with flash-attn
echo ""
echo "=== Building with FlashAttention-2 ==="
time cargo build --release -p crabinfer-server --no-default-features --features "cuda,flash-attn,providers" 2>&1
echo ""
echo "=== Setup complete! Ready for model testing ==="
