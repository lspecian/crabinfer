# CrabInfer Server — multi-stage Dockerfile for Linux (CPU-only)
#
# Build:
#   docker build -t crabinfer-server .
#
# Run:
#   docker run -p 8080:8080 -v /path/to/models:/models crabinfer-server \
#     --model /models/your-model.gguf

# ---------- Stage 1: Build ----------
FROM rust:1.82-bookworm AS builder

WORKDIR /build

# Copy workspace manifests first for layer caching
COPY Cargo.toml Cargo.lock ./
COPY crabinfer-core/Cargo.toml crabinfer-core/Cargo.toml
COPY crabinfer-server/Cargo.toml crabinfer-server/Cargo.toml
COPY crabinfer-cli/Cargo.toml crabinfer-cli/Cargo.toml

# Create stub source files so cargo can resolve dependencies
RUN mkdir -p crabinfer-core/src crabinfer-server/src crabinfer-cli/src && \
    echo "fn main() {}" > crabinfer-core/src/lib.rs && \
    echo "fn main() {}" > crabinfer-server/src/lib.rs && \
    echo "fn main() {}" > crabinfer-server/src/main.rs && \
    echo "fn main() {}" > crabinfer-cli/src/main.rs

# Pre-build dependencies (cached unless Cargo.toml/lock changes)
RUN cargo build --release -p crabinfer-server \
    --no-default-features --features "providers" 2>/dev/null || true

# Copy real source
COPY crabinfer-core/ crabinfer-core/
COPY crabinfer-server/ crabinfer-server/
COPY crabinfer-cli/ crabinfer-cli/

# Touch sources to invalidate the stub builds
RUN touch crabinfer-core/src/lib.rs crabinfer-server/src/lib.rs crabinfer-server/src/main.rs

# Build server binary (CPU-only: no Metal on Linux)
RUN cargo build --release -p crabinfer-server \
    --no-default-features --features "providers"

# ---------- Stage 2: Runtime ----------
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/crabinfer-server /usr/local/bin/crabinfer-server

# Default model directory
RUN mkdir -p /models

EXPOSE 8080

ENTRYPOINT ["crabinfer-server"]
CMD ["--model", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8080"]
