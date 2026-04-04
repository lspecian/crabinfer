---
phase: 06-embedding-model-loader
plan: 01
subsystem: serving/models
tags: [bert, embedding, safetensors, architecture-detection, nomic-bert]
dependency_graph:
  requires: []
  provides: [BertEmbeddingRunner, NomicBertRunner, masked_mean_pool, ModelArchitecture::Bert, ModelArchitecture::NomicBert]
  affects: [crabinfer-core/src/serving/safetensors_loader.rs, crabinfer-core/src/serving/models/]
tech_stack:
  added: [candle-transformers::models::bert::BertModel, candle_nn::LayerNorm, candle_nn::embedding]
  patterns: [TDD, masked-mean-pooling, RoPE-inline, SwiGLU-split-fc1]
key_files:
  created: [crabinfer-core/src/serving/models/bert.rs]
  modified:
    - crabinfer-core/src/serving/models/mod.rs
    - crabinfer-core/src/serving/safetensors_loader.rs
decisions:
  - "BertEmbeddingRunner uses candle-transformers BertModel::load() for full encoder forward; VarBuilder::from_tensors used in tests"
  - "NomicBertRunner implements custom encoder inline (not candle-transformers) — nomic-bert has non-standard QKV fused weight and SwiGLU with combined fc1"
  - "clone_model() reloads from disk for embedding models (not cheaply clonable) with panic on failure"
  - "broadcast_matmul via unsqueeze(0) pattern for 3D x 2D tensor matmul (candle requires same rank)"
  - "masked_mean_pool uses F32 casting + clamp to avoid division by zero / NaN"
  - "NomicBert RoPE uses cos/sin split at half — rotate-half convention matching HF transformers"
metrics:
  duration: 6 minutes
  completed_date: "2026-04-04"
  tasks: 2
  files: 3
---

# Phase 06 Plan 01: BERT/NomicBert Embedding Runners Summary

**One-liner:** BERT and NomicBert embedding runners implementing ModelRunner::embed() with masked mean pooling, L2 normalization, and full encoder forward passes.

## What Was Built

Two encoder-only embedding model runners added to the serving stack:

1. **`BertEmbeddingRunner`** — wraps `candle_transformers::models::bert::BertModel`. Loads any standard BERT-compatible model (BERT, RoBERTa, gte-small) via the existing candle-transformers implementation. `embed()` reshapes flat token IDs to `[1, seq_len]`, creates zero token_type_ids and ones attention_mask, runs the full encoder, then applies masked mean pool + L2 normalization to return a `[hidden_size]` tensor.

2. **`NomicBertRunner`** — custom encoder for nomic-embed-text-v1.5. Implements the non-standard architecture inline:
   - Word embeddings only (no position embeddings — RoPE applied in attention)
   - Pre-norm LayerNorm (standard, not RMSNorm)
   - Bidirectional attention with fused QKV weight `Wqkv` shape `[3*hidden, hidden]`
   - RoPE with base=1000 (not 10000), applied to Q and K
   - SwiGLU MLP with combined `fc1` weight `[2*n_inner, n_embd]` split into gate+up
   - Final LayerNorm after all layers

3. **`masked_mean_pool`** — shared helper, `[batch, seq_len, hidden_size]` + `[batch, seq_len]` mask → `[batch, hidden_size]` L2-normalized embeddings. Excludes padding tokens via broadcast multiply + sum.

4. **Architecture detection** — `ModelArchitecture::Bert` and `ModelArchitecture::NomicBert` variants added. `detect()` checks NomicBert before Bert (more specific first). Dispatch in `load_model_from_safetensors_with_backend` before existing Phi3/Mistral cases.

## Test Results

- 22 architecture detection tests: all pass (including new Bert, NomicBert, RoBERTa)
- 8 bert module tests: all pass
- 62 serving::models total: all pass (no regressions)
- Workspace compiles cleanly

## Deviations from Plan

**1. [Rule 1 - Bug] broadcast_matmul pattern for 3D x 2D tensor matmul**
- **Found during:** Task 2 NomicBert embed shape test
- **Issue:** `candle matmul` requires same-rank tensors. NomicBertLayer used `x_norm.matmul(&wqkv.t()?)` where `x_norm` is `[batch, seq, hidden]` and `wqkv.t()` is `[hidden, 3*hidden]` — rank mismatch.
- **Fix:** Added `unsqueeze(0)` to weight tensors before `broadcast_matmul`: `wqkv.t()?.unsqueeze(0)?` and `fc1.t()?.unsqueeze(0)?`.
- **Files modified:** crabinfer-core/src/serving/models/bert.rs
- **Commit:** 01be423

**2. [Rule 1 - Bug] Scalar division in attention (no broadcast_div needed)**
- **Found during:** Task 2 implementation
- **Issue:** Original code used `broadcast_div` with a full scalar tensor; candle supports `/ f64` operator directly for scalars.
- **Fix:** Replaced with `(attn_scores / scale)?` where `scale: f64`.
- **Files modified:** crabinfer-core/src/serving/models/bert.rs
- **Commit:** 01be423

## Self-Check: PASSED

- crabinfer-core/src/serving/models/bert.rs: FOUND
- crabinfer-core/src/serving/models/mod.rs: FOUND
- crabinfer-core/src/serving/safetensors_loader.rs: FOUND
- Commit 01be423: FOUND
