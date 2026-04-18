---
phase: 07-server-wiring-last-mile
plan: "03"
subsystem: server-api
tags: [openai-compat, completions, legacy-api, cache-salt, pcch-01]
dependency_graph:
  requires: [07-02]
  provides: [v1-completions-route, CompletionRequest, CompletionResponse, CompletionChoice]
  affects: [crabinfer-server/src/types/openai.rs, crabinfer-server/src/routes/openai.rs, crabinfer-server/src/routes/mod.rs, crabinfer-server/src/lib.rs]
tech_stack:
  added: []
  patterns: [OpenAI legacy completions API, serving-engine-required-503, cache-salt-propagation]
key_files:
  created: []
  modified:
    - crabinfer-server/src/types/openai.rs
    - crabinfer-server/src/routes/openai.rs
    - crabinfer-server/src/routes/mod.rs
    - crabinfer-server/src/lib.rs
decisions:
  - "503 (not legacy fallback) returned when serving engine missing for /v1/completions"
  - "logprobs field serializes as JSON null (not omitted) per OpenAI client expectations"
  - "Debug derived on Usage struct to satisfy CompletionResponse derive requirements"
  - "stream:true accepted but ignored in v1 (documented in handler doc-comment)"
metrics:
  duration: "3 minutes"
  completed_date: "2026-04-18"
  tasks_completed: 2
  files_modified: 4
---

# Phase 07 Plan 03: POST /v1/completions Handler Summary

**One-liner:** OpenAI legacy text completion endpoint (`POST /v1/completions`) with 503-on-no-engine and cache_salt tenant isolation.

## What Was Built

Closed the missing-flow gap identified in the v1.0 audit: the server was returning 404 for `POST /v1/completions`. OpenAI legacy clients and many third-party tools that speak only the legacy completions API were failing silently.

### Task 1: CompletionRequest/Response/Choice Types (TDD)

Added to `crabinfer-server/src/types/openai.rs`:

- `CompletionRequest`: model (required), prompt (required), max_tokens, temperature, top_p, stream, cache_salt (all optional via `serde(default)`)
- `CompletionResponse`: id, object ("text_completion"), created, model, choices, usage
- `CompletionChoice`: text, index, finish_reason, logprobs (serializes as JSON null, not omitted — per OpenAI client expectations)
- `Usage` struct gained `#[derive(Debug)]` to satisfy `CompletionResponse` trait requirements

Four serde tests pass:
1. `test_completion_request_deser` — minimal JSON deserializes with all optionals None
2. `test_completion_request_with_optionals` — all fields populated deserialize correctly
3. `test_completion_response_ser` — response serializes with correct `object: "text_completion"`
4. `test_completion_choice_logprobs_serializes_as_null` — logprobs key present in JSON (not absent)

### Task 2: Handler, Route Registration, Startup Log

- `pub async fn completions(...)` added to `routes/openai.rs`:
  - Returns HTTP 503 with clear message when `serving_engine` is None (no panic, no legacy fallback)
  - Validates prompt is non-empty (400 if empty)
  - Propagates `cache_salt` to `SamplingParams.cache_salt` (PCCH-01 consistency with chat path)
  - Token collection loop with timeout mirrors `serving_chat_completions` pattern
  - Correct response shape: `object: "text_completion"`, `choices[0].text`, `logprobs: null`
  - TTFT metric recorded on first token
  - LoRA adapter parsed from model field (matches chat path)
  - `stream: true` accepted but ignored in v1 (documented in doc-comment)

- `.route("/v1/completions", post(openai::completions))` registered in `routes/mod.rs`

- `tracing::info!("  POST http://{}/v1/completions", addr)` added to startup log in `lib.rs`

## Verification Results

- `cargo build --package crabinfer-server --no-default-features` — clean build
- `cargo test --package crabinfer-server --lib --no-default-features` — 91 tests pass (0 failures)
- Route registration confirmed: `/v1/completions` present in both `routes/mod.rs` and `lib.rs`
- Handler confirmed: `pub async fn completions` at routes/openai.rs line 676

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added Debug derive to Usage struct**
- **Found during:** Task 1 (TDD RED → GREEN)
- **Issue:** `CompletionResponse` derives `Debug` and references `Usage`, but `Usage` lacked `#[derive(Debug)]`
- **Fix:** Added `Debug` to `Usage`'s derive list
- **Files modified:** `crabinfer-server/src/types/openai.rs`
- **Commit:** 93c8c27

**2. [Out of scope] Pre-existing clippy error in crabinfer-core**
- `crabinfer-core/src/serving/kernels/cpu_backend.rs:26` — mutable borrow from immutable input (pre-existing)
- Not introduced by this plan; logged to deferred items
- `crabinfer-server` itself has zero new clippy warnings

## Self-Check: PASSED

- `crabinfer-server/src/types/openai.rs` — modified (CompletionRequest/Response/Choice types + tests)
- `crabinfer-server/src/routes/openai.rs` — modified (completions handler)
- `crabinfer-server/src/routes/mod.rs` — modified (/v1/completions route)
- `crabinfer-server/src/lib.rs` — modified (startup log)
- Commit 93c8c27 exists (Task 1 types + tests)
- Commit f8b7a91 exists (Task 2 handler + route + log)
