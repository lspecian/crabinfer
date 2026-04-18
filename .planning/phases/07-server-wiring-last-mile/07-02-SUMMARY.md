---
phase: 07-server-wiring-last-mile
plan: "02"
subsystem: api
tags: [cache, tenant-isolation, sampling-params, openai, anthropic, serving]

# Dependency graph
requires:
  - phase: 04-prefix-caching
    provides: "SamplingParams.cache_salt field and BlockHash::from_tokens_salted() for tenant-isolated KV cache hashing"
provides:
  - "cache_salt: Option<String> on ChatCompletionRequest with #[serde(default)]"
  - "cache_salt: Option<String> on MessagesRequest with #[serde(default)]"
  - "cache_salt propagated at all 4 SamplingParams construction sites (2 openai.rs + 2 anthropic.rs)"
affects:
  - "07-server-wiring-last-mile (remaining plans)"
  - "integration tests touching serving engine"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CrabInfer extension fields grouped alongside priority on request types with #[serde(default)]"
    - "cache_salt: req.cache_salt.clone() propagation pattern at SamplingParams construction sites"

key-files:
  created: []
  modified:
    - crabinfer-server/src/types/openai.rs
    - crabinfer-server/src/types/anthropic.rs
    - crabinfer-server/src/routes/openai.rs
    - crabinfer-server/src/routes/anthropic.rs

key-decisions:
  - "cache_salt placed adjacent to priority field in ChatCompletionRequest for consistent CrabInfer extension grouping"
  - "serde(default) ensures omitting cache_salt from request body preserves None — fully backward compatible"
  - "guided.rs SamplingParams sites intentionally excluded per PCCH-01 scope (chat surfaces only)"

patterns-established:
  - "CrabInfer API extensions use #[serde(default)] + Option<T> on request structs, placed after standard OpenAI/Anthropic fields"

requirements-completed: [PCCH-01]

# Metrics
duration: 4min
completed: 2026-04-18
---

# Phase 07 Plan 02: cache_salt API wiring Summary

**`cache_salt: Option<String>` wired end-to-end from HTTP request body through all 4 SamplingParams construction sites, activating tenant-isolated KV cache hashing (PCCH-01 closure)**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-18T05:23:14Z
- **Completed:** 2026-04-18T05:28:01Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Added `cache_salt: Option<String>` with `#[serde(default)]` to `ChatCompletionRequest` (openai.rs) and `MessagesRequest` (anthropic.rs)
- Propagated `cache_salt: req.cache_salt.clone()` to all 4 SamplingParams construction sites: `serving_chat_completions` and `serving_chat_completions_stream` in openai.rs, `serving_messages` and `serving_messages_stream` in anthropic.rs
- Added 5 tests: 4 deserialization tests (2 per API) + 1 compile-time marker test in routes/openai.rs
- PCCH-01 closed: clients can now submit `{"cache_salt": "tenant-foo"}` and trigger tenant-isolated prefix cache block hashing

## Task Commits

Each task was committed atomically:

1. **Task 1: Add cache_salt field to ChatCompletionRequest and MessagesRequest types with deserialization tests** - `d0e70ee` (feat)
2. **Task 2: Propagate cache_salt to all four SamplingParams construction sites (openai.rs + anthropic.rs)** - `a67b5c1` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `crabinfer-server/src/types/openai.rs` - Added `cache_salt` field + 2 deserialization tests
- `crabinfer-server/src/types/anthropic.rs` - Added `cache_salt` field + 2 deserialization tests
- `crabinfer-server/src/routes/openai.rs` - Propagated cache_salt at both SamplingParams sites; added marker test
- `crabinfer-server/src/routes/anthropic.rs` - Propagated cache_salt at both SamplingParams sites

## Decisions Made
- `cache_salt` placed adjacent to `priority` in `ChatCompletionRequest` to group CrabInfer extensions consistently
- `#[serde(default)]` ensures backward compatibility — requests without `cache_salt` produce `None` exactly as before
- `routes/guided.rs` SamplingParams sites excluded per PCCH-01 scope (guided endpoint is separate from OpenAI + Anthropic chat surfaces)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None - all 4 SamplingParams construction sites updated cleanly. Build and 87 tests pass.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Tenant isolation via `cache_salt` is now fully activatable from the API
- `routes/guided.rs` SamplingParams sites could be extended in a follow-up plan if guided endpoint tenant isolation is needed
- All 87 crabinfer-server lib tests green

---
*Phase: 07-server-wiring-last-mile*
*Completed: 2026-04-18*
