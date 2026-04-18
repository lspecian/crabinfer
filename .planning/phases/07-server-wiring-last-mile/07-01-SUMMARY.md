---
phase: 07-server-wiring-last-mile
plan: 01
subsystem: serving
tags: [routing, worker-pool, cache-aware, config, cli, toml, env-vars]

# Dependency graph
requires:
  - phase: 05-wiring-fixes
    provides: WorkerPool::new_with_policy() and RoutingPolicy::CacheAware enum in worker_pool.rs

provides:
  - routing_policy field threaded end-to-end: TOML -> env -> CLI -> ServerConfig -> WorkerPool::new_with_policy()
  - --routing-policy CLI flag registered with Clap
  - CRABINFER_ROUTING_POLICY env var support
  - parse_routing_policy() helper with graceful unknown-string fallback
  - WORK-03 closed: cache-aware routing now activatable from user-facing config

affects: [08-server-wiring-last-mile, integration-testing, docs]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Config field threading: CrabInferConfig -> CliOverrides -> ServerConfig -> use site, using merge! macro"
    - "Unknown string graceful degradation: match arm falls back to default with tracing::warn!, never panics"

key-files:
  created: []
  modified:
    - crabinfer-server/src/config.rs
    - crabinfer-server/src/lib.rs
    - crabinfer-server/src/main.rs

key-decisions:
  - "parse_routing_policy() as private helper near WorkerPool construction site — pure function, easily unit-testable"
  - "Tasks 1 and 2 combined into single compile unit per plan note — ServerConfig field needed before config.rs tests compile"
  - "Graceful unknown-string degradation: tracing::warn! + RoundRobin fallback, never panic — matches production safety requirement"

patterns-established:
  - "New server config fields follow 7-step pattern: CrabInferConfig -> CliOverrides -> merge_cli_overrides -> to_server_config -> apply_cli_overrides -> apply_env_overrides -> use site"

requirements-completed: [WORK-03]

# Metrics
duration: 12min
completed: 2026-04-18
---

# Phase 07 Plan 01: Server Wiring Last Mile Summary

**routing_policy: Option<String> wired end-to-end through TOML/env/CLI config pipeline into WorkerPool::new_with_policy(), closing WORK-03 (cache-aware routing unreachable from user config)**

## Performance

- **Duration:** 12 min
- **Started:** 2026-04-18T00:00:00Z
- **Completed:** 2026-04-18T00:12:00Z
- **Tasks:** 2 (executed as 1 combined atomic change)
- **Files modified:** 3

## Accomplishments
- `routing_policy: Option<String>` added to `CrabInferConfig`, `CliOverrides`, and `ServerConfig`
- `WorkerPool::new(handles)` replaced with `WorkerPool::new_with_policy(handles, policy, block_size)` at lib.rs construction site
- `--routing-policy` Clap arg registered in Cli struct, propagated through `CliOverrides` to `ServerConfig`
- `CRABINFER_ROUTING_POLICY` env var supported via `apply_env_overrides`
- `parse_routing_policy()` private helper with graceful unknown-string fallback (warning + RoundRobin, never panic)
- 5 new tests pass (4 config tests + 1 parse helper test), all 86 existing server tests remain green

## Task Commits

Each task was committed atomically:

1. **Tasks 1+2: Add routing_policy field to config pipeline and wire WorkerPool** - `4fca68f` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `crabinfer-server/src/config.rs` - routing_policy on CrabInferConfig and CliOverrides, merge/apply/env wiring, 4 tests
- `crabinfer-server/src/lib.rs` - routing_policy on ServerConfig, parse_routing_policy() helper, WorkerPool swap, startup log, test
- `crabinfer-server/src/main.rs` - --routing-policy Clap arg, CliOverrides literal population

## Decisions Made
- Tasks 1 and 2 combined into single compile unit (ServerConfig field needed before config.rs to_server_config compiles)
- parse_routing_policy() placed as private function near load_serving_engine for locality
- Graceful degradation pattern used: unknown routing-policy string logs warning, falls back to RoundRobin — consistent with plan requirement of no-panic on unknown values

## Deviations from Plan

None - plan executed exactly as written. The plan itself noted Tasks 1+2 should be combined and that ordering was accounted for.

## Issues Encountered
- Default feature is `metal` which requires macOS; on Linux (H100 machine) must use `--no-default-features --features cpu-only` for build/test. This is pre-existing environment behavior, not introduced by this plan.
- clippy error in crabinfer-core/src/serving/kernels/cpu_backend.rs (mutable borrow from immutable input) is pre-existing, not introduced by this plan.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- WORK-03 closed: `crabinfer serve --routing-policy cache-aware --workers 4` now activates CacheAware routing
- Default behavior (no flag, no TOML key) remains RoundRobin — fully backward compatible
- Ready for next plans in Phase 07

---
*Phase: 07-server-wiring-last-mile*
*Completed: 2026-04-18*
