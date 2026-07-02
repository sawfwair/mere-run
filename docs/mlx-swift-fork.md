# mlx-swift fork dependency

`Package.swift` pins mlx-swift to a fork instead of upstream:

- Repository: <https://github.com/sawfwair/mlx-swift>
- Pinned revision: `7450480eb9f9c0a41649bd3348fce13f7892ad29`
  (branch `mere-run/compiled-call-evallock` = upstream **v0.31.4** `dc43e62`
  plus one commit)
- Same patch rebased onto **v0.31.6** and pre-pushed for the next upgrade:
  branch `mere-run/compiled-call-evallock-0.31.6`
  (`f004aa1d9734494cc36c53b828d861a6159f584c`)

## What the patch changes and why

One commit: `CompiledFunction.innerCall` no longer takes the global `evalLock`
around `mlx_detail_compile` + `mlx_closure_apply`. A cache-hit apply only
constructs graph nodes — the same work every ordinary MLX op does without a
lock — and a tracing call that re-enters eval takes the recursive `evalLock`
itself. The per-function `NSLock` (held by `call`) still serializes each
compiled function.

Measured on an M4 Max (Gemma4 12B decode, 96 compiled calls/token):

| | per-token wall | per-call cost |
|---|---|---|
| upstream 0.31.4 | 70.8 ms (2.6× slower than uncompiled) | ~0.45 ms |
| patched | 28.2 ms (parity) | ~5 µs (= one raw op) |

This affects more than explicit `compile(...)` users: every MLXNN compiled
activation (`geluApproximate`, the swiglu helpers, etc.) pays the per-call
cost, so any per-token decode loop built on MLXNN inherits the cliff.

## Upstream status

As of 2026-07-02, upstream latest is v0.31.6 and
`Source/MLX/Transforms+Compile.swift` is unchanged since 2026-05-07 — the
issue exists in every released version through 0.31.6. The patch is
upstream-PR-ready; once it (or an equivalent fix) lands, drop the fork by
restoring `.package(url: "https://github.com/ml-explore/mlx-swift", from: ...)`.

## Upgrading mlx-swift while the fork is needed

1. In the fork, branch from the new upstream tag and cherry-pick the patch
   commit (it has applied cleanly across 0.31.4 → 0.31.6; if it ever
   conflicts, re-read the current `innerCall` before resolving).
2. Push the branch, update the `revision:` in `Package.swift`, and run
   `swift package resolve`.
3. Re-verify:
   - `MERERUN_BENCHMARK_COMPILE_OVERHEAD=1 swift test --filter
     MLXCompiledFunctionOverheadTests` — per-call µs should stay ~raw-op cost.
   - `MERERUN_TEST_MLX_DEVICE=gpu swift test --filter
     "Gemma4DecodeFusedKernelsTests|Gemma4KVQuantizationTests|Gemma4SlidingKVCacheDecodeStateTests"`
   - A `model benchmark gemma4-kv` decode spot-check.

## Scope notes

- The Linux/CUDA build path (`useLinuxPrebuiltMLX`) does not consume this SPM
  dependency and is unaffected by the patch either way.
- Do not edit `.build/checkouts/` to change dependency behavior — checkout
  edits silently vanish on the next `swift package resolve/update`. That
  failure mode is why the fork exists.
