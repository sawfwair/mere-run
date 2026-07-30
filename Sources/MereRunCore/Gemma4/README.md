# Gemma4

Gemma 4 chat model runtime and tokenizer/template support.

- `Gemma4Config.swift`: typed text model configuration.
- `Gemma4TokenizerAndTemplate.swift`: chat-template and tokenizer boundary.
- `Gemma4CanonicalChatTemplate.swift`: checksum-gated canonical-template overlay
  for known stale Google/MLX model packages. The E4B generation primer remains
  separate from the shared 12B/26B/31B template.
- `Gemma4Model.swift`: native model layers.
- `Gemma4Generator.swift`: `ChatGenerator` integration.
- `Gemma4ToolParser.swift`: tool-call parsing.

Keep OpenAI-style tool/message adaptation typed before passing into tokenizer
library boundaries.

Canonical templates are applied only when the package template has the exact
known-stale SHA-256 and the decoded model profile matches a released Gemma 4
shape. Current or custom package templates remain authoritative.

## Terminal text-prefill A/B

`MERERUN_GEMMA4_TERMINAL_PREFILL_ROW=1` enables a text-only experimental path
for eligible batch-one, multi-token prefills. Every final-layer K/V row is
still normalized, rotated, and appended to the cache, while Q, attention
output, the residual chain, and the final MLP run only for the consumed last
row. Unified vision inputs, shared-K/V layouts, per-layer-input layouts,
single-token decode, speculation, and ineligible final layers keep the full
graph.

Installed-checkpoint Metal A/Bs on July 30, 2026 measured roughly 2.0% lower
prefill time for Gemma 4 12B BF16, 2.2% for Gemma 4 12B 4-bit, and 3.4% for
Gemma 4 Turbo. Greedy responses matched on the exercised behavior cases, but
the BF16 checkpoint's full logit vector differed by as much as 0.25 because
single-query and full-query attention can select different reduction kernels.
The path therefore remains opt-in until sampled-distribution and broader
quality gates pass.
