# SOV-Transcribe model runtime

This internal target owns the Whisper audio encoder, stacked-frame projector,
DFlash draft model, and greedy target verification for SOV-Transcribe-1.
`MereRunGemmaModel` owns the Gemma 4 target decoder. `AudioSTT/SOVTranscribe`
owns checkpoint loading, tokenizer boundaries, audio preparation, and requests.

The draft uses raw target embedding rows and outputs after the configured
zero-based target layers. Sliding draft layers are causal within the block;
full draft layers are bidirectional. Only verified target context enters draft
caches. On rejection, verification supplies the accepted KV rows and hidden
states. The decoder appends those rows to the untouched base cache, preserving
history evicted by the trial's sliding window. It replays only when the trial
cannot supply every proposed row. This avoids a second target forward pass on
ordinary rejected rounds.

The attention split also matches `DFlashAttention` and the mask mapping in
[the pinned upstream implementation](https://huggingface.co/salesforce-sovereign-intelligence/SOV-Transcribe-1-DFlash/blob/8c0b3f3a98507d2e3dd35b1bee0f147fa3bded05/modeling_dflash.py),
which selects sliding causal and full bidirectional masks explicitly.

Target prefill uses chunks of up to 1024 tokens. The benchmark can override this
for tuning and records the chosen size alongside verification-reuse counts.

Pipelined generation retains proposal IDs as MLX arrays,
starts draft evaluation asynchronously, and constructs verification before a
combined host read. Verified predictions supply the next anchor where possible.
Committed draft and target cache branches evaluate through the next round's
dependencies. Optional stage diagnostics are separate from release benchmarks.
Acceptance counters distinguish matching prefixes from drafts emitted before EOS.

The explicit rollback experiment uses `SOVRollbackKVCache`. Trials write into spare
linear storage and commit by changing the valid length. Compaction preserves the
pre-trial window; rejected rows stay outside the valid prefix. Full-attention
storage grows in 256-row steps. This cache is confined to SOV research modes.
Fixed and confidence-truncated draft lengths remain separate experiments.
`proposedLengths` records each verified proposal count; `adaptive-after-first`
keeps the first block unchanged and uses confidence on subsequent rounds.

Run `swift test --filter SOV` and `./scripts/check.sh`. Small numerical fixtures
and synthetic generation tests do not qualify released checkpoint quality or
Metal performance. See `docs/runtime/sov-transcribe.md` for usage and validation.

The benchmark-only `--experimental-dynamic-int8` path packs both FFN matrices
once after checkpoint loading. It keeps original BF16 parameters, uses live
activation ranges, and executes int8 TensorOps with int32 accumulation on an M5
GPU. The fastest candidate uses per-row scales and K blocks of 256 values.
The kernel also supports bounded per-block power-of-two exponents for research;
its additional rescaling costs more than it saves on these recordings.
FC2 applies an orthonormal block Hadamard transform to both
operands; the activation transform, range reduction, and packing share one GPU
kernel. The projector, attention, normalization, decoder, and full padded frame
count retain their original arithmetic. Pack preparation must run again after
any parameter replacement. This experimental route is not the default speech
provider and does not execute on the standalone ANE.

For the custom GPU tests and the independent FP32 fixtures, use:

```bash
MERERUN_TEST_MLX_DEVICE=gpu MLX_ENABLE_TF32=0 swift test \
  --filter 'SOVDynamicInt8Tests|SOVTranscribeModelTests|SOVVerifiedPrefixTests'
```

Disabling TF32 preserves the NumPy fixtures' existing FP32 tolerance. The BF16
performance benchmarks use the default runtime precision settings.
