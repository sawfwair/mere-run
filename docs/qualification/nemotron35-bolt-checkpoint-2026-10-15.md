# Nemotron 3.5 Bolt checkpoint qualification, 2026-10-15

The original packed checkpoint passed 12 of 12 synthetic exact-answer probes on an Apple M3 Max with 30 GPU cores and 36 GiB of unified memory. Text passed 8/8 and images passed 4/4. The complete diagnostic run, including two capped decode runs, finished in 172.7 seconds after admission.

The checkpoint remains pinned to `96a5d63027ac013c8ef37fb438f3b4ec5fbd3ba6`. All 90 weight shards matched their upstream SHA-256 hashes. Weights, the Hub cache, inputs, raw outputs, traces, and logs are stored under `/Volumes/T7/mere-run-qualification/nemotron35-bolt-2026-10-15`.

## Measurements

The debug harness used native Metal kernels, greedy decoding, and a 4,096-token context limit. Image fixtures were 512 x 512 pixels; every image request had one image. The reasoning probe enabled thinking. Other scored cases disabled it.

| Measurement | Observed result |
| --- | --- |
| Text exact-answer checks | 8/8 |
| Image exact-answer checks | 4/4 |
| MLX peak allocation | 22.96 GiB |
| Kernel peak process physical footprint | 26.38 GiB |
| Peak process RSS | 7.05 GiB |
| Debug 128-token decode runs | 6.55 and 4.47 tokens/s |
| Fresh-instance load in debug harness | 31.50 s |
| First generated piece, fresh debug instance | 35.60 s |

The two debug speed runs produced identical outputs. A background optimized build was active during this diagnostic suite. Use the separate optimized CLI measurements below for release throughput. MLX peaks include resident model weights and reset per case. RSS and kernel physical-footprint peaks are cumulative high-water marks for the process. RSS alone does not represent the Metal working set.

| Image probe | Exact answer | Full warm request | First piece |
| --- | --- | --- | --- |
| chart | `B` | 11.65 s | 11.48 s |
| count | `3` | 11.06 s | 10.79 s |
| ocr | `RIVER 4827` | 14.13 s | 13.05 s |
| spatial | `red` | 11.09 s | 10.81 s |

Prompt timing includes image decoding, resizing, vision encoding, tokenization, and language prefill. The image request timings include those stages and generation, with the model already loaded.

## Optimized CLI

Two optimized CLI text runs generated 128 tokens each at 14.92 and 16.50 tokens/s during decode. Their outputs were identical. Each run loaded a fresh model instance. The optimized chart invocation returned `B`; its prompt preparation and prefill took 11.50 seconds and the CLI-reported runtime was 41.36 seconds, including model loading (41.84 seconds of process wall time).

The maximum kernel physical footprint across the optimized CLI runs was 27.14 GiB.

## Runtime corrections found by the checkpoint

The first full load reached a 49.0 GB process footprint because Foundation file-read buffers survived alongside MLX copies. Draining autoreleased buffers at each shard boundary removed the duplicate checkpoint retention. The next run exposed flattened logits at the shared decoder boundary; the runtime now preserves batch, position, and vocabulary dimensions.

The initial image attempt then hit a Metal timeout. The final runtime requests GPU residency through the shared MLX memory manager, caps the request below Metal's recommended working set, and restores the previous limit when active work ends. It uses 64-token prefill chunks and synchronizes decode work every eight blocks. The complete text/image diagnostic suite passed after these changes.

## Reproduce optimized measurements

Run from the native Bolt worktree after `swift build -c release --product mere.run`:

```bash
MERERUN_MODELS_DIR=/Volumes/T7/mere-run-qualification/nemotron35-bolt-2026-10-15/models \
/usr/bin/time -l .build/release/mere.run text chat \
  --model vision-chat-nemotron-35-bolt-24gb --require-installed \
  --prompt "Write a numbered list of the integers from 1 to 100, one per line." \
  --max-tokens 128 --temperature 0 --top-p 1 --context-size 4096 \
  --no-thinking --stats --quiet
```

Repeat the invocation for the second fresh-instance run. The 128-token cap is
intentional for throughput measurement. The chart invocation used the same
sampling settings, `--max-tokens 32`, and the generated `chart.png` fixture.

## Scope and reproduction

These are small synthetic diagnostics. They do not establish broad capability, natural-image performance, full upstream numerical parity, long-context behavior, or capacity on a different physical device. The model's general quality qualification remains false in the receipt. No hosted CI result is claimed.

Use the [runtime guide](../runtime/nemotron-bolt.md#run-checkpoint-diagnostics) to regenerate the fixtures and run the suite. The [machine-readable receipt](./nemotron35-bolt-checkpoint-2026-10-15.json) includes every completed probe's prompt, expected answer, raw response, timing, and memory measurements. All earlier attempt logs remain on T7.
