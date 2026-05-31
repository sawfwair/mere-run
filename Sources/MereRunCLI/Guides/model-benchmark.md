# Model Benchmark

## Purpose

Run focused local model benchmarks that measure implementation tradeoffs without changing serving defaults.

```bash
mere.run model benchmark gemma4-kv \
  --model text-chat-gemma4-turbo \
  --decode-tokens 48 \
  --json
```

Run a small promotion matrix by varying prompt fixture size and decode length:

```bash
mere.run model benchmark gemma4-kv \
  --model text-chat-gemma4-turbo \
  --prompt-repeat-values 32,128,220 \
  --decode-token-values 32,128 \
  --json
```

## Gemma4 KV Benchmark

`model benchmark gemma4-kv` runs a fixed-token comparison for the selected
Gemma4 checkpoint:

- `default`: the model's normal Gemma4 KV cache settings.
- `polar2`: model-default prefill, then packed 2-bit PolarKV from token 0 for decode.

The command disables EOS stopping so both variants decode exactly
`--decode-tokens`. Output includes prompt tokens, generated tokens, load time,
prefill time, KV conversion time, decode time, TTFT, prefill tok/s, decode tok/s,
end-to-end tok/s, and process resident memory before and after each variant.

## Prompt Control

Use one of:

- `--prompt`: inline prompt text.
- `--prompt-file`: UTF-8 prompt file.
- `--prompt-repeat`: repeat count for the built-in deterministic fixture.
- `--prompt-repeat-values`: comma-separated fixture sizes for a benchmark matrix.
- `--decode-token-values`: comma-separated decode lengths for a benchmark matrix.

The fixture prompt is for runtime comparison only; it is not a quality eval.

## Notes

- Pull `text-chat-gemma4-turbo` before running the benchmark.
- Prefer release builds for final numbers.
- Treat memory values as process resident snapshots, not peak memory.

## Sources

- `Sources/MereRunCLI/Commands/ModelBenchmarkCommand.swift`
- `Sources/MereRunCore/Gemma4/Gemma4Generator.swift`
- `Sources/MereRunCore/Gemma4/Gemma4KVQuantization.swift`
