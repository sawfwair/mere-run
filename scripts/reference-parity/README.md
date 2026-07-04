# Reference-parity harness (Q35 / Qwen-family)

Byte-identical greedy parity gates compare this runtime against itself, so they
cannot catch a whole class of bug: a systematically wrong forward pass that is
wrong the same way on both sides of a refactor. This harness compares the
native runtime against `mlx_lm` running the **same installed checkpoint**, at
two levels:

1. **Token streams** — greedy decode of the same rendered prompt on both
   stacks; any divergence in the first tokens is a red flag.
2. **Per-layer hidden states** — the runtime dumps per-stage stats when
   `MERERUN_Q35_DEBUG_LAYER_DUMP=<path>` is set (embeddings, every decoder
   layer, final norm; float32 L2 norm + leading values of the last prompt
   position). `mlxlm_layer_dump.py` produces the matching dump from `mlx_lm`,
   and `compare_layers.py` reports the first diverging stage.

This harness found the `norm_topk_prob` default bug (2026-07-04): checkpoints
that omit the key were routed with un-renormalized top-k scores, dampening
every MoE block ~25% by mid-stack and biasing logits toward repetition.

## Setup

```bash
uv venv --python 3.14 /tmp/mlxlm-venv
uv pip install --python /tmp/mlxlm-venv/bin/python mlx-lm "transformers<5"
```

## Run

```bash
# 1. reference side (writes ref_layers.jsonl + ref_ids.txt)
/tmp/mlxlm-venv/bin/python scripts/reference-parity/mlxlm_layer_dump.py \
  "$HOME/Library/Application Support/MereRun/models/text-agent-ornith-35b-mlx" \
  "<system prompt>" "<user prompt>" ref_layers.jsonl ref_ids.txt

# 2. native side (same prompts)
MERERUN_Q35_DEBUG_LAYER_DUMP=our_layers.jsonl \
MERERUN_Q35_DEBUG_PROMPT_TOKENS=our_ids.txt \
swift run -c release mere.run text chat --model text-agent-ornith-35b-mlx \
  --thinking --temperature 0 --max-tokens 4 -s "<system prompt>" -p "<user prompt>"

# 3. compare
python3 scripts/reference-parity/compare_layers.py \
  our_layers.jsonl ref_layers.jsonl our_ids.txt ref_ids.txt
```

Interpretation: prompt ids must be identical (else the bug is in the
template/tokenizer, not numerics). Norm ratios drifting ±2% late in the stack
is normal quantized-kernel accumulation noise; a *systematic* ratio away from
1.0 that compounds layer over layer is a real defect, and the first stage
where it appears names the guilty component.
