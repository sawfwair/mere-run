# Clairic extraction runtime

Clairic and Clairic Flash reuse Qwen3.5 text and vision in `MereRunQwenModel`.
This directory owns the checkpoint catalog, tokenizer, request handling, and
slot orchestration. The model target owns the JSON slot grammar. `text extract`
selects the runtime using the managed model ID or a local checkpoint marker.

Each field gets a release-format prompt. Slots reuse a long shared prefix,
including image embeddings, by forking their hybrid caches. Bounded groups of active slots share
Qwen batched decode, with separate attention and recurrent state for each
field. A byte-level JSON grammar masks invalid tokens before greedy sampling.
The slot's confidence includes EOS and uses the model probabilities
renormalized over the admissible vocabulary. `p_null` independently scores
the bare null token on the first step. The structured response reports
grammar completion, value validity, and unique literal text or number spans.

A list expands into indexed slots and retains the order and duplicates of
the completed, non-null answers. Object slots use sorted property keys.
String length and pattern constraints and numeric bounds are validated after
generation. The native path implements ordinary slots and literal text
grounding. Cloudflare's typed heads and supplementary recovery passes remain
outside this implementation.

See [Clairic](../../../docs/runtime/clairic.md) for supported fields and media.
