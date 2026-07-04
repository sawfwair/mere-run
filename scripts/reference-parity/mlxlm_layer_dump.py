import json
import sys

import mlx.core as mx
from mlx_lm import load

def main(model_path, system_prompt, user_prompt, out_path, ids_out_path):
    model, tokenizer = load(model_path)
    ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
    )
    open(ids_out_path, "w").write(",".join(str(i) for i in ids))

    records = []

    def record(stage, hidden):
        last = hidden[0, -1, :].astype(mx.float32)
        mx.eval(last)
        norm = float(mx.sqrt((last * last).sum()).item())
        head = [float(x) for x in last[:8].tolist()]
        records.append({"stage": stage, "norm": norm, "head": head})

    # locate inner transformer: object with a .layers list and a .norm
    def find_inner(module, depth=0):
        if depth > 4:
            return None
        layers = getattr(module, "layers", None)
        if isinstance(layers, list) and layers and hasattr(module, "norm"):
            return module
        for _, child in module.children().items():
            if hasattr(child, "children"):
                found = find_inner(child, depth + 1)
                if found is not None:
                    return found
        return None

    inner = find_inner(model)
    assert inner is not None, "could not locate transformer with .layers/.norm"

    class LayerProxy:
        def __init__(self, orig, stage):
            object.__setattr__(self, "_orig", orig)
            object.__setattr__(self, "_stage", stage)

        def __call__(self, *args, **kwargs):
            out = self._orig(*args, **kwargs)
            record(self._stage, out)
            return out

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_orig"), name)

    class EmbedProxy:
        def __init__(self, orig):
            object.__setattr__(self, "_orig", orig)

        def __call__(self, x):
            out = self._orig(x)
            if out.ndim == 3 and out.shape[1] > 1:
                record("embeddings", out)
            return out

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_orig"), name)

    inner.layers = [LayerProxy(l, f"layer{i}") for i, l in enumerate(inner.layers)]
    if hasattr(inner, "embed_tokens"):
        inner.embed_tokens = EmbedProxy(inner.embed_tokens)

    orig_norm = inner.norm
    class NormProxy:
        def __call__(self, x):
            out = orig_norm(x)
            if out.ndim == 3 and out.shape[1] > 1:
                record("final_norm", out)
            return out

        def __getattr__(self, name):
            return getattr(orig_norm, name)

    inner.norm = NormProxy()

    logits = model(mx.array(ids)[None])
    mx.eval(logits)

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} stages, {len(ids)} prompt tokens")


if __name__ == "__main__":
    main(*sys.argv[1:6])
