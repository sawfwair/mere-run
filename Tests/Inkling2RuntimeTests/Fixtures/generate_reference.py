"""Independent NumPy reference for the published Inkling 2 equations.

Run with Python 3 and NumPy. No Swift or MLX code is imported. The fixture
exercises both attention types, depth residual boundaries, clipping, softcaps,
router correction, and static/dynamic convolution. All reference math is FP64.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
rng = np.random.default_rng(20261113)
config = json.loads((ROOT / "config.json").read_text())
c = config["text_config"]
c.update(hidden_size=4, num_hidden_layers=4, vocab_size=12, unpadded_vocab_size=11,
         num_attention_heads=2, num_key_value_heads=1, head_dim=4,
         swa_num_attention_heads=2, swa_num_key_value_heads=1, swa_head_dim=4,
         sliding_window_size=3, local_layer_ids=[0, 1, 3], d_rel=2, rel_extent=3,
         model_max_length=32, log_scaling_n_floor=2, sconv_kernel_size=3,
         qk_l2_clip_threshold=0.6, rel_l2_clip_threshold=0.4,
         n_layers_per_attn_res_block=2, n_routed_experts=3, num_experts_per_tok=2,
         intermediate_size=4, mlp_gate_softcap=0.4, mlp_up_softcap=0.7,
         route_scale=1.5, final_logit_softcapping=0.8)
config["eos_token_id"] = 10
v = config["vision_config"]
v.update(hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
         native_patch_size=2, adapter_merge_size=2, projection_hidden_dim=6,
         mlp_ratio=2, posemb_grid_size=3)
config.pop("mtp_config", None)
(ROOT / "tiny-config.json").write_text(json.dumps(config, indent=2) + "\n")
weights = {}


def add(key, shape, norm=False):
    weights[key] = rng.normal(0, 0.25, shape) + (1 if norm else 0)


p = "model.llm."
add(p + "embed.weight", (12, 4))
for key in ["embed_norm.weight", "norm.weight"]:
    add(p + key, (4,), True)
add(p + "unembed.weight", (12, 4))
add(p + "attn_res.output_query", (4,))
for i in range(4):
    l = p + f"layers.{i}."
    local = i != 2
    if i:
        add(p + f"attn_res.attn_queries.{i - 1}", (4,) if local else (3, 4))
    add(p + f"attn_res.mlp_queries.{i}", (4,))
    for key in (["attn_norm"] if local else ["q_input_norm", "k_input_norm", "v_input_norm"]) + ["mlp_norm"]:
        add(l + key + ".weight", (4,), True)
    for key in ["attn_sconv", "mlp_sconv"]:
        add(l + key + ".weight", (4, 1, 3))
    for key, shape in {"wq_du": (8, 4), "wk_dv": (4, 4), "wv_dv": (4, 4),
                       "wr_du": (4, 4), "wo_ud": (4, 8), "gate_attn": (8, 4), "attn_sink_proj": (2, 4)}.items():
        add(l + "attn." + key + ".weight", shape)
    add(l + "attn.q_l2_clip.scale", (2, 4), True)
    add(l + "attn.rel_l2_clip.scale", (2, 2), True)
    add(l + "attn.rel_logits_proj.proj", (2, 3))
    for key in ["k_sconv", "v_sconv"]:
        if local:
            add(l + "attn." + key + ".weight", (4, 1, 3))
        else:
            add(l + "attn." + key + ".static_w", (3, 4))
            add(l + "attn." + key + ".weight_proj", (3, 4))
    add(l + "mlp.gate.weight", (3, 4))
    weights[l + "mlp.gate.bias"] = np.array([-0.4, 0.3, 0.1])
    add(l + "mlp.experts.w13_weight", (3, 8, 4))
    add(l + "mlp.experts.w2_weight", (3, 4, 4))

vp = "model.visual."
for key, shape in {"backbone.embeddings.class_embedding": (8,), "backbone.embeddings.cls_position_embedding": (8,),
                   "backbone.embeddings.patch_embedding.weight": (8, 12), "backbone.embeddings.position_embedding": (9, 8),
                   "backbone.ln_pre.weight": (8,), "backbone.ln_pre.bias": (8,),
                   "proj_in.weight": (6, 32), "proj_out.weight": (4, 6)}.items():
    add(vp + key, shape, key.endswith("ln_pre.weight"))
for i in range(2):
    l = vp + f"backbone.layers.{i}."
    for key in ["ln_1", "ln_2"]:
        add(l + key + ".weight", (8,), True)
        add(l + key + ".bias", (8,))
    for key in ["ls_1", "ls_2"]:
        add(l + key + ".gamma", (8,))
    for key, output, input_ in [("self_attn.q_proj", 8, 8), ("self_attn.k_proj", 8, 8),
                               ("self_attn.v_proj", 8, 8), ("self_attn.out_proj", 8, 8),
                               ("mlp.c_fc", 16, 8), ("mlp.c_proj", 8, 16)]:
        add(l + key + ".weight", (output, input_))
        add(l + key + ".bias", (output,))


def rms(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def softmax(x):
    y = np.exp(x - x.max(axis=-1, keepdims=True))
    return y / y.sum(axis=-1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clip(x, threshold):
    return x * np.minimum(1, threshold / (np.sqrt(np.sum(x * x, axis=-1, keepdims=True)) + 1e-6))


def norm(x, key):
    return rms(x) * weights[key + ".weight"]


def linear(x, key, bias=False):
    return x @ weights[key + ".weight"].T + (weights[key + ".bias"] if bias else 0)


def mix(blocks, partial, key):
    values = rms(np.stack(blocks + [partial], axis=-2))
    query = weights[key].reshape(-1, 4)
    return np.moveaxis(softmax(query @ values.swapaxes(-1, -2) / 2) @ values, -2, 0)


def conv(x, key, dynamic=None):
    y = x.copy()
    for t in range(len(x)):
        for lag in range(min(3, t + 1)):
            if dynamic is None:
                tap = weights[key + ".weight"][:, 0, 2 - lag]
            else:
                tap = weights[key + ".static_w"][lag] + (dynamic[t] @ weights[key + ".weight_proj"].T)[lag]
            y[t] += x[t - lag] * tap
    return y / np.sqrt(4 if dynamic is None else 7)


def attention(inputs, i):
    a = p + f"layers.{i}.attn."
    qx, kx, vx = inputs
    n = len(qx)
    q = clip(linear(qx, a + "wq_du").reshape(n, 2, 4) * weights[a + "q_l2_clip.scale"], 0.6)
    k = clip(conv(linear(kx, a + "wk_dv"), a + "k_sconv", kx if i == 2 else None).reshape(n, 1, 4), 0.6)
    value = rms(conv(linear(vx, a + "wv_dv"), a + "v_sconv", vx if i == 2 else None).reshape(n, 1, 4))
    relative = clip(linear(qx, a + "wr_du").reshape(n, 2, 2) * weights[a + "rel_l2_clip.scale"], 0.4)
    if i == 2:
        tau = 1 + 0.1 * np.log(np.maximum(np.arange(1, n + 1) / 2, 1))
        q *= tau[:, None, None]
        relative *= tau[:, None, None]
    bias = relative @ weights[a + "rel_logits_proj.proj"]
    sink = linear(qx, a + "attn_sink_proj")
    output = np.zeros((n, 2, 4))
    for t in range(n):
        for head in range(2):
            allowed = list(range(0 if i == 2 else max(0, t - 2), t + 1))
            logits = [q[t, head] @ k[s, 0] / 2 + (bias[t, head, t - s] if t - s < 3 else 0) for s in allowed]
            prob = softmax(np.array(logits + [sink[t, head]]))[:-1]
            output[t, head] = sum(prob[j] * value[s, 0] for j, s in enumerate(allowed))
    gate = sigmoid(linear(qx, a + "gate_attn"))
    return linear(output.reshape(n, 8) * gate, a + "wo_ud")


def moe(x, key):
    scores = sigmoid(linear(x, key + "gate"))
    indices = np.argsort(-(scores + weights[key + "gate.bias"]), axis=-1)[:, :2]
    output = np.zeros_like(x)
    for t in range(len(x)):
        routing = scores[t, indices[t]]
        routing = routing / routing.sum() * 1.5
        for j, expert in enumerate(indices[t]):
            gu = (x[t] @ weights[key + "experts.w13_weight"][expert].T).reshape(4, 2)
            g, u = gu[:, 0], gu[:, 1]
            hidden = np.tanh(g / 0.4) * 0.4 * sigmoid(g) * np.tanh(u / 0.7) * 0.7
            output[t] += routing[j] * (hidden @ weights[key + "experts.w2_weight"][expert].T)
    return output


tokens = [1, 4, 2, 6, 3, 5, 1]
h = norm(weights[p + "embed.weight"][tokens], p + "embed_norm")
blocks = []
for i in range(4):
    l = p + f"layers.{i}."
    residual = h.copy()
    inputs = [h] * 3 if i == 0 else list(mix(blocks, h, p + f"attn_res.attn_queries.{i - 1}"))
    if i != 2:
        inputs = [norm(inputs[0], l + "attn_norm")] * 3
    else:
        inputs = [norm(x, l + name + "_input_norm") for x, name in zip(inputs, ["q", "k", "v"])]
    a = conv(attention(inputs, i), l + "attn_sconv")
    if i % 2 == 0:
        blocks.append(residual)
        partial = a
    else:
        partial = residual + a
    mixed = mix(blocks, partial, p + f"attn_res.mlp_queries.{i}")[0]
    h = partial + conv(moe(norm(mixed, l + "mlp_norm"), l + "mlp."), l + "mlp_sconv")
h = norm(mix(blocks, h, p + "attn_res.output_query")[0], p + "norm")
logits = np.tanh(linear(h, p + "unembed") / 0.8) * 0.8


def layer_norm(x, key):
    return ((x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
            * weights[key + ".weight"] + weights[key + ".bias"])


def gelu(x):
    import math
    return x * 0.5 * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))


# An asymmetric 2x3 token image exercises interpolation and merged patch order.
pixels = rng.uniform(-1, 1, (6, 4, 4, 3))
patches = pixels.reshape(6, 2, 2, 2, 2, 3).transpose(0, 1, 3, 2, 4, 5).reshape(-1, 12)
e = vp + "backbone.embeddings."
embedded = linear(patches, e + "patch_embedding")
coords = [(r * 2 + y, c_ * 2 + x) for r in range(2) for c_ in range(3) for y in range(2) for x in range(2)]
for index, (y, x) in enumerate(coords):
    fy, fx = max(0, (y + 0.5) * 3 / 4 - 0.5), max(0, (x + 0.5) * 3 / 6 - 0.5)
    y0, x0 = int(fy), int(fx)
    for yy, wy in [(y0, 1 - (fy - y0)), (min(y0 + 1, 2), fy - y0)]:
        for xx, wx in [(x0, 1 - (fx - x0)), (min(x0 + 1, 2), fx - x0)]:
            embedded[index] += wy * wx * weights[e + "position_embedding"][yy * 3 + xx]
cls = weights[e + "class_embedding"] + weights[e + "cls_position_embedding"]
vh = layer_norm(np.concatenate([cls[None], embedded]), vp + "backbone.ln_pre")
angles = np.array([[x, x, y, y] for y, x in [(0, 0)] + [(y + 1, x + 1) for y, x in coords]])[:, None]
for i in range(2):
    l = vp + f"backbone.layers.{i}."
    z = layer_norm(vh, l + "ln_1")
    q, k, value = [linear(z, l + "self_attn." + n, True).reshape(-1, 2, 4) for n in ["q_proj", "k_proj", "v_proj"]]
    def rotate(z):
        pair = z.reshape(-1, 2, 2, 2)
        rotated = np.stack([-pair[..., 1], pair[..., 0]], axis=-1).reshape(z.shape)
        return z * np.cos(angles) + rotated * np.sin(angles)
    q, k = rotate(q).transpose(1, 0, 2), rotate(k).transpose(1, 0, 2)
    attended = (softmax(q @ k.swapaxes(-1, -2) / 2) @ value.transpose(1, 0, 2)).transpose(1, 0, 2).reshape(-1, 8)
    vh += weights[l + "ls_1.gamma"] * linear(attended, l + "self_attn.out_proj", True)
    vh += weights[l + "ls_2.gamma"] * linear(gelu(linear(layer_norm(vh, l + "ln_2"), l + "mlp.c_fc", True)), l + "mlp.c_proj", True)
image_features = linear(gelu(linear(vh[1:].reshape(6, 32), vp + "proj_in")), vp + "proj_out")

payload = {k: {"shape": list(a.shape), "values": a.flatten().tolist()} for k, a in weights.items()}
(ROOT / "weights.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")
(ROOT / "reference.json").write_text(json.dumps({"tokens": tokens, "logits": logits[:, :11].flatten().tolist(),
    "pixels": pixels.flatten().tolist(), "imageFeatures": image_features.flatten().tolist()}, indent=2) + "\n")
