"""Independent scalar DASH and tiny decoder references; Python standard library only."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent


def softmax(values):
    exps = [math.exp(x - max(values)) for x in values]
    return [x / sum(exps) for x in exps]


def entmax(values):
    z = [(x - max(values)) / 2 for x in values]
    threshold = 0
    for rho in range(1, len(z) + 1):
        mu = sum(z[:rho]) / rho
        mu2 = sum(x * x for x in z[:rho]) / rho
        trial = mu - math.sqrt(max((1 - rho * (mu2 - mu * mu)) / rho, 0))
        if trial <= z[rho - 1]:
            threshold = trial
    return [max(x - threshold, 0) ** 2 for x in z]


def attention_reference():
    count, dim = 41, 4
    def values(phase):
        return [round(math.sin((i + 1) * phase), 7) for i in range(count * dim)]
    q, k, v, iq, ik = [values(p) for p in [0.17, 0.23, 0.31, 0.41, 0.13]]
    head = [0.2, -0.1, 0.3, -0.4]
    tokens = [0, 128827] + [128815] * 10 + [128827] + [128815] * 10
    tokens += [7, 128827] + [8] * 8 + [128827] + [9] * 7
    last = tokens.index(7) - 1
    markers, query, images = 0, [], []
    for position, token in enumerate(tokens):
        markers += token == 128827
        query.append(markers - 1 if position <= last else max(-1, markers - 3))
        images.append(markers - 1 if token == 128815 else -1)
    blocks = (count + 1) // 2
    summaries = []
    for block in range(blocks):
        rows = [ik[j * dim:(j + 1) * dim] for j in range(block * 2, min(count, block * 2 + 2))]
        probabilities = softmax([sum(a * b for a, b in zip(row, head)) / 2 for row in rows])
        summaries.append([sum(p * row[c] for p, row in zip(probabilities, rows)) for c in range(dim)])
    output = []
    for position in range(count):
        remote = [b for b in range(blocks) if b <= position // 2 - 2]
        same = [b for b in remote if any(page >= 0 and page == query[position]
                for page in images[b * 2:min(count, b * 2 + 2)])]
        candidates = same or remote
        scores = [sum(iq[position * dim + c] * summaries[b][c] for c in range(dim)) for b in range(blocks)]
        x = [scores[b] if b in candidates else -10000 for b in range(blocks)]
        selected = sorted(range(blocks), key=lambda b: -x[b])[:3]
        priors = entmax([x[b] for b in selected])[:2]
        selected = selected[:2]
        valid = [b in candidates for b in selected]
        logs = [math.log(max(p, 0.0001)) for p in priors]
        center = sum(p for p, ok in zip(logs, valid) if ok) / max(sum(valid), 1)
        biases = {b: p - center for b, p, ok in zip(selected, logs, valid) if ok}
        read = {j: 0 for j in range(max(0, position - 3), position + 1)}
        for block, bias in biases.items():
            for j in range(block * 2, min(count, block * 2 + 2)):
                if j <= position - 4:
                    read[j] = bias
        keys = list(read)
        scores = [sum(q[position * dim + c] * k[j * dim + c] for c in range(dim)) / 2 + read[j] for j in keys]
        probabilities = softmax(scores)
        output += [sum(p * v[j * dim + c] for p, j in zip(probabilities, keys)) for c in range(dim)]
    return dict(tokens=tokens, q=q, k=k, v=v, indexQ=iq, indexK=ik, head=head, output=output)


def decoder_reference():
    config = json.loads((ROOT / 'config.json').read_text())
    config.update(hidden_size=8, intermediate_size=12, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=2, vocab_size=11,
                  n_routed_experts=4, n_shared_experts=2, num_experts_per_tok=2,
                  moe_intermediate_size=3)
    config['sparse_attention_config'].update(sparse_index_dim=4, dash_scale_per_layer=[4, 4],
                                            dash_floor_per_layer=[0.0001, 0.0001])
    tensors = {}

    def matrix(name, rows, cols):
        seed = sum((i + 1) * ord(c) for i, c in enumerate(name))
        values = [round(math.sin(seed + i * 0.7) * 0.15, 7) for i in range(rows * cols)]
        tensors[name + '.weight'] = dict(shape=[rows, cols], values=values)

    def norm(name, width):
        tensors[name + '.weight'] = dict(shape=[width], values=[1 + i * 0.01 for i in range(width)])

    def mlp_weights(prefix, width):
        matrix(prefix + '.gate_proj', width, 8)
        matrix(prefix + '.up_proj', width, 8)
        matrix(prefix + '.down_proj', 8, width)

    for layer in range(2):
        prefix = f'model.layers.{layer}'
        norm(prefix + '.input_layernorm', 8)
        norm(prefix + '.post_attention_layernorm', 8)
        for projection in ['q', 'k', 'v', 'o']:
            matrix(prefix + f'.self_attn.{projection}_proj', 8, 8)
        for projection in ['q', 'k']:
            matrix(prefix + f'.self_attn.dash_router.{projection}_proj', 4, 8)
            norm(prefix + f'.self_attn.dash_router.{projection}_norm', 4)
        tensors[prefix + '.self_attn.dash_router.head_cls'] = dict(shape=[4], values=[0.2, -0.1, 0.3, -0.4])
        if layer == 0:
            mlp_weights(prefix + '.mlp', 12)
        else:
            matrix(prefix + '.mlp.gate', 4, 8)
            mlp_weights(prefix + '.mlp.shared_experts', 6)
            for expert in range(4):
                mlp_weights(prefix + f'.mlp.experts.{expert}', 3)
    norm('model.norm', 8)
    matrix('lm_head', 11, 8)

    def linear(x, name):
        tensor = tensors[name + '.weight']
        rows, cols = tensor['shape']
        w = tensor['values']
        return [sum(x[j] * w[i * cols + j] for j in range(cols)) for i in range(rows)]

    def rms(x, name):
        factor = 1 / math.sqrt(sum(v * v for v in x) / len(x) + 1e-6)
        return [v * factor * w for v, w in zip(x, tensors[name + '.weight']['values'])]

    def mlp(x, prefix):
        gate = linear(x, prefix + '.gate_proj')
        up = linear(x, prefix + '.up_proj')
        activated = [a / (1 + math.exp(-a)) * b for a, b in zip(gate, up)]
        return linear(activated, prefix + '.down_proj')

    tokens = [0, 128827, 128815, 128815, 7, 128827, 8, 9, 10]
    page = [0, 1, 1, 1, 0, 1, 1, 1, 1]
    position = [0, 0, 1, 2, 1, 3, 4, 5, 6]

    def rotate(x, row):
        result = []
        for head in range(2):
            values = x[head * 4:head * 4 + 4]
            for dim in range(4):
                angle = position[row] if dim % 2 == 0 else page[row] % 2
                rotated = -values[dim + 2] if dim < 2 else values[dim - 2]
                result.append(values[dim] * math.cos(angle) + rotated * math.sin(angle))
        return result

    inputs = [[round(math.sin(i * 0.3 + j * 0.4), 7) for j in range(8)] for i in range(len(tokens))]
    hidden = inputs
    for layer in range(2):
        prefix = f'model.layers.{layer}'
        normalized = [rms(x, prefix + '.input_layernorm') for x in hidden]
        queries = [rotate(linear(x, prefix + '.self_attn.q_proj'), i) for i, x in enumerate(normalized)]
        keys = [rotate(linear(x, prefix + '.self_attn.k_proj'), i) for i, x in enumerate(normalized)]
        values = [linear(x, prefix + '.self_attn.v_proj') for x in normalized]
        updated = []
        for row in range(len(tokens)):
            attended = []
            for head in range(2):
                scores = [sum(queries[row][head * 4 + c] * keys[j][head * 4 + c] for c in range(4)) / 2 for j in range(row + 1)]
                probabilities = softmax(scores)
                attended += [sum(p * values[j][head * 4 + c] for j, p in enumerate(probabilities)) for c in range(4)]
            projected = linear(attended, prefix + '.self_attn.o_proj')
            residual = [a + b for a, b in zip(hidden[row], projected)]
            x = rms(residual, prefix + '.post_attention_layernorm')
            if layer == 0:
                delta = mlp(x, prefix + '.mlp')
            else:
                scores = softmax(linear(x, prefix + '.mlp.gate'))
                selected = sorted(range(4), key=lambda i: -scores[i])[:2]
                routed = [(scores[e], mlp(x, prefix + f'.mlp.experts.{e}')) for e in selected]
                shared = mlp(x, prefix + '.mlp.shared_experts')
                delta = [shared[c] + sum(p * expert[c] for p, expert in routed) for c in range(8)]
            updated.append([a + b for a, b in zip(residual, delta)])
        hidden = updated
    logits = [linear(rms(x, 'model.norm'), 'lm_head') for x in hidden]
    return dict(config=config, tensors=tensors, tokens=tokens, inputs=inputs, logits=logits)


if __name__ == '__main__':
    for name, reference in [('attention', attention_reference()), ('decoder', decoder_reference())]:
        (ROOT / f'{name}.json').write_text(json.dumps(reference, indent=2) + '\n')
