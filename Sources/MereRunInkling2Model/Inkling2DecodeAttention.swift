import MLX

#if os(macOS)
enum Inkling2DecodeAttention {
    static let maxKeys = 1_024

    private static let kernel = MLXFast.metalKernel(
        name: "mererun_inkling2_decode_attention",
        inputNames: ["query", "key", "value", "relative", "profiles", "sinks", "gate"],
        outputNames: ["output"],
        source: """
            const uint head = threadgroup_position_in_grid.x;
            const uint batch = threadgroup_position_in_grid.y;
            const uint token = threadgroup_position_in_grid.z;
            const uint lane = thread_index_in_simdgroup;
            const uint group = simdgroup_index_in_threadgroup;
            const uint tid = thread_index_in_threadgroup;
            const uint keys = key_shape[2];
            const uint end = keys - query_shape[1] + token + 1;
            const uint start = LOCAL && end > EXTENT ? end - EXTENT : 0;
            const uint head_index = (batch * query_shape[1] + token) * HEADS + head;
            const uint query_base = head_index * WIDTH;
            const uint kv_head = batch * KV_HEADS + head / (HEADS / KV_HEADS);
            const uint kv_base = kv_head * keys * WIDTH;
            threadgroup float logits[MAX_KEYS];
            threadgroup float partials[4];
            const float sink = float(sinks[head_index]);

            for (uint k = start + group; k < end; k += 4) {
                float dot = 0;
                for (uint d = lane; d < WIDTH; d += 32) {
                    dot += float(query[query_base + d]) * float(key[kv_base + k * WIDTH + d]);
                }
                const uint distance = end - 1 - k;
                float bias = 0;
                if (distance < EXTENT) {
                    for (uint d = lane; d < RELATIVE_WIDTH; d += 32) {
                        bias += float(relative[head_index * RELATIVE_WIDTH + d]) * float(profiles[d * EXTENT + distance]);
                    }
                }
                const float score = simd_sum(dot) * rsqrt(float(WIDTH)) + simd_sum(bias);
                if (lane == 0) logits[k] = score;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float largest = sink;
            for (uint k = start + tid; k < end; k += 128) largest = max(largest, logits[k]);
            largest = simd_max(largest);
            if (lane == 0) partials[group] = largest;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            largest = max(max(partials[0], partials[1]), max(partials[2], partials[3]));
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float sum = tid == 0 ? exp(sink - largest) : 0;
            for (uint k = start + tid; k < end; k += 128) {
                const float weight = exp(logits[k] - largest);
                logits[k] = weight;
                sum += weight;
            }
            sum = simd_sum(sum);
            if (lane == 0) partials[group] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inverse = 1 / (partials[0] + partials[1] + partials[2] + partials[3]);
            for (uint k = start + tid; k < end; k += 128) logits[k] = float(T(logits[k] * inverse));
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint d = tid; d < WIDTH; d += 128) {
                float attended = 0;
                for (uint k = start; k < end; ++k) {
                    attended += logits[k] * float(value[kv_base + k * WIDTH + d]);
                }
                const float gain = 1 / (1 + exp(-float(gate[query_base + d])));
                output[query_base + d] = T(float(T(attended)) * gain);
            }
            """
    )

    static func apply(query: MLXArray, keys: MLXArray, values: MLXArray, relative: MLXArray,
                      profiles: MLXArray, sinks: MLXArray, gate: MLXArray, local: Bool = false) -> MLXArray {
        let batch = query.dim(0), tokens = query.dim(1), heads = query.dim(2), width = query.dim(3)
        return kernel([query, keys, values, relative, profiles, sinks, gate],
                      template: [("T", query.dtype), ("HEADS", heads), ("KV_HEADS", keys.dim(1)),
                                 ("WIDTH", width), ("RELATIVE_WIDTH", relative.dim(3)),
                                 ("EXTENT", profiles.dim(1)), ("MAX_KEYS", maxKeys), ("LOCAL", local)],
                      grid: (heads * 128, batch, tokens), threadGroup: (128, 1, 1),
                      outputShapes: [[batch, tokens, heads * width]], outputDTypes: [query.dtype])[0]
    }
}
#endif
