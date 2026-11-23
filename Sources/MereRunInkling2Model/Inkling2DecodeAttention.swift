import MLX

#if os(macOS)
enum Inkling2DecodeAttention {
    static let maxKeys = 1_024
    private static let wideScoreMinKeys = 512
    private static let usesWideScoreGroups = GPU.deviceInfo().architecture == "applegpu_g17s"

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

            for (uint k = start + group; k < end; k += SCORE_GROUPS) {
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

            // Keep the four-group softmax and its reduction order.
            float largest = sink;
            if (SCORE_GROUPS == 4 || tid < 128) {
                for (uint k = start + tid; k < end; k += 128) largest = max(largest, logits[k]);
                largest = simd_max(largest);
                if (lane == 0) partials[group] = largest;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            largest = max(max(partials[0], partials[1]), max(partials[2], partials[3]));
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (SCORE_GROUPS == 4 || tid < 128) {
                float sum = tid == 0 ? exp(sink - largest) : 0;
                for (uint k = start + tid; k < end; k += 128) {
                    const float weight = exp(logits[k] - largest);
                    logits[k] = weight;
                    sum += weight;
                }
                sum = simd_sum(sum);
                if (lane == 0) partials[group] = sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inverse = 1 / (partials[0] + partials[1] + partials[2] + partials[3]);
            for (uint k = start + tid; k < end; k += 32 * SCORE_GROUPS) logits[k] = float(T(logits[k] * inverse));
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint d = tid; d < WIDTH; d += 32 * SCORE_GROUPS) {
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
        let scoreGroups = batch == 1 && tokens > 1 && keys.dim(2) >= wideScoreMinKeys && usesWideScoreGroups ? 16 : 4
        let threads = scoreGroups * 32
        return kernel([query, keys, values, relative, profiles, sinks, gate],
                      template: [("T", query.dtype), ("HEADS", heads), ("KV_HEADS", keys.dim(1)),
                                 ("WIDTH", width), ("RELATIVE_WIDTH", relative.dim(3)),
                                 ("EXTENT", profiles.dim(1)), ("MAX_KEYS", maxKeys), ("LOCAL", local), ("SCORE_GROUPS", scoreGroups)],
                      grid: (heads * threads, batch, tokens), threadGroup: (threads, 1, 1),
                      outputShapes: [[batch, tokens, heads * width]], outputDTypes: [query.dtype])[0]
    }
}

enum Inkling2MaskedAttention {
    static let maxKeys = 8_192

    private static let kernel = MLXFast.metalKernel(
        name: "mererun_inkling2_masked_attention_logits",
        inputNames: ["scores", "relative", "sinks"], outputNames: ["logits"],
        source: """
            const uint k = thread_position_in_grid.x;
            const uint row = thread_position_in_grid.y;
            const uint keys = scores_shape[3], heads = scores_shape[1], queries = scores_shape[2];
            const uint token = row % queries, head = (row / queries) % heads;
            const uint batch = row / (queries * heads);
            const uint end = keys - queries + token + 1;
            const uint query_head = (batch * queries + token) * heads + head;
            const uint distance = end - 1 - k;
            float score = -INFINITY;
            if (k == keys) {
                score = float(sinks[query_head]);
            } else if (k < end && (!LOCAL || distance < EXTENT)) {
                const float bias = distance < EXTENT ? relative[query_head * EXTENT + distance] : 0.0f;
                score = scores[row * keys + k] + bias;
            }
            logits[row * (keys + 1) + k] = score;
            """
    )

    static func probabilities(
        scores: MLXArray, relative: MLXArray, sinks: MLXArray, divisor: Float, dtype: DType, local: Bool
    ) -> MLXArray {
        let keys = scores.dim(3), rows = scores.size / keys
        let shape = [scores.dim(0), scores.dim(1), scores.dim(2), keys + 1]
        let logits = kernel([scores / divisor, relative, sinks],
                            template: [("EXTENT", relative.dim(-1)), ("LOCAL", local)],
                            grid: (keys + 1, rows, 1), threadGroup: (256, 1, 1),
                            outputShapes: [shape], outputDTypes: [.float32])[0]
        // Keep the MLX softmax reduction and round probabilities before multiplying by values.
        return softmax(logits, axis: -1)[.ellipsis, ..<keys].asType(dtype)
    }
}
#endif
