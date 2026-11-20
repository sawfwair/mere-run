#if os(macOS)
import Foundation
import MLX
import MLXFast
import MereRunTensor

enum Q35PrefillGDNRecurrence {
    @TaskLocal static var enabled =
        ProcessInfo.processInfo.environment["MERERUN_Q38_PREFILL_GDN_WIDE_RECURRENCE"] == "1"

    static func eligible(q: MLXArray, k: MLXArray, v: MLXArray, keyHeads: Int, valueHeads: Int) -> Bool {
        enabled && QuantizedPrefillContext.mode == .gdnRepaired && q.dim(0) == 1 && q.dim(1) >= 512
            && keyHeads == 16 && valueHeads == 48 && q.dim(3) == 128 && v.dim(3) == 128
            && q.dtype == .bfloat16 && k.dtype == .bfloat16 && v.dtype == .bfloat16
    }

    // Each SIMD group reuses a four-element slice of K and Q for eight
    // independent value columns. Each column retains the original 32-lane
    // dot product, SIMD sum, and FP32 state update order.
    static let kernel = MLXFast.metalKernel(
        name: "q35_gdn_prefill_eight_columns",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"], outputNames: ["y", "state_out"],
        source: """
            constexpr int columns = 8;
            const int head = thread_position_in_grid.z;
            const int keyHead = head / (Hv / Hk);
            const int column = thread_position_in_grid.y * columns;
            const int lane = thread_index_in_simdgroup;
            const int base = (head * Dv + column) * Dk + 4 * lane;
            float state[columns][4];
            for (int c = 0; c < columns; ++c)
                for (int j = 0; j < 4; ++j) state[c][j] = state_in[base + c * Dk + j];

            for (int t = 0; t < T; ++t) {
                const int at = t * Hv + head;
                const int input = (t * Hk + keyHead) * Dk + 4 * lane;
                const float4 key = float4(*reinterpret_cast<device const bfloat4*>(k + input));
                const float4 query = float4(*reinterpret_cast<device const bfloat4*>(q + input));
                const float decay = g[at], mix = beta[at];
                for (int c = 0; c < columns; ++c) {
                    float remembered = 0.0f;
                    for (int j = 0; j < 4; ++j) {
                        state[c][j] *= decay;
                        remembered += state[c][j] * key[j];
                    }
                    const int output = at * Dv + column + c;
                    const float delta = (float(v[output]) - simd_sum(remembered)) * mix;
                    float result = 0.0f;
                    for (int j = 0; j < 4; ++j) {
                        state[c][j] += key[j] * delta;
                        result += state[c][j] * query[j];
                    }
                    result = simd_sum(result);
                    if (lane == 0) y[output] = InT(result);
                }
            }
            for (int c = 0; c < columns; ++c)
                for (int j = 0; j < 4; ++j) state_out[base + c * Dk + j] = state[c][j];
            """
    )
}
#endif
