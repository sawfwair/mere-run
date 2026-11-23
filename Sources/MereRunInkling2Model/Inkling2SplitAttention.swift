import Foundation
import MLX

#if os(macOS)
// Attention kernels adapted from mlx-serve (MIT). See THIRD_PARTY_NOTICES.md.
enum Inkling2SplitAttention {
    struct Bank {
        let position: MLXArray
        let keys: MLXArray
        let values: MLXArray
    }

    private static let splitInputs = ["q", "k", "v", "rel", "sinks", "sizes", "scale"]
    private static let split = MLXFast.metalKernel(
        name: "mererun_inkling2_split_attention", inputNames: splitInputs, outputNames: ["PART", "ML"],
        source: splitSource(banked: false), header: "#include <metal_simdgroup_matrix>\n", ensureRowContiguous: false
    )
    private static let splitBank = MLXFast.metalKernel(
        name: "mererun_inkling2_split_attention_bank", inputNames: splitInputs + ["past", "tailK", "tailV"],
        outputNames: ["PART", "ML"], source: splitSource(banked: true),
        header: "#include <metal_simdgroup_matrix>\n", ensureRowContiguous: false
    )

    private static let packedRows = MLXFast.metalKernel(
        name: "mererun_inkling2_split_attention_packed_query_rows", inputNames: splitInputs,
        outputNames: ["PART", "ML"], source: splitSource(banked: false, packedRows: true),
        header: "#include <metal_simdgroup_matrix>\n", ensureRowContiguous: false
    )
    private static let packedRowsBank = MLXFast.metalKernel(
        name: "mererun_inkling2_split_attention_bank_packed_query_rows", inputNames: splitInputs + ["past", "tailK", "tailV"],
        outputNames: ["PART", "ML"], source: splitSource(banked: true, packedRows: true),
        header: "#include <metal_simdgroup_matrix>\n", ensureRowContiguous: false
    )

    private static func splitSource(banked: Bool, packedRows: Bool = false) -> String {
        let keyCount = banked ? "past + L" : "sizes[1]"
        let tailHeads = banked ? """
            const device T* tail_k_head = tailK + kvh * tailK_strides[1];
            const device T* tail_v_head = tailV + kvh * tailV_strides[1];
            """ : ""
        let keyAddress = banked
            ? "key < past ? k_head + key * k_strides[2] : tail_k_head + (key - past) * tailK_strides[2]"
            : "k_head + key * k_strides[2]"
        let valueAddress = banked
            ? "key < past ? v_head + key * v_strides[2] : tail_v_head + (key - past) * tailV_strides[2]"
            : "v_head + key * v_strides[2]"
        let rowMapping = packedRows ? """
            const int row = int(sg) * 8 + laneRow;
            const int head = kvh * G + (min(row / Q_ROWS, G - 1));
            const int fm = row % Q_ROWS;
            const bool row_ok = row < G * Q_ROWS;
            """ : """
            const int head = kvh * G + int(sg);
            const int fm = laneRow;
            const bool row_ok = fm < L;
            """
        return """
            // Inkling 2, up to 8 rows behind a cache: relattn.metal's read split over the keys, the G = HQ / HKV query heads of a
            // KV head sharing each K/V tile (MLX's attention reads the keys once per row and head). grid (splits * 32 G, HKV, 1),
            // threadgroup (32 G, 1, 1): a threadgroup takes keys [first + split * KPS, + KPS) of one KV head, SIMD group g its
            // query head kv * G + g, the rows as one 8-row fragment. q [1, HQ, L, D] and k, v [1, HKV, S, D] (T, any strides over
            // a unit last axis), the rows at the last L keys; rel [1, L, HQ, R] (T) each row's bias at the distances below R;
            // sinks [1, L, HQ] f32, the first split starting each row from its sink; sizes = { L, S, first }. WINDOW > 0 keeps
            // the keys fewer than WINDOW behind a row. PART [splits, L, HQ, D] f32 = the unnormalised sums, ML [splits, L, HQ, 2]
            // f32 = each row's running max (log2 units) and sum.
            constexpr float LOG2E = 1.4426950408889634f;
            constexpr int BK = 16;
            constexpr int LDK = BK + 8;
            constexpr int LDV = D + 8;
            constexpr int G = HQ / HKV;
            const int L = sizes[0];
            const int S = \(keyCount);
            const int split = threadgroup_position_in_grid.x;
            const int kvh = threadgroup_position_in_grid.y;
            const uint tid = thread_position_in_threadgroup.x;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const float qscale = scale * LOG2E;
            const short laneRow = ((lane / 4) & 4) + (lane / 2) % 4;
            const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
            \(rowMapping)
            const int diag = S - L + fm;
            const int kbeg = sizes[2] + split * KPS;
            const int kend = min(S, kbeg + KPS);

            threadgroup half Ks[D * LDK];
            threadgroup half Vs[BK * LDV];
            simdgroup_half8x8 q_frag[D / 8];
            for (int db = 0; db < D / 8; ++db) {
                half2 qv = half2(0.0h);
                if (row_ok) {
                    const device T* qp = q + head * q_strides[1] + fm * q_strides[2] + db * 8 + fn;
                    qv = half2(float(qp[0]), float(qp[1]));
                }
                q_frag[db].thread_elements()[0] = qv.x;
                q_frag[db].thread_elements()[1] = qv.y;
            }
            simdgroup_float8x8 acc[D / 8];
            for (int db = 0; db < D / 8; ++db) acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            // A finite floor: a split whose keys a row cannot see leaves it at weight 0 in the combine.
            float row_max = -1e30f, row_sum = 0.0f;
            if (split == 0 && row_ok) {
                row_max = sinks[fm * sinks_strides[1] + head * sinks_strides[2]] * LOG2E;
                row_sum = 1.0f;
            }
            const device T* bias = rel + (size_t(row_ok ? fm : 0) * HQ + head) * R;
            const device T* k_head = k + kvh * k_strides[1];
            const device T* v_head = v + kvh * v_strides[1];
            \(tailHeads)
            for (int k0 = kbeg; k0 < kend; k0 += BK) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int c = tid; c < BK * D / 8; c += 32 * \(packedRows ? "SIMD_GROUPS" : "G")) {
                    const int r = c / (D / 8);
                    const int col = (c % (D / 8)) * 8;
                    const int key = k0 + r;
                    vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
                    vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
                    if (key < kend) {
                        auto kp = (const device vec<T, 4>*)((\(keyAddress)) + col);
                        auto vp = (const device vec<T, 4>*)((\(valueAddress)) + col);
                        kv[0] = kp[0]; kv[1] = kp[1]; vv[0] = vp[0]; vv[1] = vp[1];
                    }
                    for (int j = 0; j < 8; ++j) {
                        Ks[(col + j) * LDK + r] = half(float(kv[j / 4][j % 4]));
                        Vs[r * LDV + col + j] = half(float(vv[j / 4][j % 4]));
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                \(packedRows ? "if (sg * 8 < uint(G * Q_ROWS)) {" : "")
                simdgroup_float8x8 scores[BK / 8];
                for (int kb = 0; kb < BK / 8; ++kb) {
                    scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    for (int db = 0; db < D / 8; ++db) {
                        simdgroup_half8x8 k_frag;
                        simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                        simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
                    }
                }
                float tile_max = -INFINITY;
                for (int kb = 0; kb < BK / 8; ++kb) {
                    for (int e = 0; e < 2; ++e) {
                        const int key = k0 + kb * 8 + fn + e;
                        const int d = diag - key;
                        const bool visible = row_ok && key < kend && d >= 0 && (WINDOW == 0 || d < WINDOW);
                        float s = -INFINITY;
                        if (visible) {
                            s = scores[kb].thread_elements()[e] * qscale;
                            if (d < R) s += float(bias[d]) * LOG2E;
                        }
                        scores[kb].thread_elements()[e] = s;
                        tile_max = max(tile_max, s);
                    }
                }
                tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
                tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
                const float new_max = max(row_max, tile_max);
                const float factor = fast::exp2(row_max - new_max);
                float tile_sum = 0.0f;
                simdgroup_half8x8 probs[BK / 8];
                for (int kb = 0; kb < BK / 8; ++kb) {
                    for (int e = 0; e < 2; ++e) {
                        const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                        probs[kb].thread_elements()[e] = half(p);
                        tile_sum += p;
                    }
                }
                tile_sum += simd_shuffle_xor(tile_sum, 1);
                tile_sum += simd_shuffle_xor(tile_sum, 8);
                row_sum = row_sum * factor + tile_sum;
                row_max = new_max;
                for (int db = 0; db < D / 8; ++db) {
                    acc[db].thread_elements()[0] *= factor;
                    acc[db].thread_elements()[1] *= factor;
                    for (int kb = 0; kb < BK / 8; ++kb) {
                        simdgroup_half8x8 v_frag;
                        simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                        simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
                    }
                }
                \(packedRows ? "}" : "")
            }
            if (row_ok) {
                const size_t at = (size_t(split) * L + fm) * HQ + head;
                for (int db = 0; db < D / 8; ++db) {
                    PART[at * D + db * 8 + fn] = acc[db].thread_elements()[0];
                    PART[at * D + db * 8 + fn + 1] = acc[db].thread_elements()[1];
                }
                if (fn == 0) {
                    ML[at * 2] = row_max;
                    ML[at * 2 + 1] = row_sum;
                }
            }
            """
    }

    private static let join = MLXFast.metalKernel(
        name: "mererun_inkling2_combine_attention",
        inputNames: ["PART", "ML", "GATE"], outputNames: ["OUT"],
        source: """
            // Inkling 2: relsplit.metal's splits combined and times the output gate. grid (D, L * HQ): one thread per output
            // element. PART [NS, L * HQ, D] f32, ML [NS, L * HQ, 2] f32, GATE [L * HQ * D] f32 (sigmoid). OUT [1, L, HQ * D] (T)
            // = the read rounded to T, times its gate, rounded to T (as gated.metal).
            const uint d = thread_position_in_grid.x;
            const uint lh = thread_position_in_grid.y;
            const uint NS = PART_shape[0];
            const uint LH = PART_shape[1];
            float m = -INFINITY;
            for (uint s = 0; s < NS; s++) m = max(m, ML[(s * LH + lh) * 2]);
            float num = 0.0f, den = 0.0f;
            for (uint s = 0; s < NS; s++) {
                const float w = precise::exp2(ML[(s * LH + lh) * 2] - m);
                num += w * PART[(s * LH + lh) * D + d];
                den += w * ML[(s * LH + lh) * 2 + 1];
            }
            OUT[lh * D + d] = T(float(T(num / den)) * GATE[lh * D + d]);
            """
    )

    static func apply(query: MLXArray, keys: MLXArray, values: MLXArray, states: MLXArray, profiles: MLXArray,
                      sinks: MLXArray, gate: MLXArray, window: Int, packQueryRows: Bool, bank: Bank? = nil) -> MLXArray {
        let rows = query.dim(1), heads = query.dim(2), width = query.dim(3)
        let count = keys.dim(2) + (bank == nil ? 0 : rows)
        let kv = keys.dim(1), groups = heads / kv
        let first = window == 0 ? 0 : max(0, count - rows - window + 1)
        let tiles = (count - first + 127) / 128
        let sizes = MLXArray([Int32(rows), Int32(count), Int32(first)])
        let scale = MLXArray(1 / Float(width).squareRoot())
        let bias = matmul(states, profiles.asType(states.dtype))
        var inputs = [query.transposed(0, 2, 1, 3), keys, values, bias, sinks.asType(.float32), sizes, scale]
        if let bank { inputs += [bank.position, bank.keys, bank.values] }
        let packed = packQueryRows && (4...5).contains(rows)
        let simdGroups = packed ? (rows * groups + 7) / 8 : groups
        let kernel = packed ? (bank == nil ? packedRows : packedRowsBank) : (bank == nil ? split : splitBank)
        let groupArgument: [(String, any KernelTemplateArg)] = packed
            ? [("SIMD_GROUPS", simdGroups), ("Q_ROWS", rows)]
            : []
        let parts = kernel(
            inputs,
            template: [("T", query.dtype), ("D", width), ("HQ", heads), ("HKV", kv), ("R", profiles.dim(1)),
                       ("WINDOW", window), ("KPS", 128)] + groupArgument,
            grid: (tiles * 32 * simdGroups, kv, 1), threadGroup: (32 * simdGroups, 1, 1),
            outputShapes: [[tiles, rows * heads, width], [tiles, rows * heads, 2]], outputDTypes: [.float32, .float32]
        )
        return join(
            [parts[0], parts[1], sigmoid(gate.asType(.float32))],
            template: [("T", query.dtype), ("D", width)],
            grid: (width, rows * heads, 1), threadGroup: (width, 1, 1),
            outputShapes: [[1, rows, heads * width]], outputDTypes: [query.dtype]
        )[0]
    }
}
#endif
