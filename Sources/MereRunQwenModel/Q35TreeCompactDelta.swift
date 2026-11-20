import MLX
import MLXFast

#if os(macOS)
func q35TreeCompactDelta(
    q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray,
    numKeyHeads: Int, numValueHeads: Int, valueHeadDim: Int, tree: Q35VerificationTree
) -> MLXArray {
    let slots = tree.recurrentSlots
    return Q35TreeCompactDeltaMetal.kernel(
        [q, k, v, g, beta, state, tree.devicePlan?.slotReads ?? MLXArray(slots.reads.map(Int32.init)),
         tree.devicePlan?.slotWrites ?? MLXArray(slots.writes.map(Int32.init)),
         tree.devicePlan?.slotOrder ?? MLXArray(slots.order.map(Int32.init))],
        template: [("InT", q.dtype), ("S", tree.count), ("Slots", max(1, slots.count)),
                   ("Hk", numKeyHeads), ("Hv", numValueHeads), ("Dk", q.dim(3)), ("Dv", valueHeadDim)],
        grid: (32, valueHeadDim, numValueHeads), threadGroup: (32, 4, 1),
        outputShapes: [[1, tree.count, numValueHeads, valueHeadDim]], outputDTypes: [q.dtype]
    )[0]
}

enum Q35TreeCompactDeltaMetal {
    // Named arrays and constant switch cases keep slot accesses addressable as
    // registers. A dynamically indexed thread-local matrix can spill to memory.
    private static let declarations = (0..<8).map {
        "float slot\($0)[Slots > \($0) ? n_per_t : 1];"
    }.joined(separator: "\n")
    private static let reads = (0..<8).map {
        """
        case \($0):
            if constexpr (Slots > \($0)) {
                for (int i = 0; i < n_per_t; ++i) state[i] = slot\($0)[i];
            }
            break;
        """
    }.joined(separator: "\n")
    private static let writes = (0..<8).map {
        """
        case \($0):
            if constexpr (Slots > \($0)) {
                for (int i = 0; i < n_per_t; ++i) slot\($0)[i] = state[i];
            }
            break;
        """
    }.joined(separator: "\n")
    static let kernel = MLXFast.metalKernel(
        name: "q35_compact_parent_delta_v2",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "read_slots", "write_slots", "rows"],
        outputNames: ["y"],
        source: source
    )

    static let source = """
            const uint hv = thread_position_in_grid.z;
            const uint hk = hv / (Hv / Hk);
            const uint lane = thread_position_in_threadgroup.x;
            const uint dv = thread_position_in_grid.y;
            constexpr int n_per_t = Dk / 32;
            \(declarations)
            for (int step = 0; step < S; ++step) {
                const int row = rows[step];
                float state[n_per_t];
                switch (read_slots[row]) {
                    case -1:
                        for (int i = 0; i < n_per_t; ++i)
                            state[i] = state_in[(hv * Dv + dv) * Dk + n_per_t * lane + i];
                        break;
                    \(reads)
                }
                const auto q_ = q + (row * Hk + hk) * Dk;
                const auto k_ = k + (row * Hk + hk) * Dk;
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    const int idx = n_per_t * lane + i;
                    state[i] = state[i] * g[row * Hv + hv];
                    kv_mem += state[i] * k_[idx];
                }
                kv_mem = simd_sum(kv_mem);
                const float delta = (v[(row * Hv + hv) * Dv + dv] - kv_mem) * beta[row * Hv + hv];
                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    const int idx = n_per_t * lane + i;
                    state[i] = state[i] + k_[idx] * delta;
                    out += state[i] * q_[idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) y[(row * Hv + hv) * Dv + dv] = InT(out);
                switch (write_slots[row]) { \(writes) }
            }
            """
}
#endif
