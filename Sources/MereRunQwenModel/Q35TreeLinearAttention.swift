import MLX
import MLXFast

func q35TreePrework(
    qkv: MLXArray, convState: MLXArray, convWeight: MLXArray,
    numKeyHeads: Int, numValueHeads: Int, keyHeadDim: Int, valueHeadDim: Int,
    rmsNormWeight: MLXArray?, tree: Q35VerificationTree
) -> Q35GDNPreworkOutput {
    #if os(macOS)
    if let output = q35GDNPreworkMetal(
        qkv: qkv, convState: convState, convWeight: convWeight,
        numKeyHeads: numKeyHeads, numValueHeads: numValueHeads,
        keyHeadDim: keyHeadDim, valueHeadDim: valueHeadDim, tree: tree
    ) { return output }
    #endif
    let outputs = tree.paths.map { path in
        let output = q35GDNPreworkOps(
            qkv: Q35VerificationTree.take(qkv, rows: path), convState: convState,
            convWeight: convWeight, numKeyHeads: numKeyHeads, numValueHeads: numValueHeads,
            keyHeadDim: keyHeadDim, valueHeadDim: valueHeadDim, rmsNormWeight: rmsNormWeight
        )
        let row = path.count - 1
        return Q35GDNPreworkOutput(q: output.q[0..., row..., 0..., 0...],
            k: output.k[0..., row..., 0..., 0...], v: output.v[0..., row..., 0..., 0...], convState: convState)
    }
    return Q35GDNPreworkOutput(q: concatenated(outputs.map(\.q), axis: 1),
        k: concatenated(outputs.map(\.k), axis: 1), v: concatenated(outputs.map(\.v), axis: 1), convState: convState)
}

func q35TreeDeltaUpdate(
    q: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray,
    aLog: MLXArray, dtBias: MLXArray, state: MLXArray?,
    numKeyHeads: Int, numValueHeads: Int, valueHeadDim: Int, tree: Q35VerificationTree?,
    saveGatesIn cache: Q35LinearCache? = nil
) -> (MLXArray, MLXArray) {
    guard let tree else {
        return q35GatedDeltaUpdate(q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
            state: state, numKeyHeads: numKeyHeads, numValueHeads: numValueHeads, valueHeadDim: valueHeadDim)
    }
    let base = state?.asType(.float32)
        ?? MLXArray.zeros([1, numValueHeads, valueHeadDim, q.dim(3)], dtype: .float32)
    #if os(macOS)
    if Device.defaultDevice().deviceType == .gpu, q.dim(3).isMultiple(of: 32) {
        let g = Q35CompiledOperations.current.computeG(aLog, a, dtBias)
        let beta = MLX.sigmoid(b).asType(.float32)
        if Q35TreePipeline.enabled { cache?.verificationGates = (g, beta) }
        let output: MLXArray
        if q.dim(3) == 128 {
            output = q35TreeCompactDelta(q: q, k: k, v: v, g: g, beta: beta, state: base,
                numKeyHeads: numKeyHeads, numValueHeads: numValueHeads, valueHeadDim: valueHeadDim, tree: tree)
        } else {
            output = q35TreeIndependentDelta(q: q, k: k, v: v, g: g, beta: beta, state: base,
                numKeyHeads: numKeyHeads, numValueHeads: numValueHeads, valueHeadDim: valueHeadDim, tree: tree)
        }
        return (output, base)
    }
    #endif
    let outputs = tree.paths.map { path in
        let output = q35GatedDeltaUpdate(
            q: Q35VerificationTree.take(q, rows: path), k: Q35VerificationTree.take(k, rows: path),
            v: Q35VerificationTree.take(v, rows: path), a: Q35VerificationTree.take(a, rows: path),
            b: Q35VerificationTree.take(b, rows: path), aLog: aLog, dtBias: dtBias,
            state: base, numKeyHeads: numKeyHeads, numValueHeads: numValueHeads, valueHeadDim: valueHeadDim
        ).0
        return output[0..., (path.count - 1)..., 0..., 0...]
    }
    return (concatenated(outputs, axis: 1), base)
}

#if os(macOS)
func q35TreeIndependentDelta(
    q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray,
    numKeyHeads: Int, numValueHeads: Int, valueHeadDim: Int, tree: Q35VerificationTree
) -> MLXArray {
    let width = tree.count
    let paths = tree.paths.flatMap { $0 + Array(repeating: 0, count: width - $0.count) }
    return Q35TreeDeltaMetal.kernel(
        [q, k, v, g, beta, state, tree.devicePlan?.paths ?? MLXArray(paths.map(Int32.init)),
         tree.devicePlan?.depths ?? MLXArray(tree.depths.map(Int32.init))],
        template: [("InT", q.dtype), ("S", width), ("Hk", numKeyHeads), ("Hv", numValueHeads),
                   ("Dk", q.dim(3)), ("Dv", valueHeadDim)],
        grid: (32, valueHeadDim, width * numValueHeads), threadGroup: (32, 4, 1),
        outputShapes: [[1, width, numValueHeads, valueHeadDim]], outputDTypes: [q.dtype]
    )[0]
}

private enum Q35TreeDeltaMetal {
    static let kernel = MLXFast.metalKernel(
        name: "q35_parent_indexed_gated_delta_v1",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "paths", "depths"], outputNames: ["y"],
        source: """
            const uint row = thread_position_in_grid.z / Hv;
            const uint hv = thread_position_in_grid.z % Hv;
            const uint hk = hv / (Hv / Hk);
            const uint lane = thread_position_in_threadgroup.x;
            const uint dv = thread_position_in_grid.y;
            constexpr int n_per_t = Dk / 32;
            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i)
                state[i] = state_in[(hv * Dv + dv) * Dk + n_per_t * lane + i];
            for (int depth = 0; depth <= depths[row]; ++depth) {
                const int node = paths[row * S + depth];
                const auto q_ = q + (node * Hk + hk) * Dk;
                const auto k_ = k + (node * Hk + hk) * Dk;
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    const int idx = n_per_t * lane + i;
                    state[i] = state[i] * g[node * Hv + hv];
                    kv_mem += state[i] * k_[idx];
                }
                kv_mem = simd_sum(kv_mem);
                const float delta = (v[(node * Hv + hv) * Dv + dv] - kv_mem) * beta[node * Hv + hv];
                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    const int idx = n_per_t * lane + i;
                    state[i] = state[i] + k_[idx] * delta;
                    out += state[i] * q_[idx];
                }
                out = simd_sum(out);
                if (depth == depths[row] && thread_index_in_simdgroup == 0)
                    y[(row * Hv + hv) * Dv + dv] = InT(out);
            }
            """
    )
}
#endif
