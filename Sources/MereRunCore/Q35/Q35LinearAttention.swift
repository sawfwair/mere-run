import Foundation
import MLX
import MLXFast
import MLXNN

private let q35PreciseSwigluCompiled = compile(shapeless: true) { hiddenStates, gate, x in
    let gate = MLXNN.silu(gate.asType(.float32))
    return (gate * x.asType(.float32)).asType(hiddenStates.dtype)
}

final class Q35RMSNormGated: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    private let eps: Float

    init(hiddenSize: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.ones([hiddenSize])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray?) -> MLXArray {
        let normalized = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
        guard let gate else {
            return normalized.asType(hiddenStates.dtype)
        }
        return q35PreciseSwigluCompiled(hiddenStates, gate, normalized)
    }
}

@inline(__always)
private func q35Swiglu(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    MLXNN.silu(gate) * up
}

private let q35RmsNormNoWeightCompiled = compile(shapeless: true) { x in
    let squaredMean = (x * x).mean(axis: -1, keepDims: true)
    return x * rsqrt(squaredMean + MLXArray(1e-6).asType(x.dtype))
}

private func q35RmsNormNoWeight(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    guard eps == 1e-6 else {
        let squaredMean = (x * x).mean(axis: -1, keepDims: true)
        return x * rsqrt(squaredMean + MLXArray(eps).asType(x.dtype))
    }
    return q35RmsNormNoWeightCompiled(x)
}

private func q35ConcatenatedOptional(_ lhs: MLXArray?, _ rhs: MLXArray?) -> MLXArray?? {
    switch (lhs, rhs) {
    case (nil, nil):
        return .some(nil)
    case let (lhs?, rhs?):
        return .some(MLX.concatenated([lhs, rhs], axis: 0))
    default:
        return nil
    }
}

func q35FusedPortableQuantizedLinear(_ lhs: Linear, _ rhs: Linear) -> PortableQuantizedLinear? {
    guard let lhs = lhs as? PortableQuantizedLinear,
          let rhs = rhs as? PortableQuantizedLinear,
          lhs.shape.1 == rhs.shape.1,
          lhs.groupSize == rhs.groupSize,
          lhs.bits == rhs.bits,
          lhs.mode == rhs.mode,
          let bias = q35ConcatenatedOptional(lhs.bias, rhs.bias),
          let biases = q35ConcatenatedOptional(lhs.biases, rhs.biases) else {
        return nil
    }

    let weight = MLX.concatenated([lhs.weight, rhs.weight], axis: 0)
    let scales = MLX.concatenated([lhs.scales, rhs.scales], axis: 0)
    MLX.eval(weight, scales)
    if let bias { MLX.eval(bias) }
    if let biases { MLX.eval(biases) }

    return PortableQuantizedLinear(
        weight: weight,
        bias: bias,
        scales: scales,
        biases: biases,
        groupSize: lhs.groupSize,
        bits: lhs.bits,
        mode: lhs.mode
    )
}

private let q35ComputeGCompiled = compile(shapeless: true) { aLog, a, dtBias in
    let dt = softplus(a.asType(.float32) + dtBias.asType(.float32).reshaped(1, 1, dtBias.dim(0)))
    let decayBase = MLX.exp(aLog.asType(.float32)).reshaped(1, 1, dtBias.dim(0))
    return MLX.exp(-decayBase * dt)
}

@inline(__always)
private func q35ComputeG(aLog: MLXArray, a: MLXArray, dtBias: MLXArray) -> MLXArray {
    q35ComputeGCompiled(aLog, a, dtBias)
}

#if os(macOS)
private enum Q35GatedDeltaMetalKernel {
    static let kernel = MLXFast.metalKernel(
        name: "q35_gated_delta_step",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            for (int t = 0; t < T; ++t) {
              float kv_mem = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * g_[hv_idx];
                kv_mem += state[i] * k_[s_idx];
              }
              kv_mem = simd_sum(kv_mem);

              auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

              float out = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * delta;
                out += state[i] * q_[s_idx];
              }
              out = simd_sum(out);
              if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
              }

              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
            """
    )
}
#endif

func q35GatedDeltaUpdateOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray?,
    numKeyHeads: Int,
    numValueHeads: Int,
    valueHeadDim: Int
) -> (MLXArray, MLXArray) {
    let batch = q.dim(0)
    let sequence = q.dim(1)
    let keyHeadDim = q.dim(3)
    let outputDType = q.dtype

    let beta = MLX.sigmoid(b).asType(.float32)
    let g = q35ComputeG(aLog: aLog, a: a, dtBias: dtBias)

    var qExpanded = q.asType(.float32)
    var kExpanded = k.asType(.float32)
    let v32 = v.asType(.float32)
    let repeatFactor = max(1, numValueHeads / max(1, numKeyHeads))
    if repeatFactor > 1 {
        qExpanded = MLX.repeated(qExpanded, count: repeatFactor, axis: 2)
        kExpanded = MLX.repeated(kExpanded, count: repeatFactor, axis: 2)
    }

    var recurrent = state
        .map { $0.asType(.float32) }
        ?? MLXArray.zeros([batch, numValueHeads, valueHeadDim, keyHeadDim], dtype: .float32)
    var outputs: [MLXArray] = []
    outputs.reserveCapacity(sequence)

    for t in 0..<sequence {
        let qStep = qExpanded[0..., t, 0..., 0...]
        let kStep = kExpanded[0..., t, 0..., 0...]
        let vStep = v32[0..., t, 0..., 0...]
        let gStep = g[0..., t, 0...]
        let betaStep = beta[0..., t, 0...]

        let decay = gStep.reshaped(batch, numValueHeads, 1, 1)
        let kBroadcast = kStep.reshaped(batch, numValueHeads, 1, keyHeadDim)
        let qBroadcast = qStep.reshaped(batch, numValueHeads, 1, keyHeadDim)

        let decayed = recurrent * decay
        let kvMem = (decayed * kBroadcast).sum(axis: -1)
        let delta = (vStep - kvMem) * betaStep.reshaped(batch, numValueHeads, 1)

        recurrent = decayed + kBroadcast * delta.reshaped(batch, numValueHeads, valueHeadDim, 1)
        let outStep = (recurrent * qBroadcast).sum(axis: -1)
        outputs.append(outStep)
    }

    let y: MLXArray
    if outputs.count == 1 {
        y = outputs[0].expandedDimensions(axis: 1)
    } else {
        y = MLX.stacked(outputs, axis: 1)
    }

    return (y.asType(outputDType), recurrent)
}

func q35GatedDeltaUpdate(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray?,
    numKeyHeads: Int,
    numValueHeads: Int,
    valueHeadDim: Int
) -> (MLXArray, MLXArray) {
    #if os(macOS)
    if let fast = q35GatedDeltaUpdateMetal(
        q: q,
        k: k,
        v: v,
        a: a,
        b: b,
        aLog: aLog,
        dtBias: dtBias,
        state: state,
        numKeyHeads: numKeyHeads,
        numValueHeads: numValueHeads,
        valueHeadDim: valueHeadDim
    ) {
        return fast
    }
    #endif

    return q35GatedDeltaUpdateOps(
        q: q,
        k: k,
        v: v,
        a: a,
        b: b,
        aLog: aLog,
        dtBias: dtBias,
        state: state,
        numKeyHeads: numKeyHeads,
        numValueHeads: numValueHeads,
        valueHeadDim: valueHeadDim
    )
}

#if os(macOS)
func q35GatedDeltaUpdateMetal(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray?,
    numKeyHeads: Int,
    numValueHeads: Int,
    valueHeadDim: Int
) -> (MLXArray, MLXArray)? {
    guard Device.defaultDevice().deviceType == .gpu else {
        return nil
    }

    let batch = q.dim(0)
    let sequence = q.dim(1)
    let keyHeadDim = q.dim(3)
    guard keyHeadDim % 32 == 0,
          numKeyHeads > 0,
          numValueHeads > 0,
          numValueHeads % numKeyHeads == 0 else {
        return nil
    }

    let beta = MLX.sigmoid(b).asType(.float32)
    let g = q35ComputeG(aLog: aLog, a: a, dtBias: dtBias)
    let recurrent = state
        .map { $0.asType(.float32) }
        ?? MLXArray.zeros([batch, numValueHeads, valueHeadDim, keyHeadDim], dtype: .float32)

    let outputs = Q35GatedDeltaMetalKernel.kernel(
        [q, k, v, g, beta, recurrent, sequence],
        template: [
            ("InT", q.dtype),
            ("StT", recurrent.dtype),
            ("Dk", keyHeadDim),
            ("Dv", valueHeadDim),
            ("Hk", numKeyHeads),
            ("Hv", numValueHeads),
        ],
        grid: (32, valueHeadDim, batch * numValueHeads),
        threadGroup: (32, 4, 1),
        outputShapes: [
            [batch, sequence, numValueHeads, valueHeadDim],
            recurrent.shape,
        ],
        outputDTypes: [q.dtype, recurrent.dtype]
    )

    return (outputs[0], outputs[1])
}
#endif

public final class Q35LinearCache: @unchecked Sendable {
    var convState: MLXArray?
    var recurrentState: MLXArray?

    public init() {}

    public func reset() {
        convState = nil
        recurrentState = nil
    }

    public func fork() -> Q35LinearCache {
        let copy = Q35LinearCache()
        copy.convState = convState
        copy.recurrentState = recurrentState
        return copy
    }

    public func batched(with caches: [Q35LinearCache]) -> Q35LinearCache? {
        guard !caches.isEmpty else { return nil }

        let batchedConvState = Self.batchedState(caches.map(\.convState))
        let batchedRecurrentState = Self.batchedState(caches.map(\.recurrentState))
        guard batchedConvState.isValid, batchedRecurrentState.isValid else {
            return nil
        }

        let copy = Q35LinearCache()
        copy.convState = batchedConvState.value
        copy.recurrentState = batchedRecurrentState.value
        return copy
    }

    public func unbatchedRows(count: Int) -> [Q35LinearCache]? {
        guard count > 0 else { return nil }
        let convRows = Self.unbatchedRows(convState, count: count)
        let recurrentRows = Self.unbatchedRows(recurrentState, count: count)
        guard convRows.isValid, recurrentRows.isValid else {
            return nil
        }
        return (0..<count).map { index in
            let copy = Q35LinearCache()
            copy.convState = convRows.values?[index]
            copy.recurrentState = recurrentRows.values?[index]
            return copy
        }
    }

    private static func batchedState(_ states: [MLXArray?]) -> (isValid: Bool, value: MLXArray?) {
        let present = states.compactMap { $0 }
        if present.isEmpty {
            return (true, nil)
        }
        guard present.count == states.count else {
            return (false, nil)
        }
        let expectedShape = Array(present[0].shape.dropFirst())
        guard present.allSatisfy({ Array($0.shape.dropFirst()) == expectedShape }) else {
            return (false, nil)
        }
        return (true, concatenated(present, axis: 0))
    }

    private static func unbatchedRows(_ state: MLXArray?, count: Int) -> (isValid: Bool, values: [MLXArray]?) {
        guard let state else {
            return (true, nil)
        }
        guard state.dim(0) == count else {
            return (false, nil)
        }
        let rows = (0..<count).map { index in
            switch state.ndim {
            case 3:
                return state[index..<(index + 1), 0..., 0...]
            case 4:
                return state[index..<(index + 1), 0..., 0..., 0...]
            default:
                return state[index..<(index + 1)]
            }
        }
        return (true, rows)
    }
}

final class Q35LinearAttention: Module {
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "dt_bias") var dtBias: MLXArray
    @ModuleInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: Q35RMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    private let hiddenSize: Int
    private let numKeyHeads: Int
    private let numValueHeads: Int
    private let keyHeadDim: Int
    private let valueHeadDim: Int
    private let keyDim: Int
    private let valueDim: Int
    private let convDim: Int
    private let convKernelSize: Int
    private var fusedInProjQKVZ: Linear?
    private var fusedInProjBA: Linear?

    init(config: Q35Config) {
        let text = config.textConfig
        self.hiddenSize = text.hiddenSize
        self.numKeyHeads = text.linearNumKeyHeads
        self.numValueHeads = text.linearNumValueHeads
        self.keyHeadDim = text.linearKeyHeadDim
        self.valueHeadDim = text.linearValueHeadDim
        self.keyDim = text.linearNumKeyHeads * text.linearKeyHeadDim
        self.valueDim = text.linearNumValueHeads * text.linearValueHeadDim
        self.convDim = self.keyDim * 2 + self.valueDim
        self.convKernelSize = max(1, text.linearConvKernelDim)

        precondition(
            self.numValueHeads % max(1, self.numKeyHeads) == 0,
            "Qwen-family linear attention requires num_value_heads divisible by num_key_heads"
        )

        self._inProjQKV.wrappedValue = Linear(text.hiddenSize, self.keyDim * 2 + self.valueDim, bias: false)
        self._inProjZ.wrappedValue = Linear(text.hiddenSize, self.valueDim, bias: false)
        self._inProjA.wrappedValue = Linear(text.hiddenSize, self.numValueHeads, bias: false)
        self._inProjB.wrappedValue = Linear(text.hiddenSize, self.numValueHeads, bias: false)
        self._conv1d.wrappedValue = Conv1d(
            inputChannels: self.convDim,
            outputChannels: self.convDim,
            kernelSize: text.linearConvKernelDim,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: self.convDim,
            bias: false
        )
        self._dtBias.wrappedValue = MLXArray.ones([self.numValueHeads])
        self._aLog.wrappedValue = MLXArray.zeros([self.numValueHeads])
        self._norm.wrappedValue = Q35RMSNormGated(hiddenSize: self.valueHeadDim, eps: text.rmsNormEps)
        self._outProj.wrappedValue = Linear(self.valueDim, text.hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: Q35LinearCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)
        let keep = max(0, convKernelSize - 1)

        let qkv: MLXArray
        let z: MLXArray
        if fusedInProjQKVZ == nil {
            fusedInProjQKVZ = q35FusedPortableQuantizedLinear(inProjQKV, inProjZ)
        }
        if let fusedInProjQKVZ {
            let qkvz = fusedInProjQKVZ(x)
            qkv = qkvz[.ellipsis, 0..<convDim]
            z = qkvz[.ellipsis, convDim...].reshaped(batch, sequence, numValueHeads, valueHeadDim)
        } else {
            qkv = inProjQKV(x)
            z = inProjZ(x).reshaped(batch, sequence, numValueHeads, valueHeadDim)
        }

        let b: MLXArray
        let a: MLXArray
        if fusedInProjBA == nil {
            fusedInProjBA = q35FusedPortableQuantizedLinear(inProjB, inProjA)
        }
        if let fusedInProjBA {
            let ba = fusedInProjBA(x)
            b = ba[.ellipsis, 0..<numValueHeads]
            a = ba[.ellipsis, numValueHeads...]
        } else {
            b = inProjB(x)
            a = inProjA(x)
        }

        let convState = cache?.convState
            ?? MLXArray.zeros([batch, keep, convDim], dtype: x.dtype)
        let convInput = MLX.concatenated([convState, qkv], axis: 1)

        if let cache {
            if keep > 0 {
                cache.convState = convInput[0..., (convInput.dim(1) - keep)..., 0...]
            } else {
                cache.convState = MLXArray.zeros([batch, 0, convDim], dtype: x.dtype)
            }
        }

        let convOut = MLXNN.silu(conv1d(convInput))

        let qConv = convOut[.ellipsis, 0..<keyDim]
        let kConv = convOut[.ellipsis, keyDim..<(2 * keyDim)]
        let vConv = convOut[.ellipsis, (2 * keyDim)...]

        var q = qConv.reshaped(batch, sequence, numKeyHeads, keyHeadDim)
        var k = kConv.reshaped(batch, sequence, numKeyHeads, keyHeadDim)
        let v = vConv.reshaped(batch, sequence, numValueHeads, valueHeadDim)

        let invScale = 1.0 / sqrt(Float(max(1, keyHeadDim)))
        q = q35RmsNormNoWeight(q) * MLXArray(invScale * invScale).asType(q.dtype)
        k = q35RmsNormNoWeight(k) * MLXArray(invScale).asType(k.dtype)

        let (updated, state) = q35GatedDeltaUpdate(
            q: q,
            k: k,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: cache?.recurrentState,
            numKeyHeads: numKeyHeads,
            numValueHeads: numValueHeads,
            valueHeadDim: valueHeadDim
        )
        cache?.recurrentState = state

        let normalized = norm(updated, gate: z)
        return outProj(normalized.reshaped(batch, sequence, valueDim))
    }
}
