import Foundation
import MLX
import MLXFast
import MLXNN

final class Q35RMSNormGated: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    private let eps: Float
    private let activation: String

    init(hiddenSize: Int, eps: Float, activation: String = "silu") {
        self._weight.wrappedValue = MLXArray.ones([hiddenSize])
        self.eps = eps
        self.activation = activation
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray?) -> MLXArray {
        let normalized = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
        guard let gate else {
            return normalized.asType(hiddenStates.dtype)
        }
        if activation == "sigmoid" {
            return (normalized.asType(.float32) * MLX.sigmoid(gate.asType(.float32)))
                .asType(hiddenStates.dtype)
        }
        return Q35CompiledOperations.current.preciseSwiglu(hiddenStates, gate, normalized)
    }
}

@inline(__always)
private func q35Swiglu(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    q35Silu(gate) * up
}

private func q35RmsNormNoWeight(
    _ x: MLXArray,
    weight: MLXArray? = nil,
    eps: Float = 1e-6
) -> MLXArray {
    MLXFast.rmsNorm(
        x,
        weight: weight ?? MLXArray.ones([x.dim(x.ndim - 1)], dtype: x.dtype),
        eps: eps
    )
}

struct Q35GDNPreworkOutput {
    let q: MLXArray
    let k: MLXArray
    let v: MLXArray
    let convState: MLXArray
}

func q35GDNPreworkOps(
    qkv: MLXArray,
    convState: MLXArray,
    convWeight: MLXArray,
    numKeyHeads: Int,
    numValueHeads: Int,
    keyHeadDim: Int,
    valueHeadDim: Int,
    rmsNormWeight: MLXArray? = nil,
    normalizeInFloat32: Bool = false
) -> Q35GDNPreworkOutput {
    let batch = qkv.dim(0)
    let sequence = qkv.dim(1)
    let convDim = qkv.dim(2)
    let keep = convWeight.dim(1) - 1
    let convInput = MLX.concatenated([convState, qkv], axis: 1)
    let nextState = convInput[0..., (convInput.dim(1) - keep)..., 0...]
    let convOut = q35Silu(
        MLX.conv1d(convInput, convWeight, groups: convDim)
    )
    let keyDim = numKeyHeads * keyHeadDim
    let qConv = convOut[.ellipsis, 0..<keyDim]
    let kConv = convOut[.ellipsis, keyDim..<(2 * keyDim)]
    let vConv = convOut[.ellipsis, (2 * keyDim)...]

    var q = qConv.reshaped(batch, sequence, numKeyHeads, keyHeadDim)
    var k = kConv.reshaped(batch, sequence, numKeyHeads, keyHeadDim)
    let v = vConv.reshaped(batch, sequence, numValueHeads, valueHeadDim)
    let invScale = 1.0 / sqrt(Float(max(1, keyHeadDim)))
    if normalizeInFloat32 {
        // Qwen4Exp's reference/FLA normalizes Q/K in FP32 before the
        // recurrent update. BF16 RMSNorm followed by scaling introduces
        // an extra rounding step and changes epsilon from a sum to a mean.
        q = q.asType(.float32)
        k = k.asType(.float32)
        q = q * MLX.rsqrt(MLX.square(q).sum(axis: -1, keepDims: true) + 1e-6) * invScale
        k = k * MLX.rsqrt(MLX.square(k).sum(axis: -1, keepDims: true) + 1e-6)
        return Q35GDNPreworkOutput(q: q, k: k, v: v.asType(.float32), convState: nextState)
    }
    q = q35RmsNormNoWeight(q, weight: rmsNormWeight) * MLXArray(invScale * invScale).asType(q.dtype)
    k = q35RmsNormNoWeight(k, weight: rmsNormWeight) * MLXArray(invScale).asType(k.dtype)

    return Q35GDNPreworkOutput(q: q, k: k, v: v, convState: nextState)
}

#if os(macOS)
private enum Q35GDNPreworkMetalKernel {
    // Adapted under Apache-2.0 from the Qwen GDN verification prework source
    // documented in THIRD_PARTY_NOTICES.md.
    static let kernel = MLXFast.metalKernel(
        name: "q35_gdn_verify_prework",
        inputNames: ["qkv", "conv_state", "conv_w", "q_scale", "k_scale"],
        outputNames: ["q_out", "k_out", "v_out", "conv_out"],
        source: """
            uint lane = thread_position_in_threadgroup.x;
            uint row = threadgroup_position_in_grid.y;
            uint logical_head = threadgroup_position_in_grid.z;
            constexpr uint q_heads = uint(HK);
            constexpr uint k_head_base = uint(HK);
            constexpr uint v_head_base = 2 * uint(HK);
            bool is_q = logical_head < q_heads;
            bool is_k = logical_head >= k_head_base && logical_head < v_head_base;
            uint head = is_q ? logical_head
                       : (is_k ? logical_head - k_head_base : logical_head - v_head_base);
            uint channel_base = is_q ? head * uint(DK)
                               : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
                                       : 2 * uint(HK) * uint(DK) + head * uint(DV));
            T activated[4];
            float sumsq = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                uint channel = channel_base + lane * 4 + i;
                float acc = 0.0f;
                for (uint tap = 0; tap < 4; ++tap) {
                    uint input_row = row + tap;
                    const T xv = input_row < uint(NKEEP)
                        ? conv_state[input_row * uint(C) + channel]
                        : qkv[(input_row - uint(NKEEP)) * uint(C) + channel];
                    acc += float(xv) * float(conv_w[channel * 4 + tap]);
                }
                const T conv = T(acc);
                T sy = T(1) / (T(1) + metal::exp(metal::abs(conv)));
                const T act = conv * ((conv < T(0)) ? sy : T(1) - sy);
                activated[i] = act;
                float value = float(act);
                sumsq += value * value;
            }
            if (is_q || is_k) {
                sumsq = simd_sum(sumsq);
                float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
                const T scale = is_q ? q_scale : k_scale;
                uint out_base = (row * uint(HK) + head) * uint(DK) + lane * 4;
                for (uint i = 0; i < 4; ++i) {
                    const T rms = T(1) * T(float(activated[i]) * inv);
                    const T value = scale * rms;
                    if (is_q) {
                        q_out[out_base + i] = value;
                    } else {
                        k_out[out_base + i] = value;
                    }
                }
            } else {
                uint out_base = (row * uint(HV) + head) * uint(DV) + lane * 4;
                for (uint i = 0; i < 4; ++i) {
                    v_out[out_base + i] = activated[i];
                }
            }
            if (row == 0) {
                for (uint state_row = 0; state_row < uint(NKEEP); ++state_row) {
                    uint input_row = uint(S) + state_row;
                    uint state_base = state_row * uint(C) + channel_base + lane * 4;
                    for (uint i = 0; i < 4; ++i) {
                        uint channel = channel_base + lane * 4 + i;
                        conv_out[state_base + i] = input_row < uint(NKEEP)
                            ? conv_state[input_row * uint(C) + channel]
                            : qkv[(input_row - uint(NKEEP)) * uint(C) + channel];
                    }
                }
            }
            """
    )
}

func q35GDNPreworkMetal(
    qkv: MLXArray,
    convState: MLXArray,
    convWeight: MLXArray,
    numKeyHeads: Int,
    numValueHeads: Int,
    keyHeadDim: Int,
    valueHeadDim: Int
) -> Q35GDNPreworkOutput? {
    guard Device.defaultDevice().deviceType == .gpu,
          qkv.ndim == 3,
          qkv.dim(0) == 1,
          (1...9).contains(qkv.dim(1)),
          qkv.dtype == .bfloat16,
          convState.shape == [1, 3, qkv.dim(2)],
          convState.dtype == .bfloat16,
          convWeight.shape == [qkv.dim(2), 4, 1],
          convWeight.dtype == .bfloat16,
          keyHeadDim == 128,
          valueHeadDim == 128,
          qkv.dim(2) == 2 * numKeyHeads * keyHeadDim + numValueHeads * valueHeadDim else {
        return nil
    }

    let sequence = qkv.dim(1)
    let convDim = qkv.dim(2)
    let invScale = 1.0 / sqrt(Float(keyHeadDim))
    let qScale = MLXArray(invScale * invScale).asType(.bfloat16)
    let kScale = MLXArray(invScale).asType(.bfloat16)
    let outputs = Q35GDNPreworkMetalKernel.kernel(
        [qkv, convState, convWeight, qScale, kScale],
        template: [
            ("T", qkv.dtype),
            ("HK", numKeyHeads),
            ("HV", numValueHeads),
            ("DK", keyHeadDim),
            ("DV", valueHeadDim),
            ("NKEEP", 3),
            ("C", convDim),
            ("S", sequence),
        ],
        grid: (32, sequence, 2 * numKeyHeads + numValueHeads),
        threadGroup: (32, 1, 1),
        outputShapes: [
            [1, sequence, numKeyHeads, keyHeadDim],
            [1, sequence, numKeyHeads, keyHeadDim],
            [1, sequence, numValueHeads, valueHeadDim],
            [1, 3, convDim],
        ],
        outputDTypes: Array(repeating: qkv.dtype, count: 4)
    )
    return Q35GDNPreworkOutput(
        q: outputs[0],
        k: outputs[1],
        v: outputs[2],
        convState: outputs[3]
    )
}
#endif

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

@inline(__always)
private func q35ComputeG(aLog: MLXArray, a: MLXArray, dtBias: MLXArray) -> MLXArray {
    Q35CompiledOperations.current.computeG(aLog, a, dtBias)
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

fileprivate struct Q35LinearVerificationReplay {
    let baseConvState: MLXArray
    let baseRecurrentState: MLXArray?
    let qkv: MLXArray
    let q: MLXArray
    let k: MLXArray
    let v: MLXArray
    let a: MLXArray
    let b: MLXArray
    let aLog: MLXArray
    let dtBias: MLXArray
    let numKeyHeads: Int
    let numValueHeads: Int
    let valueHeadDim: Int
    let convKeep: Int
}

public final class Q35LinearCache: @unchecked Sendable {
    var convState: MLXArray?
    var recurrentState: MLXArray?
    fileprivate var verificationReplay: Q35LinearVerificationReplay?
    var pleConvState: MLXArray?
    var pleTokenContext: MLXArray?
    var pleVerificationReplay: Q38PLEVerificationReplay?

    public init() {}

    public func reset() {
        convState = nil
        recurrentState = nil
        verificationReplay = nil
        pleConvState = nil
        pleTokenContext = nil
        pleVerificationReplay = nil
    }

    public func fork() -> Q35LinearCache {
        let copy = Q35LinearCache()
        copy.convState = convState
        copy.recurrentState = recurrentState
        copy.pleConvState = pleConvState
        copy.pleTokenContext = pleTokenContext
        return copy
    }

    func canRestoreVerificationPrefix(tokenCount: Int) -> Bool {
        guard let verificationReplay else { return false }
        if pleConvState != nil || pleTokenContext != nil {
            guard let pleVerificationReplay,
                  pleVerificationReplay.tokenCount == verificationReplay.q.dim(1) else { return false }
        }
        return tokenCount > 0 && tokenCount <= verificationReplay.q.dim(1)
    }

    /// Commits a prefix of a speculative verification without reading target
    /// weights again. Convolution state is sliced from the saved projection;
    /// recurrent state replays only the already-computed GDN inputs. PLE's
    /// convolution and n-gram histories are sliced to the same accepted prefix.
    func restoreVerificationPrefix(tokenCount: Int) -> Bool {
        guard canRestoreVerificationPrefix(tokenCount: tokenCount),
              let replay = verificationReplay else {
            return false
        }
        defer { commitVerification() }

        if tokenCount == replay.q.dim(1) {
            return true
        }

        let qkvPrefix = replay.qkv[0..., 0..<tokenCount, 0...]
        let convInput = MLX.concatenated([replay.baseConvState, qkvPrefix], axis: 1)
        let nextConvState: MLXArray
        if replay.convKeep == 0 {
            nextConvState = MLXArray.zeros(
                [qkvPrefix.dim(0), 0, qkvPrefix.dim(2)],
                dtype: qkvPrefix.dtype
            )
        } else {
            let start = convInput.dim(1) - replay.convKeep
            nextConvState = convInput[0..., start..., 0...]
        }

        let (_, nextRecurrentState) = q35GatedDeltaUpdate(
            q: replay.q[0..., 0..<tokenCount, 0..., 0...],
            k: replay.k[0..., 0..<tokenCount, 0..., 0...],
            v: replay.v[0..., 0..<tokenCount, 0..., 0...],
            a: replay.a[0..., 0..<tokenCount, 0...],
            b: replay.b[0..., 0..<tokenCount, 0...],
            aLog: replay.aLog,
            dtBias: replay.dtBias,
            state: replay.baseRecurrentState,
            numKeyHeads: replay.numKeyHeads,
            numValueHeads: replay.numValueHeads,
            valueHeadDim: replay.valueHeadDim
        )
        convState = nextConvState
        recurrentState = nextRecurrentState
        if let ple = pleVerificationReplay {
            let nextPLEConv = ple.convolutionInput[
                0..., tokenCount..<(tokenCount + ple.convolutionStateLength), 0...
            ]
            let nextPLETokens = ple.tokenHistory[0..., tokenCount..<(tokenCount + ple.tokenContextLength)]
            pleConvState = nextPLEConv
            pleTokenContext = nextPLETokens
            MLX.asyncEval(nextPLEConv, nextPLETokens)
        }
        MLX.asyncEval(nextConvState, nextRecurrentState)
        return true
    }

    func commitVerification() {
        verificationReplay = nil
        pleVerificationReplay = nil
    }

    public func batched(with caches: [Q35LinearCache]) -> Q35LinearCache? {
        guard !caches.isEmpty else { return nil }

        let batchedConvState = Self.batchedState(caches.map(\.convState))
        let batchedRecurrentState = Self.batchedState(caches.map(\.recurrentState))
        let batchedPLEConvState = Self.batchedState(caches.map(\.pleConvState))
        let batchedPLETokenContext = Self.batchedState(caches.map(\.pleTokenContext))
        guard batchedConvState.isValid,
              batchedRecurrentState.isValid,
              batchedPLEConvState.isValid,
              batchedPLETokenContext.isValid else {
            return nil
        }

        let copy = Q35LinearCache()
        copy.convState = batchedConvState.value
        copy.recurrentState = batchedRecurrentState.value
        copy.pleConvState = batchedPLEConvState.value
        copy.pleTokenContext = batchedPLETokenContext.value
        return copy
    }

    public func unbatchedRows(count: Int) -> [Q35LinearCache]? {
        guard count > 0 else { return nil }
        let convRows = Self.unbatchedRows(convState, count: count)
        let recurrentRows = Self.unbatchedRows(recurrentState, count: count)
        let pleConvRows = Self.unbatchedRows(pleConvState, count: count)
        let pleTokenRows = Self.unbatchedRows(pleTokenContext, count: count)
        guard convRows.isValid,
              recurrentRows.isValid,
              pleConvRows.isValid,
              pleTokenRows.isValid else {
            return nil
        }
        return (0..<count).map { index in
            let copy = Q35LinearCache()
            copy.convState = convRows.values?[index]
            copy.recurrentState = recurrentRows.values?[index]
            copy.pleConvState = pleConvRows.values?[index]
            copy.pleTokenContext = pleTokenRows.values?[index]
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
    private let qkNormWeightBF16: MLXArray
    private let isQwen4Exp: Bool
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
        self.qkNormWeightBF16 = MLXArray.ones([self.keyHeadDim], dtype: .bfloat16)
        self.isQwen4Exp = text.isQwen4Exp

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
        self._norm.wrappedValue = Q35RMSNormGated(
            hiddenSize: self.valueHeadDim,
            eps: text.rmsNormEps,
            activation: text.outputGateType
        )
        self._outProj.wrappedValue = Linear(self.valueDim, text.hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: Q35LinearCache?,
        targetVerify: Bool = false
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
        let recurrentState = cache?.recurrentState
        let rmsNormWeight = qkv.dtype == .bfloat16 ? qkNormWeightBF16 : nil
        let prework: Q35GDNPreworkOutput
        #if os(macOS)
        if !isQwen4Exp, cache != nil,
           let fused = q35GDNPreworkMetal(
               qkv: qkv,
               convState: convState,
               convWeight: conv1d.weight,
               numKeyHeads: numKeyHeads,
               numValueHeads: numValueHeads,
               keyHeadDim: keyHeadDim,
               valueHeadDim: valueHeadDim
           ) {
            prework = fused
        } else {
            prework = q35GDNPreworkOps(
                qkv: qkv,
                convState: convState,
                convWeight: conv1d.weight,
                numKeyHeads: numKeyHeads,
                numValueHeads: numValueHeads,
                keyHeadDim: keyHeadDim,
                valueHeadDim: valueHeadDim,
                rmsNormWeight: rmsNormWeight,
                normalizeInFloat32: isQwen4Exp
            )
        }
        #else
        prework = q35GDNPreworkOps(
            qkv: qkv,
            convState: convState,
            convWeight: conv1d.weight,
            numKeyHeads: numKeyHeads,
            numValueHeads: numValueHeads,
            keyHeadDim: keyHeadDim,
            valueHeadDim: valueHeadDim,
            rmsNormWeight: rmsNormWeight,
            normalizeInFloat32: isQwen4Exp
        )
        #endif
        cache?.convState = prework.convState

        let (updated, state) = q35GatedDeltaUpdate(
            q: prework.q,
            k: prework.k,
            v: prework.v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: recurrentState,
            numKeyHeads: numKeyHeads,
            numValueHeads: numValueHeads,
            valueHeadDim: valueHeadDim
        )
        cache?.recurrentState = state
        if targetVerify, let cache {
            cache.verificationReplay = Q35LinearVerificationReplay(
                baseConvState: convState,
                baseRecurrentState: recurrentState,
                qkv: qkv,
                q: prework.q,
                k: prework.k,
                v: prework.v,
                a: a,
                b: b,
                aLog: aLog,
                dtBias: dtBias,
                numKeyHeads: numKeyHeads,
                numValueHeads: numValueHeads,
                valueHeadDim: valueHeadDim,
                convKeep: keep
            )
        } else {
            cache?.commitVerification()
        }

        let normalized = norm(updated.asType(qkv.dtype), gate: z)
        return outProj(normalized.reshaped(batch, sequence, valueDim))
    }
}
