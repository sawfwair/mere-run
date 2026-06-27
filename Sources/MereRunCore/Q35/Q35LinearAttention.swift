import Foundation
import MLX
import MLXNN

final class Q35RMSNormGated: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    private let eps: Float

    init(hiddenSize: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.ones([hiddenSize])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray?) -> MLXArray {
        let dtype = hiddenStates.dtype
        let hidden32 = hiddenStates.asType(.float32)
        let variance = (hidden32 * hidden32).mean(axis: -1, keepDims: true)
        let scale = weight.reshaped([1, 1, 1, hiddenStates.dim(hiddenStates.ndim - 1)])
        var normalized = (hidden32 * rsqrt(variance + MLXArray(eps))).asType(dtype)
        normalized = normalized * scale
        if let gate {
            normalized = normalized * MLXNN.silu(gate.asType(.float32))
        }
        return normalized.asType(dtype)
    }
}

@inline(__always)
private func q35Swiglu(_ gate: MLXArray, _ up: MLXArray) -> MLXArray {
    MLXNN.silu(gate) * up
}

@inline(__always)
private func q35RmsNormNoWeight(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let squaredMean = (x * x).mean(axis: -1, keepDims: true)
    return x * rsqrt(squaredMean + MLXArray(eps).asType(x.dtype))
}

@inline(__always)
private func q35ComputeG(aLog: MLXArray, a: MLXArray, dtBias: MLXArray) -> MLXArray {
    let dt = softplus(a.asType(.float32) + dtBias.asType(.float32).reshaped(1, 1, dtBias.dim(0)))
    let decayBase = MLX.exp(aLog.asType(.float32)).reshaped(1, 1, dtBias.dim(0))
    return MLX.exp(-decayBase * dt)
}

private func q35GatedDeltaUpdate(
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

        let qkv = inProjQKV(x)
        let z = inProjZ(x).reshaped(batch, sequence, numValueHeads, valueHeadDim)
        let b = inProjB(x)
        let a = inProjA(x)

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
