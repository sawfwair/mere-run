import Foundation
import MLX
import MLXFast
import MLXNN

final class ACEStepTimbreEncoder: Module {
    let config: ACEStepConfig
    let rope: RoPE

    @ModuleInfo(key: "embed_tokens") var embedTokens: Linear
    @ModuleInfo(key: "layers") var layers: [ACEStepEncoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ParameterInfo(key: "special_token") var specialToken: MLXArray

    init(config: ACEStepConfig) {
        self.config = config
        self.rope = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: config.ropeTheta,
            scale: 1.0
        )

        self._embedTokens.wrappedValue = Linear(config.timbreHiddenDim, config.hiddenSize, bias: true)
        self._layers.wrappedValue = (0..<config.numTimbreEncoderHiddenLayers).map { ACEStepEncoderLayer(config: config, layerIdx: $0) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._specialToken.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
    }

    func callAsFunction(
        referAudioAcousticHiddenStatesPacked: MLXArray,
        referAudioOrderMask: MLXArray
    ) -> (embeddings: MLXArray, attentionMask: MLXArray) {
        var x = embedTokens(referAudioAcousticHiddenStatesPacked)

        let seqLen = x.dim(1)
        let fullMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard config.useSlidingWindow, let window = config.slidingWindow, window > 0 else { return fullMask }
            return ACEStepAttentionMasks.bidirectionalMask(attentionMask: nil, seqLen: seqLen, slidingWindow: window)
        }()

        for layer in layers {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode = (layer.attentionType == "sliding_attention") ? slidingMask : fullMask
            x = layer(x, mask: mask, rope: rope)
        }

        x = norm(x)

        // Match reference implementation: treat the first token as the aggregated embedding.
        let packedEmbeddings = x[0..., 0, 0...]
        return Self.unpackTimbreEmbeddings(packedEmbeddings: packedEmbeddings, referAudioOrderMask: referAudioOrderMask)
    }

    private static func unpackTimbreEmbeddings(
        packedEmbeddings: MLXArray, // [N, D]
        referAudioOrderMask: MLXArray // [N]
    ) -> (embeddings: MLXArray, attentionMask: MLXArray) {
        let N = packedEmbeddings.dim(0)
        let D = packedEmbeddings.dim(1)

        let order = referAudioOrderMask.asType(.int32)
        MLX.eval(order)
        let orderValues = order.asArray(Int32.self)
        let B = Int((orderValues.max() ?? 0) + 1)

        var indicesByBatch: [[Int]] = Array(repeating: [], count: B)
        indicesByBatch.reserveCapacity(B)
        for (idx, b) in orderValues.enumerated() {
            let bi = Int(b)
            if bi >= 0 && bi < B {
                indicesByBatch[bi].append(idx)
            }
        }

        let maxCount = max(1, indicesByBatch.map(\.count).max() ?? 0)

        var rows: [MLXArray] = []
        rows.reserveCapacity(B)
        var masks: [Int32] = []
        masks.reserveCapacity(B * maxCount)

        for b in 0..<B {
            let ids = indicesByBatch[b]
            let count = ids.count
            masks.append(contentsOf: Array(repeating: 1, count: count))
            masks.append(contentsOf: Array(repeating: 0, count: maxCount - count))

            if count == 0 {
                rows.append(MLXArray.zeros([1, maxCount, D], dtype: packedEmbeddings.dtype))
                continue
            }

            let gatherIdx = MLXArray(ids.map { Int32($0) }.map(Float32.init), [count]).asType(.int32)
            var gathered = MLX.take(packedEmbeddings, gatherIdx, axis: 0) // [count, D]
            if count < maxCount {
                let pad = MLXArray.zeros([maxCount - count, D], dtype: packedEmbeddings.dtype)
                gathered = MLX.concatenated([gathered, pad], axis: 0)
            }
            rows.append(gathered.reshaped(1, maxCount, D))
        }

        let unpacked = MLX.concatenated(rows, axis: 0) // [B, maxCount, D]
        let mask = MLXArray(masks.map(Float32.init), [B, maxCount]).asType(.int32)
        return (unpacked, mask)
    }
}

