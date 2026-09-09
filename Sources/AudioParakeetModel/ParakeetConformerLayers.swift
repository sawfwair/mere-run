import Foundation
import MLX
import MLXFast
import MLXNN

final class ParakeetRelPositionalEncoding {
    private let modelDim: Int
    private(set) var maxLen: Int
    private let scale: Float
    private var pe: MLXArray

    init(modelDim: Int, maxLen: Int, scaleInput: Bool) {
        precondition(modelDim % 2 == 0)
        self.modelDim = modelDim
        self.maxLen = max(1, maxLen)
        self.scale = scaleInput ? sqrt(Float(modelDim)) : 1.0
        self.pe = ParakeetRelPositionalEncoding.makeEncoding(modelDim: modelDim, maxLen: self.maxLen)
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
        let inputLen = x.dim(1) + offset
        if inputLen > maxLen {
            maxLen = inputLen + 1
            pe = Self.makeEncoding(modelDim: modelDim, maxLen: maxLen)
        }

        let scaled = x * scale

        let bufferLen = pe.dim(1)
        let start = max(0, bufferLen / 2 - (inputLen - 1))
        let end = min(bufferLen, bufferLen / 2 + (inputLen - 1) + 1)
        let posEmb = pe[0..., start..<end, 0...].asType(x.dtype)

        return (scaled, posEmb)
    }

    private static func makeEncoding(modelDim: Int, maxLen: Int) -> MLXArray {
        let total = 2 * maxLen - 1
        let half = modelDim / 2
        let logTimescaleIncrement = logf(10_000) / Float(max(1, half - 1))

        var invTimescales = [Float](repeating: 0, count: half)
        for i in 0..<half {
            invTimescales[i] = expf(Float(i) * -logTimescaleIncrement)
        }

        var data = [Float](repeating: 0, count: total * modelDim)
        for row in 0..<total {
            let position = Float(maxLen - 1 - row)
            let rowOffset = row * modelDim
            for i in 0..<half {
                let value = position * invTimescales[i]
                data[rowOffset + i * 2] = sinf(value)
                data[rowOffset + i * 2 + 1] = cosf(value)
            }
        }

        return MLXArray(data).reshaped(1, total, modelDim)
    }
}

final class ParakeetMultiHeadAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    init(heads: Int, modelDim: Int, bias: Bool) {
        self.heads = heads
        self.headDim = modelDim / max(1, heads)
        self.scale = pow(Float(headDim), -0.5)

        self._linearQ.wrappedValue = Linear(modelDim, modelDim, bias: bias)
        self._linearK.wrappedValue = Linear(modelDim, modelDim, bias: bias)
        self._linearV.wrappedValue = Linear(modelDim, modelDim, bias: bias)
        self._linearOut.wrappedValue = Linear(modelDim, modelDim, bias: bias)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)

        var q = linearQ(x)
        var k = linearK(x)
        var v = linearV(x)

        q = q.reshaped(batch, sequence, heads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, sequence, heads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, sequence, heads, headDim).transposed(0, 2, 1, 3)

        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        let merged = attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, heads * headDim)
        return linearOut(merged)
    }
}

final class ParakeetRelPositionMultiHeadAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear
    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    init(heads: Int, modelDim: Int, bias: Bool) {
        self.heads = heads
        self.headDim = modelDim / max(1, heads)
        self.scale = pow(Float(headDim), -0.5)

        self._linearQ.wrappedValue = Linear(modelDim, modelDim, bias: bias)
        self._linearK.wrappedValue = Linear(modelDim, modelDim, bias: bias)
        self._linearV.wrappedValue = Linear(modelDim, modelDim, bias: bias)
        self._linearOut.wrappedValue = Linear(modelDim, modelDim, bias: bias)
        self._linearPos.wrappedValue = Linear(modelDim, modelDim, bias: false)
        self._posBiasU.wrappedValue = MLX.zeros([heads, headDim], dtype: .float32)
        self._posBiasV.wrappedValue = MLX.zeros([heads, headDim], dtype: .float32)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let querySeq = x.dim(1)

        var q = linearQ(x)
        var k = linearK(x)
        var v = linearV(x)
        var p = linearPos(posEmb)

        let keySeq = k.dim(1)
        let posLen = p.dim(1)

        q = q.reshaped(batch, querySeq, heads, headDim)
        let qU = (q + posBiasU.reshaped(1, 1, heads, headDim)).transposed(0, 2, 1, 3)
        let qV = (q + posBiasV.reshaped(1, 1, heads, headDim)).transposed(0, 2, 1, 3)

        k = k.reshaped(batch, keySeq, heads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, keySeq, heads, headDim).transposed(0, 2, 1, 3)
        p = p.reshaped(batch, posLen, heads, headDim).transposed(0, 2, 1, 3)

        var matrixBD = MLX.matmul(qV, p.transposed(0, 1, 3, 2))
        matrixBD = relShift(matrixBD)
        matrixBD = matrixBD[0..., 0..., 0..., 0..<keySeq] * scale

        let attended = MLXFast.scaledDotProductAttention(
            queries: qU,
            keys: k,
            values: v,
            scale: scale,
            mask: .array(matrixBD)
        )

        let merged = attended.transposed(0, 2, 1, 3).reshaped(batch, querySeq, heads * headDim)
        return linearOut(merged)
    }

    private func relShift(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let heads = x.dim(1)
        let timeQ = x.dim(2)
        let posLen = x.dim(3)

        let padding = MLX.zeros([batch, heads, timeQ, 1], dtype: x.dtype)
        var shifted = MLX.concatenated([padding, x], axis: 3)
        shifted = shifted.reshaped(batch, heads, posLen + 1, timeQ)
        shifted = shifted[0..., 0..., 1..<(posLen + 1), 0...]
        shifted = shifted.reshaped(batch, heads, timeQ, posLen)

        return shifted
    }
}

final class ParakeetFeedForward: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(modelDim: Int, hiddenDim: Int, useBias: Bool) {
        self._linear1.wrappedValue = Linear(modelDim, hiddenDim, bias: useBias)
        self._linear2.wrappedValue = Linear(hiddenDim, modelDim, bias: useBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

final class ParakeetConvolutionModule: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "batch_norm") var batchNorm: BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    init(config: ParakeetEncoderConfig) {
        self._pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: config.modelDim,
            outputChannels: config.modelDim * 2,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            groups: 1,
            bias: config.useBias
        )
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: config.modelDim,
            outputChannels: config.modelDim,
            kernelSize: config.convKernelSize,
            stride: 1,
            padding: (config.convKernelSize - 1) / 2,
            groups: config.modelDim,
            bias: config.useBias
        )
        self._batchNorm.wrappedValue = BatchNorm(featureCount: config.modelDim)
        self._pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: config.modelDim,
            outputChannels: config.modelDim,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            groups: 1,
            bias: config.useBias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = pointwiseConv1(x)
        y = glu(y, axis: 2)
        y = depthwiseConv(y)
        y = batchNorm(y)
        y = silu(y)
        y = pointwiseConv2(y)
        return y
    }
}

final class ParakeetConformerBlock: Module {
    @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var feedForward1: ParakeetFeedForward

    @ModuleInfo(key: "norm_self_att") var normSelfAtt: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttentionRelPos: ParakeetRelPositionMultiHeadAttention

    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "conv") var conv: ParakeetConvolutionModule

    @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var feedForward2: ParakeetFeedForward

    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    let useRelativePosition: Bool

    init(config: ParakeetEncoderConfig) {
        let ffHidden = config.modelDim * config.ffExpansionFactor

        self._normFeedForward1.wrappedValue = LayerNorm(dimensions: config.modelDim)
        self._feedForward1.wrappedValue = ParakeetFeedForward(
            modelDim: config.modelDim,
            hiddenDim: ffHidden,
            useBias: config.useBias
        )

        self._normSelfAtt.wrappedValue = LayerNorm(dimensions: config.modelDim)
        self._selfAttentionRelPos.wrappedValue = ParakeetRelPositionMultiHeadAttention(
            heads: config.heads,
            modelDim: config.modelDim,
            bias: config.useBias
        )

        self._normConv.wrappedValue = LayerNorm(dimensions: config.modelDim)
        self._conv.wrappedValue = ParakeetConvolutionModule(config: config)

        self._normFeedForward2.wrappedValue = LayerNorm(dimensions: config.modelDim)
        self._feedForward2.wrappedValue = ParakeetFeedForward(
            modelDim: config.modelDim,
            hiddenDim: ffHidden,
            useBias: config.useBias
        )

        self._normOut.wrappedValue = LayerNorm(dimensions: config.modelDim)
        self.useRelativePosition = config.selfAttentionModel == "rel_pos"
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray?) -> MLXArray {
        var hidden = x
        hidden = hidden + 0.5 * feedForward1(normFeedForward1(hidden))

        let attentionInput = normSelfAtt(hidden)
        if useRelativePosition {
            let positional = posEmb ?? MLX.zeros([
                attentionInput.dim(0),
                attentionInput.dim(1) * 2 - 1,
                attentionInput.dim(2),
            ], dtype: attentionInput.dtype)
            hidden = hidden + selfAttentionRelPos(attentionInput, posEmb: positional)
        } else {
            hidden = hidden + attentionInput
        }

        hidden = hidden + conv(normConv(hidden))
        hidden = hidden + 0.5 * feedForward2(normFeedForward2(hidden))

        return normOut(hidden)
    }
}
