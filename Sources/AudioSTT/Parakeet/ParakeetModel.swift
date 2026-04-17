import Foundation
import MLX
import MLXFast
import MLXNN

protocol ParakeetDecodingModel: AnyObject {
    var config: ParakeetModelConfig { get }
    func decode(_ mel: MLXArray) -> [ParakeetAlignedResult]
}

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

final class ParakeetDwStridingSubsampling: Module {
    let conv: [Module]
    @ModuleInfo(key: "out") var out: Linear

    let samplingNum: Int
    let convChannels: Int
    let stride: Int = 2
    let kernelSize: Int = 3
    let padding: Int = 1

    init(config: ParakeetEncoderConfig) {
        self.samplingNum = max(1, Int(log2(Double(max(1, config.subsamplingFactor)))))
        self.convChannels = config.subsamplingConvChannels

        precondition(config.subsamplingFactor > 0 && (config.subsamplingFactor & (config.subsamplingFactor - 1) == 0))
        precondition(config.subsamplingFactor <= 8, "Parakeet subsampling factors above 8 are not supported.")

        var finalFreqDim = config.featIn
        for _ in 0..<samplingNum {
            finalFreqDim = ((finalFreqDim + 2 * padding - kernelSize) / stride) + 1
        }

        self.conv = [
            Conv2d(
                inputChannels: 1,
                outputChannels: convChannels,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: IntOrPair(padding),
                groups: 1,
                bias: true
            ),
            Identity(),
            Conv2d(
                inputChannels: convChannels,
                outputChannels: convChannels,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: IntOrPair(padding),
                groups: convChannels,
                bias: true
            ),
            Conv2d(
                inputChannels: convChannels,
                outputChannels: convChannels,
                kernelSize: IntOrPair(1),
                stride: IntOrPair(1),
                padding: IntOrPair(0),
                groups: 1,
                bias: true
            ),
            Identity(),
            Conv2d(
                inputChannels: convChannels,
                outputChannels: convChannels,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: IntOrPair(padding),
                groups: convChannels,
                bias: true
            ),
            Conv2d(
                inputChannels: convChannels,
                outputChannels: convChannels,
                kernelSize: IntOrPair(1),
                stride: IntOrPair(1),
                padding: IntOrPair(0),
                groups: 1,
                bias: true
            )
        ]

        self._out.wrappedValue = Linear(convChannels * finalFreqDim, config.modelDim)
    }

    func callAsFunction(_ x: MLXArray, lengths: [Int]) -> (MLXArray, [Int]) {
        func convLayer(at index: Int) -> Conv2d {
            guard let layer = conv[index] as? Conv2d else {
                preconditionFailure("Expected Conv2d at conv[\(index)] in ParakeetDwStridingSubsampling.")
            }
            return layer
        }

        var outLengths = lengths
        for _ in 0..<samplingNum {
            outLengths = outLengths.map { length in
                let value = Int(floor(Double(length + 2 * padding - kernelSize) / Double(stride)) + 1)
                return max(0, value)
            }
        }

        var hidden = x.expandedDimensions(axis: 1) // [B,1,T,F]
        hidden = hidden.transposed(0, 2, 3, 1) // [B,T,F,1]

        hidden = relu(convLayer(at: 0)(hidden))
        if samplingNum >= 2 {
            hidden = convLayer(at: 2)(hidden)
            hidden = convLayer(at: 3)(hidden)
            hidden = relu(hidden)
        }
        if samplingNum >= 3 {
            hidden = convLayer(at: 5)(hidden)
            hidden = convLayer(at: 6)(hidden)
            hidden = relu(hidden)
        }

        hidden = hidden.transposed(0, 3, 1, 2) // [B,C,T,F]
        let batch = hidden.dim(0)
        let channels = hidden.dim(1)
        let time = hidden.dim(2)
        let freq = hidden.dim(3)
        hidden = hidden.transposed(0, 2, 1, 3).reshaped(batch, time, channels * freq)
        hidden = out(hidden)

        return (hidden, outLengths)
    }
}

final class ParakeetConformer: Module {
    @ModuleInfo(key: "pre_encode") var preEncode: ParakeetDwStridingSubsampling
    @ModuleInfo(key: "layers") var layers: [ParakeetConformerBlock]

    let config: ParakeetEncoderConfig
    let posEnc: ParakeetRelPositionalEncoding?

    init(config: ParakeetEncoderConfig) {
        self.config = config
        self._preEncode.wrappedValue = ParakeetDwStridingSubsampling(config: config)
        self._layers.wrappedValue = (0..<config.layers).map { _ in ParakeetConformerBlock(config: config) }

        if config.selfAttentionModel == "rel_pos" {
            self.posEnc = ParakeetRelPositionalEncoding(
                modelDim: config.modelDim,
                maxLen: config.posEmbMaxLen,
                scaleInput: config.xScaling
            )
        } else {
            self.posEnc = nil
        }
    }

    func callAsFunction(_ x: MLXArray, lengths: [Int]? = nil) -> (MLXArray, [Int]) {
        let inLengths = lengths ?? Array(repeating: x.dim(1), count: x.dim(0))
        var (hidden, outLengths) = preEncode(x, lengths: inLengths)

        var posEmb: MLXArray?
        if let posEnc {
            let output = posEnc(hidden)
            hidden = output.0
            posEmb = output.1
        }

        for layer in layers {
            hidden = layer(hidden, posEmb: posEmb)
        }

        return (hidden, outLengths)
    }
}

final class ParakeetLSTMStack: Module {
    @ModuleInfo(key: "lstm") var lstm: [LSTM]

    let hiddenSize: Int

    init(inputSize: Int, hiddenSize: Int, numLayers: Int, bias: Bool) {
        self.hiddenSize = hiddenSize
        self._lstm.wrappedValue = (0..<numLayers).map { index in
            LSTM(
                inputSize: index == 0 ? inputSize : hiddenSize,
                hiddenSize: hiddenSize,
                bias: bias
            )
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        state: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        var output = x
        var nextH: [MLXArray] = []
        var nextC: [MLXArray] = []
        nextH.reserveCapacity(lstm.count)
        nextC.reserveCapacity(lstm.count)

        for index in 0..<lstm.count {
            let hidden = state?.0[index, 0..., 0...]
            let cell = state?.1[index, 0..., 0...]

            let (allH, allC) = lstm[index](output, hidden: hidden, cell: cell)
            output = allH

            let lastStep = max(0, allH.dim(1) - 1)
            nextH.append(allH[0..., lastStep, 0...])
            nextC.append(allC[0..., lastStep, 0...])
        }

        return (
            output,
            (
                MLX.stacked(nextH, axis: 0),
                MLX.stacked(nextC, axis: 0)
            )
        )
    }
}

final class ParakeetPredictNetwork: Module {
    final class Prediction: Module {
        @ModuleInfo(key: "embed") var embed: Embedding
        @ModuleInfo(key: "dec_rnn") var decRNN: ParakeetLSTMStack

        init(config: ParakeetRNNTDecoderConfig) {
            let embeddingSize = config.blankAsPad ? config.vocabSize + 1 : config.vocabSize
            self._embed.wrappedValue = Embedding(
                embeddingCount: embeddingSize,
                dimensions: config.prednet.predHidden
            )

            self._decRNN.wrappedValue = ParakeetLSTMStack(
                inputSize: config.prednet.predHidden,
                hiddenSize: config.prednet.rnnHiddenSize ?? config.prednet.predHidden,
                numLayers: config.prednet.predRnnLayers,
                bias: true
            )
        }
    }

    @ModuleInfo(key: "prediction") var prediction: Prediction

    private let predHidden: Int

    init(config: ParakeetRNNTDecoderConfig) {
        self.predHidden = config.prednet.predHidden
        self._prediction.wrappedValue = Prediction(config: config)
    }

    func callAsFunction(
        _ y: MLXArray?,
        state: (MLXArray, MLXArray)?
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let embedded: MLXArray
        if let y {
            embedded = prediction.embed(y)
        } else {
            let batch = state?.0.dim(1) ?? 1
            embedded = MLX.zeros([batch, 1, predHidden], dtype: prediction.embed.weight.dtype)
        }

        return prediction.decRNN(embedded, state: state)
    }
}

final class ParakeetJointNetwork: Module {
    final class Activation: Module {
        let kind: String

        init(kind: String) {
            self.kind = kind.lowercased()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            switch kind {
            case "relu":
                relu(x)
            case "sigmoid":
                sigmoid(x)
            case "tanh":
                tanh(x)
            default:
                relu(x)
            }
        }
    }

    @ModuleInfo(key: "pred") var pred: Linear
    @ModuleInfo(key: "enc") var enc: Linear
    let joint_net: (Activation, Identity, Linear)

    init(config: ParakeetJointConfig) {
        self._pred.wrappedValue = Linear(
            config.jointnet.predHidden,
            config.jointnet.jointHidden,
            bias: true
        )
        self._enc.wrappedValue = Linear(
            config.jointnet.encoderHidden,
            config.jointnet.jointHidden,
            bias: true
        )
        self.joint_net = (
            Activation(kind: config.jointnet.activation),
            Identity(),
            Linear(
                config.jointnet.jointHidden,
                config.numClasses + 1 + config.numExtraOutputs,
                bias: true
            )
        )
    }

    func callAsFunction(_ encInput: MLXArray, _ predInput: MLXArray) -> MLXArray {
        let encProj = enc(encInput)
        let predProj = pred(predInput)
        let hidden = encProj.expandedDimensions(axis: 2) + predProj.expandedDimensions(axis: 1)
        return joint_net.2(joint_net.1(joint_net.0(hidden)))
    }
}

final class ParakeetConvASRDecoder: Module {
    @ModuleInfo(key: "decoder_layers.0") var projection: Conv1d

    let temperature: Float = 1.0

    init(config: ParakeetCTCDecoderConfig) {
        let classCount = (config.numClasses > 0 ? config.numClasses : config.vocabulary.count) + 1
        self._projection.wrappedValue = Conv1d(
            inputChannels: config.featIn,
            outputChannels: classCount,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            groups: 1,
            bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        logSoftmax(projection(x) / temperature, axis: -1)
    }
}

class ParakeetBaseModel: Module, ParakeetDecodingModel {
    let config: ParakeetModelConfig
    @ModuleInfo(key: "encoder") var encoder: ParakeetConformer

    init(config: ParakeetModelConfig) {
        self.config = config
        self._encoder.wrappedValue = ParakeetConformer(config: config.encoder)
    }

    func decode(_ mel: MLXArray) -> [ParakeetAlignedResult] {
        []
    }

    var timePerEncoderStep: TimeInterval {
        TimeInterval(config.encoder.subsamplingFactor * config.preprocessor.hopLength)
            / TimeInterval(config.preprocessor.sampleRate)
    }

    func normalizeBatch(_ mel: MLXArray) -> MLXArray {
        if mel.ndim == 2 {
            return mel.expandedDimensions(axis: 0)
        }
        return mel
    }

    func tokenText(_ token: Int) -> String {
        ParakeetTokenizer.decode(tokens: [token], vocabulary: config.vocabulary)
    }
}

class ParakeetTDTModel: ParakeetBaseModel {
    @ModuleInfo(key: "decoder") var decoder: ParakeetPredictNetwork
    @ModuleInfo(key: "joint") var joint: ParakeetJointNetwork

    let durations: [Int]
    let maxSymbols: Int?

    override init(config: ParakeetModelConfig) {
        self.durations = config.tdtDurations ?? [0, 1, 2, 3, 4]
        self.maxSymbols = config.maxSymbols

        self._decoder.wrappedValue = ParakeetPredictNetwork(config: config.rnntDecoder ?? ParakeetRNNTDecoderConfig(
            blankAsPad: true,
            vocabSize: max(1, config.vocabulary.count),
            prednet: ParakeetPredictNetConfig(predHidden: 640, predRnnLayers: 2, rnnHiddenSize: nil)
        ))
        self._joint.wrappedValue = ParakeetJointNetwork(config: config.joint ?? ParakeetJointConfig(
            numClasses: max(1, config.vocabulary.count),
            vocabulary: config.vocabulary,
            jointnet: ParakeetJointNetConfig(jointHidden: 640, activation: "relu", encoderHidden: config.encoder.modelDim, predHidden: config.rnntDecoder?.prednet.predHidden ?? 640),
            numExtraOutputs: max(1, (config.tdtDurations ?? [0, 1]).count)
        ))

        super.init(config: config)
    }

    override func decode(_ mel: MLXArray) -> [ParakeetAlignedResult] {
        let batch = normalizeBatch(mel)
        let (encoded, lengths) = encoder(batch)
        MLX.eval(encoded)

        var results: [ParakeetAlignedResult] = []
        results.reserveCapacity(batch.dim(0))

        let vocabularySize = config.vocabulary.count
        let stepSeconds = timePerEncoderStep

        for b in 0..<batch.dim(0) {
            let features = encoded[b..<(b + 1), 0..., 0...]
            let maxLength = min(lengths[b], features.dim(1))

            var lastToken = vocabularySize
            var tokens: [ParakeetAlignedToken] = []
            var time = 0
            var newSymbols = 0
            var state: (MLXArray, MLXArray)?

            while time < maxLength {
                let feature = features[0..., time..<(time + 1), 0...]

                let tokenInput: MLXArray?
                if lastToken == vocabularySize {
                    tokenInput = nil
                } else {
                    tokenInput = MLXArray([Int32(lastToken)]).reshaped(1, 1)
                }

                let (decoderOutput, proposedState) = decoder(tokenInput, state: state)
                let jointOutput = joint(feature, decoderOutput.asType(feature.dtype))

                let predictionSlice = jointOutput[0, 0, 0, 0..<(vocabularySize + 1)]
                let durationSlice = jointOutput[0, 0, 0, (vocabularySize + 1)..<jointOutput.dim(3)]

                let predictedToken = Int(MLX.argMax(predictionSlice).item(Int32.self))
                let decisionIndex = durationSlice.count > 0
                    ? Int(MLX.argMax(durationSlice).item(Int32.self))
                    : 1
                let clampedDecision = min(max(0, decisionIndex), max(0, durations.count - 1))
                let durationSteps = max(0, durations[clampedDecision])

                if predictedToken != vocabularySize {
                    let start = TimeInterval(time) * stepSeconds
                    let duration = TimeInterval(durationSteps) * stepSeconds
                    tokens.append(
                        ParakeetAlignedToken(
                            id: predictedToken,
                            text: tokenText(predictedToken),
                            start: start,
                            duration: duration
                        )
                    )
                    lastToken = predictedToken
                    state = proposedState
                }

                time += durationSteps
                newSymbols += 1

                if durationSteps != 0 {
                    newSymbols = 0
                } else if let maxSymbols, maxSymbols <= newSymbols {
                    time += 1
                    newSymbols = 0
                }
            }

            let sentences = ParakeetAlignment.tokensToSentences(tokens)
            results.append(ParakeetAlignment.sentencesToResult(sentences))
        }

        return results
    }
}

class ParakeetRNNTModel: ParakeetBaseModel {
    @ModuleInfo(key: "decoder") var decoder: ParakeetPredictNetwork
    @ModuleInfo(key: "joint") var joint: ParakeetJointNetwork

    let maxSymbols: Int?

    override init(config: ParakeetModelConfig) {
        self.maxSymbols = config.maxSymbols

        self._decoder.wrappedValue = ParakeetPredictNetwork(config: config.rnntDecoder ?? ParakeetRNNTDecoderConfig(
            blankAsPad: true,
            vocabSize: max(1, config.vocabulary.count),
            prednet: ParakeetPredictNetConfig(predHidden: 640, predRnnLayers: 2, rnnHiddenSize: nil)
        ))
        self._joint.wrappedValue = ParakeetJointNetwork(config: config.joint ?? ParakeetJointConfig(
            numClasses: max(1, config.vocabulary.count),
            vocabulary: config.vocabulary,
            jointnet: ParakeetJointNetConfig(jointHidden: 640, activation: "relu", encoderHidden: config.encoder.modelDim, predHidden: config.rnntDecoder?.prednet.predHidden ?? 640),
            numExtraOutputs: 0
        ))

        super.init(config: config)
    }

    override func decode(_ mel: MLXArray) -> [ParakeetAlignedResult] {
        let batch = normalizeBatch(mel)
        let (encoded, lengths) = encoder(batch)
        MLX.eval(encoded)

        var results: [ParakeetAlignedResult] = []
        results.reserveCapacity(batch.dim(0))

        let vocabularySize = config.vocabulary.count
        let stepSeconds = timePerEncoderStep

        for b in 0..<batch.dim(0) {
            let features = encoded[b..<(b + 1), 0..., 0...]
            let maxLength = min(lengths[b], features.dim(1))

            var lastToken = vocabularySize
            var tokens: [ParakeetAlignedToken] = []
            var time = 0
            var newSymbols = 0
            var state: (MLXArray, MLXArray)?

            while time < maxLength {
                let feature = features[0..., time..<(time + 1), 0...]

                let tokenInput: MLXArray?
                if lastToken == vocabularySize {
                    tokenInput = nil
                } else {
                    tokenInput = MLXArray([Int32(lastToken)]).reshaped(1, 1)
                }

                let (decoderOutput, proposedState) = decoder(tokenInput, state: state)
                let jointOutput = joint(feature, decoderOutput.asType(feature.dtype))
                let predictedToken = Int(MLX.argMax(jointOutput).item(Int32.self))

                if predictedToken != vocabularySize {
                    let start = TimeInterval(time) * stepSeconds
                    tokens.append(
                        ParakeetAlignedToken(
                            id: predictedToken,
                            text: tokenText(predictedToken),
                            start: start,
                            duration: stepSeconds
                        )
                    )
                    lastToken = predictedToken
                    state = proposedState

                    newSymbols += 1
                    if let maxSymbols, maxSymbols <= newSymbols {
                        time += 1
                        newSymbols = 0
                    }
                } else {
                    time += 1
                    newSymbols = 0
                }
            }

            let sentences = ParakeetAlignment.tokensToSentences(tokens)
            results.append(ParakeetAlignment.sentencesToResult(sentences))
        }

        return results
    }
}

class ParakeetCTCModel: ParakeetBaseModel {
    @ModuleInfo(key: "decoder") var decoder: ParakeetConvASRDecoder

    override init(config: ParakeetModelConfig) {
        self._decoder.wrappedValue = ParakeetConvASRDecoder(config: config.ctcDecoder ?? ParakeetCTCDecoderConfig(
            featIn: config.encoder.modelDim,
            numClasses: max(1, config.vocabulary.count),
            vocabulary: config.vocabulary
        ))
        super.init(config: config)
    }

    override func decode(_ mel: MLXArray) -> [ParakeetAlignedResult] {
        let batch = normalizeBatch(mel)
        let (encoded, lengths) = encoder(batch)
        let logits = decoder(encoded)
        MLX.eval(logits)

        var results: [ParakeetAlignedResult] = []
        results.reserveCapacity(batch.dim(0))

        let vocabulary = config.vocabulary
        let blank = vocabulary.count
        let stepSeconds = timePerEncoderStep

        for b in 0..<batch.dim(0) {
            let featureLength = min(lengths[b], logits.dim(1))
            let predictions = logits[b, 0..<featureLength, 0...]
            let bestTokens = MLX.argMax(predictions, axis: 1)

            var hypothesis: [ParakeetAlignedToken] = []
            var tokenBoundaries: [(Int, Int?)] = []
            var previousToken = -1

            for t in 0..<featureLength {
                let token = Int(bestTokens[t].item(Int32.self))
                if token == blank {
                    continue
                }
                if token == previousToken {
                    continue
                }

                if previousToken != -1, let previousStart = tokenBoundaries.last?.0 {
                    let start = TimeInterval(previousStart) * stepSeconds
                    let end = TimeInterval(t) * stepSeconds
                    hypothesis.append(
                        ParakeetAlignedToken(
                            id: previousToken,
                            text: ParakeetTokenizer.decode(tokens: [previousToken], vocabulary: vocabulary),
                            start: start,
                            duration: max(0, end - start)
                        )
                    )
                }

                tokenBoundaries.append((t, nil))
                previousToken = token
            }

            if previousToken != -1, let previousStart = tokenBoundaries.last?.0 {
                var lastNonBlank = max(0, featureLength - 1)
                if featureLength > 1 {
                    for t in stride(from: featureLength - 1, through: previousStart, by: -1) {
                        let token = Int(bestTokens[t].item(Int32.self))
                        if token != blank {
                            lastNonBlank = t
                            break
                        }
                    }
                }

                let start = TimeInterval(previousStart) * stepSeconds
                let end = TimeInterval(lastNonBlank + 1) * stepSeconds
                hypothesis.append(
                    ParakeetAlignedToken(
                        id: previousToken,
                        text: ParakeetTokenizer.decode(tokens: [previousToken], vocabulary: vocabulary),
                        start: start,
                        duration: max(0, end - start)
                    )
                )
            }

            let sentences = ParakeetAlignment.tokensToSentences(hypothesis)
            results.append(ParakeetAlignment.sentencesToResult(sentences))
        }

        return results
    }
}

class ParakeetTDTCTCModel: ParakeetTDTModel {
    @ModuleInfo(key: "ctc_decoder") var ctcDecoder: ParakeetConvASRDecoder

    init(config: ParakeetModelConfig, ctcConfig: ParakeetCTCDecoderConfig?) {
        self._ctcDecoder.wrappedValue = ParakeetConvASRDecoder(config: ctcConfig ?? ParakeetCTCDecoderConfig(
            featIn: config.encoder.modelDim,
            numClasses: max(1, config.vocabulary.count),
            vocabulary: config.vocabulary
        ))
        super.init(config: config)
    }
}

enum ParakeetModelFactory {
    static func build(config: ParakeetModelConfig) -> any ParakeetDecodingModel {
        switch config.variant {
        case .tdt:
            return ParakeetTDTModel(config: config)
        case .tdtCTC:
            return ParakeetTDTCTCModel(config: config, ctcConfig: config.ctcDecoder)
        case .rnnt:
            return ParakeetRNNTModel(config: config)
        case .ctc:
            return ParakeetCTCModel(config: config)
        }
    }
}
