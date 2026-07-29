// Adapted from mlx-audio-swift at commit 4266f988d170a83017d1e82e2e4654602f277f1d.
// Copyright (c) 2025 Prince Canuma. Licensed under the MIT License.
import Foundation
import MLX
import MLXNN

// MARK: - FastConformer Encoder Components

/// Depthwise-striding convolutional subsampling (factor=8).
private class ConvSubsampling: Module {
    @ModuleInfo var layers_0: Conv2d
    @ModuleInfo var layers_2: Conv2d
    @ModuleInfo var layers_3: Conv2d
    @ModuleInfo var layers_5: Conv2d
    @ModuleInfo var layers_6: Conv2d
    @ModuleInfo var linear: Linear

    init(_ config: FCEncoderConfig) {
        let convChannels = config.subsamplingConvChannels
        let featOut = config.hiddenSize
        let ks = config.subsamplingConvKernelSize
        let stride = config.subsamplingConvStride
        let pad = (ks - 1) / 2
        let ksPair = IntOrPair((ks, ks))
        let stridePair = IntOrPair((stride, stride))
        let padPair = IntOrPair((pad, pad))

        self._layers_0.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: convChannels,
            kernelSize: ksPair, stride: stridePair, padding: padPair
        )
        self._layers_2.wrappedValue = Conv2d(
            inputChannels: convChannels, outputChannels: convChannels,
            kernelSize: ksPair, stride: stridePair, padding: padPair,
            groups: convChannels
        )
        self._layers_3.wrappedValue = Conv2d(
            inputChannels: convChannels, outputChannels: convChannels,
            kernelSize: 1
        )
        self._layers_5.wrappedValue = Conv2d(
            inputChannels: convChannels, outputChannels: convChannels,
            kernelSize: ksPair, stride: stridePair, padding: padPair,
            groups: convChannels
        )
        self._layers_6.wrappedValue = Conv2d(
            inputChannels: convChannels, outputChannels: convChannels,
            kernelSize: 1
        )

        let featIn = config.numMelBins
        let linearIn = convChannels * Int(ceil(Double(featIn) / 8.0))
        self._linear.wrappedValue = Linear(linearIn, featOut)
    }

    /// - Parameters:
    ///   - x: `(batch, featDim, time)` mel spectrogram
    ///   - lengths: `(batch,)` frame lengths
    /// - Returns: `(x: (batch, time/8, hiddenSize), lengths: (batch,))`
    func callAsFunction(_ x: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        // (batch, feat, time) → NHWC: (batch, time, feat, 1)
        var h = x.transposed(0, 2, 1).expandedDimensions(axis: -1)

        h = relu(layers_0(h))
        h = relu(layers_3(layers_2(h)))
        h = relu(layers_6(layers_5(h)))

        // NHWC → (b, t, c, f) for flatten
        let (b, t, f, c) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        h = h.transposed(0, 1, 3, 2).reshaped(b, t, c * f)
        h = linear(h)

        // floor((L - 1) / 2) + 1 per stride-2 stage
        var outLengths = lengths.asType(.float32)
        for _ in 0..<3 {
            outLengths = MLX.floor((outLengths - 1) / 2).asType(.int32) + 1
        }

        return (h, outLengths)
    }
}

/// Relative positional encoding for Conformer (Transformer-XL style).
private class RelPositionalEncoding: Module {
    let dModel: Int

    init(dModel: Int) {
        self.dModel = dModel
    }

    /// Generate relative positional encoding.
    /// - Parameter x: `(batch, time, dModel)`
    /// - Returns: `(1, 2*time-1, dModel)`
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let seqLen = x.dim(1)
        let positions = MLXArray(stride(from: seqLen - 1, through: -(seqLen - 1), by: -1).map { Float($0) })

        let dim = MLXArray(stride(from: 0, to: dModel, by: 2).map { Float($0) })
        let divTerm = MLX.exp(dim * Float(-log(10000.0) / Double(dModel)))

        let angles = positions.expandedDimensions(axis: 1) * divTerm.expandedDimensions(axis: 0)
        // Build PE: interleave sin/cos
        let sinAngles = MLX.sin(angles)
        let cosAngles = MLX.cos(angles)
        // (posLen, dModel/2, 2) → (posLen, dModel)
        let pe = MLX.stacked([sinAngles, cosAngles], axis: -1)
            .reshaped(positions.dim(0), dModel)
        return pe.expandedDimensions(axis: 0).asType(x.dtype)
    }
}

/// Multi-head attention with relative positional encoding (Transformer-XL).
private class RelPositionMultiHeadAttention: Module {
    let h: Int
    let dK: Int
    let sDK: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear

    @ParameterInfo(key: "bias_u") var biasU: MLXArray
    @ParameterInfo(key: "bias_v") var biasV: MLXArray

    init(_ config: FCEncoderConfig) {
        let nFeat = config.hiddenSize
        let nHead = config.numAttentionHeads
        h = nHead
        dK = nFeat / nHead
        sDK = sqrt(Float(dK))

        self._qProj.wrappedValue = Linear(nFeat, nFeat, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(nFeat, nFeat, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(nFeat, nFeat, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(nFeat, nFeat, bias: config.attentionBias)
        self._relativeKProj.wrappedValue = Linear(nFeat, nFeat, bias: false)

        self._biasU.wrappedValue = MLXArray.zeros([nHead, dK])
        self._biasV.wrappedValue = MLXArray.zeros([nHead, dK])
    }

    private func relShift(_ x: MLXArray) -> MLXArray {
        let (b, headCount, qlen, posLen) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        // Pad left
        var padded = MLX.padded(x, widths: [.init((0, 0)), .init((0, 0)), .init((0, 0)), .init((1, 0))])
        padded = padded.reshaped(b, headCount, posLen + 1, qlen)
        padded = padded[0..., 0..., 1..., 0...].reshaped(b, headCount, qlen, posLen)
        return padded
    }

    func callAsFunction(
        query: MLXArray, key: MLXArray, value: MLXArray,
        mask: MLXArray? = nil, posEmb: MLXArray? = nil
    ) -> MLXArray {
        let nBatch = query.dim(0)

        let q = qProj(query).reshaped(nBatch, -1, h, dK).transposed(0, 2, 1, 3)
        let k = kProj(key).reshaped(nBatch, -1, h, dK).transposed(0, 2, 1, 3)
        let v = vProj(value).reshaped(nBatch, -1, h, dK).transposed(0, 2, 1, 3)

        let qT = q.transposed(0, 2, 1, 3)

        let p = relativeKProj(posEmb!).reshaped(1, -1, h, dK).transposed(0, 2, 1, 3)

        let qWithBiasU = (qT + biasU).transposed(0, 2, 1, 3)
        let qWithBiasV = (qT + biasV).transposed(0, 2, 1, 3)

        let matrixAC = MLX.matmul(qWithBiasU, k.transposed(0, 1, 3, 2))
        var matrixBD = MLX.matmul(qWithBiasV, p.transposed(0, 1, 3, 2))
        matrixBD = relShift(matrixBD)
        matrixBD = matrixBD[0..., 0..., 0..., ..<matrixAC.dim(3)]

        var scores = (matrixAC + matrixBD) / sDK

        if let mask {
            scores = MLX.where(mask, MLXArray(-1e4).asType(scores.dtype), scores)
        }

        var attn = softmax(scores, axis: -1)
        if let mask {
            attn = MLX.where(mask, MLXArray(Float(0)).asType(scores.dtype), attn)
        }

        let out = MLX.matmul(attn, v)
        let reshaped = out.transposed(0, 2, 1, 3).reshaped(nBatch, -1, h * dK)
        return oProj(reshaped)
    }
}

/// Conformer feed-forward module.
private class ConformerFeedForward: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(dModel: Int, dFf: Int) {
        self._linear1.wrappedValue = Linear(dModel, dFf)
        self._linear2.wrappedValue = Linear(dFf, dModel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

/// Batch normalization using stored running statistics (inference mode only).
private class BatchNorm1d: Module {
    let eps: Float
    var weight: MLXArray
    var bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray

    init(numFeatures: Int, eps: Float = 1e-5) {
        self.eps = eps
        weight = MLXArray.ones([numFeatures])
        bias = MLXArray.zeros([numFeatures])
        self._runningMean.wrappedValue = MLXArray.zeros([numFeatures])
        self._runningVar.wrappedValue = MLXArray.ones([numFeatures])
    }

    /// Apply batch norm using running stats.
    /// - Parameter x: `(batch, time, features)`
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x - runningMean) / MLX.sqrt(runningVar + eps) * weight + bias
    }
}

/// Conformer convolution module with GLU, depthwise conv, and batch norm.
private class ConformerConvolution: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo var norm: BatchNorm1d
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    init(_ config: FCEncoderConfig) {
        let dModel = config.hiddenSize
        let kernelSize = config.convKernelSize

        self._pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel * 2,
            kernelSize: 1, bias: true
        )
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel,
            kernelSize: kernelSize,
            padding: (kernelSize - 1) / 2,
            groups: dModel,
            bias: true
        )
        self._norm.wrappedValue = BatchNorm1d(numFeatures: dModel)
        self._pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel,
            kernelSize: 1, bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = pointwiseConv1(x)

        // GLU
        let parts = MLX.split(h, parts: 2, axis: -1)
        h = parts[0] * sigmoid(parts[1])

        h = depthwiseConv(h)
        h = norm(h)
        h = silu(h)
        h = pointwiseConv2(h)
        return h
    }
}

/// Single Conformer encoder layer: FF1 → Self-Attn → Conv → FF2 → LN
private class ConformerLayer: Module {
    let fcFactor: Float = 0.5

    @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var feedForward1: ConformerFeedForward
    @ModuleInfo(key: "norm_self_att") var normSelfAtt: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: RelPositionMultiHeadAttention
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo var conv: ConformerConvolution
    @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var feedForward2: ConformerFeedForward
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    init(_ config: FCEncoderConfig) {
        let dModel = config.hiddenSize
        let dFf = config.intermediateSize

        self._normFeedForward1.wrappedValue = LayerNorm(dimensions: dModel)
        self._feedForward1.wrappedValue = ConformerFeedForward(dModel: dModel, dFf: dFf)
        self._normSelfAtt.wrappedValue = LayerNorm(dimensions: dModel)
        self._selfAttn.wrappedValue = RelPositionMultiHeadAttention(config)
        self._normConv.wrappedValue = LayerNorm(dimensions: dModel)
        self._conv.wrappedValue = ConformerConvolution(config)
        self._normFeedForward2.wrappedValue = LayerNorm(dimensions: dModel)
        self._feedForward2.wrappedValue = ConformerFeedForward(dModel: dModel, dFf: dFf)
        self._normOut.wrappedValue = LayerNorm(dimensions: dModel)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var residual = x
        var h = normFeedForward1(x)
        h = feedForward1(h)
        residual = residual + h * fcFactor

        h = normSelfAtt(residual)
        h = selfAttn(query: h, key: h, value: h, mask: mask, posEmb: posEmb)
        residual = residual + h

        h = normConv(residual)
        h = conv(h)
        residual = residual + h

        h = normFeedForward2(residual)
        h = feedForward2(h)
        residual = residual + h * fcFactor

        return normOut(residual)
    }
}

/// FastConformer encoder with conv subsampling and Conformer layers.
private class FastConformerEncoder: Module {
    let scaleInput: Bool
    let hiddenSize: Int

    @ModuleInfo var subsampling: ConvSubsampling
    var layers: [ConformerLayer]
    @ModuleInfo(key: "pos_enc") var posEnc: RelPositionalEncoding

    init(_ config: FCEncoderConfig) {
        scaleInput = config.scaleInput
        hiddenSize = config.hiddenSize

        self._subsampling.wrappedValue = ConvSubsampling(config)
        layers = (0..<config.numHiddenLayers).map { _ in ConformerLayer(config) }
        self._posEnc.wrappedValue = RelPositionalEncoding(dModel: config.hiddenSize)
    }

    /// Run ConvSubsampling only (for streaming pre-encode).
    func preEncode(_ audioSignal: MLXArray, length: MLXArray) -> (MLXArray, MLXArray) {
        subsampling(audioSignal, lengths: length)
    }

    /// Run Conformer layers on pre-encoded embeddings.
    /// - Returns: `(batch, hiddenSize, time)` channels-first
    func encode(_ embeddings: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        var x = embeddings
        if scaleInput {
            x = x * Float(sqrt(Double(hiddenSize)))
        }

        let posEmb = posEnc(x)
        for layer in layers {
            x = layer(x, posEmb: posEmb)
        }

        x = x.transposed(0, 2, 1)
        return (x, lengths)
    }

    /// Full forward: ConvSubsampling + Conformer layers.
    func callAsFunction(_ audioSignal: MLXArray, length: MLXArray) -> (MLXArray, MLXArray) {
        let (x, lengths) = preEncode(audioSignal, length: length)
        return encode(x, lengths: lengths)
    }
}

// MARK: - Transformer Encoder Components (BART-style)

/// Standard multi-head attention for the Transformer encoder.
private class TransformerAttention: Module {
    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: TFEncoderConfig) {
        embedDim = config.dModel
        numHeads = config.encoderAttentionHeads
        headDim = embedDim / numHeads
        scale = pow(Float(headDim), -0.5)

        self._qProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        self._kProj.wrappedValue = Linear(embedDim, embedDim, bias: config.kProjBias)
        self._vProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        self._outProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
    }

    func callAsFunction(
        query: MLXArray, key: MLXArray, value: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let (b, t, _) = (query.dim(0), query.dim(1), query.dim(2))

        let q = qProj(query).reshaped(b, t, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(key).reshaped(b, -1, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(value).reshaped(b, -1, numHeads, headDim).transposed(0, 2, 1, 3)

        var scores = MLX.matmul(q * scale, k.transposed(0, 1, 3, 2))

        if let mask {
            scores = scores + mask
        }

        let attn = softmax(scores, axis: -1)
        let out = MLX.matmul(attn, v).transposed(0, 2, 1, 3).reshaped(b, t, embedDim)
        return outProj(out)
    }
}

/// Single Transformer encoder layer (post-LN, BART-style).
private class TransformerEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: TransformerAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: TFEncoderConfig) {
        self._selfAttn.wrappedValue = TransformerAttention(config)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
        self._fc1.wrappedValue = Linear(config.dModel, config.encoderFfnDim)
        self._fc2.wrappedValue = Linear(config.encoderFfnDim, config.dModel)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: config.layerNormEps)
    }

    /// Post-LN: Attn → Add → LN → FFN(ReLU) → Add → LN
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var residual = x
        var h = selfAttn(query: x, key: x, value: x, mask: mask)
        h = residual + h
        h = selfAttnLayerNorm(h)

        residual = h
        h = relu(fc1(h))
        h = fc2(h)
        h = residual + h
        h = finalLayerNorm(h)

        return h
    }
}

/// Transformer encoder with learned positional embeddings.
private class TransformerEncoder: Module {
    @ModuleInfo(key: "embed_positions") var embedPositions: Embedding
    var layers: [TransformerEncoderLayer]

    init(_ config: TFEncoderConfig) {
        self._embedPositions.wrappedValue = Embedding(embeddingCount: config.maxSourcePositions, dimensions: config.dModel)
        layers = (0..<config.encoderLayers).map { _ in TransformerEncoderLayer(config) }
    }

    func callAsFunction(_ encoderStates: MLXArray, encoderMask: MLXArray? = nil) -> MLXArray {
        let seqLen = encoderStates.dim(1)
        let positions = MLXArray(0..<seqLen)
        var x = encoderStates + embedPositions(positions)

        var attnMask: MLXArray? = nil
        if let encoderMask {
            // Invert mask: True where valid → large negative where invalid
            let inverted = (1 - encoderMask.asType(.float32)) * -1e4
            attnMask = inverted.expandedDimensions(axes: [1, 2])
        }

        for layer in layers {
            x = layer(x, mask: attnMask)
        }

        return x
    }
}

// MARK: - Sortformer Modules

/// Sortformer output modules: projection + feedforward + speaker sigmoid.
private class SortformerModules: Module {
    let nSpk: Int

    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
    @ModuleInfo(key: "first_hidden_to_hidden") var firstHiddenToHidden: Linear
    @ModuleInfo(key: "single_hidden_to_spks") var singleHiddenToSpks: Linear
    @ModuleInfo(key: "hidden_to_spks") var hiddenToSpks: Linear

    init(_ config: ModulesConfig) {
        nSpk = config.numSpeakers

        self._encoderProj.wrappedValue = Linear(config.fcDModel, config.tfDModel)
        self._firstHiddenToHidden.wrappedValue = Linear(config.tfDModel, config.tfDModel)
        self._singleHiddenToSpks.wrappedValue = Linear(config.tfDModel, config.numSpeakers)
        self._hiddenToSpks.wrappedValue = Linear(2 * config.tfDModel, config.numSpeakers)
    }

    func forwardSpeakerSigmoids(_ hiddenOut: MLXArray) -> MLXArray {
        var h = relu(hiddenOut)
        h = firstHiddenToHidden(h)
        h = relu(h)
        let spkPreds = singleHiddenToSpks(h)
        return sigmoid(spkPreds)
    }

    static func lengthToMask(_ lengths: MLXArray, maxLength: Int) -> MLXArray {
        let arange = MLXArray(0..<maxLength)
        return arange.expandedDimensions(axis: 0) .< lengths.expandedDimensions(axis: 1)
    }
}

// MARK: - Main Model

public class SortformerModel: Module {
    public let config: SortformerConfig

    @ModuleInfo(key: "fc_encoder") private var fcEncoder: FastConformerEncoder
    @ModuleInfo(key: "tf_encoder") private var tfEncoder: TransformerEncoder
    @ModuleInfo(key: "sortformer_modules") private var sortformerModules: SortformerModules

    public init(_ config: SortformerConfig) {
        self.config = config
        self._fcEncoder.wrappedValue = FastConformerEncoder(config.fcEncoderConfig)
        self._tfEncoder.wrappedValue = TransformerEncoder(config.tfEncoderConfig)
        self._sortformerModules.wrappedValue = SortformerModules(config.modulesConfig)
    }

    public var modelDtype: DType {
        sortformerModules.encoderProj.weight.dtype
    }

    /// Full forward pass.
    /// - Parameters:
    ///   - audioSignal: `(batch, nMels, time)` mel features
    ///   - audioSignalLength: `(batch,)` feature lengths
    /// - Returns: `(batch, diarFrameCount, numSpeakers)`
    public func callAsFunction(_ audioSignal: MLXArray, audioSignalLength: MLXArray) -> MLXArray {
        let signal = audioSignal.asType(modelDtype)
        var (embSeq, embSeqLength) = fcEncoder(signal, length: audioSignalLength)
        embSeq = embSeq.transposed(0, 2, 1)

        embSeq = sortformerModules.encoderProj(embSeq)

        let encoderMask = SortformerModules.lengthToMask(embSeqLength, maxLength: embSeq.dim(1))
        let transEmbSeq = tfEncoder(embSeq, encoderMask: encoderMask)
        let preds = sortformerModules.forwardSpeakerSigmoids(transEmbSeq)
        return preds * encoderMask.expandedDimensions(axis: 2)
    }

    // MARK: - Offline Inference

    public func generate(
        audio: MLXArray,
        sampleRate: Int = 16000,
        threshold: Float = 0.5,
        minDuration: Float = 0.0,
        mergeGap: Float = 0.0
    ) throws -> DiarizationOutput {
        let startTime = Date()
        let processor = config.processorConfig
        guard sampleRate == processor.samplingRate else {
            throw SortformerDiarizationError.unsupportedSampleRate(
                actual: sampleRate,
                expected: processor.samplingRate
            )
        }
        guard audio.size > 0 else {
            throw SortformerDiarizationError.emptyAudio
        }

        var waveform = audio.asType(.float32)
        if waveform.ndim > 1 {
            waveform = MLX.mean(waveform, axis: -1)
        }

        let (trimmed, trimOffset) = trimSilence(waveform, sampleRate: processor.samplingRate)
        waveform = trimmed
        let trimOffsetSeconds = Float(trimOffset) / Float(processor.samplingRate)
        waveform = (1.0 / (MLX.abs(waveform).max() + 1e-3)) * waveform

        let features = extractMelFeatures(
            waveform,
            sampleRate: processor.samplingRate,
            nFft: processor.nFft,
            hopLength: processor.hopLength,
            winLength: processor.winLength,
            nMels: processor.featureSize,
            preemphasisCoeff: processor.preemphasis
        )
        let featureLengths = MLXArray([Int32(features.dim(2))])
        let predictions = self(features, audioSignalLength: featureLengths)
        eval(predictions)

        let subsamplingFactor = config.fcEncoderConfig.subsamplingFactor
        let frameDuration = Float(processor.hopLength * subsamplingFactor) / Float(processor.samplingRate)
        var segments = Self.predsToSegments(
            predictions[0],
            frameDuration: frameDuration,
            threshold: threshold,
            minDuration: minDuration,
            mergeGap: mergeGap
        )

        if trimOffset > 0 {
            segments = segments.map {
                DiarizationSegment(
                    start: $0.start + trimOffsetSeconds,
                    end: $0.end + trimOffsetSeconds,
                    speaker: $0.speaker
                )
            }
        }

        return DiarizationOutput(
            segments: segments,
            numSpeakers: Set(segments.map(\.speaker)).count,
            totalTime: Date().timeIntervalSince(startTime)
        )
    }
    // MARK: - Postprocessing

    public static func predsToSegments(
        _ preds: MLXArray,
        frameDuration: Float,
        threshold: Float = 0.5,
        minDuration: Float = 0.0,
        mergeGap: Float = 0.0
    ) -> [DiarizationSegment] {
        let numFrames = preds.dim(0)
        let numSpeakers = preds.dim(1)
        var segments = [DiarizationSegment]()

        // Single bulk GPU->CPU readback (row-major [frame, speaker]); all change
        // detection runs in pure Swift to avoid per-frame .item() round-trips.
        let flat = preds.asType(.float32).reshaped([-1]).asArray(Float.self)

        for spk in 0..<numSpeakers {
            var spkSegments = [DiarizationSegment]()
            var segStart = -1
            for f in 0..<numFrames {
                let active = flat[f * numSpeakers + spk] > threshold
                if active {
                    if segStart < 0 { segStart = f }
                } else if segStart >= 0 {
                    let startTime = Float(segStart) * frameDuration
                    let endTime = Float(f) * frameDuration
                    if endTime - startTime >= minDuration {
                        spkSegments.append(DiarizationSegment(start: startTime, end: endTime, speaker: spk))
                    }
                    segStart = -1
                }
            }
            if segStart >= 0 {
                let startTime = Float(segStart) * frameDuration
                let endTime = Float(numFrames) * frameDuration
                if endTime - startTime >= minDuration {
                    spkSegments.append(DiarizationSegment(start: startTime, end: endTime, speaker: spk))
                }
            }

            if mergeGap > 0 && spkSegments.count > 1 {
                var merged = [spkSegments[0]]
                for seg in spkSegments.dropFirst() {
                    if seg.start - merged.last!.end <= mergeGap {
                        merged[merged.count - 1] = DiarizationSegment(
                            start: merged.last!.start, end: seg.end, speaker: seg.speaker
                        )
                    } else {
                        merged.append(seg)
                    }
                }
                spkSegments = merged
            }

            segments.append(contentsOf: spkSegments)
        }

        segments.sort { $0.start < $1.start }
        return segments
    }

    // MARK: - Weight Sanitization & Loading

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        let skipKeys: Set<String> = ["num_batches_tracked"]

        let alreadyConverted = weights.keys.contains { $0.contains("subsampling.layers_") }

        for (k, var v) in weights {
            if skipKeys.contains(where: { k.contains($0) }) { continue }

            var newK = k

            if !alreadyConverted {
                if newK.contains("fc_encoder.subsampling.layers.") {
                    newK = newK.replacingOccurrences(of: "subsampling.layers.", with: "subsampling.layers_")
                }

                // Conv2d: PyTorch (O,I,H,W) → MLX (O,H,W,I)
                if newK.contains("subsampling") && newK.contains("weight") && !newK.contains("linear") {
                    if v.ndim == 4 {
                        v = v.transposed(0, 2, 3, 1)
                    }
                }

                // Conv1d: PyTorch (O,I,K) → MLX (O,K,I)
                if (newK.contains("pointwise_conv1") || newK.contains("pointwise_conv2") || newK.contains("depthwise_conv"))
                    && newK.contains("weight") {
                    if v.ndim == 3 {
                        v = v.transposed(0, 2, 1)
                    }
                }
            }

            sanitized[newK] = v
        }

        return sanitized
    }

    static func runtimeCompatibleWeights(
        _ weights: [String: MLXArray],
        promoteFloat16: Bool
    ) -> [String: MLXArray] {
        guard promoteFloat16 else { return weights }
        return weights.mapValues { weight in
            weight.dtype == .float16 ? weight.asType(.float32) : weight
        }
    }

    public static func fromModelDirectory(_ modelURL: URL) throws -> SortformerModel {
        // Load config
        let configURL = modelURL.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(SortformerConfig.self, from: configData)

        let model = SortformerModel(config)

        // Load weights
        let weightFiles = try FileManager.default.contentsOfDirectory(
            at: modelURL, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }

        var allWeights = [String: MLXArray]()
        for file in weightFiles {
            let weights = try loadArrays(url: file)
            for (k, v) in weights {
                allWeights[k] = v
            }
        }

        let sanitized = sanitize(allWeights)
        #if os(Linux) && arch(x86_64)
        // mlx-swift cannot construct Float16 host scalars on x86_64. Promoting
        // the compact Sortformer checkpoint keeps scalar model operations on
        // the native CUDA path without triggering that host-only initializer.
        let runtimeWeights = runtimeCompatibleWeights(sanitized, promoteFloat16: true)
        #else
        let runtimeWeights = sanitized
        #endif
        try model.update(parameters: ModuleParameters.unflattened(runtimeWeights), verify: .noUnusedKeys)
        eval(model.parameters())

        return model
    }
}
