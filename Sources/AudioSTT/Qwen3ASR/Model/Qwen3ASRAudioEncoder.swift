import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Audio Tower (Conv2d + Transformer Encoder)

// MARK: - Sinusoidal Position Embedding

private final class SinusoidalPositionEmbedding {
    private let embedding: MLXArray

    init(length: Int, channels: Int, maxTimescale: Float = 10000.0) {
        precondition(channels % 2 == 0, "SinusoidalPositionEmbedding requires even channels.")
        let half = channels / 2
        let logTimescaleIncrement = log(maxTimescale) / Float(max(1, half - 1))
        let invTimescales = MLX.exp(
            MLXArray(Array(0..<half).map { Float($0) * -logTimescaleIncrement })
        )
        let positions = MLXArray(Array(0..<length).map { Float($0) }).reshaped(length, 1)
        let scaledTime = positions * invTimescales.reshaped(1, half)
        let sinVals = MLX.sin(scaledTime)
        let cosVals = MLX.cos(scaledTime)
        self.embedding = MLX.concatenated([sinVals, cosVals], axis: 1)
    }

    func callAsFunction(_ seqlen: Int) -> MLXArray {
        embedding[0..<seqlen, 0...]
    }
}

/// Audio encoder with Conv2d frontend and transformer layers
/// Matches the Qwen3-ASR architecture with:
/// - 3x Conv2d layers (downsample_hidden_size=480)
/// - Linear projection to d_model=896
/// - 18 transformer encoder layers
/// - Output projection to 1024
public final class Qwen3ASRAudioTower: Module {
    let config: Qwen3ASRAudioEncoderConfig

    // Conv2d frontend
    @ModuleInfo(key: "conv2d1") var conv2d1: Conv2d
    @ModuleInfo(key: "conv2d2") var conv2d2: Conv2d
    @ModuleInfo(key: "conv2d3") var conv2d3: Conv2d
    @ModuleInfo(key: "conv_out") var convOut: Linear
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    private let positionalEmbedding: SinusoidalPositionEmbedding

    // Transformer encoder
    @ModuleInfo(key: "layers") var layers: [Qwen3ASREncoderLayer]

    // Output projection: 896 -> 1024
    @ModuleInfo(key: "proj1") var proj1: Linear
    @ModuleInfo(key: "proj2") var proj2: Linear

    public init(config: Qwen3ASRAudioEncoderConfig) {
        self.config = config

        let hiddenSize = config.downsampleHiddenSize  // 480

        // Conv2d frontend: [B, 1, 128, T] -> [B, 480, 16, T/8]
        // Each conv downsamples mel + time by 2.
        self._conv2d1.wrappedValue = Conv2d(
            inputChannels: 1,
            outputChannels: hiddenSize,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(1)
        )
        self._conv2d2.wrappedValue = Conv2d(
            inputChannels: hiddenSize,
            outputChannels: hiddenSize,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(1)
        )
        self._conv2d3.wrappedValue = Conv2d(
            inputChannels: hiddenSize,
            outputChannels: hiddenSize,
            kernelSize: IntOrPair(3),
            stride: IntOrPair(2),
            padding: IntOrPair(1)
        )

        // Project flattened conv output to d_model
        // After 3 convs: [B, 480, 16, T] -> flatten -> [B, T, 480*16=7680] -> [B, T, 896]
        self._convOut.wrappedValue = Linear(hiddenSize * 16, config.dModel, bias: false)
        self.positionalEmbedding = SinusoidalPositionEmbedding(
            length: config.maxSourcePositions,
            channels: config.dModel
        )

        // Transformer encoder layers
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            Qwen3ASREncoderLayer(config: config)
        }

        self._lnPost.wrappedValue = LayerNorm(
            dimensions: config.dModel,
            eps: config.layerNormEps
        )

        // Output projection: d_model (896) -> output_dim (1024)
        self._proj1.wrappedValue = Linear(config.dModel, config.dModel)
        self._proj2.wrappedValue = Linear(config.dModel, config.outputDim)
    }

    /// Forward pass
    /// - Parameter melSpec: Mel spectrogram [B, nMels, T] (e.g., [1, 128, T])
    /// - Returns: Encoded features [B, T, outputDim] (e.g., [1, T, 1024])
    public func callAsFunction(_ melSpec: MLXArray) -> MLXArray {
        // Input: [B, nMels=128, T]
        let batch = melSpec.dim(0)
        let melBins = melSpec.dim(1)
        let totalFrames = melSpec.dim(2)

        let chunkSize = max(1, config.nWindow * 2)
        let outputPerChunk = convOutputLength(chunkSize)

        // Feature lengths (no padding in our pipeline)
        let featureLens = Array(repeating: totalFrames, count: batch)
        let afterCnnLens = featureLens.map { getFeatExtractOutputLength($0, chunkSize: chunkSize, outputPerChunk: outputPerChunk) }

        // Split into chunks
        var chunks: [MLXArray] = []
        var chunkLengths: [Int] = []
        chunks.reserveCapacity(batch)
        chunkLengths.reserveCapacity(batch)

        for b in 0..<batch {
            let featLen = featureLens[b]
            let numChunks = Int(ceil(Double(featLen) / Double(chunkSize)))
            var pos = 0
            for j in 0..<numChunks {
                let remaining = featLen - pos
                let clen = min(chunkSize, max(0, remaining))
                guard clen > 0 else { continue }
                let chunk = melSpec[b, 0..., pos..<(pos + clen)]
                chunks.append(chunk)
                chunkLengths.append(clen)
                pos += clen
            }
        }

        guard !chunks.isEmpty else {
            return MLX.zeros([1, 0, config.outputDim], dtype: melSpec.dtype)
        }

        let maxChunkLen = chunkLengths.max() ?? 0
        var paddedChunks: [MLXArray] = []
        paddedChunks.reserveCapacity(chunks.count)

        for (idx, chunk) in chunks.enumerated() {
            let clen = chunkLengths[idx]
            if clen < maxChunkLen {
                let padWidth = maxChunkLen - clen
                let pad = MLX.zeros([melBins, padWidth], dtype: chunk.dtype)
                paddedChunks.append(MLX.concatenated([chunk, pad], axis: 1))
            } else {
                paddedChunks.append(chunk)
            }
        }

        var x = MLX.stacked(paddedChunks, axis: 0)
        // MLX Conv2d expects [B, H, W, C] (channels last)
        x = x.expandedDimensions(axis: 3)  // [B, nMels, T, 1]

        // Conv2d frontend with GELU activation
        x = gelu(conv2d1(x))
        x = gelu(conv2d2(x))
        x = gelu(conv2d3(x))

        // Reshape: [B, F, T, C] -> [B, T, C*F]
        let B = x.dim(0)
        let F = x.dim(1)
        let T = x.dim(2)
        let C = x.dim(3)
        x = x.transposed(0, 2, 3, 1).reshaped(B, T, C * F)

        // Project to d_model
        x = convOut(x)  // [B, T, 896]

        // Add sinusoidal positional embeddings
        let posEmb = positionalEmbedding(T)
        x = x + posEmb.reshaped(1, T, posEmb.dim(1))

        // Trim padding per chunk
        let chunkOutLens = chunkLengths.map { convOutputLength($0) }
        let maxLenAfterCnn = chunkOutLens.max() ?? 0

        var hiddenList: [MLXArray] = []
        hiddenList.reserveCapacity(chunkOutLens.count)
        for i in 0..<B {
            let validLen = chunkOutLens[i]
            if validLen > 0 {
                hiddenList.append(x[i, 0..<validLen, 0...])
            }
        }

        var hiddenStates = MLX.concatenated(hiddenList, axis: 0)
        let seqLen = hiddenStates.dim(0)
        if seqLen == 0 {
            return MLX.zeros([1, 0, config.outputDim], dtype: hiddenStates.dtype)
        }

        // Build block attention mask
        let ratio = max(1, config.nWindowInfer / max(1, config.nWindow * 2))
        let windowAfterCnn = maxLenAfterCnn * ratio

        var cuChunkLens: [Int] = [0]
        for len in afterCnnLens {
            if windowAfterCnn > 0 {
                let numFull = len / windowAfterCnn
                if numFull > 0 {
                    cuChunkLens.append(contentsOf: Array(repeating: windowAfterCnn, count: numFull))
                }
                let remainder = len % windowAfterCnn
                if remainder != 0 {
                    cuChunkLens.append(remainder)
                }
            } else {
                cuChunkLens.append(len)
            }
        }

        var cuSeqlens: [Int] = []
        cuSeqlens.reserveCapacity(cuChunkLens.count)
        var running = 0
        for len in cuChunkLens {
            running += len
            cuSeqlens.append(running)
        }
        if cuSeqlens.last != seqLen {
            cuSeqlens = cuSeqlens.filter { $0 <= seqLen }
            if cuSeqlens.last != seqLen {
                cuSeqlens.append(seqLen)
            }
        }

        let mask = buildBlockAttentionMask(seqLen: seqLen, cuSeqlens: cuSeqlens, dtype: hiddenStates.dtype)
        let maskMode = MLXFast.ScaledDotProductAttentionMaskMode.array(mask)

        hiddenStates = hiddenStates.reshaped(1, seqLen, config.dModel)

        // Apply transformer encoder layers
        for layer in layers {
            hiddenStates = layer(hiddenStates, mask: maskMode)
        }

        hiddenStates = lnPost(hiddenStates)

        // Project to output dimension: 896 -> 1024
        hiddenStates = gelu(proj1(hiddenStates))
        hiddenStates = proj2(hiddenStates)

        return hiddenStates  // [1, T, 1024]
    }

    private func convOutputLength(_ input: Int) -> Int {
        guard input > 0 else { return 0 }
        func convStep(_ length: Int) -> Int {
            (length - 1) / 2 + 1
        }
        let l1 = convStep(input)
        let l2 = convStep(l1)
        let l3 = convStep(l2)
        return l3
    }

    private func getFeatExtractOutputLength(_ input: Int, chunkSize: Int, outputPerChunk: Int) -> Int {
        guard input > 0 else { return 0 }
        let fullChunks = input / chunkSize
        let remainder = input % chunkSize
        return fullChunks * outputPerChunk + convOutputLength(remainder)
    }

    private func buildBlockAttentionMask(
        seqLen: Int,
        cuSeqlens: [Int],
        dtype: DType
    ) -> MLXArray {
        var data = [Float](repeating: -1.0e9, count: seqLen * seqLen)
        let count = max(0, cuSeqlens.count - 1)
        for i in 0..<count {
            let start = cuSeqlens[i]
            let end = cuSeqlens[i + 1]
            guard end > start else { continue }
            for r in start..<end {
                let rowBase = r * seqLen
                for c in start..<end {
                    data[rowBase + c] = 0.0
                }
            }
        }
        return MLXArray(data).reshaped(1, 1, seqLen, seqLen).asType(dtype)
    }
}

// MARK: - Encoder Layer

final class Qwen3ASREncoderLayer: Module {
    let config: Qwen3ASRAudioEncoderConfig

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3ASREncoderAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(config: Qwen3ASRAudioEncoderConfig) {
        self.config = config

        self._selfAttn.wrappedValue = Qwen3ASREncoderAttention(config: config)
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.dModel,
            eps: config.layerNormEps
        )
        self._fc1.wrappedValue = Linear(config.dModel, config.ffnDim)
        self._fc2.wrappedValue = Linear(config.ffnDim, config.dModel)
        self._finalLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.dModel,
            eps: config.layerNormEps
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        // Pre-norm self-attention
        var residual = x
        var hidden = selfAttnLayerNorm(x)
        hidden = selfAttn(hidden, mask: mask)
        hidden = residual + hidden

        // Pre-norm FFN
        residual = hidden
        hidden = finalLayerNorm(hidden)
        hidden = gelu(fc1(hidden))
        hidden = fc2(hidden)
        hidden = residual + hidden

        return hidden
    }
}

// MARK: - Encoder Attention

final class Qwen3ASREncoderAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(config: Qwen3ASRAudioEncoderConfig) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.dModel / config.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)

        self._qProj.wrappedValue = Linear(config.dModel, config.dModel)
        self._kProj.wrappedValue = Linear(config.dModel, config.dModel)
        self._vProj.wrappedValue = Linear(config.dModel, config.dModel)
        self._outProj.wrappedValue = Linear(config.dModel, config.dModel)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // Reshape to [B, numHeads, L, headDim]
        queries = queries.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)

        // Bidirectional attention (no causal mask)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )

        // Reshape back
        let out = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return outProj(out)
    }
}
