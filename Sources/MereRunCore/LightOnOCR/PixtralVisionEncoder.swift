import Foundation
import MLX
import MLXNN

// MARK: - Pixtral Vision Encoder Configuration

public struct PixtralVisionConfiguration: Sendable {
    public let hiddenSize: Int       // 1024
    public let numHiddenLayers: Int  // 24
    public let numAttentionHeads: Int // 16
    public let headDim: Int          // 64
    public let intermediateSize: Int // 4096
    public let patchSize: Int        // 14
    public let imageSize: Int        // 1540
    public let numChannels: Int      // 3
    public let ropeTheta: Float      // 10000

    public init(
        hiddenSize: Int = 1024,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        headDim: Int = 64,
        intermediateSize: Int = 4096,
        patchSize: Int = 14,
        imageSize: Int = 1540,
        numChannels: Int = 3,
        ropeTheta: Float = 10000
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.headDim = headDim
        self.intermediateSize = intermediateSize
        self.patchSize = patchSize
        self.imageSize = imageSize
        self.numChannels = numChannels
        self.ropeTheta = ropeTheta
    }
}

// MARK: - Pixtral Patch Conv (direct Conv2d, no wrapper)

// Note: LightOnOCR uses a direct Conv2d for patch_conv, not a nested structure.
// The weight key is "patch_conv.weight" directly on the encoder.

// MARK: - Pixtral 2D RoPE (matches mlx-vlm implementation exactly)

final class PixtralRotaryEmbedding {
    private let invFreq: MLXArray  // Precomputed frequency table [maxPatches, headDim]
    private let maxPatchesPerSide: Int

    init(config: PixtralVisionConfiguration) {
        let dim = config.headDim  // 64
        let base = config.ropeTheta  // 10000
        self.maxPatchesPerSide = config.imageSize / config.patchSize  // 110

        // Compute base frequencies: 1 / (theta^(2i/dim)) for i in [0, dim/2)
        // freqs shape: [dim/2] = [32]
        var freqsArray: [Float] = []
        for i in stride(from: 0, to: dim, by: 2) {
            let freq = 1.0 / pow(base, Float(i) / Float(dim))
            freqsArray.append(freq)
        }
        let freqs = MLXArray(freqsArray)  // [32]

        // Split into even (for height) and odd (for width) indexed frequencies
        // freqs[::2] -> indices 0, 2, 4, ... -> 16 values for height
        // freqs[1::2] -> indices 1, 3, 5, ... -> 16 values for width
        var freqsHArray: [Float] = []
        var freqsWArray: [Float] = []
        for i in stride(from: 0, to: freqsArray.count, by: 2) {
            freqsHArray.append(freqsArray[i])
        }
        for i in stride(from: 1, to: freqsArray.count, by: 2) {
            freqsWArray.append(freqsArray[i])
        }
        let freqsH = MLXArray(freqsHArray)  // [16]
        let freqsW = MLXArray(freqsWArray)  // [16]

        // Create position indices
        let h = MLXArray(Array(0..<maxPatchesPerSide).map { Float($0) })  // [maxPatches]
        let w = MLXArray(Array(0..<maxPatchesPerSide).map { Float($0) })  // [maxPatches]

        // Outer products: position * frequency
        // freqs_h[i, j] = h[i] * freqsH[j]
        let freqsHOuter = h.expandedDimensions(axis: 1) * freqsH  // [maxPatches, 16]
        let freqsWOuter = w.expandedDimensions(axis: 1) * freqsW  // [maxPatches, 16]

        // Tile to create 2D grid:
        // freqsHOuter: [H, 16] -> [H, 1, 16] -> tile [H, W, 16]
        // freqsWOuter: [W, 16] -> [1, W, 16] -> tile [H, W, 16]
        let freqsHTiled = MLX.tiled(
            freqsHOuter.expandedDimensions(axis: 1),
            repetitions: [1, maxPatchesPerSide, 1]
        )  // [H, W, 16]
        let freqsWTiled = MLX.tiled(
            freqsWOuter.expandedDimensions(axis: 0),
            repetitions: [maxPatchesPerSide, 1, 1]
        )  // [H, W, 16]

        // Concatenate H and W frequencies: [H, W, 32]
        let invFreq2D = MLX.concatenated([freqsHTiled, freqsWTiled], axis: -1)

        // Reshape to [H*W, 32] then duplicate to [H*W, 64]
        let invFreqFlat = invFreq2D.reshaped(maxPatchesPerSide * maxPatchesPerSide, dim / 2)
        self.invFreq = MLX.concatenated([invFreqFlat, invFreqFlat], axis: -1)  // [H*W, 64]

    }

    /// Get rotary embeddings for given position IDs
    func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        // positionIds: [seqLen] - indices into invFreq table
        let freqs = invFreq[positionIds]  // [seqLen, headDim]
        let cos = MLX.cos(freqs).asType(x.dtype)
        let sin = MLX.sin(freqs).asType(x.dtype)
        return (cos, sin)
    }

    /// Compute position IDs for a grid of patches
    func positionIds(gridH: Int, gridW: Int) -> MLXArray {
        // position_id = h * max_width + w
        var ids: [Int32] = []
        for h in 0..<gridH {
            for w in 0..<gridW {
                ids.append(Int32(h * maxPatchesPerSide + w))
            }
        }
        return MLXArray(ids)
    }
}

// MARK: - Pixtral Attention (no QK norm for vision encoder)

final class PixtralAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") private var qProj: Linear
    @ModuleInfo(key: "k_proj") private var kProj: Linear
    @ModuleInfo(key: "v_proj") private var vProj: Linear
    @ModuleInfo(key: "o_proj") private var oProj: Linear

    init(config: PixtralVisionConfiguration) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        let hiddenSize = config.hiddenSize
        self._qProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._oProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray,
        rotaryCos: MLXArray,
        rotarySin: MLXArray
    ) -> MLXArray {
        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        // Project Q, K, V
        var q = qProj(x)
        var k = kProj(x)
        let v = vProj(x)

        // Reshape to multi-head: [batch, seq, numHeads, headDim]
        q = q.reshaped(batch, seqLen, numHeads, headDim)
        k = k.reshaped(batch, seqLen, numHeads, headDim)
        var vReshaped = v.reshaped(batch, seqLen, numHeads, headDim)

        // Apply rotary embeddings
        q = applyRotary(q, cos: rotaryCos, sin: rotarySin)
        k = applyRotary(k, cos: rotaryCos, sin: rotarySin)

        // Transpose for attention: [batch, numHeads, seq, headDim]
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        vReshaped = vReshaped.transposed(0, 2, 1, 3)

        // Scaled dot-product attention
        var attn = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
        attn = softmax(attn, axis: -1)
        var out = MLX.matmul(attn, vReshaped)

        // Reshape back: [batch, seq, hidden]
        out = out.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim)

        return oProj(out)
    }

    private func applyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        // x: [batch, seq, heads, headDim]
        // cos, sin: [seq, headDim]
        let cosExpanded = cos.expandedDimensions(axes: [0, 2])  // [1, seq, 1, headDim]
        let sinExpanded = sin.expandedDimensions(axes: [0, 2])

        // Split x into two halves for rotation
        let x1 = x[0..., 0..., 0..., 0..<(headDim / 2)]
        let x2 = x[0..., 0..., 0..., (headDim / 2)...]

        // Rotate: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        let rotated = MLX.concatenated([-x2, x1], axis: -1)

        return x * cosExpanded + rotated * sinExpanded
    }
}

// MARK: - Pixtral MLP

final class PixtralMLP: Module {
    @ModuleInfo(key: "gate_proj") private var gateProj: Linear
    @ModuleInfo(key: "up_proj") private var upProj: Linear
    @ModuleInfo(key: "down_proj") private var downProj: Linear

    init(config: PixtralVisionConfiguration) {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize

        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SiLU gated MLP
        let gate = silu(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}

// MARK: - Pixtral Transformer Block

final class PixtralTransformerBlock: Module {
    @ModuleInfo(key: "attention") private var attention: PixtralAttention
    @ModuleInfo(key: "feed_forward") private var feedForward: PixtralMLP
    @ModuleInfo(key: "attention_norm") private var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") private var ffnNorm: RMSNorm

    init(config: PixtralVisionConfiguration) {
        self._attention.wrappedValue = PixtralAttention(config: config)
        self._feedForward.wrappedValue = PixtralMLP(config: config)
        self._attentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
    }

    func callAsFunction(
        _ x: MLXArray,
        rotaryCos: MLXArray,
        rotarySin: MLXArray
    ) -> MLXArray {
        // Pre-norm attention
        var h = x + attention(attentionNorm(x), rotaryCos: rotaryCos, rotarySin: rotarySin)
        // Pre-norm FFN
        h = h + feedForward(ffnNorm(h))
        return h
    }
}

// MARK: - Pixtral Vision Encoder

public final class PixtralVisionEncoder: Module {
    public let config: PixtralVisionConfiguration

    // Direct Conv2d for patch embedding (no bias, weight key = "patch_conv.weight")
    @ModuleInfo(key: "patch_conv") private var patchConv: Conv2d
    @ModuleInfo(key: "ln_pre") private var lnPre: RMSNorm
    @ModuleInfo(key: "transformer") private var transformer: TransformerLayers

    private let rope: PixtralRotaryEmbedding

    public init(config: PixtralVisionConfiguration) {
        self.config = config
        self.rope = PixtralRotaryEmbedding(config: config)

        // Conv2d patch embedding: [3, H, W] -> [hidden, H/patch, W/patch]
        self._patchConv.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize),
            bias: false
        )
        self._lnPre.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)

        self._transformer.wrappedValue = TransformerLayers(config: config)
    }

    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        // pixelValues: [batch, channels, height, width] (NCHW format from image loading)
        let height = pixelValues.dim(2)
        let width = pixelValues.dim(3)

        // Compute grid size after patching
        let gridH = height / config.patchSize
        let gridW = width / config.patchSize

        // Convert NCHW -> NHWC for MLX Conv2d
        let nhwc = pixelValues.transposed(0, 2, 3, 1)  // [batch, height, width, channels]

        // Patch embedding via Conv2d - output is [batch, gridH, gridW, hidden]
        let out = patchConv(nhwc)
        let b = out.dim(0)
        let h = out.dim(1)
        let w = out.dim(2)
        let c = out.dim(3)

        // Flatten spatial dims: [batch, h*w, hidden]
        var hiddenStates = out.reshaped(b, h * w, c)

        // Pre-norm
        hiddenStates = lnPre(hiddenStates)

        // Compute position IDs and rotary embeddings
        let positionIds = rope.positionIds(gridH: gridH, gridW: gridW)
        let (rotaryCos, rotarySin) = rope(hiddenStates, positionIds: positionIds)

        // Process through transformer blocks
        hiddenStates = transformer(hiddenStates, rotaryCos: rotaryCos, rotarySin: rotarySin)

        return hiddenStates
    }

    /// Encode an image and return vision features
    public func encode(pixelValues: MLXArray) -> MLXArray {
        return callAsFunction(pixelValues)
    }
}

// MARK: - Transformer Layers Wrapper (for nested "layers" key)

final class TransformerLayers: Module {
    @ModuleInfo(key: "layers") private var layers: [PixtralTransformerBlock]

    init(config: PixtralVisionConfiguration) {
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            PixtralTransformerBlock(config: config)
        }
    }

    func callAsFunction(_ x: MLXArray, rotaryCos: MLXArray, rotarySin: MLXArray) -> MLXArray {
        var hiddenStates = x
        for block in layers {
            hiddenStates = block(hiddenStates, rotaryCos: rotaryCos, rotarySin: rotarySin)
        }
        return hiddenStates
    }
}
