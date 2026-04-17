import Foundation
import MLX
import MLXNN

// MARK: - DINOv3 Vision Encoder
//
// Architecture based on safetensors inspection:
// - embeddings.cls_token, mask_token, register_tokens (4)
// - embeddings.patch_embeddings (Conv2d with bias)
// - layer.N (40 transformer layers with RoPE)
// - Hidden size: 4096, Heads: 32, Patch size: 16

// MARK: - DINOv3 2D Rotary Embeddings

final class DINOv3RotaryEmbedding {
    private let headDim: Int
    private let base: Float
    private let invFreq: [Float]

    init(headDim: Int, base: Float = 100.0) {
        self.headDim = headDim
        self.base = base

        // Matches transformers DINOv3ViTRopePositionEmbedding:
        //   inv_freq = 1 / base ** arange(0, 1, 4/head_dim)
        let count = max(1, headDim / 4)
        let step = 4.0 / Float(headDim)
        var inv: [Float] = []
        inv.reserveCapacity(count)
        for i in 0..<count {
            let exponent = Float(i) * step
            inv.append(1.0 / pow(base, exponent))
        }
        self.invFreq = inv
    }

    /// Compute rotary embeddings for a 2D grid of positions
    func callAsFunction(gridH: Int, gridW: Int, dtype: DType = .bfloat16) -> (cos: MLXArray, sin: MLXArray) {
        // Matches transformers:
        // - patch centers normalized to [-1, +1]
        // - angles = 2π * coords[:,:,None] * inv_freq[None,None,:]
        // - flatten -> tile(2) -> cos/sin
        guard gridH > 0, gridW > 0 else {
            return (cos: MLX.zeros([0, headDim], dtype: dtype), sin: MLX.zeros([0, headDim], dtype: dtype))
        }

        let twoPi = Float(2.0 * Double.pi)
        let seq = gridH * gridW

        // Precompute normalized patch-center coordinates in [-1, 1]
        var coordsH: [Float] = []
        coordsH.reserveCapacity(gridH)
        for h in 0..<gridH {
            let v = (Float(h) + 0.5) / Float(gridH)
            coordsH.append(2.0 * v - 1.0)
        }
        var coordsW: [Float] = []
        coordsW.reserveCapacity(gridW)
        for w in 0..<gridW {
            let v = (Float(w) + 0.5) / Float(gridW)
            coordsW.append(2.0 * v - 1.0)
        }

        // Build angles (float32) with the same ordering as torch meshgrid(indexing="ij") then flatten:
        // (y, x) with x varying fastest.
        var angles: [Float] = []
        angles.reserveCapacity(seq * headDim)
        for h in 0..<gridH {
            let y = coordsH[h]
            for w in 0..<gridW {
                let x = coordsW[w]

                // First half: [y * invFreq..., x * invFreq...]
                for f in invFreq {
                    angles.append(twoPi * y * f)
                }
                for f in invFreq {
                    angles.append(twoPi * x * f)
                }
                // Tile(2): repeat the half-angles to reach headDim.
                for f in invFreq {
                    angles.append(twoPi * y * f)
                }
                for f in invFreq {
                    angles.append(twoPi * x * f)
                }
            }
        }

        // Defensive: handle unexpected dims (e.g. non-multiple-of-4 head dims).
        let expected = seq * headDim
        if angles.count != expected {
            return (cos: MLX.zeros([seq, headDim], dtype: dtype), sin: MLX.zeros([seq, headDim], dtype: dtype))
        }

        let fullAngles = MLXArray(angles).reshaped(seq, headDim).asType(.float32)
        let cos = MLX.cos(fullAngles).asType(dtype)
        let sin = MLX.sin(fullAngles).asType(dtype)
        return (cos, sin)
    }
}

// MARK: - DINOv3 Attention

final class DINOv3Attention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") private var qProj: Linear
    @ModuleInfo(key: "k_proj") private var kProj: Linear
    @ModuleInfo(key: "v_proj") private var vProj: Linear
    @ModuleInfo(key: "o_proj") private var oProj: Linear

    init(config: DINOv3Config) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        let hiddenSize = config.hiddenSize
        self._qProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._oProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
    }

    func callAsFunction(
        _ x: MLXArray,
        rotaryCos: MLXArray? = nil,
        rotarySin: MLXArray? = nil,
        numPrefixTokens: Int = 0
    ) -> MLXArray {
        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        // Project Q, K, V
        var q = qProj(x)
        var k = kProj(x)
        var v = vProj(x)

        // Reshape to multi-head: [batch, seq, numHeads, headDim]
        q = q.reshaped(batch, seqLen, numHeads, headDim)
        k = k.reshaped(batch, seqLen, numHeads, headDim)
        v = v.reshaped(batch, seqLen, numHeads, headDim)

        // Apply rotary embeddings to patch tokens only (skip cls/register tokens)
        if let cos = rotaryCos, let sin = rotarySin, numPrefixTokens < seqLen {
            let qPatches = q[0..., numPrefixTokens..., 0..., 0...]
            let kPatches = k[0..., numPrefixTokens..., 0..., 0...]

            let qRotated = applyRotary(qPatches, cos: cos, sin: sin)
            let kRotated = applyRotary(kPatches, cos: cos, sin: sin)

            // Concatenate back with prefix tokens
            if numPrefixTokens > 0 {
                let qPrefix = q[0..., 0..<numPrefixTokens, 0..., 0...]
                let kPrefix = k[0..., 0..<numPrefixTokens, 0..., 0...]
                q = MLX.concatenated([qPrefix, qRotated], axis: 1)
                k = MLX.concatenated([kPrefix, kRotated], axis: 1)
            } else {
                q = qRotated
                k = kRotated
            }
        }

        // Transpose for attention: [batch, numHeads, seq, headDim]
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        // Scaled dot-product attention
        var attn = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
        attn = softmax(attn, axis: -1)
        var out = MLX.matmul(attn, v)

        // Reshape back: [batch, seq, hidden]
        out = out.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim)

        return oProj(out)
    }

    private func applyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        // x: [batch, seq, heads, headDim]
        // cos, sin: [seq, headDim]
        let cosExpanded = cos.expandedDimensions(axes: [0, 2])
        let sinExpanded = sin.expandedDimensions(axes: [0, 2])

        // Split x into two halves for rotation
        let x1 = x[0..., 0..., 0..., 0..<(headDim / 2)]
        let x2 = x[0..., 0..., 0..., (headDim / 2)...]

        // Rotate
        let rotated = MLX.concatenated([-x2, x1], axis: -1)

        return x * cosExpanded + rotated * sinExpanded
    }
}

// MARK: - DINOv3 MLP (SwiGLU)

final class DINOv3MLP: Module {
    @ModuleInfo(key: "gate_proj") private var gateProj: Linear
    @ModuleInfo(key: "up_proj") private var upProj: Linear
    @ModuleInfo(key: "down_proj") private var downProj: Linear

    init(config: DINOv3Config) {
        // SwiGLU: gate and up project to intermediate, down projects back
        // intermediate is 8192 for 7B model (2x hidden)
        let intermediateSize = config.hiddenSize * 2  // 4096 * 2 = 8192
        self._gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: true)
        self._upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: true)
        self._downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SwiGLU activation: silu(gate) * up, then down
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - DINOv3 Layer Scale

final class DINOv3LayerScale: Module {
    @ModuleInfo(key: "lambda1") private var lambda1: MLXArray

    init(hiddenSize: Int) {
        self._lambda1.wrappedValue = MLX.ones([hiddenSize])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x * lambda1
    }
}

// MARK: - DINOv3 Layer

final class DINOv3Layer: Module {
    @ModuleInfo(key: "attention") private var attention: DINOv3Attention
    @ModuleInfo(key: "mlp") private var mlp: DINOv3MLP
    @ModuleInfo(key: "norm1") private var norm1: LayerNorm
    @ModuleInfo(key: "norm2") private var norm2: LayerNorm
    @ModuleInfo(key: "layer_scale1") private var layerScale1: DINOv3LayerScale
    @ModuleInfo(key: "layer_scale2") private var layerScale2: DINOv3LayerScale

    init(config: DINOv3Config) {
        self._attention.wrappedValue = DINOv3Attention(config: config)
        self._mlp.wrappedValue = DINOv3MLP(config: config)
        self._norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._layerScale1.wrappedValue = DINOv3LayerScale(hiddenSize: config.hiddenSize)
        self._layerScale2.wrappedValue = DINOv3LayerScale(hiddenSize: config.hiddenSize)
    }

    func callAsFunction(
        _ x: MLXArray,
        rotaryCos: MLXArray?,
        rotarySin: MLXArray?,
        numPrefixTokens: Int
    ) -> MLXArray {
        // Pre-norm style with residual and layer scaling
        var h = x + layerScale1(attention(norm1(x), rotaryCos: rotaryCos, rotarySin: rotarySin, numPrefixTokens: numPrefixTokens))
        h = h + layerScale2(mlp(norm2(h)))
        return h
    }
}

// MARK: - DINOv3 Embeddings

final class DINOv3Embeddings: Module {
    @ModuleInfo(key: "cls_token") private var clsToken: MLXArray
    @ModuleInfo(key: "mask_token") private var maskToken: MLXArray
    @ModuleInfo(key: "register_tokens") private var registerTokens: MLXArray
    @ModuleInfo(key: "patch_embeddings") private var patchEmbeddings: Conv2d

    let patchSize: Int
    let numRegisterTokens: Int

    init(config: DINOv3Config) {
        self.patchSize = config.patchSize
        self.numRegisterTokens = config.numRegisterTokens

        // Special tokens
        self._clsToken.wrappedValue = MLX.zeros([1, 1, config.hiddenSize])
        self._maskToken.wrappedValue = MLX.zeros([1, 1, config.hiddenSize])
        self._registerTokens.wrappedValue = MLX.zeros([1, config.numRegisterTokens, config.hiddenSize])

        // Patch embedding via Conv2d
        self._patchEmbeddings.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize),
            bias: true
        )
    }

    func callAsFunction(_ pixelValues: MLXArray) -> (embeddings: MLXArray, numPrefixTokens: Int, gridH: Int, gridW: Int) {
        // pixelValues: [batch, channels, height, width] (NCHW)
        let batch = pixelValues.dim(0)

        // Convert to NHWC for MLX Conv2d
        let nhwc = pixelValues.transposed(0, 2, 3, 1)

        // Patch embedding
        let patchEmbeds = patchEmbeddings(nhwc)  // [batch, gridH, gridW, hidden]
        let gridH = patchEmbeds.dim(1)
        let gridW = patchEmbeds.dim(2)
        let hidden = patchEmbeds.dim(3)

        // Flatten spatial: [batch, numPatches, hidden]
        let patches = patchEmbeds.reshaped(batch, gridH * gridW, hidden)

        // Expand special tokens for batch
        let clsExpanded = MLX.broadcast(clsToken, to: [batch, 1, hidden])
        let regExpanded = MLX.broadcast(registerTokens, to: [batch, numRegisterTokens, hidden])

        // Concatenate: [cls, registers, patches]
        let embeddings = MLX.concatenated([clsExpanded, regExpanded, patches], axis: 1)
        let numPrefixTokens = 1 + numRegisterTokens  // cls + registers

        return (embeddings, numPrefixTokens, gridH, gridW)
    }
}

// MARK: - DINOv3 Vision Model

public final class DINOv3VisionModel: Module {
    public let config: DINOv3Config

    @ModuleInfo(key: "embeddings") private var embeddings: DINOv3Embeddings
    @ModuleInfo(key: "layer") private var layers: [DINOv3Layer]
    @ModuleInfo(key: "norm") private var norm: LayerNorm

    private let rope: DINOv3RotaryEmbedding

    public init(config: DINOv3Config) {
        self.config = config
        self.rope = DINOv3RotaryEmbedding(headDim: config.headDim, base: config.ropeTheta)

        self._embeddings.wrappedValue = DINOv3Embeddings(config: config)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            DINOv3Layer(config: config)
        }
        self._norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    /// Forward pass returning CLS token and patch embeddings
    public func callAsFunction(_ pixelValues: MLXArray) -> (cls: MLXArray, patches: MLXArray) {
        // Embed patches and prepend special tokens
        let (embeds, numPrefixTokens, gridH, gridW) = embeddings(pixelValues)

        // Compute rotary embeddings for patch positions
        let (rotaryCos, rotarySin) = rope(gridH: gridH, gridW: gridW, dtype: embeds.dtype)

        // Process through transformer layers
        var hiddenStates = embeds
        for layer in layers {
            hiddenStates = layer(
                hiddenStates,
                rotaryCos: rotaryCos,
                rotarySin: rotarySin,
                numPrefixTokens: numPrefixTokens
            )
        }

        // Final layer norm
        hiddenStates = norm(hiddenStates)

        // Extract CLS token and patches
        let cls = hiddenStates[0..., 0, 0...]  // [batch, hidden]
        let patches = hiddenStates[0..., numPrefixTokens..., 0...]  // [batch, numPatches, hidden]

        return (cls: cls, patches: patches)
    }

    /// Encode images and return patch embeddings for i2l (averaged across patches)
    public func encode(_ pixelValues: MLXArray) -> MLXArray {
        let (cls, _) = callAsFunction(pixelValues)
        // Return CLS token (or could average patches)
        return cls
    }
}
