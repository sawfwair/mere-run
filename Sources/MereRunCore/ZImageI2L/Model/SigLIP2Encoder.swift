import Foundation
import MLX
import MLXNN

// MARK: - SigLIP2 Vision Encoder
//
// Architecture based on safetensors inspection:
// - embeddings.patch_embedding (Conv2d with bias)
// - embeddings.position_embedding (learnable)
// - encoder.layers.N (40 transformer layers)
// - head (attention pooling with probe)
// - post_layernorm
//
// Hidden size: 1536, Patch size: 16, Heads: 16

// MARK: - SigLIP2 Attention

final class SigLIP2Attention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") private var qProj: Linear
    @ModuleInfo(key: "k_proj") private var kProj: Linear
    @ModuleInfo(key: "v_proj") private var vProj: Linear
    @ModuleInfo(key: "out_proj") private var outProj: Linear

    init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = 1.0 / sqrt(Float(headDim))

        self._qProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._kProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._vProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        self._outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        // Project Q, K, V
        var q = qProj(x)
        var k = kProj(x)
        var v = vProj(x)

        // Reshape to multi-head: [batch, seq, numHeads, headDim]
        q = q.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

        // Scaled dot-product attention
        var attn = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale

        if let mask = mask {
            attn = attn + mask
        }

        attn = softmax(attn, axis: -1)
        var out = MLX.matmul(attn, v)

        // Reshape back: [batch, seq, hidden]
        out = out.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim)

        return outProj(out)
    }
}

// MARK: - SigLIP2 MLP

final class SigLIP2MLP: Module {
    @ModuleInfo(key: "fc1") private var fc1: Linear
    @ModuleInfo(key: "fc2") private var fc2: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._fc1.wrappedValue = Linear(hiddenSize, intermediateSize, bias: true)
        self._fc2.wrappedValue = Linear(intermediateSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // GELU tanh approximation (matching SigLIP2's gelu_pytorch_tanh)
        return fc2(geluApproximate(fc1(x)))
    }
}

// MARK: - SigLIP2 Encoder Layer

final class SigLIP2EncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SigLIP2Attention
    @ModuleInfo(key: "mlp") private var mlp: SigLIP2MLP
    @ModuleInfo(key: "layer_norm1") private var layerNorm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") private var layerNorm2: LayerNorm

    init(config: SigLIP2Config) {
        self._selfAttn.wrappedValue = SigLIP2Attention(
            hiddenSize: config.hiddenSize,
            numHeads: config.numAttentionHeads
        )
        self._mlp.wrappedValue = SigLIP2MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize
        )
        self._layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
        self._layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Pre-norm style
        var h = x + selfAttn(layerNorm1(x))
        h = h + mlp(layerNorm2(h))
        return h
    }
}

// MARK: - SigLIP2 Encoder

final class SigLIP2Encoder: Module {
    @ModuleInfo(key: "layers") private var layers: [SigLIP2EncoderLayer]

    init(config: SigLIP2Config) {
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            SigLIP2EncoderLayer(config: config)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hiddenStates = x
        for layer in layers {
            hiddenStates = layer(hiddenStates)
        }
        return hiddenStates
    }
}

// MARK: - SigLIP2 Embeddings

final class SigLIP2Embeddings: Module {
    @ModuleInfo(key: "patch_embedding") private var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") private var positionEmbedding: Embedding

    let patchSize: Int
    let numPatches: Int

    init(config: SigLIP2Config) {
        self.patchSize = config.patchSize
        self.numPatches = config.numPatches

        // Conv2d for patch embedding: [3, H, W] -> [hidden, H/patch, W/patch]
        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize),
            bias: true
        )

        // Learnable position embeddings
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.numPatches,
            dimensions: config.hiddenSize
        )
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        // pixelValues: [batch, channels, height, width] (NCHW)
        // Convert to NHWC for MLX Conv2d
        let nhwc = pixelValues.transposed(0, 2, 3, 1)

        // Patch embedding
        let patchEmbeds = patchEmbedding(nhwc)  // [batch, gridH, gridW, hidden]
        let batch = patchEmbeds.dim(0)
        let gridH = patchEmbeds.dim(1)
        let gridW = patchEmbeds.dim(2)
        let hidden = patchEmbeds.dim(3)

        // Flatten spatial: [batch, numPatches, hidden]
        var embeddings = patchEmbeds.reshaped(batch, gridH * gridW, hidden)

        // Add position embeddings
        let posIds = MLXArray(Array(0..<(gridH * gridW)).map { Int32($0) })
        embeddings = embeddings + positionEmbedding(posIds)

        return embeddings
    }
}

// MARK: - SigLIP2 Attention Pooling Head

final class SigLIP2Head: Module {
    @ModuleInfo(key: "attention") private var attention: MultiHeadAttentionPooling
    @ModuleInfo(key: "layernorm") private var layernorm: LayerNorm
    @ModuleInfo(key: "mlp") private var mlp: SigLIP2MLP
    @ModuleInfo(key: "probe") private var probe: MLXArray  // Learnable probe token

    init(config: SigLIP2Config) {
        self._attention.wrappedValue = MultiHeadAttentionPooling(hiddenSize: config.hiddenSize)
        self._layernorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._mlp.wrappedValue = SigLIP2MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        self._probe.wrappedValue = MLX.zeros([1, 1, config.hiddenSize])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        // Expand probe for batch: [1, 1, hidden] -> [batch, 1, hidden]
        let probeExpanded = MLX.broadcast(probe, to: [batch, 1, x.dim(2)])

        // Attention pooling: probe attends to all patches
        var hiddenState = attention(probeExpanded, x)

        // Residual connection around layernorm + mlp (matching the reference implementation)
        let residual = hiddenState
        hiddenState = layernorm(hiddenState)
        hiddenState = residual + mlp(hiddenState)

        return hiddenState.squeezed(axis: 1)  // [batch, hidden]
    }
}

// MARK: - Multi-Head Attention Pooling (simplified)

final class MultiHeadAttentionPooling: Module {
    @ModuleInfo(key: "in_proj_weight") private var inProjWeight: MLXArray
    @ModuleInfo(key: "in_proj_bias") private var inProjBias: MLXArray
    @ModuleInfo(key: "out_proj") private var outProj: Linear

    let hiddenSize: Int
    let numHeads: Int = 16
    var headDim: Int { hiddenSize / numHeads }

    init(hiddenSize: Int) {
        self.hiddenSize = hiddenSize
        // Combined QKV projection
        self._inProjWeight.wrappedValue = MLX.zeros([hiddenSize * 3, hiddenSize])
        self._inProjBias.wrappedValue = MLX.zeros([hiddenSize * 3])
        self._outProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ query: MLXArray, _ key: MLXArray) -> MLXArray {
        // query: [batch, 1, hidden] (probe)
        // key: [batch, seq, hidden] (patches)
        let batch = query.dim(0)
        let seqLen = key.dim(1)

        // Project query from probe
        let qkv = MLX.matmul(query, inProjWeight.T) + inProjBias
        let q = qkv[0..., 0..., 0..<hiddenSize]

        // Project key and value from patches
        let kvProj = MLX.matmul(key, inProjWeight.T) + inProjBias
        let k = kvProj[0..., 0..., hiddenSize..<(hiddenSize * 2)]
        let v = kvProj[0..., 0..., (hiddenSize * 2)...]

        // Reshape for multi-head attention
        let qReshaped = q.reshaped(batch, 1, numHeads, headDim).transposed(0, 2, 1, 3)
        let kReshaped = k.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)
        let vReshaped = v.reshaped(batch, seqLen, numHeads, headDim).transposed(0, 2, 1, 3)

        // Attention
        let scale = 1.0 / sqrt(Float(headDim))
        var attn = MLX.matmul(qReshaped, kReshaped.transposed(0, 1, 3, 2)) * scale
        attn = softmax(attn, axis: -1)
        var out = MLX.matmul(attn, vReshaped)

        // Reshape back
        out = out.transposed(0, 2, 1, 3).reshaped(batch, 1, hiddenSize)

        return outProj(out)
    }
}

// MARK: - SigLIP2 Vision Model

public final class SigLIP2VisionModel: Module {
    public let config: SigLIP2Config

    @ModuleInfo(key: "embeddings") private var embeddings: SigLIP2Embeddings
    @ModuleInfo(key: "encoder") private var encoder: SigLIP2Encoder
    @ModuleInfo(key: "post_layernorm") private var postLayernorm: LayerNorm
    @ModuleInfo(key: "head") private var head: SigLIP2Head

    public init(config: SigLIP2Config) {
        self.config = config

        self._embeddings.wrappedValue = SigLIP2Embeddings(config: config)
        self._encoder.wrappedValue = SigLIP2Encoder(config: config)
        self._postLayernorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
        self._head.wrappedValue = SigLIP2Head(config: config)
    }

    /// Forward pass returning both pooled output and patch embeddings
    public func callAsFunction(_ pixelValues: MLXArray) -> (pooled: MLXArray, patches: MLXArray) {
        // Embed patches
        var hiddenStates = embeddings(pixelValues)

        // Encode through transformer
        hiddenStates = encoder(hiddenStates)

        // Post layer norm
        hiddenStates = postLayernorm(hiddenStates)

        // Pooled output via attention head
        let pooled = head(hiddenStates)

        return (pooled: pooled, patches: hiddenStates)
    }

    /// Encode images and return pooled embeddings for i2l
    public func encode(_ pixelValues: MLXArray) -> MLXArray {
        let (pooled, _) = callAsFunction(pixelValues)
        return pooled  // [batch, hidden]
    }

}
