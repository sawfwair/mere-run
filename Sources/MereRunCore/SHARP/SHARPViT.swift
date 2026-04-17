import Foundation
import MLX
import MLXNN

public struct SharpViTConfig: Sendable {
    public var imageSize: Int
    public var patchSize: Int
    public var inChannels: Int
    public var embedDim: Int
    public var depth: Int
    public var numHeads: Int
    public var initValues: Float
    public var mlpRatio: Float
    public var qkvBias: Bool

    public init(
        imageSize: Int,
        patchSize: Int,
        inChannels: Int,
        embedDim: Int,
        depth: Int,
        numHeads: Int,
        initValues: Float,
        mlpRatio: Float = 4.0,
        qkvBias: Bool = true
    ) {
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.inChannels = inChannels
        self.embedDim = embedDim
        self.depth = depth
        self.numHeads = numHeads
        self.initValues = initValues
        self.mlpRatio = mlpRatio
        self.qkvBias = qkvBias
    }
}

enum SharpMonodepthPresets {
    static func vitConfig(_ preset: String) -> SharpViTConfig? {
        switch preset {
        case "dinov2l16_384":
            return SharpViTConfig(
                imageSize: 384,
                patchSize: 16,
                inChannels: 3,
                embedDim: 1024,
                depth: 24,
                numHeads: 16,
                initValues: 1e-5
            )
        case "tiny16_64":
            // Small preset for unit tests and local shape checks.
            return SharpViTConfig(
                imageSize: 16,
                patchSize: 16,
                inChannels: 3,
                embedDim: 64,
                depth: 4,
                numHeads: 4,
                initValues: 1e-5
            )
        default:
            return nil
        }
    }

    static func encoderDims(_ preset: String) -> [Int]? {
        switch preset {
        case "dinov2l16_384":
            return [256, 512, 1024, 1024]
        case "tiny16_64":
            return [16, 32, 64, 64]
        default:
            return nil
        }
    }

    static func hookIDs(_ preset: String) -> [Int]? {
        switch preset {
        case "dinov2l16_384":
            return [5, 11, 17, 23]
        case "tiny16_64":
            return [0, 1, 2, 3]
        default:
            return nil
        }
    }
}

final class SharpViTPatchEmbed: Module, @unchecked Sendable {
    @ModuleInfo(key: "proj") var proj: Conv2d

    let imageSize: Int
    let patchSize: Int

    init(config: SharpViTConfig) {
        self.imageSize = config.imageSize
        self.patchSize = config.patchSize
        self._proj.wrappedValue = Conv2d(
            inputChannels: config.inChannels,
            outputChannels: config.embedDim,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize),
            padding: IntOrPair(0),
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (tokens: MLXArray, gridHeight: Int, gridWidth: Int) {
        // Input is NCHW. MLX Conv2d expects NHWC.
        let nhwc = x.transposed(0, 2, 3, 1)
        let y = proj(nhwc)
        let batch = y.dim(0)
        let gridHeight = y.dim(1)
        let gridWidth = y.dim(2)
        let channels = y.dim(3)
        let tokens = y.reshaped(batch, gridHeight * gridWidth, channels)
        return (tokens, gridHeight, gridWidth)
    }
}

final class SharpViTAttention: Module, @unchecked Sendable {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(config: SharpViTConfig) {
        precondition(config.embedDim % config.numHeads == 0, "embedDim must be divisible by numHeads.")

        self.embedDim = config.embedDim
        self.numHeads = config.numHeads
        self.headDim = config.embedDim / config.numHeads
        self.scale = 1.0 / sqrt(Float(headDim))
        self._qkv.wrappedValue = Linear(config.embedDim, config.embedDim * 3, bias: config.qkvBias)
        self._proj.wrappedValue = Linear(config.embedDim, config.embedDim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)

        let qkvTensor = qkv(x).reshaped(batch, seqLen, 3, numHeads, headDim)
        var q = qkvTensor[0..., 0..., 0, 0..., 0...]
        var k = qkvTensor[0..., 0..., 1, 0..., 0...]
        var v = qkvTensor[0..., 0..., 2, 0..., 0...]

        // [B, S, H, D] -> [B, H, S, D]
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        var attention = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
        attention = softmax(attention, axis: -1)
        var output = MLX.matmul(attention, v)

        // [B, H, S, D] -> [B, S, C]
        output = output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, embedDim)
        return proj(output)
    }
}

final class SharpViTMLP: Module, @unchecked Sendable {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(config: SharpViTConfig) {
        let hidden = Int(Float(config.embedDim) * config.mlpRatio)
        self._fc1.wrappedValue = Linear(config.embedDim, hidden, bias: true)
        self._fc2.wrappedValue = Linear(hidden, config.embedDim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

final class SharpViTLayerScale: Module, @unchecked Sendable {
    @ModuleInfo(key: "gamma") var gamma: MLXArray

    init(dim: Int, initValue: Float) {
        self._gamma.wrappedValue = MLX.full([dim], values: MLXArray(initValue), dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gamma
    }
}

final class SharpViTBlock: Module, @unchecked Sendable {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attention: SharpViTAttention
    @ModuleInfo(key: "ls1") var layerScale1: SharpViTLayerScale
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SharpViTMLP
    @ModuleInfo(key: "ls2") var layerScale2: SharpViTLayerScale

    init(config: SharpViTConfig) {
        self._norm1.wrappedValue = LayerNorm(dimensions: config.embedDim)
        self._attention.wrappedValue = SharpViTAttention(config: config)
        self._layerScale1.wrappedValue = SharpViTLayerScale(dim: config.embedDim, initValue: config.initValues)
        self._norm2.wrappedValue = LayerNorm(dimensions: config.embedDim)
        self._mlp.wrappedValue = SharpViTMLP(config: config)
        self._layerScale2.wrappedValue = SharpViTLayerScale(dim: config.embedDim, initValue: config.initValues)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x + layerScale1(attention(norm1(x)))
        y = y + layerScale2(mlp(norm2(y)))
        return y
    }
}

public final class SharpVisionTransformer: Module, @unchecked Sendable {
    public let config: SharpViTConfig
    private let intermediateFeatureIDs: Set<Int>?

    @ModuleInfo(key: "patch_embed") var patchEmbed: SharpViTPatchEmbed
    @ModuleInfo(key: "cls_token") var clsToken: MLXArray
    @ModuleInfo(key: "pos_embed") var posEmbed: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [SharpViTBlock]
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "head") var head: Linear

    public init(config: SharpViTConfig, intermediateFeatureIDs: [Int]? = nil) {
        self.config = config
        if let intermediateFeatureIDs {
            self.intermediateFeatureIDs = Set(intermediateFeatureIDs)
        } else {
            self.intermediateFeatureIDs = nil
        }

        self._patchEmbed.wrappedValue = SharpViTPatchEmbed(config: config)

        let grid = config.imageSize / config.patchSize
        let sequenceLength = 1 + (grid * grid)

        self._clsToken.wrappedValue = MLX.zeros([1, 1, config.embedDim], dtype: .float32)
        self._posEmbed.wrappedValue = MLX.zeros([1, sequenceLength, config.embedDim], dtype: .float32)
        self._blocks.wrappedValue = (0..<config.depth).map { _ in SharpViTBlock(config: config) }
        self._norm.wrappedValue = LayerNorm(dimensions: config.embedDim)
        self._head.wrappedValue = Linear(config.embedDim, config.embedDim, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> (features: MLXArray, intermediateFeatures: [Int: MLXArray]) {
        let (patchTokens, _, _) = patchEmbed(x)
        let batch = patchTokens.dim(0)
        let hidden = patchTokens.dim(2)

        let cls = MLX.broadcast(clsToken, to: [batch, 1, hidden]).asType(patchTokens.dtype)
        var embeddings = MLX.concatenated([cls, patchTokens], axis: 1)

        if embeddings.dim(1) == posEmbed.dim(1) {
            embeddings = embeddings + posEmbed.asType(embeddings.dtype)
        }

        var intermediate: [Int: MLXArray] = [:]
        for (idx, block) in blocks.enumerated() {
            embeddings = block(embeddings)
            if intermediateFeatureIDs?.contains(idx) == true {
                intermediate[idx] = embeddings
            }
        }

        let normalized = norm(embeddings)
        let features = reshapeFeature(normalized)
        return (features, intermediate)
    }

    func reshapeFeature(_ embeddings: MLXArray) -> MLXArray {
        let batch = embeddings.dim(0)
        let sequenceLength = embeddings.dim(1)
        let channels = embeddings.dim(2)

        // Drop class token.
        let patchTokens = embeddings[0..., 1..., 0...]
        let numPatches = sequenceLength - 1
        let gridSize = max(1, Int(round(sqrt(Double(numPatches)))))
        let width = max(1, numPatches / gridSize)

        let nhwc = patchTokens.reshaped(batch, gridSize, width, channels)
        return nhwc.transposed(0, 3, 1, 2)
    }

    public func internalResolution() -> Int {
        config.imageSize
    }
}

public func createSharpViT(
    preset: String,
    intermediateFeatureIDs: [Int]? = nil
) -> SharpVisionTransformer {
    guard let config = SharpMonodepthPresets.vitConfig(preset) else {
        fatalError("Unsupported SHARP ViT preset: \(preset)")
    }

    return SharpVisionTransformer(config: config, intermediateFeatureIDs: intermediateFeatureIDs)
}
