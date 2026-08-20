import MLX
import MLXFast
import MLXNN

final class NemotronOmniVisionAttention: Module {
    private let headCount = 16
    private let headDimension = 80
    private let scale: Float = 0.111_803_4

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var projection: Linear

    init(hiddenSize: Int) {
        self._qkv.wrappedValue = Linear(hiddenSize, hiddenSize * 3, bias: true)
        self._projection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let packed = qkv(input)
            .reshaped(batch, sequence, 3, headCount, headDimension)
            .transposed(2, 0, 3, 1, 4)
        let queries = packed[0, 0..., 0..., 0..., 0...]
        let keys = packed[1, 0..., 0..., 0..., 0...]
        let values = packed[2, 0..., 0..., 0..., 0...]
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .none
        )
        return projection(
            attended.transposed(0, 2, 1, 3)
                .reshaped(batch, sequence, headCount * headDimension)
        )
    }
}

final class NemotronOmniVisionMLP: Module {
    @ModuleInfo(key: "fc1") var first: Linear
    @ModuleInfo(key: "fc2") var second: Linear

    init(hiddenSize: Int) {
        self._first.wrappedValue = Linear(hiddenSize, hiddenSize * 4, bias: true)
        self._second.wrappedValue = Linear(hiddenSize * 4, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        second(MLXNN.gelu(first(input)))
    }
}

final class NemotronOmniVisionBlock: Module {
    @ModuleInfo(key: "norm1") var firstNorm: LayerNorm
    @ModuleInfo(key: "attn") var attention: NemotronOmniVisionAttention
    @ModuleInfo(key: "norm2") var secondNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NemotronOmniVisionMLP

    init(hiddenSize: Int) {
        self._firstNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6)
        self._attention.wrappedValue = NemotronOmniVisionAttention(hiddenSize: hiddenSize)
        self._secondNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6)
        self._mlp.wrappedValue = NemotronOmniVisionMLP(hiddenSize: hiddenSize)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let attended = input + attention(firstNorm(input))
        return attended + mlp(secondNorm(attended))
    }
}

final class NemotronOmniVisionPatchGenerator: Module {
    @ModuleInfo(key: "embedder") var imageEmbedder: Linear
    @ModuleInfo(key: "video_embedder") var videoEmbedder: Linear
    @ParameterInfo(key: "pos_embed") var positionEmbedding: MLXArray
    @ParameterInfo(key: "class_token") var classToken: MLXArray

    private let patchSize: Int
    private let hiddenSize: Int
    private let temporalPatchSize: Int

    init(patchSize: Int, hiddenSize: Int, temporalPatchSize: Int) {
        self.patchSize = patchSize
        self.hiddenSize = hiddenSize
        self.temporalPatchSize = temporalPatchSize
        self._imageEmbedder.wrappedValue = Linear(
            3 * patchSize * patchSize,
            hiddenSize,
            bias: false
        )
        self._videoEmbedder.wrappedValue = Linear(
            temporalPatchSize * 3 * patchSize * patchSize,
            hiddenSize,
            bias: false
        )
        self._positionEmbedding.wrappedValue = MLX.zeros(
            [1, 128 * 128, hiddenSize],
            dtype: .bfloat16
        )
        self._classToken.wrappedValue = MLX.zeros([10, hiddenSize], dtype: .bfloat16)
        super.init()
    }

    func imagePatches(_ pixels: MLXArray) -> (embeddings: MLXArray, height: Int, width: Int) {
        precondition(pixels.ndim == 4 && pixels.dim(3) == 3)
        let batch = pixels.dim(0)
        let patchHeight = pixels.dim(1) / patchSize
        let patchWidth = pixels.dim(2) / patchSize
        let patches = pixels
            .reshaped(batch, patchHeight, patchSize, patchWidth, patchSize, 3)
            .transposed(0, 1, 3, 5, 2, 4)
            .reshaped(batch, patchHeight * patchWidth, 3 * patchSize * patchSize)
        return (imageEmbedder(patches), patchHeight, patchWidth)
    }

    func videoPatches(_ frames: MLXArray) -> (embeddings: MLXArray, height: Int, width: Int) {
        precondition(frames.ndim == 4 && frames.dim(3) == 3)
        var pixels = frames
        let remainder = pixels.dim(0) % temporalPatchSize
        if remainder != 0 {
            let last = pixels[(pixels.dim(0) - 1)..., 0..., 0..., 0...]
            let padding = MLX.repeated(
                last,
                count: temporalPatchSize - remainder,
                axis: 0
            )
            pixels = MLX.concatenated([pixels, padding], axis: 0)
        }
        let groupCount = pixels.dim(0) / temporalPatchSize
        let patchHeight = pixels.dim(1) / patchSize
        let patchWidth = pixels.dim(2) / patchSize
        let patches = pixels
            .reshaped(
                groupCount,
                temporalPatchSize,
                patchHeight,
                patchSize,
                patchWidth,
                patchSize,
                3
            )
            .transposed(0, 2, 4, 1, 6, 3, 5)
            .reshaped(
                groupCount,
                patchHeight * patchWidth,
                temporalPatchSize * 3 * patchSize * patchSize
            )
        return (videoEmbedder(patches), patchHeight, patchWidth)
    }

    func addPositionAndClassTokens(
        _ patchEmbeddings: MLXArray,
        patchHeight: Int,
        patchWidth: Int
    ) -> MLXArray {
        let positions = interpolatedPositions(height: patchHeight, width: patchWidth)
            .asType(patchEmbeddings.dtype)
        let patches = patchEmbeddings + positions
        let classes = MLX.broadcast(
            classToken.expandedDimensions(axis: 0).asType(patches.dtype),
            to: [patches.dim(0), classToken.dim(0), hiddenSize]
        )
        return MLX.concatenated([classes, patches], axis: 1)
    }

    private func interpolatedPositions(height: Int, width: Int) -> MLXArray {
        let maximumDimension = max(height, width)
        var positions = positionEmbedding.reshaped(1, 128, 128, hiddenSize)
        if maximumDimension != 128 {
            let scale = Float(maximumDimension) / 128
            positions = Upsample(
                scaleFactor: .array([scale, scale]),
                mode: .linear(alignCorners: false)
            )(positions.asType(.float32)).asType(positionEmbedding.dtype)
        }
        positions = positions[0..., 0..<height, 0..<width, 0...]
        return positions.reshaped(1, height * width, hiddenSize)
    }
}

final class NemotronOmniVisionTransformer: Module {
    @ModuleInfo(key: "patch_generator") var patchGenerator: NemotronOmniVisionPatchGenerator
    @ModuleInfo(key: "blocks") var blocks: [NemotronOmniVisionBlock]

    init(config: NemotronOmniConfig) {
        self._patchGenerator.wrappedValue = NemotronOmniVisionPatchGenerator(
            patchSize: config.vision.patchSize,
            hiddenSize: config.visionHiddenSize,
            temporalPatchSize: config.vision.videoTemporalPatchSize
        )
        self._blocks.wrappedValue = (0..<32).map { _ in
            NemotronOmniVisionBlock(hiddenSize: config.visionHiddenSize)
        }
        super.init()
    }

    func encodeImage(_ pixels: MLXArray) -> (features: MLXArray, height: Int, width: Int) {
        let patches = patchGenerator.imagePatches(pixels)
        return (
            forward(
                patchGenerator.addPositionAndClassTokens(
                    patches.embeddings,
                    patchHeight: patches.height,
                    patchWidth: patches.width
                )
            ),
            patches.height,
            patches.width
        )
    }

    func encodeVideo(_ pixels: MLXArray) -> (features: MLXArray, height: Int, width: Int) {
        let patches = patchGenerator.videoPatches(pixels)
        return (
            forward(
                patchGenerator.addPositionAndClassTokens(
                    patches.embeddings,
                    patchHeight: patches.height,
                    patchWidth: patches.width
                )
            ),
            patches.height,
            patches.width
        )
    }

    private func forward(_ input: MLXArray) -> MLXArray {
        var hidden = input
        for block in blocks {
            hidden = block(hidden)
        }
        return hidden[0..., 10..., 0...]
    }
}

final class NemotronOmniVisionProjector: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "linear1") var first: Linear
    @ModuleInfo(key: "linear2") var second: Linear

    init(config: NemotronOmniConfig) {
        let inputSize = config.visionHiddenSize * 4
        self._norm.wrappedValue = RMSNorm(dimensions: inputSize, eps: 1e-5)
        self._first.wrappedValue = Linear(
            inputSize,
            config.projectorHiddenSize,
            bias: false
        )
        self._second.wrappedValue = Linear(
            config.projectorHiddenSize,
            config.language.hiddenSize,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let hidden = first(norm(input))
        let activated = MLX.square(MLX.maximum(hidden, MLXArray(0).asType(hidden.dtype)))
        return second(activated)
    }
}

final class NemotronOmniVisionTower: Module {
    @ModuleInfo(key: "model") var transformer: NemotronOmniVisionTransformer
    @ModuleInfo(key: "projector") var projector: NemotronOmniVisionProjector

    init(config: NemotronOmniConfig) {
        self._transformer.wrappedValue = NemotronOmniVisionTransformer(config: config)
        self._projector.wrappedValue = NemotronOmniVisionProjector(config: config)
        super.init()
    }

    func encodeImage(_ pixels: MLXArray) -> MLXArray {
        let encoded = transformer.encodeImage(pixels)
        return projector(pixelShuffle(encoded.features, height: encoded.height, width: encoded.width))
    }

    func encodeVideo(_ pixels: MLXArray) -> MLXArray {
        let encoded = transformer.encodeVideo(pixels)
        return projector(pixelShuffle(encoded.features, height: encoded.height, width: encoded.width))
    }

    private func pixelShuffle(_ input: MLXArray, height: Int, width: Int) -> MLXArray {
        precondition(height.isMultiple(of: 2) && width.isMultiple(of: 2))
        let batch = input.dim(0)
        let channels = input.dim(2)
        return input
            .reshaped(batch, height, width / 2, channels * 2)
            .transposed(0, 2, 1, 3)
            .reshaped(batch, width / 2, height / 2, channels * 4)
            .transposed(0, 2, 1, 3)
            .reshaped(batch, height / 2 * width / 2, channels * 4)
    }
}
