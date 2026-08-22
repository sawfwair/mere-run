import Foundation
import MLX
import MLXNN

struct SenseNovaU15VisionRoPEKey: Hashable {
    let height: Int
    let width: Int
    let dtype: DType
}

struct SenseNovaU15VisionRoPETables {
    let xCosine: MLXArray
    let xSine: MLXArray
    let yCosine: MLXArray
    let ySine: MLXArray

    var arrays: [MLXArray] { [xCosine, xSine, yCosine, ySine] }
}

final class SenseNovaU15VisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "dense_embedding") var denseEmbedding: Conv2d

    private let hiddenSize: Int
    private let patchSize: Int
    private let downsampleFactor: Int
    private let ropeBase: Float
    private var cachedRoPE: (key: SenseNovaU15VisionRoPEKey, tables: SenseNovaU15VisionRoPETables)?

    init(config: SenseNovaU15Config) {
        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: config.visionConfig.numberOfChannels,
            outputChannels: config.visionConfig.hiddenSize,
            kernelSize: IntOrPair(config.visionConfig.patchSize),
            stride: IntOrPair(config.visionConfig.patchSize),
            bias: true
        )
        let downsampleFactor = Int(1 / config.downsampleRatio)
        self._denseEmbedding.wrappedValue = Conv2d(
            inputChannels: config.visionConfig.hiddenSize,
            outputChannels: config.llmConfig.hiddenSize,
            kernelSize: IntOrPair(downsampleFactor),
            stride: IntOrPair(downsampleFactor),
            bias: true
        )
        self.hiddenSize = config.visionConfig.hiddenSize
        self.patchSize = config.visionConfig.patchSize
        self.downsampleFactor = downsampleFactor
        self.ropeBase = config.visionConfig.ropeTheta
        super.init()
    }

    /// Input is normalized NHWC pixels. Output is the merged image-token sequence.
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        precondition(pixels.ndim == 4 && pixels.dim(0) == 1)
        let gridHeight = pixels.dim(1) / patchSize
        let gridWidth = pixels.dim(2) / patchSize
        var patches = MLXNN.gelu(patchEmbedding(pixels))
            .reshaped(gridHeight * gridWidth, hiddenSize)
        let tables = cachedRoPETables(height: gridHeight, width: gridWidth, dtype: patches.dtype)
        patches = Self.applyVisionRoPE(patches, tables: tables)
        patches = patches.reshaped(1, gridHeight, gridWidth, hiddenSize)
        let merged = denseEmbedding(patches)
        return merged.reshaped(1, gridHeight / downsampleFactor * gridWidth / downsampleFactor, -1)
    }

    private func cachedRoPETables(height: Int, width: Int, dtype: DType) -> SenseNovaU15VisionRoPETables {
        let key = SenseNovaU15VisionRoPEKey(height: height, width: width, dtype: dtype)
        if cachedRoPE?.key == key, let tables = cachedRoPE?.tables { return tables }
        let tables = Self.makeRoPETables(
            height: height,
            width: width,
            hiddenSize: hiddenSize,
            base: ropeBase,
            dtype: dtype
        )
        cachedRoPE = (key, tables)
        return tables
    }

    static func makeRoPETables(
        height: Int,
        width: Int,
        hiddenSize: Int,
        base: Float,
        dtype: DType
    ) -> SenseNovaU15VisionRoPETables {
        let half = hiddenSize / 2
        let xPositions = (0..<height).flatMap { _ in (0..<width).map(Float.init) }
        let yPositions = (0..<height).flatMap { row in [Float](repeating: Float(row), count: width) }
        let x = makeInterleavedRoPETable(positions: xPositions, dimension: half, base: base, dtype: dtype)
        let y = makeInterleavedRoPETable(positions: yPositions, dimension: half, base: base, dtype: dtype)
        return SenseNovaU15VisionRoPETables(
            xCosine: x.cosine,
            xSine: x.sine,
            yCosine: y.cosine,
            ySine: y.sine
        )
    }

    static func applyVisionRoPE(_ input: MLXArray, tables: SenseNovaU15VisionRoPETables) -> MLXArray {
        let half = input.dim(-1) / 2
        return MLX.concatenated([
            applyInterleavedRoPE(
                input[0..., 0..<half],
                cosine: tables.xCosine,
                sine: tables.xSine
            ),
            applyInterleavedRoPE(
                input[0..., half...],
                cosine: tables.yCosine,
                sine: tables.ySine
            ),
        ], axis: -1)
    }

    private static func makeInterleavedRoPETable(
        positions: [Float],
        dimension: Int,
        base: Float,
        dtype: DType
    ) -> (cosine: MLXArray, sine: MLXArray) {
        let indices = MLXArray(stride(from: Float(0), to: Float(dimension), by: 2))
        let inverseFrequencies = 1 / MLX.pow(MLXArray(base), indices / Float(dimension))
        let frequencies = MLXArray(positions)[0..., .newAxis] * inverseFrequencies[.newAxis, 0...]
        return (MLX.cos(frequencies).asType(dtype), MLX.sin(frequencies).asType(dtype))
    }

    private static func applyInterleavedRoPE(
        _ input: MLXArray,
        cosine: MLXArray,
        sine: MLXArray
    ) -> MLXArray {
        let even = input[0..., .stride(by: 2)]
        let odd = input[0..., .stride(from: 1, by: 2)]
        let rotatedEven = even * cosine - odd * sine
        let rotatedOdd = even * sine + odd * cosine
        return MLX.stacked([rotatedEven, rotatedOdd], axis: -1).reshaped(input.shape)
    }
}

final class SenseNovaU15VisionModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: SenseNovaU15VisionEmbeddings

    init(config: SenseNovaU15Config) {
        self._embeddings.wrappedValue = SenseNovaU15VisionEmbeddings(config: config)
        super.init()
    }
}

final class SenseNovaU15TimestepEmbedder: Module {
    @ModuleInfo(key: "mlp") var mlp: (Linear, SiLUModule, Linear)
    private let frequencyEmbeddingSize = 256

    init(hiddenSize: Int) {
        self._mlp.wrappedValue = (
            Linear(frequencyEmbeddingSize, hiddenSize, bias: true),
            SiLUModule(),
            Linear(hiddenSize, hiddenSize, bias: true)
        )
        super.init()
    }

    func callAsFunction(_ timesteps: MLXArray) -> MLXArray {
        let half = frequencyEmbeddingSize / 2
        let frequencies = MLX.exp(
            -MLX.log(MLXArray(Float(10_000)))
                * MLXArray(0..<half).asType(.float32)
                / Float(half)
        )
        let arguments = timesteps.asType(.float32)[0..., .newAxis] * frequencies[.newAxis, 0...]
        let embedding = MLX.concatenated([MLX.cos(arguments), MLX.sin(arguments)], axis: -1)
        return mlp.2(mlp.1(mlp.0(embedding)))
    }
}

final class SenseNovaU15ConvDecoder: Module {
    @ModuleInfo(key: "conv1") var firstConvolution: Conv2d
    @ModuleInfo(key: "conv2") var secondConvolution: Conv2d

    init(hiddenSize: Int) {
        precondition(hiddenSize.isMultiple(of: 16))
        let intermediate = hiddenSize / 4
        self._firstConvolution.wrappedValue = Conv2d(
            inputChannels: intermediate,
            outputChannels: intermediate,
            kernelSize: 3,
            padding: 1,
            bias: true
        )
        self._secondConvolution.wrappedValue = Conv2d(
            inputChannels: intermediate / 4,
            outputChannels: 192,
            kernelSize: 3,
            padding: 1,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray, tokenHeight: Int, tokenWidth: Int) -> MLXArray {
        var pixels = hidden.reshaped(hidden.dim(0), tokenHeight, tokenWidth, hidden.dim(-1))
        pixels = MLXNN.gelu(firstConvolution(Self.pixelShuffle(pixels, factor: 2)))
        pixels = secondConvolution(Self.pixelShuffle(pixels, factor: 2))
        return Self.pixelShuffle(pixels, factor: 8)
    }

    static func pixelShuffle(_ input: MLXArray, factor: Int) -> MLXArray {
        let batch = input.dim(0)
        let height = input.dim(1)
        let width = input.dim(2)
        let channels = input.dim(3) / (factor * factor)
        return input
            .reshaped(batch, height, width, channels, factor, factor)
            .transposed(0, 1, 4, 2, 5, 3)
            .reshaped(batch, height * factor, width * factor, channels)
    }
}

final class SenseNovaU15FlowModules: Module {
    @ModuleInfo(key: "vision_model_mot_gen") var generationVisionModel: SenseNovaU15VisionModel
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: SenseNovaU15TimestepEmbedder
    @ModuleInfo(key: "noise_scale_embedder") var noiseScaleEmbedder: SenseNovaU15TimestepEmbedder
    @ModuleInfo(key: "fm_head") var flowHead: SenseNovaU15ConvDecoder

    init(config: SenseNovaU15Config) {
        self._generationVisionModel.wrappedValue = SenseNovaU15VisionModel(config: config)
        self._timestepEmbedder.wrappedValue = SenseNovaU15TimestepEmbedder(hiddenSize: config.llmConfig.hiddenSize)
        self._noiseScaleEmbedder.wrappedValue = SenseNovaU15TimestepEmbedder(hiddenSize: config.llmConfig.hiddenSize)
        self._flowHead.wrappedValue = SenseNovaU15ConvDecoder(hiddenSize: config.llmConfig.hiddenSize)
        super.init()
    }
}

final class SenseNovaU15Model: Module {
    @ModuleInfo(key: "vision_model") var visionModel: SenseNovaU15VisionModel
    @ModuleInfo(key: "language_model") var languageModel: SenseNovaU15CausalLM
    @ModuleInfo(key: "fm_modules") var flowModules: SenseNovaU15FlowModules

    init(config: SenseNovaU15Config) {
        self._visionModel.wrappedValue = SenseNovaU15VisionModel(config: config)
        self._languageModel.wrappedValue = SenseNovaU15CausalLM(config: config.llmConfig)
        self._flowModules.wrappedValue = SenseNovaU15FlowModules(config: config)
        super.init()
    }
}
