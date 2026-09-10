import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor
#if DEBUG
import MLXRandom
#endif

public final class MiniMaxH3Transformer: Module {
    public let configuration: MiniMaxH3TransformerConfiguration
    package var usesLayerwiseEvaluation = false
    package var clearsCacheAfterLayerwiseEvaluation = true
    package var usesBlockwiseCompilation = false
    package var usesFusedPostAttention = true
    package var exactKernelMode: MiniMaxH3ExactKernelMode = .disabled {
        didSet {
            for block in blocks {
                block.exactKernelMode = exactKernelMode
            }
            compiledBlockRunner = nil
            compiledBlockForwards = nil
        }
    }
    package var enabledExactKernelStages = Set(MiniMaxH3ExactKernelStage.allCases) {
        didSet {
            for block in blocks {
                block.enabledExactKernelStages = enabledExactKernelStages
            }
            compiledBlockRunner = nil
            compiledBlockForwards = nil
        }
    }
    package var exactKernelDispatchHandler: ((MiniMaxH3ExactKernelStage) -> Void)? {
        didSet {
            for block in blocks {
                block.exactKernelDispatchHandler = exactKernelDispatchHandler
            }
        }
    }
    package var exactKernelFallbackHandler: ((MiniMaxH3ExactKernelStage, String) -> Void)? {
        didSet {
            for block in blocks {
                block.exactKernelFallbackHandler = exactKernelFallbackHandler
            }
        }
    }
    package internal(set) var usesResidentBF16 = false
    package var maximumAttentionQueryTokensPerKernel = 1_024
    package var maximumAttentionHeadsPerKernel: Int?
    package var maximumAttentionKernelsPerEvaluation = 4
    package var dynamicSparseAttentionPolicy: DynamicSparseAttentionPolicy?
    package var dynamicSparseAttentionStepIndex = 0
    package var dynamicSparseAttentionStepCount = 0
    package var dynamicSparseAttentionLogHandler: ((String) -> Void)?
    package var blockTimingHandler: ((Int, TimeInterval, Memory.Snapshot) -> Void)?
    package var fastH3BlockPhaseTimingHandler: ((Int, TimeInterval, TimeInterval, TimeInterval) -> Void)?
    package var activeBlockIndices: Set<Int>?
    @ModuleInfo(key: "video_patch_proj") package var videoInput: Linear
    @ModuleInfo(key: "audio_patch_proj") package var audioInput: Linear
    @ModuleInfo(key: "condition_proj") package var textInput: Linear
    @ModuleInfo(key: "time_embedder") var timeEmbedder: MiniMaxH3TimeEmbedding?
    @ModuleInfo(key: "token_refiner") var tokenRefiner: MiniMaxH3TokenRefiner
    @ModuleInfo(key: "blocks") var blocks: [MiniMaxH3TransformerBlock]
    @ModuleInfo(key: "final_layer") var finalLayer: MiniMaxH3FinalLayer

    let inverseFrequencies: MLXArray
    let timeFrequencies: MLXArray
    var compiledBlockRunner: MiniMaxH3TransformerBlock?
    var compiledBlockForwards: MiniMaxH3CompiledBlockForwards?
    var dynamicSparseAttentionGateResults: [String: Bool] = [:]
    var fastH3CompressionGates: [MiniMaxH3FastH3CompressionGate?]

    package var supportsAffineQ8ExactKernels: Bool {
        !blocks.isEmpty && blocks.allSatisfy(\.supportsAffineQ8ExactKernels)
    }

    package static func requiresTiledFeedForwardEvaluationBoundary(
        rowCount: Int,
        feedForwardSize: Int,
        itemSize: Int
    ) -> Bool {
        precondition(rowCount > 0 && feedForwardSize > 0 && itemSize > 0)
        return UInt64(rowCount)
            * UInt64(feedForwardSize)
            * UInt64(itemSize) > UInt64(UInt32.max)
    }

    #if DEBUG
    package func feedForwardForBenchmark(_ value: MLXArray, blockIndex: Int) -> MLXArray {
        precondition(blocks.indices.contains(blockIndex))
        return blocks[blockIndex].feedForwardForBenchmark(value)
    }

    package func feedForwardInputForBenchmark(_ value: MLXArray, blockIndex: Int) -> MLXArray {
        precondition(blocks.indices.contains(blockIndex))
        return blocks[blockIndex].feedForwardInputForBenchmark(value)
    }

    package func feedForwardOutputForBenchmark(_ value: MLXArray, blockIndex: Int) -> MLXArray {
        precondition(blocks.indices.contains(blockIndex))
        return blocks[blockIndex].feedForwardOutputForBenchmark(value)
    }
    #endif

    package var affineQ8ExactKernelBlockCount: Int {
        blocks.count(where: \.supportsAffineQ8ExactKernels)
    }
    var adaLNWeightsAvailable: Bool

    public init(
        configuration: MiniMaxH3TransformerConfiguration = .init(),
        includeAdaLN: Bool = true
    ) {
        self.adaLNWeightsAvailable = includeAdaLN
        self.configuration = configuration
        self.fastH3CompressionGates = Array(repeating: nil, count: configuration.layerCount)
        self._videoInput.wrappedValue = Linear(
            configuration.videoPatchDimension,
            configuration.hiddenSize,
            bias: true
        )
        self._audioInput.wrappedValue = Linear(
            configuration.audioLatentChannels,
            configuration.hiddenSize,
            bias: true
        )
        self._textInput.wrappedValue = Linear(
            configuration.textDimension,
            configuration.hiddenSize,
            bias: true
        )
        self._timeEmbedder.wrappedValue = includeAdaLN
            ? MiniMaxH3TimeEmbedding(configuration: configuration)
            : nil
        self._tokenRefiner.wrappedValue = MiniMaxH3TokenRefiner(configuration: configuration)
        self._blocks.wrappedValue = (0..<configuration.layerCount).map { _ in
            MiniMaxH3TransformerBlock(configuration: configuration, includeAdaLN: includeAdaLN)
        }
        self._finalLayer.wrappedValue = MiniMaxH3FinalLayer(
            configuration: configuration,
            includeAdaLN: includeAdaLN
        )
        let frequencies = (0..<configuration.ropeFrequencyCount).map { index in
            1 / pow(configuration.ropeTheta, Float(index) / Float(configuration.ropeFrequencyCount))
        }
        self.inverseFrequencies = MLXArray(frequencies)
        let halfTimeDimension = configuration.timeFrequencyDimension / 2
        self.timeFrequencies = MLXArray((0..<halfTimeDimension).map { index in
            exp(-log(Float(10_000)) * Float(index) / Float(halfTimeDimension))
        })
    }

    package var estimatedResidentBF16ByteCount: UInt64 {
        leafModules().flattened().reduce(into: UInt64(0)) { total, entry in
            guard let linear = entry.1 as? Linear else { return }
            total += miniMaxH3ResidentBF16ByteCount(linear)
        }
    }

    package var activeBlockCount: Int {
        activeBlockIndices?.count ?? blocks.count
    }

    package var usesFastH3VSA: Bool {
        !fastH3CompressionGates.isEmpty
            && fastH3CompressionGates.allSatisfy { $0 != nil }
    }

    package func installFastH3CompressionGate(_ weight: MLXArray, blockIndex: Int) {
        precondition(fastH3CompressionGates.indices.contains(blockIndex))
        precondition(weight.shape == [
            configuration.attentionHeadCount * configuration.attentionHeadDimension,
            configuration.hiddenSize,
        ])
        fastH3CompressionGates[blockIndex] = MiniMaxH3FastH3CompressionGate(weight: weight)
        MLX.eval(weight)
        compiledBlockRunner = nil
        compiledBlockForwards = nil
    }

    package func installFastH3QuantizedCompressionGate(
        codes: MLXArray,
        scales: MLXArray,
        biases: MLXArray,
        groupSize: Int,
        bits: Int,
        blockIndex: Int
    ) {
        precondition(fastH3CompressionGates.indices.contains(blockIndex))
        let outputDimension = configuration.attentionHeadCount
            * configuration.attentionHeadDimension
        precondition(bits == 8 && groupSize == 64)
        precondition(codes.shape == [outputDimension, configuration.hiddenSize * bits / 32])
        precondition(scales.shape == [outputDimension, configuration.hiddenSize / groupSize])
        precondition(biases.shape == scales.shape)
        fastH3CompressionGates[blockIndex] = MiniMaxH3FastH3CompressionGate(
            codes: codes,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits
        )
        MLX.eval(codes, scales, biases)
        compiledBlockRunner = nil
        compiledBlockForwards = nil
    }

    /// Expands compact quantized storage into resident bf16 linear weights.
    /// Each transformer/refiner block is replaced and the Metal cache is
    /// cleared before moving to the next one, bounding conversion residency
    /// instead of retaining a second whole-model copy.
    package func materializeResidentBF16() -> MiniMaxH3ResidentBF16Materialization {
        compiledBlockRunner = nil
        compiledBlockForwards = nil

        func materialize(_ module: Module) {
            _ = miniMaxH3MaterializeResidentBF16(in: module)
            miniMaxH3EvaluateParameters(in: module)
            MLX.Memory.clearCache()
        }

        for block in tokenRefiner.blocks {
            materialize(block)
        }
        for block in blocks {
            materialize(block)
        }
        if let timeEmbedder {
            materialize(timeEmbedder)
        }
        materialize(finalLayer)

        var rootReplacements: [(String, Module)] = []
        for (path, linear) in [
            ("video_patch_proj", videoInput),
            ("audio_patch_proj", audioInput),
            ("condition_proj", textInput),
        ] {
            guard let resident = miniMaxH3ResidentBF16Linear(linear) else { continue }
            rootReplacements.append((path, resident.linear))
        }
        if !rootReplacements.isEmpty {
            update(modules: ModuleChildren.unflattened(rootReplacements))
        }

        // Sharded BF16 checkpoints already contain dense Linear modules, so
        // conversion alone is a no-op. Evaluating the complete parameter tree
        // here makes the requested residency real instead of charging each
        // transformer block to the first denoise pass.
        miniMaxH3EvaluateParameters(in: self)
        MLX.Memory.clearCache()

        let linears = leafModules().flattened().compactMap { $0.1 as? Linear }
        usesResidentBF16 = !linears.isEmpty && !linears.contains { $0 is QuantizedLinear }
        return .init(
            linearCount: linears.count,
            byteCount: linears.reduce(into: UInt64(0)) { total, linear in
                total += miniMaxH3ResidentBF16ByteCount(linear)
            }
        )
    }

}
