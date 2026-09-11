import Foundation
import MediaIO
@preconcurrency import MLX
import MLXNN

public struct MarigoldV2InferenceConfiguration: Equatable, Sendable {
    /// The VAE downsamples by eight and the transformer packs 2x2 latent
    /// patches, so both image sides must be multiples of sixteen.
    public static let alignment = 16

    /// Longest side the transformer runs at unless the caller raises it. The
    /// published figures are about 17 GB of accelerator memory at 1024 and
    /// 29 GB at 2048, so the default keeps a single run inside a 24 GB machine.
    public static let defaultMaximumEdge = 1_024

    public let checkpoint: MarigoldV2DepthCheckpoint
    public let maximumEdge: Int?

    public init(
        checkpoint: MarigoldV2DepthCheckpoint = MarigoldV2Repository.installedCheckpoint,
        maximumEdge: Int? = defaultMaximumEdge
    ) {
        self.checkpoint = checkpoint
        self.maximumEdge = maximumEdge.map { max(Self.alignment, $0) }
    }

    /// Size the transformer runs at: the source size reduced under `maximumEdge`,
    /// then aligned to the patch grid. Artifacts are resampled back to the source
    /// resolution afterwards.
    public func inferenceSize(width: Int, height: Int) -> (width: Int, height: Int) {
        var targetWidth = Double(width)
        var targetHeight = Double(height)
        if let maximumEdge {
            let longest = max(targetWidth, targetHeight)
            if longest > Double(maximumEdge) {
                let scale = Double(maximumEdge) / longest
                targetWidth *= scale
                targetHeight *= scale
            }
        }
        return (width: Self.align(targetWidth), height: Self.align(targetHeight))
    }

    static func align(_ value: Double) -> Int {
        max(1, Int((value / Double(alignment)).rounded())) * alignment
    }
}

public struct MarigoldV2RunResult: Sendable {
    public let export: MarigoldV2DepthExportResult
    public let checkpoint: MarigoldV2DepthCheckpoint
    public let inferenceWidth: Int
    public let inferenceHeight: Int
    public let promptTokenCount: Int
    public let adapterPairCount: Int
    public let vaeDecoderTensorCount: Int
    public let modelLoadSeconds: Double
    public let inferenceSeconds: Double
    public let postprocessSeconds: Double
}

public enum MarigoldV2GeneratorError: Error, Equatable, LocalizedError, Sendable {
    case missingModelFiles([String])
    case unexpectedDecodedElementCount(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .missingModelFiles(let paths):
            return "Marigold V2 install is incomplete. Missing: \(paths.joined(separator: ", "))."
        case let .unexpectedDecodedElementCount(expected, actual):
            return "Marigold V2 decoded \(actual) depth samples, expected \(expected)."
        }
    }
}

/// Runs Marigold V2 monocular depth on a single image.
///
/// The pipeline is one rectified-flow step at a fixed timestep with no classifier-free
/// guidance: encode the image into the Qwen VAE latent space, predict a velocity,
/// step once, decode, and read depth from the mean of the decoded channels.
public actor MarigoldV2Generator {
    /// Fixed inference timestep, normalized to [0, 1]. The transformer's sinusoidal
    /// embedding rescales it by 1,000 internally, reproducing the reference's 499.
    static let timestep: Float = 0.499

    private var loaded: LoadedModel?

    struct LoadedModel {
        let rootURL: URL
        let checkpoint: MarigoldV2DepthCheckpoint
        let transformer: MMDiT
        let vae: QwenImageEditVAE
        let prompt: MarigoldV2PromptConditioning
        let adapterPairCount: Int
        let vaeDecoderTensorCount: Int
    }

    public init() {}

    public func generate(
        imageURL: URL,
        outputDirectory: URL,
        model: String? = nil,
        configuration: MarigoldV2InferenceConfiguration = MarigoldV2InferenceConfiguration(),
        progress: (@Sendable (String) -> Void)? = nil
    ) async throws -> MarigoldV2RunResult {
        let standardizedImage = imageURL.standardizedFileURL
        let admittedInput = try VFXImageInputSnapshotBatch.capture([standardizedImage])
        defer { admittedInput.cleanup() }
        let snapshotURL = admittedInput.snapshotURLs[0]

        progress?("Decoding \(standardizedImage.lastPathComponent)")
        let image = try MediaImageIO.decode(snapshotURL)
        let dimensions = try VFXImageInputValidator.validate(
            width: image.width,
            height: image.height,
            path: standardizedImage.path
        )
        let inferenceSize = configuration.inferenceSize(
            width: dimensions.width,
            height: dimensions.height
        )

        progress?("Resolving Marigold V2 weights")
        let loadStart = Date()
        let loadedModel = try await loadModelIfNeeded(
            requestedModel: model,
            checkpoint: configuration.checkpoint,
            progress: progress
        )
        let loadSeconds = Date().timeIntervalSince(loadStart)

        progress?("Running single-step depth inference at \(inferenceSize.width)x\(inferenceSize.height)")
        let inferenceStart = Date()
        let resized = try MediaImageIO.resized(
            image,
            width: inferenceSize.width,
            height: inferenceSize.height
        )
        let rgb = Self.rgbNCHWSigned(resized)
        let rawDepth = Self.predictDepth(rgb: rgb, model: loadedModel)
        let inferenceSeconds = Date().timeIntervalSince(inferenceStart)
        let expectedSamples = inferenceSize.width * inferenceSize.height
        guard rawDepth.count == expectedSamples else {
            throw MarigoldV2GeneratorError.unexpectedDecodedElementCount(
                expected: expectedSamples,
                actual: rawDepth.count
            )
        }

        progress?("Normalizing and resampling to \(dimensions.width)x\(dimensions.height)")
        let postprocessStart = Date()
        let normalized = MarigoldV2DepthNormalizer.normalize(
            raw: rawDepth,
            parameterization: configuration.checkpoint.parameterization
        )
        let depth = Self.resampleBilinear(
            normalized.values,
            width: inferenceSize.width,
            height: inferenceSize.height,
            targetWidth: dimensions.width,
            targetHeight: dimensions.height
        )
        let postprocessSeconds = Date().timeIntervalSince(postprocessStart)

        progress?("Writing EXR, preview, and manifest artifacts")
        let provenance = GeometryModelProvenance(
            modelID: MarigoldV2Repository.modelId,
            upstreamRepository: MarigoldV2Repository.upstreamRepoId,
            upstreamRevision: MarigoldV2Repository.upstreamRevision,
            license: MarigoldV2Repository.license,
            weightsSHA256: MarigoldV2Repository.trainablesPin.sha256
        )
        let export = try MarigoldV2DepthArtifactExporter.export(
            depth: depth,
            width: dimensions.width,
            height: dimensions.height,
            inferenceWidth: inferenceSize.width,
            inferenceHeight: inferenceSize.height,
            statistics: normalized.statistics,
            checkpoint: configuration.checkpoint,
            inputURL: standardizedImage,
            outputDirectory: outputDirectory,
            provenance: provenance,
            inputRecord: admittedInput.inputRecords[0]
        )

        return MarigoldV2RunResult(
            export: export,
            checkpoint: configuration.checkpoint,
            inferenceWidth: inferenceSize.width,
            inferenceHeight: inferenceSize.height,
            promptTokenCount: loadedModel.prompt.tokenCount,
            adapterPairCount: loadedModel.adapterPairCount,
            vaeDecoderTensorCount: loadedModel.vaeDecoderTensorCount,
            modelLoadSeconds: loadSeconds,
            inferenceSeconds: inferenceSeconds,
            postprocessSeconds: postprocessSeconds
        )
    }

    public func unload() {
        loaded = nil
        MLX.Memory.clearCache()
    }

    // MARK: - Inference

    /// One rectified-flow step, then decode. Returns the per-pixel depth the
    /// decoder produced, still in the model's own affine-invariant range.
    private static func predictDepth(rgb: MLXArray, model: LoadedModel) -> [Float] {
        // Match the image runtime: encode in the VAE's own precision, then run
        // the transformer step in bfloat16 as the checkpoints were trained.
        let latents = model.vae.encodeConditioning(rgb).asType(.bfloat16)
        let latentHeight = latents.dim(2)
        let latentWidth = latents.dim(3)

        let packed = QwenImageEditLatentCreator.packLatents(latents).asType(.bfloat16)
        let timestep = MLXArray([Self.timestep]).asType(.bfloat16)
        let prediction = model.transformer(
            hiddenStates: packed,
            timestep: timestep,
            contextEmbeds: model.prompt.embeddings,
            contextMask: model.prompt.mask,
            imageShapes: [(temporal: 1, height: latentHeight / 2, width: latentWidth / 2)],
            outputTokenCount: packed.dim(1)
        )

        let velocity = QwenImageEditLatentCreator.unpackLatents(
            prediction,
            height: latentHeight,
            width: latentWidth,
            channels: latents.dim(1)
        )
        // The checkpoints predict velocity, so one Euler step from the encoded
        // latent is a subtraction rather than the eps-prediction addition.
        let stepped = latents - velocity.asType(latents.dtype)

        let pixels = model.vae.decodeGenerated(stepped)
        // Depth is carried identically on all three decoded channels; averaging
        // them is what the reference reads out.
        let depth = pixels.mean(axis: 1).asType(.float32)
        MLX.eval(depth)
        return depth.asArray(Float.self)
    }

    // MARK: - Loading

    private func loadModelIfNeeded(
        requestedModel: String?,
        checkpoint: MarigoldV2DepthCheckpoint,
        progress: (@Sendable (String) -> Void)?
    ) async throws -> LoadedModel {
        let resolved = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: requestedModel,
            defaultModelID: MarigoldV2Repository.modelId,
            allowAutoDownload: false
        )
        let rootURL = resolved.url.standardizedFileURL
        if let loaded, loaded.rootURL == rootURL, loaded.checkpoint == checkpoint {
            return loaded
        }

        let resources = MarigoldV2Resources(rootURL: rootURL, checkpoint: checkpoint)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw MarigoldV2GeneratorError.missingModelFiles(missing.map(\.path))
        }
        try MarigoldV2Repository.verifyInstalledArtifacts(rootURL: rootURL)

        let configs = try MarigoldV2ModelConfigs.load(from: resources)

        progress?("Loading the frozen transformer")
        let transformer = try Self.loadTransformer(resources: resources, config: configs.transformer)

        progress?("Installing Marigold V2 adapters")
        let adapterPairCount = try MarigoldV2LoRAAdapter.install(
            url: resources.trainablesURL,
            into: transformer
        )
        MLX.eval(transformer)
        MLX.Memory.clearCache()

        progress?("Loading the VAE")
        let vae = QwenImageEditVAE(config: configs.vae)
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.vaeWeightsURL,
            to: vae.underlyingVAE,
            dtype: .bfloat16,
            verify: [.shapeMismatch],
            mapper: QwenImageEditVAE.weightMapper
        )
        let vaeDecoderTensorCount = try MarigoldV2VAEDecoder.applyIfPresent(
            url: resources.trainablesURL,
            to: vae
        )
        MLX.eval(vae)
        MLX.Memory.clearCache()

        progress?("Loading precomputed prompt conditioning")
        let prompt = try MarigoldV2PromptEmbeddingLoader.load(
            embedsURL: resources.promptEmbedsURL,
            maskURL: resources.promptMaskURL
        )

        let model = LoadedModel(
            rootURL: rootURL,
            checkpoint: checkpoint,
            transformer: transformer,
            vae: vae,
            prompt: prompt,
            adapterPairCount: adapterPairCount,
            vaeDecoderTensorCount: vaeDecoderTensorCount
        )
        loaded = model
        return model
    }

    private static func loadTransformer(
        resources: MarigoldV2Resources,
        config: QwenImageEditTransformerConfig
    ) throws -> MMDiT {
        let quantConfigURL = resources.rootURL.appendingPathComponent("quantization_config.json")
        if FileManager.default.fileExists(atPath: quantConfigURL.path) {
            let quantConfig = try QuantizedWeightLoader.loadConfig(from: quantConfigURL)
            let arrays = try QuantizedWeightLoader.loadArrays(
                from: resources.quantizedTransformerWeightsURL
            )
            let factory = DenseLayerFactory(arrays: arrays, quantConfig: quantConfig)
            let transformer = MMDiT(config: config, factory: factory)
            try transformer.update(
                parameters: ModuleParameters.unflattened(arrays),
                verify: [.shapeMismatch, .noUnusedKeys]
            )
            return transformer
        }

        let transformer = MMDiT(config: config)
        try Self.applyTransformerWeights(resources: resources, into: transformer)
        // Marigold trains its adapters against a 4-bit quantized transformer, so
        // the base is quantized before the adapters are installed on top.
        MLXNN.quantize(model: transformer, groupSize: 64, bits: 4) { _, module in
            if let linear = module as? Linear {
                let (_, inputDim) = linear.shape
                return inputDim % 64 == 0
            }
            return true
        }
        return transformer
    }

    private static func applyTransformerWeights(
        resources: MarigoldV2Resources,
        into model: MMDiT
    ) throws {
        let mapper = QwenImageEditGenerator.transformerWeightMapper(config: model.config)
        let indexURL = resources.transformerWeightsIndexURL
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let index = try JSONDecoder().decode(
                HFSafetensorsIndex.self,
                from: try Data(contentsOf: indexURL)
            )
            try QwenImageEditGenerator.validateTransformerCheckpointCoverage(
                rawKeys: Set(index.weightMap.keys),
                model: model
            )
            try HFSafetensorsWeightsLoader.applyShardedWeights(
                indexURL: indexURL,
                to: model,
                dtype: .bfloat16,
                verify: [.shapeMismatch],
                mapper: mapper
            )
            return
        }

        try QwenImageEditGenerator.validateTransformerCheckpointCoverage(
            rawKeys: Set(try SafetensorsStreamingLoader.metadata(
                url: resources.transformerWeightsURL
            ).keys),
            model: model
        )
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.transformerWeightsURL,
            to: model,
            dtype: .bfloat16,
            verify: [.shapeMismatch],
            mapper: mapper
        )
    }

    // MARK: - Pixel helpers

    /// Packs an image as `[1, 3, H, W]` scaled to [-1, 1], the range the Qwen VAE
    /// encoder expects.
    static func rgbNCHWSigned(_ image: MediaImage) -> MLXArray {
        let pixelCount = image.width * image.height
        var values = [Float](repeating: 0, count: pixelCount * 3)
        for pixel in 0..<pixelCount {
            let source = pixel * 4
            for channel in 0..<3 {
                values[channel * pixelCount + pixel] =
                    Float(image.rgba8[source + channel]) / 127.5 - 1
            }
        }
        return MLXArray(values).reshaped(1, 3, image.height, image.width)
    }

    /// Resamples a single-channel map back to the source resolution.
    static func resampleBilinear(
        _ values: [Float],
        width: Int,
        height: Int,
        targetWidth: Int,
        targetHeight: Int
    ) -> [Float] {
        if width == targetWidth && height == targetHeight {
            return values
        }
        var output = [Float](repeating: 0, count: targetWidth * targetHeight)
        let xScale = Double(width) / Double(targetWidth)
        let yScale = Double(height) / Double(targetHeight)
        for row in 0..<targetHeight {
            let sourceY = min(Double(height) - 1, max(0, (Double(row) + 0.5) * yScale - 0.5))
            let y0 = Int(sourceY.rounded(.down))
            let y1 = min(height - 1, y0 + 1)
            let yWeight = Float(sourceY - Double(y0))
            for column in 0..<targetWidth {
                let sourceX = min(Double(width) - 1, max(0, (Double(column) + 0.5) * xScale - 0.5))
                let x0 = Int(sourceX.rounded(.down))
                let x1 = min(width - 1, x0 + 1)
                let xWeight = Float(sourceX - Double(x0))

                let top = values[y0 * width + x0] * (1 - xWeight) + values[y0 * width + x1] * xWeight
                let bottom = values[y1 * width + x0] * (1 - xWeight) + values[y1 * width + x1] * xWeight
                output[row * targetWidth + column] = top * (1 - yWeight) + bottom * yWeight
            }
        }
        return output
    }
}
