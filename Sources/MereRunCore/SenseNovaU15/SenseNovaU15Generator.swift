import Foundation
import MLX
import MLXNN
import MLXRandom

public final class SenseNovaU15Generator: ImageGenerator {
    private struct LoadedModel {
        let rootURL: URL
        let config: SenseNovaU15Config
        let tokenizer: SenseNovaU15Tokenizer
        let model: SenseNovaU15Model
    }

    private struct Conditioning {
        let prompt: SenseNovaU15Tokenizer.EncodedPrompt
        let caches: [SenseNovaU15KVCache]
    }

    private var loaded: LoadedModel?

    public init() {}

    deinit { unload() }

    public func unload() {
        loaded = nil
        clearMemory()
    }

    public func generate(
        _ request: GenerationRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> GenerationResult {
        guard request.width > 0, request.height > 0,
              request.width.isMultiple(of: 32), request.height.isMultiple(of: 32) else {
            throw SenseNovaU15Error.invalidImageSize(width: request.width, height: request.height)
        }
        let rootURL = try resolveModelRoot(request)
        let loaded = try await loadModelIfNeeded(rootURL: rootURL, progressHandler: progressHandler)
        let references = ([request.inputImage].compactMap { $0 } + request.referenceImages)
        progressHandler?(GenerationProgress(stage: .encodingReferenceImages, stepIndex: 0, totalSteps: max(1, references.count)))
        let preparedReferences = try references.enumerated().map { index, url in
            let prepared = try SenseNovaU15ImageIO.prepareReference(
                url,
                patchSize: loaded.config.patchSize,
                downsampleRatio: loaded.config.downsampleRatio,
                imageCount: references.count
            )
            progressHandler?(GenerationProgress(
                stage: .encodingReferenceImages,
                stepIndex: index + 1,
                totalSteps: references.count
            ))
            return prepared
        }

        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: 1))
        let grids = preparedReferences.map { (height: $0.gridHeight, width: $0.gridWidth) }
        let conditionalPrompt = references.isEmpty
            ? loaded.tokenizer.textToImage(request.prompt)
            : loaded.tokenizer.imageEdit(request.prompt, grids: grids)
        let conditional = try prefill(
            prompt: conditionalPrompt,
            references: preparedReferences,
            model: loaded.model,
            config: loaded.config
        )
        let guidance = Float(request.guidanceScale)
        let guidanceCondition: Conditioning?
        if guidance > 1 {
            let prompt = references.isEmpty
                ? loaded.tokenizer.unconditional(request.negativePrompt ?? "")
                : loaded.tokenizer.imageOnly(grids: grids)
            guidanceCondition = try prefill(
                prompt: prompt,
                references: references.isEmpty ? [] : preparedReferences,
                model: loaded.model,
                config: loaded.config
            )
        } else {
            guidanceCondition = nil
        }
        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 1, totalSteps: 1))

        let seed = request.seed ?? deterministicSeed(request.prompt)
        let result = denoise(
            model: loaded.model,
            config: loaded.config,
            conditional: conditional,
            guidanceCondition: guidanceCondition,
            guidanceScale: guidance,
            width: request.width,
            height: request.height,
            steps: request.steps,
            timestepShift: request.sigmaShift ?? loaded.config.timestepShift,
            seed: seed,
            progressHandler: progressHandler
        )
        MLX.eval(result)
        progressHandler?(GenerationProgress(stage: .saving, stepIndex: 0, totalSteps: 1))
        try FileManager.default.createDirectory(
            at: request.outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try SenseNovaU15ImageIO.save(result, to: request.outputURL)
        progressHandler?(GenerationProgress(stage: .saving, stepIndex: 1, totalSteps: 1))
        clearMemory()
        return GenerationResult(outputURL: request.outputURL, seed: seed)
    }

    private func prefill(
        prompt: SenseNovaU15Tokenizer.EncodedPrompt,
        references: [SenseNovaU15ImageIO.PreparedImage],
        model: SenseNovaU15Model,
        config: SenseNovaU15Config
    ) throws -> Conditioning {
        let tokenIDs = MLXArray(prompt.tokenIDs.map(Int32.init)).reshaped(1, -1)
        let embeddings = model.languageModel.model.embed(tokenIDs)
        if !references.isEmpty {
            let visualEmbeddings = references.map { model.visionModel.embeddings($0.pixels) }
            let replacements = visualEmbeddings.count == 1
                ? visualEmbeddings[0][0, 0..., 0...]
                : MLX.concatenated(visualEmbeddings.map { $0[0, 0..., 0...] }, axis: 0)
            guard replacements.dim(0) == prompt.imageContextPositions.count else {
                throw SenseNovaU15Error.imageTokenMismatch(
                    expected: prompt.imageContextPositions.count,
                    actual: replacements.dim(0)
                )
            }
            embeddings[0, MLXArray(prompt.imageContextPositions.map(Int32.init)), 0...] = replacements
        }
        let indexes = indexesArray(prompt)
        let caches = (0..<config.llmConfig.numberOfHiddenLayers).map { _ in SenseNovaU15KVCache() }
        let hidden = model.languageModel.model.forward(
            embeddings: embeddings,
            indexes: indexes,
            expert: .understanding,
            caches: caches,
            updateCache: true
        )
        MLX.eval([hidden] + caches.flatMap { [$0.keys, $0.values].compactMap { $0 } })
        Memory.clearCache()
        return Conditioning(prompt: prompt, caches: caches)
    }

    private func denoise(
        model: SenseNovaU15Model,
        config: SenseNovaU15Config,
        conditional: Conditioning,
        guidanceCondition: Conditioning?,
        guidanceScale: Float,
        width: Int,
        height: Int,
        steps: Int,
        timestepShift: Float,
        seed: UInt64,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) -> MLXArray {
        let tokenHeight = height / 32
        let tokenWidth = width / 32
        let imageTokenCount = tokenHeight * tokenWidth
        let noiseScale = SenseNovaU15Scheduler.noiseScale(
            imageTokenCount: imageTokenCount,
            baseImageTokenCount: config.noiseScaleBaseImageSequenceLength,
            baseScale: config.noiseScale,
            maximum: config.noiseScaleMaxValue,
            mode: config.noiseScaleMode
        )
        let initial = MLXRandom.normal(
            [1, height, width, 3],
            key: MLXRandom.key(seed)
        ).asType(.bfloat16) * noiseScale
        var imageTokens = Self.patchify(initial, patchSize: 32)
        let timesteps = SenseNovaU15Scheduler.timesteps(steps: steps, shift: timestepShift)
        let conditionalIndexes = generationIndexes(
            tokenHeight: tokenHeight,
            tokenWidth: tokenWidth,
            prefixTime: (conditional.prompt.timeIndexes.max() ?? -1) + 1
        )
        let guidanceIndexes = guidanceCondition.map {
            generationIndexes(
                tokenHeight: tokenHeight,
                tokenWidth: tokenWidth,
                prefixTime: ($0.prompt.timeIndexes.max() ?? -1) + 1
            )
        }
        let noiseEmbedding: MLXArray?
        if config.addNoiseScaleEmbedding {
            let normalizedNoiseScale = noiseScale / config.noiseScaleMaxValue
            let noiseScaleValues = MLXArray(
                [Float](repeating: normalizedNoiseScale, count: imageTokenCount)
            )
            noiseEmbedding = model.flowModules.noiseScaleEmbedder(noiseScaleValues)
        } else {
            noiseEmbedding = nil
        }
        MLX.eval([noiseEmbedding].compactMap { $0 })
        for step in 0..<steps {
            progressHandler?(GenerationProgress(stage: .denoising, stepIndex: step, totalSteps: steps))
            let timestep = timesteps[step]
            let pixels = Self.unpatchify(
                imageTokens,
                patchSize: 32,
                height: height,
                width: width
            )
            let timestepValues = MLXArray([Float](repeating: timestep, count: imageTokenCount))
            var timeEmbeddings = model.flowModules.timestepEmbedder(timestepValues)
            if let noiseEmbedding { timeEmbeddings = timeEmbeddings + noiseEmbedding }
            let imageEmbeddings = model.flowModules.generationVisionModel.embeddings(pixels)
                + timeEmbeddings.reshaped(1, imageTokenCount, -1)
            let conditionalVelocity = velocity(
                model: model,
                imageEmbeddings: imageEmbeddings,
                imageTokens: imageTokens,
                indexes: conditionalIndexes,
                caches: conditional.caches,
                timestep: timestep,
                tEpsilon: config.tEpsilon,
                tokenHeight: tokenHeight,
                tokenWidth: tokenWidth
            )
            let predictedVelocity: MLXArray
            if let guidanceCondition, let guidanceIndexes, guidanceScale > 1 {
                let baselineVelocity = velocity(
                    model: model,
                    imageEmbeddings: imageEmbeddings,
                    imageTokens: imageTokens,
                    indexes: guidanceIndexes,
                    caches: guidanceCondition.caches,
                    timestep: timestep,
                    tEpsilon: config.tEpsilon,
                    tokenHeight: tokenHeight,
                    tokenWidth: tokenWidth
                )
                predictedVelocity = baselineVelocity
                    + guidanceScale * (conditionalVelocity - baselineVelocity)
            } else {
                predictedVelocity = conditionalVelocity
            }
            imageTokens = imageTokens + (timesteps[step + 1] - timestep) * predictedVelocity
            MLX.eval(imageTokens)
            Memory.clearCache()
        }
        progressHandler?(GenerationProgress(stage: .denoising, stepIndex: steps, totalSteps: steps))
        return Self.unpatchify(imageTokens, patchSize: 32, height: height, width: width)
    }

    private func velocity(
        model: SenseNovaU15Model,
        imageEmbeddings: MLXArray,
        imageTokens: MLXArray,
        indexes: MLXArray,
        caches: [SenseNovaU15KVCache],
        timestep: Float,
        tEpsilon: Float,
        tokenHeight: Int,
        tokenWidth: Int
    ) -> MLXArray {
        let hidden = model.languageModel.model.forward(
            embeddings: imageEmbeddings,
            indexes: indexes,
            expert: .generation,
            caches: caches,
            updateCache: false
        )
        let predictedPixels = model.flowModules.flowHead(
            hidden,
            tokenHeight: tokenHeight,
            tokenWidth: tokenWidth
        )
        let predictedTokens = Self.patchify(predictedPixels, patchSize: 32)
        return (predictedTokens - imageTokens) / max(1 - timestep, tEpsilon)
    }

    private func indexesArray(_ prompt: SenseNovaU15Tokenizer.EncodedPrompt) -> MLXArray {
        MLXArray(
            prompt.timeIndexes + prompt.heightIndexes + prompt.widthIndexes,
            [3, prompt.tokenIDs.count]
        )
    }

    private func generationIndexes(
        tokenHeight: Int,
        tokenWidth: Int,
        prefixTime: Int32
    ) -> MLXArray {
        let count = tokenHeight * tokenWidth
        let time = [Int32](repeating: prefixTime, count: count)
        let height = (0..<count).map { Int32($0 / tokenWidth) }
        let width = (0..<count).map { Int32($0 % tokenWidth) }
        return MLXArray(time + height + width, [3, count])
    }

    static func patchify(_ pixels: MLXArray, patchSize: Int) -> MLXArray {
        let batch = pixels.dim(0)
        let height = pixels.dim(1) / patchSize
        let width = pixels.dim(2) / patchSize
        return pixels
            .reshaped(batch, height, patchSize, width, patchSize, 3)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(batch, height * width, patchSize * patchSize * 3)
    }

    static func unpatchify(
        _ patches: MLXArray,
        patchSize: Int,
        height: Int,
        width: Int
    ) -> MLXArray {
        let batch = patches.dim(0)
        let gridHeight = height / patchSize
        let gridWidth = width / patchSize
        return patches
            .reshaped(batch, gridHeight, gridWidth, patchSize, patchSize, 3)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(batch, height, width, 3)
    }

    private func loadModelIfNeeded(
        rootURL: URL,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> LoadedModel {
        if let loaded, loaded.rootURL == rootURL { return loaded }
        unload()
        let resources = SenseNovaU15Resources(rootURL: rootURL)
        let missing = resources.validate()
        guard missing.isEmpty else { throw SenseNovaU15Error.missingModelFiles(missing) }
        progressHandler?(GenerationProgress(stage: .loadingModel, stepIndex: 0, totalSteps: 1))
        let config = try SenseNovaU15Config.load(from: resources)
        let tokenizer = try await SenseNovaU15Tokenizer.load(from: resources)
        let model = SenseNovaU15Model(config: config)
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.weightsIndexURL,
            singleURL: resources.singleWeightsURL,
            to: model,
            dtype: .bfloat16,
            verify: .shapeMismatch,
            mapper: Self.mapWeight,
            progressHandler: { shard in
                progressHandler?(GenerationProgress(
                    stage: .loadingTransformer,
                    stepIndex: shard.shardIndex + 1,
                    totalSteps: shard.shardCount
                ))
            }
        )
        let loaded = LoadedModel(rootURL: rootURL, config: config, tokenizer: tokenizer, model: model)
        self.loaded = loaded
        progressHandler?(GenerationProgress(stage: .loadingModel, stepIndex: 1, totalSteps: 1))
        return loaded
    }

    private static func mapWeight(_ key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key == "language_model.lm_head.weight" { return [] }
        let isConvolution = key.hasSuffix("patch_embedding.weight")
            || key.hasSuffix("dense_embedding.weight")
            || key == "fm_modules.fm_head.conv1.weight"
            || key == "fm_modules.fm_head.conv2.weight"
        return [(key, isConvolution ? HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(value) : value)]
    }

    private func resolveModelRoot(_ request: GenerationRequest) throws -> URL {
        if let model = request.model {
            let url = URL(fileURLWithPath: model).standardizedFileURL
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        if let installed = ManagedModelResolver.resolveInstalledModel(id: SenseNovaU15Resources.modelID) {
            return installed
        }
        throw ModelResolver.ResolverError.modelNotFound(
            .senseNovaU15,
            searched: [MereRunModelPaths.modelDir(SenseNovaU15Resources.modelID)],
            upstreamRepoId: SenseNovaU15Resources.repository
        )
    }

    private func deterministicSeed(_ prompt: String) -> UInt64 {
        prompt.utf8.reduce(UInt64(0xcbf2_9ce4_8422_2325)) { ($0 ^ UInt64($1)) &* 0x100_0000_01b3 }
    }

    private func clearMemory() {
        MLX.eval(MLXArray([]))
        Memory.clearCache()
    }
}
