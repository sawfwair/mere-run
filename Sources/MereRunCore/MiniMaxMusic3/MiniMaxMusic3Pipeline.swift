import Foundation
import MLX
import MLXRandom

public struct MiniMaxMusic3GenerationOptions: Sendable {
    public var caption: String
    public var lyrics: String
    public var durationSeconds: Float
    public var minimumFrames: Int?
    public var maximumFrames: Int?
    public var inferenceSteps: Int
    public var seed: UInt64
    public var guidanceScale: Float

    public init(
        caption: String,
        lyrics: String,
        durationSeconds: Float = 60,
        minimumFrames: Int? = nil,
        maximumFrames: Int? = nil,
        inferenceSteps: Int = 30,
        seed: UInt64 = 0,
        guidanceScale: Float = 1.7
    ) {
        self.caption = caption
        self.lyrics = lyrics
        self.durationSeconds = durationSeconds
        self.minimumFrames = minimumFrames
        self.maximumFrames = maximumFrames
        self.inferenceSteps = inferenceSteps
        self.seed = seed
        self.guidanceScale = guidanceScale
    }
}

public struct MiniMaxMusic3GenerationResult {
    public let waveform: MLXArray
    public let sampleRate: Int
    public let frameCount: Int
}

public enum MiniMaxMusic3Progress: Sendable, Equatable {
    case semantic(frame: Int, maximum: Int)
    case denoise(chunk: Int, chunkCount: Int, step: Int, stepCount: Int)
    case decode(chunk: Int, chunkCount: Int)
}

public final class MiniMaxMusic3Pipeline {
    private let residentModels: MiniMaxMusic3Models?
    private let resources: MiniMaxMusic3Resources?
    public let loadingStrategy: MiniMaxMusic3LoadingStrategy
    public let performanceMode: MiniMaxMusic3PerformanceMode

    public init(models: MiniMaxMusic3Models) {
        self.residentModels = models
        self.resources = nil
        self.loadingStrategy = .resident
        self.performanceMode = .reference
    }

    public init(
        resources: MiniMaxMusic3Resources,
        loadingStrategy: MiniMaxMusic3LoadingStrategy = .staged,
        performanceMode: MiniMaxMusic3PerformanceMode = .optimized
    ) throws {
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw MiniMaxMusic3Error.missingResources(missing)
        }
        self.resources = resources
        self.loadingStrategy = loadingStrategy
        self.performanceMode = performanceMode
        self.residentModels = loadingStrategy == .resident
            ? try MiniMaxMusic3ModelLoader.load(
                from: resources,
                performanceMode: performanceMode
            )
            : nil
    }

    public func generate(
        options: MiniMaxMusic3GenerationOptions,
        progress: (@Sendable (MiniMaxMusic3Progress) -> Void)? = nil
    ) throws -> MiniMaxMusic3GenerationResult {
        guard !options.caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MiniMaxMusic3Error.invalidPrompt("the caption is empty")
        }
        guard !options.lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MiniMaxMusic3Error.invalidPrompt("the lyrics are empty")
        }
        guard options.durationSeconds > 0 else {
            throw MiniMaxMusic3Error.invalidPrompt("duration must be positive")
        }
        guard options.inferenceSteps > 0 else {
            throw MiniMaxMusic3Error.invalidPrompt("inference steps must be positive")
        }
        if let maximumFrames = options.maximumFrames,
           !(1...MiniMaxMusic3Prompt.maxAudioFrames).contains(maximumFrames)
        {
            throw MiniMaxMusic3Error.invalidPrompt(
                "maximum frames must be between 1 and \(MiniMaxMusic3Prompt.maxAudioFrames)"
            )
        }
        if let minimumFrames = options.minimumFrames,
           !(1...MiniMaxMusic3Prompt.maxAudioFrames).contains(minimumFrames)
        {
            throw MiniMaxMusic3Error.invalidPrompt(
                "minimum frames must be between 1 and \(MiniMaxMusic3Prompt.maxAudioFrames)"
            )
        }
        if let minimumFrames = options.minimumFrames,
           let maximumFrames = options.maximumFrames,
           minimumFrames > maximumFrames
        {
            throw MiniMaxMusic3Error.invalidPrompt(
                "minimum frames cannot exceed maximum frames"
            )
        }

        let generator = MLXRandom.RandomState(seed: options.seed)
        let frameHiddens = try autoregressiveStage(
            options: options,
            generator: generator,
            progress: progress
        )
        if loadingStrategy == .staged {
            MLX.Memory.clearCache()
        }
        let chunks = try flowStage(
            frameHiddens: frameHiddens,
            steps: options.inferenceSteps,
            guidanceScale: options.guidanceScale,
            generator: generator,
            progress: progress
        )
        if loadingStrategy == .staged {
            MLX.Memory.clearCache()
        }
        let decoded = try vocoderStage(chunks: chunks, progress: progress)
        if loadingStrategy == .staged {
            MLX.Memory.clearCache()
        }
        let waveform = decoded.waveform
        MLX.eval(waveform)
        return MiniMaxMusic3GenerationResult(
            waveform: waveform,
            sampleRate: decoded.sampleRate,
            frameCount: frameHiddens.dim(1)
        )
    }

    private func autoregressiveStage(
        options: MiniMaxMusic3GenerationOptions,
        generator: MLXRandom.RandomState,
        progress: (@Sendable (MiniMaxMusic3Progress) -> Void)?
    ) throws -> MLXArray {
        let models = try residentModels.map {
            MiniMaxMusic3AutoregressiveModels(
                languageModel: $0.languageModel,
                depthDecoder: $0.depthDecoder,
                tokenizer: $0.tokenizer
            )
        } ?? MiniMaxMusic3ModelLoader.loadAutoregressive(
            from: requiredResources(),
            performanceMode: performanceMode
        )
        let tokenIDs = try promptTokenIDs(
            caption: options.caption,
            lyrics: options.lyrics,
            tokenizer: models.tokenizer
        )
        let frameHiddens = try generateSemanticFrames(
            textIDs: tokenIDs,
            durationSeconds: options.durationSeconds,
            minimumFrames: options.minimumFrames,
            maximumFrames: options.maximumFrames,
            languageModel: models.languageModel,
            depthDecoder: models.depthDecoder,
            generator: generator,
            progress: progress
        )
        MLX.eval(frameHiddens)
        return frameHiddens
    }

    private func flowStage(
        frameHiddens: MLXArray,
        steps: Int,
        guidanceScale: Float,
        generator: MLXRandom.RandomState,
        progress: (@Sendable (MiniMaxMusic3Progress) -> Void)?
    ) throws -> [MLXArray] {
        let models = try residentModels.map {
            MiniMaxMusic3FlowModels(
                conditionEncoder: $0.conditionEncoder,
                transformer: $0.transformer
            )
        } ?? MiniMaxMusic3ModelLoader.loadFlow(
            from: requiredResources(),
            performanceMode: performanceMode
        )
        let chunks = denoise(
            frameHiddens: frameHiddens,
            steps: steps,
            guidanceScale: guidanceScale,
            conditionEncoder: models.conditionEncoder,
            transformer: models.transformer,
            generator: generator,
            progress: progress
        )
        MLX.eval(chunks)
        return chunks
    }

    private func vocoderStage(
        chunks: [MLXArray],
        progress: (@Sendable (MiniMaxMusic3Progress) -> Void)?
    ) throws -> (waveform: MLXArray, sampleRate: Int) {
        let vocoder = try residentModels?.vocoder
            ?? MiniMaxMusic3ModelLoader.loadVocoder(from: requiredResources())
        let waveform = decode(chunks: chunks, vocoder: vocoder, progress: progress)
        MLX.eval(waveform)
        return (waveform, vocoder.configuration.samplingRate)
    }

    private func requiredResources() throws -> MiniMaxMusic3Resources {
        guard let resources else {
            throw MiniMaxMusic3Error.invalidPrompt("staged resources are unavailable")
        }
        return resources
    }

    private func promptTokenIDs(
        caption: String,
        lyrics: String,
        tokenizer: ACEStep5HzLMTokenizer
    ) throws -> MLXArray {
        let prompt = MiniMaxMusic3Prompt.assemble(caption: caption, lyrics: lyrics)
        let conditional = tokenizer.encode(prompt, addSpecialTokens: false)
        guard conditional.count <= MiniMaxMusic3Prompt.maxPromptTokens else {
            throw MiniMaxMusic3Error.invalidPrompt(
                "the assembled prompt has \(conditional.count) tokens; maximum is \(MiniMaxMusic3Prompt.maxPromptTokens)"
            )
        }
        var unconditional = conditional
        if unconditional.count > 3 {
            for index in 1..<(unconditional.count - 2) {
                unconditional[index] = MiniMaxMusic3Prompt.audioCFGTokenID
            }
        }
        return MLXArray((conditional + unconditional).map(Int32.init)).reshaped(2, conditional.count)
    }

    private func generateSemanticFrames(
        textIDs: MLXArray,
        durationSeconds: Float,
        minimumFrames explicitMinimumFrames: Int?,
        maximumFrames explicitMaximumFrames: Int?,
        languageModel: MiniMaxMusic3LanguageModel,
        depthDecoder: MiniMaxMusic3DepthDecoder,
        generator: MLXRandom.RandomState,
        progress: (@Sendable (MiniMaxMusic3Progress) -> Void)?
    ) throws -> MLXArray {
        let caches = languageModel.makeCache()
        let textEmbeddings = languageModel.embed(tokenIDs: textIDs)
        var lastHidden = languageModel.hidden(
            embeddings: textEmbeddings,
            cache: caches,
            lastPositionOnly: true
        ).squeezed(axis: 1)
        let minimumFrames = explicitMinimumFrames ?? 0
        let durationFrames = Int(durationSeconds * Float(MiniMaxMusic3Prompt.frameRate))
        let requestedFrames = explicitMaximumFrames
            ?? max(durationFrames, minimumFrames)
        let maximumFrames = min(requestedFrames, MiniMaxMusic3Prompt.maxAudioFrames)
        guard maximumFrames > 0 else {
            throw MiniMaxMusic3Error.invalidPrompt("duration is shorter than one 25 Hz audio frame")
        }
        guard minimumFrames <= maximumFrames else {
            throw MiniMaxMusic3Error.invalidPrompt(
                "minimum duration requires more than \(maximumFrames) audio frames"
            )
        }

        var frameHiddens: [MLXArray] = []
        frameHiddens.reserveCapacity(maximumFrames)
        for frameIndex in 0...maximumFrames {
            let logits = languageModel.logits(lastHidden)
            let allowEnd = frameHiddens.count >= minimumFrames
            let sampled = performanceMode == .reference
                ? sampleSemanticReference(logits: logits, allowEnd: allowEnd, generator: generator)
                : sampleSemantic(
                    logits: logits,
                    compactHead: languageModel.usesCompactSemanticHead,
                    allowEnd: allowEnd,
                    generator: generator
                )
            let sampledID = sampled.item(Int.self)
            if sampledID == MiniMaxMusic3Prompt.audioEndTokenID {
                break
            }

            let semantic = sampled.asType(.int32) - MLXArray(Int32(MiniMaxMusic3Prompt.audioCodeOffset))
            let duplicatedSemantic = MLX.repeated(semantic.reshaped(1), count: 2, axis: 0)
            let depth = generateDepthCodes(
                lastHidden: lastHidden,
                semanticCode: duplicatedSemantic,
                languageModel: languageModel,
                depthDecoder: depthDecoder,
                generator: generator
            )
            var conditioningToEvaluate: MLXArray?
            if frameIndex > 0 {
                let conditioning = MLX.concatenated([lastHidden[0..<1], depth.hidden], axis: -1)
                if performanceMode == .reference {
                    MLX.eval(conditioning)
                } else {
                    conditioningToEvaluate = conditioning
                }
                frameHiddens.append(conditioning)
                progress?(.semantic(frame: frameHiddens.count, maximum: maximumFrames))
                if frameHiddens.count >= maximumFrames {
                    if performanceMode != .reference {
                        MLX.eval(conditioning)
                    }
                    break
                }
            }

            let feedback = embedAudioFrame(
                depth.codes,
                languageModel: languageModel,
                depthDecoder: depthDecoder
            )
            lastHidden = languageModel.hidden(
                embeddings: feedback,
                cache: caches,
                lastPositionOnly: true
            ).squeezed(axis: 1)
            if performanceMode == .reference {
                MLX.eval(lastHidden)
            } else if let conditioningToEvaluate {
                MLX.eval(lastHidden, conditioningToEvaluate)
            } else {
                MLX.eval(lastHidden)
            }
        }

        guard !frameHiddens.isEmpty else {
            throw MiniMaxMusic3Error.generatedNoFrames
        }
        return MLX.stacked(frameHiddens, axis: 1)
    }

    private func sampleSemanticReference(
        logits: MLXArray,
        allowEnd: Bool,
        generator: MLXRandom.RandomState
    ) -> MLXArray {
        let guided = allowEnd
            ? Self.guidedSemanticLogitsReference(logits)
            : Self.guidedSemanticLogits(logits, allowEnd: false)
        return sampleTopK(guided[0], generator: generator)
    }

    /// Keep the released full-vocabulary graph byte-for-byte available for
    /// seeded parity checks. Even graph-equivalent mask construction can alter
    /// lazy evaluation and therefore the autoregressive sampling trajectory.
    private static func guidedSemanticLogitsReference(_ logits: MLXArray) -> MLXArray {
        let vocabulary = logits.dim(-1)
        let tokenIndices = MLXArray((0..<vocabulary).map(Int32.init)).reshaped(1, vocabulary)
        let semanticStart = MLXArray(Int32(MiniMaxMusic3Prompt.audioCodeOffset))
        let semanticEnd = MLXArray(Int32(
            MiniMaxMusic3Prompt.audioCodeOffset + MiniMaxMusic3Prompt.semanticVocabularySize
        ))
        let allowedSemantic = (tokenIndices .>= semanticStart) .&& (tokenIndices .< semanticEnd)
        let allowedEnd = tokenIndices .== MLXArray(Int32(MiniMaxMusic3Prompt.audioEndTokenID))
        let allowed = allowedSemantic .|| allowedEnd
        let masked = MLX.where(allowed, logits.asType(.float32), MLXArray(-Float.infinity))
        let conditional = masked[0..<1]
        let unconditional = masked[1..<2]
        var guided = unconditional + (conditional - unconditional) * MLXArray(Float(1.5))
        let firstTopIndex = vocabulary - min(50, vocabulary)
        let topIndices = MLX.argPartition(
            conditional,
            kth: firstTopIndex,
            axis: -1
        )[0, firstTopIndex...]
        let threshold = conditional[0].take(topIndices, axis: -1).min()
        guided = MLX.where(conditional .< threshold, MLXArray(-Float.infinity), guided)
        return MLX.where(allowed, guided, MLXArray(-Float.infinity))
    }

    private func sampleSemantic(
        logits: MLXArray,
        compactHead: Bool,
        allowEnd: Bool,
        generator: MLXRandom.RandomState
    ) -> MLXArray {
        let guided = Self.guidedSemanticLogits(logits, allowEnd: allowEnd)
        let sampled = sampleTopK(guided[0], generator: generator)
        guard compactHead else { return sampled }
        return MLX.where(
            sampled .== MLXArray(Int32(0)),
            MLXArray(Int32(MiniMaxMusic3Prompt.audioEndTokenID)),
            sampled + MLXArray(Int32(MiniMaxMusic3Prompt.audioCodeOffset - 1))
        ).asType(.int32)
    }

    static func guidedSemanticLogits(
        _ logits: MLXArray,
        allowEnd: Bool = true
    ) -> MLXArray {
        let vocabulary = logits.dim(-1)
        if vocabulary == MiniMaxMusic3Prompt.semanticVocabularySize + 1 {
            let eligible: MLXArray
            if allowEnd {
                eligible = logits.asType(.float32)
            } else {
                eligible = MLX.concatenated(
                    [
                        MLXArray.full(
                            [logits.dim(0), 1],
                            values: MLXArray(-Float.infinity)
                        ),
                        logits[0..., 1...].asType(.float32),
                    ],
                    axis: -1
                )
            }
            return guidedTopK(eligible)
        }

        let tokenIndices = MLX.arange(vocabulary, dtype: .int32).reshaped(1, vocabulary)
        let semanticStart = MLXArray(Int32(MiniMaxMusic3Prompt.audioCodeOffset))
        let semanticEnd = MLXArray(Int32(
            MiniMaxMusic3Prompt.audioCodeOffset + MiniMaxMusic3Prompt.semanticVocabularySize
        ))
        let allowedSemantic = (tokenIndices .>= semanticStart) .&& (tokenIndices .< semanticEnd)
        let allowedEnd = allowEnd
            ? tokenIndices .== MLXArray(Int32(MiniMaxMusic3Prompt.audioEndTokenID))
            : MLXArray.zeros(tokenIndices.shape, dtype: .bool)
        let allowed = allowedSemantic .|| allowedEnd

        // MiniMax masks the language-model vocabulary before choosing the
        // conditional branch's top candidates. Text-token logits must never
        // influence the audio-code threshold.
        let masked = MLX.where(allowed, logits.asType(.float32), MLXArray(-Float.infinity))
        return guidedTopK(masked, allowed: allowed)
    }

    private static func guidedTopK(
        _ logits: MLXArray,
        allowed: MLXArray? = nil
    ) -> MLXArray {
        let vocabulary = logits.dim(-1)
        let conditional = logits[0..<1]
        let unconditional = logits[1..<2]
        var guided = unconditional + (conditional - unconditional) * MLXArray(Float(1.5))
        let firstTopIndex = vocabulary - min(50, vocabulary)
        let topIndices = MLX.argPartition(conditional, kth: firstTopIndex, axis: -1)[0, firstTopIndex...]
        let threshold = conditional[0].take(topIndices, axis: -1).min()
        guided = MLX.where(conditional .< threshold, MLXArray(-Float.infinity), guided)
        guard let allowed else { return guided }
        return MLX.where(allowed, guided, MLXArray(-Float.infinity))
    }

    private func generateDepthCodes(
        lastHidden: MLXArray,
        semanticCode: MLXArray,
        languageModel: MiniMaxMusic3LanguageModel,
        depthDecoder: MiniMaxMusic3DepthDecoder,
        generator: MLXRandom.RandomState
    ) -> (codes: MLXArray, hidden: MLXArray) {
        var sequence = [depthDecoder.projection(lastHidden).expandedDimensions(axis: 1)]
        let semanticEmbedding = languageModel.embed(
            tokenIDs: semanticCode + MLXArray(Int32(MiniMaxMusic3Prompt.audioCodeOffset))
        )
        sequence.append(depthDecoder.projection(semanticEmbedding).expandedDimensions(axis: 1))
        var codes = [semanticCode]
        var hiddenParts: [MLXArray] = []
        let caches = performanceMode.usesOptimizedGraph
            ? depthDecoder.makeCache()
            : nil
        for codebook in 1..<depthDecoder.configuration.numCodebooks {
            let input = if caches == nil || codebook == 1 {
                MLX.concatenated(sequence, axis: 1)
            } else {
                sequence[sequence.count - 1]
            }
            let hidden = depthDecoder(input, cache: caches)[0..., -1, 0...]
            hiddenParts.append(hidden[0..<1])
            let logits = depthDecoder.logits(hidden, codebookIndex: codebook - 1)
            let conditional = logits[0..<1].asType(.float32)
            let unconditional = logits[1..<2].asType(.float32)
            let guided = unconditional + (conditional - unconditional) * MLXArray(Float(1.5))
            let sampled = sampleTopK(guided[0], generator: generator)
            let duplicated = MLX.repeated(sampled.reshaped(1), count: 2, axis: 0)
            codes.append(duplicated)
            if codebook < depthDecoder.configuration.numCodebooks - 1 {
                let embedding = depthDecoder.embedResidualCodes(
                    duplicated,
                    codebookIndex: codebook - 1
                )
                sequence.append(depthDecoder.projection(embedding).expandedDimensions(axis: 1))
            }
        }
        return (
            MLX.stacked(codes, axis: 1),
            MLX.concatenated(hiddenParts, axis: -1)
        )
    }

    private func sampleTopK(
        _ logits: MLXArray,
        generator: MLXRandom.RandomState
    ) -> MLXArray {
        let finite = MLX.nanToNum(
            logits.asType(.float32),
            nan: -1e9,
            posInf: 1e9,
            negInf: -1e9
        )
        return MLXRandom.categorical(
            applyingTopK(finite, topK: min(50, finite.dim(-1))),
            key: generator
        ).asType(.int32)
    }

    private func embedAudioFrame(
        _ frameCodes: MLXArray,
        languageModel: MiniMaxMusic3LanguageModel,
        depthDecoder: MiniMaxMusic3DepthDecoder
    ) -> MLXArray {
        var embedding = languageModel.embed(
            tokenIDs: frameCodes[0..., 0] + MLXArray(Int32(MiniMaxMusic3Prompt.audioCodeOffset))
        )
        for codebook in 1..<depthDecoder.configuration.numCodebooks {
            embedding = embedding + depthDecoder.embedResidualCodes(
                frameCodes[0..., codebook],
                codebookIndex: codebook - 1
            ).asType(embedding.dtype)
        }
        let scale = 1 / Float(depthDecoder.configuration.numCodebooks).squareRoot()
        return (embedding * MLXArray(scale)).expandedDimensions(axis: 1)
    }

    private func denoise(
        frameHiddens: MLXArray,
        steps: Int,
        guidanceScale: Float,
        conditionEncoder: MiniMaxMusic3ConditionEncoder,
        transformer: MiniMaxMusic3Transformer,
        generator: MLXRandom.RandomState,
        progress: (@Sendable (MiniMaxMusic3Progress) -> Void)?
    ) -> [MLXArray] {
        if performanceMode == .reference {
            return denoiseReference(
                frameHiddens: frameHiddens,
                steps: steps,
                guidanceScale: guidanceScale,
                conditionEncoder: conditionEncoder,
                transformer: transformer,
                generator: generator,
                progress: progress
            )
        }
        let starts = MiniMaxMusic3Prompt.chunkStarts(frameCount: frameHiddens.dim(1))
        var previousLatent: MLXArray?
        var previousCondition: MLXArray?
        var chunks: [MLXArray] = []
        chunks.reserveCapacity(starts.count)

        for (chunkIndex, start) in starts.enumerated() {
            let end = min(start + 200, frameHiddens.dim(1))
            var condition = conditionEncoder(frameHiddens[0..., start..<end, 0...])
                .asType(.bfloat16)
            let overlap = min(previousLatent?.dim(2) ?? 0, condition.dim(1))
            if overlap > 0, let previousCondition {
                condition = MLX.concatenated(
                    [previousCondition[0..., 0..<overlap, 0...], condition[0..., overlap..., 0...]],
                    axis: 1
                )
            }
            var latents = MLXRandom.normal([
                1,
                transformer.configuration.inChannels,
                condition.dim(1),
            ], key: generator).asType(condition.dtype)
            let noisePrompt = overlap > 0 ? latents[0..., 0..., 0..<overlap] : nil
            let zeroCondition = MLXArray.zeros(condition.shape, dtype: condition.dtype)
            let rotary = transformer.rotaryCache(
                latentLength: condition.dim(1),
                dtype: condition.dtype
            )
            let stepSize = MLXArray(1 / Float(steps)).asType(condition.dtype)

            for step in 0..<steps {
                let time = Float(step) / Float(steps)
                if overlap > 0, let previousLatent, let noisePrompt {
                    let blended = (1 - (1 - 1e-6) * time) * noisePrompt
                        + time * previousLatent[0..., 0..., 0..<overlap]
                    latents = MLX.concatenated(
                        [blended, latents[0..., 0..., overlap...]],
                        axis: 2
                    )
                }
                let timestep = MLXArray([time]).asType(latents.dtype)
                let batchedLatents = MLX.concatenated([latents, latents], axis: 0)
                let batchedTimestep = MLX.repeated(timestep, count: 2, axis: 0)
                let batchedCondition = MLX.concatenated([condition, zeroCondition], axis: 0)
                let velocities = transformer(
                    latents: batchedLatents,
                    timestep: batchedTimestep,
                    condition: batchedCondition,
                    rotary: rotary
                )
                let conditional = velocities[0..<1]
                let unconditional = velocities[1..<2]
                let velocity = unconditional + (conditional - unconditional) * MLXArray(guidanceScale)
                latents = latents + velocity * stepSize
                MLX.eval(latents)
                progress?(.denoise(
                    chunk: chunkIndex + 1,
                    chunkCount: starts.count,
                    step: step + 1,
                    stepCount: steps
                ))
            }
            if overlap > 0, let previousLatent {
                latents = MLX.concatenated(
                    [previousLatent[0..., 0..., 0..<overlap], latents[0..., 0..., overlap...]],
                    axis: 2
                )
            }
            let overlapStart = max(0, latents.dim(2) - 344)
            let overlapEnd = max(overlapStart, latents.dim(2) - 172)
            previousLatent = latents[0..., 0..., overlapStart..<overlapEnd]
            previousCondition = condition[0..., overlapStart..<overlapEnd, 0...]
            MLX.eval(latents, previousLatent!, previousCondition!)
            chunks.append(latents)
            MLX.Memory.clearCache()
        }
        return chunks
    }

    private func denoiseReference(
        frameHiddens: MLXArray,
        steps: Int,
        guidanceScale: Float,
        conditionEncoder: MiniMaxMusic3ConditionEncoder,
        transformer: MiniMaxMusic3Transformer,
        generator: MLXRandom.RandomState,
        progress: (@Sendable (MiniMaxMusic3Progress) -> Void)?
    ) -> [MLXArray] {
        let starts = MiniMaxMusic3Prompt.chunkStarts(frameCount: frameHiddens.dim(1))
        var previousLatent: MLXArray?
        var previousCondition: MLXArray?
        var chunks: [MLXArray] = []
        chunks.reserveCapacity(starts.count)

        for (chunkIndex, start) in starts.enumerated() {
            let end = min(start + 200, frameHiddens.dim(1))
            var condition = conditionEncoder(frameHiddens[0..., start..<end, 0...])
                .asType(.bfloat16)
            let overlap = min(previousLatent?.dim(2) ?? 0, condition.dim(1))
            if overlap > 0, let previousCondition {
                condition = MLX.concatenated(
                    [previousCondition[0..., 0..<overlap, 0...], condition[0..., overlap..., 0...]],
                    axis: 1
                )
            }
            var latents = MLXRandom.normal([
                1,
                transformer.configuration.inChannels,
                condition.dim(1),
            ], key: generator).asType(condition.dtype)
            let noisePrompt = overlap > 0 ? latents[0..., 0..., 0..<overlap] : nil

            for step in 0..<steps {
                let time = Float(step) / Float(steps)
                if overlap > 0, let previousLatent, let noisePrompt {
                    let blended = (1 - (1 - 1e-6) * time) * noisePrompt
                        + time * previousLatent[0..., 0..., 0..<overlap]
                    latents = MLX.concatenated(
                        [blended, latents[0..., 0..., overlap...]],
                        axis: 2
                    )
                }
                let timestep = MLXArray([time]).asType(latents.dtype)
                let conditional = transformer(
                    latents: latents,
                    timestep: timestep,
                    condition: condition
                )
                let unconditional = transformer(
                    latents: latents,
                    timestep: timestep,
                    condition: MLXArray.zeros(condition.shape, dtype: condition.dtype)
                )
                let velocity = unconditional + (conditional - unconditional) * MLXArray(guidanceScale)
                latents = latents + velocity * MLXArray(1 / Float(steps))
                MLX.eval(latents)
                progress?(.denoise(
                    chunk: chunkIndex + 1,
                    chunkCount: starts.count,
                    step: step + 1,
                    stepCount: steps
                ))
            }
            if overlap > 0, let previousLatent {
                latents = MLX.concatenated(
                    [previousLatent[0..., 0..., 0..<overlap], latents[0..., 0..., overlap...]],
                    axis: 2
                )
            }
            let overlapStart = max(0, latents.dim(2) - 344)
            let overlapEnd = max(overlapStart, latents.dim(2) - 172)
            previousLatent = latents[0..., 0..., overlapStart..<overlapEnd]
            previousCondition = condition[0..., overlapStart..<overlapEnd, 0...]
            MLX.eval(latents, previousLatent!, previousCondition!)
            chunks.append(latents)
            MLX.Memory.clearCache()
        }
        return chunks
    }

    private func decode(
        chunks: [MLXArray],
        vocoder: MiniMaxMusic3Vocoder,
        progress: (@Sendable (MiniMaxMusic3Progress) -> Void)?
    ) -> MLXArray {
        var waveforms: [MLXArray] = []
        waveforms.reserveCapacity(chunks.count)
        for (index, chunk) in chunks.enumerated() {
            let waveform = vocoder(chunk.asType(.bfloat16))
            let left = index == 0 ? 0 : 86 * 512
            let right = index == chunks.count - 1 ? 0 : 258 * 512
            let sampleEnd = waveform.dim(2) - right
            let cropped = waveform[0..., 0..., left..<sampleEnd]
            MLX.eval(cropped)
            waveforms.append(cropped)
            progress?(.decode(chunk: index + 1, chunkCount: chunks.count))
        }
        return MLX.clip(MLX.concatenated(waveforms, axis: 2).asType(.float32), min: -1, max: 1)
    }
}
