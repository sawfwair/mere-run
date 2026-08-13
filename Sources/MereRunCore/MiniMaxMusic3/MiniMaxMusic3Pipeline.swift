import Foundation
import MLX
import MLXRandom

public struct MiniMaxMusic3GenerationOptions: Sendable {
    public var caption: String
    public var lyrics: String
    public var durationSeconds: Float
    public var inferenceSteps: Int
    public var seed: UInt64
    public var guidanceScale: Float

    public init(
        caption: String,
        lyrics: String,
        durationSeconds: Float = 60,
        inferenceSteps: Int = 30,
        seed: UInt64 = 0,
        guidanceScale: Float = 1.7
    ) {
        self.caption = caption
        self.lyrics = lyrics
        self.durationSeconds = durationSeconds
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
    public let models: MiniMaxMusic3Models

    public init(models: MiniMaxMusic3Models) {
        self.models = models
    }

    public convenience init(resources: MiniMaxMusic3Resources) throws {
        try self.init(models: MiniMaxMusic3ModelLoader.load(from: resources))
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

        MLXRandom.seed(options.seed)
        let tokenIDs = try promptTokenIDs(caption: options.caption, lyrics: options.lyrics)
        let frameHiddens = try generateSemanticFrames(
            textIDs: tokenIDs,
            durationSeconds: options.durationSeconds,
            progress: progress
        )
        let chunks = denoise(
            frameHiddens: frameHiddens,
            steps: options.inferenceSteps,
            guidanceScale: options.guidanceScale,
            progress: progress
        )
        let waveform = decode(chunks: chunks, progress: progress)
        MLX.eval(waveform)
        return MiniMaxMusic3GenerationResult(
            waveform: waveform,
            sampleRate: models.vocoder.configuration.samplingRate,
            frameCount: frameHiddens.dim(1)
        )
    }

    private func promptTokenIDs(caption: String, lyrics: String) throws -> MLXArray {
        let prompt = MiniMaxMusic3Prompt.assemble(caption: caption, lyrics: lyrics)
        let conditional = models.tokenizer.encode(prompt, addSpecialTokens: false)
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
        progress: (@Sendable (MiniMaxMusic3Progress) -> Void)?
    ) throws -> MLXArray {
        let languageModel = models.languageModel
        let caches = languageModel.makeCache()
        let textEmbeddings = languageModel.embed(tokenIDs: textIDs)
        var lastHidden = languageModel.hidden(
            embeddings: textEmbeddings,
            cache: caches,
            lastPositionOnly: true
        ).squeezed(axis: 1)
        let requestedFrames = Int(durationSeconds * Float(MiniMaxMusic3Prompt.frameRate))
        let maximumFrames = min(requestedFrames, MiniMaxMusic3Prompt.maxAudioFrames)
        guard maximumFrames > 0 else {
            throw MiniMaxMusic3Error.invalidPrompt("duration is shorter than one 25 Hz audio frame")
        }

        var frameHiddens: [MLXArray] = []
        frameHiddens.reserveCapacity(maximumFrames)
        for frameIndex in 0...maximumFrames {
            let sampled = sampleSemantic(logits: languageModel.logits(lastHidden))
            let sampledID = sampled.item(Int.self)
            if sampledID == MiniMaxMusic3Prompt.audioEndTokenID {
                break
            }

            let semantic = sampled.asType(.int32) - MLXArray(Int32(MiniMaxMusic3Prompt.audioCodeOffset))
            let duplicatedSemantic = MLX.repeated(semantic.reshaped(1), count: 2, axis: 0)
            let depth = generateDepthCodes(lastHidden: lastHidden, semanticCode: duplicatedSemantic)
            if frameIndex > 0 {
                let conditioning = MLX.concatenated([lastHidden[0..<1], depth.hidden], axis: -1)
                MLX.eval(conditioning)
                frameHiddens.append(conditioning)
                progress?(.semantic(frame: frameHiddens.count, maximum: maximumFrames))
                if frameHiddens.count >= maximumFrames {
                    break
                }
            }

            let feedback = embedAudioFrame(depth.codes)
            lastHidden = languageModel.hidden(
                embeddings: feedback,
                cache: caches,
                lastPositionOnly: true
            ).squeezed(axis: 1)
            MLX.eval(lastHidden)
        }

        guard !frameHiddens.isEmpty else {
            throw MiniMaxMusic3Error.generatedNoFrames
        }
        return MLX.stacked(frameHiddens, axis: 1)
    }

    private func sampleSemantic(logits: MLXArray) -> MLXArray {
        let guided = Self.guidedSemanticLogits(logits)
        return sampledTokenArray(
            logits: guided[0],
            config: GenerationConfig(
                temperature: 1,
                topK: 50,
                topP: 1,
                repetitionPenalty: nil
            ),
            previousTokenIndices: nil,
            banMask: nil
        )
    }

    static func guidedSemanticLogits(_ logits: MLXArray) -> MLXArray {
        let vocabulary = logits.dim(-1)
        let tokenIndices = MLXArray((0..<vocabulary).map(Int32.init)).reshaped(1, vocabulary)
        let semanticStart = MLXArray(Int32(MiniMaxMusic3Prompt.audioCodeOffset))
        let semanticEnd = MLXArray(Int32(
            MiniMaxMusic3Prompt.audioCodeOffset + MiniMaxMusic3Prompt.semanticVocabularySize
        ))
        let allowedSemantic = (tokenIndices .>= semanticStart) .&& (tokenIndices .< semanticEnd)
        let allowedEnd = tokenIndices .== MLXArray(Int32(MiniMaxMusic3Prompt.audioEndTokenID))
        let allowed = allowedSemantic .|| allowedEnd

        // MiniMax masks the language-model vocabulary before choosing the
        // conditional branch's top candidates. Text-token logits must never
        // influence the audio-code threshold.
        let masked = MLX.where(allowed, logits.asType(.float32), MLXArray(-Float.infinity))
        let conditional = masked[0..<1]
        let unconditional = masked[1..<2]
        var guided = unconditional + (conditional - unconditional) * MLXArray(Float(1.5))
        let firstTopIndex = vocabulary - min(50, vocabulary)
        let topIndices = MLX.argPartition(conditional, kth: firstTopIndex, axis: -1)[0, firstTopIndex...]
        let threshold = conditional[0].take(topIndices, axis: -1).min()
        guided = MLX.where(conditional .< threshold, MLXArray(-Float.infinity), guided)
        return MLX.where(allowed, guided, MLXArray(-Float.infinity))
    }

    private func generateDepthCodes(
        lastHidden: MLXArray,
        semanticCode: MLXArray
    ) -> (codes: MLXArray, hidden: MLXArray) {
        let depthDecoder = models.depthDecoder
        var sequence = [depthDecoder.projection(lastHidden).expandedDimensions(axis: 1)]
        let semanticEmbedding = models.languageModel.embed(
            tokenIDs: semanticCode + MLXArray(Int32(MiniMaxMusic3Prompt.audioCodeOffset))
        )
        sequence.append(depthDecoder.projection(semanticEmbedding).expandedDimensions(axis: 1))
        var codes = [semanticCode]
        var hiddenParts: [MLXArray] = []
        for codebook in 1..<depthDecoder.configuration.numCodebooks {
            let hidden = depthDecoder(MLX.concatenated(sequence, axis: 1))[0..., -1, 0...]
            hiddenParts.append(hidden[0..<1])
            let logits = depthDecoder.logits(hidden, codebookIndex: codebook - 1)
            let conditional = logits[0..<1].asType(.float32)
            let unconditional = logits[1..<2].asType(.float32)
            let guided = unconditional + (conditional - unconditional) * MLXArray(Float(1.5))
            let sampled = sampledTokenArray(
                logits: guided[0],
                config: GenerationConfig(
                    temperature: 1,
                    topK: 50,
                    topP: 1,
                    repetitionPenalty: nil
                ),
                previousTokenIndices: nil,
                banMask: nil
            )
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

    private func embedAudioFrame(_ frameCodes: MLXArray) -> MLXArray {
        var embedding = models.languageModel.embed(
            tokenIDs: frameCodes[0..., 0] + MLXArray(Int32(MiniMaxMusic3Prompt.audioCodeOffset))
        )
        for codebook in 1..<models.depthDecoder.configuration.numCodebooks {
            embedding = embedding + models.depthDecoder.embedResidualCodes(
                frameCodes[0..., codebook],
                codebookIndex: codebook - 1
            ).asType(embedding.dtype)
        }
        let scale = 1 / Float(models.depthDecoder.configuration.numCodebooks).squareRoot()
        return (embedding * MLXArray(scale)).expandedDimensions(axis: 1)
    }

    private func denoise(
        frameHiddens: MLXArray,
        steps: Int,
        guidanceScale: Float,
        progress: (@Sendable (MiniMaxMusic3Progress) -> Void)?
    ) -> [MLXArray] {
        let starts = MiniMaxMusic3Prompt.chunkStarts(frameCount: frameHiddens.dim(1))
        var previousLatent: MLXArray?
        var previousCondition: MLXArray?
        var chunks: [MLXArray] = []
        chunks.reserveCapacity(starts.count)

        for (chunkIndex, start) in starts.enumerated() {
            let end = min(start + 200, frameHiddens.dim(1))
            var condition = models.conditionEncoder(frameHiddens[0..., start..<end, 0...])
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
                models.transformer.configuration.inChannels,
                condition.dim(1),
            ]).asType(condition.dtype)
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
                let conditional = models.transformer(
                    latents: latents,
                    timestep: timestep,
                    condition: condition
                )
                let unconditional = models.transformer(
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
        progress: (@Sendable (MiniMaxMusic3Progress) -> Void)?
    ) -> MLXArray {
        var waveforms: [MLXArray] = []
        waveforms.reserveCapacity(chunks.count)
        for (index, chunk) in chunks.enumerated() {
            let waveform = models.vocoder(chunk.asType(.bfloat16))
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
