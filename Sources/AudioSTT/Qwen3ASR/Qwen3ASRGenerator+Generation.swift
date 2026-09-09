import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioCore
import AudioCodecs
import MereRunCore

extension Qwen3ASRGenerator {
    // MARK: - Transcription Generation

    func decodeSamples(
        _ samples: [Float],
        request: ASRStreamingRequest,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        let output = try await decodeSamplesDetailed(
            samples,
            request: request,
            progressHandler: progressHandler
        )
        return output.result
    }

    func decodeSamplesDetailed(
        _ samples: [Float],
        request: ASRStreamingRequest,
        precomputedMelSpec: Qwen3ASRStreamingMel? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> Qwen3ASRStreamingDecodeOutput {
        guard let thinker, let tokenizer, let melExtractor else {
            throw Qwen3ASRError.modelsNotLoaded
        }

        let sampleRate = max(1, request.sampleRate)
        let audioDuration = TimeInterval(samples.count) / TimeInterval(sampleRate)
        let decodeStarted = Date()

        progressHandler?(ASRProgress(stage: .extractingFeatures, message: "Extracting mel spectrogram..."))
        let melSpec = precomputedMelSpec?.value ?? melExtractor.extract(from: samples)
        MLX.eval(melSpec)
        let melFinished = Date()
        if Self.debugEnabled {
            let melMean = MLX.mean(melSpec).item(Float.self)
            let melStd = MLX.sqrt(MLX.variance(melSpec)).item(Float.self)
            let melMin = MLX.min(melSpec).item(Float.self)
            let melMax = MLX.max(melSpec).item(Float.self)
            let message = "[ASR DEBUG] melSpec stats mean=\(melMean) std=\(melStd) min=\(melMin) max=\(melMax)\n"
            FileHandle.standardError.write(Data(message.utf8))
        }

        progressHandler?(ASRProgress(stage: .extractingFeatures, message: "Encoding audio..."))
        var audioFeatures = thinker.encodeAudio(melSpec)
        if audioFeatures.dtype != .bfloat16 {
            audioFeatures = audioFeatures.asType(.bfloat16)
        }
        MLX.eval(audioFeatures)
        let audioEncodingFinished = Date()
        if Self.debugEnabled {
            let message = "[ASR DEBUG] melSpec shape=\(melSpec.shape) audioFeatures shape=\(audioFeatures.shape) dtype=\(audioFeatures.dtype)\n"
            FileHandle.standardError.write(Data(message.utf8))
        }

        progressHandler?(ASRProgress(stage: .transcribing, message: "Transcribing..."))
        let transcription = try generateTranscription(
            audioFeatures: audioFeatures,
            thinker: thinker,
            tokenizer: tokenizer,
            task: request.task,
            language: request.language,
            maxTokens: request.maxTokens,
            progressHandler: progressHandler
        )
        let generationFinished = Date()

        if Self.debugEnabled {
            let message = String(
                format: "[ASR DEBUG] timing audio=%.3fs mel=%.1fms encoder=%.1fms decoder=%.1fms total=%.1fms\n",
                audioDuration,
                melFinished.timeIntervalSince(decodeStarted) * 1_000,
                audioEncodingFinished.timeIntervalSince(melFinished) * 1_000,
                generationFinished.timeIntervalSince(audioEncodingFinished) * 1_000,
                generationFinished.timeIntervalSince(decodeStarted) * 1_000
            )
            FileHandle.standardError.write(Data(message.utf8))
        }

        Memory.clearCache()

        let result = ASRResult(
            text: transcription.text,
            language: request.language,
            duration: audioDuration
        )
        return Qwen3ASRStreamingDecodeOutput(
            result: result,
            tokensGenerated: transcription.tokensGenerated
        )
    }

    private func generateTranscription(
        audioFeatures: MLXArray,
        thinker: Qwen3ASRThinker,
        tokenizer: Qwen3ASRTokenizer,
        task: ASRTask,
        language: String?,
        maxTokens: Int,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) throws -> Qwen3ASRDecodedTranscription {
        let audioLen = audioFeatures.dim(1)
        let prompt = tokenizer.createQwen3ASRPrompt(
            audioPlaceholderCount: audioLen,
            language: language,
            supportedLanguages: modelConfig?.supportLanguages
        )
        let output = try generateWithPrompt(
            prompt,
            audioFeatures: audioFeatures,
            thinker: thinker,
            tokenizer: tokenizer,
            maxTokens: maxTokens,
            progressHandler: progressHandler
        )
        return Qwen3ASRDecodedTranscription(
            text: cleanOutput(output.text),
            tokensGenerated: output.tokensGenerated
        )
    }

    private func isTrivialOutput(_ text: String, language: String?) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return true
        }
        if trimmed.contains("<asr_text>") {
            let cleaned = trimmed.replacingOccurrences(of: "<asr_text>", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if cleaned.isEmpty {
                return true
            }
            let langValue = (language?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false)
                ? language!.trimmingCharacters(in: .whitespacesAndNewlines)
                : "None"
            if cleaned == "language \(langValue)" || cleaned == "language" || cleaned == "language None" {
                return true
            }
        }
        return false
    }

    private func cleanOutput(_ text: String) -> String {
        text
            .replacingOccurrences(of: "<asr_text>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func assistantPrefixFor(language: String?) -> String {
        let trimmedLang = language?.trimmingCharacters(in: .whitespacesAndNewlines)
        let value = (trimmedLang?.isEmpty == false) ? trimmedLang! : "None"
        return "language \(value)"
    }

    private func instructionFor(task: ASRTask, language: String?) -> String {
        let trimmedLang = language?.trimmingCharacters(in: .whitespacesAndNewlines)
        switch task {
        case .transcribe:
            if let trimmedLang, !trimmedLang.isEmpty {
                return "Transcribe the audio in \(trimmedLang)."
            }
            return "Transcribe the audio."
        case .translate:
            if let trimmedLang, !trimmedLang.isEmpty {
                return "Translate the audio to \(trimmedLang)."
            }
            return "Translate the audio to English."
        }
    }

    private func shouldUseMRoPE(_ thinker: Qwen3ASRThinker) -> Bool {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_ASR_USE_MROPE"]?.lowercased() else {
            return false
        }
        let enabled = raw == "1" || raw == "true" || raw == "yes"
        guard enabled else { return false }
        return thinker.config.textConfig.ropeScaling?.mropeSection?.isEmpty == false
    }

    private func buildAudioPositionIds(promptTokens: [Int], audioTokenId: Int) -> MLXArray? {
        guard let audioStart = promptTokens.firstIndex(of: audioTokenId) else {
            return nil
        }

        var audioLen = 0
        var idx = audioStart
        while idx < promptTokens.count && promptTokens[idx] == audioTokenId {
            audioLen += 1
            idx += 1
        }

        guard audioLen > 0 else { return nil }

        let seqLen = promptTokens.count
        var positions = Array(repeating: [Int32](), count: 3)
        for d in 0..<3 { positions[d].reserveCapacity(seqLen) }

        // Text before audio: standard 1D positions.
        if audioStart > 0 {
            for i in 0..<audioStart {
                let val = Int32(i)
                positions[0].append(val)
                positions[1].append(val)
                positions[2].append(val)
            }
        }

        // Audio tokens: vary only the first dimension; keep others at 0.
        let audioBase = audioStart
        for i in 0..<audioLen {
            positions[0].append(Int32(audioBase + i))
            positions[1].append(0)
            positions[2].append(0)
        }

        // Text after audio: continue sequentially after audio.
        let tokensAfter = seqLen - audioStart - audioLen
        if tokensAfter > 0 {
            let textBase = audioBase + audioLen
            for i in 0..<tokensAfter {
                let val = Int32(textBase + i)
                positions[0].append(val)
                positions[1].append(val)
                positions[2].append(val)
            }
        }

        let flat = positions.flatMap { $0 }
        return MLXArray(flat, [3, 1, seqLen])
    }

    private enum SamplingMode {
        case greedy
        case topP(temperature: Float, topP: Float)
    }

    private func generateWithPrompt(
        _ promptTokens: [Int],
        audioFeatures: MLXArray,
        thinker: Qwen3ASRThinker,
        tokenizer: Qwen3ASRTokenizer,
        maxTokens: Int,
        progressHandler: (@Sendable (ASRProgress) -> Void)?,
        sampling: SamplingMode = .greedy
    ) throws -> Qwen3ASRDecodedTranscription {
        let inputIds = MLXArray(promptTokens.map { Int32($0) }).reshaped(1, -1)
        let positionIds = shouldUseMRoPE(thinker)
            ? buildAudioPositionIds(promptTokens: promptTokens, audioTokenId: tokenizer.audioTokenId)
            : nil

        // Prefill
        let cache = thinker.makeCache()
        var logits = thinker(
            inputIds: inputIds,
            audioFeatures: audioFeatures,
            cache: cache,
            positionIds: positionIds,
            lastPositionOnly: true
        )
        MLX.eval(logits)

        // Generate tokens
        var generatedTokens: [Int] = []
        let eosTokenId = tokenizer.eosTokenId
        let padTokenId = tokenizer.padTokenId

        if Self.debugEnabled {
            let firstLogits = logits[0..., (logits.dim(1) - 1), 0...].squeezed(axis: 0)
            let topk = MLX.argSort(firstLogits, axis: -1)[(-5)...]
            let topkTokens = topk.asArray(Int32.self).reversed()
            let topkStrings = topkTokens.map { token in
                let id = Int(token)
                return "\(id):\"\(tokenizer.decode([id]))\""
            }
            let message = "[ASR DEBUG] top5=\(topkStrings.joined(separator: ", "))\n"
            FileHandle.standardError.write(Data(message.utf8))
        }

        if Self.pipelinedDecodeEnabled {
            // Depth-1 pipelined decode: the sampled token stays on GPU and
            // feeds the next forward directly; the previous step's token is
            // read back while the current step executes. The legacy loop
            // synchronized twice per token (sample readback + eval).
            var pendingToken: MLXArray?
            for step in 0..<maxTokens {
                let lastLogits = logits[0..., (logits.dim(1) - 1), 0...]
                let tokenArray = sampleTokenArray(logits: lastLogits, mode: sampling)
                logits = thinker(inputIds: tokenArray.reshaped(1, 1), cache: cache)
                asyncEval([logits, tokenArray])

                if let previous = pendingToken {
                    pendingToken = nil
                    let value = previous.item(Int.self)
                    if value == eosTokenId || value == padTokenId {
                        if Self.debugEnabled {
                            FileHandle.standardError.write(Data("[ASR DEBUG] Hit EOS at step \(step - 1)\n".utf8))
                        }
                        break
                    }
                    generatedTokens.append(value)
                    if generatedTokens.count % 10 == 0 {
                        progressHandler?(ASRProgress(
                            stage: .transcribing,
                            tokensGenerated: generatedTokens.count,
                            message: "Generated \(generatedTokens.count) tokens..."
                        ))
                    }
                    if generatedTokens.count % 50 == 0 {
                        Memory.clearCache()
                    }
                }
                pendingToken = tokenArray
            }
            if let previous = pendingToken {
                let value = previous.item(Int.self)
                if value != eosTokenId && value != padTokenId {
                    generatedTokens.append(value)
                }
            }
        } else {
            for step in 0..<maxTokens {
                let lastLogits = logits[0..., (logits.dim(1) - 1), 0...]
                let nextToken = sampleToken(logits: lastLogits, mode: sampling, previousTokens: generatedTokens)

                if nextToken == eosTokenId || nextToken == padTokenId {
                    if Self.debugEnabled {
                        FileHandle.standardError.write(Data("[ASR DEBUG] Hit EOS at step \(step)\n".utf8))
                    }
                    break
                }

                generatedTokens.append(nextToken)

                if step > 0 && step % 10 == 0 {
                    progressHandler?(ASRProgress(
                        stage: .transcribing,
                        tokensGenerated: step,
                        message: "Generated \(step) tokens..."
                    ))
                }

                // Generate next
                let nextInput = MLXArray([Int32(nextToken)]).reshaped(1, 1)
                logits = thinker(inputIds: nextInput, cache: cache)
                MLX.eval(logits)

                if step % 50 == 0 {
                    Memory.clearCache()
                }
            }
        }

        // Decode to text
        let decoded = tokenizer.decode(generatedTokens)
        if Self.debugEnabled {
            let preview = generatedTokens.prefix(20).map(String.init).joined(separator: ", ")
            let message = "[ASR DEBUG] generated=\(generatedTokens.count) tokens [\(preview)]\n"
            FileHandle.standardError.write(Data(message.utf8))
            FileHandle.standardError.write(Data("[ASR DEBUG] decoded=\"\(decoded)\"\n".utf8))
        }
        return Qwen3ASRDecodedTranscription(
            text: decoded,
            tokensGenerated: generatedTokens.count
        )
    }

    /// GPU-side variant of `sampleToken`: identical math, but the result
    /// stays on GPU as a 0-d array so the decode loop can feed it straight
    /// into the next forward without a host readback.
    private func sampleTokenArray(
        logits: MLXArray,
        mode: SamplingMode
    ) -> MLXArray {
        let squeezed = logits.squeezed(axis: 0)
        switch mode {
        case .greedy:
            return argMax(squeezed, axis: -1).asType(.int32)
        case .topP(let temperature, let topP):
            var scores = squeezed
            if scores.dtype == .bfloat16 {
                scores = scores.asType(.float32)
            }
            let probs = softmax(scores / temperature, axis: -1)
            let sortedIndices = argSort(probs, axis: -1)
            let sortedProbs = probs.take(sortedIndices, axis: -1)
            let cumulativeProbs = cumsum(sortedProbs, axis: -1)
            let topProbs = MLX.where(
                cumulativeProbs .> (1 - topP),
                sortedProbs,
                MLXArray.zeros(like: sortedProbs)
            )
            let sortedToken = categorical(MLX.log(topProbs + 1e-10))
            return sortedIndices.take(sortedToken.reshaped(1), axis: -1)
                .squeezed(axis: 0).asType(.int32)
        }
    }

    private func sampleToken(
        logits: MLXArray,
        mode: SamplingMode,
        previousTokens: [Int]
    ) -> Int {
        let squeezed = logits.squeezed(axis: 0)
        switch mode {
        case .greedy:
            return argMax(squeezed, axis: -1).item(Int.self)
        case .topP(let temperature, let topP):
            var scores = squeezed
            if scores.dtype == .bfloat16 {
                scores = scores.asType(.float32)
            }
            let probs = softmax(scores / temperature, axis: -1)
            let sortedIndices = argSort(probs, axis: -1)
            let sortedProbs = probs.take(sortedIndices, axis: -1)
            let cumulativeProbs = cumsum(sortedProbs, axis: -1)
            let topProbs = MLX.where(
                cumulativeProbs .> (1 - topP),
                sortedProbs,
                MLXArray.zeros(like: sortedProbs)
            )
            let sortedToken = categorical(MLX.log(topProbs + 1e-10))
            return sortedIndices[sortedToken].item(Int.self)
        }
    }
}

private struct Qwen3ASRDecodedTranscription: Sendable {
    let text: String
    let tokensGenerated: Int
}
