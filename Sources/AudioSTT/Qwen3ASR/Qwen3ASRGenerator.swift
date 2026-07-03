import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioCore
import AudioCodecs
import MereRunCore

// MARK: - Qwen3 ASR Generator

/// Native Swift implementation for Qwen3-ASR speech recognition
public actor Qwen3ASRGenerator: ASRGenerator {

    // MARK: - Private State

    private var thinker: Qwen3ASRThinker?
    private var tokenizer: Qwen3ASRTokenizer?
    private var melExtractor: MelSpectrogram?
    private var modelConfig: Qwen3ASRModelConfig?
    private var loadedModelPath: String?

    private let modelId: String
    private static let debugEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_ASR_DEBUG"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }()

    /// Depth-1 pipelined decode (default on): the sampled token feeds the
    /// next forward as a GPU array and the previous token is read back while
    /// the current step executes. MERERUN_STT_PIPELINED_DECODE=0 restores
    /// the legacy two-syncs-per-token loop.
    private static let pipelinedDecodeEnabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MERERUN_STT_PIPELINED_DECODE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return raw != "0" && raw != "false" && raw != "off"
    }()

    public init(modelId: String = Qwen3ASRResources.defaultModelId) {
        self.modelId = modelId
    }

    // MARK: - Public API

    /// Transcribe audio to text
    public func transcribe(
        _ request: ASRRequest,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        try await transcribe(request, modelPath: nil, progressHandler: progressHandler)
    }

    /// Transcribe with explicit model path
    public func transcribe(
        _ request: ASRRequest,
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)

        if loadedModelPath != rootURL.path {
            progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading Qwen3-ASR model..."))
            try await loadModels(from: rootURL, progressHandler: progressHandler)
        }

        progressHandler?(ASRProgress(stage: .loadingAudio, message: "Loading audio..."))
        let audio = try AudioReader.readAudio(from: request.audioURL)
        let streamingRequest = ASRStreamingRequest(
            language: request.language,
            task: request.task,
            maxTokens: request.maxTokens,
            sampleRate: Qwen3ASRResources.sampleRate
        )
        return try await decodeSamples(
            audio,
            request: streamingRequest,
            progressHandler: progressHandler
        )
    }

    public func makeStreamingSession(
        _ request: ASRStreamingRequest
    ) async throws -> any ASRStreamingSession {
        guard request.sampleRate == Qwen3ASRResources.sampleRate else {
            throw ASRStreamingError.invalidInput(
                "Qwen3 ASR streaming expects sampleRate=\(Qwen3ASRResources.sampleRate)."
            )
        }
        guard request.decodeIntervalMs > 0 else {
            throw ASRStreamingError.invalidInput("decodeIntervalMs must be > 0.")
        }
        guard request.minDecodeAudioMs >= 0 else {
            throw ASRStreamingError.invalidInput("minDecodeAudioMs must be >= 0.")
        }

        if thinker == nil || tokenizer == nil || melExtractor == nil || modelConfig == nil {
            try await prepare()
        }
        return Qwen3ASRStreamingSession(
            request: request,
            decode: { [generator = self] samples in
                try await generator.decodeSamplesDetailed(
                    samples,
                    request: request,
                    progressHandler: nil
                )
            }
        )
    }

    /// Pre-load models without transcribing
    public func prepare(
        modelPath: String? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws {
        let rootURL = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
        if loadedModelPath != rootURL.path {
            progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading Qwen3-ASR model..."))
            try await loadModels(from: rootURL, progressHandler: progressHandler)
        }
    }

    /// Unload models from memory
    public func unload() {
        thinker = nil
        tokenizer = nil
        melExtractor = nil
        modelConfig = nil
        loadedModelPath = nil
        Memory.clearCache()
    }

    // MARK: - Model Resolution

    private func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> URL {
        do {
            let resolved = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: modelPath ?? modelId,
                defaultModelID: Qwen3ASRResources.defaultModelId,
                progress: { event in
                    switch event {
                    case .downloading(let percent):
                        progressHandler?(
                            ASRProgress(stage: .loadingModel, message: "Downloading model... \(percent)%")
                        )
                    case .extracting:
                        progressHandler?(
                            ASRProgress(stage: .loadingModel, message: "Extracting model...")
                        )
                    }
                }
            )
            return resolved.url
        } catch let error as ManagedModelResolver.ResolverError {
            throw Qwen3ASRError.downloadFailed(error.localizedDescription)
        }
    }

    private func mapModelLoaderError(_ error: PretrainedModelLoader.LoadError) -> Qwen3ASRError {
        switch error {
        case .unsupportedModelId(let modelId):
            return .unsupportedModelId(modelId)
        case .missingFiles(let files):
            return .missingFiles(files)
        case .downloadFailed(let message):
            return .downloadFailed(message)
        }
    }

    // MARK: - Model Loading

    private func loadModels(
        from rootURL: URL,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws {
        let resources = Qwen3ASRResources(rootURL: rootURL)
        let missingFiles = resources.validate()
        guard missingFiles.isEmpty else {
            throw Qwen3ASRError.missingFiles(missingFiles.map { $0.lastPathComponent })
        }

        let config = try Qwen3ASRModelConfig.load(from: resources.configURL)
        let thinker = Qwen3ASRThinker(config: config)

        progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading model weights..."))
        try loadModelWeights(resources: resources, thinker: thinker)

        progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading tokenizer..."))
        let tokenizer = try Qwen3ASRTokenizer.load(from: rootURL, config: config)

        let melExtractor = MelSpectrogram()

        self.thinker = thinker
        self.tokenizer = tokenizer
        self.melExtractor = melExtractor
        self.modelConfig = config
        self.loadedModelPath = rootURL.path
    }

    private func loadModelWeights(
        resources: Qwen3ASRResources,
        thinker: Qwen3ASRThinker
    ) throws {
        let indexURL = resources.modelIndexURL
        let singleURL = resources.modelWeightsURL
        let fm = FileManager.default

        let arrays: [String: MLXArray]
        if fm.fileExists(atPath: indexURL.path) {
            arrays = try HFSafetensorsWeightsLoader.loadShardedArrays(indexURL: indexURL)
        } else if fm.fileExists(atPath: singleURL.path) {
            arrays = try MLX.loadArrays(url: singleURL)
        } else {
            throw Qwen3ASRError.weightsNotFound(indexURL)
        }

        let mapped = mapASRWeights(
            arrays,
            tieWordEmbeddings: thinker.config.textConfig.tieWordEmbeddings
        )

        if HFSafetensorsWeightsLoader.isQuantized(mapped) {
            let bits = thinker.config.quantizationBits ?? 8
            let groupSize = thinker.config.quantizationGroupSize ?? 64
            try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                mapped,
                to: thinker,
                groupSize: groupSize,
                bits: bits
            )
        } else {
            var updates: [(String, MLXArray)] = []
            updates.reserveCapacity(mapped.count)
            for (key, value) in mapped {
                let casted = HFSafetensorsWeightsLoader.castIfNeeded(value, dtype: .bfloat16)
                updates.append((key, casted))
            }
            try thinker.update(parameters: ModuleParameters.unflattened(updates), verify: .none)
        }
    }

    // MARK: - Transcription Generation

    fileprivate func decodeSamples(
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

    fileprivate func decodeSamplesDetailed(
        _ samples: [Float],
        request: ASRStreamingRequest,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> Qwen3ASRStreamingDecodeOutput {
        guard let thinker, let tokenizer, let melExtractor else {
            throw Qwen3ASRError.modelsNotLoaded
        }

        let sampleRate = max(1, request.sampleRate)
        let audioDuration = TimeInterval(samples.count) / TimeInterval(sampleRate)

        progressHandler?(ASRProgress(stage: .extractingFeatures, message: "Extracting mel spectrogram..."))
        let melSpec = melExtractor.extract(from: samples)
        MLX.eval(melSpec)
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
            positionIds: positionIds
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

struct Qwen3ASRStreamingDecodeOutput: Sendable {
    let result: ASRResult
    let tokensGenerated: Int
}

actor Qwen3ASRStreamingSession: ASRStreamingSession {
    nonisolated let events: AsyncThrowingStream<ASRStreamingEvent, Error>

    private let request: ASRStreamingRequest
    private let decode: @Sendable ([Float]) async throws -> Qwen3ASRStreamingDecodeOutput
    private let continuation: AsyncThrowingStream<ASRStreamingEvent, Error>.Continuation

    private var melBuffer: IncrementalMelSpectrogram
    private var decodeCadence: StreamingDecodeCadence
    private var decodeCount = 0
    private var lastPartialText: String?
    private var lastResult: ASRResult?
    private var finalEmitted = false
    private var didFinish = false
    private var canceled = false
    private var decodeInFlight = false
    private var pendingDecode = false

    init(
        request: ASRStreamingRequest,
        decode: @escaping @Sendable ([Float]) async throws -> Qwen3ASRStreamingDecodeOutput
    ) {
        self.request = request
        self.decode = decode
        self.melBuffer = IncrementalMelSpectrogram(sampleRate: request.sampleRate)
        self.decodeCadence = StreamingDecodeCadence(
            sampleRate: request.sampleRate,
            decodeIntervalMs: request.decodeIntervalMs,
            minDecodeAudioMs: request.minDecodeAudioMs
        )

        var capturedContinuation: AsyncThrowingStream<ASRStreamingEvent, Error>.Continuation?
        self.events = AsyncThrowingStream { continuation in
            capturedContinuation = continuation
        }
        self.continuation = capturedContinuation!
    }

    func feed(samples: [Float]) async throws {
        guard !samples.isEmpty else { return }
        guard !didFinish, !canceled else {
            throw ASRStreamingError.invalidState("Cannot feed samples after finish/cancel.")
        }

        melBuffer.append(samples)
        guard decodeCadence.shouldDecode(bufferedSampleCount: melBuffer.sampleCount) else { return }
        try await scheduleDecode(force: false)
    }

    func finish() async throws {
        if canceled || didFinish { return }

        didFinish = true
        while decodeInFlight {
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        try await scheduleDecode(force: true)
        emitFinalIfNeeded()
        continuation.finish()
    }

    func cancel() async {
        guard !canceled else { return }
        canceled = true
        didFinish = true
        continuation.finish()
    }

    private func scheduleDecode(force: Bool) async throws {
        pendingDecode = true
        if decodeInFlight { return }

        decodeInFlight = true
        defer { decodeInFlight = false }

        var forceNextDecode = force
        while pendingDecode || forceNextDecode {
            pendingDecode = false
            if canceled { return }

            let snapshot = melBuffer.snapshotSamples()
            guard !snapshot.isEmpty else {
                forceNextDecode = false
                continue
            }

            if !decodeCadence.shouldDecode(bufferedSampleCount: snapshot.count, force: forceNextDecode) {
                forceNextDecode = false
                continue
            }

            let start = Date()
            let output = try await decode(snapshot)
            if canceled { return }

            let latencyMs = Date().timeIntervalSince(start) * 1_000
            decodeCount += 1
            decodeCadence.markDecoded(sampleCount: snapshot.count)
            lastResult = output.result

            if output.result.text != lastPartialText {
                lastPartialText = output.result.text
                continuation.yield(.partial(text: output.result.text))
            }

            let stats = ASRStreamingStats(
                decodeCount: decodeCount,
                totalAudioSeconds: Double(snapshot.count) / Double(max(1, request.sampleRate)),
                lastDecodeLatencyMs: latencyMs,
                tokensGenerated: output.tokensGenerated
            )
            continuation.yield(.stats(stats))
            forceNextDecode = false
        }
    }

    private func emitFinalIfNeeded() {
        guard !finalEmitted else { return }
        finalEmitted = true

        if let lastResult {
            continuation.yield(.final(result: lastResult))
            return
        }

        let duration = melBuffer.totalAudioSeconds
        continuation.yield(.final(result: ASRResult(text: "", language: request.language, duration: duration)))
    }
}

// MARK: - Weight Key Mapping

private func mapASRWeightKey(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
    var mapped = key

    // Drop optional thinker prefix
    if mapped.hasPrefix("thinker.") {
        mapped = String(mapped.dropFirst("thinker.".count))
    }

    // Only accept known roots
    guard mapped.hasPrefix("audio_tower.")
        || mapped.hasPrefix("model.")
        || mapped.hasPrefix("lm_head.") else {
        return []
    }

    // Conv2d weights: convert only if stored as OIHW
    if mapped.contains("conv2d") && mapped.hasSuffix(".weight") {
        return [(mapped, convertConv2DWeightIfNeeded(value))]
    }

    return [(mapped, value)]
}

private func mapASRWeights(
    _ arrays: [String: MLXArray],
    tieWordEmbeddings: Bool
) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]
    mapped.reserveCapacity(arrays.count)

    for (key, value) in arrays {
        var mappedKey = key
        if mappedKey.hasPrefix("thinker.") {
            mappedKey = String(mappedKey.dropFirst("thinker.".count))
        }

        if !(mappedKey.hasPrefix("audio_tower.")
            || mappedKey.hasPrefix("model.")
            || mappedKey.hasPrefix("lm_head.")) {
            continue
        }

        if tieWordEmbeddings, mappedKey.hasPrefix("lm_head.") {
            continue
        }

        let mappedValue: MLXArray
        if mappedKey.contains("conv2d") && mappedKey.hasSuffix(".weight") {
            mappedValue = convertConv2DWeightIfNeeded(value)
        } else {
            mappedValue = value
        }

        mapped[mappedKey] = mappedValue
    }

    return mapped
}

private func convertConv2DWeightIfNeeded(_ value: MLXArray) -> MLXArray {
    guard value.ndim == 4 else { return value }
    let s = value.shape
    // Qwen3-ASR conv kernels are 3x3. Detect layout by which dims are 3.
    if s[1] == 3 && s[2] == 3 {
        // Already OHWI (out, kH, kW, in)
        return value
    }
    if s[2] == 3 && s[3] == 3 {
        // OIHW -> OHWI
        return HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(value)
    }
    return value
}

// MARK: - Errors

public enum Qwen3ASRError: LocalizedError {
    case modelsNotLoaded
    case unsupportedModelId(String)
    case missingFiles([String])
    case weightsNotFound(URL)
    case downloadFailed(String)
    case extractionFailed
    case transcriptionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelsNotLoaded:
            return "ASR models not loaded"
        case .unsupportedModelId(let modelId):
            return "Unsupported model id: \(modelId)"
        case .missingFiles(let files):
            return "Missing required files: \(files.joined(separator: ", "))"
        case .weightsNotFound(let url):
            return "Weights not found at \(url.path)"
        case .downloadFailed(let message):
            return "Download failed: \(message)"
        case .extractionFailed:
            return "Failed to prepare model files"
        case .transcriptionFailed(let message):
            return "Transcription failed: \(message)"
        }
    }
}
