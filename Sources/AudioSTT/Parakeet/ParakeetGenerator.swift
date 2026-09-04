import Foundation
import MLX
import MLXNN
import AudioCore
import AudioCodecs
import MereRunCore

public actor ParakeetGenerator: ASRGenerator {
    private var model: (any ParakeetDecodingModel)?
    private var audioPreprocessor: ParakeetAudioPreprocessor?
    private var modelConfig: ParakeetModelConfig?
    private var loadedModelPath: String?

    private let modelId: String
    private let executionProvider: ParakeetExecutionProvider

    private static let debugEnabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_ASR_DEBUG"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }()

    public init(
        modelId: String = ParakeetResources.defaultModelId,
        executionProvider: ParakeetExecutionProvider = .mlx
    ) {
        self.modelId = modelId
        self.executionProvider = executionProvider
    }

    init(
        preparedModel: any ParakeetDecodingModel,
        audioPreprocessor: ParakeetAudioPreprocessor,
        modelConfig: ParakeetModelConfig,
        executionProvider: ParakeetExecutionProvider = .mlx
    ) {
        self.modelId = ParakeetResources.defaultModelId
        self.executionProvider = executionProvider
        self.model = preparedModel
        self.audioPreprocessor = audioPreprocessor
        self.modelConfig = modelConfig
        self.loadedModelPath = "<prepared>"
    }

    public func transcribe(
        _ request: ASRRequest,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        try await transcribe(request, modelPath: nil, progressHandler: progressHandler)
    }

    public func transcribe(
        _ request: ASRRequest,
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        let root = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)

        if loadedModelPath != root.path {
            progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading Parakeet model..."))
            try await loadModel(from: root, progressHandler: progressHandler)
        }

        progressHandler?(ASRProgress(stage: .loadingAudio, message: "Loading audio..."))
        let audio = try AudioReader.readAudio(from: request.audioURL)
        return try await transcribePrepared(
            samples: audio,
            language: request.language,
            progressHandler: progressHandler
        )
    }

    public func transcribe(
        samples: [Float],
        language: String? = nil,
        modelPath: String? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        let root = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)

        if loadedModelPath != root.path {
            progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading Parakeet model..."))
            try await loadModel(from: root, progressHandler: progressHandler)
        }

        return try await transcribePrepared(
            samples: samples,
            language: language,
            progressHandler: progressHandler
        )
    }

    public func transcribePrepared(
        samples: [Float],
        language: String? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ASRResult {
        try await transcribePreparedMeasured(
            samples: samples,
            language: language,
            progressHandler: progressHandler
        ).result
    }

    public func transcribePreparedMeasured(
        samples: [Float],
        language: String? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws -> ParakeetMeasuredTranscription {
        try await Stream.withNewDefaultStream(isolation: self) {
            // Keep this closure genuinely asynchronous under -O. MLX's async overload
            // installs task-local CPU and GPU streams that survive executor hops; without
            // a suspension the optimizer can collapse the body onto the caller thread and
            // `.gpu` operations fall back to MLX's missing thread-local default stream.
            await Task.yield()
            guard let model, let audioPreprocessor, let modelConfig else {
                throw ParakeetError.modelNotLoaded
            }

            return try decodeMeasured(
                samples: samples,
                language: language,
                model: model,
                audioPreprocessor: audioPreprocessor,
                modelConfig: modelConfig,
                progressHandler: progressHandler
            )
        }
    }

    private func decodeMeasured(
        samples: [Float],
        language: String?,
        model: any ParakeetDecodingModel,
        audioPreprocessor: ParakeetAudioPreprocessor,
        modelConfig: ParakeetModelConfig,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) throws -> ParakeetMeasuredTranscription {
        let totalStarted = ParakeetMonotonicClock.now()
        var timings = ParakeetPipelineTimings()
        let audioDuration = TimeInterval(samples.count) / TimeInterval(modelConfig.preprocessor.sampleRate)

        let aligned: ParakeetAlignedResult
        switch executionProvider {
        case .mlx:
            timings.windowCount = 1
            aligned = try decodeWindow(
                samples: samples,
                model: model,
                audioPreprocessor: audioPreprocessor,
                modelConfig: modelConfig,
                timings: &timings,
                progressHandler: progressHandler
            )
        case .coreML:
            aligned = try decodeCoreMLWindows(
                samples: samples,
                model: model,
                audioPreprocessor: audioPreprocessor,
                modelConfig: modelConfig,
                timings: &timings,
                progressHandler: progressHandler
            )
        }

        let alignmentStarted = ParakeetMonotonicClock.now()
        let flattened = aligned.sentences.flatMap(\.tokens)
        let tokenAlignments = ParakeetAlignment.toASRTokenAlignments(flattened)
        let sentenceAlignments = ParakeetAlignment.toASRSentenceAlignments(aligned.sentences)
        timings.alignmentSeconds += ParakeetMonotonicClock.seconds(since: alignmentStarted)

        if Self.debugEnabled {
            let message = "[ASR DEBUG] parakeet_backend=\(modelConfig.variant.rawValue) tokens=\(flattened.count)\n"
            FileHandle.standardError.write(Data(message.utf8))
            let preview = flattened.prefix(24).map { "\($0.id):\($0.text)" }.joined(separator: " | ")
            FileHandle.standardError.write(Data("[ASR DEBUG] parakeet_tokens_preview \(preview)\n".utf8))
        }

        let cleanupStarted = ParakeetMonotonicClock.now()
        Memory.clearCache()
        timings.cleanupSeconds += ParakeetMonotonicClock.seconds(since: cleanupStarted)
        timings.totalSeconds = ParakeetMonotonicClock.seconds(since: totalStarted)

        return ParakeetMeasuredTranscription(
            result: ASRResult(
                text: aligned.text,
                language: language,
                duration: audioDuration,
                tokenAlignments: tokenAlignments,
                sentenceAlignments: sentenceAlignments
            ),
            timings: timings
        )
    }

    private func decodeCoreMLWindows(
        samples: [Float],
        model: any ParakeetDecodingModel,
        audioPreprocessor: ParakeetAudioPreprocessor,
        modelConfig: ParakeetModelConfig,
        timings: inout ParakeetPipelineTimings,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) throws -> ParakeetAlignedResult {
        let sampleRate = modelConfig.preprocessor.sampleRate
        let ranges = ParakeetCoreMLWindowing.sampleRanges(
            sampleCount: samples.count,
            sampleRate: sampleRate
        )
        timings.windowCount = ranges.count
        var mergedTokens: [ParakeetAlignedToken] = []
        let batchSize = max(1, model.preferredWindowBatchSize)
        for batchStart in stride(from: 0, to: ranges.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, ranges.count)
            let batchRanges = Array(ranges[batchStart..<batchEnd])
            var mels: [MLXArray] = []
            mels.reserveCapacity(batchRanges.count)

            for (offset, range) in batchRanges.enumerated() {
                let index = batchStart + offset
                progressHandler?(ASRProgress(
                    stage: .extractingFeatures,
                    message: "Extracting log-mel features (window \(index + 1) of \(ranges.count))..."
                ))
                let featureStarted = ParakeetMonotonicClock.now()
                let mel = audioPreprocessor.logMelSpectrogram(from: Array(samples[range]))
                MLX.eval(mel)
                timings.featureExtractionSeconds += ParakeetMonotonicClock.seconds(since: featureStarted)
                mels.append(mel)
                debugMel(mel)
            }

            progressHandler?(ASRProgress(
                stage: .transcribing,
                message: batchRanges.count == 1
                    ? "Transcribing with Parakeet (window \(batchStart + 1) of \(ranges.count))..."
                    : "Transcribing with Parakeet (windows \(batchStart + 1)-\(batchEnd) of \(ranges.count))..."
            ))
            var modelTimings = ParakeetModelTimings()
            let decoded = try model.decodeWindows(mels, timings: &modelTimings)
            timings.encoderSeconds += modelTimings.encoderSeconds
            timings.decoderSeconds += modelTimings.decoderSeconds
            timings.alignmentSeconds += modelTimings.alignmentSeconds
            guard decoded.count == batchRanges.count else {
                throw ParakeetError.unexpectedDecoderBatchCount(
                    expected: batchRanges.count,
                    actual: decoded.count
                )
            }

            for (offset, result) in decoded.enumerated() {
                let mergeStarted = ParakeetMonotonicClock.now()
                let range = batchRanges[offset]
                let timeOffset = TimeInterval(range.lowerBound) / TimeInterval(sampleRate)
                let shifted = ParakeetCoreMLWindowing.shiftedTokens(from: result, by: timeOffset)
                mergedTokens = ParakeetAlignment.mergeLongestCommonSubsequence(
                    mergedTokens,
                    shifted,
                    overlapDuration: ParakeetCoreMLWindowing.overlapSeconds,
                    windowOverlap: timeOffset..<(timeOffset + ParakeetCoreMLWindowing.overlapSeconds)
                )
                timings.windowMergeSeconds += ParakeetMonotonicClock.seconds(since: mergeStarted)
            }
        }
        let alignmentStarted = ParakeetMonotonicClock.now()
        let result = ParakeetAlignment.sentencesToResult(
            ParakeetAlignment.tokensToSentences(mergedTokens)
        )
        timings.alignmentSeconds += ParakeetMonotonicClock.seconds(since: alignmentStarted)
        return result
    }

    private func decodeWindow(
        samples: [Float],
        model: any ParakeetDecodingModel,
        audioPreprocessor: ParakeetAudioPreprocessor,
        modelConfig: ParakeetModelConfig,
        windowIndex: Int = 0,
        windowCount: Int = 1,
        timings: inout ParakeetPipelineTimings,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) throws -> ParakeetAlignedResult {
        let windowSuffix = windowCount > 1
            ? " (window \(windowIndex + 1) of \(windowCount))"
            : ""

        progressHandler?(ASRProgress(
            stage: .extractingFeatures,
            message: "Extracting log-mel features\(windowSuffix)..."
        ))
        let featureStarted = ParakeetMonotonicClock.now()
        let mel = audioPreprocessor.logMelSpectrogram(from: samples)
        MLX.eval(mel)
        timings.featureExtractionSeconds += ParakeetMonotonicClock.seconds(since: featureStarted)

        debugMel(mel)

        progressHandler?(ASRProgress(
            stage: .transcribing,
            message: "Transcribing with Parakeet\(windowSuffix)..."
        ))
        var modelTimings = ParakeetModelTimings()
        let decoded = try model.decode(mel, timings: &modelTimings)
        timings.encoderSeconds += modelTimings.encoderSeconds
        timings.decoderSeconds += modelTimings.decoderSeconds
        timings.alignmentSeconds += modelTimings.alignmentSeconds
        return decoded.first ?? ParakeetAlignedResult(text: "", sentences: [])
    }

    private func debugMel(_ mel: MLXArray) {
        guard Self.debugEnabled else { return }
        let melMin = MLX.min(mel).item(Float.self)
        let melMax = MLX.max(mel).item(Float.self)
        let melMean = MLX.mean(mel).item(Float.self)
        FileHandle.standardError.write(
            Data("[ASR DEBUG] parakeet_mel shape=\(mel.shape) mean=\(melMean) min=\(melMin) max=\(melMax)\n".utf8)
        )
    }

    public func prepare(
        modelPath: String? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil
    ) async throws {
        let root = try await resolveModelRoot(modelPath: modelPath, progressHandler: progressHandler)
        if loadedModelPath != root.path {
            progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading Parakeet model..."))
            try await loadModel(from: root, progressHandler: progressHandler)
        }
    }

    public func unload() {
        model = nil
        audioPreprocessor = nil
        modelConfig = nil
        loadedModelPath = nil
        Memory.clearCache()
    }

    public func supportedLanguageCodes(modelPath: String? = nil) async throws -> Set<String> {
        let root = try await resolveModelRoot(modelPath: modelPath, progressHandler: nil)
        let resources = ParakeetResources(rootURL: root)
        let config = try ParakeetModelConfig.load(from: resources.configURL)
        return config.supportedLanguageCodes
    }

    private func resolveModelRoot(
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> URL {
        do {
            let resolved = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: modelPath ?? modelId,
                defaultModelID: ParakeetResources.defaultModelId,
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
            throw ParakeetError.downloadFailed(error.localizedDescription)
        }
    }

    private func mapModelLoaderError(_ error: PretrainedModelLoader.LoadError) -> ParakeetError {
        switch error {
        case .unsupportedModelId(let modelId):
            return .unsupportedModelId(modelId)
        case .missingFiles(let files):
            return .missingFiles(files)
        case .downloadFailed(let message):
            return .downloadFailed(message)
        }
    }

    private func loadModel(
        from rootURL: URL,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws {
        try await Stream.withNewDefaultStream(isolation: self) {
            try loadModelOnTaskSafeStream(
                from: rootURL,
                progressHandler: progressHandler
            )
        }
    }

    /// Model construction and weight evaluation create lazy MLX arrays that retain their
    /// originating stream. The live session decodes on a later cooperative-executor thread,
    /// so preparation must use MLX's cross-thread streams too—not only the decode call.
    private func loadModelOnTaskSafeStream(
        from rootURL: URL,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) throws {
        let resources = ParakeetResources(rootURL: rootURL)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw ParakeetError.missingFiles(missing.map(\.lastPathComponent))
        }

        let config = try ParakeetModelConfig.load(from: resources.configURL)
        let includeMLXEncoder: Bool
        switch executionProvider {
        case .mlx:
            guard config.packaging == .completeMLX else {
                throw ParakeetError.coreMLProviderRequired
            }
            includeMLXEncoder = true
        case .coreML:
            includeMLXEncoder = false
        }
        let model = ParakeetModelFactory.build(
            config: config,
            includeMLXEncoder: includeMLXEncoder
        )

        switch executionProvider {
        case .mlx:
            break
        case .coreML(let artifactURL):
            guard let baseModel = model as? ParakeetBaseModel,
                  config.variant == .tdt || config.variant == .tdtCTC else {
                throw ParakeetError.unsupportedCoreMLVariant(config.variant.rawValue)
            }
            if config.packaging == .coreMLHybrid,
               rootURL.resolvingSymlinksInPath().standardizedFileURL
               != artifactURL.resolvingSymlinksInPath().standardizedFileURL {
                throw ParakeetError.coreMLHybridArtifactMismatch
            }
            let loadedArtifact = try ParakeetCoreMLManifest.load(
                artifactURL: artifactURL,
                config: config
            )
            baseModel.externalEncoder = try ParakeetCoreMLEncoder(
                manifest: loadedArtifact.manifest,
                modelURL: loadedArtifact.modelURL
            )
            if loadedArtifact.manifest.coreMLDecoder != nil,
               let tdtModel = baseModel as? ParakeetTDTModel {
                tdtModel.externalDecoder = try ParakeetCoreMLDecoder(
                    artifactURL: artifactURL,
                    manifest: loadedArtifact.manifest,
                    config: config
                )
            }
        }

        guard let module = model as? Module else {
            throw ParakeetError.modelNotLoaded
        }

        progressHandler?(ASRProgress(stage: .loadingModel, message: "Loading Parakeet weights..."))

        let quantization: ModelWeightsLoader.QuantizationParams?
        if let bits = config.quantizationBits, let group = config.quantizationGroupSize {
            quantization = .init(bits: bits, groupSize: group)
        } else {
            quantization = nil
        }

        let mapper: (String, MLXArray) -> [(String, MLXArray)]
        let verification: Module.VerifyUpdate
        switch executionProvider {
        case .mlx:
            mapper = { key, value in [(key, value)] }
            verification = .noUnusedKeys
        case .coreML:
            mapper = { key, value in
                key.hasPrefix("encoder.") ? [] : [(key, value)]
            }
            verification = .all
        }

        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.modelIndexURL,
            singleURL: resources.modelWeightsURL,
            to: module,
            dtype: .bfloat16,
            verify: verification,
            mapper: mapper,
            quantization: quantization
        )

        // Parakeet uses BatchNorm/Dropout in the encoder stack; keep inference in eval mode.
        module.train(false)

        MLX.eval(module.parameters())

        self.model = model
        self.modelConfig = config
        self.audioPreprocessor = try ParakeetAudioPreprocessor(config: config.preprocessor)
        self.loadedModelPath = rootURL.path
    }
}

public enum ParakeetError: LocalizedError {
    case modelNotLoaded
    case unsupportedModelId(String)
    case missingFiles([String])
    case downloadFailed(String)
    case extractionFailed
    case unsupportedCoreMLVariant(String)
    case coreMLUnavailable
    case coreMLProviderRequired
    case missingMLXEncoder
    case coreMLHybridArtifactMismatch
    case decoderBatchTooLarge(actual: Int, maximum: Int)
    case unexpectedDecoderBatchCount(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Parakeet model is not loaded."
        case .unsupportedModelId(let modelId):
            return "Unsupported model id: \(modelId)"
        case .missingFiles(let files):
            return "Parakeet model files missing: \(files.joined(separator: ", "))"
        case .downloadFailed(let message):
            return message
        case .extractionFailed:
            return "Failed to prepare model files"
        case .unsupportedCoreMLVariant(let variant):
            return "The Core ML Parakeet encoder supports TDT checkpoints only; found \(variant)."
        case .coreMLUnavailable:
            return "The Core ML Parakeet encoder is unavailable on this platform."
        case .coreMLProviderRequired:
            return "This Parakeet package requires the Core ML provider."
        case .missingMLXEncoder:
            return "The Parakeet MLX encoder is not available in this package."
        case .coreMLHybridArtifactMismatch:
            return "The compact Parakeet decoder and Core ML encoder must come from the same artifact."
        case .decoderBatchTooLarge(let actual, let maximum):
            return "The Parakeet decoder batch has \(actual) windows; the maximum is \(maximum)."
        case .unexpectedDecoderBatchCount(let expected, let actual):
            return "The Parakeet decoder returned \(actual) windows; expected \(expected)."
        }
    }
}
