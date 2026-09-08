import Foundation
import MLX
import MLXNN
import AudioCore
import AudioCodecs
import MereRunCore

extension ParakeetGenerator {
    func decodeMeasured(
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
            samples: samples,
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
                let windowIndex = batchStart + offset
                let previousEnd = windowIndex > 0 ? ranges[windowIndex - 1].upperBound : 0
                let overlapEnd = TimeInterval(previousEnd) / TimeInterval(sampleRate)
                let shifted = ParakeetCoreMLWindowing.shiftedTokens(from: result, by: timeOffset)
                mergedTokens = ParakeetAlignment.mergeLongestCommonSubsequence(
                    mergedTokens,
                    shifted,
                    overlapDuration: ParakeetCoreMLWindowing.overlapSeconds,
                    windowOverlap: timeOffset..<overlapEnd
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
}
