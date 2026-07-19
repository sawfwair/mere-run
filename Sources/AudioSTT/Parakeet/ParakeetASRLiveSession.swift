import Foundation
import AudioCore

/// Low-latency utterance coordinator for Parakeet using the live ASR V1 event contract.
///
/// Parakeet is not an autoregressive streaming model. This session keeps one model
/// resident and re-decodes the bounded active utterance to produce partial revisions.
public actor ParakeetASRLiveSession {
    public nonisolated let events: AsyncThrowingStream<ASRLiveEvent, Error>

    private let request: ASRStreamingRequest
    private let configuration: ASRLiveConfiguration
    private let transcribe: @Sendable ([Float], String?) async throws -> ASRResult
    private let continuation: AsyncThrowingStream<ASRLiveEvent, Error>.Continuation

    private var pendingSamples: [Float] = []
    private var processingTask: Task<Void, Never>?
    private var processingError: Error?
    private var preRoll: [Float] = []
    private var utteranceSamples: [Float] = []
    private var totalSampleCount = 0
    private var utteranceStartSample = 0
    private var lastSpeechSample = 0
    private var lastDecodeSampleCount = 0
    private var lastDecodedResult: ASRResult?
    private var lastPartialText = ""
    private var decodeCount = 0
    private var utteranceId = UUID().uuidString.lowercased()
    private var revision = 0
    private var noiseRMS: Float = 0.002
    private var finished = false
    private var cancelled = false
    private var finalEmitted = false

    public init(
        generator: ParakeetGenerator,
        request: ASRStreamingRequest,
        configuration: ASRLiveConfiguration = ASRLiveConfiguration()
    ) {
        self.request = request
        self.configuration = configuration
        self.transcribe = { samples, language in
            try await generator.transcribePrepared(samples: samples, language: language)
        }
        var captured: AsyncThrowingStream<ASRLiveEvent, Error>.Continuation?
        self.events = AsyncThrowingStream { captured = $0 }
        self.continuation = captured!
    }

    init(
        request: ASRStreamingRequest,
        configuration: ASRLiveConfiguration = ASRLiveConfiguration(),
        transcribe: @escaping @Sendable ([Float], String?) async throws -> ASRResult
    ) {
        self.request = request
        self.configuration = configuration
        self.transcribe = transcribe
        var captured: AsyncThrowingStream<ASRLiveEvent, Error>.Continuation?
        self.events = AsyncThrowingStream { captured = $0 }
        self.continuation = captured!
    }

    public func feed(samples: [Float]) async throws {
        if let processingError { throw processingError }
        guard !finished else {
            throw ASRStreamingError.invalidState("Cannot feed live ASR after finish.")
        }
        guard !samples.isEmpty else { return }
        let maximumPendingSamples = request.sampleRate * configuration.maxQueuedAudioMs / 1_000
        guard pendingSamples.count + samples.count <= maximumPendingSamples else {
            throw ASRStreamingError.invalidInput("backpressure_exceeded: live ASR input exceeded the bounded queue.")
        }

        pendingSamples.append(contentsOf: samples)
        startPendingAudioProcessing()
    }

    public func finish(reason: ASRLiveFinishReason = .eof) async throws {
        if let processingError { throw processingError }
        guard !finished else { return }
        finished = true
        if let processingTask {
            await processingTask.value
        }
        if let processingError { throw processingError }
        if !utteranceSamples.isEmpty {
            try await commitUtterance(endSample: totalSampleCount)
        }
        emitFinal(reason)
        continuation.finish()
    }

    public func cancel() {
        guard !finished else { return }
        cancelled = true
        finished = true
        processingTask?.cancel()
        pendingSamples.removeAll(keepingCapacity: false)
        utteranceSamples.removeAll(keepingCapacity: false)
        emitFinal(.cancelled)
        continuation.finish()
    }

    private func startPendingAudioProcessing() {
        guard processingTask == nil else { return }
        processingTask = Task { [weak self] in
            await self?.drainPendingAudio()
        }
    }

    private func drainPendingAudio() async {
        let analysisWindowSamples = max(1, request.sampleRate / 10)
        do {
            while !pendingSamples.isEmpty {
                try Task.checkCancellation()
                let count = min(pendingSamples.count, analysisWindowSamples)
                let window = Array(pendingSamples.prefix(count))
                pendingSamples.removeFirst(count)
                try await feedAnalysisWindow(window)
            }
        } catch is CancellationError where cancelled {
            // Cancellation intentionally discards the active utterance.
        } catch {
            processingError = error
            finished = true
            pendingSamples.removeAll(keepingCapacity: false)
            utteranceSamples.removeAll(keepingCapacity: false)
            continuation.finish(throwing: error)
        }
        processingTask = nil
    }

    private func feedAnalysisWindow(_ samples: [Float]) async throws {
        totalSampleCount += samples.count
        let rms = rootMeanSquare(samples)
        let speechThreshold = max(Float(0.005), noiseRMS * Float(3.5))
        let containsSpeech = rms >= speechThreshold

        if !containsSpeech {
            noiseRMS = (noiseRMS * Float(0.97)) + (rms * Float(0.03))
        }

        if utteranceSamples.isEmpty {
            appendPreRoll(samples)
            guard containsSpeech else { return }
            utteranceSamples = preRoll
            preRoll.removeAll(keepingCapacity: true)
            utteranceStartSample = max(0, totalSampleCount - utteranceSamples.count)
            lastSpeechSample = totalSampleCount
            lastDecodeSampleCount = 0
            lastDecodedResult = nil
            lastPartialText = ""
            decodeCount = 0
            utteranceId = UUID().uuidString.lowercased()
            revision = 0
        } else {
            utteranceSamples.append(contentsOf: samples)
            if containsSpeech {
                lastSpeechSample = totalSampleCount
            }
        }

        let minimumSamples = request.sampleRate * configuration.minDecodeAudioMs / 1_000
        let decodeIntervalSamples = request.sampleRate * configuration.decodeIntervalMs / 1_000
        if decodeCount == 0,
           utteranceSamples.count >= minimumSamples,
           utteranceSamples.count - lastDecodeSampleCount >= decodeIntervalSamples {
            try await decodePartial()
        }

        let silenceSamples = request.sampleRate * configuration.silenceMs / 1_000
        let maximumSamples = request.sampleRate * configuration.maxUtteranceMs / 1_000
        if !containsSpeech, totalSampleCount - lastSpeechSample >= silenceSamples {
            try await commitUtterance(endSample: lastSpeechSample)
        } else if utteranceSamples.count >= maximumSamples {
            try await commitUtterance(endSample: totalSampleCount)
        }
    }

    private func decodePartial() async throws {
        let samples = utteranceSamples
        let started = ContinuousClock.now
        let result = try await transcribe(samples, request.language)
        let latency = started.duration(to: .now)
        lastDecodeSampleCount = samples.count
        lastDecodedResult = result
        decodeCount += 1
        let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty, text != lastPartialText {
            revision += 1
            lastPartialText = text
            continuation.yield(.partial(ASRLiveTranscript(
                utteranceId: utteranceId,
                revision: revision,
                text: text,
                startMs: utteranceStartSample * 1_000 / request.sampleRate,
                endMs: totalSampleCount * 1_000 / request.sampleRate
            )))
        }
        continuation.yield(.stats(
            ASRStreamingStats(
                decodeCount: decodeCount,
                totalAudioSeconds: Double(samples.count) / Double(request.sampleRate),
                lastDecodeLatencyMs: milliseconds(latency),
                tokensGenerated: text.split(whereSeparator: \.isWhitespace).count
            ),
            queuedAudioMs: pendingSamples.count * 1_000 / request.sampleRate
        ))
    }

    private func commitUtterance(endSample: Int) async throws {
        let requiredCount = max(0, min(utteranceSamples.count, endSample - utteranceStartSample))
        let samples = Array(utteranceSamples.prefix(requiredCount))
        let result: ASRResult
        if requiredCount == lastDecodeSampleCount, let lastDecodedResult {
            result = lastDecodedResult
        } else if let lastDecodedResult, lastDecodeSampleCount > 0 {
            let overlapSamples = request.sampleRate / 2
            let tailStart = max(0, lastDecodeSampleCount - overlapSamples)
            let tail = Array(samples.dropFirst(tailStart))
            let tailResult = try await transcribe(tail, request.language)
            if tailStart == 0 {
                result = tailResult
            } else {
                result = ASRResult(
                    text: mergeTranscripts(lastDecodedResult.text, tailResult.text),
                    language: tailResult.language ?? lastDecodedResult.language,
                    duration: Double(requiredCount) / Double(request.sampleRate)
                )
            }
        } else {
            result = try await transcribe(samples, request.language)
        }
        let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty {
            revision += 1
            continuation.yield(.commit(ASRLiveTranscript(
                utteranceId: utteranceId,
                revision: revision,
                text: text,
                startMs: utteranceStartSample * 1_000 / request.sampleRate,
                endMs: max(utteranceStartSample, endSample) * 1_000 / request.sampleRate
            )))
        }
        utteranceSamples.removeAll(keepingCapacity: true)
        preRoll.removeAll(keepingCapacity: true)
        lastDecodeSampleCount = 0
        lastDecodedResult = nil
        lastPartialText = ""
        decodeCount = 0
    }

    private func appendPreRoll(_ samples: [Float]) {
        preRoll.append(contentsOf: samples)
        let maximum = request.sampleRate * configuration.preRollMs / 1_000
        if preRoll.count > maximum {
            preRoll.removeFirst(preRoll.count - maximum)
        }
    }

    private func emitFinal(_ reason: ASRLiveFinishReason) {
        guard !finalEmitted else { return }
        finalEmitted = true
        continuation.yield(.final(reason))
    }

    private func rootMeanSquare(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let sum = samples.reduce(Float.zero) { $0 + ($1 * $1) }
        return sqrt(sum / Float(samples.count))
    }

    private func milliseconds(_ duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds) * 1_000
            + Double(components.attoseconds) / 1_000_000_000_000_000
    }

    private func mergeTranscripts(_ prefix: String, _ suffix: String) -> String {
        let prefixWords = prefix.split(whereSeparator: \.isWhitespace).map(String.init)
        let suffixWords = suffix.split(whereSeparator: \.isWhitespace).map(String.init)
        guard !prefixWords.isEmpty else { return suffixWords.joined(separator: " ") }
        guard !suffixWords.isEmpty else { return prefixWords.joined(separator: " ") }

        let maximumOverlap = min(prefixWords.count, suffixWords.count)
        let overlap = stride(from: maximumOverlap, through: 1, by: -1).first { count in
            zip(prefixWords.suffix(count), suffixWords.prefix(count)).allSatisfy {
                normalizedWord($0) == normalizedWord($1)
            }
        } ?? 0
        return (prefixWords.dropLast(overlap) + suffixWords).joined(separator: " ")
    }

    private func normalizedWord(_ value: String) -> String {
        String(value.lowercased().filter { $0.isLetter || $0.isNumber })
    }
}
