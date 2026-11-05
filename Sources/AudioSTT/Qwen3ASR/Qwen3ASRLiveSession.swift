import Foundation
import AudioCore

public struct ASRLiveConfiguration: Sendable, Hashable {
    public var decodeIntervalMs: Int
    public var minDecodeAudioMs: Int
    public var silenceMs: Int
    public var preRollMs: Int
    public var maxUtteranceMs: Int
    public var maxQueuedAudioMs: Int

    public init(
        decodeIntervalMs: Int = 2_000,
        minDecodeAudioMs: Int = 1_600,
        silenceMs: Int = 900,
        preRollMs: Int = 300,
        maxUtteranceMs: Int = 30_000,
        maxQueuedAudioMs: Int = 5_000
    ) {
        self.decodeIntervalMs = decodeIntervalMs
        self.minDecodeAudioMs = minDecodeAudioMs
        self.silenceMs = silenceMs
        self.preRollMs = preRollMs
        self.maxUtteranceMs = maxUtteranceMs
        self.maxQueuedAudioMs = maxQueuedAudioMs
    }
}

public enum ASRLiveFinishReason: String, Sendable, Hashable, Codable {
    case eof
    case stopped
    case cancelled
    case maxDuration = "max_duration"
}

public struct ASRLiveTranscript: Sendable, Hashable, Codable {
    public let utteranceId: String
    public let revision: Int
    public let text: String
    public let startMs: Int
    public let endMs: Int
}

public enum ASRLiveEvent: Sendable, Hashable {
    case partial(ASRLiveTranscript)
    case commit(ASRLiveTranscript)
    case stats(ASRStreamingStats, queuedAudioMs: Int)
    case final(ASRLiveFinishReason)
}

public typealias Qwen3ASRLiveConfiguration = ASRLiveConfiguration
public typealias Qwen3ASRLiveFinishReason = ASRLiveFinishReason
public typealias Qwen3ASRLiveTranscript = ASRLiveTranscript
public typealias Qwen3ASRLiveEvent = ASRLiveEvent

/// Quality-oriented utterance coordinator for live Qwen ASR.
///
/// Audio is segmented with an adaptive RMS noise floor. Each committed utterance
/// gets a fresh Qwen streaming session so decode cost is bounded over long calls.
public actor Qwen3ASRLiveSession {
    public nonisolated let events: AsyncThrowingStream<Qwen3ASRLiveEvent, Error>

    private let request: ASRStreamingRequest
    private let configuration: Qwen3ASRLiveConfiguration
    private let makeSession: @Sendable (ASRStreamingRequest) async throws -> any ASRStreamingSession
    private let audioClock = Qwen3ASRLiveAudioClock()
    private let continuation: AsyncThrowingStream<Qwen3ASRLiveEvent, Error>.Continuation

    private var session: (any ASRStreamingSession)?
    private var eventTask: Task<(result: ASRResult, revision: Int), Error>?
    private var pendingSamples: [Float] = []
    private var processingTask: Task<Void, Never>?
    private var preRoll: [Float] = []
    private var totalSampleCount = 0
    private var utteranceStartSample = 0
    private var lastSpeechSample = 0
    private var utteranceId = UUID().uuidString.lowercased()
    private var revision = 0
    private var noiseRMS: Float = 0.002
    private var phase: ASRLivePhase = .accepting
    private var finalizationTask: Task<Void, Error>?
    private var cancellationTask: Task<Void, Never>?

    public init(
        generator: Qwen3ASRGenerator,
        request: ASRStreamingRequest,
        configuration: Qwen3ASRLiveConfiguration = Qwen3ASRLiveConfiguration()
    ) {
        self.request = request
        self.configuration = configuration
        self.makeSession = { request in
            try await generator.makeStreamingSession(request)
        }
        var captured: AsyncThrowingStream<Qwen3ASRLiveEvent, Error>.Continuation?
        self.events = AsyncThrowingStream { captured = $0 }
        self.continuation = captured!
    }

    init(
        request: ASRStreamingRequest,
        configuration: Qwen3ASRLiveConfiguration = Qwen3ASRLiveConfiguration(),
        makeSession: @escaping @Sendable (ASRStreamingRequest) async throws -> any ASRStreamingSession
    ) {
        self.request = request
        self.configuration = configuration
        self.makeSession = makeSession
        var captured: AsyncThrowingStream<Qwen3ASRLiveEvent, Error>.Continuation?
        self.events = AsyncThrowingStream { captured = $0 }
        self.continuation = captured!
    }

    public func feed(samples: [Float]) async throws {
        if let error = phase.failure { throw error }
        guard phase.isAccepting else {
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

    private func startPendingAudioProcessing() {
        guard processingTask == nil else { return }
        processingTask = Task { [weak self] in
            await self?.drainPendingAudio()
        }
    }

    private func drainPendingAudio() async {
        let analysisWindowSamples = max(1, request.sampleRate / 10)
        do {
            while !pendingSamples.isEmpty, phase.canProcess {
                try Task.checkCancellation()
                let count = min(pendingSamples.count, analysisWindowSamples)
                let window = Array(pendingSamples.prefix(count))
                pendingSamples.removeFirst(count)
                try await feedAnalysisWindow(window)
            }
        } catch {
            if !phase.isCancelled {
                await fail(error)
            }
        }
        processingTask = nil
    }

    private func feedAnalysisWindow(_ samples: [Float]) async throws {
        let chunkStart = totalSampleCount
        totalSampleCount += samples.count
        await audioClock.set(totalSampleCount)
        guard phase.canProcess else { return }
        let rms = rootMeanSquare(samples)
        let speechThreshold = max(Float(0.005), noiseRMS * Float(3.5))
        let containsSpeech = rms >= speechThreshold

        if !containsSpeech {
            noiseRMS = (noiseRMS * Float(0.97)) + (rms * Float(0.03))
        }

        if session == nil {
            appendPreRoll(samples)
            guard containsSpeech else { return }
            let buffered = preRoll
            preRoll.removeAll(keepingCapacity: true)
            utteranceStartSample = max(0, totalSampleCount - buffered.count)
            lastSpeechSample = totalSampleCount
            try await startUtterance()
            guard phase.canProcess else { return }
            try await session?.feed(samples: buffered)
            return
        }

        try await session?.feed(samples: samples)
        guard phase.canProcess else { return }
        if containsSpeech {
            lastSpeechSample = totalSampleCount
        }

        let silenceSamples = request.sampleRate * configuration.silenceMs / 1_000
        let maximumSamples = request.sampleRate * configuration.maxUtteranceMs / 1_000
        let utteranceSamples = totalSampleCount - utteranceStartSample
        if !containsSpeech, totalSampleCount - lastSpeechSample >= silenceSamples {
            try await commitUtterance(endSample: lastSpeechSample)
        } else if utteranceSamples >= maximumSamples {
            try await commitUtterance(endSample: chunkStart + samples.count)
        }
    }

    public func finish(reason: ASRLiveFinishReason = .eof) async throws {
        if let cancellationTask { return await cancellationTask.value }
        if let finalizationTask { return try await awaitFinalization(finalizationTask) }
        if let error = phase.failure {
            if let processingTask { await processingTask.value }
            throw error
        }
        guard phase.isAccepting else { return }
        phase = .finishing
        let task = Task { try await self.finalize(reason: reason) }
        finalizationTask = task
        try await awaitFinalization(task)
    }

    private func awaitFinalization(_ task: Task<Void, Error>) async throws {
        let result = await task.result
        if let cancellationTask { await cancellationTask.value }
        try result.get()
    }

    private func finalize(reason: ASRLiveFinishReason) async throws {
        do {
            if let processingTask { await processingTask.value }
            if let error = phase.failure { throw error }
            guard phase.canProcess else { return }
            if session != nil { try await commitUtterance(endSample: totalSampleCount, startNext: false) }
            guard phase.canProcess else { return }
            phase = .finished
            emitFinal(reason)
            continuation.finish()
        } catch {
            guard !phase.isCancelled else { return }
            await fail(error)
            throw error
        }
    }

    public func cancel() async {
        if let cancellationTask { return await cancellationTask.value }
        if phase.failure != nil {
            if let processingTask { await processingTask.value }
            if let finalizationTask { _ = await finalizationTask.result }
            return
        }
        guard phase.canProcess else { return }
        phase = .cancelling
        let task = Task { await self.cancelAndDrain() }
        cancellationTask = task
        emitFinal(.cancelled)
        continuation.finish()
        await task.value
    }

    private func cancelAndDrain() async {
        processingTask?.cancel()
        finalizationTask?.cancel()
        pendingSamples.removeAll(keepingCapacity: false)
        let reader = eventTask
        reader?.cancel()
        await session?.cancel()
        if let processingTask { await processingTask.value }
        if let finalizationTask { _ = await finalizationTask.result }
        if let reader { _ = await reader.result }
        session = nil
        eventTask = nil
        preRoll.removeAll(keepingCapacity: false)
        phase = .cancelled
    }

    private func startUtterance() async throws {
        utteranceId = UUID().uuidString.lowercased()
        revision = 0
        let next = try await makeSession(request)
        guard phase.canProcess else {
            await next.cancel()
            return
        }
        session = next
        let id = utteranceId
        let start = utteranceStartSample
        let sampleRate = request.sampleRate
        let audioClock = audioClock
        let continuation = continuation
        eventTask = Task {
            var finalResult: ASRResult?
            var localRevision = 0
            var localLastPartial = ""
            for try await event in next.events {
                switch event {
                case .partial(let text):
                    let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !normalized.isEmpty, normalized != localLastPartial else { continue }
                    localRevision += 1
                    localLastPartial = normalized
                    let end = await audioClock.current()
                    continuation.yield(.partial(Qwen3ASRLiveTranscript(
                        utteranceId: id,
                        revision: localRevision,
                        text: normalized,
                        startMs: start * 1_000 / sampleRate,
                        endMs: end * 1_000 / sampleRate
                    )))
                case .stats(let stats):
                    let end = await audioClock.current()
                    let audioMs = Int((stats.totalAudioSeconds * 1_000).rounded())
                    let currentUtteranceMs = max(0, end - start) * 1_000 / sampleRate
                    continuation.yield(.stats(stats, queuedAudioMs: max(0, currentUtteranceMs - audioMs)))
                case .final(let result):
                    finalResult = result
                }
            }
            return (
                finalResult ?? ASRResult(text: "", language: nil, duration: 0),
                localRevision
            )
        }
    }

    private func commitUtterance(endSample: Int, startNext: Bool = true) async throws {
        guard let active = session, let eventTask else { return }
        try await active.finish(requiredSampleCount: max(0, endSample - utteranceStartSample))
        guard phase.canProcess else { return }
        let output = try await eventTask.value
        guard phase.canProcess else { return }
        let result = output.result
        let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty {
            revision = output.revision + 1
            continuation.yield(.commit(Qwen3ASRLiveTranscript(
                utteranceId: utteranceId,
                revision: revision,
                text: text,
                startMs: utteranceStartSample * 1_000 / request.sampleRate,
                endMs: max(utteranceStartSample, endSample) * 1_000 / request.sampleRate
            )))
        }
        session = nil
        self.eventTask = nil
        preRoll.removeAll(keepingCapacity: true)
        if startNext, phase.isAccepting {
            utteranceStartSample = totalSampleCount
        }
    }

    private func fail(_ error: Error) async {
        phase = .failed(error)
        pendingSamples.removeAll(keepingCapacity: false)
        continuation.finish(throwing: error)
        let reader = eventTask
        reader?.cancel()
        await session?.cancel()
        if let reader { _ = await reader.result }
        session = nil
        eventTask = nil
        preRoll.removeAll(keepingCapacity: false)
    }

    private func appendPreRoll(_ samples: [Float]) {
        preRoll.append(contentsOf: samples)
        let maximum = request.sampleRate * configuration.preRollMs / 1_000
        if preRoll.count > maximum {
            preRoll.removeFirst(preRoll.count - maximum)
        }
    }

    private func emitFinal(_ reason: Qwen3ASRLiveFinishReason) {
        continuation.yield(.final(reason))
    }

    private func rootMeanSquare(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let sum = samples.reduce(Float.zero) { $0 + ($1 * $1) }
        return sqrt(sum / Float(samples.count))
    }
}

private actor Qwen3ASRLiveAudioClock {
    private var sampleCount = 0

    func set(_ value: Int) {
        sampleCount = value
    }

    func current() -> Int {
        sampleCount
    }
}
