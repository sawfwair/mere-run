import Foundation
import MLX
import AudioCore
import AudioCodecs

struct Qwen3ASRStreamingMel: @unchecked Sendable {
    let value: MLXArray
}

struct Qwen3ASRStreamingDecodeOutput: Sendable {
    let result: ASRResult
    let tokensGenerated: Int
}

actor Qwen3ASRStreamingSession: ASRStreamingSession {
    nonisolated let events: AsyncThrowingStream<ASRStreamingEvent, Error>

    private let request: ASRStreamingRequest
    private let decode: @Sendable ([Float], Qwen3ASRStreamingMel) async throws -> Qwen3ASRStreamingDecodeOutput
    private let continuation: AsyncThrowingStream<ASRStreamingEvent, Error>.Continuation
    private let melExtractor = MelSpectrogram()

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
    private var forcePendingDecode = false
    private var forcedDecodeSampleCount: Int?
    private var decodeTask: Task<Void, Never>?
    private var decodeFailure: Error?

    init(
        request: ASRStreamingRequest,
        decode: @escaping @Sendable ([Float], Qwen3ASRStreamingMel) async throws -> Qwen3ASRStreamingDecodeOutput
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
        let maximumQueuedSamples = request.sampleRate * request.maxQueuedAudioMs / 1_000
        let queuedSamples = melBuffer.sampleCount - decodeCadence.lastDecodedSampleCount
        guard queuedSamples <= maximumQueuedSamples else {
            throw ASRStreamingError.invalidInput(
                "backpressure_exceeded: more than \(request.maxQueuedAudioMs) ms of audio is waiting for decode."
            )
        }
        guard decodeCadence.shouldDecode(bufferedSampleCount: melBuffer.sampleCount) else { return }
        scheduleDecode(force: false)
        // Give the decode task a chance to snapshot the newest audio without
        // making ingestion wait for model inference.
        await Task.yield()
    }

    func finish() async throws {
        try await completeFinalization(
            requiredSampleCount: melBuffer.sampleCount,
            alwaysForceDecode: true
        )
    }

    func finish(requiredSampleCount: Int) async throws {
        try await completeFinalization(
            requiredSampleCount: requiredSampleCount,
            alwaysForceDecode: false
        )
    }

    private func completeFinalization(
        requiredSampleCount: Int,
        alwaysForceDecode: Bool
    ) async throws {
        if canceled || didFinish { return }
        guard requiredSampleCount >= 0, requiredSampleCount <= melBuffer.sampleCount else {
            throw ASRStreamingError.invalidInput(
                "requiredSampleCount must be within the streamed audio range."
            )
        }

        didFinish = true
        if let decodeTask {
            await decodeTask.value
        }
        if let decodeFailure {
            throw decodeFailure
        }
        if alwaysForceDecode || decodeCadence.lastDecodedSampleCount < requiredSampleCount {
            scheduleDecode(force: true, sampleCount: requiredSampleCount)
            if let decodeTask {
                await decodeTask.value
            }
            if let decodeFailure {
                throw decodeFailure
            }
        }
        emitFinalIfNeeded()
        continuation.finish()
    }

    func cancel() async {
        guard !canceled else { return }
        canceled = true
        didFinish = true
        decodeTask?.cancel()
        continuation.finish()
    }

    private func scheduleDecode(force: Bool, sampleCount: Int? = nil) {
        pendingDecode = true
        forcePendingDecode = forcePendingDecode || force
        if force {
            forcedDecodeSampleCount = sampleCount ?? melBuffer.sampleCount
        }
        guard decodeTask == nil else { return }

        decodeTask = Task { [weak self] in
            await self?.runDecodeLoop()
        }
    }

    private func runDecodeLoop() async {
        do {
            try await performDecodeLoop()
        } catch is CancellationError {
            // Cancellation is terminal and intentionally does not surface as a decode error.
        } catch {
            decodeFailure = error
            continuation.finish(throwing: error)
        }
        decodeTask = nil
    }

    private func performDecodeLoop() async throws {
        if decodeInFlight { return }

        decodeInFlight = true
        defer { decodeInFlight = false }

        while pendingDecode || forcePendingDecode {
            let forceNextDecode = forcePendingDecode
            let requestedSampleCount = forceNextDecode ? forcedDecodeSampleCount : nil
            pendingDecode = false
            forcePendingDecode = false
            forcedDecodeSampleCount = nil
            if canceled { return }

            let snapshot = if let requestedSampleCount {
                melBuffer.snapshotSamples(count: requestedSampleCount)
            } else {
                melBuffer.snapshotSamples()
            }
            guard !snapshot.isEmpty else {
                continue
            }

            if !decodeCadence.shouldDecode(bufferedSampleCount: snapshot.count, force: forceNextDecode) {
                continue
            }

            let start = Date()
            let melSpec = Qwen3ASRStreamingMel(
                value: melBuffer.extract(using: melExtractor, sampleCount: snapshot.count)
            )
            let output = try await decode(snapshot, melSpec)
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
