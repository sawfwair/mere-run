import Foundation

/// Supplies model audio without publishing output files or acquiring admission permits.
public protocol SpeechSynthesisExecutor: Sendable {
    func generate(
        _ request: TTSRequest,
        progressHandler: (@Sendable (TTSProgress) -> Void)?
    ) async throws -> AudioWaveform

    func generateStream(
        _ request: TTSRequest,
        options: TTSStreamingOptions
    ) -> AsyncThrowingStream<TTSStreamingEvent, Error>
}

public extension SpeechSynthesisExecutor {
    func generateStream(
        _ request: TTSRequest,
        options: TTSStreamingOptions
    ) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        AsyncThrowingStream { $0.finish(throwing: SpeechSynthesisError.streamingUnsupported) }
    }
}

public struct SpeechSynthesisOutcome: Sendable {
    public let result: TTSResult
    public let artifact: AudioExportFile
}

/// Owns validated execution and artifact publication. The caller owns model residency and admission.
public enum SpeechSynthesisOperation {
    public static func execute(
        _ plan: SpeechSynthesisPlan,
        executor: any SpeechSynthesisExecutor,
        progressHandler: (@Sendable (TTSProgress) -> Void)? = nil
    ) async throws -> SpeechSynthesisOutcome {
        guard plan.streamingOptions == nil else { throw SpeechSynthesisError.wrongExecutionMode }
        try plan.validateForExecution()
        let audio = try await executor.generate(plan.request, progressHandler: progressHandler)
        try Task.checkCancellation()
        guard !audio.samples.isEmpty else { throw SpeechSynthesisError.emptyAudio }
        progressHandler?(TTSProgress(stage: .saving, message: "Saving audio..."))
        try createOutputDirectory(plan.request.outputURL)
        let artifact = try AudioExportService.write(audio, plan: plan.exportPlan, to: plan.request.outputURL)
        return SpeechSynthesisOutcome(
            result: TTSResult(audioURL: artifact.url, duration: audio.duration, sampleRate: audio.sampleRate),
            artifact: artifact
        )
    }

    public static func stream(
        _ plan: SpeechSynthesisPlan,
        executor: any SpeechSynthesisExecutor
    ) throws -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        guard let options = plan.streamingOptions else { throw SpeechSynthesisError.wrongExecutionMode }
        try plan.validateForExecution()
        return AsyncThrowingStream { continuation in
            let task = Task {
                var writer: AudioExportStream?
                defer { writer?.cancel() }
                do {
                    var sampleRate: Int?
                    var completion: TTSResult?
                    for try await event in executor.generateStream(plan.request, options: options) {
                        try Task.checkCancellation()
                        guard completion == nil else {
                            throw SpeechSynthesisError.invalidStream("Speech stream emitted events after its final result.")
                        }
                        switch event {
                        case .token:
                            if case .terminated = continuation.yield(event) { throw CancellationError() }
                        case .audioChunk(let samples, let rate):
                            if let sampleRate, sampleRate != rate {
                                throw SpeechSynthesisError.invalidStream("Speech stream changed its sample rate.")
                            }
                            guard !samples.isEmpty else { continue }
                            if writer == nil {
                                // Validate the format before creating the output directory.
                                _ = try AudioWaveform(interleaved: samples, channels: 1, sampleRate: rate)
                                try createOutputDirectory(plan.request.outputURL)
                                writer = try AudioExportStream(plan: plan.exportPlan, to: plan.request.outputURL, sampleRate: rate)
                                sampleRate = rate
                            }
                            try writer?.append(samples: samples)
                            if case .terminated = continuation.yield(event) { throw CancellationError() }
                        case .completed(let result):
                            completion = result
                        }
                    }
                    try Task.checkCancellation()
                    guard let completion else {
                        throw SpeechSynthesisError.invalidStream("Streaming TTS completed without a final result.")
                    }
                    guard let writer, let sampleRate else { throw SpeechSynthesisError.emptyAudio }
                    guard completion.sampleRate == sampleRate,
                          completion.audioURL.standardizedFileURL == plan.request.outputURL.standardizedFileURL else {
                        throw SpeechSynthesisError.invalidStream("Speech stream result does not match its output file and sample rate.")
                    }
                    let artifact = try writer.finish()
                    let result = TTSResult(audioURL: artifact.url,
                                           duration: Double(artifact.statistics.sampleCount) / Double(sampleRate),
                                           sampleRate: sampleRate)
                    continuation.yield(.completed(result: result))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable reason in
                if case .cancelled = reason { task.cancel() }
            }
        }
    }

    private static func createOutputDirectory(_ output: URL) throws {
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
    }
}
