import Foundation
import AudioCore

/// Parakeet adapter for the shared resident utterance coordinator.
public actor ParakeetASRLiveSession {
    private let session: ASRUtteranceLiveSession
    public nonisolated let events: AsyncThrowingStream<ASRLiveEvent, Error>

    public init(generator: ParakeetGenerator, request: ASRStreamingRequest,
                configuration: ASRLiveConfiguration = ASRLiveConfiguration()) {
        session = ASRUtteranceLiveSession(request: request, configuration: configuration) { samples, language in
            try await generator.transcribePrepared(samples: samples, language: language)
        }
        events = session.events
    }

    init(request: ASRStreamingRequest, configuration: ASRLiveConfiguration = ASRLiveConfiguration(),
         transcribe: @escaping @Sendable ([Float], String?) async throws -> ASRResult) {
        session = ASRUtteranceLiveSession(request: request, configuration: configuration, transcribe: transcribe)
        events = session.events
    }

    public func feed(samples: [Float]) async throws { try await session.feed(samples: samples) }
    public func finish(reason: ASRLiveFinishReason = .eof) async throws { try await session.finish(reason: reason) }
    public func cancel() async { await session.cancel() }
}
