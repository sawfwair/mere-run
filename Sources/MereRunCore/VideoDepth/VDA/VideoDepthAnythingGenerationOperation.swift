import Foundation

/// Resolves bounded frame defaults through the native video-depth limits.
public struct VideoDepthAnythingGenerationSettings: Equatable, Sendable {
    public let inputSize: Int
    public let maximumFrameCount: Int

    public init(
        inputSize: Int = VideoDepthAnythingLimits.defaultInputSize,
        maximumFrameCount: Int? = nil
    ) throws {
        self.maximumFrameCount = try VideoDepthAnythingLimits.validateRequest(
            inputSize: inputSize, maximumFrameCount: maximumFrameCount
        )
        self.inputSize = inputSize
    }
}

public struct VideoDepthAnythingGenerationRequest: Equatable, Sendable {
    public let videoURL: URL
    public let outputDirectory: URL
    public let model: String?
    public let settings: VideoDepthAnythingGenerationSettings

    public init(videoURL: URL, outputDirectory: URL, model: String? = nil, settings: VideoDepthAnythingGenerationSettings) {
        self.videoURL = videoURL.standardizedFileURL
        self.outputDirectory = outputDirectory.standardizedFileURL
        self.model = model
        self.settings = settings
    }
}

/// Owns per-request execution and awaited cleanup within the caller's admission.
public enum VideoDepthAnythingGenerationOperation {
    public typealias ProgressHandler = @Sendable (VideoDepthAnythingProgress) -> Void

    public struct Runtime: Sendable {
        public let generate: @Sendable (VideoDepthAnythingGenerationRequest, ProgressHandler?) async throws -> VideoDepthAnythingRunResult
        public let unload: @Sendable () async -> Void

        public init(
            generate: @escaping @Sendable (VideoDepthAnythingGenerationRequest, ProgressHandler?) async throws -> VideoDepthAnythingRunResult,
            unload: @escaping @Sendable () async -> Void
        ) {
            self.generate = generate
            self.unload = unload
        }
    }

    /// Checks the current file and byte ceiling without decoding or creating output.
    public static func prepare(_ request: VideoDepthAnythingGenerationRequest) throws {
        try Task.checkCancellation()
        guard FileManager.default.fileExists(atPath: request.videoURL.path) else {
            throw VideoDepthAnythingGeneratorError.inputVideoNotFound(request.videoURL.path)
        }
        let bytes = try ModelArtifactPin.fileByteCount(request.videoURL)
        guard bytes <= VideoDepthAnythingLimits.maximumEncodedVideoBytes else {
            throw VFXImageInputValidationError.encodedByteLimitExceeded(
                path: request.videoURL.path, bytes: bytes, maximum: VideoDepthAnythingLimits.maximumEncodedVideoBytes
            )
        }
    }

    /// Uses native bounded snapshot/decode admission and releases its temporary inputs.
    public static func preflight(
        _ request: VideoDepthAnythingGenerationRequest
    ) async throws -> VideoDepthAnythingPreflightResult {
        try prepare(request)
        return try await VideoDepthAnythingGenerator.preflight(
            videoURL: request.videoURL, model: request.model,
            inputSize: request.settings.inputSize, maximumFrameCount: request.settings.maximumFrameCount
        )
    }

    public static func execute(
        _ request: VideoDepthAnythingGenerationRequest,
        prepareRuntime: @Sendable () throws -> Void = {},
        progress: ProgressHandler? = nil,
        makeRuntime: @Sendable () -> Runtime = nativeRuntime
    ) async throws -> VideoDepthAnythingRunResult {
        try prepare(request)
        try prepareRuntime()
        try Task.checkCancellation()
        let runtime = makeRuntime()
        let result: VideoDepthAnythingRunResult
        do {
            result = try await runtime.generate(request, progress)
            try Task.checkCancellation()
        } catch {
            await runtime.unload()
            throw error
        }
        await runtime.unload()
        try Task.checkCancellation()
        return result
    }

    public static func nativeRuntime() -> Runtime {
        let generator = VideoDepthAnythingGenerator()
        return Runtime(
            generate: { request, progress in
                try await generator.generate(
                    videoURL: request.videoURL, outputDirectory: request.outputDirectory,
                    model: request.model, inputSize: request.settings.inputSize,
                    maximumFrameCount: request.settings.maximumFrameCount, progress: progress
                )
            },
            unload: { await generator.unload() }
        )
    }
}
