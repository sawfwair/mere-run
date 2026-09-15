import Foundation

/// Shares native view limits, camera conditioning, and validated scene-export settings.
public struct DepthAnything3GenerationSettings: Equatable, Sendable {
    public static let defaultProcessResolution = 504
    public let processResolution: Int
    public let referenceViewStrategy: DepthAnything3ReferenceViewStrategy
    public let knownCameras: [DepthAnything3KnownCamera]?
    public let export: MultiViewGeometryExportConfiguration

    public init(
        processResolution: Int = defaultProcessResolution,
        referenceViewStrategy: DepthAnything3ReferenceViewStrategy = .saddleBalanced,
        knownCameras: [DepthAnything3KnownCamera]? = nil,
        confidencePercentile: Double = MultiViewGeometryExportConfiguration.defaultConfidencePercentile,
        maximumPointCount: Int = MultiViewGeometryExportConfiguration.defaultMaximumPointCount
    ) throws {
        try DepthAnything3Limits.validateRequest(viewCount: 1, processResolution: processResolution)
        if let knownCameras {
            for (index, camera) in knownCameras.enumerated() {
                try DepthAnything3CameraValidation.validate(camera, index: index)
            }
        }
        self.processResolution = processResolution
        self.referenceViewStrategy = referenceViewStrategy
        self.knownCameras = knownCameras
        self.export = try MultiViewGeometryExportConfiguration(
            confidencePercentile: confidencePercentile, maximumPointCount: maximumPointCount
        )
    }

    public func validate(viewCount: Int) throws {
        try DepthAnything3Limits.validateRequest(viewCount: viewCount, processResolution: processResolution)
        if let knownCameras, knownCameras.count != viewCount {
            throw DepthAnything3GeneratorError.cameraCountMismatch(images: viewCount, cameras: knownCameras.count)
        }
    }
}

public struct DepthAnything3GenerationRequest: Equatable, Sendable {
    public let imageURLs: [URL]
    public let outputDirectory: URL
    public let model: String?
    public let settings: DepthAnything3GenerationSettings

    public init(imageURLs: [URL], outputDirectory: URL, model: String? = nil, settings: DepthAnything3GenerationSettings) {
        self.imageURLs = imageURLs.map(\.standardizedFileURL)
        self.outputDirectory = outputDirectory.standardizedFileURL
        self.model = model
        self.settings = settings
    }
}

public struct DepthAnything3GenerationResult: Sendable {
    public let run: DepthAnything3RunResult
    public let export: MultiViewGeometryExportResult
    public let exportSeconds: Double
}

/// Owns per-request execution and awaited cleanup within the caller's admission.
public enum DepthAnything3GenerationOperation {
    public typealias ProgressHandler = @Sendable (DepthAnything3Progress) -> Void

    public struct Runtime: Sendable {
        public let generate: @Sendable (DepthAnything3GenerationRequest, ProgressHandler?) async throws -> DepthAnything3RunResult
        public let unload: @Sendable () async -> Void

        public init(
            generate: @escaping @Sendable (DepthAnything3GenerationRequest, ProgressHandler?) async throws -> DepthAnything3RunResult,
            unload: @escaping @Sendable () async -> Void
        ) {
            self.generate = generate
            self.unload = unload
        }
    }

    /// Inspects current headers without creating output or resolving a checkpoint.
    public static func prepare(_ request: DepthAnything3GenerationRequest) throws {
        try Task.checkCancellation()
        try request.settings.validate(viewCount: request.imageURLs.count)
        for url in request.imageURLs where !FileManager.default.fileExists(atPath: url.path) {
            throw DepthAnything3GeneratorError.imageNotFound(url.path)
        }
        let dimensions = try DepthAnything3Limits.validateImageURLs(request.imageURLs)
        try DepthAnything3CameraValidation.validate(
            request.settings.knownCameras, sourceDimensions: dimensions
        )
    }

    public static func execute(
        _ request: DepthAnything3GenerationRequest,
        prepareRuntime: @Sendable () throws -> Void = {},
        progress: ProgressHandler? = nil,
        makeRuntime: @Sendable () -> Runtime = nativeRuntime
    ) async throws -> DepthAnything3GenerationResult {
        try prepare(request)
        try prepareRuntime()
        try Task.checkCancellation()
        let runtime = makeRuntime()
        let result: DepthAnything3GenerationResult
        do {
            let run = try await runtime.generate(request, progress)
            try Task.checkCancellation()
            let exportStart = Date()
            let export = try MultiViewGeometryExporter.export(
                run: run, outputDirectory: request.outputDirectory, configuration: request.settings.export
            )
            result = DepthAnything3GenerationResult(
                run: run, export: export, exportSeconds: Date().timeIntervalSince(exportStart)
            )
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
        let generator = DepthAnything3Generator()
        return Runtime(
            generate: { request, progress in
                try await generator.generate(
                    imageURLs: request.imageURLs, model: request.model,
                    knownCameras: request.settings.knownCameras,
                    referenceViewStrategy: request.settings.referenceViewStrategy,
                    processResolution: request.settings.processResolution, progress: progress
                )
            },
            unload: { await generator.unload() }
        )
    }
}
