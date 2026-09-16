import Foundation
import MediaIO

public enum MarigoldV2GenerationError: Error, Equatable, LocalizedError, Sendable {
    case nativeResolutionConflictsWithMaximumEdge
    case maximumEdgeBelowAlignment(Int)
    case unknownCheckpoint(String)
    case inputNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .nativeResolutionConflictsWithMaximumEdge:
            "Native resolution and a maximum edge cannot be combined."
        case .maximumEdgeBelowAlignment:
            "Maximum edge must be at least \(MarigoldV2GenerationSettings.minimumMaximumEdge)."
        case .unknownCheckpoint(let name):
            "Unknown checkpoint '\(name)'. Expected one of: "
                + MarigoldV2GenerationSettings.checkpointNames.joined(separator: ", ")
        case .inputNotFound(let path):
            "Input image not found: \(path)"
        }
    }
}

/// Validates caller values before constructing the inference configuration.
public struct MarigoldV2GenerationSettings: Equatable, Sendable {
    public static let defaultCheckpoint = MarigoldV2Repository.installedCheckpoint
    public static let defaultMaximumEdge = MarigoldV2InferenceConfiguration.defaultMaximumEdge
    /// Both image sides align to the patch grid, so a smaller cap cannot be honored.
    public static let minimumMaximumEdge = MarigoldV2InferenceConfiguration.alignment
    public static let checkpointNames = MarigoldV2DepthCheckpoint.allCases.map(\.rawValue)
    public let configuration: MarigoldV2InferenceConfiguration

    /// `maximumEdge` bounds the longest inference edge; `nativeResolution` runs at the
    /// source size instead and cannot be combined with a cap. A nil `checkpoint` selects
    /// the managed install's checkpoint.
    public init(checkpoint: String? = nil, maximumEdge: Int? = nil, nativeResolution: Bool = false) throws {
        if nativeResolution && maximumEdge != nil {
            throw MarigoldV2GenerationError.nativeResolutionConflictsWithMaximumEdge
        }
        if let maximumEdge, maximumEdge < Self.minimumMaximumEdge {
            throw MarigoldV2GenerationError.maximumEdgeBelowAlignment(maximumEdge)
        }
        configuration = MarigoldV2InferenceConfiguration(
            checkpoint: try Self.checkpoint(named: checkpoint),
            maximumEdge: nativeResolution ? nil : (maximumEdge ?? Self.defaultMaximumEdge)
        )
    }

    /// The published variant named case-insensitively; nil or blank selects the managed
    /// install's checkpoint.
    public static func checkpoint(named name: String?) throws -> MarigoldV2DepthCheckpoint {
        guard let name, !name.isEmpty else { return defaultCheckpoint }
        guard let checkpoint = MarigoldV2DepthCheckpoint(rawValue: name.lowercased()) else {
            throw MarigoldV2GenerationError.unknownCheckpoint(name)
        }
        return checkpoint
    }
}

/// Carries the same settings from CLI translation into native execution.
public struct MarigoldV2GenerationRequest: Equatable, Sendable {
    public static let defaultModelID = ModelResolver.ModelID.visionDepthMarigoldV2.rawValue
    public let imageURL: URL
    public let outputDirectory: URL
    public let model: String?
    public let settings: MarigoldV2GenerationSettings

    public init(imageURL: URL, outputDirectory: URL, model: String? = nil, settings: MarigoldV2GenerationSettings) {
        self.imageURL = imageURL.standardizedFileURL
        self.outputDirectory = outputDirectory.standardizedFileURL
        self.model = model
        self.settings = settings
    }
}

/// Describes an observational plan without loading weights or creating output.
public struct MarigoldV2GenerationPlan: Equatable, Sendable {
    public let request: MarigoldV2GenerationRequest
    public let dimensions: VFXImageInputDimensions
    /// The aligned size the transformer runs at; artifacts are resampled back to `dimensions`.
    public let inferenceWidth: Int
    public let inferenceHeight: Int
    /// Whether the managed Marigold model is installed, regardless of `request.model`.
    public let managedModelInstalled: Bool

    public init(
        request: MarigoldV2GenerationRequest,
        imageWidth: Int,
        imageHeight: Int,
        managedModelInstalled: Bool
    ) throws {
        self.request = request
        dimensions = try VFXImageInputValidator.validate(
            width: imageWidth, height: imageHeight, path: request.imageURL.path
        )
        let inference = request.settings.configuration.inferenceSize(width: imageWidth, height: imageHeight)
        inferenceWidth = inference.width
        inferenceHeight = inference.height
        self.managedModelInstalled = managedModelInstalled
    }
}

/// Owns one request's runtime and awaited release. Admission stays with the caller.
public enum MarigoldV2GenerationOperation {
    public typealias ProgressHandler = @Sendable (String) -> Void

    public struct Runtime: Sendable {
        public let generate: @Sendable (MarigoldV2GenerationRequest, ProgressHandler?) async throws -> MarigoldV2RunResult
        public let unload: @Sendable () async -> Void

        public init(
            generate: @escaping @Sendable (MarigoldV2GenerationRequest, ProgressHandler?) async throws -> MarigoldV2RunResult,
            unload: @escaping @Sendable () async -> Void
        ) {
            self.generate = generate
            self.unload = unload
        }
    }

    public static func prepare(_ request: MarigoldV2GenerationRequest) throws -> MarigoldV2GenerationPlan {
        try Task.checkCancellation()
        guard FileManager.default.fileExists(atPath: request.imageURL.path) else {
            throw MarigoldV2GenerationError.inputNotFound(request.imageURL.path)
        }
        let size = try MediaImageIO.size(of: request.imageURL)
        let installed = ManagedModelResolver.resolveInstalledModel(id: MarigoldV2GenerationRequest.defaultModelID)
        return try MarigoldV2GenerationPlan(
            request: request, imageWidth: size.width, imageHeight: size.height, managedModelInstalled: installed != nil
        )
    }

    public static func execute(
        _ request: MarigoldV2GenerationRequest,
        prepareRuntime: @Sendable () throws -> Void = {},
        progress: ProgressHandler? = nil,
        makeRuntime: @Sendable () -> Runtime = nativeRuntime
    ) async throws -> MarigoldV2RunResult {
        // Recheck mutable input headers even when the caller previously planned.
        _ = try prepare(request)
        try prepareRuntime()
        try Task.checkCancellation()
        let runtime = makeRuntime()
        let result: MarigoldV2RunResult
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

    /// The generator snapshots the input before decoding it, so a file replaced during
    /// execution cannot change what the manifest records.
    public static func nativeRuntime() -> Runtime {
        let generator = MarigoldV2Generator()
        return Runtime(
            generate: { request, progress in
                try await generator.generate(
                    imageURL: request.imageURL, outputDirectory: request.outputDirectory,
                    model: request.model, configuration: request.settings.configuration, progress: progress
                )
            },
            unload: { await generator.unload() }
        )
    }
}
