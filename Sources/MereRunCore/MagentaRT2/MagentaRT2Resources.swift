import Foundation

public struct MagentaRT2Resources: Hashable, Sendable {
    public static let upstreamRepoId = "google/magenta-realtime-2"
    public static let upstreamRevision = "010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc"
    public static let smallModelId = ModelResolver.ModelID.magentaRT2Small.rawValue
    public static let baseModelId = ModelResolver.ModelID.magentaRT2Base.rawValue
    public static let sampleRate = 48_000
    public static let frameRate = 25
    public static let frameSamples = 1_920
    public static let channels = 2

    public let rootURL: URL
    public let modelID: String

    public init(rootURL: URL, modelID: String) {
        self.rootURL = rootURL
        self.modelID = modelID
    }

    public var modelName: String {
        Self.modelName(for: modelID)
    }

    public var modelURL: URL {
        rootURL.appendingPathComponent("models/\(modelName)/\(modelName).mlxfn", isDirectory: false)
    }

    public var modelStateURL: URL {
        rootURL.appendingPathComponent("models/\(modelName)/\(modelName)_state.safetensors", isDirectory: false)
    }

    public var resourcesURL: URL {
        rootURL.appendingPathComponent("resources", isDirectory: true)
    }

    public var spectrostreamEncoderURL: URL {
        resourcesURL.appendingPathComponent("spectrostream/spectrostream_encoder.mlxfn", isDirectory: false)
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        requiredRelativePaths()
            .map { rootURL.appendingPathComponent($0, isDirectory: false) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public static func modelName(for modelID: String) -> String {
        modelID == baseModelId ? "mrt2_base" : "mrt2_small"
    }

    public static func isMagentaRT2Model(_ value: String?) -> Bool {
        guard let value else { return false }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return trimmed == smallModelId
            || trimmed == baseModelId
            || trimmed == upstreamRepoId.lowercased()
    }

    public static func resolve(
        requestedModel: String?,
        defaultModelID: String = smallModelId,
        fileManager: FileManager = .default,
        progress: (@Sendable (PretrainedModelLoader.ProgressEvent) -> Void)? = nil
    ) async throws -> MagentaRT2Resources {
        if let explicit = explicitPath(from: requestedModel, fileManager: fileManager) {
            let modelID = inferredModelID(in: explicit, fileManager: fileManager) ?? defaultModelID
            return MagentaRT2Resources(rootURL: explicit, modelID: modelID)
        }

        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: requestedModel,
            defaultModelID: defaultModelID,
            fileManager: fileManager,
            progress: progress
        )
        let modelID = resolution.spec.id
        return MagentaRT2Resources(rootURL: resolution.url, modelID: modelID)
    }

    public static func inferredModelID(in rootURL: URL, fileManager: FileManager = .default) -> String? {
        let base = rootURL.appendingPathComponent("models/mrt2_base/mrt2_base.mlxfn")
        if fileManager.fileExists(atPath: base.path) {
            return baseModelId
        }
        let small = rootURL.appendingPathComponent("models/mrt2_small/mrt2_small.mlxfn")
        if fileManager.fileExists(atPath: small.path) {
            return smallModelId
        }
        return nil
    }

    public static func looksLikeMagentaRT2Root(_ rootURL: URL, fileManager: FileManager = .default) -> Bool {
        guard inferredModelID(in: rootURL, fileManager: fileManager) != nil else {
            return false
        }
        let musicCoCa = rootURL.appendingPathComponent("resources/musiccoca", isDirectory: true)
        return fileManager.fileExists(atPath: musicCoCa.path)
    }

    private func requiredRelativePaths() -> [String] {
        [
            "models/\(modelName)/\(modelName).mlxfn",
            "models/\(modelName)/\(modelName)_state.safetensors",
            "resources/musiccoca/audio_preprocessor.tflite",
            "resources/musiccoca/mapper.tflite",
            "resources/musiccoca/music_encoder.tflite",
            "resources/musiccoca/pretrained_vector_quantizer.tflite",
            "resources/musiccoca/spm.model",
            "resources/musiccoca/text_encoder.tflite",
            "resources/spectrostream/decoder.safetensors",
            "resources/spectrostream/encoder.safetensors",
            "resources/spectrostream/quantizer.safetensors",
            "resources/spectrostream/spectrostream_encoder.mlxfn",
        ]
    }

    private static func explicitPath(from requestedModel: String?, fileManager: FileManager) -> URL? {
        guard let requestedModel else { return nil }
        let trimmed = requestedModel.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let url = URL(fileURLWithPath: trimmed).standardizedFileURL
        return fileManager.fileExists(atPath: url.path) ? url : nil
    }
}

public struct MagentaRT2Controls: Hashable, Sendable {
    public var styleConditioning: MagentaRT2StyleConditioning
    public var temperature: Float
    public var topK: Int32
    public var cfgMusicCoCa: Float
    public var cfgNotes: Float
    public var cfgDrums: Float
    public var drumless: Bool
    public var unmaskWidth: Int32
    public var seedRotation: Int32
    public var prefillSilence: Bool
    public var prefillDurationSeconds: Float

    public init(
        styleConditioning: MagentaRT2StyleConditioning = .streaming,
        temperature: Float = 1.0,
        topK: Int32 = 100,
        cfgMusicCoCa: Float = 3.0,
        cfgNotes: Float = 5.0,
        cfgDrums: Float = 1.0,
        drumless: Bool = false,
        unmaskWidth: Int32 = 0,
        seedRotation: Int32 = 0,
        prefillSilence: Bool = false,
        prefillDurationSeconds: Float = 1.64
    ) {
        self.styleConditioning = styleConditioning
        self.temperature = temperature
        self.topK = topK
        self.cfgMusicCoCa = cfgMusicCoCa
        self.cfgNotes = cfgNotes
        self.cfgDrums = cfgDrums
        self.drumless = drumless
        self.unmaskWidth = unmaskWidth
        self.seedRotation = seedRotation
        self.prefillSilence = prefillSilence
        self.prefillDurationSeconds = prefillDurationSeconds
    }
}

public enum MagentaRT2StyleConditioning: String, CaseIterable, Hashable, Sendable {
    case streaming
    case full

    public var musicCoCaTokenCount: Int32 {
        switch self {
        case .streaming:
            return 6
        case .full:
            return 12
        }
    }
}

public struct MagentaRT2Frame: Hashable, Sendable {
    public let left: [Float]
    public let right: [Float]

    public init(left: [Float], right: [Float]) {
        self.left = left
        self.right = right
    }

    public var interleavedStereo: [Float] {
        let count = min(left.count, right.count)
        var samples: [Float] = []
        samples.reserveCapacity(count * 2)
        for index in 0..<count {
            samples.append(left[index])
            samples.append(right[index])
        }
        return samples
    }
}

public enum MagentaRT2Error: LocalizedError, Sendable {
    case unsupportedRuntime
    case invalidDuration(Float)
    case invalidControl(String)
    case missingAssets([URL])
    case engineInitializationFailed(String)
    case modelLoadFailed(String)
    case promptEncodingFailed
    case generationFailed(frame: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedRuntime:
            return "Magenta RT2 requires Apple Silicon macOS and a built vendor/magentart.xcframework. Run scripts/rebuild_magentart_xcframework.sh, then rebuild mere.run."
        case .invalidDuration(let duration):
            return "Duration must be > 0 seconds; received \(duration)."
        case .invalidControl(let message):
            return message
        case .missingAssets(let urls):
            return "Magenta RT2 model is incomplete: \(urls.map(\.path).joined(separator: ", "))"
        case .engineInitializationFailed(let path):
            return "Failed to initialize Magenta RT2 assets from \(path)."
        case .modelLoadFailed(let path):
            return "Failed to load Magenta RT2 model from \(path)."
        case .promptEncodingFailed:
            return "Magenta RT2 prompt encoding failed."
        case .generationFailed(let frame):
            return "Magenta RT2 generation failed at frame \(frame)."
        }
    }
}
