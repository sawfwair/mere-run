import Foundation

public enum ACEStepTask: String, CaseIterable, Codable, Hashable, Sendable {
    case textToMusic = "text2music"
    case repaint
    case cover
    case coverNoFSQ = "cover-nofsq"
    case extract
    case lego
    case complete

    public var requiresSourceAudio: Bool {
        self != .textToMusic
    }

    public var locksDurationToSource: Bool {
        switch self {
        case .cover, .coverNoFSQ, .repaint, .extract, .lego:
            true
        case .textToMusic, .complete:
            false
        }
    }

    public var usesFSQCoverHints: Bool {
        self == .cover
    }

    public var skipsLanguageModel: Bool {
        switch self {
        case .cover, .coverNoFSQ, .repaint, .extract:
            true
        case .textToMusic, .lego, .complete:
            false
        }
    }

    public var requiresBaseCheckpoint: Bool {
        switch self {
        case .extract, .lego, .complete:
            true
        case .textToMusic, .repaint, .cover, .coverNoFSQ:
            false
        }
    }

    public func instruction(
        trackName: String? = nil,
        completeTrackClasses: [String] = []
    ) -> String {
        switch self {
        case .textToMusic:
            return "Fill the audio semantic mask based on the given conditions:"
        case .repaint:
            return "Repaint the mask area based on the given conditions:"
        case .cover, .coverNoFSQ:
            return "Generate audio semantic tokens based on the given conditions:"
        case .extract:
            if let trackName = Self.nonEmpty(trackName) {
                return "Extract the \(trackName.uppercased()) track from the audio:"
            } else {
                return "Extract the track from the audio:"
            }
        case .lego:
            if let trackName = Self.nonEmpty(trackName) {
                return "Generate the \(trackName.uppercased()) track based on the audio context:"
            } else {
                return "Generate the track based on the audio context:"
            }
        case .complete:
            let classes = completeTrackClasses
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            if classes.isEmpty {
                return "Complete the input track:"
            } else {
                return "Complete the input track with \(classes.map { $0.uppercased() }.joined(separator: " | ")):"
            }
        }
    }

    private static func nonEmpty(_ value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }
}

public enum ACEStepCheckpointVariant: String, CaseIterable, Codable, Hashable, Sendable {
    case base
    case sft
    case turbo
    case xlBase = "xl-base"
    case xlSFT = "xl-sft"
    case xlTurbo = "xl-turbo"

    public var isXL: Bool {
        switch self {
        case .xlBase, .xlSFT, .xlTurbo:
            true
        case .base, .sft, .turbo:
            false
        }
    }

    public var isTurbo: Bool {
        switch self {
        case .turbo, .xlTurbo:
            true
        case .base, .sft, .xlBase, .xlSFT:
            false
        }
    }

    public var supportsBaseTasks: Bool {
        switch self {
        case .base, .xlBase:
            true
        case .sft, .turbo, .xlSFT, .xlTurbo:
            false
        }
    }

    /// Official upstream inference defaults for the distilled and
    /// classifier-free-guidance checkpoint families.
    public var defaultInferenceSteps: Int {
        isTurbo ? 8 : 50
    }

    public var defaultShift: Float {
        isTurbo ? 3 : 1
    }

    public var defaultGuidanceScale: Float {
        isTurbo ? 1 : 7
    }

    public func supports(_ task: ACEStepTask) -> Bool {
        !task.requiresBaseCheckpoint || supportsBaseTasks
    }

    public func validate(_ task: ACEStepTask) throws {
        guard supports(task) else {
            throw ACEStepCapabilityError.unsupportedTask(task: task, variant: self)
        }
    }

    public static func detect(
        modelRootURL: URL,
        config: ACEStepConfig
    ) -> ACEStepCheckpointVariant {
        let name = modelRootURL.lastPathComponent.lowercased()
        let isXL = name.contains("xl-")
            || name.contains("-xl")
            || config.hiddenSize > 2_048

        if config.isTurbo || name.contains("turbo") {
            return isXL ? .xlTurbo : .turbo
        }
        if name.contains("sft") {
            return isXL ? .xlSFT : .sft
        }
        return isXL ? .xlBase : .base
    }

    public static func load(
        modelRootURL: URL,
        fileManager: FileManager = .default
    ) throws -> ACEStepCheckpointVariant {
        let resources = ACEStepResources(rootURL: modelRootURL)
        let missing = resources.validate(fileManager: fileManager)
        if !missing.isEmpty {
            throw ACEStepCapabilityError.missingCheckpointFiles(missing)
        }
        let config = try JSONDecoder().decode(
            ACEStepConfig.self,
            from: Data(contentsOf: resources.configURL)
        )
        return detect(modelRootURL: modelRootURL, config: config)
    }
}

public enum ACEStepCapabilityError: LocalizedError {
    case missingCheckpointFiles([URL])
    case unsupportedTask(task: ACEStepTask, variant: ACEStepCheckpointVariant)

    public var errorDescription: String? {
        switch self {
        case .missingCheckpointFiles(let urls):
            let list = urls.map(\.path).joined(separator: "\n")
            return "Missing ACE-Step checkpoint resources:\n\(list)"
        case .unsupportedTask(let task, let variant):
            return "ACE-Step task '\(task.rawValue)' is not supported by \(variant.rawValue) checkpoints. "
                + "Use an ACE-Step Base checkpoint for extract, lego, and complete."
        }
    }
}

public enum ACEStepChunkMaskMode: String, CaseIterable, Codable, Hashable, Sendable {
    case auto
    case explicit
}

public enum ACEStepRepaintMode: String, CaseIterable, Codable, Hashable, Sendable {
    case conservative
    case balanced
    case aggressive
}

public struct ACEStepRepaintConfiguration: Codable, Hashable, Sendable {
    public var startSeconds: Float
    public var endSeconds: Float
    public var chunkMaskMode: ACEStepChunkMaskMode
    public var mode: ACEStepRepaintMode
    public var strength: Float

    public init(
        startSeconds: Float = 0,
        endSeconds: Float = -1,
        chunkMaskMode: ACEStepChunkMaskMode = .auto,
        mode: ACEStepRepaintMode = .balanced,
        strength: Float = 0.5
    ) {
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.chunkMaskMode = chunkMaskMode
        self.mode = mode
        self.strength = strength
    }

    public var injectionRatio: Float {
        switch mode {
        case .aggressive:
            0
        case .conservative:
            1
        case .balanced:
            1 - clampedStrength
        }
    }

    public var latentCrossfadeFrames: Int {
        switch mode {
        case .aggressive:
            0
        case .conservative:
            25
        case .balanced:
            Int((25 * Double(1 - clampedStrength)).rounded(.toNearestOrEven))
        }
    }

    public var waveformCrossfadeSeconds: Float {
        switch mode {
        case .aggressive:
            0
        case .conservative:
            0.05
        case .balanced:
            0.05 * (1 - clampedStrength)
        }
    }

    public func latentRange(
        totalFrames: Int,
        framesPerSecond: Float = 25
    ) -> Range<Int> {
        precondition(totalFrames > 0, "totalFrames must be positive.")
        let start = max(0, min(Int(startSeconds * framesPerSecond), totalFrames - 1))
        let resolvedEnd = endSeconds > startSeconds
            ? endSeconds
            : Float(totalFrames) / framesPerSecond
        let end = max(start + 1, min(Int(resolvedEnd * framesPerSecond), totalFrames))
        return start..<end
    }

    private var clampedStrength: Float {
        min(max(strength, 0), 1)
    }
}
