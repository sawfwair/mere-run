import Foundation

/// User-supplied image settings before model defaults and adapter recipes are applied.
public struct ImageGenerationOptions: Codable, Sendable, Hashable {
    public var prompt: String
    public var negativePrompt: String?
    public var outputURL: URL
    public var width: Int
    public var height: Int
    public var steps: Int?
    public var guidanceScale: Double?
    public var seed: UInt64?
    public var inputImage: URL?
    public var referenceImages: [URL]
    public var strength: Double?
    public var keepOriginalAspect: Bool
    public var maxSequenceLength: Int
    public var loras: [ImageLoRAReference]
    public var sigmaShift: Float?
    public var sigmas: [Float]?
    public var kreaConditioningRebalance: Krea2ConditioningRebalance?
    public var kreaBaseQuantizationBits: Int?
    public var mask: URL?
    public var outpaint: ImageOutpaintInsets?
    public var maskFeather: Int

    public init(
        prompt: String,
        negativePrompt: String? = nil,
        outputURL: URL,
        width: Int = 1024,
        height: Int = 1024,
        steps: Int? = nil,
        guidanceScale: Double? = nil,
        seed: UInt64? = nil,
        inputImage: URL? = nil,
        referenceImages: [URL] = [],
        strength: Double? = nil,
        keepOriginalAspect: Bool = false,
        maxSequenceLength: Int = 512,
        loras: [ImageLoRAReference] = [],
        sigmaShift: Float? = nil,
        sigmas: [Float]? = nil,
        kreaConditioningRebalance: Krea2ConditioningRebalance? = nil,
        kreaBaseQuantizationBits: Int? = nil,
        mask: URL? = nil,
        outpaint: ImageOutpaintInsets? = nil,
        maskFeather: Int = 8
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.outputURL = outputURL
        self.width = width
        self.height = height
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.seed = seed
        self.inputImage = inputImage
        self.referenceImages = referenceImages
        self.strength = strength
        self.keepOriginalAspect = keepOriginalAspect
        self.maxSequenceLength = maxSequenceLength
        self.loras = loras
        self.sigmaShift = sigmaShift
        self.sigmas = sigmas
        self.kreaConditioningRebalance = kreaConditioningRebalance
        self.kreaBaseQuantizationBits = kreaBaseQuantizationBits
        self.mask = mask
        self.outpaint = outpaint
        self.maskFeather = maskFeather
    }
}

public struct ImageLoRAReference: Codable, Sendable, Hashable {
    public let raw: String
    public let reference: String
    public let scale: Double

    public init(raw: String, reference: String, scale: Double) {
        self.raw = raw
        self.reference = reference
        self.scale = scale
    }

    public static func parse(_ arguments: [String], defaultScale: Double) throws -> [Self] {
        guard defaultScale.isFinite else {
            throw ImageGenerationIssue("lora_scale_invalid", "--lora-scale must be finite")
        }
        return try arguments.map { raw in
            let separator = raw.lastIndex(of: "=")
            let reference: String
            let scale: Double
            if let separator {
                reference = String(raw[..<separator])
                let rawScale = String(raw[raw.index(after: separator)...])
                guard let parsed = Double(rawScale), parsed.isFinite else {
                    throw ImageGenerationIssue("lora_scale_invalid", "--lora scale must be finite (got \(rawScale)).")
                }
                scale = parsed
            } else {
                reference = raw
                scale = defaultScale
            }
            guard !reference.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ImageGenerationIssue("lora_reference_invalid", "--lora must be PATH_OR_ID[=SCALE].")
            }
            return Self(raw: raw, reference: reference, scale: scale)
        }
    }
}

/// Stable diagnostic identifiers let transports present errors without parsing prose.
public struct ImageGenerationIssue: Codable, LocalizedError, Equatable, Sendable {
    public let code: String
    public let message: String
    public var errorDescription: String? { message }

    public init(_ code: String, _ message: String) {
        self.code = code
        self.message = message
    }
}

public enum ImageGenerationBackend: String, CaseIterable, Codable, Sendable {
    case flux1, flux2Klein, zImageTurbo, hiDreamO1, senseNovaU15, krea2, ideogram4, qwenImageEdit

    public init(manifest: MereRunModelManifest) throws {
        switch manifest.family {
        case .flux1: self = .flux1
        case .klein: self = .flux2Klein
        case .zimage: self = .zImageTurbo
        case .hidream: self = .hiDreamO1
        case .senseNova: self = .senseNovaU15
        case .krea: self = .krea2
        case .ideogram: self = .ideogram4
        case .qwen where manifest.engine == .qwenImageEdit: self = .qwenImageEdit
        default:
            throw ImageGenerationIssue("model_family_unsupported", "Unsupported image generation model: \(manifest.id).")
        }
    }
}

/// Compatibility choices belong to the caller, while their implementation has one owner.
public struct ImageGenerationPolicy: Codable, Sendable, Hashable {
    public var kleinUsesManifestDefaults: Bool
    public var kleinInputAsReference: Bool
    public var fallbackSteps: Int
    public var fallbackGuidanceScale: Double

    public init(
        kleinUsesManifestDefaults: Bool = true,
        kleinInputAsReference: Bool = true,
        fallbackSteps: Int = 4,
        fallbackGuidanceScale: Double = 1
    ) {
        self.kleinUsesManifestDefaults = kleinUsesManifestDefaults
        self.kleinInputAsReference = kleinInputAsReference
        self.fallbackSteps = fallbackSteps
        self.fallbackGuidanceScale = fallbackGuidanceScale
    }

    public static let manifestDefaults = Self()
}

/// Classifies a local path or managed identifier without downloading anything.
/// Callers retain their own missing-model guidance and observational behavior.
public enum ImageGenerationModelSelection: Sendable {
    case local(URL)
    case managed(ModelResolver.ModelID)
    case unknown(String)

    public init(_ selector: String, fileManager: FileManager = .default) {
        let path = URL(fileURLWithPath: selector).standardizedFileURL
        if fileManager.fileExists(atPath: path.path) {
            self = .local(path)
        } else if let id = ModelResolver.ModelID(rawValue: selector) {
            self = .managed(id)
        } else {
            self = .unknown(selector)
        }
    }

    public func resolveRoot(fileManager: FileManager = .default) throws -> URL {
        switch self {
        case .local(let url): return url
        case .managed(let id): return try ModelResolver(fileManager: fileManager).resolve(id).rootURL
        case .unknown(let selector):
            throw ImageGenerationIssue("model_unknown", "Model path not found and not a known model id: \(selector).")
        }
    }
}
