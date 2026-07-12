import Foundation

public enum Wan2ResourcesError: LocalizedError, Sendable {
    case invalidConfiguration(URL, String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let url, let reason):
            return "Invalid Wan2 configuration at \(url.path): \(reason)"
        }
    }
}

public struct Wan2TI2VConfiguration: Codable, Hashable, Sendable {
    public let modelType: String
    public let modelVersion: String
    public let patchSize: [Int]
    public let textLen: Int
    public let inDim: Int
    public let dim: Int
    public let ffnDim: Int
    public let textDim: Int
    public let outDim: Int
    public let numHeads: Int
    public let numLayers: Int
    public let vaeStride: [Int]
    public let vaeZDim: Int
    public let sampleShift: Double
    public let sampleSteps: Int
    public let sampleGuideScale: Double
    public let sampleFPS: Int
    public let frameNum: Int
    public let maxArea: Int
    public let quantization: Wan2QuantizationConfiguration?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case modelVersion = "model_version"
        case patchSize = "patch_size"
        case textLen = "text_len"
        case inDim = "in_dim"
        case dim
        case ffnDim = "ffn_dim"
        case textDim = "text_dim"
        case outDim = "out_dim"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case vaeStride = "vae_stride"
        case vaeZDim = "vae_z_dim"
        case sampleShift = "sample_shift"
        case sampleSteps = "sample_steps"
        case sampleGuideScale = "sample_guide_scale"
        case sampleFPS = "sample_fps"
        case frameNum = "frame_num"
        case maxArea = "max_area"
        case quantization
    }

    public func validationIssues() -> [String] {
        var issues: [String] = []
        if modelType != "ti2v" { issues.append("model_type must be ti2v") }
        if modelVersion != "2.2" { issues.append("model_version must be 2.2") }
        if patchSize != [1, 2, 2] { issues.append("patch_size must be [1, 2, 2]") }
        if inDim != 48 || outDim != 48 { issues.append("in_dim and out_dim must both be 48") }
        if dim != 3_072 { issues.append("dim must be 3072") }
        if ffnDim != 14_336 { issues.append("ffn_dim must be 14336") }
        if numHeads != 24 { issues.append("num_heads must be 24") }
        if numLayers != 30 { issues.append("num_layers must be 30") }
        if vaeStride != [4, 16, 16] { issues.append("vae_stride must be [4, 16, 16]") }
        if vaeZDim != 48 { issues.append("vae_z_dim must be 48") }
        if maxArea != 704 * 1_280 { issues.append("max_area must be 901120") }
        return issues
    }
}

public struct Wan2QuantizationConfiguration: Codable, Hashable, Sendable {
    public let bits: Int
    public let groupSize: Int

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
    }
}

public struct Wan2Resources: Hashable, Sendable {
    public static let modelID = "video-wan22-ti2v-5b-mlx"
    public static let managedRepoID = "SceneWorks/wan2.2-ti2v-5b-mlx"
    public static let managedRevision = "bb1b055249614cf9d7cf4373fbdbc184b77dee88"
    public static let officialRepoID = "Wan-AI/Wan2.2-TI2V-5B"
    public static let officialRevision = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
    public static let defaultNegativePrompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    public static let snapshotPatterns = [
        "LICENSE",
        "README.md",
        "config.json",
        "model.safetensors",
        "t5_encoder.safetensors",
        "tokenizer.json",
        "vae.safetensors",
    ]

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var configURL: URL { rootURL.appendingPathComponent("config.json") }
    public var transformerURL: URL { rootURL.appendingPathComponent("model.safetensors") }
    public var textEncoderURL: URL { rootURL.appendingPathComponent("t5_encoder.safetensors") }
    public var tokenizerURL: URL { rootURL.appendingPathComponent("tokenizer.json") }
    public var vaeURL: URL { rootURL.appendingPathComponent("vae.safetensors") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        [configURL, transformerURL, textEncoderURL, tokenizerURL, vaeURL]
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public func loadConfiguration() throws -> Wan2TI2VConfiguration {
        let configuration: Wan2TI2VConfiguration
        do {
            configuration = try JSONDecoder().decode(
                Wan2TI2VConfiguration.self,
                from: Data(contentsOf: configURL)
            )
        } catch {
            throw Wan2ResourcesError.invalidConfiguration(configURL, error.localizedDescription)
        }
        let issues = configuration.validationIssues()
        guard issues.isEmpty else {
            throw Wan2ResourcesError.invalidConfiguration(configURL, issues.joined(separator: "; "))
        }
        return configuration
    }
}
