import Foundation

public enum MiniMaxH3ResourcesError: LocalizedError, Sendable {
    case invalidConfiguration(URL, String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let url, let reason):
            return "Invalid MiniMax-H3 configuration at \(url.path): \(reason)"
        }
    }
}

public struct MiniMaxH3QuantizationConfiguration: Codable, Hashable, Sendable {
    public let bits: Int
    public let groupSize: Int
    public let mode: String

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
        case mode
    }
}

public struct MiniMaxH3Configuration: Decodable, Hashable, Sendable {
    public let modelType: String
    public let task: String
    public let hiddenSize: Int
    public let layerCount: Int
    public let refinerLayerCount: Int
    public let attentionHeadCount: Int
    public let attentionHeadDimension: Int
    public let feedForwardSize: Int
    public let videoLatentChannels: Int
    public let audioLatentChannels: Int
    public let patchSize: [Int]
    public let textDimension: Int
    public let timeFrequencyDimension: Int
    public let timeEmbeddingHiddenSize: Int
    public let timeEmbeddingDimension: Int
    public let videoFlowShift: Float
    public let audioFlowShift: Float
    public let sampleSteps: Int
    public let quantization: MiniMaxH3QuantizationConfiguration?
    public let textEncoderQuantization: MiniMaxH3QuantizationConfiguration?

    private struct Transformer: Decodable {
        let hiddenSize: Int
        let layerCount: Int
        let attentionHeadCount: Int
        let attentionHeadDimension: Int
        let feedForwardSize: Int
        let videoLatentChannels: Int
        let audioLatentChannels: Int
        let textDimension: Int
        let timeEmbeddingHiddenSize: Int?
        let timeEmbeddingDimension: Int
        let ropeFrequencyCount: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case layerCount = "num_layers"
            case attentionHeadCount = "num_attention_heads"
            case attentionHeadDimension = "attention_head_dim"
            case feedForwardSize = "ffn_hidden_size"
            case videoLatentChannels = "latents_dim"
            case audioLatentChannels = "audio_latents_dim"
            case textDimension = "text_dim"
            case timeEmbeddingHiddenSize = "time_embed_hidden_dim"
            case timeEmbeddingDimension = "time_embed_dim"
            case ropeFrequencyCount = "rope_inv_freq_len"
        }
    }

    private struct SigmaShifts: Decodable {
        let video: Float
        let audio: Float
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case partition
        case transformer
        case sigmaShifts = "sigma_shift_scales"
        case quantization
        case textEncoderQuantization = "text_encoder_quantization"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let transformer = try container.decode(Transformer.self, forKey: .transformer)
        let shifts = try container.decode(SigmaShifts.self, forKey: .sigmaShifts)
        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.task = try container.decode(String.self, forKey: .partition)
        self.hiddenSize = transformer.hiddenSize
        self.layerCount = transformer.layerCount
        self.refinerLayerCount = 2
        self.attentionHeadCount = transformer.attentionHeadCount
        self.attentionHeadDimension = transformer.attentionHeadDimension
        self.feedForwardSize = transformer.feedForwardSize
        self.videoLatentChannels = transformer.videoLatentChannels
        self.audioLatentChannels = transformer.audioLatentChannels
        self.patchSize = [1, 2, 2]
        self.textDimension = transformer.textDimension
        self.timeFrequencyDimension = 256
        self.timeEmbeddingHiddenSize = transformer.timeEmbeddingHiddenSize ?? transformer.hiddenSize
        self.timeEmbeddingDimension = transformer.timeEmbeddingDimension
        self.videoFlowShift = shifts.video
        self.audioFlowShift = shifts.audio
        self.sampleSteps = 31
        self.quantization = try container.decodeIfPresent(
            MiniMaxH3QuantizationConfiguration.self,
            forKey: .quantization
        )
        self.textEncoderQuantization = try container.decodeIfPresent(
            MiniMaxH3QuantizationConfiguration.self,
            forKey: .textEncoderQuantization
        ) ?? quantization
    }

    public func validationIssues() -> [String] {
        var issues: [String] = []
        if modelType != "minimax_h3" { issues.append("model_type must be minimax_h3") }
        if task != "fl2va" && task != "ref2va" { issues.append("task must be fl2va or ref2va") }
        if hiddenSize != 5_376 { issues.append("hidden_size must be 5376") }
        if layerCount != 50 { issues.append("num_layers must be 50") }
        if refinerLayerCount != 2 { issues.append("num_refiner_layers must be 2") }
        if attentionHeadCount != 56 || attentionHeadDimension != 128 {
            issues.append("attention geometry must be 56 heads x 128")
        }
        if feedForwardSize != 14_336 { issues.append("ffn_dim must be 14336") }
        if videoLatentChannels != 24 || audioLatentChannels != 32 {
            issues.append("video/audio latent channels must be 24/32")
        }
        if patchSize != [1, 2, 2] { issues.append("patch_size must be [1, 2, 2]") }
        if textDimension != 5_120 { issues.append("text_dim must be 5120") }
        if timeFrequencyDimension != 256
            || timeEmbeddingHiddenSize != 5_376
            || timeEmbeddingDimension != 2_688 {
            issues.append("time embedding must be 256 -> 5376 -> 2688")
        }
        if videoFlowShift <= 0 || audioFlowShift <= 0 { issues.append("flow shifts must be positive") }
        if sampleSteps < 2 { issues.append("sample_steps must be at least 2") }
        return issues
    }
}

public struct MiniMaxH3Resources: Sendable {
    private struct SafetensorsHeader: Decodable {
        let metadata: [String: String]?

        enum CodingKeys: String, CodingKey {
            case metadata = "__metadata__"
        }
    }

    public static let fl2vaModelID = "video-minimax-h3-fl2va-mlx"
    public static let ref2vaModelID = "video-minimax-h3-ref2va-mlx"
    public static let sourceRepository = "MiniMaxAI/MiniMax-H3"
    public static let sourceRevision = "ec19cc6daf5d8add9417c18e86b6b58cc6c55027"
    public static let artifactRepository = "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit"
    public static let artifactRevision = "32bfc37f1dc8bd331394573859a627bc0aa9822b"
    public static let conversionSourceRepository = "Comfy-Org/MiniMax-H3"
    public static let conversionSourceRevision = "fd70b39279d1ae6eb214c903f53e1bec3af19a77"
    public static let ref2vaSourceSHA256 = "9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9"
    public static let ref2vaConvertedSHA256 = "c3ddde0dc29503281cd4c03c1f82b9cb640f4670da68caa5f55e3cec8f2045e8"

    public static let requiredFiles = [
        "config.json",
        "transformer.safetensors",
        "text_encoder.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_vae.safetensors",
        "audio_vae.safetensors",
        "LICENSE",
        "NOTICE",
        "MODIFICATIONS.md",
    ]

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var configURL: URL { rootURL.appending(path: "config.json") }
    public var transformerWeightsURL: URL { rootURL.appending(path: "transformer.safetensors") }
    public var textEncoderWeightsURL: URL { rootURL.appending(path: "text_encoder.safetensors") }
    public var tokenizerURL: URL { rootURL }
    public var videoVAEWeightsURL: URL { rootURL.appending(path: "video_vae.safetensors") }
    public var audioVAEWeightsURL: URL { rootURL.appending(path: "audio_vae.safetensors") }
    public var adaLNCacheURL: URL { rootURL.appending(path: MiniMaxH3AdaLNCache.filename) }

    func adaLNCacheSourceIdentity() throws -> String {
        if let inheritedIdentity = try transformerMetadata()["adaln_cache_source_identity"] {
            return inheritedIdentity
        }
        let values = try transformerWeightsURL.resourceValues(forKeys: [
            .fileSizeKey,
            .contentModificationDateKey,
        ])
        guard let fileSize = values.fileSize, let modificationDate = values.contentModificationDate else {
            throw MiniMaxH3AdaLNCacheError.incompatible("transformer file identity is unavailable")
        }
        let nanoseconds = Int64((modificationDate.timeIntervalSince1970 * 1_000_000_000).rounded())
        return "\(fileSize):\(nanoseconds)"
    }

    func transformerMetadata() throws -> [String: String] {
        let handle = try FileHandle(forReadingFrom: transformerWeightsURL)
        defer { try? handle.close() }
        guard let lengthData = try handle.read(upToCount: MemoryLayout<UInt64>.size),
              lengthData.count == MemoryLayout<UInt64>.size else {
            throw MiniMaxH3ResourcesError.invalidConfiguration(
                transformerWeightsURL,
                "safetensors header length is missing"
            )
        }
        let rawLength = lengthData.withUnsafeBytes { bytes in
            bytes.loadUnaligned(as: UInt64.self)
        }
        let headerLength = UInt64(littleEndian: rawLength)
        guard headerLength > 0, headerLength <= 64 * 1_024 * 1_024 else {
            throw MiniMaxH3ResourcesError.invalidConfiguration(
                transformerWeightsURL,
                "safetensors header length is invalid"
            )
        }
        guard let headerData = try handle.read(upToCount: Int(headerLength)),
              headerData.count == Int(headerLength) else {
            throw MiniMaxH3ResourcesError.invalidConfiguration(
                transformerWeightsURL,
                "safetensors header is truncated"
            )
        }
        do {
            return try JSONDecoder().decode(SafetensorsHeader.self, from: headerData).metadata ?? [:]
        } catch {
            throw MiniMaxH3ResourcesError.invalidConfiguration(
                transformerWeightsURL,
                "safetensors metadata is invalid: \(error.localizedDescription)"
            )
        }
    }

    func requiresAdaLNCache() throws -> Bool {
        try transformerMetadata()["cache_covered_weights_omitted"] == "true"
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        Self.requiredFiles
            .map { rootURL.appending(path: $0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public func loadConfiguration() throws -> MiniMaxH3Configuration {
        let configuration: MiniMaxH3Configuration
        do {
            configuration = try JSONDecoder().decode(
                MiniMaxH3Configuration.self,
                from: Data(contentsOf: configURL)
            )
        } catch {
            throw MiniMaxH3ResourcesError.invalidConfiguration(configURL, error.localizedDescription)
        }
        let issues = configuration.validationIssues()
        guard issues.isEmpty else {
            throw MiniMaxH3ResourcesError.invalidConfiguration(configURL, issues.joined(separator: "; "))
        }
        return configuration
    }
}
