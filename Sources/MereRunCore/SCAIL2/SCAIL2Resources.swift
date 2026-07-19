import Foundation

public enum SCAIL2ResourcesError: LocalizedError, Sendable {
    case invalidConfiguration(URL, String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let url, let reason):
            return "Invalid SCAIL-2 configuration at \(url.path): \(reason)"
        }
    }
}

public struct SCAIL2Configuration: Codable, Hashable, Sendable {
    public let className: String
    public let modelType: String
    public let patchSize: [Int]
    public let textLength: Int
    public let inputChannels: Int
    public let maskChannels: Int
    public let hiddenSize: Int
    public let feedForwardSize: Int
    public let frequencySize: Int
    public let textEmbeddingSize: Int
    public let outputChannels: Int
    public let headCount: Int
    public let layerCount: Int
    public let epsilon: Float
    public let vaeStride: [Int]
    public let vaeLatentChannels: Int
    public let sampleSteps: Int
    public let sampleShift: Float
    public let guidanceScale: Float
    public let framesPerSecond: Int
    public let segmentLength: Int
    public let segmentOverlap: Int

    enum CodingKeys: String, CodingKey {
        case className = "_class_name"
        case modelType = "model_type"
        case patchSize = "patch_size"
        case textLength = "text_len"
        case inputChannels = "in_dim"
        case maskChannels = "mask_dim"
        case hiddenSize = "dim"
        case feedForwardSize = "ffn_dim"
        case frequencySize = "freq_dim"
        case textEmbeddingSize = "text_dim"
        case outputChannels = "out_dim"
        case headCount = "num_heads"
        case layerCount = "num_layers"
        case epsilon = "eps"
        case vaeStride = "vae_stride"
        case vaeLatentChannels = "vae_z_dim"
        case sampleSteps = "sample_steps"
        case sampleShift = "sample_shift"
        case guidanceScale = "sample_guide_scale"
        case framesPerSecond = "sample_fps"
        case segmentLength = "segment_len"
        case segmentOverlap = "segment_overlap"
    }

    public init(
        className: String = "WanSCAILModel",
        modelType: String = "scail2",
        patchSize: [Int] = [1, 2, 2],
        textLength: Int = 512,
        inputChannels: Int = 20,
        maskChannels: Int = 28,
        hiddenSize: Int = 5_120,
        feedForwardSize: Int = 13_824,
        frequencySize: Int = 256,
        textEmbeddingSize: Int = 4_096,
        outputChannels: Int = 16,
        headCount: Int = 40,
        layerCount: Int = 40,
        epsilon: Float = 1e-6,
        vaeStride: [Int] = [4, 8, 8],
        vaeLatentChannels: Int = 16,
        sampleSteps: Int = 40,
        sampleShift: Float = 3,
        guidanceScale: Float = 5,
        framesPerSecond: Int = 16,
        segmentLength: Int = 81,
        segmentOverlap: Int = 5
    ) {
        self.className = className
        self.modelType = modelType
        self.patchSize = patchSize
        self.textLength = textLength
        self.inputChannels = inputChannels
        self.maskChannels = maskChannels
        self.hiddenSize = hiddenSize
        self.feedForwardSize = feedForwardSize
        self.frequencySize = frequencySize
        self.textEmbeddingSize = textEmbeddingSize
        self.outputChannels = outputChannels
        self.headCount = headCount
        self.layerCount = layerCount
        self.epsilon = epsilon
        self.vaeStride = vaeStride
        self.vaeLatentChannels = vaeLatentChannels
        self.sampleSteps = sampleSteps
        self.sampleShift = sampleShift
        self.guidanceScale = guidanceScale
        self.framesPerSecond = framesPerSecond
        self.segmentLength = segmentLength
        self.segmentOverlap = segmentOverlap
    }

    public func validationIssues() -> [String] {
        var issues: [String] = []
        if className != "WanSCAILModel" { issues.append("_class_name must be WanSCAILModel") }
        if modelType != "scail2" { issues.append("model_type must be scail2") }
        if patchSize != [1, 2, 2] { issues.append("patch_size must be [1, 2, 2]") }
        if textLength != 512 { issues.append("text_len must be 512") }
        if inputChannels != 20 { issues.append("in_dim must be 20") }
        if maskChannels != 28 { issues.append("mask_dim must be 28") }
        if hiddenSize != 5_120 { issues.append("dim must be 5120") }
        if feedForwardSize != 13_824 { issues.append("ffn_dim must be 13824") }
        if frequencySize != 256 { issues.append("freq_dim must be 256") }
        if textEmbeddingSize != 4_096 { issues.append("text_dim must be 4096") }
        if outputChannels != 16 { issues.append("out_dim must be 16") }
        if headCount != 40 { issues.append("num_heads must be 40") }
        if layerCount != 40 { issues.append("num_layers must be 40") }
        if epsilon != 1e-6 { issues.append("eps must be 1e-6") }
        if vaeStride != [4, 8, 8] { issues.append("vae_stride must be [4, 8, 8]") }
        if vaeLatentChannels != 16 { issues.append("vae_z_dim must be 16") }
        if sampleSteps <= 0 { issues.append("sample_steps must be positive") }
        if sampleShift <= 0 { issues.append("sample_shift must be positive") }
        if guidanceScale <= 0 { issues.append("sample_guide_scale must be positive") }
        if framesPerSecond <= 0 { issues.append("sample_fps must be positive") }
        if segmentLength < 1 || (segmentLength - 1) % 4 != 0 {
            issues.append("segment_len must equal 1 modulo 4")
        }
        if segmentOverlap < 1 || segmentOverlap >= segmentLength || (segmentOverlap - 1) % 4 != 0 {
            issues.append("segment_overlap must equal 1 modulo 4 and be smaller than segment_len")
        }
        return issues
    }
}

public struct SCAIL2Resources: Hashable, Sendable {
    public static let modelID = "video-scail2-14b-mlx"
    public static let upstreamRepoID = "zai-org/SCAIL-2"
    public static let upstreamRevision = "150cc0ca4e98e50e60b9295dacde39442fdccab2"

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var configURL: URL { rootURL.appendingPathComponent("config.json") }
    public var transformerURL: URL { rootURL.appendingPathComponent("model.safetensors") }
    public var transformerIndexURL: URL { rootURL.appendingPathComponent("model.safetensors.index.json") }
    public var clipURL: URL { rootURL.appendingPathComponent("clip.safetensors") }
    public var textEncoderURL: URL { rootURL.appendingPathComponent("t5_encoder.safetensors") }
    public var textEncoderIndexURL: URL { rootURL.appendingPathComponent("t5_encoder.safetensors.index.json") }
    public var tokenizerURL: URL { rootURL.appendingPathComponent("tokenizer.json") }
    public var vaeURL: URL { rootURL.appendingPathComponent("vae.safetensors") }
    public var sourceLicenseURL: URL { rootURL.appendingPathComponent("LICENSE-SCAIL-2") }
    public var sourceReadmeURL: URL { rootURL.appendingPathComponent("README-SCAIL-2.md") }
    public var provenanceURL: URL { rootURL.appendingPathComponent("provenance.json") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing = [
            configURL, clipURL, tokenizerURL, vaeURL,
            sourceLicenseURL, sourceReadmeURL, provenanceURL,
        ]
            .filter { !fileManager.fileExists(atPath: $0.path) }
        if !fileManager.fileExists(atPath: transformerURL.path) {
            missing.append(contentsOf: missingIndexOrShards(
                indexURL: transformerIndexURL,
                fileManager: fileManager
            ))
        }
        if !fileManager.fileExists(atPath: textEncoderURL.path) {
            missing.append(contentsOf: missingIndexOrShards(
                indexURL: textEncoderIndexURL,
                fileManager: fileManager
            ))
        }
        return missing
    }

    private func missingIndexOrShards(
        indexURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        guard fileManager.fileExists(atPath: indexURL.path),
              let data = try? Data(contentsOf: indexURL),
              let index = try? JSONDecoder().decode(HFSafetensorsIndex.self, from: data),
              !index.shardFilenames.isEmpty else {
            return [indexURL]
        }
        return index.shardFilenames
            .map { indexURL.deletingLastPathComponent().appendingPathComponent($0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public func loadConfiguration() throws -> SCAIL2Configuration {
        let configuration: SCAIL2Configuration
        do {
            configuration = try JSONDecoder().decode(
                SCAIL2Configuration.self,
                from: Data(contentsOf: configURL)
            )
        } catch {
            throw SCAIL2ResourcesError.invalidConfiguration(configURL, error.localizedDescription)
        }
        let issues = configuration.validationIssues()
        guard issues.isEmpty else {
            throw SCAIL2ResourcesError.invalidConfiguration(configURL, issues.joined(separator: "; "))
        }
        return configuration
    }
}
