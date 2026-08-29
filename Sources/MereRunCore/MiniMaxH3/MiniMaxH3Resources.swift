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

struct MiniMaxH3Ref2VAConversionReceipt: Decodable, Sendable {
    struct FileIdentity: Decodable, Sendable {
        let byteCount: Int64
        let filename: String
        let repository: String?
        let revision: String?
        let sha256: String

        enum CodingKeys: String, CodingKey {
            case byteCount = "byte_count"
            case filename
            case repository
            case revision
            case sha256
        }
    }

    let converter: String
    let converterVersion: Int
    let partition: String
    let source: FileIdentity
    let output: FileIdentity
    let sourceConvRotGroups: [String: Int]
    let quantization: MiniMaxH3QuantizationConfiguration

    enum CodingKeys: String, CodingKey {
        case converter
        case converterVersion = "converter_version"
        case partition
        case source
        case output
        case sourceConvRotGroups = "source_convrot_groups"
        case quantization
    }
}

private struct MiniMaxH3Ref2VASourceManifest: Decodable, Sendable {
    struct Artifact: Decodable, Sendable {
        let format: String
        let partition: String
        let repository: String
    }

    struct AdaLNCache: Decodable, Sendable {
        struct Schedule: Decodable, Sendable {
            let audioFlowShift: Float
            let pointCount: Int
            let videoFlowShift: Float

            enum CodingKeys: String, CodingKey {
                case audioFlowShift = "audio_flow_shift"
                case pointCount = "point_count"
                case videoFlowShift = "video_flow_shift"
            }
        }

        let byteCount: Int64
        let format: String
        let path: String
        let schedule: Schedule
        let schemaVersion: Int
        let sha256: String
        let sourceIdentity: String

        enum CodingKeys: String, CodingKey {
            case byteCount = "byte_count"
            case format
            case path
            case schedule
            case schemaVersion = "schema_version"
            case sha256
            case sourceIdentity = "source_identity"
        }
    }

    let artifact: Artifact
    let adaLNCache: AdaLNCache
    let schemaVersion: Int

    enum CodingKeys: String, CodingKey {
        case artifact
        case adaLNCache = "adaln_cache"
        case schemaVersion = "schema_version"
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
    public enum TransformerWeightsLayout: Sendable, Hashable {
        case single(URL)
        case shardedBF16(indexURL: URL)
    }

    public enum TransformerStorage: String, Codable, Sendable {
        case shardedBF16 = "sharded-bf16"
        case compactBF16 = "compact-bf16"
        case affineQ8 = "affine-q8"
        case affineQ4 = "affine-q4"

        public var supportsFL2VATurboAdapters: Bool {
            self != .affineQ4
        }
    }

    private struct SafetensorsHeader: Decodable {
        let metadata: [String: String]?

        enum CodingKeys: String, CodingKey {
            case metadata = "__metadata__"
        }
    }

    public static let fl2vaModelID = "video-minimax-h3-fl2va-mlx"
    public static let fl2vaBF16ModelID = "video-minimax-h3-fl2va-bf16-mlx"
    public static let fl2vaQ8ModelID = "video-minimax-h3-fl2va-8bit-mlx"
    public static let ref2vaModelID = "video-minimax-h3-ref2va-mlx"
    public static let sourceRepository = "MiniMaxAI/MiniMax-H3"
    public static let sourceRevision = "ec19cc6daf5d8add9417c18e86b6b58cc6c55027"
    public static let artifactRepository = "Sawfwair/MiniMax-H3-FL2VA-MLX-4bit"
    public static let artifactRevision = "e1244ad93d60c737c7e0f065a1c9372f3de7caf8"
    public static let legacyBF16ArtifactRepository = "pipenetwork/MiniMax-H3-MLX-bf16"
    public static let legacyBF16ArtifactRevision = "1486555759eed9e3037edf29f9e055a0713bab2f"
    public static let compactBF16ArtifactRepository = "Sawfwair/MiniMax-H3-FL2VA-MLX-BF16"
    public static let compactBF16ArtifactRevision = "4ce4b1d870f7b1b0c75672fd4f2867c1f5df7b5f"
    public static let q8ArtifactRepository = "Sawfwair/MiniMax-H3-FL2VA-MLX-8bit"
    public static let q8ArtifactRevision = "86500cb6ebec22c006597e41840b26ef1099fdd7"
    public static let ref2vaArtifactRepository = "Sawfwair/MiniMax-H3-Ref2VA-MLX-8bit"
    public static let ref2vaArtifactRevision = "61dc387ef1a7166425cdacd63c2340598dcc364f"
    public static let bf16TransformerDirectory = "transformer-bf16"
    public static let bf16TextEncoderDirectory = "text-encoder-bf16"
    public static let officialTransformerSourceIdentity =
        "MiniMaxAI/MiniMax-H3@ec19cc6daf5d8add9417c18e86b6b58cc6c55027:"
        + "FL2VA/transformer:index-sha256:fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb"
    public static let conversionSourceRepository = "Comfy-Org/MiniMax-H3"
    public static let conversionSourceRevision = "fd70b39279d1ae6eb214c903f53e1bec3af19a77"
    public static let ref2vaSourceFilename = "minimax_h3_ref2va_int8_convrot.safetensors"
    public static let ref2vaSourceSHA256 = "9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9"
    public static let ref2vaSourceByteCount: Int64 = 34_038_894_550
    public static let ref2vaConvertedSHA256 = "234f22f69f8d40d6ed81cceed8259fa287f3c9417d40fba5274e3a7aa84e18a2"
    public static let ref2vaConvertedByteCount: Int64 = 36_024_412_656
    public static let ref2vaAdaLNCacheSHA256 = "2cbe9e3324ef2cc5108a3ba7f1219d84079ff00a017f604fd86300005cc64fcd"
    public static let ref2vaAdaLNCacheByteCount: Int64 = 873_820_740
    public static let ref2vaAdaLNCacheSourceIdentity =
        "sha256:\(ref2vaConvertedSHA256)"

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
    public static let compactArtifactFiles = requiredFiles + [
        "adaln_cache.safetensors",
        "SOURCE_MANIFEST.json",
        "transformer.conversion.json",
        "SHA256SUMS",
    ]
    public static let cachePackFiles = [MiniMaxH3AdaLNCachePack.filename]
        + MiniMaxH3AdaLNCachePack.productionSchedules.map(\.filename)
    public static let compactBF16AndQ8ArtifactFiles = requiredFiles + cachePackFiles + [
        "adaln_cache.refresh.json",
        "SOURCE_MANIFEST.json",
        "transformer.conversion.json",
        "SHA256SUMS",
    ]
    public static let bf16SupportArtifactFiles = compactArtifactFiles.filter {
        $0 != "transformer.safetensors" && $0 != MiniMaxH3AdaLNCache.filename
    }
    public static let ref2vaArtifactFiles = requiredFiles + [
        "adaln_cache.safetensors",
        "SOURCE_MANIFEST.json",
        "transformer.conversion.json",
        "SHA256SUMS",
    ]
    public static let bf16ArtifactFiles = [
        "README.md",
        "LICENSE",
        "config.json",
        "model.safetensors.index.json",
        "model-*.safetensors",
    ]
    public static let bf16ShardFilenames = (1...13).map {
        String(format: "model-%05d-of-00013.safetensors", $0)
    }
    public static let bf16TextEncoderShardFilenames = (1...14).map {
        String(format: "model-%05d-of-00014.safetensors", $0)
    }

    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var configURL: URL { rootURL.appending(path: "config.json") }
    public var transformerWeightsURL: URL { rootURL.appending(path: "transformer.safetensors") }
    public var bf16TransformerRootURL: URL {
        rootURL.appending(path: Self.bf16TransformerDirectory, directoryHint: .isDirectory)
    }
    public var bf16TransformerIndexURL: URL {
        bf16TransformerRootURL.appending(path: "model.safetensors.index.json")
    }
    public var bf16TextEncoderRootURL: URL {
        rootURL.appending(path: Self.bf16TextEncoderDirectory, directoryHint: .isDirectory)
    }
    public var bf16TextEncoderIndexURL: URL {
        bf16TextEncoderRootURL.appending(path: "model.safetensors.index.json")
    }
    public var textEncoderWeightsURL: URL { rootURL.appending(path: "text_encoder.safetensors") }
    public var tokenizerURL: URL { rootURL }
    public var videoVAEWeightsURL: URL { rootURL.appending(path: "video_vae.safetensors") }
    public var audioVAEWeightsURL: URL { rootURL.appending(path: "audio_vae.safetensors") }
    public var adaLNCacheURL: URL { rootURL.appending(path: MiniMaxH3AdaLNCache.filename) }
    public var adaLNCachePackIndexURL: URL {
        rootURL.appending(path: MiniMaxH3AdaLNCachePack.filename)
    }
    public var conversionReceiptURL: URL { rootURL.appending(path: "transformer.conversion.json") }

    public func transformerWeightsLayout(fileManager: FileManager = .default) -> TransformerWeightsLayout? {
        if fileManager.fileExists(atPath: bf16TransformerIndexURL.path) {
            return .shardedBF16(indexURL: bf16TransformerIndexURL)
        }
        if fileManager.fileExists(atPath: transformerWeightsURL.path) {
            return .single(transformerWeightsURL)
        }
        return nil
    }

    public var usesShardedBF16Transformer: Bool {
        if case .shardedBF16 = transformerWeightsLayout() { return true }
        return false
    }

    public func transformerStorage() throws -> TransformerStorage {
        if usesShardedBF16Transformer { return .shardedBF16 }
        let metadata = try transformerMetadata()
        switch metadata["precision"]?.lowercased() {
        case "bf16":
            return .compactBF16
        case "q8", "int8", "8bit":
            return .affineQ8
        case "q4", "int4", "4bit":
            return .affineQ4
        default:
            let configuration = try loadConfiguration()
            switch configuration.quantization?.bits {
            case nil:
                return .compactBF16
            case 8:
                return .affineQ8
            case 4:
                return .affineQ4
            case .some(let bits):
                throw MiniMaxH3ResourcesError.invalidConfiguration(
                    configURL,
                    "unsupported transformer quantization width: \(bits)"
                )
            }
        }
    }

    public var usesShardedBF16Conditioner: Bool {
        FileManager.default.fileExists(atPath: bf16TextEncoderIndexURL.path)
    }

    func adaLNCacheSourceIdentity() throws -> String {
        if usesShardedBF16Transformer {
            return Self.officialTransformerSourceIdentity
        }
        if let inheritedIdentity = try transformerMetadata()["adaln_cache_source_identity"] {
            return inheritedIdentity
        }
        if validateManagedRef2VAArtifact().isEmpty {
            return Self.ref2vaAdaLNCacheSourceIdentity
        }
        let resolvedWeightsURL = transformerWeightsURL.resolvingSymlinksInPath()
        let values = try resolvedWeightsURL.resourceValues(forKeys: [
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
        guard !usesShardedBF16Transformer else { return [:] }
        return try safetensorsMetadata(at: transformerWeightsURL)
    }

    private func safetensorsMetadata(at url: URL) throws -> [String: String] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let lengthData = try handle.read(upToCount: MemoryLayout<UInt64>.size),
              lengthData.count == MemoryLayout<UInt64>.size else {
            throw MiniMaxH3ResourcesError.invalidConfiguration(
                url,
                "safetensors header length is missing"
            )
        }
        let rawLength = lengthData.withUnsafeBytes { bytes in
            bytes.loadUnaligned(as: UInt64.self)
        }
        let headerLength = UInt64(littleEndian: rawLength)
        guard headerLength > 0, headerLength <= 64 * 1_024 * 1_024 else {
            throw MiniMaxH3ResourcesError.invalidConfiguration(
                url,
                "safetensors header length is invalid"
            )
        }
        guard let headerData = try handle.read(upToCount: Int(headerLength)),
              headerData.count == Int(headerLength) else {
            throw MiniMaxH3ResourcesError.invalidConfiguration(
                url,
                "safetensors header is truncated"
            )
        }
        do {
            return try JSONDecoder().decode(SafetensorsHeader.self, from: headerData).metadata ?? [:]
        } catch {
            throw MiniMaxH3ResourcesError.invalidConfiguration(
                url,
                "safetensors metadata is invalid: \(error.localizedDescription)"
            )
        }
    }

    func requiresAdaLNCache() throws -> Bool {
        if usesShardedBF16Transformer { return false }
        return try transformerMetadata()["cache_covered_weights_omitted"] == "true"
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing = Self.requiredFiles
            .filter { filename in
                filename != "transformer.safetensors"
                    && (!usesShardedBF16Conditioner || filename != "text_encoder.safetensors")
            }
            .map { rootURL.appending(path: $0) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
        switch transformerWeightsLayout(fileManager: fileManager) {
        case .single:
            break
        case .shardedBF16:
            missing += Self.bf16ShardFilenames
                .map { bf16TransformerRootURL.appending(path: $0) }
                .filter { !fileManager.fileExists(atPath: $0.path) }
        case nil:
            missing.append(transformerWeightsURL)
        }
        if usesShardedBF16Conditioner {
            if !fileManager.fileExists(atPath: bf16TextEncoderIndexURL.path) {
                missing.append(bf16TextEncoderIndexURL)
            }
            missing += Self.bf16TextEncoderShardFilenames
                .map { bf16TextEncoderRootURL.appending(path: $0) }
                .filter { !fileManager.fileExists(atPath: $0.path) }
        }
        if fileManager.fileExists(atPath: transformerWeightsURL.path),
           (try? requiresAdaLNCache()) == true {
            if fileManager.fileExists(atPath: adaLNCachePackIndexURL.path) {
                missing += Self.cachePackFiles
                    .map { rootURL.appending(path: $0) }
                    .filter { !fileManager.fileExists(atPath: $0.path) }
            } else if !fileManager.fileExists(atPath: adaLNCacheURL.path) {
                missing.append(adaLNCacheURL)
            }
        }
        return missing
    }

    func validateCompactCachePack(fileManager: FileManager = .default) -> [String] {
        let provenanceFiles = [
            "adaln_cache.refresh.json",
            "SOURCE_MANIFEST.json",
            "transformer.conversion.json",
            "SHA256SUMS",
        ]
        let missingProvenance = provenanceFiles.filter {
            !fileManager.fileExists(atPath: rootURL.appending(path: $0).path)
        }
        if !missingProvenance.isEmpty {
            return missingProvenance.map { "Missing required MiniMax-H3 provenance file: \($0)" }
        }
        do {
            let sourceIdentity = try adaLNCacheSourceIdentity()
            try MiniMaxH3AdaLNCachePack.validateClosure(
                from: rootURL,
                sourceIdentity: sourceIdentity,
                fileManager: fileManager
            )
            return []
        } catch {
            return ["Invalid MiniMax-H3 cache pack: \(error.localizedDescription)"]
        }
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

    func validateManagedRef2VAArtifact(fileManager: FileManager = .default) -> [String] {
        let sourceManifestURL = rootURL.appending(path: "SOURCE_MANIFEST.json")
        let requiredProvenance = [
            sourceManifestURL,
            conversionReceiptURL,
            rootURL.appending(path: "SHA256SUMS"),
            adaLNCacheURL,
        ]
        let missing = requiredProvenance.filter { !fileManager.fileExists(atPath: $0.path) }
        guard missing.isEmpty else {
            return missing.map { "Missing required Ref2VA provenance file: \($0.lastPathComponent)" }
        }

        let sourceManifest: MiniMaxH3Ref2VASourceManifest
        let receipt: MiniMaxH3Ref2VAConversionReceipt
        do {
            sourceManifest = try JSONDecoder().decode(
                MiniMaxH3Ref2VASourceManifest.self,
                from: Data(contentsOf: sourceManifestURL)
            )
            receipt = try JSONDecoder().decode(
                MiniMaxH3Ref2VAConversionReceipt.self,
                from: Data(contentsOf: conversionReceiptURL)
            )
        } catch {
            return ["Invalid Ref2VA provenance: \(error.localizedDescription)"]
        }

        var issues: [String] = []
        if sourceManifest.schemaVersion != 1
            || sourceManifest.artifact.format != "mere.run.minimax-h3-ref2va-mlx-8bit"
            || sourceManifest.artifact.partition != "ref2va"
            || sourceManifest.artifact.repository != Self.ref2vaArtifactRepository {
            issues.append("Ref2VA source manifest does not identify the pinned managed artifact.")
        }
        let cache = sourceManifest.adaLNCache
        if cache.path != MiniMaxH3AdaLNCache.filename
            || cache.byteCount != Self.ref2vaAdaLNCacheByteCount
            || cache.sha256 != Self.ref2vaAdaLNCacheSHA256
            || cache.format != "mere.run.minimax-h3-adaln-cache"
            || cache.schemaVersion != 2
            || cache.sourceIdentity != Self.ref2vaAdaLNCacheSourceIdentity
            || cache.schedule.pointCount != 31
            || cache.schedule.videoFlowShift != 12
            || cache.schedule.audioFlowShift != 3 {
            issues.append("Ref2VA source manifest does not match the pinned AdaLN cache.")
        }
        let resolvedCacheURL = adaLNCacheURL.resolvingSymlinksInPath()
        let actualCacheBytes = try? resolvedCacheURL.resourceValues(forKeys: [.fileSizeKey]).fileSize
        if Int64(actualCacheBytes ?? -1) != Self.ref2vaAdaLNCacheByteCount {
            issues.append("Ref2VA AdaLN cache byte count does not match the pinned artifact.")
        }
        do {
            let metadata = try safetensorsMetadata(at: adaLNCacheURL)
            if metadata["format"] != "mere.run.minimax-h3-adaln-cache"
                || metadata["schema_version"] != MiniMaxH3AdaLNCache.schemaVersion
                || metadata["source_identity"] != Self.ref2vaAdaLNCacheSourceIdentity {
                issues.append("Ref2VA AdaLN cache metadata does not match the pinned artifact.")
            }
        } catch {
            issues.append("Ref2VA AdaLN cache metadata is invalid: \(error.localizedDescription)")
        }
        if receipt.converter != "scripts/model-conversion/convert_minimax_h3_convrot.py"
            || receipt.converterVersion != 2 {
            issues.append("Ref2VA conversion receipt must use the pinned ConvRot converter version 2.")
        }
        if receipt.partition != "ref2va" {
            issues.append("Ref2VA conversion receipt partition must be ref2va.")
        }
        if receipt.source.repository != Self.conversionSourceRepository
            || receipt.source.revision != Self.conversionSourceRevision
            || receipt.source.filename != Self.ref2vaSourceFilename
            || receipt.source.byteCount != Self.ref2vaSourceByteCount
            || receipt.source.sha256 != Self.ref2vaSourceSHA256 {
            issues.append("Ref2VA conversion receipt does not match the pinned source artifact.")
        }
        if receipt.output.filename != transformerWeightsURL.lastPathComponent
            || receipt.output.repository != nil
            || receipt.output.revision != nil
            || receipt.output.byteCount != Self.ref2vaConvertedByteCount
            || receipt.output.sha256 != Self.ref2vaConvertedSHA256 {
            issues.append("Ref2VA conversion receipt does not match the pinned MLX transformer.")
        }
        if receipt.sourceConvRotGroups != ["64": 50, "256": 200] {
            issues.append("Ref2VA conversion receipt has unexpected ConvRot source-group counts.")
        }
        if receipt.quantization != MiniMaxH3QuantizationConfiguration(
            bits: 8,
            groupSize: 64,
            mode: "affine"
        ) {
            issues.append("Ref2VA conversion receipt must declare MLX affine INT8/group-64 output.")
        }
        let resolvedWeightsURL = transformerWeightsURL.resolvingSymlinksInPath()
        let actualBytes = try? resolvedWeightsURL.resourceValues(forKeys: [.fileSizeKey]).fileSize
        if Int64(actualBytes ?? -1) != Self.ref2vaConvertedByteCount {
            issues.append("Ref2VA transformer byte count does not match the pinned artifact.")
        }
        do {
            let metadata = try transformerMetadata()
            if metadata["quantization"] != "affine 8-bit g64"
                || metadata["source_repository"] != Self.conversionSourceRepository
                || metadata["source_revision"] != Self.conversionSourceRevision {
                issues.append("Ref2VA transformer metadata does not match the pinned conversion source.")
            }
        } catch {
            issues.append("Ref2VA transformer metadata is invalid: \(error.localizedDescription)")
        }
        return issues
    }
}
