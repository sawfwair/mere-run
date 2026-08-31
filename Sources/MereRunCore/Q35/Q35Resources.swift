import Foundation

public struct Q35Resources: Sendable, Hashable {
    public struct Profile: Sendable, Hashable {
        public let modelId: String
        public let upstreamRepoId: String
        public let upstreamRevision: String
        public let snapshotPatterns: [String]

        public init(
            modelId: String,
            upstreamRepoId: String,
            upstreamRevision: String,
            snapshotPatterns: [String] = Q35Resources.snapshotPatterns
        ) {
            self.modelId = modelId
            self.upstreamRepoId = upstreamRepoId
            self.upstreamRevision = upstreamRevision
            self.snapshotPatterns = snapshotPatterns
        }

        public var hubFallbackConfig: HubFallbackConfig {
            HubFallbackConfig(
                repoId: upstreamRepoId,
                revision: upstreamRevision,
                patterns: snapshotPatterns
            )
        }
    }

    public static let q36NanoModelId = "text-chat-q36-nano"
    public static let q38TwentySevenBModelId = "vision-chat-q38-27b"
    public static let q38TwentySevenB4BitModelId = "vision-chat-q38-27b-4bit"
    public static let q38FlashNextMixedModelId = "vision-chat-q38-flash-next-mixed"
    public static let q38FlashNext3BitModelId = "vision-chat-q38-flash-next-3bit"
    public static let q38FlashNext3BitNativePLEModelId = "vision-chat-q38-flash-next-3bit-native-ple"
    public static let q38FlashNext4BitModelId = "vision-chat-q38-flash-next-4bit"
    public static let bonsai27B1BitModelId = "text-chat-bonsai-27b-1bit"
    public static let bonsai27B2BitModelId = "text-chat-bonsai-27b-2bit"
    public static let ornith9BModelId = "text-agent-ornith-9b"
    public static let ornith35BMLX4BitModelId = "text-agent-ornith-35b-mlx-4bit"
    public static let ornith35BMLX6BitModelId = "text-agent-ornith-35b-mlx-6bit"
    public static let ornith35BMLX8BitModelId = "text-agent-ornith-35b-mlx-8bit"
    /// Compatibility id for the official unquantized BF16 MLX snapshot.
    public static let ornith35BMLXModelId = "text-agent-ornith-35b-mlx"
    public static let ornith35BMTPModelId = "text-agent-ornith-35b-mtp"
    public static let infinityParser2ProModelId = "vision-ocr-infinity-pro"
    public static let infinityParser2ProInt8ModelId = "vision-ocr-infinity-pro-int8"
    public static let defaultModelId = q36NanoModelId

    /// R1-style agent tunes degenerate without reasoning enabled: no-think
    /// output is signature echo or repetition loops at any temperature
    /// (HumanEval no-think scored 1/164 on Ornith 35B vs a ~90% thinking-mode
    /// model card), so these lanes default to thinking-enabled generation.
    public static func thinkingDefault(forModelId modelId: String) -> Bool {
        isQ38ModelId(modelId)
            || isBonsai27BModelId(modelId)
            || modelId == ornith9BModelId
            || isOrnith35BMLXModelId(modelId)
    }

    public static func isBonsai27BModelId(_ modelId: String) -> Bool {
        modelId == bonsai27B1BitModelId || modelId == bonsai27B2BitModelId
    }

    /// Convert the public continuous reasoning control into the three native
    /// levels accepted by the pinned Qwen3.8 chat template.
    public static func q38ReasoningEffortLabel(for strength: Double?) -> String? {
        guard let strength else { return nil }
        if strength < 1.0 / 3.0 { return "low" }
        if strength < 2.0 / 3.0 { return "medium" }
        return "xhigh"
    }

    public struct RecommendedSampling: Sendable, Hashable {
        public let temperature: Double
        public let topP: Double
        public let topK: Int

        public init(temperature: Double, topP: Double, topK: Int) {
            self.temperature = temperature
            self.topP = topP
            self.topK = topK
        }
    }

    /// Published `generation_config.json` sampling for lanes whose models ship
    /// one; callers use it when the user did not set explicit sampling.
    public static func recommendedSampling(forModelId modelId: String) -> RecommendedSampling? {
        switch modelId {
        case q38TwentySevenBModelId,
             q38TwentySevenB4BitModelId,
             q38FlashNextMixedModelId,
             q38FlashNext3BitModelId,
             q38FlashNext3BitNativePLEModelId,
             q38FlashNext4BitModelId:
            return RecommendedSampling(temperature: 1.0, topP: 0.95, topK: 20)
        case bonsai27B1BitModelId, bonsai27B2BitModelId:
            return RecommendedSampling(temperature: 0.7, topP: 0.95, topK: 20)
        case ornith9BModelId,
             ornith35BMLX4BitModelId,
             ornith35BMLX6BitModelId,
             ornith35BMLX8BitModelId,
             ornith35BMLXModelId:
            return RecommendedSampling(temperature: 1.0, topP: 0.95, topK: 20)
        default:
            return nil
        }
    }

    public static let q36NanoUpstreamRepoId = "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit"
    public static let q36NanoUpstreamRevision = "63d520640ca7461f31ba66104612135770090340"
    public static let q38TwentySevenBUpstreamRepoId = "Qwen/Qwen3.8-27B"
    public static let q38TwentySevenBUpstreamRevision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    public static let q38TwentySevenBEstimatedDownloadBytes: Int64 = 55_586_114_863
    public static let q38TwentySevenB4BitUpstreamRepoId = "EigenLabs/Qwen3.8-27B-4bit"
    public static let q38TwentySevenB4BitUpstreamRevision = "eda45ab47f465d08d6558f0353a2346e2eb9d5b3"
    public static let q38TwentySevenB4BitEstimatedDownloadBytes: Int64 = 15_520_000_000
    public static let q38FlashNextMixedUpstreamRepoId = "Sawfwair/Qwen3.8-Flash-Next-MLX-Mixed-2bit"
    public static let q38FlashNextMixedUpstreamRevision = "0bdf3edf02df271e9898f17a7882e5e6a8feb58a"
    public static let q38FlashNextMixedEstimatedDownloadBytes: Int64 = 73_095_335_934
    public static let q38FlashNext3BitUpstreamRepoId = "Sawfwair/Qwen3.8-Flash-Next-MLX-Activation-3bit"
    public static let q38FlashNext3BitUpstreamRevision = "c699bd611366cbc441377275bd1b7a6d2e18e1df"
    public static let q38FlashNext3BitEstimatedDownloadBytes: Int64 = 89_666_463_470
    public static let q38FlashNext3BitNativePLEUpstreamRepoId =
        "Sawfwair/Qwen3.8-Flash-Next-MLX-Activation-3bit-Native-PLE"
    public static let q38FlashNext3BitNativePLEUpstreamRevision =
        "1cee9301c745836e0abb8933e89cf27a38b98125"
    public static let q38FlashNext3BitNativePLEEstimatedDownloadBytes: Int64 = 89_667_182_797
    public static let q38FlashNext4BitUpstreamRepoId = "Sawfwair/Qwen3.8-Flash-Next-MLX-4bit"
    public static let q38FlashNext4BitUpstreamRevision = "6cc9bbc0fae9ce26b7670b3ed1e26d557c154506"
    public static let q38FlashNext4BitEstimatedDownloadBytes: Int64 = 104_742_357_706
    public static let q38MTPComponentPath = "mtp"
    public static let q38MTP4BitUpstreamRepoId = "morgan/qwen38-27b-mtp-r20k-lr3-q4-g64-q2-rerank"
    public static let q38MTP4BitUpstreamRevision = "fd4a99c590dd6e468c0e2a28168c235e32151a4b"
    public static let q38MTPComponentSnapshotPatterns = ["model.safetensors"]
    public static let q38VisionComponentPath = "vision"
    public static let q38VisionComponentSnapshotPatterns = [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00018.safetensors",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ]
    public static let q38VisionComponentEstimatedDownloadBytes: Int64 = 3_967_000_000
    public static let q38LicenseComponentPath = "licenses/qwen3.8"
    public static let q38LicenseComponentSnapshotPatterns = ["LICENSE"]
    public static let q38TwentySevenBContextLength = 262_144
    public static let q38TwentySevenBVisionMinPixels = 65_536
    public static let q38TwentySevenBVisionMaxPixels = 16_777_216
    public static let bonsai27B1BitUpstreamRepoId = "prism-ml/Bonsai-27B-mlx-1bit"
    public static let bonsai27B1BitUpstreamRevision = "ef22f239c670078e1507f9769bcaa66657332b96"
    public static let bonsai27B1BitEstimatedDownloadBytes: Int64 = 5_128_837_600
    public static let bonsai27B1BitContextLength = 262_144
    public static let bonsai27B2BitUpstreamRepoId = "prism-ml/Ternary-Bonsai-27B-mlx-2bit"
    public static let bonsai27B2BitUpstreamRevision = "70f75f3ad081ab840a42f3304c02c27e7f89bfb7"
    public static let bonsai27B2BitEstimatedDownloadBytes: Int64 = 8_521_085_419
    public static let bonsai27B2BitContextLength = 262_144
    public static let ornith9BUpstreamRepoId = "sahilchachra/ornith-1.0-9b-optiq-5bpw-mlx"
    public static let ornith9BUpstreamRevision = "4f9f4fc2c10ec17cbeb9dae086a7f1272c904e86"
    public static let ornith9BEstimatedDownloadBytes: Int64 = 7 * 1_073_741_824
    public static let ornith35BMLXUpstreamRepoId = "ornith-ai/Ornith-1.5-35B-A3B-MLX"
    public static let ornith35BMLXUpstreamRevision = "bcfbfccfe413e46bc7cc04188622accccd8d3c00"
    public static let ornith35BMLXEstimatedDownloadBytes: Int64 = 69_343_588_037
    public static let ornith35BMLX4BitUpstreamRepoId = "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"
    public static let ornith35BMLX4BitUpstreamRevision = "19504d912fa8fc7622bf6b1de3db5d5d890b1f02"
    public static let ornith35BMLX4BitEstimatedDownloadBytes: Int64 = 19_530_936_278
    public static let ornith35BMLX6BitUpstreamRepoId = "ornith-ai/Ornith-1.5-35B-A3B-MLX-6bit"
    public static let ornith35BMLX6BitUpstreamRevision = "585b7867b0517980293ece857b26d64e84491352"
    public static let ornith35BMLX6BitEstimatedDownloadBytes: Int64 = 28_190_535_198
    public static let ornith35BMLX8BitUpstreamRepoId = "ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit"
    public static let ornith35BMLX8BitUpstreamRevision = "02440c39bdf7365c494a7f55f2a8b104ba87562f"
    public static let ornith35BMLX8BitEstimatedDownloadBytes: Int64 = 36_850_134_639
    public static let ornith35BMTPUpstreamRepoId = "ornith-ai/Ornith-1.5-35B-A3B"
    public static let ornith35BMTPUpstreamRevision = "e4dfb35a93d4b6822a811a7676f3488514abe7e2"
    public static let ornith35BMTPShardFilename = "model-00016-of-00016.safetensors"
    public static let ornith35BMTPSnapshotPatterns = [
        "README.md",
        "model.safetensors.index.json",
        ornith35BMTPShardFilename,
    ]
    public static let ornith35BMTPEstimatedDownloadBytes: Int64 = 4_379_189_705
    public static let ornith35BMLXContextLength = 262_144
    public static let infinityParser2ProUpstreamRepoId = "infly/Infinity-Parser2-Pro"
    public static let infinityParser2ProUpstreamRevision = "1d070df7db5acca0ffa75596229070a047704f89"
    public static let infinityParser2ProInt8UpstreamRepoId = "Sawfwair/Infinity-Parser2-Pro-Int8"
    public static let infinityParser2ProInt8UpstreamRevision = "main"

    private static let profilesByModelId: [String: Profile] = [
        q36NanoModelId: Profile(
            modelId: q36NanoModelId,
            upstreamRepoId: q36NanoUpstreamRepoId,
            upstreamRevision: q36NanoUpstreamRevision
        ),
        q38TwentySevenBModelId: Profile(
            modelId: q38TwentySevenBModelId,
            upstreamRepoId: q38TwentySevenBUpstreamRepoId,
            upstreamRevision: q38TwentySevenBUpstreamRevision,
            snapshotPatterns: snapshotPatterns + [
                "generation_config.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
                "merges.txt",
                "vocab.json",
            ]
        ),
        q38TwentySevenB4BitModelId: Profile(
            modelId: q38TwentySevenB4BitModelId,
            upstreamRepoId: q38TwentySevenB4BitUpstreamRepoId,
            upstreamRevision: q38TwentySevenB4BitUpstreamRevision,
            snapshotPatterns: snapshotPatterns + [
                "generation_config.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
                "vocab.json",
            ]
        ),
        q38FlashNextMixedModelId: Profile(
            modelId: q38FlashNextMixedModelId,
            upstreamRepoId: q38FlashNextMixedUpstreamRepoId,
            upstreamRevision: q38FlashNextMixedUpstreamRevision,
            snapshotPatterns: snapshotPatterns + [
                "generation_config.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
                "README.md",
                "MERERUN_CONVERSION.json",
            ]
        ),
        q38FlashNext4BitModelId: Profile(
            modelId: q38FlashNext4BitModelId,
            upstreamRepoId: q38FlashNext4BitUpstreamRepoId,
            upstreamRevision: q38FlashNext4BitUpstreamRevision,
            snapshotPatterns: snapshotPatterns + [
                "generation_config.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
                "README.md",
                "MERERUN_CONVERSION.json",
            ]
        ),
        q38FlashNext3BitModelId: Profile(
            modelId: q38FlashNext3BitModelId,
            upstreamRepoId: q38FlashNext3BitUpstreamRepoId,
            upstreamRevision: q38FlashNext3BitUpstreamRevision,
            snapshotPatterns: snapshotPatterns + [
                "generation_config.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
                "README.md",
                "MERERUN_CONVERSION.json",
                "MERERUN_QUALIFICATION.json",
            ]
        ),
        q38FlashNext3BitNativePLEModelId: Profile(
            modelId: q38FlashNext3BitNativePLEModelId,
            upstreamRepoId: q38FlashNext3BitNativePLEUpstreamRepoId,
            upstreamRevision: q38FlashNext3BitNativePLEUpstreamRevision,
            snapshotPatterns: snapshotPatterns + [
                "generation_config.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
                "README.md",
                "MERERUN_CONVERSION.json",
                "MERERUN_QUALIFICATION.json",
                "MERERUN_PLE_STORE.json",
                "MERERUN_PLE_PACK.json",
            ]
        ),
        bonsai27B1BitModelId: Profile(
            modelId: bonsai27B1BitModelId,
            upstreamRepoId: bonsai27B1BitUpstreamRepoId,
            upstreamRevision: bonsai27B1BitUpstreamRevision
        ),
        bonsai27B2BitModelId: Profile(
            modelId: bonsai27B2BitModelId,
            upstreamRepoId: bonsai27B2BitUpstreamRepoId,
            upstreamRevision: bonsai27B2BitUpstreamRevision
        ),
        ornith9BModelId: Profile(
            modelId: ornith9BModelId,
            upstreamRepoId: ornith9BUpstreamRepoId,
            upstreamRevision: ornith9BUpstreamRevision
        ),
        ornith35BMLXModelId: Profile(
            modelId: ornith35BMLXModelId,
            upstreamRepoId: ornith35BMLXUpstreamRepoId,
            upstreamRevision: ornith35BMLXUpstreamRevision,
            snapshotPatterns: snapshotPatterns + ["generation_config.json", "README.md"]
        ),
        ornith35BMLX4BitModelId: Profile(
            modelId: ornith35BMLX4BitModelId,
            upstreamRepoId: ornith35BMLX4BitUpstreamRepoId,
            upstreamRevision: ornith35BMLX4BitUpstreamRevision,
            snapshotPatterns: snapshotPatterns + ["generation_config.json", "README.md"]
        ),
        ornith35BMLX6BitModelId: Profile(
            modelId: ornith35BMLX6BitModelId,
            upstreamRepoId: ornith35BMLX6BitUpstreamRepoId,
            upstreamRevision: ornith35BMLX6BitUpstreamRevision,
            snapshotPatterns: snapshotPatterns + ["generation_config.json", "README.md"]
        ),
        ornith35BMLX8BitModelId: Profile(
            modelId: ornith35BMLX8BitModelId,
            upstreamRepoId: ornith35BMLX8BitUpstreamRepoId,
            upstreamRevision: ornith35BMLX8BitUpstreamRevision,
            snapshotPatterns: snapshotPatterns + ["generation_config.json", "README.md"]
        ),
        infinityParser2ProModelId: Profile(
            modelId: infinityParser2ProModelId,
            upstreamRepoId: infinityParser2ProUpstreamRepoId,
            upstreamRevision: infinityParser2ProUpstreamRevision
        ),
        infinityParser2ProInt8ModelId: Profile(
            modelId: infinityParser2ProInt8ModelId,
            upstreamRepoId: infinityParser2ProInt8UpstreamRepoId,
            upstreamRevision: infinityParser2ProInt8UpstreamRevision
        ),
    ]

    public static var supportedModelIds: Set<String> {
        Set(profilesByModelId.keys)
    }

    public static func profile(for modelId: String) -> Profile? {
        let normalized = modelId.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return profilesByModelId[normalized]
    }

    public static let defaultContextLength = 16_384

    public static func defaultContextLength(forModelId modelId: String) -> Int {
        switch modelId {
        case q38TwentySevenBModelId,
             q38TwentySevenB4BitModelId,
             q38FlashNextMixedModelId,
             q38FlashNext3BitModelId,
             q38FlashNext3BitNativePLEModelId,
             q38FlashNext4BitModelId:
            q38TwentySevenBContextLength
        case bonsai27B1BitModelId:
            bonsai27B1BitContextLength
        case bonsai27B2BitModelId:
            bonsai27B2BitContextLength
        case ornith35BMLX4BitModelId,
             ornith35BMLX6BitModelId,
             ornith35BMLX8BitModelId,
             ornith35BMLXModelId:
            ornith35BMLXContextLength
        default:
            defaultContextLength
        }
    }

    public static func visionPixelBounds(forModelId modelId: String) -> (minimum: Int, maximum: Int) {
        if isQ38ModelId(modelId) {
            return (q38TwentySevenBVisionMinPixels, q38TwentySevenBVisionMaxPixels)
        }
        return (Q35Generator.qwen3VLMinPixels, Q35Generator.qwen3VLMaxPixels)
    }

    public static func isQ38ModelId(_ modelId: String) -> Bool {
        modelId == q38TwentySevenBModelId
            || modelId == q38TwentySevenB4BitModelId
            || modelId == q38FlashNextMixedModelId
            || modelId == q38FlashNext3BitModelId
            || modelId == q38FlashNext3BitNativePLEModelId
            || modelId == q38FlashNext4BitModelId
    }

    public static func isOrnith35BMLXModelId(_ modelId: String) -> Bool {
        modelId == ornith35BMLX4BitModelId
            || modelId == ornith35BMLX6BitModelId
            || modelId == ornith35BMLX8BitModelId
            || modelId == ornith35BMLXModelId
    }

    public static let snapshotPatterns = [
        "LICENSE*",
        "NOTICE*",
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "optiq_metadata.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "*.safetensors",
    ]

    public var rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var configURL: URL { rootURL.appending(path: "config.json") }
    public var modelIndexURL: URL { rootURL.appending(path: "model.safetensors.index.json") }
    public var modelWeightsURL: URL { rootURL.appending(path: "model.safetensors") }
    public var tokenizerURL: URL { rootURL.appending(path: "tokenizer.json") }
    public var tokenizerConfigURL: URL { rootURL.appending(path: "tokenizer_config.json") }
    public var chatTemplateURL: URL { rootURL.appending(path: "chat_template.jinja") }
    public var generationConfigURL: URL { rootURL.appending(path: "generation_config.json") }
    public var processorConfigURL: URL { rootURL.appending(path: "processor_config.json") }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []
        if !fileManager.fileExists(atPath: configURL.path) {
            missing.append(configURL)
        }

        let hasIndex = fileManager.fileExists(atPath: modelIndexURL.path)
        let hasSingle = fileManager.fileExists(atPath: modelWeightsURL.path)
        if !hasIndex && !hasSingle {
            missing.append(modelIndexURL)
        }

        if !fileManager.fileExists(atPath: tokenizerURL.path) {
            missing.append(tokenizerURL)
        }
        if !fileManager.fileExists(atPath: chatTemplateURL.path)
            && !fileManager.fileExists(atPath: tokenizerConfigURL.path) {
            missing.append(chatTemplateURL)
        }

        return missing
    }

    public func validateQ38MTPComponent(fileManager: FileManager = .default) -> [URL] {
        let componentRoot = rootURL.appendingPathComponent(
            Self.q38MTPComponentPath,
            isDirectory: true
        )
        return Self.q38MTPComponentSnapshotPatterns
            .map { componentRoot.appendingPathComponent($0, isDirectory: false) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public var q38VisionComponentResources: Q35Resources {
        Q35Resources(rootURL: rootURL.appendingPathComponent(
            Self.q38VisionComponentPath,
            isDirectory: true
        ))
    }

    public func validateQ38VisionComponent(fileManager: FileManager = .default) -> [URL] {
        let componentRoot = q38VisionComponentResources.rootURL
        return Self.q38VisionComponentSnapshotPatterns
            .map { componentRoot.appendingPathComponent($0, isDirectory: false) }
            .filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public func validateOrnith35BMTPCompanion(fileManager: FileManager = .default) -> [URL] {
        [
            modelIndexURL,
            rootURL.appendingPathComponent(Self.ornith35BMTPShardFilename, isDirectory: false),
        ].filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public static func normalizedRootURL(_ rootURL: URL, fileManager: FileManager = .default) -> URL {
        let standardized = rootURL.standardizedFileURL
        let directConfig = standardized.appendingPathComponent("config.json")
        if fileManager.fileExists(atPath: directConfig.path) {
            return standardized
        }

        if let children = try? fileManager.contentsOfDirectoryResolvingSymlinks(
            at: standardized,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for child in children {
                let config = child.appendingPathComponent("config.json")
                if fileManager.fileExists(atPath: config.path) {
                    return child.standardizedFileURL
                }
            }
        }

        return standardized
    }
}

public enum Q35Error: LocalizedError {
    case modelNotLoaded
    case unsupportedModelId(String)
    case missingFiles([String])
    case downloadFailed(String)
    case extractionFailed
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Qwen-family model not loaded"
        case .unsupportedModelId(let id):
            return "Unsupported Qwen-family model id: \(id)"
        case .missingFiles(let files):
            return "Missing required files: \(files.joined(separator: ", "))"
        case .downloadFailed(let message):
            return "Download failed: \(message)"
        case .extractionFailed:
            return "Failed to prepare Qwen-family model files"
        case .generationFailed(let message):
            return message
        }
    }
}
