import Foundation

/// Shared model lookup utilities for mere.run's public model families.
///
/// Phase 1 scope:
/// - Resolve known model IDs (e.g. `image-klein-max`, `image-zimage-max`) to a local model root directory.
/// - Resolve from mere.run's configured local model store (default: `~/Library/Application Support/MereRun/models/...`).
///
/// This intentionally does *not* download models; it only resolves paths.
public struct ModelResolver {
    public enum ModelID: String, CaseIterable, Hashable, Sendable {
        case kleinNano = "image-klein-nano"
        case kleinMax = "image-klein-max"
        case kleinBase = "image-klein-base"
        case kleinShared = "image-klein-shared"
        case bonsaiBinary = "image-bonsai-binary"
        case bonsaiTernary = "image-bonsai-ternary"
        case zetaNano = "image-zimage-nano"
        case zetaMax = "image-zimage-max"
        case zetaBase = "image-zimage-base"
        case hidreamO1 = "image-hidream-o1"
        case hidreamO1Dev = "image-hidream-o1-dev"
        case mebot = "text-chat-mebot"
        case psiAgent = "text-chat-psi-agent"
        case gemma4 = "text-chat-gemma4"
        case gemma4Nano = "text-chat-gemma4-nano"
        case gemma4Max = "text-chat-gemma4-max"
        case gemma4Turbo = "text-chat-gemma4-turbo"
        case gemma4TwelveB = "text-chat-gemma4-12b"
        case gemma4VisionTwelveB = "vision-chat-gemma4-12b"
        case q36Nano = "text-chat-q36-nano"
        case lfm25A1B8Bit = "text-chat-lfm25-a1b-8bit"
        case qwen35Agent9B = "text-agent-qwen35-9b"
        case q36NanoGGUF = "text-chat-q36-nano-gguf"
        case deepseekV4Flash = "text-agent-deepseek-v4-flash"
        case qwen3TTSNano = "speech-tts-qwen3-nano"
        case qwen3TTSCustomVoice = "speech-tts-qwen3-customvoice"
        case qwen3ASR = "speech-asr-qwen3"
        case parakeetASR = "speech-asr-parakeet"
        case qwen3Code = "text-code-qwen3"
        case qwen3Embedding = "text-embed-qwen3-0.6b"
        case privacyFilter = "text-anonymize-privacy-filter"
        case lightOnOCR = "vision-ocr-lighton"
        case visionSegmentSAM31 = "vision-segment-sam31"
        case visionGroundFalconPerception = "vision-ground-falcon-perception"
        case aceStep = "music-acestep"
        case ltxVideoAV = "video-ltx-av"
    }

    public enum Source: String, Hashable, Sendable {
        /// `.../models/<id>` under the configured local mere.run model store.
        case localModelStore
    }

    public struct Resolution: Hashable, Sendable {
        public let modelID: ModelID
        public let rootURL: URL
        public let source: Source

        public init(modelID: ModelID, rootURL: URL, source: Source) {
            self.modelID = modelID
            self.rootURL = rootURL
            self.source = source
        }
    }

    public enum ResolverError: LocalizedError, Sendable {
        case applicationSupportUnavailable
        case modelNotFound(ModelID, searched: [URL], upstreamRepoId: String?)

        public var errorDescription: String? {
            switch self {
            case .applicationSupportUnavailable:
                return "Could not locate Application Support directory."
            case .modelNotFound(let id, let searched, let upstreamRepoId):
                var lines: [String] = []
                lines.append("Model not found: \(id.rawValue)")
                if let upstreamRepoId {
                    lines.append("Upstream repo: \(upstreamRepoId)")
                }
                if !searched.isEmpty {
                    lines.append("Searched:")
                    lines.append(contentsOf: searched.map { "  - \($0.path)" })
                }
                return lines.joined(separator: "\n")
            }
        }
    }

    private let fileManager: FileManager

    public init(
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        self.fileManager = fileManager
        _ = environment
    }

    public func resolve(_ modelID: ModelID) throws -> Resolution {
        if let direct = resolveDirect(modelID) {
            return direct
        }

        let spec = ManagedModelCatalog.spec(for: modelID.rawValue)
        for fallbackID in spec?.resolutionFallbackIDs ?? [] {
            guard let fallbackModelID = ModelID(rawValue: fallbackID),
                  let resolved = resolveDirect(fallbackModelID) else {
                continue
            }
            return Resolution(modelID: modelID, rootURL: resolved.rootURL, source: resolved.source)
        }

        let searchedIDs = [modelID] + (spec?.resolutionFallbackIDs.compactMap(ModelID.init(rawValue:)) ?? [])
        throw ResolverError.modelNotFound(
            modelID,
            searched: searchedPaths(for: searchedIDs),
            upstreamRepoId: spec?.upstreamRepoId
        )
    }

    public func resolveIfPresent(_ modelID: ModelID) -> Resolution? {
        try? resolve(modelID)
    }

    private func candidateModelRoots() -> [URL] {
        [MereRunModelPaths.modelsDir.standardizedFileURL]
    }

    private func resolveDirect(_ modelID: ModelID) -> Resolution? {
        guard let spec = ManagedModelCatalog.spec(for: modelID.rawValue) else {
            return nil
        }

        for modelsRoot in candidateModelRoots() {
            let modelRoot = modelsRoot.appendingPathComponent(modelID.rawValue, isDirectory: true)
            if isValidModelRoot(modelRoot, spec: spec, expectedModelID: modelID) {
                return Resolution(modelID: modelID, rootURL: modelRoot, source: .localModelStore)
            }
        }
        return nil
    }

    private func searchedPaths(for modelIDs: [ModelID]) -> [URL] {
        candidateModelRoots().flatMap { modelsRoot in
            modelIDs.map { modelsRoot.appendingPathComponent($0.rawValue, isDirectory: true) }
        }
    }

    private func isValidModelRoot(
        _ url: URL,
        spec: ManagedModelSpec,
        expectedModelID: ModelID? = nil
    ) -> Bool {
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            return false
        }

        guard let expectedModelID else {
            return spec.isManagedRootComplete(url, fileManager: fileManager)
        }

        let manifestURL = url.appendingPathComponent(MereRunModelManifest.filename)
        if fileManager.fileExists(atPath: manifestURL.path),
           let loaded = try? MereRunModelManifest.loadRequired(from: url, fileManager: fileManager),
           loaded.id == expectedModelID.rawValue {
            return spec.isManagedRootComplete(url, fileManager: fileManager)
        }

        return false
    }
}
