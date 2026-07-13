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
        case klein9B = "image-klein-9b"
        case kleinBase9B = "image-klein-base-9b"
        case kleinBase = "image-klein-base"
        case kleinShared = "image-klein-shared"
        case bonsaiBinary = "image-bonsai-binary"
        case bonsaiTernary = "image-bonsai-ternary"
        case zetaNano = "image-zimage-nano"
        case zetaMax = "image-zimage-max"
        case zetaBase = "image-zimage-base"
        case hidreamO1 = "image-hidream-o1"
        case hidreamO1Dev = "image-hidream-o1-dev"
        case krea2Raw = "image-krea2-raw"
        case krea2Turbo = "image-krea2-turbo"
        case ideogram4SDNQUInt4 = "image-ideogram4-sdnq-uint4"
        case mebot = "text-chat-mebot"
        case psiAgent = "text-chat-psi-agent"
        case gemma4 = "text-chat-gemma4"
        case gemma4Nano = "text-chat-gemma4-nano"
        case gemma4Max = "text-chat-gemma4-max"
        case gemma4Turbo = "text-chat-gemma4-turbo"
        case gemma4TwelveB = "text-chat-gemma4-12b"
        case gemma4TwelveB4Bit = "text-chat-gemma4-12b-4bit"
        case gemma4VisionTwelveB = "vision-chat-gemma4-12b"
        case gemma4TwelveBMTP = "text-chat-gemma4-12b-mtp"
        case ltxGemma3TwelveB4Bit = "text-encoder-ltx-gemma3-12b-4bit"
        case q36Nano = "text-chat-q36-nano"
        case lfm25A1B8Bit = "text-chat-lfm25-a1b-8bit"
        case qwen35Agent9B = "text-agent-qwen35-9b"
        case ornith9B = "text-agent-ornith-9b"
        case ornith35B = "text-agent-ornith-35b"
        case ornith35BMLX = "text-agent-ornith-35b-mlx"
        case northMiniCode = "text-code-north-mini"
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
        case infinityParser2Flash = "vision-ocr-infinity-flash"
        case infinityParser2Pro = "vision-ocr-infinity-pro"
        case infinityParser2ProInt8 = "vision-ocr-infinity-pro-int8"
        case visionSegmentSAM31 = "vision-segment-sam31"
        case visionGroundFalconPerception = "vision-ground-falcon-perception"
        case visionGeometryMoGe2Small = "vision-geometry-moge2-small"
        case visionDepthVDASmall = "vision-depth-vda-small"
        case visionDepthVDASmallMetric = "vision-depth-vda-small-metric"
        case visionGeometryDA3Small = "vision-geometry-da3-small"
        case image3DTripoSR = "image-3d-triposr"
        case image3DInstantMeshBase = "image-3d-instantmesh-base"
        case aceStep = "music-acestep"
        case aceStepXLTurbo = "music-acestep-xl-turbo"
        case aceStepXLTurboLM4B = "music-acestep-xl-turbo-lm4b"
        case magentaRT2Small = "music-magenta-rt2-small"
        case magentaRT2Base = "music-magenta-rt2-base"
        case muScriptorSmall = "music-muscriptor-small"
        case muScriptorMedium = "music-muscriptor-medium"
        case muScriptorLarge = "music-muscriptor-large"
        case wooshDFlow = "sfx-woosh-dflow"
        case wooshFlow = "sfx-woosh-flow"
        case wooshClap = "sfx-woosh-clap"
        case wooshSynchformer = "sfx-woosh-synchformer"
        case wooshVFlow8s = "sfx-woosh-vflow-8s"
        case wooshDVFlow8s = "sfx-woosh-dvflow-8s"
        case mmaudioLarge44kV2 = "sfx-mmaudio-large-44k-v2"
        case ltxVideoAV = "video-ltx-av"
        case ltxVideo23AVMLX = "video-ltx23-av-mlx"
        case wan22TI2V5BMLX = "video-wan22-ti2v-5b-mlx"
        case dreamXWorld5BARMLX = "video-dreamx-world-5b-ar-mlx"
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
            return spec.isManagedRuntimeReady(url, fileManager: fileManager)
        }

        let manifestURL = url.appendingPathComponent(MereRunModelManifest.filename)
        if fileManager.fileExists(atPath: manifestURL.path),
           let loaded = try? MereRunModelManifest.loadRequired(from: url, fileManager: fileManager),
           loaded.id == expectedModelID.rawValue {
            return spec.isManagedRuntimeReady(url, fileManager: fileManager)
        }

        return false
    }
}
