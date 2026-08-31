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
        case senseNovaU15 = "image-sensenova-u1-5-8b-mot"
        case krea2Raw = "image-krea2-raw"
        case krea2Turbo = "image-krea2-turbo"
        case qwenImageEdit2511 = "image-qwen-edit-2511"
        case qwenImageEdit2511Lightning = "image-qwen-edit-2511-lightning"
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
        case lagunaS21 = "text-chat-laguna-s-2-1"
        case lagunaXS21 = "text-chat-laguna-xs-2-1"
        case lagunaS21DFlash = "text-chat-laguna-s-2-1-dflash"
        case inklingSmall = "text-chat-inkling-small"
        case museGlimmer30B = "vision-chat-muse-glimmer-30b"
        case museGlimmer30BDFlash2 = "vision-chat-muse-glimmer-30b-dflash2"
        case nemotron35Lightning = "text-chat-nemotron-35-lightning"
        case nemotron35LightningDSpark = "text-chat-nemotron-35-lightning-dspark"
        case nemotron3NanoOmni30BA3BBF16 = "omni-chat-nemotron3-nano-30b-a3b-bf16"
        case ltxGemma3TwelveB4Bit = "text-encoder-ltx-gemma3-12b-4bit"
        case q36Nano = "text-chat-q36-nano"
        case q38TwentySevenB = "vision-chat-q38-27b"
        case q38TwentySevenB4Bit = "vision-chat-q38-27b-4bit"
        case q38FlashNextMixed = "vision-chat-q38-flash-next-mixed"
        case q38FlashNext3Bit = "vision-chat-q38-flash-next-3bit"
        case q38FlashNext3BitNativePLE = "vision-chat-q38-flash-next-3bit-native-ple"
        case q38FlashNext4Bit = "vision-chat-q38-flash-next-4bit"
        case bonsai27B1Bit = "text-chat-bonsai-27b-1bit"
        case bonsai27B2Bit = "text-chat-bonsai-27b-2bit"
        case lfm25A1B8Bit = "text-chat-lfm25-a1b-8bit"
        case lfm25A1BBF16 = "text-chat-lfm25-a1b-bf16"
        case lfm25Small1_2BBF16 = "text-chat-lfm25-1.2b-bf16"
        case lfm25Small1_2BQAD4Bit = "text-chat-lfm25-1.2b-qad-4bit"
        case lfm25Dense2_6B4Bit = "text-chat-lfm25-2.6b-4bit"
        case lfm25Dense2_6BBF16 = "text-chat-lfm25-2.6b-bf16"
        case lfm25Dense2_6BQAD4Bit = "text-chat-lfm25-2.6b-qad-4bit"
        case lfm25A1BDSpark = "text-chat-lfm25-a1b-dspark"
        case lfm25Small1_2BDSpark = "text-chat-lfm25-1.2b-dspark"
        case lfm25Dense2_6BDSpark = "text-chat-lfm25-2.6b-dspark"
        case lfm25VL3B8Bit = "vision-chat-lfm25-3b-8bit"
        case qwen35Agent9B = "text-agent-qwen35-9b"
        case ornith9B = "text-agent-ornith-9b"
        case ornith35B = "text-agent-ornith-35b"
        case ornith35BMLX4Bit = "text-agent-ornith-35b-mlx-4bit"
        case ornith35BMLX6Bit = "text-agent-ornith-35b-mlx-6bit"
        case ornith35BMLX8Bit = "text-agent-ornith-35b-mlx-8bit"
        case ornith35BMLX = "text-agent-ornith-35b-mlx"
        case ornith35BMTP = "text-agent-ornith-35b-mtp"
        case northMiniCode = "text-code-north-mini"
        case q36NanoGGUF = "text-chat-q36-nano-gguf"
        case deepseekV4Flash = "text-agent-deepseek-v4-flash"
        case qwen3TTSNano = "speech-tts-qwen3-nano"
        case qwen3TTSCustomVoice = "speech-tts-qwen3-customvoice"
        case qwen3ASR = "speech-asr-qwen3"
        case parakeetASR = "speech-asr-parakeet"
        case sortformerDiarization = "speech-diarization-sortformer"
        case qwen3Code = "text-code-qwen3"
        case qwen3Embedding = "text-embed-qwen3-0.6b"
        case privacyFilter = "text-anonymize-privacy-filter"
        case lightOnOCR = "vision-ocr-lighton"
        case infinityParser2Pro = "vision-ocr-infinity-pro"
        case infinityParser2ProInt8 = "vision-ocr-infinity-pro-int8"
        case visionSegmentSAM31 = "vision-segment-sam31"
        case visionGroundFalconPerception = "vision-ground-falcon-perception"
        case visionFloodTerraMindBase = "vision-flood-terramind-base"
        case visionFireTerraMindBase = "vision-fire-terramind-base"
        case visionEmbedTESSERAV2Nano = "vision-embed-tessera-v2-nano"
        case visionEmbedTESSERAV2Small = "vision-embed-tessera-v2-small"
        case visionEmbedTESSERAV2Medium = "vision-embed-tessera-v2-medium"
        case visionEmbedTESSERAV2Large = "vision-embed-tessera-v2-large"
        case visionEmbedTESSERAV2Teacher = "vision-embed-tessera-v2-teacher"
        case visionEmbedOlmoEarthV12Nano = "vision-embed-olmoearth-v12-nano"
        case visionEmbedOlmoEarthV12Tiny = "vision-embed-olmoearth-v12-tiny"
        case visionEmbedOlmoEarthV12Small = "vision-embed-olmoearth-v12-small"
        case visionEmbedOlmoEarthV12Base = "vision-embed-olmoearth-v12-base"
        case visionEmbedQwen3VL2B = "vision-embed-qwen3-vl-2b"
        case visionFaceBuffaloL = "vision-face-buffalo-l"
        case visionGeometryMoGe2Small = "vision-geometry-moge2-small"
        case visionDepthVDASmall = "vision-depth-vda-small"
        case visionDepthVDASmallMetric = "vision-depth-vda-small-metric"
        case visionGeometryDA3Small = "vision-geometry-da3-small"
        case image3DTripoSR = "image-3d-triposr"
        case image3DInstantMeshBase = "image-3d-instantmesh-base"
        case image3DTrellis2 = "image-3d-trellis2-4b"
        case aceStep = "music-acestep"
        case aceStepXLBase = "music-acestep-xl-base"
        case aceStepXLSFT = "music-acestep-xl-sft"
        case aceStepXLTurbo = "music-acestep-xl-turbo"
        case aceStepXLTurboLM4B = "music-acestep-xl-turbo-lm4b"
        case aceStepLM17B = "music-acestep-lm-1.7b"
        case aceStepLM4B = "music-acestep-lm-4b"
        case miniMaxMusic3 = "music-minimax-music3"
        case magentaRT2Small = "music-magenta-rt2-small"
        case magentaRT2Base = "music-magenta-rt2-base"
        case muScriptorSmall = "music-muscriptor-small"
        case muScriptorMedium = "music-muscriptor-medium"
        case muScriptorLarge = "music-muscriptor-large"
        case roFormerViperX1297 = "music-separate-bs-roformer-viperx-1297"
        case roFormerFourStem = "music-separate-bs-roformer-4stem"
        case melRoFormerDereverb = "music-separate-mel-roformer-dereverb"
        case melRoFormerDenoise = "music-separate-mel-roformer-denoise"
        case apBWE16kTo48k = "audio-enhance-ap-bwe-16kto48k"
        case univerSRAudio = "audio-enhance-universr-audio"
        case wooshDFlow = "sfx-woosh-dflow"
        case wooshFlow = "sfx-woosh-flow"
        case wooshClap = "sfx-woosh-clap"
        case wooshSynchformer = "sfx-woosh-synchformer"
        case wooshVFlow8s = "sfx-woosh-vflow-8s"
        case wooshDVFlow8s = "sfx-woosh-dvflow-8s"
        case mmaudioLarge44kV2 = "sfx-mmaudio-large-44k-v2"
        case ltxVideoAV = "video-ltx-av"
        case ltxVideo23AVMLX = "video-ltx23-av-mlx"
        case ltxVideo23FullMLX = "video-ltx23-full-mlx"
        case ltxVideo23A2VMLX = "video-ltx23-a2vid-mlx"
        case ltxVideo25DistilledBF16 = "video-ltx25-distilled-bf16"
        case ltxVideo25FullBF16 = "video-ltx25-full-bf16"
        case wan22TI2V5BMLX = "video-wan22-ti2v-5b-mlx"
        case miniMaxH3FL2VAMLX = "video-minimax-h3-fl2va-mlx"
        case miniMaxH3FL2VABF16MLX = "video-minimax-h3-fl2va-bf16-mlx"
        case miniMaxH3FL2VAQ8MLX = "video-minimax-h3-fl2va-8bit-mlx"
        case miniMaxH3FastH3VSADataFreeMLX = "video-minimax-h3-fasth3-vsa-datafree-mlx"
        case miniMaxH3Ref2VAMLX = "video-minimax-h3-ref2va-mlx"
        case cosmos3EdgeMLX = "video-cosmos3-edge-mlx"
        case scail2Video14BMLX = "video-scail2-14b-mlx"
        case dreamXWorld5BARMLX = "video-dreamx-world-5b-ar-mlx"
    }

    public enum Source: String, Hashable, Sendable {
        /// `.../models/<id>` under the configured local mere.run model store.
        case localModelStore
        /// An explicit canonical-model-id to directory binding.
        case registeredBinding
        /// A canonical `<root>/<model-id>` directory under a registered read-only root.
        case registeredSearchRoot
    }

    public struct Resolution: Hashable, Sendable {
        public let modelID: ModelID
        public let rootURL: URL
        public let source: Source
        public let catalogRootURL: URL?

        public init(
            modelID: ModelID,
            rootURL: URL,
            source: Source,
            catalogRootURL: URL? = nil
        ) {
            self.modelID = modelID
            self.rootURL = rootURL
            self.source = source
            self.catalogRootURL = catalogRootURL
        }

        public var isExternallyManaged: Bool {
            source != .localModelStore
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
    private let locations: ModelLocationSnapshot

    public init(
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        locations: ModelLocationSnapshot? = nil
    ) {
        self.fileManager = fileManager
        self.locations = locations ?? MereRunModelLocations.snapshot(
            fileManager: fileManager,
            environment: environment
        )
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
            return Resolution(
                modelID: modelID,
                rootURL: resolved.rootURL,
                source: resolved.source,
                catalogRootURL: resolved.catalogRootURL
            )
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

    /// Ordered locations considered for a canonical model id.
    ///
    /// The writable primary store is always first, followed by explicit bindings
    /// and then canonical directories under registered search roots.
    public func locationCandidates(for modelID: ModelID) -> [ModelLocationCandidate] {
        locations.candidates(for: modelID.rawValue)
    }

    private func resolveDirect(_ modelID: ModelID) -> Resolution? {
        guard let spec = ManagedModelCatalog.spec(for: modelID.rawValue) else {
            return nil
        }

        for candidate in locations.candidates(for: modelID.rawValue) {
            if isValidModelRoot(candidate, spec: spec, expectedModelID: modelID) {
                return Resolution(
                    modelID: modelID,
                    rootURL: candidate.rootURL,
                    source: source(for: candidate.kind),
                    catalogRootURL: candidate.catalogRootURL
                )
            }
        }
        return nil
    }

    private func searchedPaths(for modelIDs: [ModelID]) -> [URL] {
        modelIDs.flatMap { modelID in
            locations.candidates(for: modelID.rawValue).map(\.rootURL)
        }
    }

    private func isValidModelRoot(
        _ candidate: ModelLocationCandidate,
        spec: ManagedModelSpec,
        expectedModelID: ModelID? = nil
    ) -> Bool {
        let url = candidate.rootURL
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            return false
        }

        guard let expectedModelID else {
            return spec.isManagedRuntimeReady(url, fileManager: fileManager)
        }

        let manifest: MereRunModelManifest?
        do {
            manifest = try MereRunModelManifest.loadIfPresent(from: url, fileManager: fileManager)
        } catch {
            return false
        }
        if let manifest {
            guard manifest.id == expectedModelID.rawValue else { return false }
            if candidate.kind.isExternallyManaged,
               spec.usageRestriction != nil,
               manifest.usageTermsAcknowledged != true,
               !candidate.usageTermsAcknowledged {
                return false
            }
            return spec.isManagedRuntimeReady(url, fileManager: fileManager)
        }

        guard candidate.kind == .registeredBinding else { return false }
        guard spec.usageRestriction == nil || candidate.usageTermsAcknowledged else { return false }
        return spec.isManagedRuntimeReady(url, fileManager: fileManager)
    }

    private func source(for kind: ModelLocationKind) -> Source {
        switch kind {
        case .primaryStore: .localModelStore
        case .registeredBinding: .registeredBinding
        case .registeredSearchRoot: .registeredSearchRoot
        }
    }
}
