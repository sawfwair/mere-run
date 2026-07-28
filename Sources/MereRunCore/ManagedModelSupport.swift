import Foundation
#if canImport(Darwin)
import Darwin
#endif

public struct MereRunMachineProfile: Hashable, Sendable {
    public let physicalMemoryBytes: UInt64
    public let processorName: String
    public let isAppleSiliconMac: Bool
    public let isLinux: Bool

    public init(
        physicalMemoryBytes: UInt64,
        processorName: String,
        isAppleSiliconMac: Bool,
        isLinux: Bool = false
    ) {
        self.physicalMemoryBytes = physicalMemoryBytes
        self.processorName = processorName
        self.isAppleSiliconMac = isAppleSiliconMac
        self.isLinux = isLinux
    }

    public var unifiedMemoryGB: Int {
        max(1, Int(physicalMemoryBytes / 1_073_741_824))
    }

    public var isSupportedRuntime: Bool {
        isAppleSiliconMac || isLinux
    }

    public static var current: MereRunMachineProfile {
        MereRunMachineProfile(
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            processorName: currentProcessorName(),
            isAppleSiliconMac: currentIsAppleSiliconMac(),
            isLinux: currentIsLinux()
        )
    }

    private static func currentIsAppleSiliconMac() -> Bool {
        #if os(macOS) && arch(arm64)
        return true
        #else
        return false
        #endif
    }

    private static func currentIsLinux() -> Bool {
        #if os(Linux)
        return true
        #else
        return false
        #endif
    }

    private static func currentProcessorName() -> String {
        #if canImport(Darwin)
        if let brand = sysctlString("machdep.cpu.brand_string"), !brand.isEmpty {
            if brand.hasPrefix("Apple ") {
                return String(brand.dropFirst("Apple ".count))
            }
            return brand
        }
        #endif
        #if os(Linux)
        return "Linux"
        #else
        return currentIsAppleSiliconMac() ? "Apple Silicon" : "Unknown processor"
        #endif
    }

    #if canImport(Darwin)
    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else {
            return nil
        }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else {
            return nil
        }
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }
    #endif
}

public struct ManagedModelCapabilityDescriptor: Hashable, Sendable {
    public let modelID: String
    public let title: String
    public let summary: String
    public let minimumUnifiedMemoryGB: Int
    public let recommendedUnifiedMemoryGB: Int
    public let isRecommendedForSetup: Bool

    public init(
        modelID: String,
        title: String,
        summary: String,
        minimumUnifiedMemoryGB: Int,
        recommendedUnifiedMemoryGB: Int,
        isRecommendedForSetup: Bool
    ) {
        self.modelID = modelID
        self.title = title
        self.summary = summary
        self.minimumUnifiedMemoryGB = minimumUnifiedMemoryGB
        self.recommendedUnifiedMemoryGB = recommendedUnifiedMemoryGB
        self.isRecommendedForSetup = isRecommendedForSetup
    }
}

public enum ManagedModelSupportStatus: String, Hashable, Sendable {
    case supported
    case unsupported
}

public struct ManagedModelSupportReport: Hashable, Sendable {
    public let spec: ManagedModelSpec
    public let descriptor: ManagedModelCapabilityDescriptor
    public let machine: MereRunMachineProfile
    public let status: ManagedModelSupportStatus
    public let reasons: [String]

    public var isSupported: Bool {
        status == .supported
    }

    public var memoryHeadroomGB: Int {
        machine.unifiedMemoryGB - descriptor.minimumUnifiedMemoryGB
    }

    public var meetsRecommendedMemory: Bool {
        machine.unifiedMemoryGB >= descriptor.recommendedUnifiedMemoryGB
    }
}

public struct ManagedChatModelBandRecommendation: Hashable, Sendable {
    public let minimumUnifiedMemoryGB: Int
    public let maximumUnifiedMemoryGB: Int?
    public let modelID: String
    public let title: String
    public let summary: String
    public let alternateModelIDs: [String]

    public init(
        minimumUnifiedMemoryGB: Int,
        maximumUnifiedMemoryGB: Int?,
        modelID: String,
        title: String,
        summary: String,
        alternateModelIDs: [String] = []
    ) {
        self.minimumUnifiedMemoryGB = minimumUnifiedMemoryGB
        self.maximumUnifiedMemoryGB = maximumUnifiedMemoryGB
        self.modelID = modelID
        self.title = title
        self.summary = summary
        self.alternateModelIDs = alternateModelIDs
    }

    public var bandLabel: String {
        if let maximumUnifiedMemoryGB {
            return "\(minimumUnifiedMemoryGB)-\(maximumUnifiedMemoryGB) GB"
        }
        return "\(minimumUnifiedMemoryGB)+ GB"
    }

    public func contains(unifiedMemoryGB: Int) -> Bool {
        guard unifiedMemoryGB >= minimumUnifiedMemoryGB else { return false }
        if let maximumUnifiedMemoryGB {
            return unifiedMemoryGB <= maximumUnifiedMemoryGB
        }
        return true
    }
}

public enum ManagedModelCapabilityCatalog {
    private static let descriptorsByID: [String: ManagedModelCapabilityDescriptor] = {
        let descriptors = [
            descriptor(
                "image-klein-nano",
                "Image, fast artistic",
                "Creates fast FLUX.2 Klein images for prompts, drafts, and lightweight image workflows.",
                minimum: 8,
                recommended: 16,
                setup: true
            ),
            descriptor(
                "image-klein-max",
                "Image, artistic quality",
                "Creates higher-fidelity FLUX.2 Klein images when the machine has enough memory for the larger stack.",
                minimum: 24,
                recommended: 32
            ),
            descriptor(
                "image-klein-9b",
                "Image, large artistic quality",
                "Installs the gated FLUX.2 Klein 9B image model for higher-capacity prompt and reference-image workflows.",
                minimum: 48,
                recommended: 64
            ),
            descriptor(
                "image-klein-base-9b",
                "Image LoRA training base",
                "Installs the undistilled FLUX.2 Klein Base 9B checkpoint for higher-capacity LoRA training and research workflows.",
                minimum: 64,
                recommended: 96
            ),
            descriptor(
                "image-klein-base",
                "Image training base",
                "Installs the FLUX.2 Klein base layout used by image-training and compatibility workflows.",
                minimum: 24,
                recommended: 32
            ),
            descriptor(
                "image-klein-shared",
                "Shared Klein components",
                "Represents shared FLUX.2 Klein components resolved from another installed Klein image model.",
                minimum: 8,
                recommended: 16
            ),
            descriptor(
                "image-bonsai-binary",
                "Image, binary Klein",
                "Runs the Bonsai Image binary FLUX.2 Klein deployment through the native Swift MLX image runtime.",
                minimum: 16,
                recommended: 24
            ),
            descriptor(
                "image-bonsai-ternary",
                "Image, compact Klein",
                "Runs the Bonsai Image ternary FLUX.2 Klein deployment through the native Swift MLX image runtime.",
                minimum: 12,
                recommended: 16
            ),
            descriptor(
                "image-zimage-nano",
                "Image Nano",
                "Creates fast 4-bit Z-Image Turbo outputs for photorealistic and general image generation.",
                minimum: 12,
                recommended: 16,
                setup: true
            ),
            descriptor(
                "image-zimage-max",
                "Image, realistic quality",
                "Creates higher-fidelity Z-Image outputs from the larger Z-Image Turbo model.",
                minimum: 48,
                recommended: 64
            ),
            descriptor(
                "image-zimage-base",
                "Image realistic base",
                "Installs the larger Z-Image base layout for advanced image workflows.",
                minimum: 48,
                recommended: 64
            ),
            descriptor(
                "image-hidream-o1-dev",
                "Image, HiDream dev",
                "Runs the distilled HiDream O1 image model for text, reference editing, and subject personalization.",
                minimum: 48,
                recommended: 64
            ),
            descriptor(
                "image-hidream-o1",
                "Image, HiDream full",
                "Runs the full HiDream O1 image model for high-quality text, reference editing, and personalization workflows.",
                minimum: 96,
                recommended: 128
            ),
            descriptor(
                Krea2RawResources.modelId,
                "Image training, Krea 2 Raw",
                "Installs the Krea 2 Raw base checkpoint for LoRA training and post-training workflows.",
                minimum: 96,
                recommended: 128
            ),
            descriptor(
                Krea2Resources.modelId,
                "Image, Krea 2 Turbo",
                "Runs the Krea 2 Turbo 8-step text-to-image model and Raw-trained LoRAs through the native Swift MLX image runtime.",
                minimum: 96,
                recommended: 128
            ),
            descriptor(
                Ideogram4Resources.modelId,
                "Image, Ideogram 4 typography",
                "Installs the SDNQ uint4 Ideogram 4 text-to-image stack with dedicated positive and unconditional transformers.",
                minimum: 48,
                recommended: 64
            ),
            descriptor(
                "text-chat-mebot",
                "Local personal chat",
                "Adds a compact local chat model for API serving and personal-agent style workflows.",
                minimum: 8,
                recommended: 16
            ),
            descriptor(
                "text-chat-psi-agent",
                "Agent chat, large",
                "Adds the larger Psi agent chat model for local reasoning-heavy assistant workflows.",
                minimum: 64,
                recommended: 96
            ),
            descriptor(
                "text-chat-gemma4",
                "Gemma 4 dense bf16 alias",
                "Resolves to dense bf16 Gemma 4 chat models for native Swift text chat; use the TurboQuant tier on 32 GB Macs.",
                minimum: 48,
                recommended: 64
            ),
            descriptor(
                Gemma4Resources.turboModelId,
                "Gemma 4 NVFP4 MoE",
                "Installs the MLX NVFP4 Gemma 4 26B-A4B-it MoE snapshot for the native Swift 32 GB tier.",
                minimum: 24,
                recommended: 32,
                setup: true
            ),
            descriptor(
                Gemma4Resources.twelveBModelId,
                "Gemma 4 12B chat",
                "Runs the dense Gemma 4 12B instruction model through the native Swift text runtime.",
                minimum: 24,
                recommended: 32
            ),
            descriptor(
                Gemma4Resources.twelveB4BitModelId,
                "Gemma 4 12B 4-bit chat",
                "Installs the MLX 4-bit Gemma 4 12B instruction snapshot for faster native Swift text chat.",
                minimum: 16,
                recommended: 24,
                setup: true
            ),
            descriptor(
                Gemma4Resources.visionTwelveBModelId,
                "Gemma 4 12B vision chat",
                "Runs the dense Gemma 4 12B unified model for single-image chat through the native Swift runtime.",
                minimum: 32,
                recommended: 48
            ),
            descriptor(
                Gemma4MTPResources.modelId,
                "Gemma 4 12B MTP assistant",
                "Installs the companion drafter used for verified Gemma 4 12B decode-tail speculation.",
                minimum: 24,
                recommended: 32
            ),
            descriptor(
                "text-chat-gemma4-nano",
                "Gemma 4 chat nano",
                "Runs the smaller Gemma 4 native Swift chat model for lower-memory machines.",
                minimum: 16,
                recommended: 24,
                setup: true
            ),
            descriptor(
                "text-chat-gemma4-max",
                "Gemma 4 chat max",
                "Runs the larger Gemma 4 native Swift chat model for higher-quality local chat.",
                minimum: 48,
                recommended: 64
            ),
            descriptor(
                LagunaResources.modelID,
                "Laguna S 2.1 agentic chat",
                "Runs Poolside's 118B-A8B Laguna S 2.1 NVFP4 model with verified DFlash speculative decoding through the native Swift runtime.",
                minimum: 96,
                recommended: 128
            ),
            descriptor(
                Q35Resources.q36NanoModelId,
                "Qwen3.6 A3B chat nano",
                "Runs the Qwen3.6 35B-A3B OptiQ 4-bit MLX/MTP snapshot through the native Qwen-family runtime.",
                minimum: 24,
                recommended: 32,
                setup: true
            ),
            descriptor(
                Q35Resources.bonsai27B1BitModelId,
                "Bonsai 27B 1-bit vision chat",
                "Runs Prism ML's dense Qwen3.6 27B reasoning and vision model from packed 1-bit MLX weights.",
                minimum: 12,
                recommended: 16
            ),
            descriptor(
                Q35Resources.bonsai27B2BitModelId,
                "Ternary Bonsai 27B 2-bit vision chat",
                "Runs Prism ML's dense Qwen3.6 27B reasoning and vision model from packed ternary 2-bit MLX weights.",
                minimum: 16,
                recommended: 24
            ),
            descriptor(
                Q35Resources.ornith9BModelId,
                "Ornith 1.0 9B OptiQ",
                "Runs DeepReinforce's Ornith 1.0 9B agentic coding model through the native Qwen-family MLX runtime.",
                minimum: 16,
                recommended: 24
            ),
            descriptor(
                Q35Resources.ornith35BMLXModelId,
                "Ornith 1.0 35B MLX Q4",
                "Runs a converted Ornith 1.0 35B Q4 MLX MoE through the native Qwen-family runtime.",
                minimum: 24,
                recommended: 32
            ),
            descriptor(
                LFM2Resources.defaultModelId,
                "LFM2.5 A1B 8-bit",
                "Runs the LiquidAI LFM2.5 8B-A1B MLX 8-bit snapshot through the native Swift LFM2 runtime.",
                minimum: 16,
                recommended: 24,
                setup: true
            ),
            descriptor(
                "text-chat-q36-nano-gguf",
                "Qwen3.6 A3B chat nano (GGUF)",
                "Runs Qwen3.6 35B-A3B GGUF through llama.cpp; the default chat model on Linux CUDA hosts.",
                minimum: 24,
                recommended: 32,
                setup: false
            ),
            descriptor(
                AgentModelResources.qwen35NineBModelId,
                "Qwen3.5 9B agent",
                "Runs the Qwen3.5 9B Q4 GGUF setup agent for lower-memory Macs.",
                minimum: 16,
                recommended: 16,
                setup: true
            ),
            descriptor(
                NorthMiniCodeResources.modelId,
                "North Mini Code Q4",
                "Runs Cohere's compact 30B-A3B coding model through the native "
                + "llama.cpp/GGUF code runtime.",
                minimum: 24,
                recommended: 32
            ),
            descriptor(
                Ornith35BCodeResources.modelId,
                "Ornith 1.0 35B Q4",
                "Runs DeepReinforce's larger Ornith coding-agent GGUF through the native "
                + "llama.cpp/GGUF code runtime.",
                minimum: 32,
                recommended: 64
            ),
            descriptor(
                "speech-tts-qwen3-nano",
                "Text to speech",
                "Synthesizes speech with Qwen3 TTS, including style prompts and voice-design workflows.",
                minimum: 8,
                recommended: 16,
                setup: true
            ),
            descriptor(
                "speech-tts-qwen3-customvoice",
                "Custom voice TTS",
                "Synthesizes speech with the Qwen3 custom-voice model layout.",
                minimum: 12,
                recommended: 16
            ),
            descriptor(
                "speech-asr-qwen3",
                "Speech recognition, multilingual",
                "Transcribes and translates speech with Qwen3 ASR.",
                minimum: 12,
                recommended: 16
            ),
            descriptor(
                "speech-asr-parakeet",
                "Speech recognition, fast",
                "Transcribes speech quickly with the Parakeet ASR runtime.",
                minimum: 8,
                recommended: 16,
                setup: true
            ),
            descriptor(
                "text-code-qwen3",
                "Qwen3-Coder Next",
                "Runs the GGUF Qwen3-Coder Next model through llama.cpp for local coding and agent sessions.",
                minimum: 64,
                recommended: 96
            ),
            descriptor(
                DeepseekV4FlashResources.defaultModelId,
                "DeepSeek V4 Flash IQ2 imatrix",
                "Premier 284B MoE agent tier served by the bundled ds4-server engine "
                + "(imatrix-tuned q2 GGUF, ~81 GB) for 96 GB+ Apple Silicon Macs.",
                minimum: 96,
                recommended: 128,
                setup: true
            ),
            descriptor(
                "text-embed-qwen3-0.6b",
                "Text embeddings",
                "Creates Qwen3 embedding vectors for local search and retrieval workflows.",
                minimum: 8,
                recommended: 16,
                setup: true
            ),
            descriptor(
                OpenAIPrivacyFilterCatalog.modelId,
                "Text anonymization",
                "Detects and replaces private text spans before prompts or logs leave a local workflow.",
                minimum: 8,
                recommended: 16,
                setup: true
            ),
            descriptor(
                "vision-ocr-lighton",
                "OCR",
                "Reads text from images and documents with the LightOn OCR runtime.",
                minimum: 8,
                recommended: 16,
                setup: true
            ),
            descriptor(
                Q35Resources.infinityParser2FlashModelId,
                "Infinity-Parser2 Flash OCR",
                "Parses document images with the native Qwen-family Infinity-Parser2 Flash runtime.",
                minimum: 24,
                recommended: 32
            ),
            descriptor(
                Q35Resources.infinityParser2ProModelId,
                "Infinity-Parser2 Pro OCR",
                "Runs the heavyweight native Infinity-Parser2 Pro eval model for document parsing.",
                minimum: 128,
                recommended: 192
            ),
            descriptor(
                Q35Resources.infinityParser2ProInt8ModelId,
                "Infinity-Parser2 Pro int8 OCR",
                "Runs the quantized native Infinity-Parser2 Pro model for quality-focused OCR evals.",
                minimum: 64,
                recommended: 96
            ),
            descriptor(
                "vision-segment-sam31",
                "Vision segmentation",
                "Segments and tracks objects in images or video frames with SAM 3.1.",
                minimum: 16,
                recommended: 24
            ),
            descriptor(
                "vision-ground-falcon-perception",
                "Vision grounding",
                "Finds and labels objects in images with Falcon Perception grounding.",
                minimum: 16,
                recommended: 24
            ),
            descriptor(
                FaceAnalysisResources.modelID,
                "Face analysis",
                "Detects faces and landmarks, creates identity embeddings, and compares people locally.",
                minimum: 8,
                recommended: 16
            ),
            descriptor(
                ModelResolver.ModelID.visionGeometryMoGe2Small.rawValue,
                "Metric image geometry",
                "Predicts metric depth, point maps, normals, masks, and camera intrinsics with MoGe-2 Small.",
                minimum: 8,
                recommended: 16
            ),
            descriptor(
                ModelResolver.ModelID.visionDepthVDASmall.rawValue,
                "Temporal relative depth",
                "Predicts temporally consistent affine-relative video depth with Video Depth Anything Small.",
                minimum: 16,
                recommended: 32
            ),
            descriptor(
                ModelResolver.ModelID.visionDepthVDASmallMetric.rawValue,
                "Temporal metric depth",
                "Predicts temporally consistent metric video depth with Video Depth Anything Small.",
                minimum: 16,
                recommended: 32
            ),
            descriptor(
                ModelResolver.ModelID.visionGeometryDA3Small.rawValue,
                "Multi-view scene geometry",
                "Predicts relative depth, confidence, intrinsics, and extrinsics across multiple views with DA3 Small.",
                minimum: 12,
                recommended: 24
            ),
            descriptor(
                ModelResolver.ModelID.image3DTripoSR.rawValue,
                "Single-image object mesh",
                "Reconstructs a normalized object mesh and vertex colors from one image with TripoSR.",
                minimum: 16,
                recommended: 32
            ),
            descriptor(
                ModelResolver.ModelID.image3DInstantMeshBase.rawValue,
                "Multi-view object mesh",
                "Reconstructs a FlexiCubes object mesh from licensed user-supplied views with InstantMesh Base.",
                minimum: 24,
                recommended: 48
            ),
            descriptor(
                ModelResolver.ModelID.image3DTrellis2.rawValue,
                "PBR O-Voxel object mesh",
                "Reconstructs a 512-resolution PBR O-Voxel mesh with the native Swift MLX TRELLIS.2 runtime.",
                minimum: 64,
                recommended: 128
            ),
            descriptor(
                "music-acestep",
                "Music generation",
                "Generates music from text prompts with the ACE-Step model stack.",
                minimum: 24,
                recommended: 32
            ),
            descriptor(
                ModelResolver.ModelID.aceStepXLBase.rawValue,
                "ACE-Step XL Base",
                "Runs the full ACE-Step 1.5 XL Base DiT with CFG and advanced source-audio tasks.",
                minimum: 32,
                recommended: 64
            ),
            descriptor(
                ModelResolver.ModelID.aceStepXLSFT.rawValue,
                "ACE-Step XL SFT",
                "Runs the full ACE-Step 1.5 XL supervised checkpoint with CFG for highest-fidelity standard generation.",
                minimum: 32,
                recommended: 64
            ),
            descriptor(
                ModelResolver.ModelID.aceStepXLTurbo.rawValue,
                "ACE-Step XL Turbo",
                "Generates higher-quality music with the ACE-Step 1.5 XL turbo DiT.",
                minimum: 24,
                recommended: 32
            ),
            descriptor(
                ModelResolver.ModelID.aceStepXLTurboLM4B.rawValue,
                "ACE-Step XL Turbo + 4B LM",
                "Adds the optional 4B 5Hz LM to the ACE-Step 1.5 XL turbo stack.",
                minimum: 32,
                recommended: 64
            ),
            descriptor(
                ModelResolver.ModelID.magentaRT2Small.rawValue,
                "Magenta RealTime 2 small",
                "Streams controllable Magenta RT2 music on Apple Silicon with the 230M model.",
                minimum: 8,
                recommended: 16
            ),
            descriptor(
                ModelResolver.ModelID.magentaRT2Base.rawValue,
                "Magenta RealTime 2 base",
                "Streams higher-quality Magenta RT2 music on Pro/Max-class Apple Silicon.",
                minimum: 32,
                recommended: 64
            ),
            descriptor(
                ModelResolver.ModelID.muScriptorSmall.rawValue,
                "MuScriptor small",
                "Transcribes full music mixes into instrument-separated MIDI with the 103M MuScriptor model.",
                minimum: 8,
                recommended: 16
            ),
            descriptor(
                ModelResolver.ModelID.muScriptorMedium.rawValue,
                "MuScriptor medium",
                "Transcribes full music mixes with the default 307M MuScriptor quality/speed trade-off.",
                minimum: 16,
                recommended: 24
            ),
            descriptor(
                ModelResolver.ModelID.muScriptorLarge.rawValue,
                "MuScriptor large",
                "Runs the highest-accuracy 1.4B MuScriptor transcription checkpoint.",
                minimum: 32,
                recommended: 64
            ),
            descriptor(
                ModelResolver.ModelID.wooshDFlow.rawValue,
                "Woosh DFlow",
                "Generates Foley and sound effects from text prompts with Sony Research Woosh.",
                minimum: 16,
                recommended: 32
            ),
            descriptor(
                ModelResolver.ModelID.wooshFlow.rawValue,
                "Woosh Flow",
                "Generates Foley and sound effects from text prompts with Sony Research Woosh's original Flow model.",
                minimum: 16,
                recommended: 32
            ),
            descriptor(
                ModelResolver.ModelID.wooshClap.rawValue,
                "Woosh CLAP",
                "Embeds and scores sound effects against text with Sony Research Woosh-CLAP.",
                minimum: 16,
                recommended: 32
            ),
            descriptor(
                ModelResolver.ModelID.wooshSynchformer.rawValue,
                "Woosh Synchformer",
                "Extracts video synchronization features for Sony Research Woosh V2A generation.",
                minimum: 16,
                recommended: 32
            ),
            descriptor(
                ModelResolver.ModelID.wooshVFlow8s.rawValue,
                "Woosh VFlow 8s",
                "Generates 8-second sound effects from video with Sony Research Woosh VFlow.",
                minimum: 32,
                recommended: 64
            ),
            descriptor(
                ModelResolver.ModelID.wooshDVFlow8s.rawValue,
                "Woosh DVFlow 8s",
                "Generates 8-second sound effects from video with Sony Research Woosh distilled VFlow.",
                minimum: 32,
                recommended: 64
            ),
            descriptor(
                ModelResolver.ModelID.mmaudioLarge44kV2.rawValue,
                "MMAudio large 44.1 kHz v2",
                "Generates 44.1 kHz sound effects from text or synchronized video with native MMAudio.",
                minimum: 32,
                recommended: 64
            ),
            descriptor(
                "video-ltx-av",
                "Video generation",
                "Generates short audio-video clips with the LTX unified AV model stack.",
                minimum: 64,
                recommended: 96
            ),
            descriptor(
                ModelResolver.ModelID.ltxVideo23AVMLX.rawValue,
                "LTX 2.3 Distilled MLX",
                "Installs the standalone distilled LTX 2.3 MLX checkpoint for fast draft renders.",
                minimum: 96,
                recommended: 128
            ),
            descriptor(
                ModelResolver.ModelID.ltxVideo23FullMLX.rawValue,
                "LTX 2.3 Full MLX",
                "Installs the final-quality dev transformer, distilled LoRA, and optional generated-audio components.",
                minimum: 96,
                recommended: 128
            ),
            descriptor(
                ModelResolver.ModelID.ltxVideo23A2VMLX.rawValue,
                "LTX 2.3 A2Vid MLX (compatibility)",
                "Preserves the legacy narrow A2Vid install ID; prefer video-ltx23-full-mlx for new installs.",
                minimum: 96,
                recommended: 128
            ),
            descriptor(
                ModelResolver.ModelID.wan22TI2V5BMLX.rawValue,
                "Wan2.2 TI2V 5B MLX",
                "Generates text- and image-conditioned pixel video with the native Swift MLX Wan2.2 runtime.",
                minimum: 64,
                recommended: 96
            ),
            descriptor(
                ModelResolver.ModelID.cosmos3EdgeMLX.rawValue,
                "Cosmos3-Edge MLX",
                "Runs NVIDIA Cosmos3-Edge video, action, navigation, and reasoner modes with native Swift MLX.",
                minimum: 32,
                recommended: 64
            ),
            descriptor(
                ModelResolver.ModelID.scail2Video14BMLX.rawValue,
                "SCAIL-2 14B MLX",
                "Animates or replaces masked subjects from reference images and driving videos with native Swift/MLX.",
                minimum: 96,
                recommended: 128
            ),
            descriptor(
                ModelResolver.ModelID.dreamXWorld5BARMLX.rawValue,
                "DreamX World 5B AR MLX",
                "Runs persistent camera-conditioned world generation with the native causal Swift MLX runtime.",
                minimum: 64,
                recommended: 128
            ),
        ]
        return Dictionary(uniqueKeysWithValues: descriptors.map { ($0.modelID, $0) })
    }()

    public static var allDescriptors: [ManagedModelCapabilityDescriptor] {
        ManagedModelCatalog.allSpecs.compactMap { descriptorsByID[$0.id] }
    }

    public static func descriptor(for spec: ManagedModelSpec) -> ManagedModelCapabilityDescriptor {
        if let descriptor = descriptorsByID[spec.id] {
            return descriptor
        }
        preconditionFailure("Missing managed model capability descriptor for \(spec.id)")
    }

    public static func descriptor(for modelID: String) -> ManagedModelCapabilityDescriptor? {
        descriptorsByID[modelID]
    }

    public static func support(
        for spec: ManagedModelSpec,
        on machine: MereRunMachineProfile = .current
    ) -> ManagedModelSupportReport {
        let descriptor = descriptor(for: spec)
        var reasons: [String] = []

        if !machine.isSupportedRuntime {
            reasons.append("Apple Silicon macOS or Linux is required.")
        }

        if spec.validationKind == .magentaRT2 && !machine.isAppleSiliconMac {
            reasons.append("Magenta RT2 requires Apple Silicon macOS.")
        }

        if machine.unifiedMemoryGB < descriptor.minimumUnifiedMemoryGB {
            reasons.append(
                "Requires at least \(descriptor.minimumUnifiedMemoryGB) GB unified memory; detected \(machine.unifiedMemoryGB) GB."
            )
        }

        return ManagedModelSupportReport(
            spec: spec,
            descriptor: descriptor,
            machine: machine,
            status: reasons.isEmpty ? .supported : .unsupported,
            reasons: reasons
        )
    }

    public static func supportReports(
        on machine: MereRunMachineProfile = .current
    ) -> [ManagedModelSupportReport] {
        ManagedModelCatalog.allSpecs.map { support(for: $0, on: machine) }
    }

    public static func recommendedSetupReports(
        on machine: MereRunMachineProfile = .current
    ) -> [ManagedModelSupportReport] {
        supportReports(on: machine).filter {
            $0.isSupported
                && $0.descriptor.isRecommendedForSetup
                && $0.spec.hasAnyManagedDownloadSource()
        }
    }

    public static func recommendedChatBandReports(
        on machine: MereRunMachineProfile = .current
    ) -> [ManagedChatModelBandRecommendation] {
        let q36ModelID = machine.isLinux ? ModelResolver.ModelID.q36NanoGGUF.rawValue : Q35Resources.q36NanoModelId
        let defaultAssistantModelID = machine.isLinux ? q36ModelID : Gemma4Resources.twelveB4BitModelId
        let defaultAssistantSummary: String
        let headroomAssistantSummary: String
        if machine.isLinux {
            defaultAssistantSummary = "Q36 GGUF stays the Linux CUDA chat default; Gemma 12B 4-bit is the Apple Silicon local-assistant default."
            headroomAssistantSummary = "Keep the optimized Q36 GGUF Linux lane; use headroom for context, concurrency, or larger alternates."
        } else {
            defaultAssistantSummary = "Gemma 12B 4-bit is the conservative default for grounded local chat; use Turbo when you want to spend more memory."
            headroomAssistantSummary = "Keep the proven Gemma local-assistant lane as default; use the headroom for context, concurrency, or larger Gemma alternates."
        }
        return [
            ManagedChatModelBandRecommendation(
                minimumUnifiedMemoryGB: 16,
                maximumUnifiedMemoryGB: 23,
                modelID: Gemma4Resources.twelveB4BitModelId,
                title: "Compact native chat",
                summary: "Best first chat pick for the smallest supported RAM band; use nano only when the 12B 4-bit lane is too tight.",
                alternateModelIDs: [Gemma4Resources.nanoModelId, LFM2Resources.defaultModelId]
            ),
            ManagedChatModelBandRecommendation(
                minimumUnifiedMemoryGB: 24,
                maximumUnifiedMemoryGB: 63,
                modelID: defaultAssistantModelID,
                title: "Default local assistant",
                summary: defaultAssistantSummary,
                alternateModelIDs: [Gemma4Resources.turboModelId, q36ModelID]
            ),
            ManagedChatModelBandRecommendation(
                minimumUnifiedMemoryGB: 64,
                maximumUnifiedMemoryGB: 95,
                modelID: defaultAssistantModelID,
                title: "Gemma chat with headroom",
                summary: headroomAssistantSummary,
                alternateModelIDs: [Gemma4Resources.maxModelId, Gemma4Resources.turboModelId, q36ModelID]
            ),
            ManagedChatModelBandRecommendation(
                minimumUnifiedMemoryGB: 96,
                maximumUnifiedMemoryGB: nil,
                modelID: DeepseekV4FlashResources.defaultModelId,
                title: "Premier agent/API chat",
                summary: "Highest-tier agent/API chat model; keep Gemma 12B 4-bit as the normal interactive local-assistant fallback.",
                alternateModelIDs: [Gemma4Resources.twelveB4BitModelId, Gemma4Resources.maxModelId, q36ModelID]
            ),
        ]
    }

    public static func recommendedCodeModelReport(
        on machine: MereRunMachineProfile = .current
    ) -> ManagedModelSupportReport? {
        let candidateIDs: [String]
        if machine.unifiedMemoryGB >= 64 {
            candidateIDs = [
                CodeGenResources.defaultModelId,
                NorthMiniCodeResources.modelId,
                AgentModelResources.qwen35NineBModelId,
            ]
        } else if machine.unifiedMemoryGB >= 24 {
            candidateIDs = [
                NorthMiniCodeResources.modelId,
                AgentModelResources.qwen35NineBModelId,
            ]
        } else {
            candidateIDs = [
                AgentModelResources.qwen35NineBModelId,
            ]
        }

        return candidateIDs.compactMap { modelID -> ManagedModelSupportReport? in
            guard let spec = ManagedModelCatalog.spec(for: modelID) else { return nil }
            let report = support(for: spec, on: machine)
            return report.isSupported ? report : nil
        }.first
    }

    public static func supportedCodeBenchmarkModelIDs(
        on machine: MereRunMachineProfile = .current
    ) -> [String] {
        [
            Q35Resources.ornith9BModelId,
            NorthMiniCodeResources.modelId,
            CodeGenResources.defaultModelId,
        ].filter { modelID in
            guard let spec = ManagedModelCatalog.spec(for: modelID),
                  spec.category == .textCode else {
                return false
            }
            return support(for: spec, on: machine).isSupported
        }
    }

    private static func descriptor(
        _ modelID: String,
        _ title: String,
        _ summary: String,
        minimum: Int,
        recommended: Int,
        setup: Bool = false
    ) -> ManagedModelCapabilityDescriptor {
        ManagedModelCapabilityDescriptor(
            modelID: modelID,
            title: title,
            summary: summary,
            minimumUnifiedMemoryGB: minimum,
            recommendedUnifiedMemoryGB: recommended,
            isRecommendedForSetup: setup
        )
    }
}
