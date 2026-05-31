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
                Q35Resources.q36NanoModelId,
                "Qwen3.6 A3B chat nano",
                "Runs the Qwen3.6 35B-A3B OptiQ 4-bit MLX/MTP snapshot through the native Qwen-family runtime.",
                minimum: 24,
                recommended: 32,
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
                "music-acestep",
                "Music generation",
                "Generates music from text prompts with the ACE-Step model stack.",
                minimum: 24,
                recommended: 32
            ),
            descriptor(
                "video-ltx-av",
                "Video generation",
                "Generates short audio-video clips with the LTX unified AV model stack.",
                minimum: 64,
                recommended: 96
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
