import Foundation

public enum LTXVideoVariant: String, CaseIterable, Codable, Sendable {
    case unifiedAV = "unified-av"
    case distilled
}

public enum LTXVideoQuality: String, CaseIterable, Codable, Sendable {
    case draft
    case final
}

public enum LTXVideoOutputMode: String, CaseIterable, Codable, Sendable {
    case videoOnly = "video-only"
    case audioVideo = "audio-video"

    public var writesAudio: Bool {
        self == .audioVideo
    }

    public var compatibilityVariant: LTXVideoVariant {
        writesAudio ? .unifiedAV : .distilled
    }
}

public enum TextResponseFormat: String, CaseIterable, Codable, Sendable {
    case text
    case jsonObject = "json_object"
}

public enum TextThinkingMode: String, CaseIterable, Codable, Sendable {
    case automatic
    case show
    case hide
}

public enum MereRunCapabilityValueKind: String, Codable, Sendable {
    case string
    case integer
    case number
    case boolean
    case file
    case directory
    case choice
}

public struct MereRunCapabilityArgument: Codable, Equatable, Sendable {
    public let name: String
    public let label: String
    public let kind: MereRunCapabilityValueKind
    public let required: Bool

    public init(name: String, label: String, kind: MereRunCapabilityValueKind, required: Bool) {
        self.name = name
        self.label = label
        self.kind = kind
        self.required = required
    }
}

public struct MereRunCapabilityOption: Codable, Equatable, Sendable {
    public let flag: String
    public let label: String
    public let kind: MereRunCapabilityValueKind
    public let required: Bool
    public let repeatable: Bool
    public let choices: [String]

    public init(
        flag: String,
        label: String,
        kind: MereRunCapabilityValueKind,
        required: Bool = false,
        repeatable: Bool = false,
        choices: [String] = []
    ) {
        self.flag = flag
        self.label = label
        self.kind = kind
        self.required = required
        self.repeatable = repeatable
        self.choices = choices
    }
}

public enum MereRunCapabilityOutputKind: String, Codable, Sendable {
    case text
    case file
    case directory
    case service
}

public struct MereRunCapabilityOutput: Codable, Equatable, Sendable {
    public let kind: MereRunCapabilityOutputKind
    public let fileExtension: String?

    enum CodingKeys: String, CodingKey {
        case kind
        case fileExtension = "file_extension"
    }

    public init(kind: MereRunCapabilityOutputKind, fileExtension: String? = nil) {
        self.kind = kind
        self.fileExtension = fileExtension
    }
}

public struct MereRunCommandCapability: Codable, Equatable, Sendable {
    public let id: String
    public let command: [String]
    public let title: String
    public let summary: String
    public let arguments: [MereRunCapabilityArgument]
    public let options: [MereRunCapabilityOption]
    public let output: MereRunCapabilityOutput

    public init(
        id: String,
        command: [String],
        title: String,
        summary: String,
        arguments: [MereRunCapabilityArgument] = [],
        options: [MereRunCapabilityOption],
        output: MereRunCapabilityOutput
    ) {
        self.id = id
        self.command = command
        self.title = title
        self.summary = summary
        self.arguments = arguments
        self.options = options
        self.output = output
    }
}

public struct MereRunCapabilityDocument: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let commands: [MereRunCommandCapability]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case commands
    }

    public init(schemaVersion: Int, commands: [MereRunCommandCapability]) {
        self.schemaVersion = schemaVersion
        self.commands = commands
    }
}

public enum MereRunCapabilityCatalog {
    public static let schemaVersion = 1

    public static let document = MereRunCapabilityDocument(
        schemaVersion: schemaVersion,
        commands: [
            textChat,
            textCode,
            textEmbed,
            textAnonymize,
            textTrainLoRA,
            videoGenerate,
            videoAnimate,
            videoCosmos3,
            videoPrepareMasks,
            videoExportLatents,
            videoSession
        ]
    )

    public static func command(id: String) -> MereRunCommandCapability? {
        document.commands.first { $0.id == id }
    }

    public static let textChat = MereRunCommandCapability(
        id: "text.chat",
        command: ["text", "chat"],
        title: "Chat",
        summary: "Run local chat, vision, JSON, LoRA, reasoning, and tool workflows.",
        options: [
            .init(flag: "--prompt", label: "Prompt", kind: .string, required: true),
            .init(flag: "--image", label: "Image", kind: .file),
            .init(flag: "--system", label: "System prompt", kind: .string),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--context-size", label: "Context size", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--top-k", label: "Top-k", kind: .integer),
            .init(flag: "--kv-bits", label: "KV bits", kind: .integer),
            .init(
                flag: "--kv-quant-scheme",
                label: "KV quantization",
                kind: .choice,
                choices: ["uniform", "polar", "turboquant"]
            ),
            .init(flag: "--kv-group-size", label: "KV group size", kind: .integer),
            .init(flag: "--quantized-kv-start", label: "Quantized KV start", kind: .integer),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(
                flag: "--response-format",
                label: "Response format",
                kind: .choice,
                choices: TextResponseFormat.allCases.map(\.rawValue)
            ),
            .init(flag: "--lora", label: "LoRA", kind: .file),
            .init(flag: "--lora-scale", label: "LoRA scale", kind: .number),
            .init(flag: "--thinking", label: "Show thinking", kind: .boolean),
            .init(flag: "--no-thinking", label: "Disable thinking", kind: .boolean),
            .init(flag: "--stats", label: "Stats", kind: .boolean),
            .init(flag: "--stream", label: "Stream", kind: .boolean),
            .init(flag: "--tools", label: "Tools", kind: .string),
            .init(flag: "--tool-loop", label: "Tool loop", kind: .boolean),
            .init(flag: "--sandbox-dir", label: "Sandbox directory", kind: .directory),
            .init(flag: "--allow-shell-exec", label: "Allow shell", kind: .boolean),
            .init(flag: "--allow-absolute-tool-paths", label: "Allow absolute paths", kind: .boolean),
            .init(flag: "--auto-approve-tools", label: "Auto-approve tools", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON preflight", kind: .boolean),
            .init(flag: "--require-installed", label: "Require installed", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let textCode = MereRunCommandCapability(
        id: "text.code",
        command: ["text", "code"],
        title: "Code",
        summary: "Run local code generation with GGUF models through llama.cpp.",
        options: [
            .init(flag: "--prompt", label: "Prompt", kind: .string, required: true),
            .init(flag: "--system", label: "System prompt", kind: .string),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--model", label: "Model", kind: .file),
            .init(flag: "--stats", label: "Stats", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean),
            .init(flag: "--stream", label: "Stream", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let textEmbed = MereRunCommandCapability(
        id: "text.embed",
        command: ["text", "embed"],
        title: "Embeddings",
        summary: "Generate native Qwen3 text embeddings.",
        arguments: [
            .init(name: "texts", label: "Texts", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--pretty", label: "Pretty JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "json")
    )

    public static let textAnonymize = MereRunCommandCapability(
        id: "text.anonymize",
        command: ["text", "anonymize"],
        title: "Anonymize",
        summary: "Detect and redact PII with the native OpenAI Privacy Filter.",
        arguments: [
            .init(name: "texts", label: "Texts", kind: .string, required: false)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--replacement", label: "Replacement template", kind: .string),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--pretty", label: "Pretty JSON", kind: .boolean),
            .init(flag: "--output", label: "Output", kind: .file)
        ],
        output: .init(kind: .file)
    )

    public static let textTrainLoRA = MereRunCommandCapability(
        id: "text.train-lora",
        command: ["text", "train-lora"],
        title: "Train text LoRA",
        summary: "Train a native text LoRA from OpenAI-style chat SFT JSONL.",
        options: [
            .init(flag: "--data", label: "Dataset", kind: .file, required: true),
            .init(flag: "--output", label: "Output", kind: .file, required: true),
            .init(flag: "--model", label: "Base model", kind: .string),
            .init(flag: "--model-path", label: "Model path", kind: .directory),
            .init(flag: "--eval", label: "Eval prompts", kind: .file),
            .init(flag: "--adapter-name", label: "Adapter name", kind: .string),
            .init(flag: "--training-steps", label: "Training steps", kind: .integer),
            .init(flag: "--batch-size", label: "Batch size", kind: .integer),
            .init(flag: "--learning-rate", label: "Learning rate", kind: .number),
            .init(flag: "--rank", label: "Rank", kind: .integer),
            .init(flag: "--alpha", label: "Alpha", kind: .number),
            .init(flag: "--max-sequence-length", label: "Sequence length", kind: .integer),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--target-modules", label: "Target modules", kind: .string),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--visualize", label: "Visualize", kind: .boolean),
            .init(flag: "--visualize-port", label: "Visualization port", kind: .integer),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors")
    )

    public static let videoGenerate = MereRunCommandCapability(
        id: "video.generate",
        command: ["video", "generate"],
        title: "Generate video",
        summary: "Generate LTX or Wan video from text, images, keyframes, or source audio.",
        arguments: [
            .init(name: "prompt", label: "Prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(
                flag: "--quality",
                label: "Quality",
                kind: .choice,
                choices: LTXVideoQuality.allCases.map(\.rawValue)
            ),
            .init(
                flag: "--output-mode",
                label: "Output mode",
                kind: .choice,
                choices: LTXVideoOutputMode.allCases.map(\.rawValue)
            ),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--width", label: "Width", kind: .integer),
            .init(flag: "--height", label: "Height", kind: .integer),
            .init(flag: "--num-frames", label: "Frames", kind: .integer),
            .init(flag: "--duration", label: "Duration", kind: .number),
            .init(flag: "--fps", label: "Frames per second", kind: .integer),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--steps", label: "Wan steps", kind: .integer),
            .init(flag: "--guidance-scale", label: "Wan guidance", kind: .number),
            .init(flag: "--shift", label: "Wan schedule shift", kind: .number),
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string),
            .init(flag: "--audio", label: "Source audio", kind: .file),
            .init(flag: "--audio-start-time", label: "Audio start", kind: .number),
            .init(flag: "--a2v-guidance-scale", label: "Audio-to-video guidance", kind: .number),
            .init(flag: "--video-cfg-guidance-scale", label: "Video CFG guidance", kind: .number),
            .init(flag: "--audio-cfg-guidance-scale", label: "Audio CFG guidance", kind: .number),
            .init(flag: "--v2a-guidance-scale", label: "Video-to-audio guidance", kind: .number),
            .init(flag: "--a2v-steps", label: "Audio-to-video steps", kind: .integer),
            .init(flag: "--image", label: "Start image", kind: .file),
            .init(flag: "--image-strength", label: "Start image strength", kind: .number),
            .init(flag: "--end-image", label: "End image", kind: .file),
            .init(flag: "--end-image-strength", label: "End image strength", kind: .number),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--timings", label: "Timings", kind: .boolean),
            .init(flag: "--timings-output", label: "Timings output", kind: .file),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "mp4")
    )

    public static let videoAnimate = MereRunCommandCapability(
        id: "video.animate",
        command: ["video", "animate"],
        title: "Animate subject",
        summary: "Animate or replace masked subjects with native SCAIL-2.",
        arguments: [
            .init(name: "prompt", label: "Prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--reference", label: "Reference image", kind: .file, required: true),
            .init(flag: "--reference-mask", label: "Reference mask", kind: .file, required: true),
            .init(flag: "--driving-video", label: "Driving video", kind: .file, required: true),
            .init(flag: "--driving-mask", label: "Driving mask", kind: .file, required: true),
            .init(flag: "--additional-reference", label: "Additional reference", kind: .file, repeatable: true),
            .init(flag: "--additional-reference-mask", label: "Additional reference mask", kind: .file, repeatable: true),
            .init(flag: "--output", label: "Output", kind: .file, required: true),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--mode", label: "Mode", kind: .choice, choices: ["animation", "replacement"]),
            .init(flag: "--profile", label: "Profile", kind: .choice, choices: ["fast", "quality"]),
            .init(flag: "--width", label: "Width", kind: .integer),
            .init(flag: "--height", label: "Height", kind: .integer),
            .init(flag: "--steps", label: "Steps", kind: .integer),
            .init(flag: "--guidance-scale", label: "Guidance", kind: .number),
            .init(flag: "--shift", label: "Shift", kind: .number),
            .init(flag: "--sampler", label: "Sampler", kind: .choice, choices: ["unipc", "euler"]),
            .init(flag: "--distilled-adapter", label: "Distilled adapter", kind: .file),
            .init(flag: "--distilled-adapter-strength", label: "Adapter strength", kind: .number),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--fps", label: "Frames per second", kind: .integer),
            .init(flag: "--segment-length", label: "Segment length", kind: .integer),
            .init(flag: "--segment-overlap", label: "Segment overlap", kind: .integer),
            .init(flag: "--tail-policy", label: "Tail policy", kind: .choice, choices: ["drop", "pad-trim"]),
            .init(flag: "--audio-source", label: "Audio source", kind: .choice, choices: ["none", "driving"]),
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "mp4")
    )

    public static let videoCosmos3 = MereRunCommandCapability(
        id: "video.cosmos3",
        command: ["video", "cosmos3"],
        title: "Cosmos3",
        summary: "Run Cosmos3-Edge generation, dynamics, policy, and reasoning modes.",
        arguments: [
            .init(name: "prompt", label: "Prompt or action task", kind: .string, required: true)
        ],
        options: [
            .init(
                flag: "--mode",
                label: "Mode",
                kind: .choice,
                choices: [
                    "text-to-image", "image-to-image", "text-to-video", "image-to-video",
                    "video-to-video", "policy", "forward-dynamics", "inverse-dynamics", "reasoner"
                ]
            ),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--actions-output", label: "Actions output", kind: .file),
            .init(flag: "--image", label: "Conditioning image", kind: .file),
            .init(flag: "--video", label: "Conditioning video", kind: .file),
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string),
            .init(flag: "--width", label: "Width", kind: .integer),
            .init(flag: "--height", label: "Height", kind: .integer),
            .init(flag: "--num-frames", label: "Frames", kind: .integer),
            .init(flag: "--steps", label: "Steps", kind: .integer),
            .init(flag: "--guidance-scale", label: "Guidance", kind: .number),
            .init(flag: "--shift", label: "Shift", kind: .number),
            .init(flag: "--schedule", label: "Schedule", kind: .choice, choices: ["nvidia", "published-karras"]),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--fps", label: "Frames per second", kind: .integer),
            .init(flag: "--condition-latent-frame", label: "Conditioned frames", kind: .string),
            .init(flag: "--keep-video-tail", label: "Keep video tail", kind: .boolean),
            .init(flag: "--action-domain", label: "Action domain", kind: .string),
            .init(flag: "--action-file", label: "Action file", kind: .file),
            .init(flag: "--action-chunk-size", label: "Action chunk", kind: .integer),
            .init(flag: "--action-resolution", label: "Action resolution", kind: .integer),
            .init(flag: "--action-viewpoint", label: "Action viewpoint", kind: .choice, choices: ["ego_view", "third_person_view", "wrist_view", "concat_view"]),
            .init(flag: "--max-new-tokens", label: "Reasoner tokens", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--max-video-frames", label: "Reasoner video frames", kind: .integer),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file)
    )

    public static let videoPrepareMasks = MereRunCommandCapability(
        id: "video.prepare-masks",
        command: ["video", "prepare-masks"],
        title: "Prepare SCAIL-2 masks",
        summary: "Create immutable palette-safe SCAIL-2 masks with native SAM 3.1.",
        options: [
            .init(flag: "--plan", label: "Mask plan", kind: .file, required: true),
            .init(flag: "--output-dir", label: "Output directory", kind: .directory, required: true),
            .init(flag: "--preview-frame", label: "Preview frame", kind: .integer),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .directory)
    )

    public static let videoExportLatents = MereRunCommandCapability(
        id: "video.export-latents",
        command: ["video", "export-latents"],
        title: "Export video latents",
        summary: "Run native LTX denoising and export final latents.",
        arguments: [
            .init(name: "prompt", label: "Prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--width", label: "Width", kind: .integer),
            .init(flag: "--height", label: "Height", kind: .integer),
            .init(flag: "--num-frames", label: "Frames", kind: .integer),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors")
    )

    public static let videoSession = MereRunCommandCapability(
        id: "video.session",
        command: ["video", "session"],
        title: "Resident LTX session",
        summary: "Keep an LTX 2.3 runtime resident for JSONL generation requests.",
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .service)
    )
}
