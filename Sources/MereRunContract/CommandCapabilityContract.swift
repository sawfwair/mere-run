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

/// How prominently a shell should surface an option. `essential` options are the
/// two to four controls a shell keeps visible next to the prompt, `standard`
/// options fill the regular inspector sections, and `expert` options collapse
/// under an advanced disclosure.
public enum MereRunCapabilityOptionTier: String, Codable, Sendable {
    case essential
    case standard
    case expert
}

/// A numeric range hint for integer and number options. Every field is optional
/// so a contract entry can declare only a lower bound or only a step.
public struct MereRunCapabilityRange: Codable, Equatable, Sendable {
    public let min: Double?
    public let max: Double?
    public let step: Double?

    public init(min: Double? = nil, max: Double? = nil, step: Double? = nil) {
        self.min = min
        self.max = max
        self.step = step
    }
}

/// Shared group names shells use to section options. Capabilities may use other
/// strings; these are the ones the built-in shells recognize.
public enum MereRunCapabilityOptionGroup {
    public static let prompt = "Prompt"
    public static let inputs = "Inputs"
    public static let output = "Output"
    public static let modelAndAdapters = "Model & adapters"
    public static let sampling = "Sampling"
    public static let run = "Run"
}

public struct MereRunCapabilityOption: Codable, Equatable, Sendable {
    public let flag: String
    public let label: String
    public let kind: MereRunCapabilityValueKind
    public let required: Bool
    public let repeatable: Bool
    public let choices: [String]
    /// The CLI's static ArgumentParser default rendered as the CLI would parse it
    /// (`"1024"`, `"0.7"`, `"peak"`). Absent when the default is machine- or
    /// model-specific.
    public let defaultValue: String?
    /// Section name for shells; see `MereRunCapabilityOptionGroup`.
    public let group: String?
    public let tier: MereRunCapabilityOptionTier?
    public let range: MereRunCapabilityRange?
    /// Flag of another option on the same capability that must be set for this
    /// option to have any effect.
    public let dependsOn: String?

    enum CodingKeys: String, CodingKey {
        case flag
        case label
        case kind
        case required
        case repeatable
        case choices
        case defaultValue = "default_value"
        case group
        case tier
        case range
        case dependsOn = "depends_on"
    }

    public init(
        flag: String,
        label: String,
        kind: MereRunCapabilityValueKind,
        required: Bool = false,
        repeatable: Bool = false,
        choices: [String] = [],
        defaultValue: String? = nil,
        group: String? = nil,
        tier: MereRunCapabilityOptionTier? = nil,
        range: MereRunCapabilityRange? = nil,
        dependsOn: String? = nil
    ) {
        self.flag = flag
        self.label = label
        self.kind = kind
        self.required = required
        self.repeatable = repeatable
        self.choices = choices
        self.defaultValue = defaultValue
        self.group = group
        self.tier = tier
        self.range = range
        self.dependsOn = dependsOn
    }
}

extension MereRunCapabilityOption {
    /// Shared presentation for commands without a bespoke form. Authored metadata always wins.
    fileprivate func withPresentation(outputFlag: String?) -> Self {
        let section: String
        if flag == outputFlag || ["--output", "--json-output", "--mask-output-dir", "--structured-prompt-output"].contains(flag) {
            section = MereRunCapabilityOptionGroup.output
        } else if ["--prompt", "--text", "--query", "--system", "--system-prompt", "--negative-prompt", "--lyrics"].contains(flag) {
            section = MereRunCapabilityOptionGroup.prompt
        } else if ["--model", "--model-root", "--lora", "--lora-scale", "--adapter"].contains(flag) {
            section = MereRunCapabilityOptionGroup.modelAndAdapters
        } else if [.file, .directory].contains(kind) {
            section = MereRunCapabilityOptionGroup.inputs
        } else if ["--seed", "--steps", "--cfg", "--temperature", "--top-p", "--top-k", "--max-tokens", "--width", "--height"].contains(flag) {
            section = MereRunCapabilityOptionGroup.sampling
        } else {
            section = MereRunCapabilityOptionGroup.run
        }
        return Self(flag: flag, label: label, kind: kind, required: required, repeatable: repeatable,
            choices: choices, defaultValue: defaultValue, group: group ?? section,
            tier: tier ?? (required ? .essential : .standard), range: range, dependsOn: dependsOn)
    }
}

/// What a successful run leaves behind when the caller passes no destination.
/// `text` prints its result to stdout, `service` runs until it is stopped, and
/// `file` and `directory` always write the artifact, at a default path when the
/// caller names none.
public enum MereRunCapabilityOutputKind: String, Codable, Sendable {
    case text
    case file
    case directory
    case service
}

/// The artifact one run produces, and how the caller asks for it.
///
/// A `text` or `service` capability that still declares a `flag` writes that
/// artifact only when the flag is passed (`optional` is then `true`); `kind`
/// describes what the run does without it. Look `flag` up in the capability's
/// `options` to learn whether it names a file or a directory.
public struct MereRunCapabilityOutput: Codable, Equatable, Sendable {
    public let kind: MereRunCapabilityOutputKind
    /// The written artifact's extension when it is a file, and the command
    /// always writes the same one. Absent for directories and for commands
    /// whose extension follows another option (`speech diarize --format`).
    public let fileExtension: String?
    /// The option whose value names the destination. Absent when the run writes
    /// nothing, or chooses the location itself (`adapter pull`, `image run-plan`).
    public let flag: String?
    /// True when the artifact is written only if `flag` is passed.
    public let optional: Bool

    enum CodingKeys: String, CodingKey {
        case kind
        case fileExtension = "file_extension"
        case flag
        case optional
    }

    public init(
        kind: MereRunCapabilityOutputKind,
        fileExtension: String? = nil,
        flag: String? = nil,
        optional: Bool = false
    ) {
        self.kind = kind
        self.fileExtension = fileExtension
        self.flag = flag
        self.optional = optional
    }

    /// `flag` and `optional` are additive: a document written before they
    /// existed decodes with no destination flag and a mandatory artifact.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        kind = try container.decode(MereRunCapabilityOutputKind.self, forKey: .kind)
        fileExtension = try container.decodeIfPresent(String.self, forKey: .fileExtension)
        flag = try container.decodeIfPresent(String.self, forKey: .flag)
        optional = try container.decodeIfPresent(Bool.self, forKey: .optional) ?? false
    }

    /// Absent fields stay absent so a decoder that predates them sees the same
    /// JSON it always did.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(kind, forKey: .kind)
        try container.encodeIfPresent(fileExtension, forKey: .fileExtension)
        try container.encodeIfPresent(flag, forKey: .flag)
        if optional { try container.encode(true, forKey: .optional) }
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
        self.options = options.map { $0.withPresentation(outputFlag: output.flag) }
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

private typealias Group = MereRunCapabilityOptionGroup

/// The machine-readable flags shared by every long-running generation command.
/// `--receipt` prints the final `{"event":"result",...}` line; `--progress-json`
/// streams `{"event":"progress",...}` lines on stderr.
private let receiptOption = MereRunCapabilityOption(
    flag: "--receipt",
    label: "Result receipt",
    kind: .boolean,
    group: Group.run,
    tier: .expert
)

private let progressJSONOption = MereRunCapabilityOption(
    flag: "--progress-json",
    label: "Progress JSON",
    kind: .boolean,
    group: Group.run,
    tier: .expert
)

public enum MereRunCapabilityCatalog {
    public static let schemaVersion = 1

    /// The `--progress-json` stderr event shape, one JSON object per line.
    /// `step` is 0-based while a stage is in progress; every determinate stage
    /// ends with exactly one event whose `step == total_steps`, written when
    /// the stage completes or at the end of the run at the latest.
    /// `total_steps == 0` marks an indeterminate stage (token streaming with no
    /// known length) that carries no terminal event.
    public static let progressEventExample =
        #"{"event":"progress","stage":"denoising","step":2,"total_steps":4}"#

    /// The `--receipt` final stdout line. The first output is the primary
    /// artifact; sidecars follow with a `role`. It is printed only after a
    /// successful run, so `exit` is always `0`; `--receipt` is rejected
    /// together with `--preflight`, which produces no result.
    public static let resultReceiptExample =
        #"{"event":"result","exit":0,"outputs":[{"kind":"image","path":"/abs/out.png"}]}"#

    /// Capability ids whose commands print the `--receipt` line.
    public static let receiptCapabilityIDs: [String] = [
        "image.generate", "video.generate", "music.generate", "sfx.generate",
        "speech.synthesize", "speech.transcribe",
        "vision.ground", "vision.segment", "vision.track"
    ]

    /// Capability ids whose commands stream `--progress-json` events.
    public static let progressJSONCapabilityIDs: [String] = [
        "image.generate", "video.generate", "music.generate", "sfx.generate", "speech.synthesize"
    ]

    public static let document = MereRunCapabilityDocument(
        schemaVersion: schemaVersion,
        commands: [
            textChat,
            textCode,
            textEmbed,
            textAnonymize,
            textTrainLoRA,
            imageGenerate,
            imageTrainLoRA,
            imageValidate,
            imageDatasetDiscover,
            imageRunPlan,
            imageVisualizeRun,
            imageReconstruct3D,
            imageReconstruct3DTrellis2,
            imageReconstruct3DMultiview,
            visionEmbed,
            visionInspect,
            visionCaption,
            visionOCR,
            visionGround,
            visionSegment,
            visionTrack,
            visionTrackLive,
            visionFaceDetect,
            visionFaceEmbed,
            visionFaceCompare,
            visionFaceBatch,
            visionPose,
            visionFlow,
            visionDepthVideo,
            visionGeometry,
            visionGeometryMultiview,
            audioEnhance,
            audioGenerate,
            musicGenerate,
            musicAnalyze,
            musicTranscribe,
            musicSeparate,
            musicRealtime,
            musicTrainAdapter,
            musicServe,
            videoGenerate,
            videoRetake,
            videoDubIt,
            videoAnimate,
            videoCosmos3,
            videoPrepareMasks,
            videoExportLatents,
            videoSession,
            adapterList,
            adapterPull,
            runList,
            runInspect,
            runWatch,
            runFetch,
            runCancel,
            runRetry,
            evaluationPackValidate,
            evaluationRun,
            evaluationPromote,
            worldServe,
            visionServe,
            status,
            gate,
            modelStorage,
            modelLocationList,
            modelLocationAdd,
            modelLocationRemove,
            modelLocationBind,
            modelLocationUnbind,
            modelGarbageCollect,
            modelRuntimeGet,
            modelRuntimeSet,
            setup,
            agentOnboard,
            agentStatus,
            agentInstallPi,
            agentStart,
            modelList,
            modelCapabilities,
            modelPull,
            modelInfo,
            modelRemove,
            modelRepairManifests,
            modelOptimize,
            modelBenchmarkQ36MTP,
            modelBenchmarkLagunaDFlash,
            modelBenchmarkParakeetCoreML,
            modelBenchmarkChat,
            modelBenchmarkCode,
            modelBenchmarkFused,
            modelBenchmarkFusedFixture,
            modelBenchmarkVLM,
            modelBenchmarkToolCalls,
            modelBenchmarkToolContinuations,
            modelBenchmarkGemma4KV,
            modelBenchmarkGemma4MTP,
            modelBenchmarkAPIWorkload,
            speechSynthesize,
            speechTranscribe,
            speechDiarize,
            speechProfileList,
            speechProfileCreate,
            speechProfileDelete,
            speechListen,
            sfxGenerate,
            sfxVideoGenerate,
            sfxAEEncode,
            sfxAEDecode,
            sfxCLAPScore,
            sfxConditionText,
            pluginList,
            pluginInfo,
            pluginInstall,
            pluginDoctor,
            pluginRun,
            pluginRollback,
            openWebUIQuickstart,
            apiServe,
            guide,
            configSet,
            configGet,
            configUnset,
            configList,
            configPath,
            geoFlood,
            geoFire,
            geoTessera,
            geoOlmoEarth
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
            .init(flag: "--prompt", label: "Prompt", kind: .string, required: true, group: Group.prompt, tier: .essential),
            .init(flag: "--image", label: "Image", kind: .file, group: Group.inputs, tier: .standard),
            .init(flag: "--system", label: "System prompt", kind: .string, group: Group.prompt, tier: .standard),
            .init(
                flag: "--max-tokens", label: "Max tokens", kind: .integer,
                defaultValue: "2048", group: Group.sampling, tier: .standard,
                range: .init(min: 1, max: 131_072, step: 1)
            ),
            .init(
                flag: "--context-size", label: "Context size", kind: .integer,
                group: Group.sampling, tier: .expert, range: .init(min: 512, max: 1_048_576, step: 1)
            ),
            .init(
                flag: "--temperature", label: "Temperature", kind: .number,
                group: Group.sampling, tier: .standard, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(
                flag: "--top-p", label: "Top-p", kind: .number,
                group: Group.sampling, tier: .standard, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--top-k", label: "Top-k", kind: .integer,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1_000, step: 1)
            ),
            .init(
                flag: "--min-p", label: "Min-p", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--kv-bits", label: "KV bits", kind: .integer,
                group: Group.run, tier: .expert, range: .init(min: 2, max: 8, step: 1)
            ),
            .init(
                flag: "--kv-quant-scheme",
                label: "KV quantization",
                kind: .choice,
                choices: ["uniform", "polar", "turboquant"],
                group: Group.run, tier: .expert, dependsOn: "--kv-bits"
            ),
            .init(
                flag: "--kv-group-size", label: "KV group size", kind: .integer,
                group: Group.run, tier: .expert, dependsOn: "--kv-bits"
            ),
            .init(
                flag: "--quantized-kv-start", label: "Quantized KV start", kind: .integer,
                group: Group.run, tier: .expert, range: .init(min: 0, step: 1), dependsOn: "--kv-bits"
            ),
            .init(flag: "--model-root", label: "Model root", kind: .directory, group: Group.modelAndAdapters, tier: .expert),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(
                flag: "--response-format",
                label: "Response format",
                kind: .choice,
                choices: TextResponseFormat.allCases.map(\.rawValue),
                defaultValue: TextResponseFormat.text.rawValue, group: Group.output, tier: .standard
            ),
            .init(flag: "--lora", label: "LoRA", kind: .file, group: Group.modelAndAdapters, tier: .standard),
            .init(
                flag: "--lora-scale", label: "LoRA scale", kind: .number,
                defaultValue: "1.0", group: Group.modelAndAdapters, tier: .standard,
                range: .init(min: 0, max: 2, step: 0.05), dependsOn: "--lora"
            ),
            .init(flag: "--thinking", label: "Show thinking", kind: .boolean, group: Group.sampling, tier: .standard),
            .init(flag: "--no-thinking", label: "Disable thinking", kind: .boolean, group: Group.sampling, tier: .standard),
            .init(
                flag: "--reasoning-effort", label: "Inkling reasoning effort", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(flag: "--stats", label: "Stats", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--stream", label: "Stream", kind: .boolean, group: Group.output, tier: .standard),
            .init(flag: "--tools", label: "Tools", kind: .string, group: Group.run, tier: .expert),
            .init(flag: "--tool-loop", label: "Tool loop", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--tools"),
            .init(
                flag: "--sandbox-dir", label: "Sandbox directory", kind: .directory,
                group: Group.run, tier: .expert, dependsOn: "--tools"
            ),
            .init(
                flag: "--allow-shell-exec", label: "Allow shell", kind: .boolean,
                group: Group.run, tier: .expert, dependsOn: "--tools"
            ),
            .init(
                flag: "--allow-absolute-tool-paths", label: "Allow absolute paths", kind: .boolean,
                group: Group.run, tier: .expert, dependsOn: "--tools"
            ),
            .init(
                flag: "--auto-approve-tools", label: "Auto-approve tools", kind: .boolean,
                group: Group.run, tier: .expert, dependsOn: "--tools"
            ),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--json", label: "JSON preflight", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--preflight"),
            .init(flag: "--require-installed", label: "Require installed", kind: .boolean, group: Group.run, tier: .expert)
        ],
        output: .init(kind: .text)
    )

    public static let textCode = MereRunCommandCapability(
        id: "text.code",
        command: ["text", "code"],
        title: "Code",
        summary: "Run local code generation with GGUF models through llama.cpp.",
        options: [
            .init(flag: "--prompt", label: "Prompt", kind: .string, required: true, group: Group.prompt, tier: .essential),
            .init(
                flag: "--system", label: "System prompt", kind: .string,
                defaultValue: "You are a helpful coding assistant.", group: Group.prompt, tier: .standard
            ),
            .init(
                flag: "--max-tokens", label: "Max tokens", kind: .integer,
                defaultValue: "2048", group: Group.sampling, tier: .standard,
                range: .init(min: 1, max: 131_072, step: 1)
            ),
            .init(
                flag: "--temperature", label: "Temperature", kind: .number,
                defaultValue: "1.0", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(
                flag: "--top-p", label: "Top-p", kind: .number,
                defaultValue: "0.95", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--min-p", label: "Min-p", kind: .number,
                defaultValue: "0.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(flag: "--model", label: "Model", kind: .file, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--stats", label: "Stats", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--stream", label: "Stream", kind: .boolean, group: Group.output, tier: .standard)
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
        output: .init(kind: .text, fileExtension: "json", flag: "--output", optional: true)
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
        output: .init(kind: .text, flag: "--output", optional: true)
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
            .init(flag: "--reasoning-effort", label: "Inkling reasoning effort", kind: .number),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--target-modules", label: "Target modules", kind: .string),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--visualize", label: "Visualize", kind: .boolean),
            .init(flag: "--visualize-port", label: "Visualization port", kind: .integer),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )

    public static let imageGenerate = MereRunCommandCapability(
        id: "image.generate",
        command: ["image", "generate"],
        title: "Generate and edit images",
        summary: "Generate, transform, or personalize images with references, structured prompts, and LoRAs.",
        options: [
            .init(flag: "--prompt", label: "Prompt", kind: .string, required: true, group: Group.prompt, tier: .essential),
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string, group: Group.prompt, tier: .standard),
            .init(
                flag: "--cfg", label: "CFG scale", kind: .number,
                group: Group.sampling, tier: .standard, range: .init(min: 0, max: 20, step: 0.5)
            ),
            .init(
                flag: "--sigma-shift", label: "Sigma shift", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 16, step: 0.1)
            ),
            .init(flag: "--output", label: "Output", kind: .file, group: Group.output, tier: .standard),
            .init(
                flag: "--width", label: "Width", kind: .integer,
                defaultValue: "1024", group: Group.output, tier: .essential, range: .init(min: 256, max: 2_048, step: 16)
            ),
            .init(
                flag: "--height", label: "Height", kind: .integer,
                defaultValue: "1024", group: Group.output, tier: .essential, range: .init(min: 256, max: 2_048, step: 16)
            ),
            .init(
                flag: "--steps", label: "Steps", kind: .integer,
                group: Group.sampling, tier: .essential, range: .init(min: 1, max: 100, step: 1)
            ),
            .init(flag: "--seed", label: "Seed", kind: .integer, group: Group.sampling, tier: .essential, range: .init(min: 0, step: 1)),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--input", label: "Input image", kind: .file, group: Group.inputs, tier: .standard),
            .init(flag: "--mask", label: "Edit mask", kind: .file, group: Group.inputs, tier: .standard, dependsOn: "--input"),
            .init(
                flag: "--outpaint", label: "Outpaint padding", kind: .string,
                group: Group.inputs, tier: .expert, dependsOn: "--input"
            ),
            .init(
                flag: "--mask-feather", label: "Mask feather", kind: .integer,
                defaultValue: "8", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 128, step: 1), dependsOn: "--input"
            ),
            .init(
                flag: "--ref-image", label: "Reference image", kind: .file, repeatable: true,
                group: Group.inputs, tier: .standard
            ),
            .init(
                flag: "--keep-original-aspect", label: "Keep original aspect", kind: .boolean,
                group: Group.inputs, tier: .expert, dependsOn: "--ref-image"
            ),
            .init(
                flag: "--strength", label: "Edit strength", kind: .number,
                group: Group.inputs, tier: .standard, range: .init(min: 0, max: 1, step: 0.05)
            ),
            .init(
                flag: "--max-sequence-length", label: "Max sequence length", kind: .integer,
                defaultValue: "512", group: Group.sampling, tier: .expert, range: .init(min: 64, max: 4_096, step: 64)
            ),
            .init(flag: "--structured-prompt", label: "Structured prompt", kind: .boolean, group: Group.prompt, tier: .standard),
            .init(
                flag: "--structured-prompt-model", label: "Prompt model", kind: .string,
                defaultValue: "text-chat-gemma4-12b-4bit", group: Group.prompt, tier: .expert, dependsOn: "--structured-prompt"
            ),
            .init(
                flag: "--structured-prompt-model-root", label: "Prompt model root", kind: .directory,
                group: Group.prompt, tier: .expert, dependsOn: "--structured-prompt"
            ),
            .init(
                flag: "--structured-prompt-max-tokens", label: "Prompt max tokens", kind: .integer,
                defaultValue: "2048", group: Group.prompt, tier: .expert,
                range: .init(min: 1, max: 8_192, step: 1), dependsOn: "--structured-prompt"
            ),
            .init(
                flag: "--structured-prompt-output", label: "Structured prompt output", kind: .file,
                group: Group.prompt, tier: .expert, dependsOn: "--structured-prompt"
            ),
            .init(flag: "--lora", label: "LoRA", kind: .file, group: Group.modelAndAdapters, tier: .standard),
            .init(
                flag: "--lora-scale", label: "LoRA scale", kind: .number,
                defaultValue: "1.0", group: Group.modelAndAdapters, tier: .standard,
                range: .init(min: 0, max: 2, step: 0.05), dependsOn: "--lora"
            ),
            .init(
                flag: "--krea-conditioning-multiplier", label: "Krea conditioning", kind: .number,
                group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--krea-conditioning-layer-weights", label: "Krea layer weights", kind: .string,
                group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--krea-base-quantization-bits",
                label: "Krea base quantization",
                kind: .choice,
                choices: ["4", "8"],
                group: Group.modelAndAdapters, tier: .expert
            ),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--json", label: "JSON", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--preflight"),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            progressJSONOption,
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "png", flag: "--output")
    )

    public static let imageTrainLoRA = MereRunCommandCapability(
        id: "image.train-lora",
        command: ["image", "train-lora"],
        title: "Train image LoRA",
        summary: "Train Krea 2 or FLUX.2 Klein adapters with recipes, previews, checkpoints, and dashboards.",
        options: [
            .init(flag: "--data", label: "Dataset", kind: .directory),
            .init(flag: "--output", label: "Output", kind: .file, required: true),
            .init(flag: "--model", label: "Base model", kind: .string),
            .init(flag: "--width", label: "Width", kind: .integer),
            .init(flag: "--height", label: "Height", kind: .integer),
            .init(flag: "--training-steps", label: "Training steps", kind: .integer),
            .init(flag: "--batch-size", label: "Batch size", kind: .integer),
            .init(flag: "--learning-rate", label: "Learning rate", kind: .number),
            .init(flag: "--rank", label: "Rank", kind: .integer),
            .init(flag: "--alpha", label: "Alpha", kind: .number),
            .init(flag: "--max-text-length", label: "Max text length", kind: .integer),
            .init(flag: "--scheduler-steps", label: "Scheduler steps", kind: .integer),
            .init(flag: "--caption-dropout", label: "Caption dropout", kind: .number),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--lite", label: "Lite targets", kind: .boolean),
            .init(flag: "--base-quantization-bits", label: "Base quantization", kind: .choice, choices: ["4", "8"]),
            .init(flag: "--exclude-preview-images", label: "Exclude preview images", kind: .boolean),
            .init(flag: "--checkpoint-interval", label: "Checkpoint interval", kind: .integer),
            .init(flag: "--resume-from", label: "Resume checkpoint", kind: .file),
            .init(flag: "--max-resolution", label: "Max resolution", kind: .integer),
            .init(flag: "--progressive", label: "Progressive resolution", kind: .boolean),
            .init(flag: "--low-ram", label: "Low RAM", kind: .boolean),
            .init(flag: "--no-compile", label: "Disable compile", kind: .boolean),
            .init(flag: "--gradient-checkpointing", label: "Gradient checkpointing", kind: .boolean),
            .init(
                flag: "--recipe",
                label: "Recipe",
                kind: .choice,
                choices: ["krea-fast-style", "krea-cinematic-style", "klein-fast-style"]
            ),
            .init(flag: "--benchmark-steps", label: "Benchmark steps", kind: .integer),
            .init(flag: "--benchmark-warmup-steps", label: "Benchmark warmup", kind: .integer),
            .init(flag: "--sample-interval", label: "Sample interval", kind: .integer),
            .init(flag: "--sample-prompt", label: "Sample prompt", kind: .string),
            .init(flag: "--sample-model", label: "Sample model", kind: .string),
            .init(flag: "--sample-steps", label: "Sample steps", kind: .integer),
            .init(flag: "--sample-cfg", label: "Sample CFG", kind: .number),
            .init(flag: "--sample-lora-scale", label: "Sample LoRA scale", kind: .number),
            .init(flag: "--sample-seed", label: "Sample seed", kind: .integer),
            .init(flag: "--visualize", label: "Visualize", kind: .boolean),
            .init(flag: "--visualize-port", label: "Visualization port", kind: .integer),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--lora-target-ranks", label: "Target ranks", kind: .string),
            .init(flag: "--lora-rank-preset", label: "Rank preset", kind: .choice, choices: ["flux2-style-128"]),
            .init(flag: "--lora-target-preset", label: "Target preset", kind: .choice, choices: ["fal-klein-fast"]),
            .init(
                flag: "--lora-target-mode",
                label: "Target mode",
                kind: .choice,
                choices: ["suffix", "transformer-linear-walk"]
            ),
            .init(
                flag: "--timestep-sampling",
                label: "Timestep sampling",
                kind: .choice,
                choices: ["uniform", "bellCurve", "contentFocused", "styleFocused", "logitNormal", "shift"]
            ),
            .init(
                flag: "--timestep-loss-weighting",
                label: "Timestep weighting",
                kind: .choice,
                choices: ["none", "weighted"]
            ),
            .init(flag: "--loss-weighting", label: "Loss weighting", kind: .choice, choices: ["none", "snr", "minSNR"]),
            .init(flag: "--timestep-low", label: "Timestep low", kind: .integer),
            .init(flag: "--timestep-high", label: "Timestep high", kind: .integer),
            .init(flag: "--lr-warmup-steps", label: "LR warmup", kind: .integer),
            .init(flag: "--no-cosine-scheduler", label: "Disable cosine scheduler", kind: .boolean),
            .init(flag: "--lr-min-factor", label: "LR minimum factor", kind: .number),
            .init(flag: "--adam-weight-decay", label: "Adam weight decay", kind: .number),
            .init(flag: "--synthetic-samples", label: "Synthetic samples", kind: .integer),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )

    public static let imageValidate = MereRunCommandCapability(
        id: "image.validate",
        command: ["image", "validate"],
        title: "Validate image runtime",
        summary: "Run deterministic VAE, encoder, transformer, and pipeline checks.",
        options: [
            .init(
                flag: "--test",
                label: "Suite",
                kind: .choice,
                choices: ["vae", "encoder", "transformer", "pipeline", "all"]
            ),
            .init(flag: "--family", label: "Family", kind: .choice, choices: ["zimage", "klein"]),
            .init(flag: "--output", label: "Output", kind: .directory),
            .init(flag: "--save-reference", label: "Save reference", kind: .boolean),
            .init(flag: "--compare", label: "Compare", kind: .boolean),
            .init(flag: "--reference-dir", label: "Reference directory", kind: .directory)
        ],
        output: .init(kind: .directory, flag: "--output")
    )

    public static let imageDatasetDiscover = MereRunCommandCapability(
        id: "image.dataset.discover",
        command: ["image", "dataset", "discover"],
        title: "Discover image datasets",
        summary: "Find trainable image-caption dataset leaves and produce preflight commands.",
        options: [
            .init(flag: "--root", label: "Root", kind: .directory, required: true),
            .init(flag: "--max-depth", label: "Max depth", kind: .integer),
            .init(flag: "--min-usable-pairs", label: "Minimum pairs", kind: .integer),
            .init(flag: "--training-output-root", label: "Training output root", kind: .directory),
            .init(flag: "--training-model", label: "Training model", kind: .string),
            .init(flag: "--training-recipe", label: "Training recipe", kind: .string),
            .init(flag: "--exclude-preview-images", label: "Exclude preview images", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let imageRunPlan = MereRunCommandCapability(
        id: "image.run-plan",
        command: ["image", "run-plan"],
        title: "Run image plan",
        summary: "Preflight, materialize, or execute a saved image workflow plan.",
        arguments: [
            .init(name: "file", label: "Plan", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--materialize", label: "Materialize run", kind: .directory)
        ],
        output: .init(kind: .file)
    )

    public static let imageVisualizeRun = MereRunCommandCapability(
        id: "image.visualize-run",
        command: ["image", "visualize-run"],
        title: "Visualize image run",
        summary: "Open the loopback dashboard for a durable image LoRA run.",
        arguments: [
            .init(name: "run-directory", label: "Run directory", kind: .directory, required: true)
        ],
        options: [
            .init(flag: "--port", label: "Port", kind: .integer)
        ],
        output: .init(kind: .service)
    )

    public static let imageReconstruct3D = MereRunCommandCapability(
        id: "image.reconstruct-3d",
        command: ["image", "reconstruct-3d"],
        title: "TripoSR reconstruction",
        summary: "Reconstruct a colored object mesh from a single image with native TripoSR.",
        arguments: [
            .init(name: "input", label: "Input image", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Output", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--resolution", label: "Resolution", kind: .integer),
            .init(flag: "--density-threshold", label: "Density threshold", kind: .number),
            .init(flag: "--foreground-ratio", label: "Foreground ratio", kind: .number),
            .init(flag: "--already-framed", label: "Already framed", kind: .boolean),
            .init(flag: "--no-vertex-colors", label: "Geometry only", kind: .boolean),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output")
    )

    public static let imageReconstruct3DTrellis2 = MereRunCommandCapability(
        id: "image.reconstruct-3d-trellis2",
        command: ["image", "reconstruct-3d-trellis2"],
        title: "TRELLIS.2 reconstruction",
        summary: "Reconstruct a native 512-resolution PBR O-Voxel asset from one image.",
        arguments: [
            .init(name: "input", label: "Input image", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Output", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--texture-seed", label: "Texture seed", kind: .integer),
            .init(flag: "--max-tokens", label: "Maximum sparse tokens", kind: .integer),
            .init(flag: "--already-framed", label: "Already framed", kind: .boolean),
            .init(flag: "--no-remesh", label: "Skip remeshing", kind: .boolean),
            .init(flag: "--remesh-band", label: "Remesh band", kind: .number),
            .init(flag: "--seal-radius", label: "Seal radius", kind: .integer),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output")
    )

    public static let imageReconstruct3DMultiview = MereRunCommandCapability(
        id: "image.reconstruct-3d-multiview",
        command: ["image", "reconstruct-3d-multiview"],
        title: "InstantMesh multiview reconstruction",
        summary: "Reconstruct a colored mesh from four or six ordered source views.",
        options: [
            .init(flag: "--view", label: "View", kind: .file, repeatable: true),
            .init(flag: "--output", label: "Output", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--cameras", label: "Cameras", kind: .file),
            .init(flag: "--resolution", label: "Resolution", kind: .integer),
            .init(flag: "--no-vertex-colors", label: "Geometry only", kind: .boolean),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output")
    )

    public static let visionInspect = MereRunCommandCapability(
        id: "vision.inspect",
        command: ["vision", "inspect"],
        title: "Inspect image",
        summary: "Describe or answer questions about an image with a local VLM.",
        arguments: [.init(name: "image", label: "Image", kind: .file, required: true)],
        options: [
            .init(flag: "--prompt", label: "Prompt", kind: .string, group: Group.prompt, tier: .essential),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(
                flag: "--max-tokens", label: "Max tokens", kind: .integer,
                defaultValue: "2048", group: Group.sampling, tier: .standard, range: .init(min: 1, max: 8_192, step: 1)
            ),
            .init(
                flag: "--temperature", label: "Temperature", kind: .number,
                defaultValue: "0.7", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(
                flag: "--top-p", label: "Top-p", kind: .number,
                defaultValue: "0.9", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 1, step: 0.01)
            )
        ],
        output: .init(kind: .text)
    )

    public static let visionEmbed = MereRunCommandCapability(
        id: "vision.embed",
        command: ["vision", "embed"],
        title: "Multimodal embeddings",
        summary: "Generate shared text and image embeddings with native Qwen3-VL.",
        options: [
            .init(flag: "--text", label: "Text inputs", kind: .string, repeatable: true),
            .init(flag: "--image", label: "Image inputs", kind: .file, repeatable: true),
            .init(flag: "--input-json", label: "JSON batch", kind: .file),
            .init(flag: "--instruction", label: "Retrieval instruction", kind: .string),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--dimensions", label: "Dimensions", kind: .integer),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--min-pixels", label: "Minimum pixels", kind: .integer),
            .init(flag: "--max-pixels", label: "Maximum pixels", kind: .integer),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--pretty", label: "Pretty JSON", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--output", optional: true)
    )

    public static let visionCaption = MereRunCommandCapability(
        id: "vision.caption",
        command: ["vision", "caption"],
        title: "Caption images",
        summary: "Generate training-friendly captions for one or more images.",
        arguments: [.init(name: "images", label: "Images", kind: .file, required: true)],
        options: [
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--output-dir", label: "Output directory", kind: .directory, group: Group.output, tier: .standard),
            .init(flag: "--prompt", label: "Prompt", kind: .string, group: Group.prompt, tier: .essential),
            .init(flag: "--prompt-file", label: "Prompt file", kind: .file, group: Group.prompt, tier: .expert),
            .init(flag: "--focus", label: "Focus", kind: .string, repeatable: true, group: Group.prompt, tier: .standard),
            .init(flag: "--trigger-token", label: "Trigger token", kind: .string, group: Group.prompt, tier: .standard),
            .init(
                flag: "--max-tokens", label: "Max tokens", kind: .integer,
                defaultValue: "96", group: Group.sampling, tier: .standard, range: .init(min: 1, max: 2_048, step: 1)
            ),
            .init(
                flag: "--temperature", label: "Temperature", kind: .number,
                defaultValue: "0.2", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(
                flag: "--top-p", label: "Top-p", kind: .number,
                defaultValue: "0.9", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 1, step: 0.01)
            )
        ],
        output: .init(kind: .directory, flag: "--output-dir")
    )

    public static let visionOCR = MereRunCommandCapability(
        id: "vision.ocr",
        command: ["vision", "ocr"],
        title: "OCR",
        summary: "Extract text with native LightOn/Infinity or external GLM/Infinity runtimes.",
        arguments: [.init(name: "images", label: "Images", kind: .file, required: true)],
        options: [
            .init(
                flag: "--backend", label: "Backend", kind: .choice, choices: ["lighton", "glm", "infinity"],
                defaultValue: "lighton", group: Group.modelAndAdapters, tier: .essential
            ),
            .init(flag: "--compare", label: "Compare", kind: .boolean, group: Group.run, tier: .expert),
            .init(
                flag: "--model", label: "LightOn model", kind: .string,
                defaultValue: "vision-ocr-lighton", group: Group.modelAndAdapters, tier: .standard
            ),
            .init(
                flag: "--glmocr-cli", label: "GLM executable", kind: .file,
                defaultValue: "glmocr", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(flag: "--glm-config", label: "GLM config", kind: .file, group: Group.modelAndAdapters, tier: .expert),
            .init(
                flag: "--infinity-runtime", label: "Infinity runtime", kind: .choice, choices: ["native", "external"],
                defaultValue: "native", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--infinity-parser-cli", label: "Parser executable", kind: .file,
                defaultValue: "parser", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--infinity-model", label: "Infinity model", kind: .string,
                defaultValue: "vision-ocr-infinity-pro-int8", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--infinity-backend",
                label: "Infinity backend",
                kind: .choice,
                choices: ["transformers", "vllm-engine", "vllm-server"],
                defaultValue: "vllm-server", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--infinity-api-url", label: "Infinity API URL", kind: .string,
                defaultValue: "http://localhost:8000/v1/chat/completions", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--infinity-api-key", label: "Infinity API key", kind: .string,
                defaultValue: "EMPTY", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--infinity-task", label: "Infinity task", kind: .choice, choices: ["doc2json", "doc2md", "custom"],
                defaultValue: "doc2json", group: Group.prompt, tier: .expert
            ),
            .init(flag: "--infinity-prompt", label: "Infinity prompt", kind: .string, group: Group.prompt, tier: .expert),
            .init(
                flag: "--infinity-output-format", label: "Infinity format", kind: .choice, choices: ["md", "json"],
                defaultValue: "md", group: Group.output, tier: .expert
            ),
            .init(
                flag: "--infinity-batch-size", label: "Batch size", kind: .integer,
                defaultValue: "1", group: Group.run, tier: .expert, range: .init(min: 1, max: 64, step: 1)
            ),
            .init(
                flag: "--infinity-model-cache-dir", label: "Model cache", kind: .directory,
                group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--infinity-min-pixels", label: "Minimum pixels", kind: .integer,
                defaultValue: "2048", group: Group.inputs, tier: .expert, range: .init(min: 1, step: 1)
            ),
            .init(
                flag: "--infinity-max-pixels", label: "Maximum pixels", kind: .integer,
                defaultValue: "16777216", group: Group.inputs, tier: .expert, range: .init(min: 1, step: 1)
            ),
            .init(flag: "--output-dir", label: "Output directory", kind: .directory, group: Group.output, tier: .standard),
            .init(
                flag: "--max-tokens", label: "Max tokens", kind: .integer,
                defaultValue: "4096", group: Group.sampling, tier: .standard, range: .init(min: 1, max: 16_384, step: 1)
            ),
            .init(
                flag: "--temperature", label: "Temperature", kind: .number,
                defaultValue: "0.2", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert)
        ],
        output: .init(kind: .text, flag: "--output-dir", optional: true)
    )

    public static let visionGround = MereRunCommandCapability(
        id: "vision.ground",
        command: ["vision", "ground"],
        title: "Ground objects",
        summary: "Ground one or more text expressions with native Falcon Perception.",
        arguments: [.init(name: "image", label: "Image", kind: .file, required: true)],
        options: [
            .init(flag: "--query", label: "Query", kind: .string, repeatable: true, group: Group.prompt, tier: .essential),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--output", label: "Annotated image", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--json-output", label: "JSON output", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--mask-output-dir", label: "Mask directory", kind: .directory, group: Group.output, tier: .standard),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--json", label: "JSON preflight", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--preflight"),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "png", flag: "--output")
    )

    public static let visionSegment = MereRunCommandCapability(
        id: "vision.segment",
        command: ["vision", "segment"],
        title: "Segment objects",
        summary: "Segment text, box, or point prompts with native SAM 3.1.",
        arguments: [.init(name: "image", label: "Image", kind: .file, required: true)],
        options: [
            .init(flag: "--prompt", label: "Text prompt", kind: .string, repeatable: true, group: Group.prompt, tier: .essential),
            .init(flag: "--box", label: "Box prompt", kind: .string, repeatable: true, group: Group.inputs, tier: .standard),
            .init(flag: "--point", label: "Point prompt", kind: .string, repeatable: true, group: Group.inputs, tier: .standard),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--output", label: "Annotated image", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--json-output", label: "JSON output", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--mask-output-dir", label: "Mask directory", kind: .directory, group: Group.output, tier: .standard),
            .init(
                flag: "--threshold", label: "Threshold", kind: .number,
                defaultValue: "0.05", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--resolution", label: "Resolution", kind: .integer,
                defaultValue: "1008", group: Group.sampling, tier: .expert, range: .init(min: 256, max: 2_048, step: 16)
            ),
            .init(flag: "--show-boxes", label: "Show boxes", kind: .boolean, group: Group.output, tier: .standard),
            .init(flag: "--multimask", label: "Multiple masks", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--json", label: "JSON preflight", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--preflight"),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "png", flag: "--output")
    )

    public static let visionTrack = MereRunCommandCapability(
        id: "vision.track",
        command: ["vision", "track"],
        title: "Track objects",
        summary: "Track text, box, or point prompted objects through video.",
        arguments: [.init(name: "video", label: "Video", kind: .file, required: true)],
        options: [
            .init(flag: "--prompt", label: "Text prompt", kind: .string, repeatable: true, group: Group.prompt, tier: .essential),
            .init(flag: "--box", label: "Box prompt", kind: .string, repeatable: true, group: Group.inputs, tier: .standard),
            .init(flag: "--point", label: "Point prompt", kind: .string, repeatable: true, group: Group.inputs, tier: .standard),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(flag: "--output", label: "Annotated video", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--json-output", label: "JSON output", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--mask-output-dir", label: "Mask directory", kind: .directory, group: Group.output, tier: .standard),
            .init(
                flag: "--init-frame", label: "Initial frame", kind: .integer,
                defaultValue: "0", group: Group.inputs, tier: .standard, range: .init(min: 0, step: 1)
            ),
            .init(
                flag: "--end-frame", label: "End frame", kind: .integer,
                group: Group.inputs, tier: .standard, range: .init(min: 0, step: 1)
            ),
            .init(
                flag: "--threshold", label: "Threshold", kind: .number,
                defaultValue: "0.05", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--resolution", label: "Resolution", kind: .integer,
                defaultValue: "1008", group: Group.sampling, tier: .expert, range: .init(min: 256, max: 2_048, step: 16)
            ),
            .init(flag: "--show-boxes", label: "Show boxes", kind: .boolean, group: Group.output, tier: .standard),
            .init(flag: "--show-labels", label: "Show labels", kind: .boolean, group: Group.output, tier: .expert),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--json", label: "JSON preflight", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--preflight"),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "mp4", flag: "--output")
    )

    public static let visionTrackLive = MereRunCommandCapability(
        id: "vision.track-live",
        command: ["vision", "track-live"],
        title: "Track camera",
        summary: "Capture a camera and track prompted objects.",
        options: [
            .init(flag: "--prompt", label: "Prompt", kind: .string, repeatable: true),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--output", label: "Annotated video", kind: .file, required: true),
            .init(flag: "--json-output", label: "JSON output", kind: .file),
            .init(flag: "--camera", label: "Camera", kind: .integer),
            .init(flag: "--duration-seconds", label: "Duration", kind: .number),
            .init(flag: "--init-frame", label: "Initial frame", kind: .integer),
            .init(flag: "--seed-search-frames", label: "Seed search", kind: .integer),
            .init(flag: "--threshold", label: "Threshold", kind: .number),
            .init(flag: "--resolution", label: "Resolution", kind: .integer),
            .init(flag: "--show-boxes", label: "Show boxes", kind: .boolean),
            .init(flag: "--show-labels", label: "Show labels", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "mp4", flag: "--output")
    )

    public static let visionFaceDetect = MereRunCommandCapability(
        id: "vision.face.detect",
        command: ["vision", "face", "detect"],
        title: "Detect faces",
        summary: "Detect faces, landmarks, and optional identity embeddings.",
        arguments: [.init(name: "image", label: "Image", kind: .file, required: true)],
        options: faceOptions + [
            .init(flag: "--max-faces", label: "Max faces", kind: .integer),
            .init(flag: "--include-embeddings", label: "Embeddings", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--json-output", optional: true)
    )

    public static let visionFaceEmbed = MereRunCommandCapability(
        id: "vision.face.embed",
        command: ["vision", "face", "embed"],
        title: "Embed face",
        summary: "Create one normalized ArcFace identity embedding.",
        arguments: [.init(name: "image", label: "Image", kind: .file, required: true)],
        options: faceOptions + [
            .init(flag: "--face-index", label: "Face index", kind: .integer)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--json-output", optional: true)
    )

    public static let visionFaceCompare = MereRunCommandCapability(
        id: "vision.face.compare",
        command: ["vision", "face", "compare"],
        title: "Compare faces",
        summary: "Compare a face from each of two images.",
        arguments: [
            .init(name: "reference", label: "Reference", kind: .file, required: true),
            .init(name: "candidate", label: "Candidate", kind: .file, required: true)
        ],
        options: faceOptions + [
            .init(flag: "--reference-face-index", label: "Reference index", kind: .integer),
            .init(flag: "--candidate-face-index", label: "Candidate index", kind: .integer)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--json-output", optional: true)
    )

    public static let visionFaceBatch = MereRunCommandCapability(
        id: "vision.face.batch",
        command: ["vision", "face", "batch"],
        title: "Batch faces",
        summary: "Analyze many images in one warm face session.",
        arguments: [.init(name: "images", label: "Images", kind: .file, required: false)],
        options: [
            .init(flag: "--input-list", label: "Input list", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--score-threshold", label: "Score threshold", kind: .number),
            .init(flag: "--execution-provider", label: "Provider", kind: .choice, choices: ["auto", "coreml", "cpu"]),
            .init(flag: "--max-faces", label: "Max faces", kind: .integer),
            .init(flag: "--include-embeddings", label: "Embeddings", kind: .boolean),
            .init(flag: "--jsonl-output", label: "JSONL output", kind: .file),
            .init(flag: "--fail-fast", label: "Fail fast", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "jsonl", flag: "--jsonl-output", optional: true)
    )

    private static let faceOptions: [MereRunCapabilityOption] = [
        .init(flag: "--model", label: "Model", kind: .string),
        .init(flag: "--score-threshold", label: "Score threshold", kind: .number),
        .init(flag: "--execution-provider", label: "Provider", kind: .choice, choices: ["auto", "coreml", "cpu"]),
        .init(flag: "--json-output", label: "JSON output", kind: .file),
        .init(flag: "--json", label: "Print JSON", kind: .boolean)
    ]

    public static let visionPose = MereRunCommandCapability(
        id: "vision.pose",
        command: ["vision", "pose"],
        title: "Pose landmarks",
        summary: "Detect body, hand, and face landmarks with native platform APIs.",
        arguments: [.init(name: "image", label: "Image", kind: .file, required: true)],
        options: [
            .init(flag: "--json-output", label: "JSON output", kind: .file),
            .init(flag: "--no-body", label: "Disable body", kind: .boolean),
            .init(flag: "--no-hands", label: "Disable hands", kind: .boolean),
            .init(flag: "--no-face", label: "Disable face", kind: .boolean),
            .init(flag: "--max-hands", label: "Max hands", kind: .integer),
            .init(flag: "--minimum-confidence", label: "Confidence", kind: .number),
            .init(flag: "--json", label: "Print JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "json", flag: "--json-output")
    )

    public static let visionFlow = MereRunCommandCapability(
        id: "vision.flow",
        command: ["vision", "flow"],
        title: "Optical flow",
        summary: "Generate dense optical flow between two images.",
        arguments: [
            .init(name: "from", label: "From", kind: .file, required: true),
            .init(name: "to", label: "To", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Flow output", kind: .file),
            .init(flag: "--json-output", label: "JSON output", kind: .file),
            .init(flag: "--accuracy", label: "Accuracy", kind: .choice, choices: ["low", "medium", "high", "very-high"]),
            .init(flag: "--json", label: "Print JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "flo", flag: "--output")
    )

    public static let visionDepthVideo = MereRunCommandCapability(
        id: "vision.depth-video",
        command: ["vision", "depth-video"],
        title: "Video depth",
        summary: "Generate temporally consistent relative or metric depth.",
        arguments: [.init(name: "input", label: "Video", kind: .file, required: true)],
        options: [
            .init(flag: "--output", label: "Output directory", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--input-size", label: "Input edge", kind: .integer),
            .init(flag: "--max-frames", label: "Max frames", kind: .integer),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "Print JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output")
    )

    public static let visionGeometry = MereRunCommandCapability(
        id: "vision.geometry",
        command: ["vision", "geometry"],
        title: "Metric geometry",
        summary: "Generate metric depth, normals, camera intrinsics, and a point cloud.",
        arguments: [.init(name: "input", label: "Image", kind: .file, required: true)],
        options: [
            .init(flag: "--output", label: "Output directory", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--resolution-level", label: "Quality", kind: .integer),
            .init(flag: "--token-count", label: "Token count", kind: .integer),
            .init(flag: "--max-points", label: "Max points", kind: .integer),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "Print JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output")
    )

    public static let visionGeometryMultiview = MereRunCommandCapability(
        id: "vision.geometry-multiview",
        command: ["vision", "geometry-multiview"],
        title: "Multi-view geometry",
        summary: "Solve relative geometry, confidence, and cameras from ordered views.",
        arguments: [.init(name: "images", label: "Images", kind: .file, required: true)],
        options: [
            .init(flag: "--output", label: "Output directory", kind: .directory),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--cameras", label: "Cameras", kind: .file),
            .init(flag: "--process-resolution", label: "Process resolution", kind: .integer),
            .init(flag: "--reference-view", label: "Reference view", kind: .choice, choices: ["first", "middle", "saddle-balanced", "saddle-similarity-range"]),
            .init(flag: "--confidence-percentile", label: "Confidence percentile", kind: .number),
            .init(flag: "--max-points", label: "Max points", kind: .integer),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "Print JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output")
    )

    public static let audioEnhance = MereRunCommandCapability(
        id: "audio.enhance",
        command: ["audio", "enhance"],
        title: "Enhance audio",
        summary: "Extend speech or general-audio bandwidth to 48 kHz with native AP-BWE or UniverSR.",
        arguments: [
            .init(name: "audio", label: "Audio", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-path", label: "Model path", kind: .directory),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--overlap", label: "AP-BWE overlap", kind: .integer),
            .init(flag: "--input-rate", label: "Input bandwidth", kind: .integer),
            .init(
                flag: "--ode-method",
                label: "UniverSR ODE method",
                kind: .choice,
                choices: ["euler", "midpoint", "rk4"]
            ),
            .init(flag: "--ode-steps", label: "UniverSR ODE steps", kind: .integer),
            .init(flag: "--guidance-scale", label: "UniverSR guidance", kind: .number),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--chunk-seconds", label: "Chunk seconds", kind: .integer),
            .init(
                flag: "--dtype",
                label: "Compute type",
                kind: .choice,
                choices: ["float16", "float32"]
            ),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output")
    )

    public static let audioGenerate = MereRunCommandCapability(
        id: "audio.generate",
        command: ["audio", "generate"],
        title: "Generate LTX audio",
        summary: "Generate audio only with the native LTX-2.5 text-to-audio pipeline.",
        arguments: [
            .init(name: "prompt", label: "Audio prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--output", label: "WAV output", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string),
            .init(flag: "--enhance-prompt", label: "Enhance prompt", kind: .boolean),
            .init(flag: "--prompt-enhancer-model", label: "Prompt enhancer", kind: .string),
            .init(flag: "--prompt-enhancer-model-root", label: "Prompt enhancer root", kind: .directory),
            .init(flag: "--duration", label: "Duration", kind: .number),
            .init(flag: "--auto-duration", label: "Auto duration range", kind: .string),
            .init(flag: "--num-frames", label: "Video-clock frames", kind: .integer),
            .init(flag: "--fps", label: "Video-clock rate", kind: .integer),
            .init(flag: "--steps", label: "Denoising steps", kind: .integer),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--audio-cfg-guidance-scale", label: "Audio CFG", kind: .number),
            .init(flag: "--audio-stg-guidance-scale", label: "Audio STG", kind: .number),
            .init(flag: "--audio-rescale", label: "Guidance rescale", kind: .number),
            .init(flag: "--audio-stg-block", label: "STG block", kind: .integer, repeatable: true),
            .init(flag: "--audio-skip-step", label: "Guidance skip", kind: .integer),
            .init(flag: "--sigmas", label: "Sigma schedule", kind: .string),
            .init(flag: "--lora", label: "Audio LoRA", kind: .string, repeatable: true),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output")
    )

    public static let musicGenerate = MereRunCommandCapability(
        id: "music.generate",
        command: ["music", "generate"],
        title: "Generate and edit music",
        summary: "Create, cover, repaint, flow-edit, rank, stem, and export production music.",
        arguments: [
            .init(name: "caption", label: "Music prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--lyrics", label: "Lyrics", kind: .string, group: Group.prompt, tier: .standard),
            .init(flag: "--lyrics-file", label: "Lyrics file", kind: .file, group: Group.prompt, tier: .expert),
            .init(flag: "--instrumental", label: "Instrumental", kind: .boolean, group: Group.prompt, tier: .standard),
            .init(flag: "--lrc-file", label: "LRC file", kind: .file, group: Group.prompt, tier: .expert),
            .init(flag: "--lrc-output", label: "LRC output", kind: .file, group: Group.output, tier: .expert),
            .init(flag: "--output", label: "Audio output", kind: .file, group: Group.output, tier: .standard),
            .init(
                flag: "--export-format", label: "Audio format", kind: .choice, choices: ["pcm16", "pcm24", "float32"],
                defaultValue: "pcm24", group: Group.output, tier: .expert
            ),
            .init(
                flag: "--normalize", label: "Normalization", kind: .choice, choices: ["none", "peak"],
                defaultValue: "peak", group: Group.output, tier: .expert
            ),
            .init(
                flag: "--target-peak-db", label: "Peak target", kind: .number,
                defaultValue: "-1.0", group: Group.output, tier: .expert, range: .init(min: -24, max: 0, step: 0.5)
            ),
            .init(
                flag: "--fade-in-ms", label: "Fade in", kind: .number,
                defaultValue: "5.0", group: Group.output, tier: .expert, range: .init(min: 0, max: 5_000, step: 1)
            ),
            .init(
                flag: "--fade-out-ms", label: "Fade out", kind: .number,
                defaultValue: "20.0", group: Group.output, tier: .expert, range: .init(min: 0, max: 5_000, step: 1)
            ),
            .init(flag: "--no-dither", label: "Disable dither", kind: .boolean, group: Group.output, tier: .expert),
            .init(flag: "--recipe-output", label: "Recipe output", kind: .file, group: Group.output, tier: .expert),
            .init(flag: "--no-recipe", label: "Disable recipe", kind: .boolean, group: Group.output, tier: .expert),
            .init(flag: "--daw-bundle", label: "DAW bundle", kind: .directory, group: Group.output, tier: .expert),
            .init(flag: "--stems", label: "Stems", kind: .string, group: Group.output, tier: .expert),
            .init(flag: "--adapter", label: "Adapter", kind: .file, repeatable: true, group: Group.modelAndAdapters, tier: .standard),
            .init(
                flag: "--adapter-kind", label: "Adapter kind", kind: .choice, choices: ["auto", "lora", "lokr"],
                defaultValue: "auto", group: Group.modelAndAdapters, tier: .expert, dependsOn: "--adapter"
            ),
            .init(
                flag: "--adapter-scale", label: "Adapter scale", kind: .number, repeatable: true,
                group: Group.modelAndAdapters, tier: .standard, range: .init(min: 0, max: 2, step: 0.05), dependsOn: "--adapter"
            ),
            .init(
                flag: "--model", label: "Model", kind: .string,
                defaultValue: "music-acestep", group: Group.modelAndAdapters, tier: .essential
            ),
            .init(flag: "--checkpoints-root", label: "Checkpoints root", kind: .directory, group: Group.modelAndAdapters, tier: .expert),
            .init(
                flag: "--decoder-subdirectory", label: "Decoder", kind: .string,
                defaultValue: "acestep-v15-turbo", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--vae-subdirectory", label: "VAE", kind: .string,
                defaultValue: "vae", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(flag: "--lm-subdirectory", label: "Language model", kind: .string, group: Group.modelAndAdapters, tier: .expert),
            .init(flag: "--lm-model", label: "LM model", kind: .string, group: Group.modelAndAdapters, tier: .expert),
            .init(flag: "--text-subdirectory", label: "Text encoder", kind: .string, group: Group.modelAndAdapters, tier: .expert),
            .init(flag: "--use-lm", label: "Use LM planning", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(flag: "--no-lm", label: "Disable LM planning", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(
                flag: "--analyze-source-audio", label: "Analyze source", kind: .boolean,
                group: Group.inputs, tier: .expert, dependsOn: "--source-audio"
            ),
            .init(
                flag: "--duration", label: "Duration", kind: .number,
                group: Group.sampling, tier: .essential, range: .init(min: 1, max: 600, step: 1)
            ),
            .init(
                flag: "--quality", label: "Quality", kind: .choice, choices: ["draft", "song", "final", "edit"],
                group: Group.sampling, tier: .essential
            ),
            .init(
                flag: "--steps", label: "Steps", kind: .integer,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 200, step: 1)
            ),
            .init(
                flag: "--shift", label: "Scheduler shift", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 0.1)
            ),
            .init(
                flag: "--infer-method", label: "Inference method", kind: .choice, choices: ["ode", "sde"],
                group: Group.sampling, tier: .expert
            ),
            .init(flag: "--sampler", label: "Sampler", kind: .choice, choices: ["euler", "heun"], group: Group.sampling, tier: .expert),
            .init(
                flag: "--guidance-scale", label: "Guidance scale", kind: .number,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 20, step: 0.1)
            ),
            .init(
                flag: "--guidance-mode", label: "Guidance mode", kind: .choice, choices: ["apg", "adg", "cfg"],
                group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--cfg-interval-start", label: "CFG start", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--cfg-interval-end", label: "CFG end", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--velocity-norm-threshold", label: "Velocity clamp", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, step: 0.1)
            ),
            .init(
                flag: "--velocity-ema-factor", label: "Velocity EMA", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(flag: "--seed", label: "Seed", kind: .integer, group: Group.sampling, tier: .essential, range: .init(min: 0, step: 1)),
            .init(
                flag: "--candidates", label: "Candidate count", kind: .integer,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 16, step: 1)
            ),
            .init(
                flag: "--keep-candidates", label: "Keep candidates", kind: .boolean,
                group: Group.output, tier: .expert, dependsOn: "--candidates"
            ),
            .init(
                flag: "--audio-cover-strength", label: "Cover strength", kind: .number,
                defaultValue: "1.0", group: Group.inputs, tier: .standard, range: .init(min: 0, max: 1, step: 0.05),
                dependsOn: "--source-audio"
            ),
            .init(
                flag: "--cover-noise-strength", label: "Cover noise", kind: .number,
                defaultValue: "0.0", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 1, step: 0.05),
                dependsOn: "--source-audio"
            ),
            .init(flag: "--retake-seed", label: "Retake seed", kind: .integer, group: Group.sampling, tier: .expert, range: .init(min: 0, step: 1)),
            .init(
                flag: "--retake-variance", label: "Retake variance", kind: .number,
                defaultValue: "0.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.05),
                dependsOn: "--retake-seed"
            ),
            .init(flag: "--vocal-language", label: "Vocal language", kind: .string, defaultValue: "en", group: Group.prompt, tier: .standard),
            .init(
                flag: "--instruction", label: "Instruction", kind: .string,
                defaultValue: "Fill the audio semantic mask based on the given conditions:", group: Group.prompt, tier: .expert
            ),
            .init(
                flag: "--task-type",
                label: "Task",
                kind: .choice,
                choices: ["text2music", "repaint", "cover", "cover-nofsq", "extract", "lego", "complete"],
                defaultValue: "text2music", group: Group.inputs, tier: .standard
            ),
            .init(flag: "--source-audio", label: "Source audio", kind: .file, group: Group.inputs, tier: .standard),
            .init(
                flag: "--reference-audio", label: "Reference audio", kind: .file, repeatable: true,
                group: Group.inputs, tier: .standard
            ),
            .init(flag: "--track-name", label: "Track name", kind: .string, group: Group.inputs, tier: .expert),
            .init(flag: "--complete-track-classes", label: "Track classes", kind: .string, group: Group.inputs, tier: .expert),
            .init(flag: "--non-cover", label: "No FSQ cover", kind: .boolean, group: Group.inputs, tier: .expert, dependsOn: "--source-audio"),
            .init(
                flag: "--repaint-start", label: "Repaint start", kind: .number,
                defaultValue: "0.0", group: Group.inputs, tier: .standard, range: .init(min: 0, step: 0.1), dependsOn: "--source-audio"
            ),
            .init(
                flag: "--repaint-end", label: "Repaint end", kind: .number,
                defaultValue: "-1.0", group: Group.inputs, tier: .standard, range: .init(min: -1, step: 0.1), dependsOn: "--source-audio"
            ),
            .init(
                flag: "--chunk-mask-mode", label: "Chunk mask", kind: .choice, choices: ["auto", "explicit"],
                defaultValue: "auto", group: Group.inputs, tier: .expert, dependsOn: "--source-audio"
            ),
            .init(
                flag: "--repaint-mode", label: "Repaint mode", kind: .choice, choices: ["conservative", "balanced", "aggressive"],
                defaultValue: "balanced", group: Group.inputs, tier: .expert, dependsOn: "--source-audio"
            ),
            .init(
                flag: "--repaint-strength", label: "Repaint strength", kind: .number,
                defaultValue: "0.5", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 1, step: 0.05),
                dependsOn: "--source-audio"
            ),
            .init(flag: "--flow-edit", label: "Flow edit", kind: .boolean, group: Group.inputs, tier: .standard, dependsOn: "--source-audio"),
            .init(flag: "--source-caption", label: "Source caption", kind: .string, group: Group.inputs, tier: .standard, dependsOn: "--flow-edit"),
            .init(flag: "--source-lyrics", label: "Source lyrics", kind: .string, group: Group.inputs, tier: .expert, dependsOn: "--flow-edit"),
            .init(
                flag: "--flow-edit-n-min", label: "Flow start", kind: .number,
                defaultValue: "0.0", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 1, step: 0.05), dependsOn: "--flow-edit"
            ),
            .init(
                flag: "--flow-edit-n-max", label: "Flow end", kind: .number,
                defaultValue: "1.0", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 1, step: 0.05), dependsOn: "--flow-edit"
            ),
            .init(
                flag: "--flow-edit-n-average", label: "Flow samples", kind: .integer,
                defaultValue: "1", group: Group.inputs, tier: .expert, range: .init(min: 1, max: 16, step: 1), dependsOn: "--flow-edit"
            ),
            .init(flag: "--bpm", label: "BPM", kind: .integer, group: Group.prompt, tier: .standard, range: .init(min: 20, max: 300, step: 1)),
            .init(flag: "--keyscale", label: "Key", kind: .string, group: Group.prompt, tier: .standard),
            .init(flag: "--timesignature", label: "Time signature", kind: .string, group: Group.prompt, tier: .standard),
            .init(
                flag: "--lm-temperature", label: "LM temperature", kind: .number,
                defaultValue: "0.85", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(
                flag: "--lm-top-k", label: "LM top-k", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1_000, step: 1)
            ),
            .init(
                flag: "--lm-top-p", label: "LM top-p", kind: .number,
                defaultValue: "0.9", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.01)
            ),
            .init(
                flag: "--lm-repetition-penalty", label: "LM repetition penalty", kind: .number,
                defaultValue: "1.0", group: Group.sampling, tier: .expert, range: .init(min: 0.5, max: 2, step: 0.05)
            ),
            .init(
                flag: "--lm-cfg-scale", label: "LM CFG scale", kind: .number,
                defaultValue: "2.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 0.1)
            ),
            .init(
                flag: "--lm-negative-prompt", label: "LM negative prompt", kind: .string,
                defaultValue: "NO USER INPUT", group: Group.sampling, tier: .expert
            ),
            .init(flag: "--metadata-duration", label: "Metadata duration", kind: .string, group: Group.prompt, tier: .expert),
            .init(flag: "--metadata-language", label: "Metadata language", kind: .string, group: Group.prompt, tier: .expert),
            .init(flag: "--no-tiled-vae", label: "Disable tiled VAE", kind: .boolean, group: Group.run, tier: .expert),
            .init(
                flag: "--vae-chunk-size", label: "VAE chunk", kind: .integer,
                defaultValue: "512", group: Group.run, tier: .expert, range: .init(min: 64, max: 4_096, step: 64)
            ),
            .init(
                flag: "--vae-overlap", label: "VAE overlap", kind: .integer,
                defaultValue: "64", group: Group.run, tier: .expert, range: .init(min: 0, max: 1_024, step: 8)
            ),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            .init(
                flag: "--temperature", label: "RT2 temperature", kind: .number,
                defaultValue: "1.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(
                flag: "--style-conditioning", label: "RT2 style", kind: .choice, choices: ["streaming", "full"],
                defaultValue: "streaming", group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--top-k", label: "RT2 top-k", kind: .integer,
                defaultValue: "100", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1_000, step: 1)
            ),
            .init(
                flag: "--cfg-musiccoca", label: "MusicCoCa CFG", kind: .number,
                defaultValue: "3.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(
                flag: "--cfg-notes", label: "Notes CFG", kind: .number,
                defaultValue: "5.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(
                flag: "--cfg-drums", label: "Drums CFG", kind: .number,
                defaultValue: "1.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(flag: "--drumless", label: "Drumless", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(
                flag: "--unmask-width", label: "Unmask width", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, step: 1)
            ),
            .init(
                flag: "--seed-rotation", label: "Seed rotation", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, step: 1)
            ),
            .init(flag: "--prefill-silence", label: "Prefill silence", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(
                flag: "--prefill-duration", label: "Prefill duration", kind: .number,
                defaultValue: "1.64", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 30, step: 0.01),
                dependsOn: "--prefill-silence"
            ),
            progressJSONOption,
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output")
    )

    public static let musicAnalyze = MereRunCommandCapability(
        id: "music.analyze",
        command: ["music", "analyze"],
        title: "Analyze music",
        summary: "Extract structured music metadata and optional ACE-Step audio codes.",
        arguments: [
            .init(name: "audio", label: "Audio", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--checkpoints-root", label: "Checkpoints root", kind: .directory),
            .init(flag: "--decoder-subdirectory", label: "Decoder", kind: .string),
            .init(flag: "--vae-subdirectory", label: "VAE", kind: .string),
            .init(flag: "--lm-subdirectory", label: "Language model", kind: .string),
            .init(flag: "--lm-model", label: "LM model", kind: .string),
            .init(flag: "--duration", label: "Duration", kind: .number),
            .init(flag: "--max-new-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--lm-temperature", label: "LM temperature", kind: .number),
            .init(flag: "--lm-top-k", label: "LM top-k", kind: .integer),
            .init(flag: "--lm-top-p", label: "LM top-p", kind: .number),
            .init(flag: "--include-raw-lm", label: "Raw LM", kind: .boolean),
            .init(flag: "--include-audio-codes", label: "Audio codes", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let musicTranscribe = MereRunCommandCapability(
        id: "music.transcribe",
        command: ["music", "transcribe"],
        title: "Transcribe music",
        summary: "Turn a full mix into separated MIDI or structured events with musical context.",
        arguments: [
            .init(name: "audio", label: "Audio", kind: .file, required: false)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-path", label: "Model path", kind: .directory),
            .init(flag: "--variant", label: "Variant", kind: .choice, choices: ["small", "medium", "large"]),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--format", label: "Format", kind: .choice, choices: ["midi", "json", "jsonl"]),
            .init(flag: "--instruments", label: "Instruments", kind: .string),
            .init(flag: "--list-instruments", label: "List instruments", kind: .boolean),
            .init(flag: "--sampling", label: "Sampling", kind: .boolean),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--max-tokens-per-chunk", label: "Tokens per chunk", kind: .integer),
            .init(flag: "--strict-eos", label: "Strict EOS", kind: .boolean),
            .init(flag: "--beam-size", label: "Beam size", kind: .integer),
            .init(flag: "--chunk-batch-size", label: "Chunk batch", kind: .integer),
            .init(flag: "--dtype", label: "Compute type", kind: .choice, choices: ["bfloat16", "float16", "float32"]),
            .init(flag: "--no-musical-context", label: "Disable context", kind: .boolean),
            .init(flag: "--context-output", label: "Context output", kind: .file),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, flag: "--output")
    )

    public static let musicSeparate = MereRunCommandCapability(
        id: "music.separate",
        command: ["music", "separate"],
        title: "Separate or restore music",
        summary: "Create stems, remove reverb, or denoise audio with native RoFormer models.",
        arguments: [
            .init(name: "audio", label: "Audio", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-path", label: "Model path", kind: .directory),
            .init(flag: "--output-dir", label: "Output directory", kind: .directory),
            .init(flag: "--overlap", label: "Overlap", kind: .integer),
            .init(
                flag: "--dtype",
                label: "Compute type",
                kind: .choice,
                choices: ["float16", "float32"]
            ),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--output-dir")
    )

    public static let musicRealtime = MereRunCommandCapability(
        id: "music.realtime",
        command: ["music", "realtime"],
        title: "Realtime music",
        summary: "Run live Magenta RT2 generation with text, interactive, and MIDI steering.",
        arguments: [
            .init(name: "prompt", label: "Prompt", kind: .string, required: false)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--duration", label: "Duration", kind: .number),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--no-play", label: "Disable playback", kind: .boolean),
            .init(flag: "--style-conditioning", label: "Style", kind: .choice, choices: ["streaming", "full"]),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-k", label: "Top-k", kind: .integer),
            .init(flag: "--cfg-musiccoca", label: "MusicCoCa CFG", kind: .number),
            .init(flag: "--cfg-notes", label: "Notes CFG", kind: .number),
            .init(flag: "--cfg-drums", label: "Drums CFG", kind: .number),
            .init(flag: "--drumless", label: "Drumless", kind: .boolean),
            .init(flag: "--unmask-width", label: "Unmask width", kind: .integer),
            .init(flag: "--seed-rotation", label: "Seed rotation", kind: .integer),
            .init(flag: "--prefill-silence", label: "Prefill silence", kind: .boolean),
            .init(flag: "--prefill-duration", label: "Prefill duration", kind: .number),
            .init(flag: "--interactive", label: "Interactive", kind: .boolean),
            .init(flag: "--list-midi-inputs", label: "List MIDI", kind: .boolean),
            .init(flag: "--midi-monitor", label: "MIDI monitor", kind: .boolean),
            .init(flag: "--midi-log-events", label: "Log MIDI", kind: .boolean),
            .init(flag: "--midi-log-raw", label: "Log raw MIDI", kind: .boolean),
            .init(flag: "--midi-input", label: "MIDI input", kind: .string),
            .init(flag: "--midi-channel", label: "MIDI channel", kind: .string),
            .init(flag: "--midi-note-offset", label: "MIDI transpose", kind: .integer),
            .init(flag: "--midi-cc", label: "MIDI CC map", kind: .string, repeatable: true),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .service, fileExtension: "wav", flag: "--output", optional: true)
    )

    public static let musicTrainAdapter = MereRunCommandCapability(
        id: "music.train-adapter",
        command: ["music", "train-adapter"],
        title: "Train music adapter",
        summary: "Train a native ACE-Step LoRA or LoKr adapter.",
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--dataset", label: "Dataset", kind: .file, required: true),
            .init(flag: "--output", label: "Output", kind: .file, required: true),
            .init(flag: "--kind", label: "Adapter kind", kind: .choice, choices: ["lora", "lokr"]),
            .init(flag: "--rank", label: "Rank", kind: .integer),
            .init(flag: "--alpha", label: "Alpha", kind: .number),
            .init(flag: "--factor", label: "LoKr factor", kind: .integer),
            .init(flag: "--steps", label: "Steps", kind: .integer),
            .init(flag: "--learning-rate", label: "Learning rate", kind: .number),
            .init(flag: "--weight-decay", label: "Weight decay", kind: .number),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--max-duration", label: "Max duration", kind: .number),
            .init(flag: "--checkpoints-root", label: "Checkpoints root", kind: .directory),
            .init(flag: "--decoder-subdirectory", label: "Decoder", kind: .string),
            .init(flag: "--vae-subdirectory", label: "VAE", kind: .string),
            .init(flag: "--text-subdirectory", label: "Text encoder", kind: .string),
            .init(flag: "--log-every", label: "Progress interval", kind: .integer)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )

    public static let musicServe = MereRunCommandCapability(
        id: "music.serve",
        command: ["music", "serve"],
        title: "Serve resident music",
        summary: "Keep ACE-Step, its language model, and adapter stack warm behind a local API.",
        options: [
            .init(flag: "--host", label: "Host", kind: .string),
            .init(flag: "--port", label: "Port", kind: .integer),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--checkpoints-root", label: "Checkpoints root", kind: .directory),
            .init(flag: "--decoder-subdirectory", label: "Decoder", kind: .string),
            .init(flag: "--vae-subdirectory", label: "VAE", kind: .string),
            .init(flag: "--lm-subdirectory", label: "Language model", kind: .string),
            .init(flag: "--lm-model", label: "LM model", kind: .string),
            .init(flag: "--text-subdirectory", label: "Text encoder", kind: .string),
            .init(flag: "--adapter", label: "Adapter", kind: .file, repeatable: true),
            .init(flag: "--adapter-kind", label: "Adapter kind", kind: .choice, choices: ["auto", "lora", "lokr"]),
            .init(flag: "--adapter-scale", label: "Adapter scale", kind: .number, repeatable: true),
            .init(flag: "--api-key", label: "API key", kind: .string)
        ],
        output: .init(kind: .service)
    )

    public static let videoGenerate = MereRunCommandCapability(
        id: "video.generate",
        command: ["video", "generate"],
        title: "Generate video",
        summary: "Generate LTX, Wan, or synchronized MiniMax-H3 video from text and ordered media conditioning.",
        arguments: [
            .init(name: "prompt", label: "Prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Output", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .essential),
            .init(
                flag: "--quality",
                label: "Quality",
                kind: .choice,
                choices: LTXVideoQuality.allCases.map(\.rawValue),
                group: Group.sampling, tier: .essential
            ),
            .init(
                flag: "--output-mode",
                label: "Output mode",
                kind: .choice,
                choices: LTXVideoOutputMode.allCases.map(\.rawValue),
                group: Group.output, tier: .essential
            ),
            .init(flag: "--model-root", label: "Model root", kind: .directory, group: Group.modelAndAdapters, tier: .expert),
            .init(flag: "--auto-duration", label: "Auto duration range", kind: .string, group: Group.sampling, tier: .expert),
            .init(
                flag: "--video-decoder", label: "Video decoder", kind: .choice, choices: ["diffusion", "convolutional"],
                group: Group.run, tier: .expert
            ),
            .init(
                flag: "--hdr", label: "HDR color space", kind: .choice, choices: ["srgb-linear", "acescg", "acescct"],
                group: Group.output, tier: .expert
            ),
            .init(
                flag: "--hdr-transfer", label: "HDR transfer", kind: .choice, choices: ["acescct", "logc3"],
                group: Group.output, tier: .expert, dependsOn: "--hdr"
            ),
            .init(flag: "--high-quality-hdr", label: "High quality HDR", kind: .boolean, group: Group.output, tier: .expert, dependsOn: "--hdr"),
            .init(flag: "--text-embeddings", label: "Precomputed text contexts", kind: .file, group: Group.inputs, tier: .expert),
            .init(
                flag: "--spatial-tile", label: "VAE spatial tile", kind: .integer,
                group: Group.run, tier: .expert, range: .init(min: 256, max: 4_096, step: 32)
            ),
            .init(
                flag: "--spatial-overlap", label: "VAE spatial overlap", kind: .integer,
                defaultValue: "256", group: Group.run, tier: .expert, range: .init(min: 0, max: 1_024, step: 32)
            ),
            .init(flag: "--skip-mp4", label: "EXR only", kind: .boolean, group: Group.output, tier: .expert, dependsOn: "--hdr"),
            .init(
                flag: "--width", label: "Width", kind: .integer,
                group: Group.output, tier: .essential, range: .init(min: 256, max: 2_048, step: 16)
            ),
            .init(
                flag: "--height", label: "Height", kind: .integer,
                group: Group.output, tier: .essential, range: .init(min: 256, max: 2_048, step: 16)
            ),
            .init(
                flag: "--num-frames", label: "Frames", kind: .integer,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 1_000, step: 1)
            ),
            .init(
                flag: "--duration", label: "Duration", kind: .number,
                group: Group.sampling, tier: .essential, range: .init(min: 1, max: 60, step: 0.5)
            ),
            .init(
                flag: "--fps", label: "Frames per second", kind: .integer,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 60, step: 1)
            ),
            .init(flag: "--seed", label: "Seed", kind: .integer, group: Group.sampling, tier: .essential, range: .init(min: 0, step: 1)),
            .init(
                flag: "--steps", label: "Denoising steps", kind: .integer,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 100, step: 1)
            ),
            .init(
                flag: "--h3-weight-mode",
                label: "MiniMax-H3 weight mode",
                kind: .choice,
                choices: ["auto", "quantized", "resident-bf16"],
                defaultValue: "auto", group: Group.modelAndAdapters, tier: .expert
            ),
            .init(
                flag: "--h3-acceleration",
                label: "MiniMax-H3 acceleration",
                kind: .choice,
                choices: [
                    "quality", "balanced", "maximum",
                    "layers-45", "layers-40", "velocity-reuse-2", "token-reduction"
                ],
                defaultValue: "quality", group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--guidance-scale", label: "Wan guidance", kind: .number,
                defaultValue: "5.0", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(
                flag: "--shift", label: "Wan schedule shift", kind: .number,
                defaultValue: "5.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 0.1)
            ),
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string, group: Group.prompt, tier: .standard),
            .init(flag: "--enhance-prompt", label: "Enhance prompt", kind: .boolean, group: Group.prompt, tier: .standard),
            .init(
                flag: "--prompt-enhancer-model", label: "Prompt enhancer", kind: .string,
                group: Group.prompt, tier: .expert, dependsOn: "--enhance-prompt"
            ),
            .init(
                flag: "--prompt-enhancer-model-root", label: "Prompt enhancer root", kind: .directory,
                group: Group.prompt, tier: .expert, dependsOn: "--enhance-prompt"
            ),
            .init(flag: "--audio", label: "Source audio", kind: .file, group: Group.inputs, tier: .standard),
            .init(
                flag: "--audio-start-time", label: "Audio start", kind: .number,
                defaultValue: "0.0", group: Group.inputs, tier: .standard, range: .init(min: 0, step: 0.1), dependsOn: "--audio"
            ),
            .init(
                flag: "--audio-max-duration", label: "Audio max duration", kind: .number,
                group: Group.inputs, tier: .expert, range: .init(min: 0, step: 0.1), dependsOn: "--audio"
            ),
            .init(
                flag: "--a2v-guidance-scale", label: "Audio-to-video guidance", kind: .number,
                defaultValue: "3.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(
                flag: "--video-cfg-guidance-scale", label: "Video CFG guidance", kind: .number,
                defaultValue: "3.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(
                flag: "--audio-cfg-guidance-scale", label: "Audio CFG guidance", kind: .number,
                defaultValue: "7.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(
                flag: "--v2a-guidance-scale", label: "Video-to-audio guidance", kind: .number,
                defaultValue: "3.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(
                flag: "--a2v-steps", label: "Audio-to-video steps", kind: .integer,
                defaultValue: "30", group: Group.sampling, tier: .expert, range: .init(min: 1, max: 100, step: 1)
            ),
            .init(
                flag: "--ltx-preset", label: "LTX preset", kind: .choice, choices: ["standard", "hq"],
                defaultValue: "standard", group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--ltx-pipeline", label: "LTX pipeline", kind: .choice, choices: ["two-stage", "dev-one-stage"],
                defaultValue: "two-stage", group: Group.sampling, tier: .expert
            ),
            .init(
                flag: "--ltx-sampler", label: "LTX sampler", kind: .choice,
                choices: ["euler", "res2s", "euler-ancestral", "cfg-plus-plus", "gradient-estimating-euler"],
                group: Group.sampling, tier: .expert
            ),
            .init(flag: "--ltx-sigmas", label: "Stage one sigmas", kind: .string, group: Group.sampling, tier: .expert),
            .init(flag: "--ltx-stage-2-sigmas", label: "Stage two sigmas", kind: .string, group: Group.sampling, tier: .expert),
            .init(
                flag: "--distilled-lora-strength-stage-1", label: "Stage one distilled strength", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.05)
            ),
            .init(
                flag: "--distilled-lora-strength-stage-2", label: "Stage two distilled strength", kind: .number,
                group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.05)
            ),
            .init(
                flag: "--ltx-sampler-eta", label: "Sampler eta", kind: .number,
                defaultValue: "0.5", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.05)
            ),
            .init(
                flag: "--video-stg-scale", label: "Video STG", kind: .number,
                defaultValue: "1.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 0.1)
            ),
            .init(
                flag: "--video-guidance-rescale", label: "Video rescale", kind: .number,
                defaultValue: "0.7", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.05)
            ),
            .init(
                flag: "--video-stg-block", label: "Video STG block", kind: .integer, repeatable: true,
                group: Group.sampling, tier: .expert, range: .init(min: 0, step: 1)
            ),
            .init(
                flag: "--video-guidance-skip-step", label: "Video guidance skip", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 1)
            ),
            .init(
                flag: "--audio-stg-scale", label: "Audio STG", kind: .number,
                defaultValue: "1.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 0.1)
            ),
            .init(
                flag: "--audio-guidance-rescale", label: "Audio rescale", kind: .number,
                defaultValue: "0.7", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 1, step: 0.05)
            ),
            .init(
                flag: "--audio-stg-block", label: "Audio STG block", kind: .integer, repeatable: true,
                group: Group.sampling, tier: .expert, range: .init(min: 0, step: 1)
            ),
            .init(
                flag: "--audio-guidance-skip-step", label: "Audio guidance skip", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 1)
            ),
            .init(flag: "--no-res2s-bong-math", label: "Disable Res2s anchor refinement", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(
                flag: "--res2s-bong-max-iterations", label: "Res2s iterations", kind: .integer,
                defaultValue: "100", group: Group.sampling, tier: .expert, range: .init(min: 1, max: 1_000, step: 1)
            ),
            .init(
                flag: "--gradient-estimation-gamma", label: "Gradient estimate gamma", kind: .number,
                defaultValue: "2.0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 10, step: 0.1)
            ),
            .init(flag: "--image", label: "Start image", kind: .file, group: Group.inputs, tier: .essential),
            .init(
                flag: "--image-strength", label: "Start image strength", kind: .number,
                defaultValue: "1.0", group: Group.inputs, tier: .standard, range: .init(min: 0, max: 1, step: 0.05), dependsOn: "--image"
            ),
            .init(flag: "--end-image", label: "End image", kind: .file, group: Group.inputs, tier: .standard, dependsOn: "--image"),
            .init(
                flag: "--end-image-strength", label: "End image strength", kind: .number,
                defaultValue: "1.0", group: Group.inputs, tier: .standard, range: .init(min: 0, max: 1, step: 0.05), dependsOn: "--end-image"
            ),
            .init(
                flag: "--image-conditioning", label: "Timed image guide", kind: .string, repeatable: true,
                group: Group.inputs, tier: .expert
            ),
            .init(
                flag: "--num-generated-keyframes", label: "Generated keyframe count", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 16, step: 1)
            ),
            .init(
                flag: "--generated-keyframe", label: "Generated keyframe", kind: .integer, repeatable: true,
                group: Group.sampling, tier: .expert, range: .init(min: 0, step: 1)
            ),
            .init(flag: "--lora", label: "LTX LoRA", kind: .string, repeatable: true, group: Group.modelAndAdapters, tier: .standard),
            .init(
                flag: "--video-conditioning", label: "IC-LoRA reference video", kind: .string, repeatable: true,
                group: Group.inputs, tier: .expert
            ),
            .init(
                flag: "--conditioning-attention-strength", label: "Reference attention", kind: .number,
                defaultValue: "1.0", group: Group.inputs, tier: .expert, range: .init(min: 0, max: 1, step: 0.05),
                dependsOn: "--video-conditioning"
            ),
            .init(
                flag: "--conditioning-attention-mask", label: "Reference attention mask", kind: .file,
                group: Group.inputs, tier: .expert, dependsOn: "--video-conditioning"
            ),
            .init(flag: "--skip-stage-2", label: "Stage one preview", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(
                flag: "--reference-downscale-factor", label: "Reference spatial scale", kind: .integer,
                group: Group.inputs, tier: .expert, range: .init(min: 1, max: 8, step: 1), dependsOn: "--video-conditioning"
            ),
            .init(
                flag: "--reference-temporal-scale-factor", label: "Reference temporal scale", kind: .integer,
                group: Group.inputs, tier: .expert, range: .init(min: 1, max: 8, step: 1), dependsOn: "--video-conditioning"
            ),
            .init(flag: "--dfr", label: "Diffusion fidelity rendering", kind: .boolean, group: Group.sampling, tier: .expert),
            .init(
                flag: "--temporal-upsample-rounds", label: "Temporal refinement rounds", kind: .integer,
                defaultValue: "0", group: Group.sampling, tier: .expert, range: .init(min: 0, max: 2, step: 1), dependsOn: "--dfr"
            ),
            .init(
                flag: "--detailing-lora", label: "Detailing IC-LoRA", kind: .string, repeatable: true,
                group: Group.modelAndAdapters, tier: .expert, dependsOn: "--dfr"
            ),
            .init(
                flag: "--detailing-reference-downscale-factor", label: "Detail reference scale", kind: .integer,
                group: Group.modelAndAdapters, tier: .expert, range: .init(min: 1, max: 8, step: 1), dependsOn: "--dfr"
            ),
            .init(flag: "--reference", label: "Ordered H3 reference", kind: .string, repeatable: true, group: Group.inputs, tier: .standard),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--json", label: "JSON", kind: .boolean, group: Group.run, tier: .expert, dependsOn: "--preflight"),
            .init(flag: "--timings", label: "Timings", kind: .boolean, group: Group.run, tier: .expert),
            .init(flag: "--timings-output", label: "Timings output", kind: .file, group: Group.run, tier: .expert),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            progressJSONOption,
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "mp4", flag: "--output")
    )

    public static let videoRetake = MereRunCommandCapability(
        id: "video.retake",
        command: ["video", "retake"],
        title: "Retake video region",
        summary: "Regenerate a bounded video and/or audio region with native LTX-2.5.",
        arguments: [
            .init(name: "prompt", label: "Replacement prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--source", label: "Source video or EXR folder", kind: .file, required: true),
            .init(flag: "--frame-rate", label: "EXR frame rate", kind: .integer),
            .init(flag: "--start-time", label: "Start time", kind: .number, required: true),
            .init(flag: "--end-time", label: "End time", kind: .number, required: true),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string),
            .init(flag: "--enhance-prompt", label: "Enhance prompt", kind: .boolean),
            .init(flag: "--prompt-enhancer-model", label: "Prompt enhancer", kind: .string),
            .init(flag: "--prompt-enhancer-model-root", label: "Prompt enhancer root", kind: .directory),
            .init(flag: "--steps", label: "Denoising steps", kind: .integer),
            .init(flag: "--sigmas", label: "Sigma schedule", kind: .string),
            .init(flag: "--lora", label: "LTX LoRA", kind: .string, repeatable: true),
            .init(flag: "--video-cfg-guidance-scale", label: "Video CFG", kind: .number),
            .init(flag: "--video-stg-scale", label: "Video STG", kind: .number),
            .init(flag: "--video-guidance-rescale", label: "Video rescale", kind: .number),
            .init(flag: "--video-modality-scale", label: "Video modality guidance", kind: .number),
            .init(flag: "--video-stg-block", label: "Video STG block", kind: .integer, repeatable: true),
            .init(flag: "--video-guidance-skip-step", label: "Video guidance skip", kind: .integer),
            .init(flag: "--audio-cfg-guidance-scale", label: "Audio CFG", kind: .number),
            .init(flag: "--audio-stg-scale", label: "Audio STG", kind: .number),
            .init(flag: "--audio-guidance-rescale", label: "Audio rescale", kind: .number),
            .init(flag: "--audio-modality-scale", label: "Audio modality guidance", kind: .number),
            .init(flag: "--audio-stg-block", label: "Audio STG block", kind: .integer, repeatable: true),
            .init(flag: "--audio-guidance-skip-step", label: "Audio guidance skip", kind: .integer),
            .init(flag: "--video-decoder", label: "Video decoder", kind: .choice, choices: ["diffusion", "convolutional"]),
            .init(flag: "--hdr", label: "HDR color space", kind: .choice, choices: ["srgb-linear", "acescg", "acescct"]),
            .init(flag: "--hdr-transfer", label: "HDR transfer", kind: .choice, choices: ["acescct", "logc3"]),
            .init(flag: "--preserve-video", label: "Preserve video", kind: .boolean),
            .init(flag: "--preserve-audio", label: "Preserve audio", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "mp4", flag: "--output")
    )

    public static let videoDubIt = MereRunCommandCapability(
        id: "video.dub-it",
        command: ["video", "dub-it"],
        title: "Dub-It",
        summary: "Rephrase a reference performance while preserving speaker identity and lip motion.",
        arguments: [
            .init(name: "prompt", label: "New performance prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--reference-video", label: "Reference AV", kind: .file, required: true),
            .init(flag: "--ic-lora", label: "Dub-It IC-LoRA", kind: .file, required: true),
            .init(flag: "--ic-lora-strength", label: "IC-LoRA strength", kind: .number),
            .init(flag: "--reference-strength", label: "Reference strength", kind: .number),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--width", label: "Width", kind: .integer),
            .init(flag: "--height", label: "Height", kind: .integer),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--image-conditioning", label: "Timed image guide", kind: .string, repeatable: true),
            .init(flag: "--stage-1-sigmas", label: "Stage one sigmas", kind: .string),
            .init(flag: "--stage-2-sigmas", label: "Stage two sigmas", kind: .string),
            .init(flag: "--enhance-prompt", label: "Enhance prompt", kind: .boolean),
            .init(flag: "--prompt-enhancer-model", label: "Prompt enhancer", kind: .string),
            .init(flag: "--prompt-enhancer-model-root", label: "Prompt enhancer root", kind: .directory),
            .init(flag: "--video-decoder", label: "Video decoder", kind: .choice, choices: ["diffusion", "convolutional"]),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "mp4", flag: "--output")
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
        output: .init(kind: .file, fileExtension: "mp4", flag: "--output")
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
        output: .init(kind: .file, flag: "--output")
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
        output: .init(kind: .directory, flag: "--output-dir")
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
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
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

    public static let adapterList = MereRunCommandCapability(
        id: "adapter.list",
        command: ["adapter", "list"],
        title: "Browse adapters",
        summary: "List verified LoRA adapters, compatibility, provenance, and install state.",
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let adapterPull = MereRunCommandCapability(
        id: "adapter.pull",
        command: ["adapter", "pull"],
        title: "Pull adapter",
        summary: "Download and checksum-verify one cataloged LoRA adapter.",
        arguments: [
            .init(name: "target", label: "Adapter id", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--force", label: "Replace install", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors")
    )

    public static let runList = MereRunCommandCapability(
        id: "run.list",
        command: ["run", "list"],
        title: "Browse durable runs",
        summary: "Find local run directories and reports or list remote Relay jobs.",
        options: [
            .init(flag: "--root", label: "Local root", kind: .directory),
            .init(flag: "--executor", label: "Remote executor", kind: .string),
            .init(flag: "--limit", label: "Remote limit", kind: .integer),
            .init(flag: "--max-depth", label: "Scan depth", kind: .integer),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let runInspect = MereRunCommandCapability(
        id: "run.inspect",
        command: ["run", "inspect"],
        title: "Inspect durable run",
        summary: "Inspect a run directory, report, plan, or remote job reference.",
        arguments: [
            .init(name: "path", label: "Run path or reference", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let runWatch = MereRunCommandCapability(
        id: "run.watch",
        command: ["run", "watch"],
        title: "Watch remote run",
        summary: "Poll a remote graph job and stream worker events until completion.",
        arguments: [
            .init(name: "reference", label: "Remote run reference", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--poll-interval", label: "Poll interval", kind: .number),
            .init(flag: "--json-stream", label: "NDJSON events", kind: .boolean),
            .init(flag: "--json", label: "Final JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let runFetch = MereRunCommandCapability(
        id: "run.fetch",
        command: ["run", "fetch"],
        title: "Fetch remote run",
        summary: "Verify and materialize a remote run and selected artifacts locally.",
        arguments: [
            .init(name: "reference", label: "Remote run reference", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--into", label: "Destination", kind: .directory, required: true),
            .init(flag: "--all-artifacts", label: "All artifacts", kind: .boolean),
            .init(flag: "--artifact", label: "Named artifact", kind: .string, repeatable: true),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .directory, flag: "--into")
    )

    public static let runCancel = MereRunCommandCapability(
        id: "run.cancel",
        command: ["run", "cancel"],
        title: "Cancel run",
        summary: "Request cooperative cancellation for a local or remote graph run.",
        arguments: [
            .init(name: "reference", label: "Run reference", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let runRetry = MereRunCommandCapability(
        id: "run.retry",
        command: ["run", "retry"],
        title: "Retry Relay run",
        summary: "Retry one immutable Relay graph job with the same bundle.",
        arguments: [
            .init(name: "reference", label: "Relay run reference", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let worldServe = MereRunCommandCapability(
        id: "world.serve",
        command: ["world", "serve"],
        title: "World session",
        summary: "Serve one warm DreamX or Cosmos3 conditioned-video world session.",
        options: [
            .init(flag: "--host", label: "Host", kind: .string),
            .init(flag: "--port", label: "Port", kind: .integer),
            .init(flag: "--api-key", label: "API key", kind: .string),
            .init(flag: "--backend", label: "Backend", kind: .choice, choices: ["dreamx", "cosmos3"]),
            .init(flag: "--base-model", label: "Base model", kind: .string),
            .init(flag: "--model", label: "World model", kind: .string),
            .init(flag: "--state-directory", label: "State directory", kind: .directory),
            .init(flag: "--prepare", label: "Warm models", kind: .boolean)
        ],
        output: .init(kind: .service)
    )

    public static let status = MereRunCommandCapability(
        id: "status",
        command: ["status"],
        title: "Status snapshot",
        summary: "Inspect the API server, loaded models, model store, and local inventory.",
        options: [
            .init(flag: "--host", label: "Host", kind: .string),
            .init(flag: "--port", label: "Port", kind: .integer),
            .init(flag: "--api-key", label: "API key", kind: .string),
            .init(flag: "--timeout-seconds", label: "Timeout", kind: .number),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let gate = MereRunCommandCapability(
        id: "gate",
        command: ["gate"],
        title: "Quality gate",
        summary: "Run installed-model correctness, determinism, and performance checks.",
        options: [
            .init(flag: "--suite", label: "Suites", kind: .string),
            .init(flag: "--update-baselines", label: "Update baselines", kind: .boolean),
            .init(flag: "--strict-perf", label: "Strict performance", kind: .boolean),
            .init(flag: "--json-output", label: "JSON report", kind: .file),
            .init(flag: "--list", label: "List checks", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--json-output", optional: true)
    )

    public static let evaluationPackValidate = MereRunCommandCapability(
        id: "eval.pack.validate",
        command: ["eval", "pack", "validate"],
        title: "Validate evaluation pack",
        summary: "Validate and content-hash a declared-file-only external evaluation pack.",
        arguments: [
            .init(name: "pack", label: "Evaluation pack", kind: .directory, required: true)
        ],
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let evaluationRun = MereRunCommandCapability(
        id: "eval.run",
        command: ["eval", "run"],
        title: "Run external evaluation",
        summary: "Run matched model, prompt, and adapter arms with resumable reports and gates.",
        arguments: [
            .init(name: "pack", label: "Evaluation pack", kind: .directory, required: true)
        ],
        options: [
            .init(flag: "--model", label: "Model slot binding", kind: .string, required: true, repeatable: true),
            .init(flag: "--adapter", label: "Adapter slot binding", kind: .string, repeatable: true),
            .init(flag: "--trials", label: "Trials", kind: .integer),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--context-size", label: "Context size", kind: .integer),
            .init(
                flag: "--logprobs",
                label: "Logprob capture",
                kind: .choice,
                choices: ["none", "summary", "tokens", "top"]
            ),
            .init(flag: "--top-logprobs", label: "Top logprobs", kind: .integer),
            .init(flag: "--allow-external-scorer", label: "Authorize scorer", kind: .boolean),
            .init(flag: "--log-responses", label: "Include responses", kind: .boolean),
            .init(flag: "--dry-run", label: "Plan only", kind: .boolean),
            .init(flag: "--checkpoint", label: "Checkpoint", kind: .file),
            .init(flag: "--resume", label: "Resume", kind: .boolean),
            .init(flag: "--case-trial-limit", label: "Row limit", kind: .integer),
            .init(flag: "--output", label: "Report", kind: .file),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--output", optional: true)
    )

    public static let evaluationPromote = MereRunCommandCapability(
        id: "eval.promote",
        command: ["eval", "promote"],
        title: "Promote evaluated artifact",
        summary: "Create a content-addressed receipt for a complete, gate-passing evaluation report.",
        arguments: [
            .init(name: "report", label: "Evaluation report", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Promotion receipt", kind: .file),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text, fileExtension: "json", flag: "--output", optional: true)
    )

    public static let modelStorage = MereRunCommandCapability(
        id: "model.storage",
        command: ["model", "storage"],
        title: "Model storage",
        summary: "Inspect physical storage, sharing, and reclaimable bytes.",
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelGarbageCollect = MereRunCommandCapability(
        id: "model.gc",
        command: ["model", "gc"],
        title: "Model storage cleanup",
        summary: "Plan or execute safe cleanup of unreferenced payloads and partial downloads.",
        options: [
            .init(flag: "--force", label: "Execute", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelRuntimeGet = MereRunCommandCapability(
        id: "model.runtime.get",
        command: ["model", "runtime", "get"],
        title: "Read runtime policy",
        summary: "Read typed API residency and default generation settings.",
        arguments: [
            .init(name: "model", label: "Model or alias", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelRuntimeSet = MereRunCommandCapability(
        id: "model.runtime.set",
        command: ["model", "runtime", "set"],
        title: "Set runtime policy",
        summary: "Update typed API residency and default generation settings.",
        arguments: [
            .init(name: "model", label: "Model or alias", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--alias", label: "Alias", kind: .string),
            .init(flag: "--clear-alias", label: "Clear alias", kind: .boolean),
            .init(flag: "--pinned", label: "Pin model", kind: .boolean),
            .init(flag: "--unpinned", label: "Unpin model", kind: .boolean),
            .init(flag: "--ttl-seconds", label: "TTL", kind: .integer),
            .init(flag: "--clear-ttl", label: "Clear TTL", kind: .boolean),
            .init(flag: "--max-context-tokens", label: "Max context", kind: .integer),
            .init(flag: "--clear-max-context-tokens", label: "Clear max context", kind: .boolean),
            .init(flag: "--max-tokens", label: "Max output", kind: .integer),
            .init(flag: "--clear-max-tokens", label: "Clear max output", kind: .boolean),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--clear-temperature", label: "Clear temperature", kind: .boolean),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--clear-top-p", label: "Clear top-p", kind: .boolean),
            .init(flag: "--min-p", label: "Min-p", kind: .number),
            .init(flag: "--clear-min-p", label: "Clear min-p", kind: .boolean),
            .init(
                flag: "--engine",
                label: "Engine",
                kind: .choice,
                choices: [
                    "text-code", "text-chat-klein", "text-chat-gemma4", "text-chat-q36",
                    "text-chat-q35", "text-chat-laguna", "text-chat-lfm2",
                    "text-chat-deepseek-v4-flash"
                ]
            ),
            .init(flag: "--clear-engine", label: "Clear engine", kind: .boolean),
            .init(
                flag: "--kv-cache-mode",
                label: "KV cache",
                kind: .choice,
                choices: ["default", "affine4", "affine8", "polar2", "auto"]
            ),
            .init(flag: "--clear-kv-cache-mode", label: "Clear KV cache", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let setup = MereRunCommandCapability(
        id: "setup",
        command: ["setup"],
        title: "Setup path",
        summary: "Plan or run guided, BYOA, or manual local setup.",
        options: [
            .init(flag: "--mode", label: "Mode", kind: .choice, choices: ["agent", "byoa", "manual"]),
            .init(flag: "--agent-model", label: "Agent tier", kind: .choice, choices: ["small", "tier", "premier"]),
            .init(flag: "--install", label: "Install", kind: .boolean),
            .init(flag: "--start", label: "Start", kind: .boolean),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--host", label: "API host", kind: .string),
            .init(flag: "--port", label: "API port", kind: .integer),
            .init(flag: "--pi-path", label: "Pi executable", kind: .file),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let agentOnboard = MereRunCommandCapability(
        id: "agent.onboard",
        command: ["agent", "onboard"],
        title: "Agent onboarding",
        summary: "Check readiness and prepare the optional Pi integration.",
        options: [
            .init(flag: "--pull-recommended", label: "Pull recommended", kind: .boolean),
            .init(flag: "--accept-model-license", label: "Accept model terms", kind: .boolean),
            .init(flag: "--install-pi", label: "Install Pi", kind: .boolean),
            .init(flag: "--configure-pi", label: "Configure Pi", kind: .boolean),
            .init(flag: "--host", label: "API host", kind: .string),
            .init(flag: "--port", label: "API port", kind: .integer),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let agentStatus = MereRunCommandCapability(
        id: "agent.status",
        command: ["agent", "status"],
        title: "Agent status",
        summary: "Inspect Pi, provider, machine, and local agent-model readiness.",
        options: [
            .init(flag: "--pi-path", label: "Pi executable", kind: .file),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let agentInstallPi = MereRunCommandCapability(
        id: "agent.install-pi",
        command: ["agent", "install-pi"],
        title: "Install Pi",
        summary: "Install or replace the optional Pi setup agent.",
        options: [
            .init(flag: "--force", label: "Force reinstall", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let agentStart = MereRunCommandCapability(
        id: "agent.start",
        command: ["agent", "start"],
        title: "Start setup agent",
        summary: "Start a guided Pi session against the local API.",
        options: [
            .init(flag: "--host", label: "API host", kind: .string),
            .init(flag: "--port", label: "API port", kind: .integer),
            .init(flag: "--pi-path", label: "Pi executable", kind: .file),
            .init(flag: "--prompt", label: "Prompt", kind: .string),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--skip-server", label: "Skip server", kind: .boolean),
            .init(flag: "--allow-unsupported", label: "Allow unsupported", kind: .boolean),
            .init(flag: "--no-bootstrap", label: "No bootstrap", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .service)
    )

    public static let modelList = MereRunCommandCapability(
        id: "model.list",
        command: ["model", "list"],
        title: "List models",
        summary: "List managed model install state.",
        options: [
            .init(flag: "--measure-sizes", label: "Measure referenced sizes", kind: .boolean),
            .init(flag: "--json", label: "JSON inventory", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelCapabilities = MereRunCommandCapability(
        id: "model.capabilities",
        command: ["model", "capabilities"],
        title: "Model capabilities",
        summary: "Inspect hardware support and setup recommendations.",
        options: [
            .init(flag: "--all", label: "Include unsupported", kind: .boolean),
            .init(flag: "--recommended", label: "Recommended only", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelPull = MereRunCommandCapability(
        id: "model.pull",
        command: ["model", "pull"],
        title: "Pull model",
        summary: "Preflight or install one or all managed models.",
        arguments: [
            .init(name: "target", label: "Model", kind: .string, required: false)
        ],
        options: [
            .init(flag: "--all", label: "All models", kind: .boolean),
            .init(flag: "--force", label: "Force download", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean),
            .init(flag: "--allow-unsupported", label: "Allow unsupported", kind: .boolean),
            .init(flag: "--accept-model-license", label: "Accept model terms", kind: .boolean),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelInfo = MereRunCommandCapability(
        id: "model.info",
        command: ["model", "info"],
        title: "Model info",
        summary: "Inspect a model manifest and resolved components.",
        arguments: [
            .init(name: "target", label: "Model", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--components", label: "Components", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelRemove = MereRunCommandCapability(
        id: "model.remove",
        command: ["model", "remove"],
        title: "Remove model",
        summary: "Remove a managed model with optional cache preservation and receipt.",
        arguments: [
            .init(name: "target", label: "Model", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--force", label: "Force", kind: .boolean),
            .init(flag: "--keep-cache", label: "Keep cache", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelRepairManifests = MereRunCommandCapability(
        id: "model.repair-manifests",
        command: ["model", "repair-manifests"],
        title: "Repair manifests",
        summary: "Restore missing manifests for known local models.",
        options: [
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelOptimize = MereRunCommandCapability(
        id: "model.optimize",
        command: ["model", "optimize"],
        title: "Optimize model",
        summary: "Build inference-only caches for a supported installed model.",
        arguments: [
            .init(name: "target", label: "Model or local root", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--force", label: "Replace cache", kind: .boolean),
            .init(flag: "--output", label: "Standalone checkpoint", kind: .directory),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text, flag: "--output", optional: true)
    )

    public static let modelBenchmarkQ36MTP = MereRunCommandCapability(
        id: "model.benchmark.q36-mtp",
        command: ["model", "benchmark", "q36-mtp"],
        title: "Qwen-family MTP benchmark",
        summary: "Run prompt, decode-length, and temperature matrices for Qwen-family MTP.",
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--prompt", label: "Prompt", kind: .string),
            .init(flag: "--prompt-file", label: "Prompt file", kind: .file),
            .init(flag: "--prompt-repeat", label: "Prompt repeat", kind: .integer),
            .init(flag: "--prompt-repeat-values", label: "Prompt repeat matrix", kind: .string),
            .init(flag: "--decode-tokens", label: "Decode tokens", kind: .integer),
            .init(flag: "--decode-token-values", label: "Decode matrix", kind: .string),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--temperature-values", label: "Temperature matrix", kind: .string),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(flag: "--mtp-block-size", label: "MTP block", kind: .integer),
            .init(flag: "--forced-mtp-min-prompt-tokens", label: "Forced MTP threshold", kind: .integer),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkLagunaDFlash = MereRunCommandCapability(
        id: "model.benchmark.laguna-dflash",
        command: ["model", "benchmark", "laguna-dflash"],
        title: "Laguna DFlash benchmark",
        summary: "Compare target-only, fixed DFlash, and adaptive Laguna decode in one resident process.",
        options: [
            .init(flag: "--laguna-path", label: "Laguna path", kind: .directory, required: true),
            .init(flag: "--laguna-dflash-path", label: "DFlash path", kind: .directory, required: true),
            .init(flag: "--decode-token-values", label: "Decode matrix", kind: .string),
            .init(flag: "--repetitions", label: "Repetitions", kind: .integer),
            .init(flag: "--laguna-dflash-tokens", label: "Speculative tokens", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--top-k", label: "Top-k", kind: .integer),
            .init(flag: "--min-p", label: "Min-p", kind: .number),
            .init(flag: "--prompt", label: "Prompt", kind: .string),
            .init(flag: "--prompt-file", label: "Prompt file", kind: .file),
            .init(
                flag: "--fixture",
                label: "Fixture",
                kind: .choice,
                choices: ["deterministic-prose", "grounded-email", "code-completion"]
            ),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(flag: "--concurrency-values", label: "Concurrency matrix", kind: .string),
            .init(flag: "--warmup-repetitions", label: "Warmups", kind: .integer),
            .init(flag: "--mixed-fixtures", label: "Mixed fixtures", kind: .boolean),
            .init(flag: "--include-automatic", label: "Adaptive routing", kind: .boolean),
            .init(flag: "--log-responses", label: "Log responses", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkParakeetCoreML = MereRunCommandCapability(
        id: "model.benchmark.parakeet-coreml",
        command: ["model", "benchmark", "parakeet-coreml"],
        title: "Parakeet Core ML benchmark",
        summary: "Measure the prepared Parakeet Core ML pipeline in one resident release process.",
        arguments: [
            .init(name: "audio", label: "Audio", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--artifact", label: "Core ML artifact", kind: .directory, required: true),
            .init(flag: "--warmups", label: "Warmups", kind: .integer),
            .init(flag: "--repetitions", label: "Repetitions", kind: .integer),
            .init(flag: "--language", label: "Language", kind: .string),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let speechSynthesize = MereRunCommandCapability(
        id: "speech.synthesize",
        command: ["speech", "synthesize"],
        title: "Synthesize speech",
        summary: "Create styled or cloned speech, including streaming output.",
        arguments: [
            .init(name: "text", label: "Text", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Output", kind: .file, required: true, group: Group.output, tier: .essential),
            .init(
                flag: "--model", label: "Model", kind: .string,
                defaultValue: "speech-tts-qwen3-nano", group: Group.modelAndAdapters, tier: .essential
            ),
            .init(
                flag: "--voice", label: "Voice", kind: .string,
                defaultValue: "A calm female voice with clear pronunciation", group: Group.prompt, tier: .essential
            ),
            .init(
                flag: "--mode", label: "Mode", kind: .choice, choices: ["style", "clone"],
                defaultValue: "style", group: Group.inputs, tier: .standard
            ),
            .init(flag: "--profile", label: "Profile", kind: .string, group: Group.inputs, tier: .standard),
            .init(flag: "--ref-audio", label: "Reference audio", kind: .file, group: Group.inputs, tier: .standard),
            .init(
                flag: "--ref-text", label: "Reference text", kind: .string,
                group: Group.inputs, tier: .standard, dependsOn: "--ref-audio"
            ),
            .init(flag: "--language", label: "Language", kind: .string, defaultValue: "auto", group: Group.prompt, tier: .standard),
            .init(
                flag: "--save-profile", label: "Save profile", kind: .string,
                group: Group.inputs, tier: .expert, dependsOn: "--ref-audio"
            ),
            .init(
                flag: "--temperature", label: "Temperature", kind: .number,
                defaultValue: "0.6", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 2, step: 0.05)
            ),
            .init(flag: "--stream", label: "Stream", kind: .boolean, group: Group.output, tier: .expert),
            .init(
                flag: "--stream-chunk-tokens", label: "Chunk tokens", kind: .integer,
                defaultValue: "25", group: Group.output, tier: .expert, range: .init(min: 1, max: 500, step: 1), dependsOn: "--stream"
            ),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            progressJSONOption,
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output")
    )

    public static let speechTranscribe = MereRunCommandCapability(
        id: "speech.transcribe",
        command: ["speech", "transcribe"],
        title: "Transcribe speech",
        summary: "Transcribe files or raw streaming audio with optional JSONL events.",
        arguments: [
            .init(name: "audio", label: "Audio", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Output", kind: .file, group: Group.output, tier: .standard),
            .init(flag: "--model", label: "Model", kind: .string, group: Group.modelAndAdapters, tier: .standard),
            .init(
                flag: "--backend", label: "Backend", kind: .choice, choices: ["auto", "parakeet", "qwen"],
                defaultValue: "auto", group: Group.modelAndAdapters, tier: .essential
            ),
            .init(
                flag: "--provider", label: "Parakeet provider", kind: .choice, choices: ["mlx", "coreml"],
                defaultValue: "mlx", group: Group.modelAndAdapters, tier: .standard
            ),
            .init(
                flag: "--coreml-encoder", label: "Core ML artifact", kind: .directory,
                group: Group.modelAndAdapters, tier: .expert, dependsOn: "--provider"
            ),
            .init(
                flag: "--task", label: "Task", kind: .choice, choices: ["transcribe", "translate"],
                defaultValue: "transcribe", group: Group.prompt, tier: .essential
            ),
            .init(flag: "--language", label: "Language", kind: .string, group: Group.prompt, tier: .standard),
            .init(
                flag: "--max-tokens", label: "Max tokens", kind: .integer,
                defaultValue: "448", group: Group.sampling, tier: .standard, range: .init(min: 1, max: 8_192, step: 1)
            ),
            .init(flag: "--stream", label: "Stream", kind: .boolean, group: Group.run, tier: .expert),
            .init(
                flag: "--stream-chunk-ms", label: "Feed interval", kind: .integer,
                defaultValue: "200", group: Group.run, tier: .expert, range: .init(min: 10, max: 5_000, step: 10), dependsOn: "--stream"
            ),
            .init(
                flag: "--stream-decode-ms", label: "Decode interval", kind: .integer,
                defaultValue: "2000", group: Group.run, tier: .expert, range: .init(min: 100, max: 10_000, step: 100), dependsOn: "--stream"
            ),
            .init(flag: "--input-format", label: "Input format", kind: .string, group: Group.inputs, tier: .expert, dependsOn: "--stream"),
            // Raw stdin protocol v1 accepts exactly 16 kHz, so the range pins
            // the single value a shell may send rather than a span.
            .init(
                flag: "--sample-rate", label: "Sample rate", kind: .integer,
                group: Group.inputs, tier: .expert, range: .init(min: 16_000, max: 16_000, step: 1),
                dependsOn: "--stream"
            ),
            .init(flag: "--jsonl", label: "JSON Lines", kind: .boolean, group: Group.output, tier: .expert, dependsOn: "--stream"),
            .init(flag: "--no-timestamps", label: "No timestamps", kind: .boolean, group: Group.output, tier: .standard),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            receiptOption
        ],
        output: .init(kind: .text, fileExtension: "txt", flag: "--output", optional: true)
    )

    public static let speechDiarize = MereRunCommandCapability(
        id: "speech.diarize",
        command: ["speech", "diarize"],
        title: "Diarize speech",
        summary: "Identify speaker activity in a local audio file as JSON or RTTM.",
        arguments: [
            .init(name: "audio", label: "Audio", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--format", label: "Format", kind: .choice, choices: ["json", "rttm"]),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--threshold", label: "Threshold", kind: .number),
            .init(flag: "--min-duration", label: "Minimum duration", kind: .number),
            .init(flag: "--merge-gap", label: "Merge gap", kind: .number),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .text, flag: "--output", optional: true)
    )

    public static let speechProfileList = MereRunCommandCapability(
        id: "speech.profile.list",
        command: ["speech", "profile", "list"],
        title: "Voice profiles",
        summary: "List saved voice-cloning profiles.",
        options: [],
        output: .init(kind: .text)
    )

    public static let speechProfileCreate = MereRunCommandCapability(
        id: "speech.profile.create",
        command: ["speech", "profile", "create"],
        title: "Create voice profile",
        summary: "Create a reusable voice-cloning profile.",
        options: [
            .init(flag: "--name", label: "Name", kind: .string, required: true),
            .init(flag: "--audio", label: "Audio", kind: .file, required: true),
            .init(flag: "--text", label: "Transcript", kind: .string),
            .init(flag: "--language", label: "Language", kind: .string),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let speechProfileDelete = MereRunCommandCapability(
        id: "speech.profile.delete",
        command: ["speech", "profile", "delete"],
        title: "Delete voice profile",
        summary: "Delete one saved voice profile.",
        options: [
            .init(flag: "--id", label: "Profile id", kind: .string, required: true)
        ],
        output: .init(kind: .text)
    )

    public static let sfxGenerate = MereRunCommandCapability(
        id: "sfx.generate",
        command: ["sfx", "generate"],
        title: "Generate sound effect",
        summary: "Generate a Woosh or MMAudio sound effect from text.",
        arguments: [
            .init(name: "prompt", label: "Prompt", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string, group: Group.prompt, tier: .standard),
            .init(flag: "--output", label: "Output", kind: .file, group: Group.output, tier: .standard),
            .init(
                flag: "--model", label: "Model", kind: .string,
                defaultValue: "sfx-woosh-dflow", group: Group.modelAndAdapters, tier: .essential
            ),
            .init(
                flag: "--duration", label: "Duration", kind: .number,
                group: Group.sampling, tier: .essential, range: .init(min: 0.5, max: 30, step: 0.5)
            ),
            .init(
                flag: "--steps", label: "Steps", kind: .integer,
                group: Group.sampling, tier: .standard, range: .init(min: 1, max: 100, step: 1)
            ),
            .init(
                flag: "--cfg", label: "CFG", kind: .number,
                defaultValue: "4.5", group: Group.sampling, tier: .standard, range: .init(min: 0, max: 20, step: 0.1)
            ),
            .init(flag: "--seed", label: "Seed", kind: .integer, group: Group.sampling, tier: .essential, range: .init(min: 0, step: 1)),
            .init(flag: "--renoise", label: "Renoise", kind: .string, group: Group.sampling, tier: .expert),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean, group: Group.run, tier: .expert),
            progressJSONOption,
            receiptOption
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output")
    )

    public static let sfxVideoGenerate = MereRunCommandCapability(
        id: "sfx.video.generate",
        command: ["sfx", "video", "generate"],
        title: "Video foley",
        summary: "Generate synchronized sound effects from video conditioning.",
        arguments: [
            .init(name: "prompt", label: "Prompt", kind: .string, required: true),
            .init(name: "input", label: "Video or features", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--negative-prompt", label: "Negative prompt", kind: .string),
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--synchformer-model", label: "Synchformer", kind: .string),
            .init(flag: "--duration", label: "Duration", kind: .number),
            .init(flag: "--steps", label: "Steps", kind: .integer),
            .init(flag: "--cfg", label: "CFG", kind: .number),
            .init(flag: "--seed", label: "Seed", kind: .integer),
            .init(flag: "--renoise", label: "Renoise", kind: .string),
            .init(flag: "--sync-batch-size", label: "Sync batch", kind: .integer),
            .init(flag: "--clip-batch-size", label: "CLIP batch", kind: .integer),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output")
    )

    public static let sfxAEEncode = MereRunCommandCapability(
        id: "sfx.ae.encode",
        command: ["sfx", "ae", "encode"],
        title: "Encode SFX latents",
        summary: "Encode audio into Woosh latent arrays.",
        arguments: [.init(name: "input", label: "Audio", kind: .file, required: true)],
        options: [
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "npy", flag: "--output")
    )

    public static let sfxAEDecode = MereRunCommandCapability(
        id: "sfx.ae.decode",
        command: ["sfx", "ae", "decode"],
        title: "Decode SFX latents",
        summary: "Decode Woosh latent arrays into audio.",
        arguments: [.init(name: "input", label: "Latents", kind: .file, required: true)],
        options: [
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "wav", flag: "--output")
    )

    public static let sfxCLAPScore = MereRunCommandCapability(
        id: "sfx.clap.score",
        command: ["sfx", "clap", "score"],
        title: "CLAP score",
        summary: "Score semantic alignment between a prompt and audio.",
        arguments: [
            .init(name: "prompt", label: "Prompt", kind: .string, required: true),
            .init(name: "audio", label: "Audio", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let sfxConditionText = MereRunCommandCapability(
        id: "sfx.condition.text",
        command: ["sfx", "condition", "text"],
        title: "SFX text conditioning",
        summary: "Export text-conditioning tensors for Woosh.",
        arguments: [.init(name: "prompt", label: "Prompt", kind: .string, required: true)],
        options: [
            .init(flag: "--output", label: "Output", kind: .file),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )

    public static let pluginList = MereRunCommandCapability(
        id: "plugin.list",
        command: ["plugin", "list"],
        title: "List plugins",
        summary: "Inspect the official or an overridden plugin catalog.",
        options: [
            .init(flag: "--catalog-url", label: "Catalog", kind: .string),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let pluginInstall = MereRunCommandCapability(
        id: "plugin.install",
        command: ["plugin", "install"],
        title: "Install plugin",
        summary: "Plan or execute an official plugin installation.",
        arguments: [.init(name: "id", label: "Plugin id", kind: .string, required: true)],
        options: [
            .init(flag: "--catalog-url", label: "Catalog", kind: .string),
            .init(flag: "--channel", label: "Channel", kind: .string),
            .init(flag: "--yes", label: "Execute", kind: .boolean),
            .init(flag: "--force", label: "Force", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let pluginDoctor = MereRunCommandCapability(
        id: "plugin.doctor",
        command: ["plugin", "doctor"],
        title: "Plugin doctor",
        summary: "Run a companion plugin's health check.",
        arguments: [.init(name: "id", label: "Plugin id", kind: .string, required: true)],
        options: [
            .init(flag: "--catalog-url", label: "Catalog", kind: .string)
        ],
        output: .init(kind: .text)
    )

    public static let openWebUIQuickstart = MereRunCommandCapability(
        id: "open-webui.quickstart",
        command: ["open-webui", "quickstart"],
        title: "Open WebUI quickstart",
        summary: "Launch and configure Open WebUI against the local API and model suite.",
        options: [
            .init(flag: "--host", label: "API host", kind: .string),
            .init(flag: "--port", label: "API port", kind: .integer),
            .init(
                flag: "--engine",
                label: "Engine",
                kind: .choice,
                choices: [
                    "text-chat-q36", "text-code", "text-chat-klein", "text-chat-gemma4",
                    "text-chat-laguna", "text-chat-lfm2", "text-chat-deepseek-v4-flash"
                ]
            ),
            .init(flag: "--webui-host", label: "WebUI host", kind: .string),
            .init(flag: "--webui-port", label: "WebUI port", kind: .integer),
            .init(flag: "--container-name", label: "Container", kind: .string),
            .init(flag: "--volume-name", label: "Volume", kind: .string),
            .init(flag: "--image", label: "Docker image", kind: .string),
            .init(flag: "--api-key", label: "API key", kind: .string),
            .init(flag: "--text-model", label: "Text model", kind: .string),
            .init(flag: "--vision-model", label: "Vision model", kind: .string),
            .init(flag: "--embedding-model", label: "Embedding model", kind: .string),
            .init(flag: "--image-model", label: "Image model", kind: .string),
            .init(flag: "--tts-model", label: "TTS model", kind: .string),
            .init(flag: "--stt-model", label: "STT model", kind: .string),
            .init(flag: "--tts-format", label: "TTS format", kind: .string),
            .init(flag: "--admin-email", label: "Admin email", kind: .string),
            .init(flag: "--admin-password", label: "Admin password", kind: .string),
            .init(flag: "--wait-seconds", label: "Health wait", kind: .integer),
            .init(flag: "--pull", label: "Pull models", kind: .boolean),
            .init(flag: "--accept-model-license", label: "Accept model terms", kind: .boolean),
            .init(flag: "--skip-server", label: "Skip API server", kind: .boolean),
            .init(flag: "--skip-docker", label: "Skip Docker", kind: .boolean),
            .init(flag: "--skip-configure", label: "Skip configure", kind: .boolean),
            .init(flag: "--reset", label: "Reset", kind: .boolean),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean)
        ],
        output: .init(kind: .service)
    )

    public static let apiServe = MereRunCommandCapability(
        id: "api.serve",
        command: ["api", "serve"],
        title: "API server",
        summary: "Serve installed models through OpenAI-compatible local APIs.",
        options: [
            .init(flag: "--port", label: "Port", kind: .integer),
            .init(flag: "--host", label: "Host", kind: .string),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(
                flag: "--engine",
                label: "Engine",
                kind: .choice,
                choices: [
                    "text-chat-q36", "text-code", "text-chat-klein", "text-chat-gemma4",
                    "text-chat-laguna", "text-chat-lfm2", "text-chat-deepseek-v4-flash"
                ]
            ),
            .init(flag: "--lora", label: "Adapter", kind: .string),
            .init(flag: "--api-key", label: "API key", kind: .string),
            .init(flag: "--rate-limit-per-minute", label: "Rate limit", kind: .integer),
            .init(flag: "--max-active-requests", label: "Active requests", kind: .integer),
            .init(flag: "--memory-guard", label: "Memory guard", kind: .choice, choices: ["off", "safe", "balanced", "aggressive", "custom"]),
            .init(flag: "--memory-guard-custom-ceiling-gb", label: "Memory ceiling", kind: .number),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(flag: "--kv-bits", label: "KV bits", kind: .number),
            .init(flag: "--kv-quant-scheme", label: "KV scheme", kind: .string),
            .init(flag: "--kv-group-size", label: "KV group", kind: .integer),
            .init(flag: "--quantized-kv-start", label: "KV start", kind: .integer),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .service)
    )

    public static let guide = MereRunCommandCapability(
        id: "guide",
        command: ["guide"],
        title: "Offline guides",
        summary: "List or read CLI-owned offline workflow guides.",
        arguments: [
            .init(name: "command-path", label: "Command path", kind: .string, required: false)
        ],
        options: [
            .init(flag: "--list", label: "List topics", kind: .boolean),
            .init(flag: "--list-models", label: "List model guides", kind: .boolean),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--json", label: "JSON", kind: .boolean),
            .init(flag: "--markdown", label: "Markdown index", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let configSet = MereRunCommandCapability(
        id: "config.set",
        command: ["config", "set"],
        title: "Set configuration",
        summary: "Persist a supported configuration value, including a secret-safe environment source.",
        arguments: [
            .init(name: "key", label: "Key", kind: .string, required: true),
            .init(name: "value", label: "Value", kind: .string, required: false)
        ],
        options: [
            .init(flag: "--from-env", label: "Environment variable", kind: .string)
        ],
        output: .init(kind: .text)
    )

    public static let configGet = MereRunCommandCapability(
        id: "config.get",
        command: ["config", "get"],
        title: "Read configuration",
        summary: "Read a persisted configuration value with secrets masked by default.",
        arguments: [
            .init(name: "key", label: "Key", kind: .string, required: true)
        ],
        options: [
            .init(flag: "--reveal", label: "Reveal secret", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let configUnset = MereRunCommandCapability(
        id: "config.unset",
        command: ["config", "unset"],
        title: "Unset configuration",
        summary: "Remove a persisted configuration value.",
        arguments: [
            .init(name: "key", label: "Key", kind: .string, required: true)
        ],
        options: [],
        output: .init(kind: .text)
    )

    // MARK: - Geospatial

    public static let geoFlood = MereRunCommandCapability(
        id: "geo.flood",
        command: ["geo", "flood"],
        title: "Flood inference",
        summary: "Run native TerraMind Flood tile inference with MLX on Apple Silicon.",
        arguments: [
            .init(name: "input", label: "Tile safetensors", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Logits output", kind: .file, required: true),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )

    public static let geoFire = MereRunCommandCapability(
        id: "geo.fire",
        command: ["geo", "fire"],
        title: "Fire inference",
        summary: "Run native TerraMind Fire tile inference with MLX on Apple Silicon.",
        arguments: [
            .init(name: "input", label: "Tile safetensors", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Logits output", kind: .file, required: true),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )

    public static let geoTessera = MereRunCommandCapability(
        id: "geo.tessera",
        command: ["geo", "tessera"],
        title: "TESSERA embeddings",
        summary: "Encode local Sentinel-1/2 time series with a native TESSERA v2 student.",
        arguments: [
            .init(name: "input", label: "Observation safetensors", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Embedding output", kind: .file, required: true),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--dimensions", label: "Dimensions", kind: .integer),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )

    public static let geoOlmoEarth = MereRunCommandCapability(
        id: "geo.olmoearth",
        command: ["geo", "olmoearth"],
        title: "OlmoEarth embeddings",
        summary: "Encode multisensor Earth observations with native OlmoEarth v1.2.",
        arguments: [
            .init(name: "input", label: "Observation safetensors", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--output", label: "Embedding output", kind: .file, required: true),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--patch-size", label: "Patch size", kind: .integer),
            .init(flag: "--input-resolution", label: "Input resolution", kind: .number),
            .init(flag: "--include-tokens", label: "Include tokens", kind: .boolean),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .file, fileExtension: "safetensors", flag: "--output")
    )

    // MARK: - Model store locations

    public static let modelLocationList = MereRunCommandCapability(
        id: "model.location.list",
        command: ["model", "location", "list"],
        title: "List model locations",
        summary: "List the writable store, read-only search roots, and explicit bindings.",
        options: [
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelLocationAdd = MereRunCommandCapability(
        id: "model.location.add",
        command: ["model", "location", "add"],
        title: "Add search root",
        summary: "Register a read-only root containing directories named for canonical model ids.",
        arguments: [
            .init(name: "path", label: "Search root", kind: .directory, required: true)
        ],
        options: [],
        output: .init(kind: .text)
    )

    public static let modelLocationRemove = MereRunCommandCapability(
        id: "model.location.remove",
        command: ["model", "location", "remove"],
        title: "Remove search root",
        summary: "Unregister a read-only search root without deleting its files.",
        arguments: [
            .init(name: "path", label: "Search root", kind: .directory, required: true)
        ],
        options: [],
        output: .init(kind: .text)
    )

    public static let modelLocationBind = MereRunCommandCapability(
        id: "model.location.bind",
        command: ["model", "location", "bind"],
        title: "Bind model directory",
        summary: "Bind a canonical model id to an arbitrary read-only directory.",
        arguments: [
            .init(name: "modelID", label: "Model id", kind: .string, required: true),
            .init(name: "path", label: "Model directory", kind: .directory, required: true)
        ],
        options: [
            .init(
                flag: "--accept-model-license",
                label: "Accept model license",
                kind: .boolean
            )
        ],
        output: .init(kind: .text)
    )

    public static let modelLocationUnbind = MereRunCommandCapability(
        id: "model.location.unbind",
        command: ["model", "location", "unbind"],
        title: "Unbind model directory",
        summary: "Remove explicit bindings without deleting model files.",
        arguments: [
            .init(name: "modelID", label: "Model id", kind: .string, required: true),
            .init(name: "path", label: "Model directory", kind: .directory, required: false)
        ],
        options: [],
        output: .init(kind: .text)
    )

    // MARK: - Model benchmarks

    public static let modelBenchmarkChat = MereRunCommandCapability(
        id: "model.benchmark.chat",
        command: ["model", "benchmark", "chat"],
        title: "Chat benchmark",
        summary: "Run a small grounded-chat evaluation slice against local assistant models.",
        options: [
            .init(flag: "--models", label: "Models", kind: .string),
            .init(flag: "--suite", label: "Suite", kind: .choice, choices: ["mere-chat-slice"]),
            .init(flag: "--cases", label: "Cases", kind: .file),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--top-k", label: "Top-k", kind: .integer),
            .init(flag: "--min-p", label: "Min-p", kind: .number),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(flag: "--concurrency", label: "Concurrency", kind: .integer),
            .init(flag: "--laguna-path", label: "Laguna path", kind: .directory),
            .init(flag: "--laguna-dflash-path", label: "DFlash path", kind: .directory),
            .init(flag: "--laguna-dflash-tokens", label: "Speculative tokens", kind: .integer),
            .init(flag: "--laguna-dflash-min-tokens", label: "Minimum speculative tokens", kind: .integer),
            .init(
                flag: "--laguna-dflash-routing",
                label: "DFlash routing",
                kind: .choice,
                choices: ["automatic", "target-only", "dflash"]
            ),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--log-responses", label: "Log responses", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkCode = MereRunCommandCapability(
        id: "model.benchmark.code",
        command: ["model", "benchmark", "code"],
        title: "Code benchmark",
        summary: "Run a real coding-evaluation slice against local coding models.",
        options: [
            .init(flag: "--models", label: "Models", kind: .string),
            .init(flag: "--laguna-path", label: "Laguna path", kind: .directory),
            .init(flag: "--laguna-dflash-path", label: "Laguna DFlash path", kind: .directory),
            .init(flag: "--laguna-dflash-tokens", label: "Laguna DFlash tokens", kind: .integer),
            .init(
                flag: "--laguna-dflash-min-tokens",
                label: "Laguna DFlash minimum tokens",
                kind: .integer
            ),
            .init(flag: "--suite", label: "Suite", kind: .choice, choices: ["humaneval-slice"]),
            .init(flag: "--tasks", label: "Tasks", kind: .string),
            .init(flag: "--humaneval-file", label: "HumanEval file", kind: .file),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--top-k", label: "Top-k", kind: .integer),
            .init(flag: "--min-p", label: "Min-p", kind: .number),
            .init(flag: "--thinking", label: "Thinking", kind: .boolean),
            .init(flag: "--execution-timeout", label: "Execution timeout", kind: .number),
            .init(flag: "--python", label: "Python", kind: .string),
            .init(
                flag: "--sandbox",
                label: "Sandbox",
                kind: .choice,
                choices: ["auto", "macos-sandbox-exec", "bubblewrap", "none"]
            ),
            .init(flag: "--allow-code-execution", label: "Allow code execution", kind: .boolean),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkFused = MereRunCommandCapability(
        id: "model.benchmark.fused",
        command: ["model", "benchmark", "fused"],
        title: "Fused quality suite",
        summary: "Run the versioned Mere Lite or Mere Comprehensive fused quality suite.",
        options: [
            .init(flag: "--suite", label: "Suite", kind: .choice, choices: ["lite", "comprehensive"]),
            .init(flag: "--models", label: "Models", kind: .string),
            .init(flag: "--manifest", label: "Manifest", kind: .file),
            .init(flag: "--external-cases", label: "External cases", kind: .file),
            .init(flag: "--cases", label: "Cases", kind: .file),
            .init(flag: "--capabilities", label: "Capabilities", kind: .string),
            .init(flag: "--trials", label: "Trials", kind: .integer),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(
                flag: "--logprobs",
                label: "Logprobs",
                kind: .choice,
                choices: ["summary", "tokens", "top"]
            ),
            .init(flag: "--top-logprobs", label: "Top logprobs", kind: .integer),
            .init(
                flag: "--performance-lane",
                label: "Performance lane",
                kind: .choice,
                choices: ["none", "native"]
            ),
            .init(flag: "--execution-timeout", label: "Execution timeout", kind: .number),
            .init(flag: "--python", label: "Python", kind: .string),
            .init(
                flag: "--sandbox",
                label: "Sandbox",
                kind: .choice,
                choices: ["auto", "macos-sandbox-exec", "bubblewrap", "none"]
            ),
            .init(flag: "--allow-code-execution", label: "Allow code execution", kind: .boolean),
            .init(flag: "--log-responses", label: "Log responses", kind: .boolean),
            .init(flag: "--checkpoint", label: "Checkpoint", kind: .file),
            .init(flag: "--resume", label: "Resume", kind: .boolean),
            .init(flag: "--case-trial-limit", label: "Case trial limit", kind: .integer),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkFusedFixture = MereRunCommandCapability(
        id: "model.benchmark.fused-fixture",
        command: ["model", "benchmark", "fused-fixture"],
        title: "Fused fixture hashes",
        summary: "Stamp or verify normalized fused-benchmark JSONL fixture hashes.",
        arguments: [
            .init(name: "input", label: "Fixture JSONL", kind: .file, required: true)
        ],
        options: [
            .init(flag: "--check", label: "Verify only", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkVLM = MereRunCommandCapability(
        id: "model.benchmark.vlm",
        command: ["model", "benchmark", "vlm"],
        title: "Vision-language benchmark",
        summary: "Compare vision-language chat models on synthetic or lmms-eval datasets.",
        options: [
            .init(flag: "--models", label: "Models", kind: .string),
            .init(
                flag: "--dataset",
                label: "Dataset",
                kind: .choice,
                choices: [
                    "synthetic-vqa-v1", "mathvista-testmini", "mmmu-val",
                    "chartqa", "docvqa-val", "mme"
                ]
            ),
            .init(flag: "--lmms-tasks", label: "lmms-eval tasks", kind: .string),
            .init(flag: "--fixture-dir", label: "Fixture directory", kind: .directory),
            .init(flag: "--output-dir", label: "Output directory", kind: .directory),
            .init(flag: "--lmms-eval-root", label: "lmms-eval root", kind: .directory),
            .init(flag: "--lmms-eval-python", label: "lmms-eval Python", kind: .string),
            .init(flag: "--external-endpoint", label: "External endpoint", kind: .boolean),
            .init(flag: "--base-url", label: "Base URL", kind: .string),
            .init(flag: "--api-key", label: "API key", kind: .string),
            .init(flag: "--host", label: "Host", kind: .string),
            .init(flag: "--port", label: "Port", kind: .integer),
            .init(flag: "--limit", label: "Limit", kind: .string),
            .init(flag: "--log-samples", label: "Log samples", kind: .boolean),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text, flag: "--output-dir", optional: true)
    )

    public static let modelBenchmarkToolCalls = MereRunCommandCapability(
        id: "model.benchmark.tool-calls",
        command: ["model", "benchmark", "tool-calls"],
        title: "Tool-call benchmark",
        summary: "Run a small tool-call selection evaluation against local chat models.",
        options: [
            .init(flag: "--models", label: "Models", kind: .string),
            .init(flag: "--cases", label: "Cases", kind: .file),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--top-k", label: "Top-k", kind: .integer),
            .init(flag: "--min-p", label: "Min-p", kind: .number),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(flag: "--laguna-path", label: "Laguna path", kind: .directory),
            .init(flag: "--laguna-dflash-path", label: "DFlash path", kind: .directory),
            .init(flag: "--laguna-dflash-tokens", label: "Speculative tokens", kind: .integer),
            .init(flag: "--laguna-dflash-min-tokens", label: "Minimum speculative tokens", kind: .integer),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--log-responses", label: "Log responses", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkToolContinuations = MereRunCommandCapability(
        id: "model.benchmark.tool-continuations",
        command: ["model", "benchmark", "tool-continuations"],
        title: "Tool continuation benchmark",
        summary: "Evaluate Gemma 4 continuation after completed tool calls.",
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--context-size", label: "Context", kind: .integer),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--log-responses", label: "Log responses", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkGemma4KV = MereRunCommandCapability(
        id: "model.benchmark.gemma4-kv",
        command: ["model", "benchmark", "gemma4-kv"],
        title: "Gemma4 KV benchmark",
        summary: "Compare Gemma4 default KV cache decode against packed PolarKV.",
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--prompt", label: "Prompt", kind: .string),
            .init(flag: "--prompt-file", label: "Prompt file", kind: .file),
            .init(flag: "--prompt-repeat", label: "Prompt repeat", kind: .integer),
            .init(flag: "--prompt-repeat-values", label: "Prompt repeat matrix", kind: .string),
            .init(flag: "--decode-tokens", label: "Decode tokens", kind: .integer),
            .init(flag: "--decode-token-values", label: "Decode matrix", kind: .string),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkGemma4MTP = MereRunCommandCapability(
        id: "model.benchmark.gemma4-mtp",
        command: ["model", "benchmark", "gemma4-mtp"],
        title: "Gemma4 MTP benchmark",
        summary: "Compare Gemma4 serial decode against verified MTP speculative decode.",
        options: [
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--model-root", label: "Model root", kind: .directory),
            .init(flag: "--prompt", label: "Prompt", kind: .string),
            .init(flag: "--prompt-file", label: "Prompt file", kind: .file),
            .init(flag: "--prompt-repeat", label: "Prompt repeat", kind: .integer),
            .init(flag: "--prompt-repeat-values", label: "Prompt repeat matrix", kind: .string),
            .init(flag: "--decode-tokens", label: "Decode tokens", kind: .integer),
            .init(flag: "--decode-token-values", label: "Decode matrix", kind: .string),
            .init(flag: "--mtp-block-size", label: "MTP block size", kind: .integer),
            .init(flag: "--mtp-min-prompt-tokens", label: "MTP minimum prompt tokens", kind: .integer),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let modelBenchmarkAPIWorkload = MereRunCommandCapability(
        id: "model.benchmark.api-workload",
        command: ["model", "benchmark", "api-workload"],
        title: "API workload benchmark",
        summary: "Replay a chat workload against a running API server and measure cache counters.",
        options: [
            .init(flag: "--host", label: "Host", kind: .string),
            .init(flag: "--port", label: "Port", kind: .integer),
            .init(flag: "--api-key", label: "API key", kind: .string),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--workload-file", label: "Workload file", kind: .file),
            .init(flag: "--turns", label: "Turns", kind: .integer),
            .init(flag: "--shared-prefix-repeat", label: "Shared prefix repeat", kind: .integer),
            .init(flag: "--max-tokens", label: "Max tokens", kind: .integer),
            .init(flag: "--temperature", label: "Temperature", kind: .number),
            .init(flag: "--top-p", label: "Top-p", kind: .number),
            .init(flag: "--concurrency", label: "Concurrency", kind: .integer),
            .init(flag: "--timeout-seconds", label: "Timeout", kind: .number),
            .init(flag: "--dry-run", label: "Dry run", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    // MARK: - Plugins, configuration, and resident services

    public static let pluginInfo = MereRunCommandCapability(
        id: "plugin.info",
        command: ["plugin", "info"],
        title: "Plugin details",
        summary: "Show one plugin's catalog entry and install command.",
        arguments: [.init(name: "id", label: "Plugin id", kind: .string, required: true)],
        options: [
            .init(flag: "--catalog-url", label: "Catalog", kind: .string),
            .init(flag: "--channel", label: "Channel", kind: .string),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let pluginRun = MereRunCommandCapability(
        id: "plugin.run",
        command: ["plugin", "run"],
        title: "Run plugin",
        summary: "Run an installed plugin without changing PATH.",
        arguments: [
            .init(name: "entrypoint", label: "Entrypoint", kind: .string, required: true)
        ],
        options: [],
        output: .init(kind: .text)
    )

    public static let pluginRollback = MereRunCommandCapability(
        id: "plugin.rollback",
        command: ["plugin", "rollback"],
        title: "Roll back plugin",
        summary: "Restore a retained signed plugin bundle.",
        arguments: [.init(name: "id", label: "Plugin id", kind: .string, required: true)],
        options: [
            .init(flag: "--yes", label: "Activate", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let configList = MereRunCommandCapability(
        id: "config.list",
        command: ["config", "list"],
        title: "List configuration",
        summary: "Show all persisted configuration values with secrets masked.",
        options: [],
        output: .init(kind: .text)
    )

    public static let configPath = MereRunCommandCapability(
        id: "config.path",
        command: ["config", "path"],
        title: "Configuration path",
        summary: "Print the path of the persisted configuration file.",
        options: [],
        output: .init(kind: .text)
    )

    public static let speechListen = MereRunCommandCapability(
        id: "speech.listen",
        command: ["speech", "listen"],
        title: "Live transcription",
        summary: "Transcribe a macOS microphone with live Qwen ASR.",
        options: [
            .init(flag: "--device", label: "Input device", kind: .string),
            .init(flag: "--list-devices", label: "List devices", kind: .boolean),
            .init(flag: "--language", label: "Language", kind: .string),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--decode-ms", label: "Decode window", kind: .integer),
            .init(flag: "--silence-ms", label: "Silence window", kind: .integer),
            .init(flag: "--quiet", label: "Quiet", kind: .boolean),
            .init(flag: "--jsonl", label: "JSONL", kind: .boolean)
        ],
        output: .init(kind: .text)
    )

    public static let visionServe = MereRunCommandCapability(
        id: "vision.serve",
        command: ["vision", "serve"],
        title: "Vision grounding server",
        summary: "Serve resident, binary-frame vision grounding over HTTP.",
        options: [
            .init(flag: "--host", label: "Host", kind: .string),
            .init(flag: "--port", label: "Port", kind: .integer),
            .init(flag: "--model", label: "Model", kind: .string),
            .init(flag: "--api-key", label: "API key", kind: .string),
            .init(flag: "--max-frame-bytes", label: "Max frame bytes", kind: .integer),
            .init(flag: "--max-batch-size", label: "Max batch size", kind: .integer),
            .init(flag: "--max-batch-bytes", label: "Max batch bytes", kind: .integer),
            .init(flag: "--preflight", label: "Preflight", kind: .boolean),
            .init(flag: "--json", label: "JSON", kind: .boolean)
        ],
        output: .init(kind: .service)
    )
}
