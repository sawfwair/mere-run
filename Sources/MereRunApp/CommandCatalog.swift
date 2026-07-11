import Foundation
import UniformTypeIdentifiers

enum CommandCategory: String, CaseIterable, Identifiable {
    case setup = "Setup"
    case models = "Models"
    case image = "Image"
    case text = "Text"
    case speech = "Speech"
    case vision = "Vision"
    case media = "Music & Video"
    case sfx = "Sound FX"
    case server = "Server"
    case custom = "Custom"

    var id: String { rawValue }
}

enum CommandTemplateID: String, CaseIterable {
    case setup
    case agentOnboard
    case agentInstallPi
    case agentStart
    case modelList
    case modelCapabilities
    case modelPull
    case modelInfo
    case modelRemove
    case modelRepairManifests
    case imageGenerate
    case imageTrainLoRA
    case imageValidate
    case textChat
    case textCode
    case textEmbed
    case textAnonymize
    case speechSynthesize
    case speechTranscribe
    case speechProfileList
    case speechProfileCreate
    case speechProfileDelete
    case visionInspect
    case visionCaption
    case visionOCR
    case visionGround
    case visionSegment
    case visionTrack
    case visionTrackLive
    case musicGenerate
    case musicAnalyze
    case musicTranscribe
    case musicRealtime
    case videoGenerate
    case videoExportLatents
    case sfxGenerate
    case sfxVideo
    case sfxAEEncode
    case sfxAEDecode
    case sfxClapScore
    case sfxConditionText
    case modelBenchmark
    case pluginList
    case pluginInstall
    case pluginDoctor
    case openWebui
    case apiServe
    case custom
}

enum CommandInputKind: Equatable {
    case none
    case file([UTType])
    case directory
    case image
    case audio
    case video

    var title: String {
        switch self {
        case .none: return "Input"
        case .file: return "File"
        case .directory: return "Directory"
        case .image: return "Image"
        case .audio: return "Audio"
        case .video: return "Video"
        }
    }

    var allowedTypes: [UTType] {
        switch self {
        case .none, .directory: return []
        case .file(let types): return types
        case .image: return [.image]
        case .audio: return [.audio]
        case .video: return [.movie, .video, .audiovisualContent]
        }
    }
}

enum CommandOutputKind: Equatable {
    case none
    case file(String)
    case directory

    var isFile: Bool {
        if case .file = self { return true }
        return false
    }
}

enum StudioChatDefaults {
    static let fallbackModelID = "text-chat-gemma4-12b-4bit"
    static let fallbackServingEngine = "text-chat-gemma4"

    static func servingEngine(for modelID: String) -> String {
        let normalized = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalized.contains("deepseek-v4-flash") {
            return "text-chat-deepseek-v4-flash"
        }
        if normalized.contains("q36") || normalized.contains("q35") {
            return "text-chat-q36"
        }
        if normalized.contains("lfm") {
            return "text-chat-lfm2"
        }
        if normalized.contains("klein") {
            return "text-chat-klein"
        }
        if normalized.contains("gemma") {
            return "text-chat-gemma4"
        }
        return fallbackServingEngine
    }

    static func shouldReplaceModelDefault(_ modelID: String, oldRecommendation: String? = nil) -> Bool {
        let normalized = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalized.isEmpty { return true }
        var replaceable: Set<String> = [
            fallbackModelID,
            "text-chat-q36-nano",
            "text-chat-gemma4",
            "text-chat-gemma4-12b",
            "text-chat-gemma4-12b-4bit"
        ]
        if let oldRecommendation, !oldRecommendation.isBlank {
            replaceable.insert(oldRecommendation)
        }
        return replaceable.contains(normalized)
    }

    static func shouldReplaceServingEngineDefault(_ engine: String, oldRecommendation: String? = nil) -> Bool {
        let normalized = engine.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalized.isEmpty { return true }
        var replaceable: Set<String> = [
            fallbackServingEngine,
            "text-chat-q36",
            "text-chat-gemma4"
        ]
        if let oldRecommendation, !oldRecommendation.isBlank {
            replaceable.insert(servingEngine(for: oldRecommendation))
        }
        return replaceable.contains(normalized)
    }
}

enum StudioCodeDefaults {
    static let fallbackModelID = "text-code-north-mini"

    static func shouldReplaceModelDefault(_ modelID: String, oldRecommendation: String? = nil) -> Bool {
        let normalized = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalized.isEmpty { return true }
        var replaceable: Set<String> = [
            fallbackModelID,
            "text-agent-qwen35-9b",
            "text-code-qwen3",
        ]
        if let oldRecommendation, !oldRecommendation.isBlank {
            replaceable.insert(oldRecommendation)
        }
        return replaceable.contains(normalized)
    }
}

struct CommandDraft: Equatable {
    var prompt = ""
    var secondaryText = ""
    var model = ""
    var inputPath = ""
    var outputPath = ""
    var width = 1024
    var height = 1024
    var steps = 4
    var seed = ""
    var cfgScale = 1.0
    var strength = 0.75
    var maxTokens = 2048
    var temperature = 0.7
    var topP = 0.9
    var durationSeconds = 10.0
    var fps = 24
    var numFrames = 65
    var host = "127.0.0.1"
    var port = 8080
    var apiKey = ""
    var engine = StudioChatDefaults.fallbackServingEngine
    var variant = "distilled"
    var backend = "auto"
    var task = "transcribe"
    var language = "auto"
    var timestamps = true
    var voiceMode = "style"
    var voiceProfile = ""
    var refAudioPath = ""
    var refText = ""
    var saveProfileName = ""
    // Vision chat (vision-capable chat models) and the agentic tool loop for `text chat`.
    var imagePath = ""
    var tools = ""
    var toolLoop = false
    var allowShellExec = false
    var sandboxDir = ""
    var setupMode = "agent"
    var agentModel = "tier"
    var quiet = false
    var force = false
    var all = false
    var json = false
    var stream = false
    var extraArguments = ""
}

struct CommandTemplate: Identifiable, Equatable {
    let id: CommandTemplateID
    let category: CommandCategory
    let title: String
    let subtitle: String
    let systemImage: String
    let promptLabel: String?
    let secondaryLabel: String?
    let inputKind: CommandInputKind
    let outputKind: CommandOutputKind
    let defaultPrompt: String
    let defaultSecondaryText: String
    let defaultModel: String
    let defaultExtraArguments: String

    init(
        id: CommandTemplateID,
        category: CommandCategory,
        title: String,
        subtitle: String,
        systemImage: String,
        promptLabel: String? = nil,
        secondaryLabel: String? = nil,
        inputKind: CommandInputKind = .none,
        outputKind: CommandOutputKind = .none,
        defaultPrompt: String = "",
        defaultSecondaryText: String = "",
        defaultModel: String = "",
        defaultExtraArguments: String = ""
    ) {
        self.id = id
        self.category = category
        self.title = title
        self.subtitle = subtitle
        self.systemImage = systemImage
        self.promptLabel = promptLabel
        self.secondaryLabel = secondaryLabel
        self.inputKind = inputKind
        self.outputKind = outputKind
        self.defaultPrompt = defaultPrompt
        self.defaultSecondaryText = defaultSecondaryText
        self.defaultModel = defaultModel
        self.defaultExtraArguments = defaultExtraArguments
    }

    func defaultDraft() -> CommandDraft {
        var draft = CommandDraft()
        draft.prompt = defaultPrompt
        draft.secondaryText = defaultSecondaryText
        draft.model = defaultModel
        draft.extraArguments = defaultExtraArguments

        switch id {
        case .textChat:
            draft.stream = true
        case .textCode:
            draft.temperature = 1.0
            draft.topP = 0.95
            draft.stream = true
        case .visionCaption, .visionOCR:
            draft.maxTokens = id == .visionCaption ? 96 : 4096
            draft.temperature = id == .visionCaption ? 0.2 : 0.2
        case .apiServe:
            draft.engine = StudioChatDefaults.fallbackServingEngine
            draft.port = 8080
        case .setup:
            draft.setupMode = "agent"
            draft.agentModel = "tier"
        case .agentStart:
            draft.prompt = defaultPrompt
            draft.port = 8080
        case .imageValidate:
            draft.backend = "all"
            draft.variant = "zimage"
        case .imageTrainLoRA:
            draft.steps = 1000
        case .videoGenerate:
            draft.width = 768
            draft.height = 512
            draft.steps = 0
        case .videoExportLatents:
            draft.width = 768
            draft.height = 512
            draft.numFrames = 65
            draft.seed = "42"
        case .musicGenerate:
            draft.steps = 8
            draft.durationSeconds = 10
        case .sfxGenerate, .sfxVideo:
            draft.steps = 4
            draft.durationSeconds = 8
        case .speechTranscribe:
            draft.backend = "auto"
            draft.maxTokens = 448
            draft.task = "transcribe"
            draft.language = "auto"
            draft.timestamps = true
        default:
            break
        }

        switch outputKind {
        case .file(let ext):
            draft.outputPath = Self.defaultOutputURL(stem: id.rawValue, extension: ext).path
        case .directory:
            draft.outputPath = Self.defaultOutputDirectory(stem: id.rawValue).path
        case .none:
            break
        }

        return draft
    }

    func validationMessage(for draft: CommandDraft) -> String? {
        if promptLabel != nil && id != .custom && draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "\(promptLabel ?? "Prompt") is required."
        }

        let optionalInputs: Set<CommandTemplateID> = [.imageGenerate, .videoGenerate]
        if inputKind != .none
            && !optionalInputs.contains(id)
            && draft.inputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "\(inputKind.title) path is required."
        }

        switch id {
        case .modelPull:
            if !draft.all && draft.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return "Choose a model id, or enable All."
            }
        case .modelRemove:
            if draft.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return "Model id is required."
            }
        case .modelInfo:
            if draft.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return "Model id or local model path is required."
            }
        case .custom:
            if ShellWords.split(draft.extraArguments).isEmpty {
                return "Enter mere.run arguments."
            }
        default:
            break
        }

        return nil
    }

    func arguments(from draft: CommandDraft) -> [String] {
        var args: [String]

        switch id {
        case .setup:
            args = ["setup", "--mode", draft.setupMode, "--agent-model", draft.agentModel]
            if draft.force { args.append("--install") }
            if draft.stream { args.append("--start") }
            if draft.quiet { args.append("--quiet") }

        case .agentOnboard:
            args = ["agent", "onboard"]
            if draft.force { args.append("--pull-recommended") }
            if draft.all { args.append("--install-pi") }
            if draft.stream { args.append("--configure-pi") }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--host", draft.host, "--port", String(draft.port)]
            if draft.quiet { args.append("--quiet") }

        case .agentInstallPi:
            args = ["agent", "install-pi"]
            if draft.force { args.append("--force") }

        case .agentStart:
            args = ["agent", "start", "--host", draft.host, "--port", String(draft.port)]
            if !draft.prompt.isBlank { args += ["--prompt", draft.prompt] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.stream { args.append("--skip-server") }
            if draft.force { args.append("--allow-unsupported") }

        case .modelList:
            args = ["model", "list"]

        case .modelCapabilities:
            args = ["model", "capabilities"]
            if draft.all { args.append("--all") }
            if draft.force { args.append("--recommended") }
            if draft.json { args.append("--json") }

        case .modelPull:
            args = ["model", "pull"]
            if draft.all {
                args.append("--all")
            } else {
                args.append(draft.model)
            }
            if draft.force { args.append("--force") }
            if draft.stream { args.append("--allow-unsupported") }
            if draft.quiet { args.append("--quiet") }

        case .modelInfo:
            args = ["model", "info", draft.model]
            if draft.all { args.append("--json") }
            if draft.force { args.append("--components") }

        case .modelRemove:
            args = ["model", "remove", draft.model]
            if draft.force { args.append("--force") }

        case .modelRepairManifests:
            args = ["model", "repair-manifests"]
            if draft.force { args.append("--dry-run") }

        case .imageGenerate:
            args = ["image", "generate", "--prompt", draft.prompt, "--output", draft.outputPath]
            args += ["--width", String(draft.width), "--height", String(draft.height)]
            args += ["--steps", String(draft.steps)]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.secondaryText.isBlank { args += ["--negative-prompt", draft.secondaryText] }
            if draft.cfgScale != 1.0 { args += ["--cfg", format(draft.cfgScale)] }
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if !draft.inputPath.isBlank {
                args += ["--input", draft.inputPath, "--strength", format(draft.strength)]
            }
            if draft.quiet { args.append("--quiet") }

        case .imageTrainLoRA:
            args = ["image", "train-lora", "--data", draft.inputPath, "--output", draft.outputPath]
            args += ["--width", String(draft.width), "--height", String(draft.height)]
            args += ["--training-steps", String(draft.steps)]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if draft.quiet { args.append("--quiet") }

        case .imageValidate:
            args = ["image", "validate", "--test", draft.backend, "--family", draft.variant]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if draft.force { args.append("--save-reference") }
            if draft.all { args.append("--compare") }

        case .textChat:
            args = ["text", "chat", "--prompt", draft.prompt]
            if !draft.secondaryText.isBlank { args += ["--system", draft.secondaryText] }
            if !draft.imagePath.isBlank { args += ["--image", draft.imagePath] }
            args += ["--max-tokens", String(draft.maxTokens), "--temperature", format(draft.temperature), "--top-p", format(draft.topP)]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.tools.isBlank { args += ["--tools", draft.tools] }
            if draft.toolLoop { args.append("--tool-loop") }
            if draft.allowShellExec { args.append("--allow-shell-exec") }
            if !draft.sandboxDir.isBlank { args += ["--sandbox-dir", draft.sandboxDir] }
            if draft.stream { args.append("--stream") }
            if draft.all { args.append("--thinking") }
            if draft.force { args.append("--stats") }
            if draft.quiet { args.append("--quiet") }

        case .textCode:
            args = ["text", "code", "--prompt", draft.prompt]
            if !draft.secondaryText.isBlank { args += ["--system", draft.secondaryText] }
            args += ["--max-tokens", String(draft.maxTokens), "--temperature", format(draft.temperature), "--top-p", format(draft.topP)]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.stream { args.append("--stream") }
            if draft.force { args.append("--stats") }
            if draft.quiet { args.append("--quiet") }

        case .textEmbed:
            args = ["text", "embed", draft.prompt]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.maxTokens > 0 { args += ["--max-tokens", String(draft.maxTokens)] }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if draft.force { args.append("--pretty") }

        case .textAnonymize:
            args = ["text", "anonymize", draft.prompt]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.maxTokens > 0 { args += ["--max-tokens", String(draft.maxTokens)] }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if draft.all { args.append("--json") }
            if draft.force { args.append("--pretty") }

        case .speechSynthesize:
            args = ["speech", "synthesize", draft.prompt, "--output", draft.outputPath]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.secondaryText.isBlank { args += ["--voice", draft.secondaryText] }
            if draft.voiceMode == "clone" { args += ["--mode", "clone"] }
            if !draft.voiceProfile.isBlank { args += ["--profile", draft.voiceProfile] }
            if !draft.refAudioPath.isBlank { args += ["--ref-audio", draft.refAudioPath] }
            if !draft.refText.isBlank { args += ["--ref-text", draft.refText] }
            if !draft.saveProfileName.isBlank { args += ["--save-profile", draft.saveProfileName] }
            if !draft.language.isBlank, draft.language != "auto" { args += ["--language", draft.language] }
            args += ["--temperature", format(draft.temperature)]
            if draft.stream { args.append("--stream") }
            if draft.quiet { args.append("--quiet") }

        case .speechTranscribe:
            args = ["speech", "transcribe", draft.inputPath]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--backend", draft.backend, "--task", draft.task, "--max-tokens", String(draft.maxTokens)]
            if !draft.language.isBlank, draft.language != "auto" { args += ["--language", draft.language] }
            if draft.stream { args.append("--stream") }
            if !draft.timestamps { args.append("--no-timestamps") }
            if draft.quiet { args.append("--quiet") }

        case .speechProfileList:
            args = ["speech", "profile", "list"]

        case .speechProfileCreate:
            args = ["speech", "profile", "create", "--name", draft.prompt, "--audio", draft.inputPath]
            if !draft.secondaryText.isBlank { args += ["--text", draft.secondaryText] }
            if !draft.language.isBlank { args += ["--language", draft.language] }
            if draft.quiet { args.append("--quiet") }

        case .speechProfileDelete:
            args = ["speech", "profile", "delete", "--id", draft.prompt]

        case .visionInspect:
            args = ["vision", "inspect", draft.inputPath, draft.prompt]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--max-tokens", String(draft.maxTokens), "--temperature", format(draft.temperature), "--top-p", format(draft.topP)]

        case .visionCaption:
            args = ["vision", "caption", draft.inputPath]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.outputPath.isBlank { args += ["--output-dir", draft.outputPath] }
            args += ["--prompt", draft.prompt, "--max-tokens", String(draft.maxTokens), "--temperature", format(draft.temperature)]

        case .visionOCR:
            args = ["vision", "ocr", draft.inputPath, "--backend", draft.backend]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            args += ["--max-tokens", String(draft.maxTokens), "--temperature", format(draft.temperature)]
            if draft.all { args.append("--compare") }
            if draft.quiet { args.append("--quiet") }

        case .visionGround:
            args = ["vision", "ground", draft.inputPath, "--query", draft.prompt]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }

        case .visionSegment:
            args = ["vision", "segment", draft.inputPath, "--prompt", draft.prompt]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if draft.force { args.append("--show-boxes") }

        case .visionTrack:
            args = ["vision", "track", draft.inputPath, "--prompt", draft.prompt]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if draft.force { args.append("--show-boxes") }

        case .visionTrackLive:
            args = ["vision", "track-live", "--prompt", draft.prompt]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            args += ["--duration-seconds", format(draft.durationSeconds)]
            if draft.force { args.append("--show-boxes") }

        case .musicGenerate:
            args = ["music", "generate", draft.prompt]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.secondaryText.isBlank { args += ["--lyrics", draft.secondaryText] }
            args += ["--duration", format(draft.durationSeconds), "--steps", String(draft.steps)]
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if draft.quiet { args.append("--quiet") }

        case .videoGenerate:
            args = ["video", "generate", draft.prompt]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--variant", draft.variant, "--width", String(draft.width), "--height", String(draft.height)]
            args += ["--num-frames", String(draft.numFrames), "--fps", String(draft.fps)]
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if !draft.inputPath.isBlank { args += ["--image", draft.inputPath, "--image-strength", format(draft.strength)] }
            if draft.quiet { args.append("--quiet") }

        case .videoExportLatents:
            args = ["video", "export-latents", draft.prompt]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--width", String(draft.width), "--height", String(draft.height)]
            args += ["--num-frames", String(draft.numFrames)]
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if draft.quiet { args.append("--quiet") }

        case .sfxGenerate:
            args = ["sfx", "generate", draft.prompt]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--duration", format(draft.durationSeconds), "--steps", String(draft.steps)]
            if draft.cfgScale != 1.0 { args += ["--cfg", format(draft.cfgScale)] }
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if draft.quiet { args.append("--quiet") }

        case .sfxVideo:
            args = ["sfx", "video", "generate", draft.prompt, draft.inputPath]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--duration", format(draft.durationSeconds), "--steps", String(draft.steps)]
            if !draft.seed.isBlank { args += ["--seed", draft.seed] }
            if draft.quiet { args.append("--quiet") }

        case .musicAnalyze:
            args = ["music", "analyze", draft.inputPath]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.all { args.append("--include-raw-lm") }
            if draft.quiet { args.append("--quiet") }

        case .musicTranscribe:
            args = ["music", "transcribe", draft.inputPath]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if draft.quiet { args.append("--quiet") }

        case .musicRealtime:
            args = ["music", "realtime", draft.prompt]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            args += ["--duration", format(draft.durationSeconds)]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            // GUI subprocesses have no audio device contract or stdin TTY, so capture to the
            // output WAV without live playback; interactive steering stays CLI-only.
            args.append("--no-play")
            if draft.quiet { args.append("--quiet") }

        case .sfxAEEncode:
            args = ["sfx", "ae", "encode", draft.inputPath]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.quiet { args.append("--quiet") }

        case .sfxAEDecode:
            args = ["sfx", "ae", "decode", draft.inputPath]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.quiet { args.append("--quiet") }

        case .sfxClapScore:
            args = ["sfx", "clap", "score", draft.prompt, draft.inputPath]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.quiet { args.append("--quiet") }

        case .sfxConditionText:
            args = ["sfx", "condition", "text", draft.prompt]
            if !draft.outputPath.isBlank { args += ["--output", draft.outputPath] }
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.quiet { args.append("--quiet") }

        case .modelBenchmark:
            args = ["model", "benchmark", "q36-mtp"]
            if !draft.model.isBlank { args += ["--model", draft.model] }
            if draft.all { args.append("--json") }

        case .pluginList:
            args = ["plugin", "list"]
            if draft.all { args.append("--json") }

        case .pluginInstall:
            args = ["plugin", "install", draft.prompt]
            if draft.force { args.append("--yes") }

        case .pluginDoctor:
            args = ["plugin", "doctor", draft.prompt]

        case .openWebui:
            args = ["open-webui", "quickstart", "--host", draft.host, "--port", String(draft.port)]
            if !draft.model.isBlank { args += ["--text-model", draft.model] }
            if !draft.apiKey.isBlank { args += ["--api-key", draft.apiKey] }

        case .apiServe:
            args = ["api", "serve", "--host", draft.host, "--port", String(draft.port), "--engine", draft.engine]
            if !draft.model.isBlank { args += ["--model", draft.model] }

        case .custom:
            return ShellWords.split(draft.extraArguments)
        }

        let extra = ShellWords.split(draft.extraArguments)
        if !extra.isEmpty {
            args.append(contentsOf: extra)
        }
        return args
    }

    private static func defaultOutputDirectory(stem: String) -> URL {
        outputRoot().appendingPathComponent(stem, isDirectory: true)
    }

    private static func defaultOutputURL(stem: String, extension ext: String) -> URL {
        let stamp = DateFormatter.mereRunTimestamp.string(from: Date())
        return outputRoot()
            .appendingPathComponent("\(stem)-\(stamp)")
            .appendingPathExtension(ext)
    }

    private static func outputRoot() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MereRun", isDirectory: true)
            .appendingPathComponent("App Outputs", isDirectory: true)
    }

    private func format(_ value: Double) -> String {
        String(format: "%.4g", value)
    }
}

enum CommandLaunchEnvironment {
    static let apiKeyEnvironmentKey = "MERERUN_API_KEY"

    static func overrides(templateID: CommandTemplateID, draft: CommandDraft) -> [String: String] {
        guard templateID == .apiServe else { return [:] }
        let apiKey = draft.apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !apiKey.isEmpty else { return [:] }
        return [apiKeyEnvironmentKey: apiKey]
    }
}

enum CommandCatalog {
    static let templates: [CommandTemplate] = [
        CommandTemplate(
            id: .setup,
            category: .setup,
            title: "Setup path",
            subtitle: "Plan or run guided, BYOA, or manual setup",
            systemImage: "wand.and.stars"
        ),
        CommandTemplate(
            id: .agentOnboard,
            category: .setup,
            title: "Agent onboarding",
            subtitle: "Check local readiness and prepare Pi integration",
            systemImage: "person.crop.circle.badge.gearshape",
            defaultModel: StudioCodeDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .agentInstallPi,
            category: .setup,
            title: "Install Pi",
            subtitle: "Install or replace the optional setup agent",
            systemImage: "square.and.arrow.down"
        ),
        CommandTemplate(
            id: .agentStart,
            category: .setup,
            title: "Start agent",
            subtitle: "Launch a guided setup session",
            systemImage: "terminal.fill",
            promptLabel: "Prompt",
            defaultPrompt: "Guide me through setting up mere.run on this machine. Start by summarizing what this Mac can run, then help me install only supported models."
        ),
        CommandTemplate(id: .modelList, category: .models, title: "List models", subtitle: "Installed and missing managed models", systemImage: "list.bullet.rectangle"),
        CommandTemplate(id: .modelCapabilities, category: .models, title: "Capabilities", subtitle: "Hardware support and recommended pulls", systemImage: "memorychip"),
        CommandTemplate(id: .modelPull, category: .models, title: "Pull model", subtitle: "Download a managed model", systemImage: "arrow.down.circle", defaultModel: "image-zimage-nano"),
        CommandTemplate(id: .modelInfo, category: .models, title: "Model info", subtitle: "Manifest and validation report", systemImage: "info.circle", defaultModel: "image-zimage-nano"),
        CommandTemplate(id: .modelRemove, category: .models, title: "Remove model", subtitle: "Delete a local model install", systemImage: "trash", defaultModel: "image-zimage-nano"),
        CommandTemplate(id: .modelRepairManifests, category: .models, title: "Repair manifests", subtitle: "Write missing known model manifests", systemImage: "wrench.and.screwdriver"),
        CommandTemplate(
            id: .imageGenerate,
            category: .image,
            title: "Generate image",
            subtitle: "Text-to-image and image-to-image",
            systemImage: "photo",
            promptLabel: "Prompt",
            secondaryLabel: "Negative prompt",
            inputKind: .image,
            outputKind: .file("png"),
            defaultPrompt: "a ceramic coffee mug in soft morning light",
            defaultModel: "image-zimage-nano"
        ),
        CommandTemplate(
            id: .imageTrainLoRA,
            category: .image,
            title: "Train LoRA",
            subtitle: "Train Krea 2 Raw adapters",
            systemImage: "slider.horizontal.3",
            inputKind: .directory,
            outputKind: .file("safetensors"),
            defaultModel: "image-krea2-raw"
        ),
        CommandTemplate(
            id: .imageValidate,
            category: .image,
            title: "Validate image stack",
            subtitle: "Run deterministic image runtime checks",
            systemImage: "checkmark.seal",
            outputKind: .directory
        ),
        CommandTemplate(
            id: .textChat,
            category: .text,
            title: "Chat",
            subtitle: "Local chat models",
            systemImage: "bubble.left.and.bubble.right",
            promptLabel: "Prompt",
            secondaryLabel: "System",
            defaultPrompt: "Summarize diffusion models in one paragraph.",
            defaultModel: StudioChatDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .textCode,
            category: .text,
            title: "Code",
            subtitle: "Local code generation",
            systemImage: "chevron.left.forwardslash.chevron.right",
            promptLabel: "Prompt",
            secondaryLabel: "System",
            defaultPrompt: "Write a tiny Swift function that formats byte counts.",
            defaultModel: StudioCodeDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .textEmbed,
            category: .text,
            title: "Embeddings",
            subtitle: "Generate JSON embedding vectors",
            systemImage: "point.3.connected.trianglepath.dotted",
            promptLabel: "Text",
            outputKind: .file("json"),
            defaultPrompt: "semantic search query",
            defaultModel: "text-embed-qwen3-0.6b"
        ),
        CommandTemplate(
            id: .textAnonymize,
            category: .text,
            title: "Anonymize",
            subtitle: "Detect and redact PII",
            systemImage: "eye.slash",
            promptLabel: "Text",
            outputKind: .file("txt"),
            defaultPrompt: "My name is Alice Smith and my email is alice@example.com",
            defaultModel: "text-anonymize-privacy-filter"
        ),
        CommandTemplate(
            id: .speechSynthesize,
            category: .speech,
            title: "Synthesize",
            subtitle: "Text to speech",
            systemImage: "waveform",
            promptLabel: "Text",
            secondaryLabel: "Voice",
            outputKind: .file("wav"),
            defaultPrompt: "Hello from mere.run.",
            defaultSecondaryText: "A calm female voice with clear pronunciation",
            defaultModel: "speech-tts-qwen3-nano"
        ),
        CommandTemplate(
            id: .speechTranscribe,
            category: .speech,
            title: "Transcribe",
            subtitle: "Speech to text",
            systemImage: "captions.bubble",
            inputKind: .audio,
            outputKind: .file("txt")
        ),
        CommandTemplate(id: .speechProfileList, category: .speech, title: "Voice profiles", subtitle: "List saved clone profiles", systemImage: "person.wave.2"),
        CommandTemplate(
            id: .speechProfileCreate,
            category: .speech,
            title: "Create voice profile",
            subtitle: "Save reference audio for clone mode",
            systemImage: "person.badge.plus",
            promptLabel: "Profile name",
            secondaryLabel: "Transcript override",
            inputKind: .audio,
            defaultPrompt: "Narration profile"
        ),
        CommandTemplate(
            id: .speechProfileDelete,
            category: .speech,
            title: "Delete voice profile",
            subtitle: "Remove a saved clone profile",
            systemImage: "person.badge.minus",
            promptLabel: "Profile UUID"
        ),
        CommandTemplate(
            id: .visionInspect,
            category: .vision,
            title: "Inspect image",
            subtitle: "Ask a VLM about an image",
            systemImage: "eye",
            promptLabel: "Question",
            inputKind: .image,
            defaultPrompt: "Describe this image."
        ),
        CommandTemplate(
            id: .visionCaption,
            category: .vision,
            title: "Caption",
            subtitle: "Write LoRA-friendly captions",
            systemImage: "text.bubble",
            promptLabel: "Instruction",
            inputKind: .image,
            outputKind: .directory,
            defaultPrompt: "Write a short, concrete caption describing the image for LoRA training. Avoid fluff."
        ),
        CommandTemplate(
            id: .visionOCR,
            category: .vision,
            title: "OCR",
            subtitle: "Extract text from images",
            systemImage: "doc.text.viewfinder",
            inputKind: .image,
            outputKind: .directory,
            defaultModel: "vision-ocr-lighton"
        ),
        CommandTemplate(
            id: .visionGround,
            category: .vision,
            title: "Ground objects",
            subtitle: "Find prompted objects in an image",
            systemImage: "scope",
            promptLabel: "Query",
            inputKind: .image,
            outputKind: .file("png"),
            defaultPrompt: "a person",
            defaultModel: "vision-ground-falcon-perception"
        ),
        CommandTemplate(
            id: .visionSegment,
            category: .vision,
            title: "Segment",
            subtitle: "Segment prompted objects",
            systemImage: "square.dashed",
            promptLabel: "Prompt",
            inputKind: .image,
            outputKind: .file("png"),
            defaultPrompt: "a person",
            defaultModel: "vision-segment-sam31"
        ),
        CommandTemplate(
            id: .visionTrack,
            category: .vision,
            title: "Track video",
            subtitle: "Track prompted objects through a clip",
            systemImage: "point.topleft.down.curvedto.point.bottomright.up",
            promptLabel: "Prompt",
            inputKind: .video,
            outputKind: .file("mp4"),
            defaultPrompt: "a person",
            defaultModel: "vision-segment-sam31"
        ),
        CommandTemplate(
            id: .visionTrackLive,
            category: .vision,
            title: "Track camera",
            subtitle: "Capture then track from a camera",
            systemImage: "video.badge.waveform",
            promptLabel: "Prompt",
            outputKind: .file("mp4"),
            defaultPrompt: "a person",
            defaultModel: "vision-segment-sam31"
        ),
        CommandTemplate(
            id: .musicGenerate,
            category: .media,
            title: "Generate music",
            subtitle: "ACE-Step or Magenta RT2 music generation",
            systemImage: "music.note",
            promptLabel: "Caption",
            secondaryLabel: "Lyrics",
            outputKind: .file("wav"),
            defaultPrompt: "upbeat electronic groove",
            defaultModel: "music-acestep"
        ),
        CommandTemplate(
            id: .videoGenerate,
            category: .media,
            title: "Generate video",
            subtitle: "Native LTX draft or unified AV generation",
            systemImage: "film",
            promptLabel: "Prompt",
            inputKind: .image,
            outputKind: .file("mp4"),
            defaultPrompt: "a cinematic drone flythrough over snowy mountains",
            defaultModel: "video-ltx23-av-mlx"
        ),
        CommandTemplate(
            id: .videoExportLatents,
            category: .media,
            title: "Export video latents",
            subtitle: "Write native LTX final latents",
            systemImage: "shippingbox",
            promptLabel: "Prompt",
            outputKind: .file("safetensors"),
            defaultPrompt: "a cinematic drone flythrough over snowy mountains",
            defaultModel: "video-ltx-av"
        ),
        CommandTemplate(
            id: .sfxGenerate,
            category: .sfx,
            title: "Generate sound effect",
            subtitle: "Woosh text-to-audio sound effects",
            systemImage: "speaker.wave.2",
            promptLabel: "Prompt",
            outputKind: .file("wav"),
            defaultPrompt: "a heavy wooden door creaking open"
        ),
        CommandTemplate(
            id: .sfxVideo,
            category: .sfx,
            title: "Video foley",
            subtitle: "Generate sound effects from a video",
            systemImage: "video.badge.waveform",
            promptLabel: "Prompt",
            inputKind: .video,
            outputKind: .file("wav"),
            defaultPrompt: "footsteps on gravel"
        ),
        CommandTemplate(
            id: .musicAnalyze,
            category: .media,
            title: "Analyze music",
            subtitle: "ACE-Step audio understanding (JSON)",
            systemImage: "waveform.badge.magnifyingglass",
            inputKind: .audio,
            defaultModel: "music-acestep"
        ),
        CommandTemplate(
            id: .musicTranscribe,
            category: .media,
            title: "Transcribe to MIDI",
            subtitle: "MuScriptor full-mix audio to instrument tracks",
            systemImage: "pianokeys",
            inputKind: .audio,
            outputKind: .file("mid"),
            defaultModel: "music-muscriptor-medium"
        ),
        CommandTemplate(
            id: .musicRealtime,
            category: .media,
            title: "Realtime music",
            subtitle: "Magenta RT2 capture to WAV",
            systemImage: "dot.radiowaves.left.and.right",
            promptLabel: "Prompt",
            outputKind: .file("wav"),
            defaultPrompt: "warm ambient pads with a slow build"
        ),
        CommandTemplate(
            id: .sfxAEEncode,
            category: .sfx,
            title: "Autoencoder · encode",
            subtitle: "Audio → Woosh latents (.npy)",
            systemImage: "arrow.down.doc",
            inputKind: .audio,
            outputKind: .file("npy")
        ),
        CommandTemplate(
            id: .sfxAEDecode,
            category: .sfx,
            title: "Autoencoder · decode",
            subtitle: "Woosh latents (.npy) → audio",
            systemImage: "arrow.up.doc",
            inputKind: .file([.data]),
            outputKind: .file("wav")
        ),
        CommandTemplate(
            id: .sfxClapScore,
            category: .sfx,
            title: "CLAP score",
            subtitle: "Score audio against a prompt (JSON)",
            systemImage: "checkmark.seal",
            promptLabel: "Prompt",
            inputKind: .audio,
            defaultPrompt: "a heavy wooden door creaking open"
        ),
        CommandTemplate(
            id: .sfxConditionText,
            category: .sfx,
            title: "Conditioning · text",
            subtitle: "Export Woosh conditioning tensors",
            systemImage: "function",
            promptLabel: "Prompt",
            outputKind: .file("safetensors"),
            defaultPrompt: "a heavy wooden door creaking open"
        ),
        CommandTemplate(
            id: .modelBenchmark,
            category: .models,
            title: "Benchmark",
            subtitle: "Focused Qwen3.6 MTP benchmark",
            systemImage: "speedometer",
            defaultModel: "text-chat-q36-nano"
        ),
        CommandTemplate(
            id: .pluginList,
            category: .server,
            title: "Plugins",
            subtitle: "List official companion plugins",
            systemImage: "puzzlepiece.extension"
        ),
        CommandTemplate(
            id: .pluginInstall,
            category: .server,
            title: "Install plugin",
            subtitle: "Install an official plugin by id",
            systemImage: "square.and.arrow.down",
            promptLabel: "Plugin id",
            defaultPrompt: "mere-runpod"
        ),
        CommandTemplate(
            id: .pluginDoctor,
            category: .server,
            title: "Plugin doctor",
            subtitle: "Run an installed plugin's doctor",
            systemImage: "stethoscope",
            promptLabel: "Plugin id",
            defaultPrompt: "mere-runpod"
        ),
        CommandTemplate(
            id: .openWebui,
            category: .server,
            title: "Open WebUI",
            subtitle: "Start the Open WebUI companion",
            systemImage: "globe",
            defaultModel: StudioChatDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .apiServe,
            category: .server,
            title: "API server",
            subtitle: "OpenAI-compatible local server",
            systemImage: "network",
            defaultModel: StudioChatDefaults.fallbackModelID
        ),
        CommandTemplate(
            id: .custom,
            category: .custom,
            title: "Raw arguments",
            subtitle: "Run any mere.run command",
            systemImage: "terminal",
            defaultExtraArguments: "--help"
        ),
    ]

    static func templates(in category: CommandCategory) -> [CommandTemplate] {
        templates.filter { $0.category == category }
    }

    static func template(id: CommandTemplateID) -> CommandTemplate? {
        templates.first { $0.id == id }
    }
}

enum ShellWords {
    static func split(_ raw: String) -> [String] {
        var words: [String] = []
        var current = ""
        var quote: Character?
        var escaping = false

        for char in raw {
            if escaping {
                current.append(char)
                escaping = false
                continue
            }

            if char == "\\" {
                escaping = true
                continue
            }

            if let activeQuote = quote {
                if char == activeQuote {
                    quote = nil
                } else {
                    current.append(char)
                }
                continue
            }

            if char == "\"" || char == "'" {
                quote = char
                continue
            }

            if char.isWhitespace {
                if !current.isEmpty {
                    words.append(current)
                    current.removeAll(keepingCapacity: true)
                }
                continue
            }

            current.append(char)
        }

        if escaping {
            current.append("\\")
        }
        if !current.isEmpty {
            words.append(current)
        }
        return words
    }
}

extension String {
    var isBlank: Bool {
        trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

extension DateFormatter {
    static let mereRunTimestamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter
    }()
}
