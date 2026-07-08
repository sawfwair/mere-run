import Foundation
import UniformTypeIdentifiers

enum StudioMode: String, CaseIterable, Codable, Identifiable {
    case createImage
    case chat
    case code
    case speak
    case listen
    case readImage
    case findObjects
    case segment
    case track
    case music
    case video
    case sfx

    var id: String { rawValue }

    /// Chat and Code are multi-turn conversations; every other mode is a single-shot run.
    var isConversational: Bool {
        self == .chat || self == .code
    }

    var title: String {
        switch self {
        case .createImage: return "Create Image"
        case .chat: return "Chat"
        case .code: return "Code"
        case .speak: return "Speak"
        case .listen: return "Listen"
        case .readImage: return "Read Image"
        case .findObjects: return "Find"
        case .segment: return "Segment"
        case .track: return "Track"
        case .music: return "Music"
        case .video: return "Video"
        case .sfx: return "Sound FX"
        }
    }

    var subtitle: String {
        switch self {
        case .createImage: return "Text or reference image to PNG"
        case .chat: return "Ask a local model"
        case .code: return "Draft code locally"
        case .speak: return "Text to natural speech"
        case .listen: return "Audio to text"
        case .readImage: return "Inspect, OCR, or caption"
        case .findObjects: return "Ground prompted objects"
        case .segment: return "Cut out prompted objects"
        case .track: return "Follow objects through video"
        case .music: return "Prompt to song"
        case .video: return "Prompt to clip"
        case .sfx: return "Prompt to sound effect"
        }
    }

    var systemImage: String {
        switch self {
        case .createImage: return "photo"
        case .chat: return "bubble.left.and.bubble.right"
        case .code: return "chevron.left.forwardslash.chevron.right"
        case .speak: return "waveform"
        case .listen: return "captions.bubble"
        case .readImage: return "doc.viewfinder"
        case .findObjects: return "scope"
        case .segment: return "square.dashed"
        case .track: return "point.topleft.down.curvedto.point.bottomright.up"
        case .music: return "music.note"
        case .video: return "film"
        case .sfx: return "speaker.wave.2"
        }
    }

    var defaultTemplateID: CommandTemplateID {
        switch self {
        case .createImage: return .imageGenerate
        case .chat: return .textChat
        case .code: return .textCode
        case .speak: return .speechSynthesize
        case .listen: return .speechTranscribe
        case .readImage: return .visionInspect
        case .findObjects: return .visionGround
        case .segment: return .visionSegment
        case .track: return .visionTrack
        case .music: return .musicGenerate
        case .video: return .videoGenerate
        case .sfx: return .sfxGenerate
        }
    }

    var acceptedTypes: [UTType] {
        switch self {
        case .createImage, .readImage, .findObjects, .segment, .chat:
            return [.image]
        case .listen:
            return [.audio]
        case .track:
            return [.movie, .video, .audiovisualContent]
        case .video:
            return [.image]
        default:
            return []
        }
    }

    var requiresAttachment: Bool {
        switch self {
        case .listen, .readImage, .findObjects, .segment, .track:
            return true
        default:
            return false
        }
    }

    var promptPlaceholder: String {
        switch self {
        case .createImage: return "Describe the image..."
        case .chat: return "Ask anything..."
        case .code: return "Describe the code you want..."
        case .speak: return "Type what to say..."
        case .listen: return "Attach audio to transcribe..."
        case .readImage: return "Ask about the image..."
        case .findObjects: return "What should mere.run find?"
        case .segment: return "What should mere.run segment?"
        case .track: return "What should mere.run track?"
        case .music: return "Describe the song..."
        case .video: return "Describe the video..."
        case .sfx: return "Describe the sound..."
        }
    }

    var emptyTitle: String {
        switch self {
        case .createImage: return "Make something visible."
        case .chat: return "Ask locally, keep it local."
        case .code: return "Turn intent into code."
        case .speak: return "Give text a voice."
        case .listen: return "Drop in audio."
        case .readImage: return "Let the image answer."
        case .findObjects: return "Find what matters."
        case .segment: return "Separate subject from scene."
        case .track: return "Follow motion through time."
        case .music: return "Score the moment."
        case .video: return "Make the frame move."
        case .sfx: return "Design a sound."
        }
    }

    var emptyMessage: String {
        switch self {
        case .createImage: return "Write a prompt, attach a reference if you have one, then create."
        case .chat: return "Ask a question and mere.run will answer with a local model."
        case .code: return "Describe a function, script, or refactor and run it on-device."
        case .speak: return "Enter the line, pick a voice style, and render a WAV."
        case .listen: return "Attach an audio file to create a transcript."
        case .readImage: return "Attach an image and ask for an inspection, OCR pass, or caption."
        case .findObjects: return "Attach an image and name the thing to locate."
        case .segment: return "Attach an image and name the object to isolate."
        case .track: return "Attach a video and name the subject to follow."
        case .music: return "Describe the sound, add lyrics if needed, and create audio."
        case .video: return "Describe a shot, optionally attach a starting image, and create a clip."
        case .sfx: return "Describe a sound effect and render a WAV."
        }
    }

    /// One-click starters shown on the empty canvas. They fill the composer, never auto-run.
    /// Attachment-first modes keep prompts short (they name the subject, not the scene).
    var examplePrompts: [String] {
        switch self {
        case .createImage:
            return [
                "A lighthouse at dusk, gouache on rough paper",
                "Brass mechanical keyboard, studio product shot",
                "Isometric cutaway of a tiny bakery, morning light"
            ]
        case .chat:
            return [
                "Explain how attention works in a transformer, briefly",
                "Draft a friendly reply declining a meeting",
                "What can I cook with mushrooms, eggs, and spinach?"
            ]
        case .code:
            return [
                "A Swift function that debounces async work",
                "A zsh script that renames photos by EXIF date",
                "A tiny Python HTTP server that logs request headers"
            ]
        case .speak:
            return [
                "Welcome aboard. Let's make something worth keeping.",
                "The forecast calls for slow mornings and warm coffee.",
                "Three… two… one… liftoff."
            ]
        case .listen:
            return []
        case .readImage:
            return [
                "Describe this scene in one paragraph",
                "What's written on the receipt?",
                "List every object on the desk"
            ]
        case .findObjects:
            return ["the red bicycle", "every coffee cup", "the person wearing a hat"]
        case .segment:
            return ["the foreground subject", "the dog", "the nearest car"]
        case .track:
            return ["the skateboarder", "the white van", "the ball"]
        case .music:
            return [
                "Slow-burn synthwave with a hopeful bridge",
                "Acoustic folk waltz, brushed drums, dusty piano",
                "90s boom-bap beat over upright bass"
            ]
        case .video:
            return [
                "Steam rising from a street-food stall at night",
                "A paper boat drifting down a rain gutter, macro",
                "Slow dolly through a sunlit greenhouse"
            ]
        case .sfx:
            return [
                "Heavy wooden door creaking open",
                "Retro arcade power-up",
                "Distant thunder rolling over hills"
            ]
        }
    }
}

/// Sidebar sections. Order here is the navigation order.
enum StudioModeGroup: String, CaseIterable, Identifiable {
    case create = "Create"
    case converse = "Converse"
    case voice = "Voice"
    case vision = "Vision"

    var id: String { rawValue }

    var modes: [StudioMode] {
        switch self {
        case .create: return [.createImage, .video, .music, .sfx]
        case .converse: return [.chat, .code]
        case .voice: return [.speak, .listen]
        case .vision: return [.readImage, .findObjects, .segment, .track]
        }
    }
}

enum StudioReadImageAction: String, CaseIterable, Codable, Identifiable {
    case inspect
    case ocr
    case caption

    var id: String { rawValue }

    var title: String {
        switch self {
        case .inspect: return "Inspect"
        case .ocr: return "OCR"
        case .caption: return "Caption"
        }
    }

    var templateID: CommandTemplateID {
        switch self {
        case .inspect: return .visionInspect
        case .ocr: return .visionOCR
        case .caption: return .visionCaption
        }
    }
}

struct StudioDraft: Codable, Equatable {
    var prompt = ""
    var secondaryText = ""
    var inputPath = ""
    var model = ""
    var width = 1024
    var height = 1024
    var steps = 4
    var seed = ""
    var durationSeconds = 10.0
    var readImageAction: StudioReadImageAction = .inspect
    // Speak voice cloning (Studio surface). "style" uses the voice description; "clone" uses a
    // saved profile or reference audio.
    var voiceMode = "style"
    var voiceProfile = ""
    var refAudioPath = ""
    var saveProfileName = ""
    // Advanced depth shared with the Advanced surface (see StudioOptionSchema). Defaults are
    // seeded from the matching template's CommandDraft so the two surfaces never drift.
    var temperature = 0.7
    var maxTokens = 2048
    var cfgScale = 1.0
    var strength = 0.75
    var language = "auto"
    var timestamps = true
    var backend = "auto"
    var fps = 24
    var numFrames = 65
    var variant = "distilled"

    mutating func reset(for mode: StudioMode) {
        let base = CommandCatalog.template(id: mode.defaultTemplateID)?.defaultDraft()
        prompt = modeDefaultPrompt(mode)
        secondaryText = modeDefaultSecondaryText(mode)
        inputPath = ""
        model = CommandCatalog.template(id: mode.defaultTemplateID)?.defaultModel ?? ""
        width = mode == .video ? 768 : 1024
        height = mode == .video ? 512 : 1024
        steps = mode == .music ? 8 : 4
        seed = ""
        durationSeconds = 10
        readImageAction = .inspect
        voiceMode = "style"
        voiceProfile = ""
        refAudioPath = ""
        saveProfileName = ""
        temperature = base?.temperature ?? 0.7
        maxTokens = base?.maxTokens ?? 2048
        cfgScale = base?.cfgScale ?? 1.0
        strength = base?.strength ?? 0.75
        language = base?.language ?? "auto"
        timestamps = base?.timestamps ?? true
        backend = base?.backend ?? "auto"
        fps = base?.fps ?? 24
        numFrames = base?.numFrames ?? 65
        variant = base?.variant ?? "distilled"
    }

    private func modeDefaultPrompt(_ mode: StudioMode) -> String {
        switch mode {
        case .listen:
            return ""
        default:
            return CommandCatalog.template(id: mode.defaultTemplateID)?.defaultPrompt ?? ""
        }
    }

    private func modeDefaultSecondaryText(_ mode: StudioMode) -> String {
        CommandCatalog.template(id: mode.defaultTemplateID)?.defaultSecondaryText ?? ""
    }
}

/// One advanced option for a Studio mode: a label plus a typed control bound to a `StudioDraft`
/// key path. This is the single source of truth for the depth controls — the Studio sheet renders
/// it, and the adapter maps the same fields to the identical CLI flags the Advanced surface emits.
struct StudioOptionField: Identifiable {
    let id: String
    let label: String
    let control: Control

    enum Control {
        case int(WritableKeyPath<StudioDraft, Int>, range: ClosedRange<Int>, step: Int)
        case double(WritableKeyPath<StudioDraft, Double>)
        case bool(WritableKeyPath<StudioDraft, Bool>)
        case text(WritableKeyPath<StudioDraft, String>, placeholder: String)
    }
}

enum StudioOptionSchema {
    static func fields(for mode: StudioMode) -> [StudioOptionField] {
        switch mode {
        case .chat, .code:
            return [
                StudioOptionField(id: "temperature", label: "Temperature", control: .double(\.temperature)),
                StudioOptionField(id: "maxTokens", label: "Max tokens", control: .int(\.maxTokens, range: 1...32_768, step: 64)),
            ]
        case .createImage:
            return [
                StudioOptionField(id: "cfg", label: "CFG scale", control: .double(\.cfgScale)),
                StudioOptionField(id: "strength", label: "Img2img strength", control: .double(\.strength)),
            ]
        case .listen:
            return [
                StudioOptionField(id: "language", label: "Language", control: .text(\.language, placeholder: "auto")),
                StudioOptionField(id: "backend", label: "Backend", control: .text(\.backend, placeholder: "auto")),
                StudioOptionField(id: "timestamps", label: "Timestamps", control: .bool(\.timestamps)),
            ]
        case .video:
            return [
                StudioOptionField(id: "variant", label: "Variant", control: .text(\.variant, placeholder: "distilled")),
                StudioOptionField(id: "fps", label: "Frames per second", control: .int(\.fps, range: 1...60, step: 1)),
                StudioOptionField(id: "numFrames", label: "Frames", control: .int(\.numFrames, range: 1...600, step: 1)),
                StudioOptionField(id: "strength", label: "Image strength", control: .double(\.strength)),
            ]
        default:
            return []
        }
    }
}

struct StudioRunRequest: Identifiable, Equatable {
    let id: UUID
    let mode: StudioMode
    let templateID: CommandTemplateID
    let template: CommandTemplate
    let draft: CommandDraft
    let createdAt: Date
    /// The conversation this run is a turn of, when chat/code. The run's `id` is a per-turn id;
    /// `conversationID` routes completion back to the owning thread.
    let conversationID: UUID?

    init(
        id: UUID = UUID(),
        mode: StudioMode,
        templateID: CommandTemplateID,
        template: CommandTemplate,
        draft: CommandDraft,
        createdAt: Date = Date(),
        conversationID: UUID? = nil
    ) {
        self.id = id
        self.mode = mode
        self.templateID = templateID
        self.template = template
        self.draft = draft
        self.createdAt = createdAt
        self.conversationID = conversationID
    }

    var expectedOutputURL: URL? {
        guard !draft.outputPath.isBlank else { return nil }
        return URL(fileURLWithPath: NSString(string: draft.outputPath).expandingTildeInPath)
    }
}

enum StudioCapabilityRequirement: Equatable {
    case managedModel(String)
    case unavailable(String)
}

enum StudioCommandError: LocalizedError, Equatable {
    case missingPrompt(String)
    case missingInput(String)
    case missingTemplate(CommandTemplateID)

    var errorDescription: String? {
        switch self {
        case .missingPrompt(let label):
            return "\(label) is required."
        case .missingInput(let label):
            return "Attach \(label.lowercased()) first."
        case .missingTemplate(let id):
            return "No command template found for \(id.rawValue)."
        }
    }
}

enum StudioCommandAdapter {
    static func makeRequest(
        mode: StudioMode,
        draft studioDraft: StudioDraft,
        conversationID: UUID? = nil
    ) throws -> StudioRunRequest {
        let templateID = templateID(for: mode, draft: studioDraft)
        guard let template = CommandCatalog.template(id: templateID) else {
            throw StudioCommandError.missingTemplate(templateID)
        }

        try validate(mode: mode, templateID: templateID, draft: studioDraft)

        var draft = template.defaultDraft()
        let prompt = studioDraft.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let secondary = studioDraft.secondaryText.trimmingCharacters(in: .whitespacesAndNewlines)

        switch mode {
        case .createImage:
            draft.prompt = prompt
            draft.secondaryText = secondary
            draft.inputPath = studioDraft.inputPath
            draft.model = studioDraft.model.isBlank ? draft.model : studioDraft.model
            draft.width = studioDraft.width
            draft.height = studioDraft.height
            draft.steps = studioDraft.steps
            draft.seed = studioDraft.seed
            draft.cfgScale = studioDraft.cfgScale
            draft.strength = studioDraft.strength

        case .chat, .code:
            draft.prompt = prompt
            draft.secondaryText = secondary
            draft.model = studioDraft.model.isBlank ? draft.model : studioDraft.model
            draft.temperature = studioDraft.temperature
            draft.maxTokens = studioDraft.maxTokens
            // Conversation turns stream so the canvas renders tokens live; the reply is
            // accumulated and think-stripped app-side.
            if conversationID != nil { draft.stream = true }
            // Vision chat: a single attached image rides with this turn (chat only, not code).
            if mode == .chat, !studioDraft.inputPath.isBlank { draft.imagePath = studioDraft.inputPath }

        case .speak:
            draft.prompt = prompt
            draft.secondaryText = secondary.isEmpty ? draft.secondaryText : secondary
            draft.model = studioDraft.model.isBlank ? draft.model : studioDraft.model
            draft.voiceMode = studioDraft.voiceMode
            if studioDraft.voiceMode == "clone" {
                draft.voiceProfile = studioDraft.voiceProfile
                draft.refAudioPath = studioDraft.refAudioPath
                draft.saveProfileName = studioDraft.saveProfileName
            }

        case .listen:
            draft.inputPath = studioDraft.inputPath
            draft.model = studioDraft.model.isBlank ? draft.model : studioDraft.model
            draft.language = studioDraft.language
            draft.backend = studioDraft.backend
            draft.timestamps = studioDraft.timestamps

        case .readImage:
            draft.inputPath = studioDraft.inputPath
            draft.prompt = prompt.isEmpty ? draft.prompt : prompt
            draft.model = studioDraft.model.isBlank ? draft.model : studioDraft.model

        case .findObjects, .segment, .track:
            draft.prompt = prompt
            draft.inputPath = studioDraft.inputPath
            draft.model = studioDraft.model.isBlank ? draft.model : studioDraft.model

        case .music:
            draft.prompt = prompt
            draft.secondaryText = secondary
            draft.model = studioDraft.model.isBlank ? draft.model : studioDraft.model
            draft.durationSeconds = studioDraft.durationSeconds
            draft.steps = studioDraft.steps
            draft.seed = studioDraft.seed

        case .video:
            draft.prompt = prompt
            draft.inputPath = studioDraft.inputPath
            draft.model = studioDraft.model.isBlank ? draft.model : studioDraft.model
            draft.width = studioDraft.width
            draft.height = studioDraft.height
            draft.seed = studioDraft.seed
            draft.variant = studioDraft.variant
            draft.fps = studioDraft.fps
            draft.numFrames = studioDraft.numFrames
            draft.strength = studioDraft.strength

        case .sfx:
            draft.prompt = prompt
            draft.model = studioDraft.model.isBlank ? draft.model : studioDraft.model
            draft.durationSeconds = studioDraft.durationSeconds
            draft.steps = studioDraft.steps
            draft.seed = studioDraft.seed
        }

        return StudioRunRequest(
            mode: mode, templateID: templateID, template: template, draft: draft,
            conversationID: conversationID
        )
    }

    static func pullRequest(for mode: StudioMode, draft: StudioDraft) throws -> StudioRunRequest? {
        guard let requirement = capabilityRequirement(for: mode, draft: draft),
              case .managedModel(let model) = requirement else {
            return nil
        }
        guard !model.isBlank else { return nil }
        guard let template = CommandCatalog.template(id: .modelPull) else {
            throw StudioCommandError.missingTemplate(.modelPull)
        }

        var commandDraft = template.defaultDraft()
        commandDraft.model = model
        return StudioRunRequest(mode: mode, templateID: .modelPull, template: template, draft: commandDraft)
    }

    static func requiredModel(for mode: StudioMode, draft: StudioDraft) -> String {
        if !draft.model.isBlank { return draft.model }
        let templateID = templateID(for: mode, draft: draft)
        return CommandCatalog.template(id: templateID)?.defaultModel ?? ""
    }

    static func capabilityRequirement(for mode: StudioMode, draft: StudioDraft) -> StudioCapabilityRequirement? {
        // Read Image actions inspect/caption use a vision-language model the CLI
        // auto-downloads on demand, so they are not gated by the managed catalog; only
        // actions with a managed default model (OCR) require a readiness check.
        if let modelID = managedCapabilityModelID(for: mode, draft: draft) {
            return .managedModel(modelID)
        }

        return nil
    }

    private static func templateID(for mode: StudioMode, draft: StudioDraft) -> CommandTemplateID {
        if mode == .readImage {
            return draft.readImageAction.templateID
        }
        return mode.defaultTemplateID
    }

    private static func managedCapabilityModelID(for mode: StudioMode, draft: StudioDraft) -> String? {
        let model = requiredModel(for: mode, draft: draft)
        if !model.isBlank {
            return model
        }

        if mode == .listen {
            return "speech-asr-parakeet"
        }

        return nil
    }

    private static func validate(
        mode: StudioMode,
        templateID: CommandTemplateID,
        draft: StudioDraft
    ) throws {
        let prompt = draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let input = draft.inputPath.trimmingCharacters(in: .whitespacesAndNewlines)

        if mode.requiresAttachment && input.isEmpty {
            throw StudioCommandError.missingInput(mode.acceptedTypes.first == .audio ? "audio" : mode.acceptedTypes.first == .movie ? "video" : "image")
        }

        // Clone voice needs a source the CLI accepts; otherwise `speech synthesize` hard-fails with
        // "Clone mode requires --profile or --ref-audio." Validate it up front instead.
        if mode == .speak, draft.voiceMode == "clone",
           draft.voiceProfile.isBlank, draft.refAudioPath.isBlank {
            throw StudioCommandError.missingPrompt("A saved voice profile or reference audio")
        }

        let promptRequired: Bool
        switch mode {
        case .listen:
            promptRequired = false
        case .readImage:
            promptRequired = templateID != .visionOCR
        default:
            promptRequired = true
        }

        if promptRequired && prompt.isEmpty {
            throw StudioCommandError.missingPrompt("Prompt")
        }
    }
}

enum StudioLibraryStatus: String, Codable, Equatable {
    case queued
    case running
    case completed
    case failed
}

enum StudioMessageRole: String, Codable, Equatable {
    case user
    case assistant
}

/// One turn in a chat/code conversation. The app owns conversation history (the CLI is
/// stateless per invocation), so these are persisted in the owning `StudioLibraryItem`.
struct StudioMessage: Codable, Identifiable, Equatable {
    let id: UUID
    var role: StudioMessageRole
    var content: String
    var createdAt: Date
    /// True for an assistant turn whose run exited non-zero; the thread is kept either way.
    var failed: Bool
    /// The image attached to this user turn (vision chat), so edit/retry resend it. Optional so
    /// older persisted threads decode unchanged.
    var imagePath: String?

    init(
        id: UUID = UUID(),
        role: StudioMessageRole,
        content: String,
        createdAt: Date = Date(),
        failed: Bool = false,
        imagePath: String? = nil
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.createdAt = createdAt
        self.failed = failed
        self.imagePath = imagePath
    }
}

struct StudioLibraryItem: Codable, Identifiable, Equatable {
    let id: UUID
    var mode: StudioMode
    var prompt: String
    var inputURL: URL?
    var outputURL: URL?
    var createdAt: Date
    var updatedAt: Date
    var status: StudioLibraryStatus
    var exitCode: Int32?
    var commandPreview: String
    var outputText: String?
    var customTitle: String?
    // Conversation channel — non-nil only for .chat / .code threads. Optional + additive so
    // legacy library.json rows (which lack these keys) decode unchanged with nil.
    var messages: [StudioMessage]? = nil
    var systemPrompt: String? = nil
    var model: String? = nil

    var displayTitle: String {
        if let customTitle, !customTitle.isBlank { return customTitle }
        if let firstUser = messages?.first(where: { $0.role == .user })?.content,
           !firstUser.isBlank {
            return firstUser
        }
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { return trimmed }
        return mode.title
    }

    var isConversation: Bool {
        messages != nil
    }
}

enum ModelReadinessState: Equatable {
    case checking
    case ready
    case missingModel(String)
    case unsupported(String)
    case unknown(String)

    var blocksRun: Bool {
        switch self {
        case .checking, .missingModel, .unsupported, .unknown:
            return true
        default:
            return false
        }
    }

    var canPull: Bool {
        if case .missingModel = self {
            return true
        }
        return false
    }

    var isChecking: Bool {
        if case .checking = self {
            return true
        }
        return false
    }

    var title: String {
        switch self {
        case .checking: return "Checking"
        case .ready: return "Ready"
        case .missingModel: return "Model needed"
        case .unsupported: return "Unsupported"
        case .unknown: return "Not checked"
        }
    }

    var message: String {
        switch self {
        case .checking:
            return "Checking local model availability."
        case .ready:
            return "This mode is ready to run locally."
        case .missingModel(let model):
            return "Download \(model) before running this mode."
        case .unsupported(let reason):
            return reason
        case .unknown(let reason):
            return reason
        }
    }
}

struct StudioModelCapability: Equatable {
    let modelID: String
    let isSupported: Bool
    let minimumUnifiedMemoryGB: Int?
    let recommendedUnifiedMemoryGB: Int?
    let download: String?
    let reason: String?

    var unavailableMessage: String? {
        guard !isSupported else { return nil }
        if let reason, !reason.isBlank {
            return reason
        }
        if let minimumUnifiedMemoryGB {
            return "Requires at least \(minimumUnifiedMemoryGB) GB unified memory."
        }
        return "\(modelID) is not supported on this Mac."
    }
}

struct StudioModelCapabilityReport: Equatable {
    let capabilitiesByID: [String: StudioModelCapability]
    let recommendedChatModelID: String?
    let recommendedCodeModelID: String?
}

enum ModelCapabilitiesParser {
    static func report(from output: String) -> StudioModelCapabilityReport {
        if let jsonReport = jsonReport(from: output) {
            return jsonReport
        }
        return StudioModelCapabilityReport(
            capabilitiesByID: capabilities(from: output),
            recommendedChatModelID: nil,
            recommendedCodeModelID: nil
        )
    }

    static func capabilities(from output: String) -> [String: StudioModelCapability] {
        var capabilities: [String: StudioModelCapability] = [:]
        var current: CapabilityBuilder?

        func flushCurrent() {
            guard let built = current?.build() else { return }
            capabilities[built.modelID] = built
        }

        for line in output.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }

            if trimmed.hasPrefix("- ") {
                flushCurrent()
                current = CapabilityBuilder(header: trimmed)
                continue
            }

            guard var builder = current else { continue }
            builder.read(fieldLine: trimmed)
            current = builder
        }

        flushCurrent()
        return capabilities
    }

    private static func jsonReport(from output: String) -> StudioModelCapabilityReport? {
        guard let data = jsonObjectData(in: output) else { return nil }

        struct Payload: Decodable {
            struct ChatBand: Decodable {
                let modelID: String
            }
            struct RecommendedModel: Decodable {
                let id: String?
                let modelID: String?
            }
            struct Model: Decodable {
                let id: String
                let supported: Bool
                let minimumUnifiedMemoryGB: Int?
                let recommendedUnifiedMemoryGB: Int?
                let download: String?
                let reasons: [String]?
            }

            let recommendedChatModel: ChatBand?
            let recommendedCodeModel: RecommendedModel?
            let models: [Model]
        }

        guard let payload = try? JSONDecoder().decode(Payload.self, from: data) else { return nil }
        let capabilities = Dictionary(uniqueKeysWithValues: payload.models.map { model in
            (
                model.id,
                StudioModelCapability(
                    modelID: model.id,
                    isSupported: model.supported,
                    minimumUnifiedMemoryGB: model.minimumUnifiedMemoryGB,
                    recommendedUnifiedMemoryGB: model.recommendedUnifiedMemoryGB,
                    download: model.download,
                    reason: model.reasons?.joined(separator: " ")
                )
            )
        })
        return StudioModelCapabilityReport(
            capabilitiesByID: capabilities,
            recommendedChatModelID: payload.recommendedChatModel?.modelID,
            recommendedCodeModelID: payload.recommendedCodeModel?.modelID
                ?? payload.recommendedCodeModel?.id
        )
    }

    private static func jsonObjectData(in text: String) -> Data? {
        guard let start = text.firstIndex(of: "{"),
              let end = text.lastIndex(of: "}"),
              start < end else { return nil }
        return String(text[start...end]).data(using: .utf8)
    }
}

private struct CapabilityBuilder {
    var modelID: String
    var isSupported: Bool
    var minimumUnifiedMemoryGB: Int?
    var recommendedUnifiedMemoryGB: Int?
    var download: String?
    var reason: String?

    init?(header: String) {
        let statusSuffix: String
        if header.hasSuffix("[supported]") {
            statusSuffix = "[supported]"
            isSupported = true
        } else if header.hasSuffix("[unsupported]") {
            statusSuffix = "[unsupported]"
            isSupported = false
        } else {
            return nil
        }

        let id = header
            .dropFirst(2)
            .dropLast(statusSuffix.count)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !id.isEmpty else { return nil }
        modelID = id
    }

    mutating func read(fieldLine: String) {
        if fieldLine.hasPrefix("memory:") {
            let values = Self.memoryValues(from: fieldLine)
            minimumUnifiedMemoryGB = values.minimum
            recommendedUnifiedMemoryGB = values.recommended
        } else if fieldLine.hasPrefix("download:") {
            download = Self.value(after: "download:", in: fieldLine)
        } else if fieldLine.hasPrefix("reason:") {
            reason = Self.value(after: "reason:", in: fieldLine)
        }
    }

    func build() -> StudioModelCapability {
        StudioModelCapability(
            modelID: modelID,
            isSupported: isSupported,
            minimumUnifiedMemoryGB: minimumUnifiedMemoryGB,
            recommendedUnifiedMemoryGB: recommendedUnifiedMemoryGB,
            download: download,
            reason: reason
        )
    }

    private static func value(after prefix: String, in line: String) -> String {
        String(line.dropFirst(prefix.count)).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func memoryValues(from line: String) -> (minimum: Int?, recommended: Int?) {
        let parts = line
            .replacingOccurrences(of: "memory:", with: "")
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }

        let minimum = parts.first { $0.hasPrefix("minimum ") }.flatMap { Self.firstInteger(in: $0) }
        let recommended = parts.first { $0.hasPrefix("recommended ") }.flatMap { Self.firstInteger(in: $0) }
        return (minimum, recommended)
    }

    private static func firstInteger(in text: String) -> Int? {
        let digits = text.drop { !$0.isNumber }.prefix { $0.isNumber }
        return Int(digits)
    }
}

/// A parsed progress update emitted by long-running CLI commands (e.g. `model pull`).
struct StudioRunProgress: Equatable {
    let label: String
    /// 0...1 when a percentage is known; nil for indeterminate (bytes-only) progress.
    let fractionCompleted: Double?
    /// Human-readable detail such as "1.2 GB / 3.4 GB" or "(45 MB/s)" or "extracting…".
    let detail: String?
}

/// Parses the carriage-return progress lines the CLI writes for downloads/extraction, e.g.
/// `[image-zimage-nano] 45%  1.2 GB / 3.4 GB` or `[image-zimage-nano] 45%  (45 MB/s)`.
enum StudioProgressParser {
    static func parse(_ rawLine: String) -> StudioRunProgress? {
        let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
        guard line.hasPrefix("["), let close = line.firstIndex(of: "]") else { return nil }
        let label = String(line[line.index(after: line.startIndex)..<close])
            .trimmingCharacters(in: .whitespaces)
        guard !label.isEmpty else { return nil }

        var rest = String(line[line.index(after: close)...]).trimmingCharacters(in: .whitespaces)
        guard !rest.isEmpty else { return nil }

        var fraction: Double?
        let tokens = rest.split(separator: " ").map(String.init)
        if let first = tokens.first, first.hasSuffix("%"), let pct = Double(first.dropLast()) {
            fraction = min(100, max(0, pct)) / 100.0
            rest = tokens.dropFirst().joined(separator: " ").trimmingCharacters(in: .whitespaces)
        }

        let collapsed = rest
            .replacingOccurrences(of: "  ", with: " ")
            .trimmingCharacters(in: .whitespaces)
        let detail = collapsed.isEmpty ? nil : collapsed

        let looksLikeProgress = fraction != nil
            || collapsed.contains("/")
            || collapsed.lowercased().contains("extracting")
            || collapsed.range(
                of: #"\b\d+(\.\d+)?\s?(B|KB|MB|GB|TB)\b"#,
                options: .regularExpression
            ) != nil
        guard looksLikeProgress else { return nil }

        return StudioRunProgress(label: label, fractionCompleted: fraction, detail: detail)
    }
}

/// Extracts the output-artifact path a `mere.run` command reports on stdout.
///
/// The CLI's contract (AGENTS.md: stdout is machine-readable, stderr is diagnostic) is that
/// media commands print the artifact path as a bare line, e.g. `/Users/me/out.png`, while the
/// directory/OCR commands print `input -> output` pairs. Progress and logs go to stderr, so a
/// media command's stdout is effectively just its result path(s). Transcription prints
/// transcript text and reports no path here — the explicit `--output` location covers it.
enum StudioResultParser {
    /// Output-artifact paths parsed from `stdout`, most-recently-emitted first.
    static func outputPaths(fromStdout stdout: String) -> [String] {
        var paths: [String] = []
        let lines = stdout
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        for line in lines.reversed() {
            if let separator = line.range(of: " -> ") ?? line.range(of: " → ") {
                // `input -> output` pair (vision ocr/caption): the artifact is the right side.
                let rhs = String(line[separator.upperBound...]).trimmingCharacters(in: .whitespaces)
                if isPathLike(rhs) { paths.append(rhs) }
            } else if isPathLike(line) {
                paths.append(line)
            }
        }
        return paths
    }

    /// True when a line is an absolute or tilde-rooted filesystem path rather than prose.
    static func isPathLike(_ candidate: String) -> Bool {
        candidate.hasPrefix("/") || candidate.hasPrefix("~/")
    }
}

enum ModelReadinessParser {
    static func state(for modelID: String, modelListOutput: String) -> ModelReadinessState {
        guard !modelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return .ready
        }

        let rows = modelListOutput
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        for row in rows {
            let fields = row.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
            guard fields.first == modelID else { continue }

            let normalized = row.lowercased()
            if normalized.contains("unsupported") {
                return .unsupported("\(modelID) is listed as unsupported on this Mac.")
            }
            if normalized.contains("installed") || normalized.contains("ready") || normalized.contains("present") {
                return .ready
            }
            return .missingModel(modelID)
        }

        return .unknown("Run model capabilities or configure model sources to check \(modelID).")
    }
}

/// A parsed snapshot of `mere.run status --json` for the Studio status pill.
struct StudioServerStatus: Equatable {
    let health: String
    let loadedModels: [String]
    let installedCount: Int

    var isReachable: Bool {
        let normalized = health.lowercased()
        return normalized == "ok" || normalized == "up" || normalized == "healthy"
            || normalized == "reachable" || !loadedModels.isEmpty
    }

    var loadedModelSummary: String? {
        loadedModels.first
    }

    static func parse(jsonStdout: String) -> StudioServerStatus? {
        guard let data = jsonObjectData(in: jsonStdout) else { return nil }
        struct Snapshot: Decodable {
            struct Server: Decodable {
                let health: String?
                let loadedModels: [String]?
            }
            struct InstalledModel: Decodable { let id: String }
            let server: Server?
            let installedModels: [InstalledModel]?
        }
        guard let snapshot = try? JSONDecoder().decode(Snapshot.self, from: data) else { return nil }
        return StudioServerStatus(
            health: snapshot.server?.health ?? "unknown",
            loadedModels: snapshot.server?.loadedModels ?? [],
            installedCount: snapshot.installedModels?.count ?? 0
        )
    }

    /// The CLI may emit diagnostics around the JSON; isolate the top-level object.
    private static func jsonObjectData(in text: String) -> Data? {
        guard let start = text.firstIndex(of: "{"),
              let end = text.lastIndex(of: "}"),
              start < end else { return nil }
        return String(text[start...end]).data(using: .utf8)
    }
}

/// One offline `guide` topic from `guide --list --json`, with the command path used to fetch it.
struct StudioGuideTopic: Identifiable, Equatable {
    let id: String
    let title: String
    let commandPath: [String]

    static func parse(listJSON: String) -> [StudioGuideTopic] {
        guard let data = jsonArrayData(in: listJSON) else { return [] }
        struct Item: Decodable { let topic: String; let title: String; let commands: [String] }
        guard let items = try? JSONDecoder().decode([Item].self, from: data) else { return [] }
        return items.map { item in
            let path = (item.commands.first ?? item.topic).split(separator: " ").map(String.init)
            return StudioGuideTopic(id: item.topic, title: item.title, commandPath: path)
        }
    }

    static func parseContent(payloadJSON: String) -> String? {
        guard let data = jsonObjectData(in: payloadJSON) else { return nil }
        struct Payload: Decodable { let content: String }
        return (try? JSONDecoder().decode(Payload.self, from: data))?.content
    }

    private static func jsonArrayData(in text: String) -> Data? {
        guard let start = text.firstIndex(of: "["),
              let end = text.lastIndex(of: "]"),
              start < end else { return nil }
        return String(text[start...end]).data(using: .utf8)
    }

    private static func jsonObjectData(in text: String) -> Data? {
        guard let start = text.firstIndex(of: "{"),
              let end = text.lastIndex(of: "}"),
              start < end else { return nil }
        return String(text[start...end]).data(using: .utf8)
    }
}

/// A saved voice-clone profile from `speech profile list` (tab-separated id/name/updatedAt).
struct StudioVoiceProfile: Identifiable, Equatable {
    let id: String
    let name: String
    let updatedAt: String

    static func parse(listOutput: String) -> [StudioVoiceProfile] {
        listOutput.components(separatedBy: .newlines).compactMap { line in
            let parts = line.components(separatedBy: "\t")
            guard parts.count >= 2 else { return nil }
            let id = parts[0].trimmingCharacters(in: .whitespaces)
            let name = parts[1].trimmingCharacters(in: .whitespaces)
            guard !id.isEmpty, !name.isEmpty else { return nil }
            return StudioVoiceProfile(id: id, name: name, updatedAt: parts.count > 2 ? parts[2] : "")
        }
    }
}
