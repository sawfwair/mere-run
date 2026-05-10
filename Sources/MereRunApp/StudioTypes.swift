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

    var id: String { rawValue }

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
        }
    }

    var acceptedTypes: [UTType] {
        switch self {
        case .createImage, .readImage, .findObjects, .segment:
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

    mutating func reset(for mode: StudioMode) {
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

struct StudioRunRequest: Identifiable, Equatable {
    let id: UUID
    let mode: StudioMode
    let templateID: CommandTemplateID
    let template: CommandTemplate
    let draft: CommandDraft
    let createdAt: Date

    init(
        id: UUID = UUID(),
        mode: StudioMode,
        templateID: CommandTemplateID,
        template: CommandTemplate,
        draft: CommandDraft,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.mode = mode
        self.templateID = templateID
        self.template = template
        self.draft = draft
        self.createdAt = createdAt
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
    static func makeRequest(mode: StudioMode, draft studioDraft: StudioDraft) throws -> StudioRunRequest {
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

        case .chat, .code:
            draft.prompt = prompt
            draft.secondaryText = secondary
            draft.model = studioDraft.model.isBlank ? draft.model : studioDraft.model

        case .speak:
            draft.prompt = prompt
            draft.secondaryText = secondary.isEmpty ? draft.secondaryText : secondary
            draft.model = studioDraft.model.isBlank ? draft.model : studioDraft.model

        case .listen:
            draft.inputPath = studioDraft.inputPath
            draft.model = studioDraft.model.isBlank ? draft.model : studioDraft.model

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
        }

        return StudioRunRequest(mode: mode, templateID: templateID, template: template, draft: draft)
    }

    static func pullRequest(for mode: StudioMode, draft: StudioDraft) throws -> StudioRunRequest? {
        guard let model = managedCapabilityModelID(for: mode, draft: draft) else {
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
        if let modelID = managedCapabilityModelID(for: mode, draft: draft) {
            return .managedModel(modelID)
        }

        if mode == .readImage {
            switch draft.readImageAction {
            case .inspect, .caption:
                return .unavailable(unmanagedVisionLanguageMessage(for: draft.readImageAction))
            case .ocr:
                return nil
            }
        }

        return nil
    }

    private static func unmanagedVisionLanguageMessage(for action: StudioReadImageAction) -> String {
        "\(action.title) uses an automatic vision-language model download "
            + "that is not listed in the managed capability catalog yet."
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
    case running
    case completed
    case failed
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

    var displayTitle: String {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { return trimmed }
        return mode.title
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
        case .checking, .missingModel, .unsupported:
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

enum ModelCapabilitiesParser {
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
