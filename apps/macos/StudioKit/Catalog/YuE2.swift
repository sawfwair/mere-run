import Foundation
import UniformTypeIdentifiers

extension CommandCatalog {
    package static let yue2Templates: [CommandTemplate] = [
        CommandTemplate(id: .musicYue2Plan, category: .media, title: "Plan YuE2 score", subtitle: "Save an editable composition from a JSON request",
            systemImage: "music.note.list", inputKind: .file([.json]), outputKind: .directory, defaultModel: "music-yue2"),
        CommandTemplate(id: .musicYue2Semantic, category: .media, title: "Generate YuE2 music tokens", subtitle: "Continue a saved score stage",
            systemImage: "music.note", inputKind: .directory, outputKind: .directory, defaultModel: "music-yue2"),
        CommandTemplate(id: .musicYue2Synthesize, category: .media, title: "Synthesize YuE2 latents", subtitle: "Continue a saved semantic stage",
            systemImage: "waveform", inputKind: .directory, outputKind: .directory, defaultModel: "music-yue2"),
        CommandTemplate(id: .musicYue2Decode, category: .media, title: "Decode YuE2 audio", subtitle: "Replay saved NumPy latents with only the VAE",
            systemImage: "waveform", inputKind: .file([.data]), outputKind: .file("wav")),
        CommandTemplate(id: .musicYue2Inspect, category: .media, title: "Inspect YuE2 stage", subtitle: "Verify saved artifacts without loading models",
            systemImage: "checkmark.seal", inputKind: .directory),
        CommandTemplate(id: .musicYue2Validate, category: .media, title: "Validate YuE2 request", subtitle: "Check a JSON request without loading models",
            systemImage: "checkmark.seal", inputKind: .file([.json])),
        CommandTemplate(id: .musicYue2Instrumentalize, category: .media, title: "Convert instrumental score", subtitle: "Move vocal score notes into the instrumental voice",
            systemImage: "pianokeys", inputKind: .file([.plainText]), outputKind: .file("abc")),
        CommandTemplate(id: .musicYue2Transcribe, category: .media, title: "Transcribe music to score", subtitle: "Experimental native SheetSage2 and MERT2",
            systemImage: "music.note.list", inputKind: .audio, outputKind: .directory, defaultModel: "music-sheetsage2"),
        CommandTemplate(id: .musicYue2Cover, category: .media, title: "Create YuE2 cover", subtitle: "Transcribe a recording and render a new interpretation",
            systemImage: "music.note", promptLabel: "Request JSON path", inputKind: .audio, outputKind: .directory, defaultModel: "music-yue2"),
        CommandTemplate(id: .musicYue2Export, category: .media, title: "Export YuE2 pipeline", subtitle: "Copy weights and licenses for offline use",
            systemImage: "shippingbox", outputKind: .directory, defaultModel: "music-yue2"),
        CommandTemplate(id: .musicYue2Batch, category: .media, title: "Run YuE2 batch", subtitle: "Process JSONL requests with individual outcomes",
            systemImage: "square.stack", inputKind: .file([.plainText]), outputKind: .directory, defaultModel: "music-yue2"),
        CommandTemplate(id: .musicYue2Serve, category: .media, title: "Serve YuE2 jobs", subtitle: "Queue local generation with reusable artifacts",
            systemImage: "server.rack", outputKind: .directory, defaultModel: "music-yue2")
    ]
}

extension CommandArguments {
    package static func musicYue2(_ id: CommandTemplateID, draft: CommandDraft) -> [String] {
        guard let capability = id.capability else { return [] }
        var arguments = ArgumentBuilder(capability.command)
        if id == .musicYue2Plan {
            if !draft.inputPath.isBlank { arguments.option(CommandFlags.MusicYue2Plan.request, draft.inputPath) }
        } else if !capability.arguments.isEmpty {
            arguments.value(draft.inputPath)
        }
        if id == .musicYue2Cover, !draft.prompt.isBlank { arguments.option(CommandFlags.MusicYue2Cover.request, draft.prompt) }
        if capability.options.contains(where: { $0.flag == CommandFlags.MusicGenerate.model }), !draft.model.isBlank {
            arguments.option(CommandFlags.MusicGenerate.model, draft.model)
        }
        if capability.options.contains(where: { $0.flag == CommandFlags.MusicGenerate.output }), !draft.outputPath.isBlank {
            arguments.option(CommandFlags.MusicGenerate.output, draft.outputPath)
        }
        return arguments.arguments
    }
}
