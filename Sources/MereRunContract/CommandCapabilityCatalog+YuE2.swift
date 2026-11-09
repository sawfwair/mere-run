import Foundation

extension MereRunCapabilityCatalog {
    static var musicYuE2: [MereRunCommandCapability] {
        let stageOptions: [MereRunCapabilityOption] = [
            .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string, defaultValue: "music-yue2"),
            .init(flag: "--output", aliases: ["-o"], label: "Output directory", kind: .directory, required: true),
            .init(flag: "--memory-mode", label: "Memory mode", kind: .choice, choices: ["staged", "resident", "low-memory"], defaultValue: "staged")
        ]
        var commands: [MereRunCommandCapability] = []
        for (name, summary) in [("plan", "Save an exact score stage from a native JSON request."),
                                ("semantic", "Generate music tokens from a saved score."),
                                ("synthesize", "Generate latents from saved music tokens."),
                                ("export", "Copy a complete local pipeline for offline use.")] {
            commands.append(yue2(name, summary, input: ["semantic", "synthesize"].contains(name) ? "input" : nil,
                options: stageOptions + (name == "plan" ? [.init(flag: "--request", label: "Request JSON", kind: .file, required: true)] : [])))
        }
        commands += [
            yue2("inspect", "Verify saved stage artifacts without loading models.", input: "input", output: .text),
            yue2("validate", "Validate a native YuE2 JSON request without loading models.", input: "request", options: [
                .init(flag: "--tokenizer", label: "Tokenizer for context preflight", kind: .file)], output: .text),
            yue2("decode", "Decode saved NumPy latents using only the VAE.", input: "input", options: [
                .init(flag: "--vae", label: "VAE directory or benchmark model ID", kind: .string),
                .init(flag: "--output", aliases: ["-o"], label: "WAV output", kind: .file, required: true),
                .init(flag: "--core-frames", label: "VAE tile frames", kind: .integer),
                .init(flag: "--full", label: "Decode the complete sequence", kind: .boolean)]),
            yue2("instrumentalize", "Transfer vocal score notes into the instrumental voice.", input: "input", options: [
                .init(flag: "--output", aliases: ["-o"], label: "ABC output", kind: .file, required: true),
                .init(flag: "--overlap", label: "Overlapping accompaniment", kind: .choice, choices: ["vocal", "error"], defaultValue: "vocal"),
                .init(flag: "--drop-chords", label: "Remove chords", kind: .boolean)]),
            yue2("batch", "Run JSONL requests and report individual failures.", input: "input", options: stageOptions + [
                .init(flag: "--validate-only", label: "Validate without generation", kind: .boolean)]),
            yue2("transcribe", "Transcribe recordings with native SheetSage2 and MERT2.", input: "input", options: [
                .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string, defaultValue: "music-sheetsage2"),
                .init(flag: "--output", aliases: ["-o"], label: "Output directory", kind: .directory, required: true),
                .init(flag: "--melody-only", label: "Remove chord conditioning", kind: .boolean),
                .init(flag: "--allow-notation-failure", label: "Keep raw analysis without a score", kind: .boolean),
                .init(flag: "--overlap", label: "Overlap seconds", kind: .number, defaultValue: "200.0"),
                .init(flag: "--lookahead", label: "Lookahead seconds", kind: .number, defaultValue: "100.0")]),
            yue2("cover", "Transcribe a recording and render a new interpretation.", input: "input", options: [
                .init(flag: "--model", aliases: ["-m"], label: "YuE2 model", kind: .string, defaultValue: "music-yue2"),
                .init(flag: "--transcriber-model", label: "Transcriber model", kind: .string, defaultValue: "music-sheetsage2"),
                .init(flag: "--request", label: "Request JSON", kind: .file, required: true),
                .init(flag: "--output", aliases: ["-o"], label: "Output directory", kind: .directory, required: true),
                .init(flag: "--full-score", label: "Keep source harmony", kind: .boolean)]),
            yue2("serve", "Serve queued native jobs with reusable artifacts.", options: [
                .init(flag: "--model", aliases: ["-m"], label: "Model", kind: .string, defaultValue: "music-yue2"),
                .init(flag: "--host", label: "Host", kind: .string, defaultValue: "127.0.0.1"),
                .init(flag: "--port", label: "Port", kind: .integer, defaultValue: "8082"),
                .init(flag: "--output", aliases: ["-o"], label: "Artifact directory", kind: .directory, required: true),
                .init(flag: "--api-key", label: "API key", kind: .string),
                .init(flag: "--max-pending", label: "Queue capacity", kind: .integer, defaultValue: "16"),
                .init(flag: "--memory-mode", label: "Memory mode", kind: .choice,
                      choices: ["staged", "resident", "low-memory"], defaultValue: "resident")], output: .service)
        ]
        return commands
    }

    private static func yue2(_ name: String, _ summary: String, input: String? = nil,
                             options: [MereRunCapabilityOption] = [], output: MereRunCapabilityOutputKind = .file) -> MereRunCommandCapability {
        let routing: MereRunCapabilityRouting?
        if name == "transcribe" {
            routing = .init(modelFlags: ["--model"], defaultModels: [.always("music-sheetsage2")],
                families: [.init(id: "sheetsage2", title: "SheetSage2", models: ["music-sheetsage2"])])
        } else if name == "decode" {
            routing = .init(modelFlags: ["--vae"], defaultModels: [.always("music-yue2")], families: [
                .init(id: "yue2-vae-standard", title: "YuE2 listening VAE", models: ["music-yue2"]),
                .init(id: "yue2-vae-legacy", title: "YuE2 benchmark VAE", models: ["music-yue2-vae-legacy"])])
        } else { routing = nil }
        return MereRunCommandCapability(id: "music.yue2.\(name)", command: ["music", "yue2", name],
            title: "YuE2 \(name)", summary: summary,
            arguments: input.map { [.init(name: $0, label: "Input",
                kind: ["semantic", "synthesize", "inspect"].contains(name) ? .directory : .file, required: true)] } ?? [],
            options: options, output: .init(kind: output, flag: options.contains { $0.flag == "--output" } ? "--output" : nil), routing: routing)
    }
}
