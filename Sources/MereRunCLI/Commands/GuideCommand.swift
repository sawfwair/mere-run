import ArgumentParser
import Foundation

struct GuideCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "guide",
        abstract: "Read offline mere.run command cookbooks.",
        discussion: """
        Prints Markdown guidance for a command topic, or JSON when --json is set.

        Examples:
          mere.run guide --list
          mere.run guide --list --markdown > guides.md
          mere.run guide music generate
          mere.run guide image generate --model image-zimage-nano
          mere.run guide music generate --json
        """
    )

    @Flag(name: [.long], help: "List available guide topics.")
    var list: Bool = false

    @Option(name: [.long], help: "Focus the guide on a supported managed model id.")
    var model: String?

    @Flag(name: [.long], help: "Emit JSON instead of Markdown.")
    var json: Bool = false

    @Flag(name: [.long], help: "Render the topic list as a Markdown table (redirect to a .md file).")
    var markdown: Bool = false

    @Argument(help: "Command path to read, for example: music generate.")
    var commandPath: [String] = []

    func run() throws {
        if list || commandPath.isEmpty {
            if markdown {
                print(try Self.renderListMarkdown())
            } else {
                print(try Self.renderList(json: json))
            }
            return
        }

        let entry = try Self.resolveEntry(commandPath: commandPath, model: model)
        print(try Self.render(entry: entry, model: model, json: json))
    }

    static func resolveEntry(commandPath: [String], model: String?) throws -> GuideTopic {
        guard let entry = GuideRegistry.topic(matching: commandPath) else {
            let requested = GuideRegistry.normalizedCommandPath(commandPath).joined(separator: " ")
            throw ValidationError(
                """
                Unknown guide topic: \(requested)
                Run `mere.run guide --list` to see available topics.
                """
            )
        }

        try GuideRegistry.validate(model: model, for: entry)
        return entry
    }

    static func render(entry: GuideTopic, model: String?, json: Bool) throws -> String {
        let rawContent = try GuideRegistry.content(for: entry)
        let content: String
        if let model, !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            content = "> Model focus: `\(model)`\n\n" + rawContent
        } else {
            content = rawContent
        }

        if json {
            let payload = GuidePayload(
                topic: entry.topic,
                title: entry.title,
                commands: entry.commands,
                models: entry.models,
                content: content
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(payload)
            guard let rendered = String(data: data, encoding: .utf8) else {
                throw ValidationError("Failed to encode guide JSON as UTF-8.")
            }
            return rendered
        }

        return content
    }

    static func renderList(json: Bool) throws -> String {
        let topics = GuideRegistry.all
        if json {
            let payload = topics.map {
                GuideListItem(
                    topic: $0.topic,
                    title: $0.title,
                    commands: $0.commands,
                    models: $0.models
                )
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(payload)
            guard let rendered = String(data: data, encoding: .utf8) else {
                throw ValidationError("Failed to encode guide list JSON as UTF-8.")
            }
            return rendered
        }

        let rows = topics.map { topic in
            GuideListRow(
                topic: topic.topic,
                command: topic.commands.first ?? "",
                models: topic.models.isEmpty ? "-" : topic.models.joined(separator: ", "),
                title: topic.title
            )
        }
        let widths = GuideListWidths(rows: rows)
        var lines = [
            GuideListRow(
                topic: "Topic",
                command: "Command",
                models: "Models",
                title: "Title"
            ).render(widths: widths),
            String(repeating: "-", count: widths.total)
        ]
        lines.append(contentsOf: rows.map { $0.render(widths: widths) })
        lines.append("")
        lines.append("Read one with: mere.run guide <command path>")
        return lines.joined(separator: "\n")
    }

    static func renderListMarkdown() throws -> String {
        let topics = GuideRegistry.all
        var lines = [
            "| Topic | Command | Models | Title | Description |",
            "| --- | --- | --- | --- | --- |",
        ]
        for topic in topics {
            let command = topic.commands.first ?? ""
            // Use <br> so long model lists wrap inside the table cell instead of widening it.
            let models = topic.models.isEmpty ? "-" : topic.models.joined(separator: "<br>")
            let description = GuideRegistry.purposeSummary(for: topic)
            lines.append(
                "| \(escapeCell(topic.topic)) | `\(escapeCell(command))` "
                    + "| \(escapeCell(models)) | \(escapeCell(topic.title)) | \(escapeCell(description)) |"
            )
        }
        return lines.joined(separator: "\n")
    }

    /// Escape characters that would break a Markdown table cell.
    private static func escapeCell(_ value: String) -> String {
        value
            .replacingOccurrences(of: "|", with: "\\|")
            .replacingOccurrences(of: "\n", with: " ")
    }
}

struct GuideTopic: Equatable {
    let topic: String
    let title: String
    let commandPaths: [[String]]
    let models: [String]
    let resourceName: String

    var commands: [String] {
        commandPaths.map { $0.joined(separator: " ") }
    }
}

enum GuideRegistry {
    static let all: [GuideTopic] = [
        GuideTopic(
            topic: "image-generate",
            title: "Image Generate",
            commandPaths: [["image", "generate"]],
            models: [
                "image-zimage-nano",
                "image-zimage-max",
                "image-zimage-base",
                "image-klein-nano",
                "image-klein-max",
                "image-klein-base",
                "image-bonsai-binary",
                "image-bonsai-ternary",
            ],
            resourceName: "image-generate.md"
        ),
        GuideTopic(
            topic: "image-validate",
            title: "Image Validate",
            commandPaths: [["image", "validate"]],
            models: [
                "image-zimage-nano",
                "image-zimage-max",
                "image-klein-max",
            ],
            resourceName: "image-validate.md"
        ),
        GuideTopic(
            topic: "text-chat",
            title: "Text Chat",
            commandPaths: [["text", "chat"]],
            models: [
                "text-chat-gemma4",
                "text-chat-gemma4-12b",
                "text-chat-gemma4-nano",
                "text-chat-gemma4-max",
                "text-chat-q36-nano",
                "text-chat-lfm25-a1b-8bit",
                "text-chat-psi-agent",
            ],
            resourceName: "text-chat.md"
        ),
        GuideTopic(
            topic: "text-code",
            title: "Text Code",
            commandPaths: [["text", "code"]],
            models: ["text-code-qwen3"],
            resourceName: "text-code.md"
        ),
        GuideTopic(
            topic: "text-embed",
            title: "Text Embed",
            commandPaths: [["text", "embed"]],
            models: ["text-embed-qwen3-0.6b"],
            resourceName: "text-embed.md"
        ),
        GuideTopic(
            topic: "text-anonymize",
            title: "Text Anonymize",
            commandPaths: [["text", "anonymize"]],
            models: ["text-anonymize-privacy-filter"],
            resourceName: "text-anonymize.md"
        ),
        GuideTopic(
            topic: "speech-synthesize",
            title: "Speech Synthesize",
            commandPaths: [["speech", "synthesize"]],
            models: [
                "speech-tts-qwen3-nano",
                "speech-tts-qwen3-customvoice",
            ],
            resourceName: "speech-synthesize.md"
        ),
        GuideTopic(
            topic: "speech-transcribe",
            title: "Speech Transcribe",
            commandPaths: [["speech", "transcribe"]],
            models: [
                "speech-asr-parakeet",
                "speech-asr-qwen3",
            ],
            resourceName: "speech-transcribe.md"
        ),
        GuideTopic(
            topic: "speech-profile",
            title: "Speech Profile",
            commandPaths: [
                ["speech", "profile"],
                ["speech", "profile", "list"],
                ["speech", "profile", "create"],
                ["speech", "profile", "delete"],
            ],
            models: [
                "speech-tts-qwen3-customvoice",
                "speech-asr-parakeet",
                "speech-asr-qwen3",
            ],
            resourceName: "speech-profile.md"
        ),
        GuideTopic(
            topic: "vision-caption",
            title: "Vision Caption",
            commandPaths: [["vision", "caption"]],
            models: [],
            resourceName: "vision-caption.md"
        ),
        GuideTopic(
            topic: "vision-inspect",
            title: "Vision Inspect",
            commandPaths: [["vision", "inspect"]],
            models: [],
            resourceName: "vision-inspect.md"
        ),
        GuideTopic(
            topic: "vision-ground",
            title: "Vision Ground",
            commandPaths: [["vision", "ground"]],
            models: ["vision-ground-falcon-perception"],
            resourceName: "vision-ground.md"
        ),
        GuideTopic(
            topic: "vision-segment",
            title: "Vision Segment",
            commandPaths: [["vision", "segment"]],
            models: ["vision-segment-sam31"],
            resourceName: "vision-segment.md"
        ),
        GuideTopic(
            topic: "vision-track",
            title: "Vision Track",
            commandPaths: [["vision", "track"]],
            models: ["vision-segment-sam31"],
            resourceName: "vision-track.md"
        ),
        GuideTopic(
            topic: "vision-track-live",
            title: "Vision Track Live",
            commandPaths: [["vision", "track-live"]],
            models: ["vision-segment-sam31"],
            resourceName: "vision-track-live.md"
        ),
        GuideTopic(
            topic: "vision-ocr",
            title: "Vision OCR",
            commandPaths: [["vision", "ocr"]],
            models: ["vision-ocr-lighton"],
            resourceName: "vision-ocr.md"
        ),
        GuideTopic(
            topic: "music-generate",
            title: "Music Generate",
            commandPaths: [["music", "generate"]],
            models: [
                "music-acestep",
                "music-magenta-rt2-small",
                "music-magenta-rt2-base",
            ],
            resourceName: "music-generate.md"
        ),
        GuideTopic(
            topic: "video-generate",
            title: "Video Generate",
            commandPaths: [["video", "generate"]],
            models: ["video-ltx-av"],
            resourceName: "video-generate.md"
        ),
        GuideTopic(
            topic: "video-export-latents",
            title: "Video Export Latents",
            commandPaths: [["video", "export-latents"]],
            models: ["video-ltx-av"],
            resourceName: "video-export-latents.md"
        ),
        GuideTopic(
            topic: "api-serve",
            title: "API Serve",
            commandPaths: [["api", "serve"]],
            models: [
                "text-code-qwen3",
                "text-chat-gemma4",
                "text-chat-gemma4-12b",
                "vision-chat-gemma4-12b",
                "text-chat-gemma4-nano",
                "text-chat-gemma4-max",
                "text-chat-q36-nano",
                "text-chat-lfm25-a1b-8bit",
                "text-chat-mebot",
            ],
            resourceName: "api-serve.md"
        ),
        GuideTopic(
            topic: "status",
            title: "Status",
            commandPaths: [["status"]],
            models: [],
            resourceName: "status.md"
        ),
        GuideTopic(
            topic: "model-list",
            title: "Model List",
            commandPaths: [["model", "list"]],
            models: [],
            resourceName: "model-list.md"
        ),
        GuideTopic(
            topic: "model-runtime",
            title: "Model Runtime",
            commandPaths: [["model", "runtime"], ["model", "runtime", "get"], ["model", "runtime", "set"]],
            models: [
                "text-code-qwen3",
                "text-agent-qwen35-9b",
                "text-chat-gemma4",
                "text-chat-gemma4-12b",
                "vision-chat-gemma4-12b",
                "text-chat-q36-nano",
                "text-chat-lfm25-a1b-8bit",
                "text-agent-deepseek-v4-flash",
                "text-chat-mebot",
            ],
            resourceName: "model-runtime.md"
        ),
        GuideTopic(
            topic: "model-benchmark",
            title: "Model Benchmark",
            commandPaths: [
                ["model", "benchmark"],
                ["model", "benchmark", "gemma4-kv"],
                ["model", "benchmark", "gemma4-mtp"],
                ["model", "benchmark", "q36-mtp"],
            ],
            models: [
                "text-chat-gemma4-turbo",
                "text-chat-gemma4-12b-4bit",
                "text-chat-q36-nano",
            ],
            resourceName: "model-benchmark.md"
        ),
        GuideTopic(
            topic: "model-capabilities",
            title: "Model Capabilities",
            commandPaths: [["model", "capabilities"]],
            models: [],
            resourceName: "model-capabilities.md"
        ),
        GuideTopic(
            topic: "model-info",
            title: "Model Info",
            commandPaths: [["model", "info"]],
            models: [],
            resourceName: "model-info.md"
        ),
        GuideTopic(
            topic: "model-pull",
            title: "Model Pull",
            commandPaths: [["model", "pull"]],
            models: [],
            resourceName: "model-pull.md"
        ),
        GuideTopic(
            topic: "model-remove",
            title: "Model Remove",
            commandPaths: [["model", "remove"]],
            models: [],
            resourceName: "model-remove.md"
        ),
        GuideTopic(
            topic: "model-repair-manifests",
            title: "Model Repair Manifests",
            commandPaths: [["model", "repair-manifests"]],
            models: [],
            resourceName: "model-repair-manifests.md"
        ),
        GuideTopic(
            topic: "setup",
            title: "Setup",
            commandPaths: [["setup"]],
            models: [
                "text-code-qwen3",
                "text-agent-qwen35-9b",
            ],
            resourceName: "setup.md"
        ),
        GuideTopic(
            topic: "agent-onboard",
            title: "Agent Onboard",
            commandPaths: [["agent", "onboard"]],
            models: [
                "text-code-qwen3",
                "text-agent-qwen35-9b",
            ],
            resourceName: "agent-onboard.md"
        ),
        GuideTopic(
            topic: "agent-install-pi",
            title: "Agent Install Pi",
            commandPaths: [["agent", "install-pi"]],
            models: [],
            resourceName: "agent-install-pi.md"
        ),
        GuideTopic(
            topic: "agent-start",
            title: "Agent Start",
            commandPaths: [["agent", "start"]],
            models: [
                "text-code-qwen3",
                "text-agent-qwen35-9b",
            ],
            resourceName: "agent-start.md"
        ),
    ]

    static func topic(matching commandPath: [String]) -> GuideTopic? {
        let normalized = normalizedCommandPath(commandPath)
        return all.first { topic in
            topic.commandPaths.contains(normalized)
        }
    }

    static func normalizedCommandPath(_ commandPath: [String]) -> [String] {
        commandPath.map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        }.filter { !$0.isEmpty }
    }

    static func validate(model: String?, for topic: GuideTopic) throws {
        guard let model else { return }
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return }
        guard topic.models.contains(normalized) else {
            let supported = topic.models.isEmpty ? "none" : topic.models.joined(separator: ", ")
            throw ValidationError(
                "Model \(model) is not covered by guide \(topic.topic). Supported model IDs: \(supported)."
            )
        }
    }

    static func content(for topic: GuideTopic) throws -> String {
        let resource = topic.resourceName as NSString
        let baseName = resource.deletingPathExtension
        let fileExtension = resource.pathExtension
        let nestedURL = Bundle.module.url(
            forResource: baseName,
            withExtension: fileExtension,
            subdirectory: "Guides"
        )
        let flatURL = Bundle.module.url(forResource: baseName, withExtension: fileExtension)
        guard let url = nestedURL ?? flatURL else {
            throw ValidationError("Guide resource missing: \(topic.resourceName)")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// First non-empty line of the guide's `## Purpose` section, used as a one-line description.
    static func purposeSummary(for topic: GuideTopic) -> String {
        guard let content = try? content(for: topic) else { return "" }
        var inPurpose = false
        for rawLine in content.components(separatedBy: "\n") {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.lowercased().hasPrefix("## purpose") {
                inPurpose = true
                continue
            }
            guard inPurpose else { continue }
            if line.hasPrefix("#") { break } // reached the next section
            if line.isEmpty { continue }
            return line
        }
        return ""
    }
}

private struct GuidePayload: Encodable {
    let topic: String
    let title: String
    let commands: [String]
    let models: [String]
    let content: String
}

private struct GuideListItem: Encodable {
    let topic: String
    let title: String
    let commands: [String]
    let models: [String]
}

private struct GuideListRow {
    let topic: String
    let command: String
    let models: String
    let title: String

    func render(widths: GuideListWidths) -> String {
        [
            topic.padding(toLength: widths.topic, withPad: " ", startingAt: 0),
            command.padding(toLength: widths.command, withPad: " ", startingAt: 0),
            models.padding(toLength: widths.models, withPad: " ", startingAt: 0),
            title,
        ].joined(separator: "  ")
    }
}

private struct GuideListWidths {
    let topic: Int
    let command: Int
    let models: Int

    init(rows: [GuideListRow]) {
        topic = max("Topic".count, rows.map(\.topic.count).max() ?? 0)
        command = max("Command".count, rows.map(\.command.count).max() ?? 0)
        models = max("Models".count, rows.map(\.models.count).max() ?? 0)
    }

    var total: Int {
        topic + command + models + "Title".count + 6
    }
}
