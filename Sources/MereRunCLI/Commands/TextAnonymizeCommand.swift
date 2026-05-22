import ArgumentParser
import Foundation
import MereRunCore

struct TextAnonymize: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "anonymize",
        abstract: "Detect and redact PII using OpenAI Privacy Filter.",
        discussion: """
        Uses the native MLX OpenAI Privacy Filter token classifier.

        Example:
          mere.run text anonymize "My name is Alice Smith and my email is alice@example.com"
          mere.run text anonymize --json --pretty "Phone: 555-1234"
          cat notes.txt | mere.run text anonymize --output redacted.txt
        """
    )

    @Argument(help: "One or more texts to anonymize. Reads stdin when omitted.")
    var texts: [String] = []

    @Option(name: [.customShort("m"), .long], help: "Model path or model id (default: text-anonymize-privacy-filter).")
    var model: String?

    @Option(name: [.long], help: "Maximum token length per input (clamped to model max).")
    var maxTokens: Int?

    @Option(name: [.long], help: "Replacement template. Supports {label} and {index}.")
    var replacement: String = "[{label}]"

    @Flag(name: [.long], help: "Emit structured JSON with spans.")
    var json: Bool = false

    @Flag(name: [.long], help: "Pretty-print JSON output.")
    var pretty: Bool = false

    @Option(name: [.customShort("o"), .long], help: "Optional output path.")
    var output: String?

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: false)

        let resolvedTexts = try readTexts()
        let resolvedModelRoot = try await resolveModelRoot()
        let resources = OpenAIPrivacyFilterResources(rootURL: resolvedModelRoot)
        let anonymizer = try OpenAIPrivacyFilterAnonymizer(resources: resources)
        let results = try anonymizer.anonymize(
            texts: resolvedTexts,
            maxTokens: maxTokens,
            replacementTemplate: replacement
        )

        let data: Data
        if json {
            let encoder = JSONEncoder()
            if pretty {
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            }
            data = try encoder.encode(AnonymizeResponse(model: OpenAIPrivacyFilterCatalog.modelId, data: results))
        } else {
            data = Data(results.map(\.anonymizedText).joined(separator: "\n").utf8)
        }

        if let outputPath = output {
            let outputURL = URL(fileURLWithPath: outputPath).standardizedFileURL
            try FileManager.default.createDirectory(
                at: outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try data.write(to: outputURL, options: [.atomic])
        }

        if let text = String(data: data, encoding: .utf8) {
            print(text)
        } else {
            throw ValidationError("Failed to encode anonymizer output as UTF-8.")
        }
    }

    private func readTexts() throws -> [String] {
        if !texts.isEmpty {
            return texts
        }

        guard !CLIStdin.isInteractive() else {
            throw ValidationError("Provide text arguments or pipe UTF-8 text on stdin.")
        }

        let input = FileHandle.standardInput.readDataToEndOfFile()
        guard !input.isEmpty, let text = String(data: input, encoding: .utf8), !text.isEmpty else {
            throw ValidationError("Provide text arguments or pipe UTF-8 text on stdin.")
        }
        return [text]
    }

    private func resolveModelRoot() async throws -> URL {
        do {
            let resolved = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: model,
                defaultModelID: OpenAIPrivacyFilterCatalog.modelId,
                progress: nil
            )
            return resolved.url
        } catch let error as ManagedModelResolver.ResolverError {
            throw ValidationError(error.localizedDescription)
        }
    }
}

private struct AnonymizeResponse: Encodable, Sendable {
    let object = "list"
    let model: String
    let data: [OpenAIPrivacyFilterAnonymizationResult]
}
