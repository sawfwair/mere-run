import ArgumentParser
import Foundation
import MereRunCore

struct TextClassify: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "classify", abstract: "Classify text with native GLiNER2.5 Decide.",
        discussion: "Read a JSON request with text and an ordered tasks array. Each task has a name, labels, "
            + "and optional prompt, descriptions, multi_label, and threshold fields. Pull the managed model first."
    )

    @Option(name: [.customShort("i"), .long], help: "JSON request file; use - or omit to read stdin.")
    var input: String?

    @Option(name: [.customShort("m"), .long], help: "Managed GLiNER ID or local checkpoint directory.")
    var model: String = GLiNERCatalog.modelID

    @Option(name: [.customShort("o"), .long], help: "Also save the JSON result to this file.")
    var output: String?

    @Flag(help: "Pretty-print JSON.")
    var pretty = false

    @Flag(help: "Validate the request and report token usage without loading weights.")
    var preflight = false

    func run() async throws {
        let data: Data
        if let input, input != "-" {
            data = try Data(contentsOf: URL(fileURLWithPath: input))
        } else {
            guard !CLIStdin.isInteractive() else { throw ValidationError("Pass --input request.json or pipe JSON to stdin.") }
            data = FileHandle.standardInput.readDataToEndOfFile()
        }
        guard data.count <= 2 * 1_024 * 1_024 else { throw ValidationError("Classification request exceeds 2 MiB.") }
        let request = try JSONDecoder().decode(GLiNERClassificationRequest.self, from: data)
        try request.validate()
        if let managed = ManagedModelCatalog.spec(for: model), managed.validationKind != .gliner25Decide {
            throw ValidationError("text classify requires a GLiNER2.5 Decide model.")
        }
        let resolved = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: model, defaultModelID: GLiNERCatalog.modelID, progress: nil)
        let operation = try GLiNERClassificationOperation(root: resolved.url, modelID: model)
        let encoder = JSONEncoder()
        encoder.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]
        let result: Data
        if preflight {
            result = try encoder.encode(operation.prepare(request))
        } else {
            try MLXBundleSupport.ensureAvailable(quiet: true)
            result = try encoder.encode(operation.predict(request))
        }
        if let output {
            let destination = URL(fileURLWithPath: output)
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try result.write(to: destination, options: .atomic)
        }
        FileHandle.standardOutput.write(result)
        FileHandle.standardOutput.write(Data([10]))
    }
}
