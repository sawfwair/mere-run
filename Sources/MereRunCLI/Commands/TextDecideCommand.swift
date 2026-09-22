import ArgumentParser
import Foundation
import MereRunCore

struct TextDecide: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "decide", abstract: "Evaluate typed questions with native Laya decision models.",
        discussion: "Read a JSON request containing state text and an ordered questions array. Output is always JSON. "
            + "Use --preflight to inspect token budgets before inference. Pull a managed model with model pull first."
    )

    @Option(name: [.customShort("i"), .long], help: "JSON request file; use - or omit to read stdin.")
    var input: String?

    @Option(name: [.customShort("m"), .long], help: "Managed Laya id or local checkpoint directory.")
    var model: String = LayaCatalog.modelID

    @Option(name: [.customShort("o"), .long], help: "Also save the JSON result to this file.")
    var output: String?

    @Flag(help: "Pretty-print JSON.")
    var pretty = false

    @Flag(help: "Validate inputs and report token budgets without loading model weights.")
    var preflight = false

    func run() async throws {
        let data: Data
        if let input, input != "-" {
            data = try Data(contentsOf: URL(fileURLWithPath: input))
        } else {
            guard !CLIStdin.isInteractive() else { throw ValidationError("Pass --input request.json or pipe a JSON request to stdin.") }
            data = FileHandle.standardInput.readDataToEndOfFile()
        }
        let request = try Self.decodeRequest(data)
        let managed = ManagedModelCatalog.spec(for: model)
        if let managed, managed.validationKind != .laya {
            throw ValidationError("text decide requires a Laya model.")
        }
        let resolved = try await ManagedModelResolver.resolveForRuntime(requestedModel: model, defaultModelID: LayaCatalog.modelID, progress: nil)
        let root = managed.map { LayaCatalog.checkpointRoot(resolved.url, modelID: $0.id) } ?? resolved.url
        let operation = try LayaDecisionOperation(root: root, modelID: managed?.id ?? root.lastPathComponent)
        let encoder = JSONEncoder()
        encoder.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]
        let result: Data
        if preflight {
            result = try encoder.encode(operation.prepare(request).plan)
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

    static func decodeRequest(_ data: Data) throws -> LayaDecisionRequest {
        guard data.count <= 2 * 1_024 * 1_024 else { throw ValidationError("The decision request must not exceed 2 MiB.") }
        let request = try JSONDecoder().decode(LayaDecisionRequest.self, from: data)
        try request.validate()
        return request
    }
}
