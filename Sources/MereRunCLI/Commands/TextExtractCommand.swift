import ArgumentParser
import Foundation
import MereRunCore

struct TextExtract: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "extract", abstract: "Extract entities, relations, and structures with native GLiNER2.5 Decide.",
        discussion: "Read a JSON request with text and entities, relations, or structures. Pull the managed model first."
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

    @Flag(help: "Split long text into overlapping chunks and merge the results.")
    var long = false

    @Flag(help: "Read a JSON array of requests and return an array of results.")
    var batch = false

    func run() async throws {
        let data: Data
        if let input, input != "-" {
            data = try Data(contentsOf: URL(fileURLWithPath: input))
        } else {
            guard !CLIStdin.isInteractive() else { throw ValidationError("Pass --input request.json or pipe JSON to stdin.") }
            data = FileHandle.standardInput.readDataToEndOfFile()
        }
        guard data.count <= 2 * 1_024 * 1_024 else { throw ValidationError("Extraction request exceeds 2 MiB.") }
        let requests = try batch
            ? JSONDecoder().decode([GLiNERExtractionRequest].self, from: data)
            : [JSONDecoder().decode(GLiNERExtractionRequest.self, from: data)]
        guard !requests.isEmpty else { throw ValidationError("Provide at least one extraction request.") }
        try requests.forEach { try $0.validate() }
        if let managed = ManagedModelCatalog.spec(for: model), managed.validationKind != .gliner25Decide {
            throw ValidationError("text extract requires a GLiNER2.5 Decide model.")
        }
        let resolved = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: model, defaultModelID: GLiNERCatalog.modelID, progress: nil)
        let operation = try GLiNERClassificationOperation(root: resolved.url, modelID: model)
        let encoder = JSONEncoder()
        encoder.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]
        let result: Data
        if preflight, long {
            let plans = try requests.map { try operation.prepareLong($0) }
            result = try batch ? encoder.encode(plans) : encoder.encode(plans[0])
        } else if preflight {
            let plans = try requests.map { try operation.prepare($0) }
            result = try batch ? encoder.encode(plans) : encoder.encode(plans[0])
        } else {
            try MLXBundleSupport.ensureAvailable(quiet: true)
            let responses = try long ? operation.predictBatchLong(requests) : operation.predictBatch(requests)
            result = try batch ? encoder.encode(responses) : encoder.encode(responses[0])
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
