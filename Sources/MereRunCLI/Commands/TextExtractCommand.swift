import ArgumentParser
import Foundation
import MereRunCore

struct TextExtract: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "extract", abstract: "Run native GLiNER2.5 Decide or Clairic extraction.",
        discussion: "Use GLiNER for entities and relations, or select Clairic for typed fields in text, images, and PDFs. Pull the model first."
    )

    @Option(name: [.customShort("i"), .long], help: "JSON request file; use - or omit to read stdin.")
    var input: String?

    @Option(name: [.customShort("m"), .long], help: "Managed GLiNER or Clairic ID, or a local checkpoint directory.")
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
        let isClairic = Self.isClairicModel(model)
        if isClairic && long { throw ValidationError("--long is available for GLiNER extraction.") }
        let data: Data
        if let input, input != "-" {
            data = try Data(contentsOf: URL(fileURLWithPath: input))
        } else {
            guard !CLIStdin.isInteractive() else { throw ValidationError("Pass --input request.json or pipe JSON to stdin.") }
            data = FileHandle.standardInput.readDataToEndOfFile()
        }
        let maximumMiB = isClairic ? 12 : 2
        guard data.count <= maximumMiB * 1_024 * 1_024 else {
            throw ValidationError("Extraction request exceeds \(maximumMiB) MiB.")
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]
        let result = try await (isClairic ? runClairic(data, encoder: encoder) : runGLiNER(data, encoder: encoder))
        if let output {
            let destination = URL(fileURLWithPath: output)
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try result.write(to: destination, options: .atomic)
        }
        FileHandle.standardOutput.write(result)
        FileHandle.standardOutput.write(Data([10]))
    }

    static func isClairicModel(_ model: String) -> Bool {
        if let spec = ManagedModelCatalog.spec(for: model) { return spec.validationKind == .clairic }
        return ClairicCatalog.identifiesCheckpoint(at: URL(fileURLWithPath: model))
    }

    private func runClairic(_ data: Data, encoder: JSONEncoder) async throws -> Data {
#if os(iOS)
        throw ValidationError("Clairic extraction requires macOS or Linux.")
#else
        let requests = try batch
            ? JSONDecoder().decode([ClairicRequest].self, from: data)
            : [JSONDecoder().decode(ClairicRequest.self, from: data)]
        guard !requests.isEmpty else { throw ValidationError("Provide at least one extraction request.") }
        let resolved = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: model, defaultModelID: ClairicCatalog.flashModelID, progress: nil)
        let operation = try ClairicOperation(root: resolved.url, modelID: resolved.source == .explicitPath ? nil : resolved.spec.id)
        if preflight {
            let plans = try requests.map { try operation.prepare($0) }
            return try batch ? encoder.encode(plans) : encoder.encode(plans[0])
        }
        try MLXBundleSupport.ensureAvailable(quiet: true)
        defer { operation.unload() }
        let responses = try requests.map { try operation.predict($0) }
        return try batch ? encoder.encode(responses) : encoder.encode(responses[0])
#endif
    }

    private func runGLiNER(_ data: Data, encoder: JSONEncoder) async throws -> Data {
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
        return result
    }
}
