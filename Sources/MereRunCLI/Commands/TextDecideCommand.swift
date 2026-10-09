import ArgumentParser
import Foundation
import MereRunCore

struct TextDecide: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "decide", abstract: "Evaluate typed questions with native Laya, Clef, or D1 decision models.",
        discussion: "Read a JSON request containing state and questions (an array for Laya, an object for Clef and D1). Output is always JSON. "
            + "Use --preflight to inspect token budgets before inference. Pull a managed model with model pull first."
    )

    @Option(name: [.customShort("i"), .long], help: "JSON request file; use - or omit to read stdin.")
    var input: String?

    @Option(name: [.customShort("m"), .long], help: "Managed Laya, Clef, or D1 id, or local checkpoint directory.")
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
        let managed = ManagedModelCatalog.spec(for: model)
        if let managed, managed.validationKind != .laya && managed.validationKind != .clef && managed.validationKind != .d1 {
            throw ValidationError("text decide requires a Laya, Clef, or D1 model.")
        }
        let isD1 = managed?.validationKind == .d1 || D1Catalog.isLocalCheckpoint(URL(fileURLWithPath: model))
        let d1Request = isD1 ? try D1DecisionRequest.decode(data) : nil
        let localClef = FileManager.default.fileExists(atPath: URL(fileURLWithPath: model).appending(path: "joint_head_config.json").path)
        let isClef = managed?.validationKind == .clef || localClef
        // Decode before model resolution so invalid inputs never trigger checkpoint work.
        let clefRequest = isClef ? try ClefDecisionRequest.decode(data) : nil
        let layaRequest = isClef || isD1 ? nil : try Self.decodeRequest(data)
        let resolved = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: model, defaultModelID: isD1 ? D1Catalog.modelID : isClef ? ClefCatalog.modelID : LayaCatalog.modelID, progress: nil)
        let encoder = JSONEncoder()
        encoder.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]
        // The file keeps the result alone; stdout adds the gate's warnings.
        func encoded(_ value: some Encodable) throws -> (file: Data, stdout: Data) {
            (try encoder.encode(value), try encoder.encode(GateWarned(value)))
        }
        let result: (file: Data, stdout: Data)
        if let request = d1Request {
            let operation = try D1DecisionOperation(root: resolved.url, modelID: managed?.id ?? resolved.url.lastPathComponent)
            defer { operation.unload() }
            if preflight { result = try encoded(operation.prepare(request)) }
            else {
                try MLXBundleSupport.ensureAvailable(quiet: true)
                result = try encoded(operation.predict(request))
            }
        } else if let request = clefRequest {
            let operation = try ClefDecisionOperation(root: resolved.url, modelID: managed?.id ?? resolved.url.lastPathComponent)
            defer { operation.unload() }
            if preflight { result = try encoded(operation.prepare(request)) }
            else {
                try MLXBundleSupport.ensureAvailable(quiet: true)
                result = try encoded(operation.predict(request))
            }
        } else {
            guard let request = layaRequest else { throw ValidationError("Missing decision request.") }
            let root = managed.map { LayaCatalog.checkpointRoot(resolved.url, modelID: $0.id) } ?? resolved.url
            let operation = try LayaDecisionOperation(root: root, modelID: managed?.id ?? root.lastPathComponent)
            if preflight { result = try encoded(operation.prepare(request).plan) }
            else {
                try MLXBundleSupport.ensureAvailable(quiet: true)
                result = try encoded(operation.predict(request))
            }
        }
        if let output {
            let destination = URL(fileURLWithPath: output)
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try result.file.write(to: destination, options: .atomic)
        }
        FileHandle.standardOutput.write(result.stdout)
        FileHandle.standardOutput.write(Data([10]))
    }

    static func decodeRequest(_ data: Data) throws -> LayaDecisionRequest {
        guard data.count <= 2 * 1_024 * 1_024 else { throw ValidationError("The decision request must not exceed 2 MiB.") }
        let request = try JSONDecoder().decode(LayaDecisionRequest.self, from: data)
        try request.validate()
        return request
    }
}
