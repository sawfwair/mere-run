import ArgumentParser
import Foundation
import MereRunCore

struct ModelOptimize: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "optimize",
        abstract: "Build inference-only caches for a supported installed model."
    )

    @Argument(help: "Canonical model id or local model root path.")
    var target: String

    @Flag(name: [.long], help: "Replace an existing compatible optimization cache.")
    var force: Bool = false

    @Flag(name: [.long], help: "Emit the result as JSON.")
    var json: Bool = false

    func run() throws {
        let rootURL = try resolveRootURL()
        let resources = MiniMaxH3Resources(rootURL: rootURL)
        let outputURL = try MiniMaxH3ModelOptimizer.optimize(
            resources: resources,
            replacing: force,
            progressHandler: { completed, total in
                CLIStderr.write("MiniMax-H3 AdaLN cache: \(completed)/\(total)\n")
            }
        )
        let bytes = (try? outputURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        let output = ModelOptimizeOutput(
            modelRoot: rootURL.path,
            artifact: outputURL.path,
            bytes: bytes
        )
        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(output)
            guard let text = String(data: data, encoding: .utf8) else {
                throw ValidationError("Could not encode optimization result as UTF-8.")
            }
            print(text)
        } else {
            print("Model optimization complete")
            print("  model: \(output.modelRoot)")
            print("  artifact: \(output.artifact)")
            print("  size: \(ByteCountFormatter.string(fromByteCount: Int64(output.bytes), countStyle: .file))")
        }
    }

    private func resolveRootURL() throws -> URL {
        let pathURL = URL(fileURLWithPath: target).standardizedFileURL
        if FileManager.default.fileExists(atPath: pathURL.path) {
            return pathURL
        }
        guard let modelID = ModelResolver.ModelID(rawValue: target) else {
            throw ValidationError("Not a path and not a known model id: \(target)")
        }
        guard modelID == .miniMaxH3FL2VAMLX || modelID == .miniMaxH3Ref2VAMLX else {
            throw ValidationError("Model optimization currently supports MiniMax-H3 MLX models.")
        }
        do {
            return try ModelResolver().resolve(modelID).rootURL
        } catch {
            throw ValidationError("Model \(target) is not installed in the local model store.")
        }
    }
}

private struct ModelOptimizeOutput: Codable {
    let modelRoot: String
    let artifact: String
    let bytes: Int
}
