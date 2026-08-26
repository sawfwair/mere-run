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
        let artifacts: [URL]
        let optimization: String
        if isLTX25ModelRoot(rootURL) {
            let resources = LTX25Resources(rootURL: rootURL)
            var kinds: [LTX25NativeModelPackKind] = [.distilled, .connector]
            if isLTX25FullModelRoot(rootURL) {
                kinds.append(.dev)
            }
            var ltxArtifacts = try kinds.map { kind in
                let result = try LTX25NativeModelPack.optimize(
                    resources: resources,
                    kind: kind,
                    replacing: force,
                    progressHandler: { completed, total in
                        if completed == total || completed.isMultiple(of: 100) {
                            CLIStderr.write(
                                "LTX 2.5 native \(kind.rawValue) pack: \(completed)/\(total) tensors\n"
                            )
                        }
                    }
                )
                return result.outputURL
            }
            _ = try LTX25TextEncoderQuantizedPack.optimize(
                resources: resources,
                replacing: force,
                progressHandler: { completed, total in
                    CLIStderr.write(
                        "LTX 2.5 Q4 text pack: \(completed)/\(total) shards\n"
                    )
                }
            )
            ltxArtifacts.append(
                contentsOf: LTX25TextEncoderQuantizedPack.artifactURLs(resources: resources)
            )
            artifacts = ltxArtifacts
            optimization = "ltx25-native-model-pack+text-q4-v1"
        } else {
            let resources = MiniMaxH3Resources(rootURL: rootURL)
            _ = try MiniMaxH3ModelOptimizer.optimize(
                resources: resources,
                replacing: force,
                progressHandler: { completed, total in
                    CLIStderr.write("MiniMax-H3 AdaLN cache: \(completed)/\(total)\n")
                }
            )
            artifacts = MiniMaxH3ModelOptimizer.artifactURLs(resources: resources)
            optimization = "minimax-h3-adaln-cache-pack-v1"
        }
        let bytes = artifacts.reduce(0) { total, url in
            total + ((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        let output = ModelOptimizeOutput(
            modelRoot: rootURL.path,
            optimization: optimization,
            artifact: artifacts[0].path,
            artifacts: artifacts.map(\.path),
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
            print("  optimization: \(output.optimization)")
            for artifact in output.artifacts {
                print("  artifact: \(artifact)")
            }
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
        guard modelID == .miniMaxH3FL2VAMLX
                || modelID == .miniMaxH3FL2VABF16MLX
                || modelID == .miniMaxH3FL2VAQ8MLX
                || modelID == .miniMaxH3Ref2VAMLX
                || modelID == .ltxVideo25DistilledBF16
                || modelID == .ltxVideo25FullBF16 else {
            throw ValidationError("Model optimization supports MiniMax-H3 MLX and LTX 2.5 models.")
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
    let optimization: String
    let artifact: String
    let artifacts: [String]
    let bytes: Int
}
