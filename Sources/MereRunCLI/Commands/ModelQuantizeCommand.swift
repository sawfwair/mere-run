import ArgumentParser
import Foundation
import MereRunCore

struct ModelQuantize: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "quantize",
        abstract: "Convert a LingBot-Video MoE checkpoint to native MLX 4-bit routed experts."
    )

    @Argument(help: "Managed model id or local LingBot-Video MoE model root.")
    var source: String = ModelResolver.ModelID.lingBotVideoMoE30BA3B.rawValue

    @Option(name: [.short, .long], help: "Output model root (default: the 4-bit LingBot MoE directory in the model store).")
    var output: String?

    @Option(name: [.long], help: "Quantized expert bit width. Only 4 is currently supported.")
    var bits: Int = 4

    @Option(name: [.customLong("group-size")], help: "Affine quantization group size.")
    var groupSize: Int = 64

    @Flag(name: [.customLong("skip-refiner")], help: "Convert only the base transformer and omit the refiner.")
    var skipRefiner: Bool = false

    @Flag(name: [.long], help: "Replace an existing output directory instead of resuming complete shards.")
    var force: Bool = false

    func validate() throws {
        guard bits == 4 else {
            throw ValidationError("--bits currently supports only 4.")
        }
        guard [32, 64, 128].contains(groupSize) else {
            throw ValidationError("--group-size must be 32, 64, or 128.")
        }
        if skipRefiner, output == nil {
            throw ValidationError("--skip-refiner requires an explicit --output so the canonical 4-bit model remains complete.")
        }
    }

    func run() throws {
        let sourceRoot = try resolveSourceRoot()
        let outputRoot = output.map(URL.init(fileURLWithPath:))
            ?? MereRunModelPaths.modelDir(LingBotVideoMoEQuantizer.defaultOutputModelID)
        let result = try LingBotVideoMoEQuantizer.quantize(
            options: .init(
                sourceRoot: sourceRoot,
                outputRoot: outputRoot,
                bits: bits,
                groupSize: groupSize,
                includeRefiner: !skipRefiner,
                force: force
            )
        ) { progress in
            let action = progress.state == .reused ? "reused" : "quantizing"
            CLIStderr.write(
                "[\(progress.component)] \(action) shard \(progress.completedShards)/\(progress.totalShards): \(progress.shard)\n"
            )
        }
        CLIStderr.write(
            "Converted \(result.quantizedShardCount) shards; reused \(result.reusedShardCount) complete shards.\n"
        )
        print(result.outputRoot.path)
    }

    private func resolveSourceRoot(fileManager: FileManager = .default) throws -> URL {
        let localPath = URL(fileURLWithPath: source).standardizedFileURL
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: localPath.path, isDirectory: &isDirectory),
           isDirectory.boolValue {
            return localPath
        }

        let managedRoot = MereRunModelPaths.modelDir(source)
        if fileManager.fileExists(atPath: managedRoot.path, isDirectory: &isDirectory),
           isDirectory.boolValue {
            return managedRoot
        }
        throw ValidationError(
            "Model source not found: \(source). Pull it first with `mere.run model pull \(source) --allow-unsupported`."
        )
    }
}
