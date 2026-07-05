import ArgumentParser
import Foundation

struct ImageDataset: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "dataset",
        abstract: "Inspect image training datasets.",
        subcommands: [
            ImageDatasetDiscover.self,
        ]
    )
}

struct ImageDatasetDiscover: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "discover",
        abstract: "Find image-caption dataset candidates under a root directory."
    )

    @Option(name: [.customLong("root")], help: "Root directory to scan for image-caption dataset leaves.")
    var root: String

    @Option(name: [.customLong("max-depth")], help: "Maximum child-directory depth to scan from --root.")
    var maxDepth: Int = 4

    @Option(name: [.customLong("min-usable-pairs")], help: "Minimum usable image-caption pairs required for a candidate to be trainable.")
    var minUsablePairs: Int = 1

    @Option(name: [.customLong("training-output-root")], help: "Optional output directory for per-candidate train-lora preflight commands.")
    var trainingOutputRoot: String?

    @Option(name: [.customLong("training-model")], help: "Optional model id/path for per-candidate train-lora preflight commands.")
    var trainingModel: String?

    @Option(name: [.customLong("training-recipe")], help: "Optional recipe for per-candidate train-lora preflight commands.")
    var trainingRecipe: String?

    @Flag(name: [.customLong("exclude-preview-images")], help: "Ignore preview*.png/jpg/webp images while inspecting candidates.")
    var excludePreviewImages: Bool = false

    @Flag(name: [.customLong("json")], help: "Emit a structured dataset discovery JSON report.")
    var json: Bool = false

    func run() throws {
        let envelope = try makeEnvelope()
        if json {
            print(try StructuredRunOutput.encode(envelope))
        } else {
            printHumanSummary(envelope)
        }
        if envelope.status == .blocked {
            throw ExitCode.failure
        }
    }

    func makeEnvelope(
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) throws -> LoRATrainingDatasetDiscoveryEnvelope {
        guard maxDepth >= 0 else {
            throw ValidationError("--max-depth must be >= 0")
        }
        guard minUsablePairs >= 1 else {
            throw ValidationError("--min-usable-pairs must be >= 1")
        }
        return LoRATrainingDatasetDiscoveryAnalyzer(
            root: root,
            maxDepth: maxDepth,
            excludePreviewImages: excludePreviewImages,
            minUsablePairs: minUsablePairs,
            trainingOutputRoot: trainingOutputRoot,
            trainingModel: trainingModel,
            trainingRecipe: trainingRecipe,
            fileManager: fileManager,
            now: now
        ).envelope()
    }

    private func printHumanSummary(_ envelope: LoRATrainingDatasetDiscoveryEnvelope) {
        print(envelope.summary)
        for candidate in envelope.result.candidates {
            let marker: String
            switch candidate.status {
            case .ok:
                marker = "ok"
            case .warning:
                marker = "warning"
            case .blocked:
                marker = "blocked"
            default:
                marker = candidate.status.rawValue
            }
            print(
                "[\(marker)] \(candidate.relativePath) " +
                "usable=\(candidate.usablePairCount) " +
                "images=\(candidate.imageCount) captions=\(candidate.captionCount)"
            )
        }
        for diagnostic in envelope.diagnostics {
            print("[\(diagnostic.severity.rawValue)] \(diagnostic.title): \(diagnostic.message)")
        }
    }
}
