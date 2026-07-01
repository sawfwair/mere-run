import ArgumentParser
import Foundation

struct ImageVisualizeRun: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "visualize-run",
        abstract: "Open a local LoRA training run viewer.",
        discussion: """
        Serves a loopback-only web UI for an existing LoRA run directory. The viewer reads run.json,
        loss CSV files, training events, samples, checkpoints, and adapter artifacts from disk.
        """
    )

    @Argument(help: "LoRA run directory containing run.json, loss CSV, events, samples, or checkpoints.")
    var runDirectory: String

    @Option(name: [.long], help: "Loopback port for the viewer.")
    var port: Int = 8787

    func run() async throws {
        guard (1...65535).contains(port) else {
            throw ValidationError("--port must be between 1 and 65535")
        }

        let url = URL(fileURLWithPath: runDirectory).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw ValidationError("Run directory not found: \(url.path)")
        }

        let viewer = LoRATrainingRunViewer(runDirectoryURL: url)
        try await viewer.run(host: "127.0.0.1", port: port)
    }
}
