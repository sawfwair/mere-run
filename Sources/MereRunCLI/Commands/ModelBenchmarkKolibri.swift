import ArgumentParser
import Foundation
import MereRunCore

struct ModelBenchmarkKolibri: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "kolibri-logprobs",
        abstract: "Score fixed token sequences with native Kolibri and export raw logits."
    )
    @Option(name: .long, help: "Converted native Kolibri checkpoint directory.") var modelRoot: String
    @Option(name: .long, help: "JSON array of fixed token sequences and scoring boundaries.") var suite: String
    @Option(name: .long, help: "Fresh directory for raw-logit files and the receipt.") var output: String
    @Option(name: .long, help: "Prefill chunk size.") var chunkSize: Int = 32

    @Option(name: .long, help: "Fresh safetensors file for calibration-only expert input second moments.")
    var calibrationOutput: String?

    func run() throws {
        try MLXBundleSupport.ensureAvailable(quiet: true)
        try KolibriLogprobBenchmark.run(modelRoot: URL(fileURLWithPath: modelRoot),
                                       suite: URL(fileURLWithPath: suite), output: URL(fileURLWithPath: output),
                                       chunkSize: chunkSize, calibrationOutput: calibrationOutput.map { URL(fileURLWithPath: $0) },
                                       progressHandler: { CLIStderr.write($0 + "\n") })
        let receipt = URL(fileURLWithPath: output).appendingPathComponent("receipt.json")
        print(String(decoding: try Data(contentsOf: receipt), as: UTF8.self))
    }
}
