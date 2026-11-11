import ArgumentParser
import Foundation
import MereRunCore
import MereRunExecution

struct ModelBenchmarkH3Image: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "h3-image",
        abstract: "Measure cold and warm native H3-Image generation with retained models.")

    @Option(name: [.long], help: "Installed original BF16 or native Q8 model directory.")
    var modelRoot: String
    @Option(name: [.long], help: "Detailed image prompt, reused for all runs.")
    var prompt: String
    @Option(name: [.long], help: "Square native canvas: 2048 or 4096 pixels.")
    var size: Int = 4_096
    @Option(name: [.long], help: "Total runs, including the first cold run; at least two.")
    var runs: Int = 3
    @Option(name: [.long], help: "Seed reused for every run.")
    var seed: UInt64 = 42
    @Option(name: [.long], help: "New directory for generated PNGs and benchmark.json.")
    var output: String
    @Flag(name: [.customLong("resident-bf16")], help: "Materialize Q8 transformer weights as resident BF16 for dense compute.")
    var residentBF16 = false
    @Option(name: [.customLong("attention-heads")], help: "Maximum attention heads per kernel, from 1 to 56.")
    var attentionHeads: Int = 28
    @Option(name: [.customLong("attention-queries")], help: "Maximum queries per attention kernel, from 1 to 32768.")
    var attentionQueries: Int = 1_024
    @Option(name: [.customLong("vae-tile-batch")], help: "Maximum VAE decode tiles per batch, from 1 to 128.")
    var vaeTileBatch: Int = 32
    @Flag(name: [.long], help: "Emit the benchmark receipt as JSON.")
    var json = false

    func validate() throws {
        guard [2_048, 4_096].contains(size), runs >= 2, (1...56).contains(attentionHeads),
              (1...32_768).contains(attentionQueries), (1...128).contains(vaeTileBatch) else {
            throw ValidationError("Use size 2048 or 4096, at least two runs, heads 1–56, queries 1–32768, and VAE tile batch 1–128.")
        }
        guard !FileManager.default.fileExists(atPath: output) else {
            throw ValidationError("Benchmark output must be a new directory.")
        }
    }

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: false)
        let directory = URL(fileURLWithPath: output).standardizedFileURL
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let generator = MiniMaxH3ImageGenerator(retainModels: true, residentBF16: residentBF16,
            attentionHeads: attentionHeads, attentionQueries: attentionQueries, vaeTileBatch: vaeTileBatch)
        var records: [Record] = []
        for index in 0..<runs {
            CLIStderr.write("H3-Image benchmark \(index + 1)/\(runs)\n")
            let image = directory.appending(path: "run-\(index).png")
            let request = GenerationRequest(prompt: prompt, width: size, height: size,
                steps: 6, guidanceScale: 1, seed: seed, outputURL: image, model: modelRoot)
            let timing = DenoisingTiming()
            let start = ProcessInfo.processInfo.systemUptime
            _ = try await generator.generate(request, progressHandler: { timing.record($0) })
            let seconds = ProcessInfo.processInfo.systemUptime - start
            let artifact = try RunArtifact.read(image)
            records.append(Record(index: index, warm: index > 0, seconds: seconds, denoisingSeconds: timing.seconds,
                                  output: image.path, sha256: artifact.sha256))
        }
        await generator.unload()
        let warmTimes = records.dropFirst().map(\.seconds).sorted()
        let middle = warmTimes.count / 2
        let median = warmTimes.count.isMultiple(of: 2) ? (warmTimes[middle - 1] + warmTimes[middle]) / 2 : warmTimes[middle]
        let receipt = Receipt(modelRoot: modelRoot, size: size, seed: seed, prompt: prompt,
            residentBF16: residentBF16, attentionHeads: attentionHeads,
            attentionQueries: attentionQueries, vaeTileBatch: vaeTileBatch,
            records: records, warmMedianSeconds: median, identicalOutputs: Set(records.map(\.sha256)).count == 1)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(receipt)
        try data.write(to: directory.appending(path: "benchmark.json"))
        if json { print(String(decoding: data, as: UTF8.self)) }
        else { print("H3-Image: cold \(records[0].seconds)s; warm median \(median)s; receipt \(directory.appending(path: "benchmark.json").path)") }
    }

    private struct Record: Codable {
        let index: Int
        let warm: Bool
        let seconds: Double
        let denoisingSeconds: Double?
        let output: String
        let sha256: String
    }
    private struct Receipt: Codable {
        let modelRoot: String
        let size: Int
        let seed: UInt64
        let prompt: String
        let residentBF16: Bool
        let attentionHeads: Int
        let attentionQueries: Int
        let vaeTileBatch: Int
        let records: [Record]
        let warmMedianSeconds: Double
        let identicalOutputs: Bool
    }

    private final class DenoisingTiming: @unchecked Sendable {
        private let lock = NSLock()
        private var start: Double?
        private var end: Double?

        func record(_ progress: GenerationProgress) {
            guard progress.stage == .denoising else { return }
            lock.withLock {
                if progress.stepIndex == 0 { start = ProcessInfo.processInfo.systemUptime }
                if progress.stepIndex == progress.totalSteps { end = ProcessInfo.processInfo.systemUptime }
            }
        }

        var seconds: Double? {
            lock.withLock {
                guard let start, let end else { return nil }
                return end - start
            }
        }
    }
}
