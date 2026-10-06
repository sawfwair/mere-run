import Foundation
import Crypto
import MLX
import MereRunKolibriModel

public enum KolibriLogprobBenchmark {
    public struct Sequence: Codable, Sendable {
        public let id: String
        public let language: String
        public let task: String
        public let split: String
        public let tokens: [Int]
        /// Index of the first target token scored, not its predicting input row.
        public let scoreStart: Int
    }
    struct CaseResult: Codable {
        let sequence: Sequence
        let logitsFile: String
        let logitsSHA256: String
        let meanNegativeLogLikelihood: Double
        let tokenCount: Int
        let elapsedSeconds: Double
    }
    struct Receipt: Codable {
        let schemaVersion: Int
        let configSHA256: String
        let indexSHA256: String
        let suiteSHA256: String
        let conversionSHA256: String?
        let calibrationSHA256: String?
        let calibrationCaseIDs: [String]
        let checkpointPath: String
        let chunkSize: Int
        let device: String
        let peakMemoryBytes: Int
        let cases: [CaseResult]
    }

    public static func run(modelRoot: URL, suite: URL, output: URL, chunkSize: Int = 32, calibrationOutput: URL? = nil, progressHandler: ((String) -> Void)? = nil) throws {
        if let calibrationOutput, FileManager.default.fileExists(atPath: calibrationOutput.path) {
            throw ChatRequestIssue("calibration-output", "use a fresh safetensors file")
        }
        guard chunkSize > 0 else { throw ChatRequestIssue("chunk-size", "must be positive") }
        let suiteData = try Data(contentsOf: suite)
        let sequences = try JSONDecoder().decode([Sequence].self, from: suiteData)
        guard !sequences.isEmpty, Set(sequences.map(\.id)).count == sequences.count else {
            throw ChatRequestIssue("suite", "must contain unique nonempty cases")
        }
        Memory.peakMemory = 0
        progressHandler?("Verifying conversion hashes")
        try KolibriWeightVerification.verifyIfPresent(root: modelRoot)
        progressHandler?("Loading native Kolibri checkpoint")
        let model = try KolibriLoader.load(root: modelRoot)
        for sequence in sequences {
            guard sequence.id == URL(fileURLWithPath: sequence.id).lastPathComponent,
                  sequence.id != ".", sequence.id != "..", !sequence.id.isEmpty,
                  sequence.tokens.count <= model.config.maxPositionEmbeddings,
                  sequence.tokens.allSatisfy({ (0..<model.config.vocabSize).contains($0) }),
                  sequence.scoreStart > 0, sequence.scoreStart < sequence.tokens.count else {
                throw ChatRequestIssue("suite", "invalid sequence \(sequence.id)")
            }
        }
        guard !FileManager.default.fileExists(atPath: output.path) else {
            throw ChatRequestIssue("output", "use a fresh output directory")
        }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
        var inputSums: [String: MLXArray] = [:]
        var inputCounts: [String: Int] = [:]
        var results: [CaseResult] = []
        for sequence in sequences {
            let caseStart = Date()
            progressHandler?("Scoring \(sequence.id)")
            if calibrationOutput != nil, sequence.split == "calibration" {
                model.observeExpertInputs { path, input in
                    let axes = Array(0..<(input.ndim - 1))
                    let squareSum = input.asType(.float32).square().sum(axes: axes)
                    let sum = inputSums[path].map { $0 + squareSum } ?? squareSum
                    eval(sum)
                    inputSums[path] = sum
                    inputCounts[path, default: 0] += input.size / input.dim(-1)
                }
            } else { model.observeExpertInputs(nil) }
            let caches = model.makeCache()
            var scored: [MLXArray] = []
            let inputCount = sequence.tokens.count - 1
            for start in stride(from: 0, to: inputCount, by: chunkSize) {
                try Task.checkCancellation()
                let end = min(start + chunkSize, inputCount)
                let input = MLXArray(sequence.tokens[start..<end].map(Int32.init)).reshaped(1, end - start)
                let logits = model(input, cache: caches)
                eval(logits)
                let scoredStart = max(start, sequence.scoreStart - 1)
                if scoredStart < end { scored.append(contiguous(logits[0, (scoredStart - start)..., 0...])) }
            }
            let logits = concatenated(scored, axis: 0).asType(.float32)
            let targets = MLXArray(sequence.tokens[sequence.scoreStart...].map(Int32.init)).expandedDimensions(axis: -1)
            let logprobs = logits - logSumExp(logits, axis: -1, keepDims: true)
            let nll = -takeAlong(logprobs, targets, axis: -1).mean().item(Float.self)
            let filename = sequence.id + ".safetensors"
            try MLX.save(arrays: ["logits": logits], url: output.appendingPathComponent(filename))
            results.append(CaseResult(sequence: sequence, logitsFile: filename,
                                      logitsSHA256: digest(try Data(contentsOf: output.appendingPathComponent(filename))),
                                      meanNegativeLogLikelihood: Double(nll), tokenCount: targets.size, elapsedSeconds: Date().timeIntervalSince(caseStart)))
        }
        model.observeExpertInputs(nil)
        if let calibrationOutput {
            guard !inputSums.isEmpty, !FileManager.default.fileExists(atPath: calibrationOutput.path) else {
                throw ChatRequestIssue("calibration-output", "requires calibration cases and a fresh safetensors file")
            }
            let means = inputSums.mapValues { $0 }
            var calibrated: [String: MLXArray] = [:]
            for (path, sum) in means { calibrated[path] = sum / Float(inputCounts[path, default: 1]) }
            try MLX.save(arrays: calibrated, url: calibrationOutput)
        }
        let conversionURL = modelRoot.appendingPathComponent("KOLIBRI_CONVERSION.json")
        let receipt = Receipt(
            schemaVersion: 1,
            configSHA256: digest(try Data(contentsOf: modelRoot.appendingPathComponent("config.json"))),
            indexSHA256: digest(try Data(contentsOf: modelRoot.appendingPathComponent("model.safetensors.index.json"))),
            suiteSHA256: digest(suiteData),
            conversionSHA256: FileManager.default.fileExists(atPath: conversionURL.path) ? digest(try Data(contentsOf: conversionURL)) : nil,
            calibrationSHA256: try calibrationOutput.map { digest(try Data(contentsOf: $0)) },
            calibrationCaseIDs: calibrationOutput == nil ? [] : sequences.filter { $0.split == "calibration" }.map(\.id),
            checkpointPath: modelRoot.path, chunkSize: chunkSize,
            device: String(describing: Device.defaultDevice()), peakMemoryBytes: Memory.snapshot().peakMemory, cases: results
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(receipt).write(to: output.appendingPathComponent("receipt.json"), options: .atomic)
    }
}
