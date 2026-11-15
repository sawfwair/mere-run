import CryptoKit
import Foundation
import AudioCodecs
import MLX
import XCTest
@testable import AudioSTT

extension SOVLowBitProfileTests {
    private struct QualityEntry: Encodable {
        let sample: SOVLowBitSample
        let measurement: SOVMeasuredTranscription
    }
    private struct QualityReport: Encodable {
        let policy: String
        let manifestSHA256: String
        let recipeSHA256: String?
        let preparedActiveBytes: Int
        let modelLoadSeconds: Double
        let operatingSystem: String
        let warmup: SOVMeasuredTranscription
        let entries: [QualityEntry]
    }

    func testHeldOutRecognition() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_MLX_DEVICE"] == "gpu", let root = env["MERERUN_TEST_SOV_PATH"],
              let draft = env["MERERUN_TEST_SOV_DFLASH_PATH"], let manifest = env["MERERUN_SOV_EVALUATION"],
              let policyName = env["MERERUN_SOV_EXPERT_POLICY"], let policy = SOVDecoderExperts(rawValue: policyName),
              let output = env["MERERUN_SOV_QUALITY_OUTPUT"] else {
            throw XCTSkip("Set GPU, SOV checkpoints, held-out manifest, expert policy, and quality output.")
        }
        let manifestData = try Data(contentsOf: URL(fileURLWithPath: manifest))
        let samples = try JSONDecoder().decode([SOVLowBitSample].self, from: manifestData)
        let recipeData = try env["MERERUN_SOV_EXPERT_RECIPE"].map { try Data(contentsOf: URL(fileURLWithPath: $0)) }
        let recipe = try recipeData.map { try JSONDecoder().decode(SOVExpertRecipe.self, from: $0) }
        let generator = SOVTranscribeGenerator(decoderExperts: policy, experimentalAttention: true,
            generationExperiment: .rollback, expertRecipe: recipe)
        let started = ProcessInfo.processInfo.systemUptime
        try await generator.prepare(modelPath: root, draftModel: draft)
        let load = ProcessInfo.processInfo.systemUptime - started
        let prepared = Memory.activeMemory
        let first = try XCTUnwrap(samples.first)
        let warmup = try await generator.transcribePreparedMeasured(samples: AudioReader.readAudio(from: URL(fileURLWithPath: first.audio)))
        var entries: [QualityEntry] = []
        let json = JSONEncoder(); json.outputFormatting = [.prettyPrinted, .sortedKeys]
        func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
        for sample in samples {
            let audio = try AudioReader.readAudio(from: URL(fileURLWithPath: sample.audio))
            let measurement = try await generator.transcribePreparedMeasured(samples: audio)
            entries.append(QualityEntry(sample: sample, measurement: measurement))
            let report = QualityReport(policy: policyName, manifestSHA256: digest(manifestData), recipeSHA256: recipeData.map(digest),
                preparedActiveBytes: prepared, modelLoadSeconds: load, operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
                warmup: warmup, entries: entries)
            try json.encode(report).write(to: URL(fileURLWithPath: output), options: .atomic)
            print("SOV held-out \(policyName) \(entries.count)/\(samples.count): \(sample.id)")
        }
        await generator.unload()
    }
}
