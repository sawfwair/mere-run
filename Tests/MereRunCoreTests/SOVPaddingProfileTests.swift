import AudioCodecs
import CryptoKit
import Foundation
import MLX
import XCTest
@testable import AudioSTT

/// Optional real-checkpoint input-policy qualification. Timings here are diagnostic.
final class SOVPaddingProfileTests: MereRunCoreTestCase {
    private struct Entry: Encodable {
        let sample: SOVLowBitSample
        let measurement: SOVMeasuredTranscription
    }
    private struct Condition: Encodable {
        let entries: [Entry]
    }
    private struct Candidate: Encodable {
        let margin: Int
        let entries: [Entry]
    }
    private struct Report: Encodable {
        let schemaVersion: Int
        let manifestSHA256: String
        let packedWeights: SOVPackedWeightsMeasurement?
        let operatingSystem: String
        let baseline: Condition
        let candidates: [Candidate]
    }

    func testHeldOutRecognition() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_MLX_DEVICE"] == "gpu", let root = env["MERERUN_TEST_SOV_PATH"],
              let draft = env["MERERUN_TEST_SOV_DFLASH_PATH"], let packed = env["MERERUN_TEST_SOV_PACKED_PATH"],
              let manifest = env["MERERUN_SOV_PADDING_EVALUATION"], let output = env["MERERUN_SOV_PADDING_OUTPUT"] else {
            throw XCTSkip("Requires GPU, SOV checkpoints, packed Q8 weights, padding evaluation manifest and output.")
        }
        let margins = try JSONDecoder().decode([Int].self, from: Data((env["MERERUN_SOV_PADDING_MARGINS"] ?? "[25,12,0]").utf8))
        XCTAssertFalse(margins.isEmpty)
        XCTAssertEqual(Set(margins).count, margins.count)
        for margin in margins { try SOVAudioTokenTrimming.validate(margin: margin) }
        let data = try Data(contentsOf: URL(fileURLWithPath: manifest))
        let samples = try JSONDecoder().decode([SOVLowBitSample].self, from: data)
        let generator = SOVTranscribeGenerator(decoderExperts: .q8Metal, experimentalAttention: true,
            generationExperiment: .rollback, packedWeights: URL(fileURLWithPath: packed))
        try await generator.prepare(modelPath: root, draftModel: draft)
        let packedMeasurement = await generator.packedWeightsMeasurement
        var baseline: [Entry] = []
        var candidates = margins.map { Candidate(margin: $0, entries: []) }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        for (index, sample) in samples.enumerated() {
            let audio = try AudioReader.readAudio(from: URL(fileURLWithPath: sample.audio))
            // Rotate complete condition order. All conditions share one prepared model.
            let policies: [Int?] = [nil] + margins.map(Optional.some)
            let offset = index % policies.count
            let order = Array(policies[offset...] + policies[..<offset])
            for margin in order {
                try await generator.setAudioTokenMarginForBenchmark(margin)
                let measurement = try await generator.transcribePreparedMeasured(samples: audio)
                let entry = Entry(sample: sample, measurement: measurement)
                if let margin {
                    let candidate = try XCTUnwrap(candidates.firstIndex { $0.margin == margin })
                    candidates[candidate] = Candidate(margin: margin, entries: candidates[candidate].entries + [entry])
                } else { baseline.append(entry) }
            }
            let report = Report(schemaVersion: 1,
                manifestSHA256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
                packedWeights: packedMeasurement, operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
                baseline: Condition(entries: baseline), candidates: candidates)
            try encoder.encode(report).write(to: URL(fileURLWithPath: output), options: .atomic)
            print("SOV padding \(index + 1)/\(samples.count): \(sample.id)")
        }
        await generator.unload()
    }
}
