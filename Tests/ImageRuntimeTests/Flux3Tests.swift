import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunImageModels

final class Flux3Tests: MLXTestCase {
    private let config = Flux3Configuration(hiddenSize: 16, heads: 2, earlyDepth: 1, singleDepth: 2,
        imageChannels: 8, textChannels: 12, axesDimensions: [2, 2, 2, 2])

    private func arrays(_ name: String) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appending(path: "Fixtures/Flux3/\(name).safetensors"))
    }

    func testTransformerMatchesPinnedBFLReference() throws {
        for dtype in [DType.float32, .bfloat16] {
            let weights = try arrays("weights").mapValues { $0.asType(dtype) }
            let model = try Flux3Transformer(configuration: config, weights: weights)
            let input = try arrays("inputs"), expected = try arrays(dtype == .float32 ? "results-fp32" : "results-bf16")
            func predict(_ image: MLXArray, _ time: Float, _ target: Float, references: Bool) throws -> MLXArray {
                try model(image: image, imageIDs: input["image_ids"]!, text: input["text"]!, textIDs: input["text_ids"]!,
                    references: references ? input["references"] : nil, referenceIDs: references ? input["reference_ids"] : nil,
                    timestep: time, nextTimestep: target)
            }
            let tolerance: Float = dtype == .float32 ? 2e-5 : 0.008
            assertClose(try predict(input["image"]!, 0.75, 0.5, references: false), expected["text_only"]!, tolerance)
            assertClose(try predict(input["image"]!, 0.413725, 0.125, references: true), expected["references"]!, tolerance)
            assertClose(try predict(input["image"]!, 0.413725, 0.25, references: true), expected["changed_target"]!, tolerance)
            let schedule = try Flux3Sampling.schedule(steps: 3)
            var state = input["image"]!
            for step in 0..<3 {
                let prediction = try predict(state, schedule[step], schedule[step + 1], references: true)
                state = state + (schedule[step + 1] - schedule[step]) * prediction.asType(.float32)
            }
            assertClose(state, expected["trajectory"]!, tolerance)
            XCTAssertThrowsError(try predict(input["image"]!, 0.5, 0.75, references: false))
        }
    }

    func testStrictWeightSchemaRejectsBaseQuantizedAndIncompletePackages() throws {
        let weights = try arrays("weights")
        XCTAssertEqual(Set(weights.keys), Set(Flux3Transformer.weightShapes(config).keys))
        XCTAssertThrowsError(try Flux3Transformer(configuration: config, weights: weights.filter { !$0.key.hasPrefix("ref_time_in.") }))
        var unexpected = weights
        unexpected["txt_in.weight_scale"] = MLXArray(1)
        XCTAssertThrowsError(try Flux3Transformer(configuration: config, weights: unexpected))
        var malformed = weights
        malformed["txt_in.weight"] = zeros([1, 1])
        XCTAssertThrowsError(try Flux3Transformer(configuration: config, weights: malformed))
    }

    func testLatentPatchOrderBatchNormAndReferenceCoordinates() {
        let mean = MLXArray(0..<(32 * 4 * 6)).asType(.float32).reshaped(1, 32, 4, 6)
        let runningMean = MLXArray(0..<128).asType(.float32) / 10
        let variance = ones([128]) * 4
        let tokens = Flux3Latents.pack(mean, runningMean: runningMean, runningVariance: variance)
        let raw = tokens[0] * sqrt(variance + 1e-4) + runningMean
        assertClose(raw[0..<8], MLXArray([Float(0), 1, 6, 7, 24, 25, 30, 31]), 1e-4)
        let restored = Flux3Latents.unpack(tokens, height: 2, width: 3, runningMean: runningMean, runningVariance: variance)
        assertClose(restored, mean, 1e-4)
        XCTAssertEqual(Flux3Latents.imageIDs(height: 1, width: 2, referenceIndex: 1).asArray(Float.self),
                       [0, 0, 0, -2, 0, 0, 0.5, -2])
        XCTAssertEqual(Flux3Latents.textIDs(length: 2).asArray(Float.self), [0, 0, 0, 0, 0, 0, 0, 1])
    }

    func testScheduleAndTextPaddingContract() throws {
        let schedule = try Flux3Sampling.schedule(steps: 16)
        XCTAssertEqual(schedule.count, 17)
        XCTAssertEqual(schedule.first, 1)
        XCTAssertEqual(schedule.last, 0)
        XCTAssertTrue(zip(schedule, schedule.dropFirst()).allSatisfy { $0 > $1 })
        XCTAssertThrowsError(try Flux3Sampling.schedule(steps: 0))
        XCTAssertThrowsError(try Flux3Sampling.schedule(steps: 2, alpha: .infinity))
        XCTAssertEqual(Flux3PromptPlan.paddedLength(tokenCount: 1), 80)
        XCTAssertEqual(Flux3PromptPlan.paddedLength(tokenCount: 80), 80)
        XCTAssertEqual(Flux3PromptPlan.paddedLength(tokenCount: 81), 160)
        XCTAssertEqual(Flux3PromptPlan.paddedLength(tokenCount: 4090), 4096)
        XCTAssertEqual(Flux3PromptPlan.outputLayers, [4, 8, 12, 16, 20, 24, 28, 32])
    }

    private func assertClose(_ actual: MLXArray, _ expected: MLXArray, _ tolerance: Float,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        let error = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
        XCTAssertLessThanOrEqual(error, tolerance, file: file, line: line)
    }
}
