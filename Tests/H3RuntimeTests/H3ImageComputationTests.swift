import Foundation
import MLX
import MLXNN
import XCTest
import MereRunMLXTestSupport
@testable import MereRunH3Model

final class H3ImageComputationTests: MLXTestCase {
    private let fixtureRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/H3Image")

    func testUpstreamFP32AndMixedBF16SixStepTrajectories() throws {
        let inputs = try MLX.loadArrays(url: fixtureRoot.appending(path: "inputs.safetensors"))
        for suffix in ["fp32", "bf16"] {
            let model = MiniMaxH3ImageTransformer(configuration: .init(hiddenSize: 12, layerCount: 2,
                refinerLayerCount: 1, attentionHeadCount: 2, attentionHeadDimension: 8, feedForwardSize: 16,
                textDimension: 10, timeFrequencyDimension: 8, timeEmbeddingHiddenSize: 12,
                timeEmbeddingDimension: 6, ropeFrequencyCount: 1))
            try model.update(parameters: ModuleParameters.unflattened(
                MLX.loadArrays(url: fixtureRoot.appending(path: "weights-\(suffix).safetensors")).map { ($0.key, $0.value) }), verify: .all)
            let expected = try MLX.loadArrays(url: fixtureRoot.appending(path: "outputs-\(suffix).safetensors"))
            let text = model.prepareText(try XCTUnwrap(inputs["text"]))
            for references in [false, true] {
                // Exercise bounded projections against the same independent upstream outputs.
                model.maximumProjectionRows = references ? 2 : 8_192
                model.maximumAttentionHeadsPerKernel = 1
                model.maximumAttentionQueryTokens = 2
                let label = references ? "edit" : "text"
                let layout = MiniMaxH3ImageLayout(textTags: [1, 0, 1],
                    referenceGrids: references ? [(2, 2)] : [], height: 4, width: 4)
                let initial = try XCTUnwrap(inputs["image"])
                let reference = try XCTUnwrap(inputs["reference"])
                var state = initial
                for step in 0..<6 {
                    let rows = references ? MLX.concatenated([reference, initial], axis: 1) : initial
                    let actual = try model(imageRows: rows, text: text, layout: layout, step: step)
                    assertClose(actual, try XCTUnwrap(expected["\(label)_step_\(step)"]), tolerance: suffix == "fp32" ? 2e-6 : 0.006)
                    let evolving = references ? MLX.concatenated([reference, state], axis: 1) : state
                    let prediction = try model(imageRows: evolving, text: text, layout: layout, step: step)
                    state = MiniMaxH3ImageSampling.step(sample: state, velocity: prediction, index: step)
                    MLX.eval(state)
                }
                assertClose(state, try XCTUnwrap(expected["\(label)_trajectory"]), tolerance: suffix == "fp32" ? 2e-6 : 0.006)
            }
        }
    }

    func testImageVAEDecoderMatchesUpstreamTwoAxisRotaryAndRegisters() throws {
        let model = MiniMaxH3ImageDecoder(configuration: .init(temporalPatchSize: 1,
            layerCount: 2, headCount: 2, headDimension: 8, rotaryDimensionRatio: 0.5))
        let weights = try MLX.loadArrays(url: fixtureRoot.appending(path: "decoder-weights.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(weights.map { ($0.key, $0.value) }), verify: .all)
        let inputs = try MLX.loadArrays(url: fixtureRoot.appending(path: "inputs.safetensors"))
        let expected = try MLX.loadArrays(url: fixtureRoot.appending(path: "decoder-output.safetensors"))
        let actual = model(try XCTUnwrap(inputs["decoder_latents"]))
        XCTAssertEqual(actual.shape, [1, 3, 32, 48])
        assertClose(actual, try XCTUnwrap(expected["rgb"]), tolerance: 3e-6)
    }

    func testVAEDecodeTileBatchesPreserveGeometryAndSamples() throws {
        let model = MiniMaxH3ImageDecoder(configuration: .init(temporalPatchSize: 1,
            layerCount: 2, headCount: 2, headDimension: 8, rotaryDimensionRatio: 0.5))
        let weights = try MLX.loadArrays(url: fixtureRoot.appending(path: "decoder-weights.safetensors"))
        try model.update(parameters: ModuleParameters.unflattened(weights.map { ($0.key, $0.value) }), verify: .all)
        let input = MLX.sin(MLXArray(0..<(2 * 24 * 65 * 65)).asType(.float32) * 0.013).reshaped(2, 24, 65, 65)
        let sequential = MiniMaxH3ImageVAE.tiled(input, heightAxis: 2, widthAxis: 3,
            outputRatio: 1, latentInput: true, maximumBatchSize: 1) { model($0) }
        MLX.eval(sequential)
        XCTAssertEqual(sequential.shape, [2, 3, 1_040, 1_040])
        for batch in [2, 3, 32] {
            let actual = MiniMaxH3ImageVAE.tiled(input, heightAxis: 2, widthAxis: 3,
                outputRatio: 1, latentInput: true, maximumBatchSize: batch) { model($0) }
            assertClose(actual, sequential, tolerance: 3e-6)
        }
    }

    func testExactModulationCacheAndPreparedLayoutMatchUpstreamSixStepTrajectories() throws {
        let inputs = try MLX.loadArrays(url: fixtureRoot.appending(path: "inputs.safetensors"))
        for suffix in ["fp32", "bf16"] {
            let model = MiniMaxH3ImageTransformer(configuration: .init(hiddenSize: 12, layerCount: 2,
                refinerLayerCount: 1, attentionHeadCount: 2, attentionHeadDimension: 8, feedForwardSize: 16,
                textDimension: 10, timeFrequencyDimension: 8, timeEmbeddingHiddenSize: 12,
                timeEmbeddingDimension: 6, ropeFrequencyCount: 1))
            try model.update(parameters: ModuleParameters.unflattened(
                MLX.loadArrays(url: fixtureRoot.appending(path: "weights-\(suffix).safetensors")).map { ($0.key, $0.value) }), verify: .all)
            let text = model.prepareText(try XCTUnwrap(inputs["text"]))
            model.installModulationCache(model.precomputeModulations())
            XCTAssertFalse(model.parameters().flattened().contains { $0.0.contains("adaln_proj") && $0.1.size > 1 })
            let expected = try MLX.loadArrays(url: fixtureRoot.appending(path: "outputs-\(suffix).safetensors"))
            for references in [false, true] {
                model.maximumProjectionRows = references ? 2 : 8_192
                model.maximumAttentionHeadsPerKernel = 1
                model.maximumAttentionQueryTokens = 2
                let layout = model.prepareLayout(MiniMaxH3ImageLayout(textTags: [1, 0, 1],
                    referenceGrids: references ? [(2, 2)] : [], height: 4, width: 4))
                XCTAssertEqual(layout.rope.cosine.dtype, suffix == "fp32" ? .float32 : .bfloat16)
                XCTAssertEqual(layout.rope.sine.dtype, layout.rope.cosine.dtype)
                var state = try XCTUnwrap(inputs["image"])
                for step in 0..<6 {
                    let rows = references ? MLX.concatenated([try XCTUnwrap(inputs["reference"]), state], axis: 1) : state
                    let velocity = try model(imageRows: rows, text: text, preparedLayout: layout, step: step)
                    state = MiniMaxH3ImageSampling.step(sample: state, velocity: velocity, index: step)
                    MLX.eval(state)
                }
                assertClose(state, try XCTUnwrap(expected["\(references ? "edit" : "text")_trajectory"]),
                            tolerance: suffix == "fp32" ? 2e-6 : 0.006)
            }
        }
    }

    func testPackedImageLayoutUsesImageSlotsAndReferenceOnlyTimeIndex() {
        let layout = MiniMaxH3ImageLayout(textTags: [1, 0, 1], referenceGrids: [(2, 4), (4, 2)], height: 4, width: 4)
        XCTAssertEqual(layout.targetRows, 7..<11)
        XCTAssertEqual(layout.adaLNIndices().asArray(Int32.self), [1, 0, 1, 3, 3, 3, 3, 0, 0, 0, 0])
        let positions = layout.positions.asArray(Float.self)
        XCTAssertEqual(stride(from: 0, to: positions.count, by: 3).map { positions[$0] }, [0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5])
        XCTAssertEqual(Array(positions.suffix(12)), [5, 0, 0, 5, 0, 16, 5, 16, 0, 5, 16, 16])
    }

    func testScheduleAndCanvasFollowReleasedRounding() {
        XCTAssertEqual(MiniMaxH3ImageSampling.sigmas.count, 7)
        XCTAssertEqual(MiniMaxH3ImageSampling.sigmas.last, 0)
        XCTAssertEqual(MiniMaxH3ImageSampling.sigmas[5], Float(1) - Float(1 - 0.23200887441635132))
        let canvas = MiniMaxH3ImageGeometry.canvas(width: 1_024, height: 1_024)
        XCTAssertEqual(canvas.width, 2_048)
        XCTAssertEqual(canvas.height, 2_048)
        let unchanged = MiniMaxH3ImageGeometry.referenceSize(width: 1_501, height: 1_025, vision: false)
        XCTAssertEqual(unchanged.width, 1_501)
        XCTAssertEqual(unchanged.height, 1_025)
        let processor = MiniMaxH3ImageGeometry.visionProcessorSize(width: 1_501, height: 1_024)
        XCTAssertEqual(processor.width, 1_504)
        XCTAssertEqual(processor.height, 1_024)
    }

    func testPositiveDataWardVelocityIncreasesSampleAsSigmaFalls() {
        var sample = MLXArray([Float(-1), 2])
        let velocity = MLXArray([Float(3), -1])
        for step in 0..<6 {
            sample = MiniMaxH3ImageSampling.step(sample: sample, velocity: velocity, index: step)
        }
        assertClose(sample, MLXArray([Float(2), 1]), tolerance: 1e-6)
    }

    private func assertClose(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        let error = MLX.max(MLX.abs(actual.asType(.float32) - expected.asType(.float32))).item(Float.self)
        XCTAssertLessThan(error, tolerance, "Maximum error: \(error)", file: file, line: line)
    }
}
