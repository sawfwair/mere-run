import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class LTXTeaCacheTests: XCTestCase {
    func testLTX25CalibrationIsGuidanceGroupAndSamplerSpecific() {
        let euler = LTXTeaCacheCalibration.ltx25(
            sampler: .euler,
            key: LTXTeaCacheKey(
                branch: .conditioned,
                stage: .primary,
                pipelineStage: .coarse
            )
        )
        let res2S = LTXTeaCacheCalibration.ltx25(
            sampler: .res2s,
            key: LTXTeaCacheKey(
                branch: .isolated,
                stage: .midpoint,
                pipelineStage: .coarse
            )
        )

        XCTAssertEqual(euler.coefficients.count, 3)
        XCTAssertEqual(euler.threshold, 0.235, accuracy: 0.0001)
        XCTAssertEqual(res2S.coefficients.count, 3)
        XCTAssertEqual(res2S.threshold, 0.39, accuracy: 0.0001)
        XCTAssertNotEqual(euler.coefficients, res2S.coefficients)
    }

    func testControllerComputesBoundariesAndReusesCachedResidual() {
        let controller = LTXTeaCacheController(
            configuration: LTXTeaCacheConfiguration(threshold: 0.7),
            sampler: .euler
        )
        let key = LTXTeaCacheKey(
            branch: .conditioned,
            stage: .primary,
            pipelineStage: .coarse
        )
        let gate = MLX.ones([1, 2, 2], dtype: .float32)
        let residual = MLX.full([1, 2, 2], values: MLXArray(Float(2)))

        XCTAssertCompute(
            controller.decide(
                request: LTXTeaCacheRequest(key: key, stepIndex: 0, stepCount: 4),
                gate: gate
            )
        )
        controller.recordComputedResidual(
            request: LTXTeaCacheRequest(key: key, stepIndex: 0, stepCount: 4),
            videoResidual: residual,
            audioResidual: residual
        )

        switch controller.decide(
            request: LTXTeaCacheRequest(key: key, stepIndex: 1, stepCount: 4),
            gate: gate
        ) {
        case .compute:
            XCTFail("Expected the first sub-threshold delta to reuse the cached residual.")
        case .reuse(let videoResidual, let audioResidual):
            XCTAssertEqual(videoResidual.shape, residual.shape)
            XCTAssertEqual(audioResidual.shape, residual.shape)
        }

        XCTAssertCompute(
            controller.decide(
                request: LTXTeaCacheRequest(key: key, stepIndex: 2, stepCount: 4),
                gate: gate
            )
        )
        XCTAssertCompute(
            controller.decide(
                request: LTXTeaCacheRequest(key: key, stepIndex: 3, stepCount: 4),
                gate: gate
            )
        )
        XCTAssertEqual(controller.metrics.computedBlockStacks, 3)
        XCTAssertEqual(controller.metrics.reusedBlockStacks, 1)
    }

    func testControllerSynchronizesReuseAcrossGuidanceBranches() {
        let controller = LTXTeaCacheController(
            configuration: LTXTeaCacheConfiguration(threshold: 1.0),
            sampler: .euler
        )
        let conditionedKey = LTXTeaCacheKey(
            branch: .conditioned,
            stage: .primary,
            pipelineStage: .coarse
        )
        let unconditionalKey = LTXTeaCacheKey(
            branch: .unconditional,
            stage: .primary,
            pipelineStage: .coarse
        )
        let gate = MLX.ones([1, 2, 2], dtype: .float32)
        let residual = MLX.full([1, 2, 2], values: MLXArray(Float(2)))

        for key in [conditionedKey, unconditionalKey] {
            let request = LTXTeaCacheRequest(key: key, stepIndex: 0, stepCount: 4)
            XCTAssertCompute(controller.decide(request: request, gate: gate))
            controller.recordComputedResidual(
                request: request,
                videoResidual: residual,
                audioResidual: residual
            )
        }

        XCTAssertReuse(
            controller.decide(
                request: LTXTeaCacheRequest(
                    key: conditionedKey,
                    stepIndex: 1,
                    stepCount: 4
                ),
                gate: gate
            )
        )
        XCTAssertReuse(
            controller.decide(
                request: LTXTeaCacheRequest(
                    key: unconditionalKey,
                    stepIndex: 1,
                    stepCount: 4
                ),
                gate: gate * MLXArray(Float(100))
            )
        )
        XCTAssertEqual(controller.metrics.computedBlockStacks, 2)
        XCTAssertEqual(controller.metrics.reusedBlockStacks, 2)
    }

    func testCalibrationModeNeverSkipsAndWritesDriftSample() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString,
            isDirectory: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = directory.appendingPathComponent("calibration.json")
        let controller = LTXTeaCacheController(
            configuration: LTXTeaCacheConfiguration(calibrationOutputURL: output),
            sampler: .res2s
        )
        let key = LTXTeaCacheKey(
            branch: .conditioned,
            stage: .midpoint,
            pipelineStage: .detail
        )
        let first = MLX.ones([1, 2, 2], dtype: .float32)
        let second = first * MLXArray(Float(2))

        XCTAssertCompute(
            controller.decide(
                request: LTXTeaCacheRequest(key: key, stepIndex: 0, stepCount: 3),
                gate: first
            )
        )
        controller.recordComputedResidual(
            request: LTXTeaCacheRequest(key: key, stepIndex: 0, stepCount: 3),
            videoResidual: first,
            audioResidual: first
        )
        XCTAssertCompute(
            controller.decide(
                request: LTXTeaCacheRequest(key: key, stepIndex: 1, stepCount: 3),
                gate: second
            )
        )
        controller.recordComputedResidual(
            request: LTXTeaCacheRequest(key: key, stepIndex: 1, stepCount: 3),
            videoResidual: second,
            audioResidual: second
        )
        try controller.writeCalibrationReport()

        let report = try JSONDecoder().decode(
            LTXTeaCacheCalibrationReport.self,
            from: Data(contentsOf: output)
        )
        XCTAssertEqual(report.sampler, LTXSamplerMode.res2s.rawValue)
        XCTAssertEqual(report.samples.count, 1)
        XCTAssertEqual(report.samples[0].inputRelativeL1, 1, accuracy: 0.0001)
        XCTAssertEqual(report.samples[0].outputRelativeL1, 1, accuracy: 0.0001)
    }

    private func XCTAssertCompute(
        _ decision: LTXTeaCacheDecision,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        if case .reuse = decision {
            XCTFail("Expected the transformer stack to compute.", file: file, line: line)
        }
    }

    private func XCTAssertReuse(
        _ decision: LTXTeaCacheDecision,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        if case .compute = decision {
            XCTFail("Expected the transformer stack to reuse its residual.", file: file, line: line)
        }
    }
}
