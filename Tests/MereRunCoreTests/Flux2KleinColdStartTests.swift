import Foundation
import MLX
import MereRunImageModels
import XCTest
@testable import MereRunCore

final class Flux2KleinColdStartTests: MereRunCoreTestCase {
    func testStagedFirstStepDefaultsToLinuxGPUOnly() {
        XCTAssertTrue(
            Flux2KleinGenerator.stagedFirstStepEvaluationEnabled(
                environment: [:], runsOnLinuxGPU: true, compileEnabled: false
            )
        )
        XCTAssertFalse(
            Flux2KleinGenerator.stagedFirstStepEvaluationEnabled(
                environment: [:], runsOnLinuxGPU: false, compileEnabled: false
            )
        )
    }

    func testStagedFirstStepHonorsExplicitOverride() {
        let key = Flux2KleinGenerator.stagedFirstStepEnvironmentKey
        XCTAssertEqual(key, "MERERUN_FLUX2_STAGED_FIRST_STEP")
        for raw in ["1", "true", " yes ", "ON"] {
            XCTAssertTrue(
                Flux2KleinGenerator.stagedFirstStepEvaluationEnabled(
                    environment: [key: raw], runsOnLinuxGPU: false, compileEnabled: false
                ),
                "\(raw) should force staging on"
            )
        }
        for raw in ["0", "false", "no", "off"] {
            XCTAssertFalse(
                Flux2KleinGenerator.stagedFirstStepEvaluationEnabled(
                    environment: [key: raw], runsOnLinuxGPU: true, compileEnabled: false
                ),
                "\(raw) should force staging off"
            )
        }
        XCTAssertTrue(
            Flux2KleinGenerator.stagedFirstStepEvaluationEnabled(
                environment: [key: "maybe"], runsOnLinuxGPU: true, compileEnabled: false
            ),
            "An unrecognized value falls back to the platform default."
        )
    }

    func testStagedFirstStepNeverRunsUnderCompile() {
        let key = Flux2KleinGenerator.stagedFirstStepEnvironmentKey
        XCTAssertFalse(
            Flux2KleinGenerator.stagedFirstStepEvaluationEnabled(
                environment: [key: "1"], runsOnLinuxGPU: true, compileEnabled: true
            )
        )
    }

    func testStagedEvaluationHandlerMaterializesStagesAndReportsSlowOnes() {
        var ticks: [TimeInterval] = [10.0, 10.2, 10.3, 15.3, 15.35]
        var reports: [String] = []
        let handler = Flux2KleinGenerator.makeStagedEvaluationHandler(
            reportEveryStage: false,
            clock: { ticks.removeFirst() },
            report: { reports.append($0) }
        )

        let embedding = MLXArray([Float(1), 2, 3]) * MLXArray(Float(2))
        handler(Flux2TransformerForwardStage(kind: .embedding, index: 0, count: 1), [embedding])
        XCTAssertEqual(embedding.asArray(Float.self), [2, 4, 6])
        XCTAssertTrue(reports.isEmpty, "A fast stage stays quiet.")

        handler(Flux2TransformerForwardStage(kind: .jointBlock, index: 0, count: 5), [MLXArray(Float(1))])
        XCTAssertTrue(reports.isEmpty)

        handler(Flux2TransformerForwardStage(kind: .jointBlock, index: 1, count: 5), [MLXArray(Float(1))])
        XCTAssertEqual(
            reports,
            ["[Flux2KleinGenerator] first_step_stage=joint_block=2/5 stage_s=5.000 elapsed_s=5.300"]
        )

        handler(Flux2TransformerForwardStage(kind: .output, index: 0, count: 1), [MLXArray(Float(1))])
        XCTAssertEqual(reports.count, 1, "Only stages at or above the slow threshold are reported.")
    }

    func testStagedEvaluationHandlerReportsEveryStageWhenAsked() {
        var ticks: [TimeInterval] = [0.0, 0.01, 0.02]
        var reports: [String] = []
        let handler = Flux2KleinGenerator.makeStagedEvaluationHandler(
            reportEveryStage: true,
            clock: { ticks.removeFirst() },
            report: { reports.append($0) }
        )
        handler(Flux2TransformerForwardStage(kind: .embedding, index: 0, count: 1), [MLXArray(Float(1))])
        handler(Flux2TransformerForwardStage(kind: .singleBlock, index: 19, count: 20), [MLXArray(Float(1))])
        XCTAssertEqual(reports, [
            "[Flux2KleinGenerator] first_step_stage=embedding stage_s=0.010 elapsed_s=0.010",
            "[Flux2KleinGenerator] first_step_stage=single_block=20/20 stage_s=0.010 elapsed_s=0.020",
        ])
        XCTAssertEqual(
            Flux2KleinGenerator.firstStepReportLine(seconds: 187.25),
            "[Flux2KleinGenerator] first_step_s=187.250 staged=1"
        )
    }
}
