import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class CLIGenerationProgressPrinterTests: XCTestCase {
    func testProgressJSONLineEncodesStageStepAndTotal() {
        let line = CLIGenerationProgressPrinter.progressJSONLine(
            GenerationProgress(stage: .denoising, stepIndex: 2, totalSteps: 4)
        )

        XCTAssertEqual(
            line,
            #"{"event":"progress","stage":"denoising","step":2,"total_steps":4}"#
        )

        let decoded = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any]
        XCTAssertEqual(decoded?["event"] as? String, "progress")
        XCTAssertEqual(decoded?["stage"] as? String, "denoising")
        XCTAssertEqual(decoded?["step"] as? Int, 2)
        XCTAssertEqual(decoded?["total_steps"] as? Int, 4)
    }

    func testProgressJSONLineEncodesLoadingStages() {
        let line = CLIGenerationProgressPrinter.progressJSONLine(
            GenerationProgress(stage: .loadingModel, stepIndex: 3, totalSteps: 7)
        )

        XCTAssertEqual(
            line,
            #"{"event":"progress","stage":"loadingModel","step":3,"total_steps":7}"#
        )
    }

    func testJSONProgressHandlerEmitsOneLinePerDistinctEvent() {
        final class Sink: @unchecked Sendable {
            var lines: [String] = []
        }

        let sink = Sink()
        let handler = CLIGenerationProgressPrinter.makeJSONProgressHandler { text in
            sink.lines.append(text)
        }

        let events = [
            GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: 1),
            GenerationProgress(stage: .denoising, stepIndex: 0, totalSteps: 4),
            GenerationProgress(stage: .denoising, stepIndex: 0, totalSteps: 4),
            GenerationProgress(stage: .denoising, stepIndex: 1, totalSteps: 4),
            GenerationProgress(stage: .denoising, stepIndex: 4, totalSteps: 4),
        ]
        events.forEach(handler)

        XCTAssertEqual(sink.lines, [
            "{\"event\":\"progress\",\"stage\":\"encodingText\",\"step\":0,\"total_steps\":1}\n",
            "{\"event\":\"progress\",\"stage\":\"denoising\",\"step\":0,\"total_steps\":4}\n",
            "{\"event\":\"progress\",\"stage\":\"denoising\",\"step\":1,\"total_steps\":4}\n",
            "{\"event\":\"progress\",\"stage\":\"denoising\",\"step\":4,\"total_steps\":4}\n",
        ])
        XCTAssertTrue(sink.lines.allSatisfy { $0.hasSuffix("\n") && !$0.dropLast().contains("\n") })
    }
}
