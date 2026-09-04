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

    func testGenericProgressLineMatchesTheImageEventShape() {
        XCTAssertEqual(
            CLIGenerationProgressPrinter.progressJSONLine(stage: "denoising", step: 2, totalSteps: 4),
            CLIGenerationProgressPrinter.progressJSONLine(
                GenerationProgress(stage: .denoising, stepIndex: 2, totalSteps: 4)
            )
        )
        XCTAssertEqual(
            CLIGenerationProgressPrinter.progressJSONLine(stage: "generating", step: 50, totalSteps: 0),
            #"{"event":"progress","stage":"generating","step":50,"total_steps":0}"#
        )
    }

    // MARK: - The documented convention: 0-based in progress, one terminal step == total_steps

    private struct Event: Equatable {
        let stage: String
        let step: Int
        let total: Int
    }

    private func events(_ lines: [String]) throws -> [Event] {
        try lines.map { line in
            XCTAssertTrue(line.hasSuffix("\n") && !line.dropLast().contains("\n"), "not one NDJSON line: \(line)")
            let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
            XCTAssertEqual(object["event"] as? String, "progress")
            return Event(
                stage: try XCTUnwrap(object["stage"] as? String),
                step: try XCTUnwrap(object["step"] as? Int),
                total: try XCTUnwrap(object["total_steps"] as? Int)
            )
        }
    }

    private func makeStream() -> (JSONProgressStream, () -> [String]) {
        final class Sink: @unchecked Sendable {
            var lines: [String] = []
        }
        let sink = Sink()
        return (JSONProgressStream { sink.lines.append($0) }, { sink.lines })
    }

    func testSFXCompletionCountsBecomeZeroBasedStepsPlusOneTerminalEvent() throws {
        let (stream, lines) = makeStream()
        // Woosh and MMAudio both report `completed` 1...N (WooshGenerator and
        // MMAudioNetwork call `progress?(step + 1, config.steps)`).
        for completed in 1...4 {
            let progress = SFXGenerate.denoiseProgressEvent(completed: completed, total: 4)
            stream.report(stage: progress.stage, step: progress.step, totalSteps: progress.totalSteps)
        }
        stream.finish()

        XCTAssertEqual(try events(lines()), [
            Event(stage: "denoising", step: 0, total: 4),
            Event(stage: "denoising", step: 1, total: 4),
            Event(stage: "denoising", step: 2, total: 4),
            Event(stage: "denoising", step: 3, total: 4),
            Event(stage: "denoising", step: 4, total: 4),
        ])
    }

    func testMiniMaxMusicCountersFlattenToZeroBasedStepsAndCloseEachStage() throws {
        let semantic = MusicGenerate.miniMaxProgressEvent(.semantic(frame: 1, maximum: 1_500), strategy: .sequential)
        XCTAssertEqual(semantic.stage, "semantic")
        XCTAssertEqual(semantic.step, 0)
        XCTAssertEqual(semantic.totalSteps, 1_500)

        let firstChunk = MusicGenerate.miniMaxProgressEvent(
            .denoise(chunk: 1, chunkCount: 3, step: 1, stepCount: 30), strategy: .sequential
        )
        XCTAssertEqual(firstChunk.stage, "denoising")
        XCTAssertEqual(firstChunk.step, 0)
        XCTAssertEqual(firstChunk.totalSteps, 90)

        let lastChunk = MusicGenerate.miniMaxProgressEvent(
            .denoise(chunk: 3, chunkCount: 3, step: 30, stepCount: 30), strategy: .sequential
        )
        XCTAssertEqual(lastChunk.step, 89)
        XCTAssertEqual(lastChunk.totalSteps, 90)

        // `sequential` denoises chunk by chunk (MiniMaxMusic3Pipeline.denoise).
        let (stream, lines) = makeStream()
        for event: MiniMaxMusic3Progress in [
            .semantic(frame: 1, maximum: 2), .semantic(frame: 2, maximum: 2),
            .denoise(chunk: 1, chunkCount: 2, step: 1, stepCount: 2), .denoise(chunk: 1, chunkCount: 2, step: 2, stepCount: 2),
            .denoise(chunk: 2, chunkCount: 2, step: 1, stepCount: 2), .denoise(chunk: 2, chunkCount: 2, step: 2, stepCount: 2),
            .decode(chunk: 1, chunkCount: 2), .decode(chunk: 2, chunkCount: 2),
        ] {
            let progress = MusicGenerate.miniMaxProgressEvent(event, strategy: .sequential)
            stream.report(stage: progress.stage, step: progress.step, totalSteps: progress.totalSteps)
        }
        stream.finish()

        XCTAssertEqual(try events(lines()), [
            Event(stage: "semantic", step: 0, total: 2),
            Event(stage: "semantic", step: 1, total: 2),
            Event(stage: "semantic", step: 2, total: 2),
            Event(stage: "denoising", step: 0, total: 4),
            Event(stage: "denoising", step: 1, total: 4),
            Event(stage: "denoising", step: 2, total: 4),
            Event(stage: "denoising", step: 3, total: 4),
            Event(stage: "denoising", step: 4, total: 4),
            Event(stage: "decoding", step: 0, total: 2),
            Event(stage: "decoding", step: 1, total: 2),
            Event(stage: "decoding", step: 2, total: 2),
        ])
    }

    func testMiniMaxMusicOverlapAverageStaysMonotonicAcrossWindows() throws {
        // `denoiseOverlapAverage` runs one flow step across every window before
        // advancing, the opposite order from `sequential`. Flattening chunk-major
        // there would make `step` fall back and close `denoising` on every step.
        let (stream, lines) = makeStream()
        for event: MiniMaxMusic3Progress in [
            .denoise(chunk: 1, chunkCount: 2, step: 1, stepCount: 2), .denoise(chunk: 2, chunkCount: 2, step: 1, stepCount: 2),
            .denoise(chunk: 1, chunkCount: 2, step: 2, stepCount: 2), .denoise(chunk: 2, chunkCount: 2, step: 2, stepCount: 2),
        ] {
            let progress = MusicGenerate.miniMaxProgressEvent(event, strategy: .overlapAverage)
            stream.report(stage: progress.stage, step: progress.step, totalSteps: progress.totalSteps)
        }
        stream.finish()

        XCTAssertEqual(try events(lines()), [
            Event(stage: "denoising", step: 0, total: 4),
            Event(stage: "denoising", step: 1, total: 4),
            Event(stage: "denoising", step: 2, total: 4),
            Event(stage: "denoising", step: 3, total: 4),
            Event(stage: "denoising", step: 4, total: 4),
        ])
    }

    func testLoadingStagesThatOnlyReportStepZeroCloseWhenTheNextStageBegins() throws {
        // Wan2 reports `encodingText`/`loadingTransformer` as step 0 of `steps`
        // and never again; `decoding` arrives already terminal.
        let (stream, lines) = makeStream()
        stream.report(stage: "encodingText", step: 0, totalSteps: 3)
        stream.report(stage: "loadingTransformer", step: 0, totalSteps: 3)
        for step in 0..<3 {
            stream.report(stage: "denoising", step: step, totalSteps: 3)
        }
        stream.report(stage: "decoding", step: 3, totalSteps: 3)
        stream.finish()

        XCTAssertEqual(try events(lines()), [
            Event(stage: "encodingText", step: 0, total: 3),
            Event(stage: "encodingText", step: 3, total: 3),
            Event(stage: "loadingTransformer", step: 0, total: 3),
            Event(stage: "loadingTransformer", step: 3, total: 3),
            Event(stage: "denoising", step: 0, total: 3),
            Event(stage: "denoising", step: 1, total: 3),
            Event(stage: "denoising", step: 2, total: 3),
            Event(stage: "denoising", step: 3, total: 3),
            Event(stage: "decoding", step: 3, total: 3),
        ])
    }

    func testSlidingWindowsRestartDenoisingAndCloseTheWindowMilestoneAtTheEnd() throws {
        // MiniMax-H3 emits the window index before each window's 0..<N denoising steps.
        let (stream, lines) = makeStream()
        for window in 0..<2 {
            stream.mark(stage: "window", step: window, totalSteps: 2)
            for step in 0..<2 {
                stream.report(stage: "denoising", step: step, totalSteps: 2)
            }
        }
        stream.finish()

        XCTAssertEqual(try events(lines()), [
            Event(stage: "window", step: 0, total: 2),
            Event(stage: "denoising", step: 0, total: 2),
            Event(stage: "denoising", step: 1, total: 2),
            Event(stage: "window", step: 1, total: 2),
            Event(stage: "denoising", step: 2, total: 2),
            Event(stage: "denoising", step: 0, total: 2),
            Event(stage: "denoising", step: 1, total: 2),
            Event(stage: "denoising", step: 2, total: 2),
            Event(stage: "window", step: 2, total: 2),
        ])
    }

    func testIndeterminateStagesCarryNoTerminalEventAndRepeatsAreDeduplicated() throws {
        let (stream, lines) = makeStream()
        stream.report(stage: "loadingModel", step: 0, totalSteps: 0)
        stream.report(stage: "generating", step: 25, totalSteps: 0)
        stream.report(stage: "generating", step: 25, totalSteps: 0)
        stream.report(stage: "generating", step: 50, totalSteps: 0)
        stream.finish()

        XCTAssertEqual(try events(lines()), [
            Event(stage: "loadingModel", step: 0, total: 0),
            Event(stage: "generating", step: 25, total: 0),
            Event(stage: "generating", step: 50, total: 0),
        ])
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
