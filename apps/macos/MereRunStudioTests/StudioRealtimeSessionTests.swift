@testable import MereRunApp
import Foundation
import XCTest

final class StudioRealtimeSessionTests: XCTestCase {
    // MARK: Transport clock

    func testTimestampUsesTwoDigitMinutes() {
        XCTAssertEqual(StudioRealtimeTransport.timestamp(0), "00:00")
        XCTAssertEqual(StudioRealtimeTransport.timestamp(252.9), "04:12")
        XCTAssertEqual(StudioRealtimeTransport.timestamp(3_601), "60:01")
        XCTAssertEqual(StudioRealtimeTransport.timestamp(-4), "00:00")
        XCTAssertEqual(StudioRealtimeTransport.timestamp(.nan), "00:00")
    }

    func testRenderedSecondsReadsRealtimeFrameProgress() throws {
        let progress = StudioProgressParser.parse("Realtime frame 6353/7500")
        XCTAssertEqual(progress?.label, "Realtime music")
        XCTAssertEqual(
            try XCTUnwrap(StudioRealtimeTransport.renderedSeconds(from: progress)),
            254.12,
            accuracy: 0.001
        )

        XCTAssertNil(StudioRealtimeTransport.renderedSeconds(from: nil))
        XCTAssertNil(
            StudioRealtimeTransport.renderedSeconds(
                from: StudioRunProgress(label: "Training", fractionCompleted: 0.5, detail: "Step 2 of 4")
            )
        )
    }

    func testStatusLineReportsLeadOverPlayback() {
        XCTAssertEqual(
            StudioRealtimeTransport.statusLine(wallElapsed: 252, rendered: 254.12),
            "04:12 · 2.1 s ahead of playback"
        )
        // Playback can never run ahead of the rendered audio, so the lead floors at zero.
        XCTAssertEqual(
            StudioRealtimeTransport.statusLine(wallElapsed: 300, rendered: 254.12),
            "04:14 · 0.0 s ahead of playback"
        )
        XCTAssertEqual(
            StudioRealtimeTransport.statusLine(wallElapsed: 7, rendered: nil),
            "00:07 · loading model"
        )
    }

    func testEngineLabelNamesMagentaRT2WithSampleRate() {
        XCTAssertEqual(StudioRealtimeTransport.engineLabel(model: "music-magenta-rt2-medium"), "Magenta RT2 · 48 kHz")
        XCTAssertEqual(StudioRealtimeTransport.engineLabel(model: ""), "Magenta RT2 · 48 kHz")
        XCTAssertEqual(StudioRealtimeTransport.engineLabel(model: "/models/custom-rt"), "/models/custom-rt")
    }

    // MARK: Blend

    func testBlendedPromptPicksEndpointsAndOrdersTheMiddle() {
        let promptA = "slow-burn synthwave, hopeful bridge"
        let promptB = "brushed drums, dusty piano"
        XCTAssertEqual(StudioRealtimeSteering.blendedPrompt(a: promptA, b: promptB, blend: 0), promptA)
        XCTAssertEqual(StudioRealtimeSteering.blendedPrompt(a: promptA, b: promptB, blend: 1), promptB)
        XCTAssertEqual(
            StudioRealtimeSteering.blendedPrompt(a: promptA, b: promptB, blend: 0.35),
            "\(promptA), \(promptB)"
        )
        XCTAssertEqual(
            StudioRealtimeSteering.blendedPrompt(a: promptA, b: promptB, blend: 0.8),
            "\(promptB), \(promptA)"
        )
    }

    func testBlendedPromptIgnoresAnEmptySide() {
        XCTAssertEqual(StudioRealtimeSteering.blendedPrompt(a: "  ambient  ", b: "", blend: 0.9), "ambient")
        XCTAssertEqual(StudioRealtimeSteering.blendedPrompt(a: "", b: "piano", blend: 0), "piano")
        XCTAssertEqual(StudioRealtimeSteering.blendedPrompt(a: "", b: "", blend: 0.5), "")
    }

    func testFormatBlendAndControlValues() {
        XCTAssertEqual(StudioRealtimeSteering.formatBlend(0.35), "0.35")
        XCTAssertEqual(StudioRealtimeSteering.formatBlend(1.4), "1.00")
        XCTAssertEqual(StudioRealtimeSteering.format(1.1), "1.1")
        XCTAssertEqual(StudioRealtimeSteering.format(4.0), "4")
        XCTAssertEqual(StudioRealtimeSteering.format(0.25), "0.25")
    }

    func testControlCommandsSpeakTheCLIProtocol() {
        XCTAssertEqual(
            StudioRealtimeSteering.controlCommands(
                temperature: 1.1,
                topK: 40,
                guidance: 4,
                noteGuidance: 5,
                drumGuidance: 1.5,
                style: "full",
                drumless: true
            ),
            ["style full", "temp 1.1", "topk 40", "mc 4", "notes 5", "drums 1.5", "drumless on"]
        )
    }

    // MARK: Session log

    func testLogLinesAreStampedWithTheSessionClock() {
        // LogLine stamps itself at creation; the session started 248.5 s before these lines.
        let startedAt = Date().addingTimeInterval(-248.5)
        let logs = [
            LogLine(stream: .stderr, text: "Realtime frame 6326/7500"),
            LogLine(stream: .system, text: "Live control → prompt brushed drums")
        ]
        XCTAssertEqual(
            StudioRealtimeSessionLog.lines(logs, startedAt: startedAt),
            ["04:08  Realtime frame 6326/7500", "04:08  Live control → prompt brushed drums"]
        )

        XCTAssertEqual(
            StudioRealtimeSessionLog.copyText(["04:08  a", "04:10  b"]),
            "04:08  a\n04:10  b"
        )
    }

    func testVisibleTailKeepsTheNewestLinesThatFit() {
        let lines = (1...10).map { "line \($0)" }
        XCTAssertEqual(
            StudioRealtimeSessionLog.visibleTail(lines, height: 120, lineHeight: 17.8),
            ["line 5", "line 6", "line 7", "line 8", "line 9", "line 10"]
        )
        XCTAssertEqual(StudioRealtimeSessionLog.visibleTail(lines, height: 10, lineHeight: 17.8), ["line 10"])
        XCTAssertEqual(StudioRealtimeSessionLog.visibleTail(["only"], height: 120, lineHeight: 17.8), ["only"])
    }

    // MARK: Job bar

    func testJobBarDetailNamesTheResidentModel() {
        XCTAssertEqual(StudioRealtimeJobBar.residentName(model: "music-magenta-rt2-medium"), "rt2-medium")
        XCTAssertEqual(StudioRealtimeJobBar.residentName(model: ""), "rt2-small")
        XCTAssertEqual(StudioRealtimeJobBar.residentName(model: "custom"), "custom")

        var components = DateComponents()
        components.year = 2026
        components.month = 9
        components.day = 3
        components.hour = 13
        components.minute = 27
        let startedAt = Calendar.current.date(from: components)!

        XCTAssertEqual(
            StudioRealtimeJobBar.detail(phase: .live, startedAt: startedAt, model: "music-magenta-rt2-medium"),
            "Session running · started 1:27 PM · rt2-medium resident"
        )
        XCTAssertEqual(
            StudioRealtimeJobBar.detail(phase: .idle, startedAt: nil, model: "music-magenta-rt2-small"),
            "No session · press play to start · rt2-small"
        )
        XCTAssertEqual(
            StudioRealtimeJobBar.detail(phase: .queued, startedAt: nil, model: "music-magenta-rt2-small"),
            "Queued behind the active job · rt2-small"
        )
        XCTAssertEqual(
            StudioRealtimeJobBar.detail(
                phase: .ended(exitCode: 0),
                startedAt: startedAt,
                model: "music-magenta-rt2-small",
                now: startedAt.addingTimeInterval(298)
            ),
            "Session ended · ran 04:58 · rt2-small"
        )
        XCTAssertEqual(
            StudioRealtimeJobBar.detail(
                phase: .ended(exitCode: 64),
                startedAt: startedAt,
                model: "music-magenta-rt2-small",
                now: startedAt.addingTimeInterval(3)
            ),
            "Session failed · exit 64 · ran 00:03"
        )
    }
}
