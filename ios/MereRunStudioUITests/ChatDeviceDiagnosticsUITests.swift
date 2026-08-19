import XCTest

/// Diagnostic: drive an on-device chat turn on real hardware and record the
/// streaming pipeline's behavior — telemetry line, partial bubble, reply.
/// Gated on MERERUN_TEST_CHAT_DIAG.
final class ChatDeviceDiagnosticsUITests: XCTestCase {
    func testOnDeviceChatTurn() throws {
        guard ProcessInfo.processInfo.environment["MERERUN_TEST_CHAT_DIAG"] != nil else {
            throw XCTSkip("Set MERERUN_TEST_CHAT_DIAG to run the on-device chat diagnostic.")
        }
        let app = XCUIApplication()
        app.launch()

        let chatTab = app.tabBars.buttons["Chat"]
        XCTAssertTrue(chatTab.waitForExistence(timeout: 20))
        chatTab.tap()

        // Use the persisted on-device lane; only open the menu if needed.
        let chip = app.buttons.matching(
            NSPredicate(format: "label CONTAINS 'this iPhone'")
        ).firstMatch
        if !chip.waitForExistence(timeout: 5) {
            let menuChip = app.buttons.matching(
                NSPredicate(format: "label CONTAINS 'Fleet default'")
            ).firstMatch
            XCTAssertTrue(menuChip.waitForExistence(timeout: 5), "A model chip must exist.")
            menuChip.tap()
            let liquid = app.buttons["Liquid Chat"]
            XCTAssertTrue(liquid.waitForExistence(timeout: 5), "Liquid Chat must be installed for this diagnostic.")
            liquid.tap()
        }

        // A persisted thread makes old bubbles look like replies; start clean.
        let newChat = app.buttons["New chat"]
        if newChat.waitForExistence(timeout: 3), newChat.isEnabled {
            newChat.tap()
        }

        let prompt = "Write a six line poem about rain, then count from 1 to 40 one number per line."
        let composer = app.textViews["chat.composer"].exists
            ? app.textViews["chat.composer"]
            : app.textFields["chat.composer"]
        XCTAssertTrue(composer.waitForExistence(timeout: 5))
        composer.tap()
        composer.typeText(prompt)
        app.buttons["Send"].tap()

        var observations: [String] = []
        var sawPartial = false
        var outcome = "timeout"
        let deadline = Date().addingTimeInterval(300)
        while Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.7))
            let debug = app.staticTexts["chat.debug"]
            if debug.exists {
                observations.append("debug: \(debug.label.prefix(80))")
            }
            let partial = app.staticTexts["chat.partial"]
            if partial.exists {
                sawPartial = true
                observations.append("PARTIAL: \(partial.label.prefix(50))")
            }
            let finished = !app.progressIndicators.firstMatch.exists
                && !partial.exists
                && app.staticTexts.allElementsBoundByIndex
                    .contains { $0.label.count > 60 && $0.label != prompt }
            if finished {
                outcome = sawPartial ? "replied-streaming" : "replied-without-partial"
                break
            }
        }
        observations.append("sawPartial=\(sawPartial)")
        let report = XCTAttachment(string: ([outcome] + observations.suffix(60)).joined(separator: "\n"))
        report.name = "chat-outcome"
        report.lifetime = .keepAlways
        add(report)
        let screenshot = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        screenshot.name = "final"
        screenshot.lifetime = .keepAlways
        add(screenshot)
        XCTAssertTrue(outcome.hasPrefix("replied"), "On-device chat did not finish: \(outcome)")
    }
}
