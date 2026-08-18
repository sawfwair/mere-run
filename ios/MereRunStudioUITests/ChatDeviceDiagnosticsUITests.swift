import XCTest

/// Diagnostic: drive on-device Bonsai Chat on real hardware and record how
/// far it gets — model load, first token, or process death. Gated on
/// MERERUN_TEST_CHAT_DIAG.
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

        // Select the on-device model from the chip menu.
        let chip = app.buttons.matching(
            NSPredicate(format: "label CONTAINS 'Fleet default' OR label CONTAINS 'this iPhone' OR label CONTAINS 'Bonsai Chat'")
        ).firstMatch
        XCTAssertTrue(chip.waitForExistence(timeout: 10), "The model chip must exist.")
        chip.tap()
        let localButton = app.buttons["Bonsai Chat"]
        XCTAssertTrue(localButton.waitForExistence(timeout: 5), "Bonsai Chat must be selectable (installed).")
        localButton.tap()

        let composer = app.textViews["chat.composer"].exists
            ? app.textViews["chat.composer"]
            : app.textFields["chat.composer"]
        XCTAssertTrue(composer.waitForExistence(timeout: 5))
        composer.tap()
        composer.typeText("Say hi in three words.")
        app.buttons["Send"].tap()

        var observations: [String] = []
        let deadline = Date().addingTimeInterval(420)
        var outcome = "timeout"
        while Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(5))
            if app.state != .runningForeground {
                outcome = "app died (state \(app.state.rawValue))"
                break
            }
            let labels = app.staticTexts.allElementsBoundByIndex.map { $0.label }
            if labels.contains(where: { $0.localizedCaseInsensitiveContains("needs a physical iPhone")
                || $0.localizedCaseInsensitiveContains("error")
                || $0.localizedCaseInsensitiveContains("busy") }) {
                outcome = "error text: \(labels.filter { $0.count > 8 }.suffix(3))"
                break
            }
            let hasReply = labels.contains { $0.count > 2 && $0 != "Say hi in three words."
                && !$0.hasPrefix("Thinking on") && !$0.hasPrefix("Chat")
                && $0.rangeOfCharacter(from: .letters) != nil
                && !$0.contains("Bonsai") && !$0.contains("New chat") }
            if hasReply && !app.progressIndicators.firstMatch.exists {
                outcome = "replied"
                break
            }
            observations.append("t: alive, texts=\(labels.count)")
        }
        let report = XCTAttachment(string: ([outcome] + observations.suffix(10)).joined(separator: "\n"))
        report.name = "chat-outcome"
        report.lifetime = .keepAlways
        add(report)
        let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        attachment.name = "final"
        attachment.lifetime = .keepAlways
        add(attachment)
        XCTAssertEqual(outcome, "replied", "On-device chat did not reply: \(outcome)")
    }
}
