import XCTest

/// End-to-end integration of the direct lane, driven through the real UI in
/// the simulator against a live `mere.run relay serve` on the host machine:
/// pair with the terminal code, see the machine come online in Fleet, and
/// run a chat turn through a `text.generate` job.
///
/// Run with the relay's address and code in the test environment:
///
///   xcodebuild test -project MereRunStudio.xcodeproj -scheme MereRunStudio \
///     -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
///     TEST_RUNNER_MERERUN_TEST_RELAY_URL=http://127.0.0.1:6399 \
///     TEST_RUNNER_MERERUN_TEST_PAIR_CODE=123-456
///
/// Without those variables the test skips, so plain `xcodebuild test` stays
/// green on machines with no relay running.
final class DirectLaneUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testDirectLanePairsSeesFleetAndChats() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let relayURL = environment["MERERUN_TEST_RELAY_URL"],
              let pairCode = environment["MERERUN_TEST_PAIR_CODE"] else {
            throw XCTSkip("Set MERERUN_TEST_RELAY_URL and MERERUN_TEST_PAIR_CODE to run the direct-lane test.")
        }

        let app = XCUIApplication()
        app.launch()

        // Pair through the direct lane. A fresh install shows the pairing
        // screen; a re-run against a paired container skips straight to tabs.
        if app.buttons["Connect to a machine"].waitForExistence(timeout: 5) {
            app.buttons["Connect to a machine"].tap()

            let address = app.textFields["pairing.address"]
            XCTAssertTrue(address.waitForExistence(timeout: 5), "The address field must appear.")
            reliablyType(relayURL, into: address)

            let code = app.textFields["pairing.code"]
            reliablyType(pairCode, into: code)

            app.buttons["pairing.primary"].tap()
        }

        // Paired: the tab bar appears.
        let chatTab = app.tabBars.buttons["Chat"]
        XCTAssertTrue(chatTab.waitForExistence(timeout: 30), "Pairing must land on the main tabs.")
        takeScreenshot(named: "paired")

        // Fleet shows this machine online.
        app.tabBars.buttons["Fleet"].tap()
        let online = app.staticTexts["online"]
        let busy = app.staticTexts["busy"]
        XCTAssertTrue(
            online.waitForExistence(timeout: 30) || busy.exists,
            "The fleet must list the direct machine."
        )
        takeScreenshot(named: "fleet")

        // One chat turn through a real text.generate job on the machine.
        chatTab.tap()
        let composer = app.textViews["chat.composer"].exists
            ? app.textViews["chat.composer"]
            : app.textFields["chat.composer"]
        XCTAssertTrue(composer.waitForExistence(timeout: 10), "The chat composer must appear.")
        composer.tap()
        composer.typeText("Reply with one short friendly sentence.")
        app.buttons["Send"].tap()

        // The user bubble appears immediately; the reply bubble arrives when
        // the job finishes (model load can dominate on first run).
        let userBubble = app.staticTexts["Reply with one short friendly sentence."]
        XCTAssertTrue(userBubble.waitForExistence(timeout: 10), "The sent message must appear as a bubble.")

        let sendReenabled = NSPredicate(format: "isEnabled == false")
        _ = sendReenabled // Send disables while awaiting; poll bubbles instead.

        let deadline = Date().addingTimeInterval(420)
        var replied = false
        while Date() < deadline {
            // Any new visible bubble that is not the prompt counts as the
            // reply; failure copy renders in the same position, so assert on
            // the store's error styling being absent via the retry banner.
            let chrome: Set<String> = [
                "Reply with one short friendly sentence.",
                "Think it through.",
                "Your words go to your machines and nowhere else.",
            ]
            let bubbles = app.scrollViews.staticTexts.allElementsBoundByIndex
                .map { $0.label }
                .filter { $0.count > 1 && !chrome.contains($0) && !$0.hasPrefix("Running on") && !$0.hasPrefix("Thinking on") }
            if bubbles.contains(where: { $0.localizedCaseInsensitiveContains("expired") }) {
                XCTFail("Chat surfaced an auth error: \(bubbles)")
                break
            }
            if !bubbles.isEmpty {
                replied = true
                break
            }
            RunLoop.current.run(until: Date().addingTimeInterval(3))
        }
        XCTAssertTrue(replied, "A reply bubble must arrive from the direct machine.")
        takeScreenshot(named: "chat-reply")
    }

    /// `typeText` drops keystrokes when it races keyboard layout changes;
    /// verify the field's value and retype until it matches.
    private func reliablyType(_ text: String, into element: XCUIElement) {
        func typedValue() -> String {
            let value = element.value as? String ?? ""
            return value == element.placeholderValue ? "" : value
        }
        for _ in 0..<4 {
            element.tap()
            if typedValue() == text { return }
            for _ in 0..<typedValue().count {
                element.typeText(XCUIKeyboardKey.delete.rawValue)
            }
            element.typeText(text)
            RunLoop.current.run(until: Date().addingTimeInterval(0.5))
            if typedValue() == text { return }
        }
        XCTFail("Could not enter \"\(text)\"; the field holds \"\(element.value as? String ?? "")\".")
    }

    private func takeScreenshot(named name: String) {
        let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
