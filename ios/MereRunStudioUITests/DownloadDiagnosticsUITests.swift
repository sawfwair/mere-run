import XCTest

/// Diagnostic: drive an on-device model download on real hardware and record
/// what the UI reports. Runs only when MERERUN_TEST_DOWNLOAD_DIAG is set —
/// this is an investigation tool, not a CI gate.
final class DownloadDiagnosticsUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = true
    }

    func testObserveLocalModelDownload() throws {
        guard ProcessInfo.processInfo.environment["MERERUN_TEST_DOWNLOAD_DIAG"] != nil else {
            throw XCTSkip("Set MERERUN_TEST_DOWNLOAD_DIAG to run the download diagnostic.")
        }

        let app = XCUIApplication()
        app.launch()

        let createTab = app.tabBars.buttons["Create"]
        XCTAssertTrue(createTab.waitForExistence(timeout: 20), "App must be paired before this diagnostic.")
        createTab.tap()

        let phoneSegment = app.buttons["This iPhone"]
        if phoneSegment.waitForExistence(timeout: 5) {
            phoneSegment.tap()
        } else {
            app.segmentedControls.buttons.element(boundBy: 1).tap()
        }
        snapshot(app, "lane")

        // Tap the first available "Get …" button (Klein nano is smallest).
        let get = app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Get '")).firstMatch
        XCTAssertTrue(get.waitForExistence(timeout: 10), "A download button must be visible.")
        let tappedLabel = get.label
        get.tap()

        // Observe for two minutes, logging every visible static text in the
        // lane so progress, stalls, and error banners are all captured.
        var observations: [String] = ["tapped: \(tappedLabel)"]
        let deadline = Date().addingTimeInterval(120)
        var tick = 0
        while Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(10))
            tick += 1
            let labels = app.scrollViews.staticTexts.allElementsBoundByIndex
                .map { $0.label }
                .filter { $0.contains("%") || $0.contains("MB") || $0.contains("…")
                    || $0.localizedCaseInsensitiveContains("error")
                    || $0.localizedCaseInsensitiveContains("fail")
                    || $0.localizedCaseInsensitiveContains("could not")
                    || $0.localizedCaseInsensitiveContains("download") }
            observations.append("t+\(tick * 10)s: \(labels.joined(separator: " | "))")
            if tick % 3 == 0 { snapshot(app, "t\(tick * 10)") }
        }
        snapshot(app, "final")

        let report = XCTAttachment(string: observations.joined(separator: "\n"))
        report.name = "download-observations"
        report.lifetime = .keepAlways
        add(report)
    }

    private func snapshot(_ app: XCUIApplication, _ name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
