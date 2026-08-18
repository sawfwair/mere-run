import XCTest

/// Screenshot harness for design iteration: launches the app in UI-preview
/// mode and captures the on-device model surfaces. Gated on
/// MERERUN_TEST_UX_SHOTS so ordinary test runs skip it.
final class UXShotsUITests: XCTestCase {
    func testCaptureOnDeviceSurfaces() throws {
        guard ProcessInfo.processInfo.environment["MERERUN_TEST_UX_SHOTS"] != nil else {
            throw XCTSkip("Set MERERUN_TEST_UX_SHOTS to capture UX screenshots.")
        }
        let app = XCUIApplication()
        app.launchArguments = ["MERERUN_UI_PREVIEW"]
        app.launch()

        XCTAssertTrue(app.tabBars.buttons["Create"].waitForExistence(timeout: 15))
        app.tabBars.buttons["Create"].tap()
        shot(app, "create")

        app.buttons["create.model"].tap()
        sleep(1)
        shot(app, "create-model-menu")

        if app.buttons["Get on-device models…"].waitForExistence(timeout: 3) {
            app.buttons["Get on-device models…"].tap()
            sleep(1)
            shot(app, "ondevice-sheet")
            app.buttons["Done"].tap()
        }

        app.tabBars.buttons["Chat"].tap()
        sleep(1)
        let chip = app.buttons.matching(NSPredicate(format: "label CONTAINS 'Fleet default'")).firstMatch
        if chip.waitForExistence(timeout: 3) {
            chip.tap()
            sleep(1)
            shot(app, "chat-model-menu")
        }
    }

    private func shot(_ app: XCUIApplication, _ name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
