import AppKit
@testable import StudioKit
@testable import StudioUI
import XCTest

@MainActor
final class StudioSidebarTests: XCTestCase {
    // MARK: - Machine status

    func testMachineStatusResolvesEveryProbeOutcome() {
        XCTAssertEqual(StudioMachineStatus(serverStatus: nil, probeTimedOut: false), .checking)
        XCTAssertEqual(StudioMachineStatus(serverStatus: nil, probeTimedOut: true), .unreachable)

        let idle = StudioServerStatus(health: "down", loadedModels: [], installedCount: 92)
        XCTAssertEqual(
            StudioMachineStatus(serverStatus: idle, probeTimedOut: false),
            .ready(installedModels: 92)
        )
        // A late answer wins over the grace-period fallback.
        XCTAssertEqual(
            StudioMachineStatus(serverStatus: idle, probeTimedOut: true),
            .ready(installedModels: 92)
        )

        let serving = StudioServerStatus(health: "ok", loadedModels: ["gemma4-e4b"], installedCount: 3)
        XCTAssertEqual(
            StudioMachineStatus(serverStatus: serving, probeTimedOut: false),
            .serving(installedModels: 3, loadedModel: "gemma4-e4b")
        )
    }

    func testMachineStatusCopy() {
        XCTAssertEqual(StudioMachineStatus.checking.summary, "Checking…")
        XCTAssertEqual(StudioMachineStatus.unreachable.summary, "Server unreachable")
        XCTAssertEqual(StudioMachineStatus.ready(installedModels: 92).summary, "Ready · 92 models")
        XCTAssertEqual(StudioMachineStatus.ready(installedModels: 1).summary, "Ready · 1 model")
        XCTAssertEqual(
            StudioMachineStatus.serving(installedModels: 3, loadedModel: "gemma4-e4b").summary,
            "Serving · 3 models"
        )

        XCTAssertEqual(
            StudioMachineStatus.serving(installedModels: 3, loadedModel: "gemma4-e4b").serverDetail,
            "Up · gemma4-e4b"
        )
        XCTAssertEqual(StudioMachineStatus.serving(installedModels: 3, loadedModel: nil).serverDetail, "Up")
        XCTAssertEqual(StudioMachineStatus.ready(installedModels: 2).serverDetail, "Not running — starts on demand")
        XCTAssertTrue(StudioMachineStatus.unreachable.serverDetail.contains("did not answer"))
        XCTAssertEqual(StudioMachineStatus.unreachable.modelsDetail, "—")
        XCTAssertEqual(StudioMachineStatus.ready(installedModels: 2).modelsDetail, "2 installed")
    }

    func testCheckingGracePeriodOutlastsTheProbeTimeout() {
        // The controller's status probe gives the CLI about a second; the footer must wait longer
        // than one probe before calling the server unreachable, and not so long that it feels stuck.
        XCTAssertGreaterThan(StudioMachineStatus.checkingGracePeriod, 2)
        XCTAssertLessThanOrEqual(StudioMachineStatus.checkingGracePeriod, 10)
    }

    // MARK: - Wordmark font

    func testBundledWordmarkFontIsFoundAndRegisters() throws {
        let url = try XCTUnwrap(MereRunTheme.Brand.fontURL(), "Caveat[wght].ttf is not in the app resource bundle")
        XCTAssertEqual(url.lastPathComponent, "Caveat[wght].ttf")
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.deletingLastPathComponent().appendingPathComponent("OFL.txt").path),
                      "the OFL license ships beside the font")

        XCTAssertTrue(MereRunTheme.Brand.register())
        XCTAssertTrue(MereRunTheme.Brand.isAvailable)
        let font = try XCTUnwrap(NSFont(name: MereRunTheme.Brand.familyName, size: MereRunTheme.Brand.wordmarkSize))
        XCTAssertEqual(font.familyName, "Caveat")
    }
}
