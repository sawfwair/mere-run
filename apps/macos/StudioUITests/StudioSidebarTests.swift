import AppKit
@testable import StudioKit
@testable import StudioUI
import XCTest

@MainActor
final class StudioSidebarTests: XCTestCase {
    // MARK: - Machine status

    func testMachineStatusResolvesEveryProbeOutcome() {
        XCTAssertEqual(StudioMachineStatus(serverStatus: nil, probeTimedOut: false, isServing: false), .checking)
        XCTAssertEqual(StudioMachineStatus(serverStatus: nil, probeTimedOut: true, isServing: false), .cliNotResponding)

        let idle = StudioServerStatus(health: "down", loadedModels: [], installedCount: 92)
        XCTAssertEqual(
            StudioMachineStatus(serverStatus: idle, probeTimedOut: false, isServing: false),
            .ready(installedModels: 92)
        )
        // A late answer wins over the grace-period fallback.
        XCTAssertEqual(
            StudioMachineStatus(serverStatus: idle, probeTimedOut: true, isServing: false),
            .ready(installedModels: 92)
        )

        let serving = StudioServerStatus(health: "ok", loadedModels: ["gemma4-e4b"], installedCount: 3)
        XCTAssertEqual(
            StudioMachineStatus(serverStatus: serving, probeTimedOut: false, isServing: true, loadedModel: "gemma4-e4b"),
            .serving(installedModels: 3, loadedModel: "gemma4-e4b")
        )
    }

    func testMachineStatusTakesServingFromTheServerMonitorNotTheSlowerStatusPoll() {
        // The 20-second `status --json` poll can lag the 2-second endpoint monitor either way; the
        // footer follows the monitor, as the Server page and the menu bar do.
        let staleUp = StudioServerStatus(health: "ok", loadedModels: ["gemma4-e4b"], installedCount: 3)
        XCTAssertEqual(
            StudioMachineStatus(serverStatus: staleUp, probeTimedOut: false, isServing: false),
            .ready(installedModels: 3)
        )
        let staleDown = StudioServerStatus(health: "down", loadedModels: [], installedCount: 3)
        XCTAssertEqual(
            StudioMachineStatus(serverStatus: staleDown, probeTimedOut: false, isServing: true),
            .serving(installedModels: 3, loadedModel: nil)
        )
    }

    func testMachineStatusCopy() {
        XCTAssertEqual(StudioMachineStatus.checking.summary, "Checking…")
        XCTAssertEqual(StudioMachineStatus.cliNotResponding.summary, "CLI not responding")
        XCTAssertEqual(StudioMachineStatus.ready(installedModels: 92).summary, "Ready · 92 models")
        XCTAssertEqual(StudioMachineStatus.ready(installedModels: 1).summary, "Ready · 1 model")
        XCTAssertEqual(
            StudioMachineStatus.serving(installedModels: 3, loadedModel: "gemma4-e4b").summary,
            "Serving"
        )

        XCTAssertEqual(
            StudioMachineStatus.serving(installedModels: 3, loadedModel: "gemma4-e4b").serverDetail,
            "Up · gemma4-e4b"
        )
        XCTAssertEqual(StudioMachineStatus.serving(installedModels: 3, loadedModel: nil).serverDetail, "Up")
        XCTAssertEqual(StudioMachineStatus.ready(installedModels: 2).serverDetail, "Not running — starts on demand")
        XCTAssertTrue(StudioMachineStatus.cliNotResponding.serverDetail.contains("did not answer"))
        XCTAssertEqual(StudioMachineStatus.cliNotResponding.modelsDetail, "—")
        XCTAssertEqual(StudioMachineStatus.ready(installedModels: 2).modelsDetail, "2 installed")
    }

    func testSkippedModelLocationsTurnTheDotYellowAndExplainThePrompt() {
        let stalled = StudioSkippedModelLocation(path: "/Volumes/MODELS", problem: .unresponsive)
        let denied = StudioSkippedModelLocation(path: "/Volumes/Home/models", problem: .denied)
        let status = StudioServerStatus(
            health: "down", loadedModels: [], installedCount: 90, skippedLocations: [stalled, denied]
        )
        let machine = StudioMachineStatus(serverStatus: status, probeTimedOut: false, isServing: false)

        XCTAssertEqual(machine, .ready(installedModels: 90, skippedLocations: [stalled, denied]))
        XCTAssertEqual(machine.summary, "Ready · 90 models")
        XCTAssertEqual(machine.dotColor, MereRunTheme.yellow)
        XCTAssertEqual(machine.locationNotice?.title, "Model drive not responding")
        XCTAssertEqual(
            machine.locationNotice?.detail,
            "/Volumes/MODELS and 1 more did not answer. macOS may be asking to allow access to it."
        )

        let deniedOnly = StudioMachineStatus.serving(installedModels: 3, loadedModel: nil, skippedLocations: [denied])
        XCTAssertEqual(deniedOnly.locationNotice?.title, "Model drive access denied")
        XCTAssertEqual(
            deniedOnly.locationNotice?.detail,
            "macOS denied access to /Volumes/Home/models. Allow MereRun in Files & Folders."
        )

        XCTAssertNil(StudioMachineStatus.ready(installedModels: 90).locationNotice)
        XCTAssertEqual(StudioMachineStatus.ready(installedModels: 90).dotColor, MereRunTheme.green)
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
