import AppKit
@testable import StudioKit
@testable import StudioUI
import XCTest

@MainActor
final class StudioMenuBarTests: XCTestCase {
    func testStateLineNamesThePhaseAndTheAddress() {
        XCTAssertEqual(
            StudioMenuBarCopy.stateLine(phase: .running, endpoint: "http://127.0.0.1:8080"),
            "Running · 127.0.0.1:8080"
        )
        XCTAssertEqual(
            StudioMenuBarCopy.stateLine(phase: .external, endpoint: "http://127.0.0.1:8080"),
            "Running outside Studio · 127.0.0.1:8080"
        )
        XCTAssertEqual(
            StudioMenuBarCopy.stateLine(phase: .failed("x"), endpoint: "http://0.0.0.0:9000"),
            "Stopped unexpectedly · 0.0.0.0:9000"
        )
    }

    func testDetailSaysWhyWhoOrWhatAndNothingWhenThereIsNothingToSay() throws {
        let runtime = try JSONDecoder().decode(StudioRuntimeSnapshot.self, from: Data("""
        {
          "admission": {
            "maxActiveRequests": 4, "activeRequests": 2, "queuedRequests": 1,
            "totalAdmittedRequests": 3, "totalCompletedRequests": 0, "totalCancelledRequests": 0
          },
          "process": { "uptimeSeconds": 9000 }
        }
        """.utf8))

        XCTAssertEqual(
            StudioMenuBarCopy.detail(phase: .failed("Error: port in use"), safety: .loopback, runtime: nil, connection: nil),
            "Error: port in use"
        )
        XCTAssertEqual(
            StudioMenuBarCopy.detail(phase: .running, safety: .loopback, runtime: runtime, connection: nil),
            "2 active · 1 queued · up 2h 30m"
        )
        XCTAssertEqual(
            StudioMenuBarCopy.detail(phase: .external, safety: .loopback, runtime: nil, connection: nil),
            "Stop it where it was started."
        )
        // A server that answers but rejects the key is up; the panel says what it answered.
        XCTAssertEqual(
            StudioMenuBarCopy.detail(
                phase: .running,
                safety: .loopback,
                runtime: nil,
                connection: "Authentication failed — check the API key"
            ),
            "Authentication failed — check the API key"
        )
        XCTAssertEqual(
            StudioMenuBarCopy.detail(phase: .stopped, safety: .exposedWithoutAuthentication, runtime: nil, connection: nil),
            "Add an API key in Settings before serving beyond this Mac."
        )
        XCTAssertNil(StudioMenuBarCopy.detail(phase: .stopped, safety: .loopback, runtime: nil, connection: nil))
        XCTAssertNil(StudioMenuBarCopy.detail(phase: .starting, safety: .loopback, runtime: nil, connection: nil))
    }

    func testUptimeKeepsTheTwoLargestUnits() {
        XCTAssertEqual(StudioMenuBarCopy.uptime(20), "1m")
        XCTAssertEqual(StudioMenuBarCopy.uptime(45 * 60), "45m")
        XCTAssertEqual(StudioMenuBarCopy.uptime(2 * 3_600), "2h")
        XCTAssertEqual(StudioMenuBarCopy.uptime(8_100), "2h 15m")
        XCTAssertEqual(StudioMenuBarCopy.uptime(3 * 86_400 + 4 * 3_600 + 59 * 60), "3d 4h")
    }

    func testResourceCopy() {
        XCTAssertEqual(StudioMenuBarCopy.percent(0.234), "23%")
        XCTAssertEqual(StudioMenuBarCopy.percent(1.4), "100%")
        XCTAssertEqual(StudioMenuBarCopy.rate(nil), "—")
        XCTAssertEqual(StudioMenuBarCopy.rate(0.2), "Idle")
        XCTAssertEqual(StudioMenuBarCopy.rate(41.6), "42 tok/s")
        XCTAssertEqual(StudioMenuBarCopy.memoryUsage(used: 44_023_414_784, total: 68_719_476_736), "41.0 of 64 GB")
        XCTAssertNil(StudioMenuBarCopy.thermal(.nominal))
        XCTAssertEqual(StudioMenuBarCopy.thermal(.serious), "Thermal: Serious")
        XCTAssertEqual(StudioMenuBarCopy.residentCount(3), "3 resident")
    }

    func testTheDockIconHidesOnlyWithTheMenuBarToComeBackFrom() throws {
        XCTAssertTrue(StudioMenuBar.hidesDockIcon(showsMenuBarExtra: true, hidesWithoutWindows: true, hasOpenWindow: false))
        XCTAssertFalse(StudioMenuBar.hidesDockIcon(showsMenuBarExtra: true, hidesWithoutWindows: true, hasOpenWindow: true))
        XCTAssertFalse(StudioMenuBar.hidesDockIcon(showsMenuBarExtra: false, hidesWithoutWindows: true, hasOpenWindow: false))
        XCTAssertFalse(StudioMenuBar.hidesDockIcon(showsMenuBarExtra: true, hidesWithoutWindows: false, hasOpenWindow: false))

        let defaults = try XCTUnwrap(UserDefaults(suiteName: "StudioMenuBarTests-\(UUID().uuidString)"))
        XCTAssertTrue(StudioMenuBar.isOn("unset", defaults: defaults), "an unset preference is on")
        defaults.set(false, forKey: "off")
        XCTAssertFalse(StudioMenuBar.isOn("off", defaults: defaults))
    }

    func testSparklinePutsTheNewestReadingAtTheRightEdge() {
        let size = CGSize(width: 118, height: 28)
        let points = StudioSparkline.points([0, 0.5, 1], maximum: 1, in: size)
        XCTAssertEqual(points.count, 3)
        XCTAssertEqual(points.last?.x, 118)
        XCTAssertEqual(points.last?.y, 2, "a full reading reaches the top margin")
        XCTAssertEqual(points.first?.y, 26, "an empty reading sits on the bottom margin")
        XCTAssertEqual(points[1].x - points[0].x, 2, "one slot per reading of a full two-minute history")
    }

    func testIconIsATemplateWhosePeriodLightsWhileServing() throws {
        _ = MereRunTheme.Brand.register()
        let serving = StudioMenuBarIcon.image(isServing: true)
        let idle = StudioMenuBarIcon.image(isServing: false)

        XCTAssertTrue(serving.isTemplate)
        XCTAssertTrue(idle.isTemplate)
        XCTAssertEqual(serving.size, NSSize(width: 22, height: 16))
        XCTAssertGreaterThan(try inkCoverage(serving), try inkCoverage(idle), "the filled period adds ink")
    }

    private func inkCoverage(_ image: NSImage) throws -> Int {
        let rep = try XCTUnwrap(NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: 44, pixelsHigh: 32, bitsPerSample: 8, samplesPerPixel: 4,
            hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
        ))
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        image.draw(in: NSRect(x: 0, y: 0, width: 44, height: 32))
        NSGraphicsContext.restoreGraphicsState()
        var inked = 0
        for x in 0..<44 {
            for y in 0..<32 where (rep.colorAt(x: x, y: y)?.alphaComponent ?? 0) > 0.5 {
                inked += 1
            }
        }
        return inked
    }
}
