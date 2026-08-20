import AppKit
@testable import MereRunApp
import XCTest

@MainActor
final class MereRunThemeTests: XCTestCase {
    func testNSColorHexParsesSRGBComponents() throws {
        let color = try XCTUnwrap(NSColor(hex: "C9A65D").usingColorSpace(.sRGB))
        XCTAssertEqual(color.redComponent, CGFloat(0xC9) / 255, accuracy: 0.001)
        XCTAssertEqual(color.greenComponent, CGFloat(0xA6) / 255, accuracy: 0.001)
        XCTAssertEqual(color.blueComponent, CGFloat(0x5D) / 255, accuracy: 0.001)
    }

    func testDynamicNSColorResolvesPerAppearance() throws {
        // The exact mechanism the theme uses: a name-less dynamic provider keyed on darkAqua.
        let dynamic = NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
                ? NSColor(hex: "000000")
                : NSColor(hex: "FFFFFF")
        }
        let light = try XCTUnwrap(NSAppearance(named: .aqua))
        let dark = try XCTUnwrap(NSAppearance(named: .darkAqua))

        var lightValue: CGFloat = -1
        var darkValue: CGFloat = -1
        light.performAsCurrentDrawingAppearance {
            lightValue = dynamic.usingColorSpace(.sRGB)?.redComponent ?? -1
        }
        dark.performAsCurrentDrawingAppearance {
            darkValue = dynamic.usingColorSpace(.sRGB)?.redComponent ?? -1
        }

        XCTAssertEqual(lightValue, 1.0, accuracy: 0.01, "light resolves to white")
        XCTAssertEqual(darkValue, 0.0, accuracy: 0.01, "dark resolves to black")
    }
}
