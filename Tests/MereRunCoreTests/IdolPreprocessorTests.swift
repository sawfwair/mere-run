import Foundation
import MediaIO
import XCTest
@testable import MereRunCore

final class IdolPreprocessorTests: XCTestCase {
    func testOpaqueInputNeedsExplicitFramingAndCompositesOnBlack() throws {
        let image = try MediaImage(width: 2, height: 2, rgba8: [255, 128, 0, 128, 0, 255, 0, 255,
            255, 0, 255, 0, 0, 0, 255, 255])
        let prepared = try IdolPreprocessor.prepare(image, size: 2, alreadyFramed: true)
        XCTAssertEqual(prepared.rgba8, [128, 64, 0, 255, 0, 255, 0, 255, 0, 0, 0, 255, 0, 0, 255, 255])
        let opaque = try MediaImage(width: 2, height: 2, rgba8: [UInt8](repeating: 255, count: 16))
        XCTAssertThrowsError(try IdolPreprocessor.prepare(opaque, size: 16))
    }

    func testAlphaErosionRejectsEmptyForegroundAndCropModeChangesFraming() throws {
        let empty = try MediaImage(width: 16, height: 16, rgba8: [UInt8](repeating: 0, count: 16 * 16 * 4))
        XCTAssertThrowsError(try IdolPreprocessor.prepare(empty, size: 16))
        var bytes = empty.rgba8
        for y in 4..<12 {
            for x in 4..<12 {
                for channel in 0..<4 { bytes[(y * 16 + x) * 4 + channel] = 255 }
            }
        }
        let object = try MediaImage(width: 16, height: 16, rgba8: bytes)
        let synthetic = try IdolPreprocessor.prepare(object, size: 16, mode: .synthetic)
        let real = try IdolPreprocessor.prepare(object, size: 16, mode: .real)
        XCTAssertEqual(synthetic.width, 16)
        XCTAssertGreaterThan(synthetic.rgba8.enumerated().filter { $0.offset % 4 == 0 }.reduce(0) { $0 + Int($1.element) },
                             real.rgba8.enumerated().filter { $0.offset % 4 == 0 }.reduce(0) { $0 + Int($1.element) })
        XCTAssertThrowsError(try IdolPreprocessor.prepare(object, size: 0))
    }

    func testReleasePinsAndMissingResourceRejection() throws {
        XCTAssertEqual(IdolResources.pins.count, 4)
        XCTAssertEqual(IdolResources.pins.map(\.byteCount).reduce(0, +), 3078979896)
        let absent = URL(fileURLWithPath: "/tmp/idol-missing-\(UUID().uuidString)")
        let resources = IdolResources(root: absent, dinoRoot: absent, vaeRoot: absent)
        XCTAssertThrowsError(try resources.verify())
    }
}
