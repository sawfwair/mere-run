import Foundation
import MediaIO
import XCTest
@testable import MereRunCore

final class FalconPerceptionResizeParityTests: XCTestCase {
    private struct Fixture: Decodable {
        let cases: [ResizeCase]
    }

    private struct ResizeCase: Decodable {
        let name: String
        let mode: String
        let width: Int
        let height: Int
        let rgba: [UInt8]
        let expectedWidth: Int
        let expectedHeight: Int
        let expectedRGB: [UInt8]
    }

    func testMediaImageResizeMatchesPillowRGBFixtures() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/falcon-bicubic-resize.json")
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        XCTAssertEqual(fixture.cases.count, 8)
        for item in fixture.cases {
            let input = try MediaImage(width: item.width, height: item.height, rgba8: item.rgba)
            let output = item.mode == "initial"
                ? FalconPerceptionProcessor.resizeIfNecessary(input, shortest: 16, longest: 32)
                : FalconPerceptionProcessor.smartResize(input, factor: 4, minPixels: 16, maxPixels: 4096)
            XCTAssertEqual(output.width, item.expectedWidth, item.name)
            XCTAssertEqual(output.height, item.expectedHeight, item.name)
            let rgb = output.rgba8.enumerated().compactMap { index, value in index % 4 < 3 ? value : nil }
            let differences = zip(rgb, item.expectedRGB).filter { $0 != $1 }.count
            XCTAssertEqual(rgb.count, item.expectedRGB.count, item.name)
            XCTAssertEqual(differences, 0, "\(item.name): \(differences) differing RGB bytes")
        }
    }

    private struct FrameDeclaration: Decodable {
        struct Frame: Decodable {
            let sourcePath: String
            let referencePath: String
            let outputPath: String
        }
        let frames: [Frame]
    }

    func testDeclaredFramePixelsMatchPillowWhenRequested() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_FALCON_RESIZE_PARITY_CASE"] else {
            throw XCTSkip("Set MERERUN_FALCON_RESIZE_PARITY_CASE for full-frame pixel qualification.")
        }
        let declaration = try JSONDecoder().decode(
            FrameDeclaration.self, from: Data(contentsOf: URL(fileURLWithPath: path))
        )
        XCTAssertFalse(declaration.frames.isEmpty)
        for frame in declaration.frames {
            let input = try MediaImageIO.decode(URL(fileURLWithPath: frame.sourcePath))
            let initial = FalconPerceptionProcessor.resizeIfNecessary(input, shortest: 256, longest: 1024)
            let output = FalconPerceptionProcessor.smartResize(initial, factor: 16)
            try MediaImageIO.writePNG(output, to: URL(fileURLWithPath: frame.outputPath))
            let reference = try MediaImageIO.decode(URL(fileURLWithPath: frame.referencePath))
            XCTAssertEqual(output.width, reference.width, frame.sourcePath)
            XCTAssertEqual(output.height, reference.height, frame.sourcePath)
            let differences = zip(output.rgba8, reference.rgba8).enumerated().filter {
                $0.offset % 4 < 3 && $0.element.0 != $0.element.1
            }.count
            XCTAssertEqual(differences, 0, "\(frame.sourcePath): \(differences) differing RGB bytes")
        }
    }
}
