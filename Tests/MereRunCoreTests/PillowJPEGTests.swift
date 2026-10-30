import Foundation
import MediaIO
import XCTest

final class PillowJPEGTests: XCTestCase {
    private var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/PillowJPEG")
    }

    func testNativeJPEGBytesMatchIndependentPillowGoldens() throws {
        struct Manifest: Decodable { let cases: [Case] }
        struct Case: Decodable { let name: String; let width: Int; let height: Int }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: fixtures.appendingPathComponent("manifest.json")))
        for item in manifest.cases {
            let image = try MediaImageIO.decodeWithPillowJPEG(fixtures.appendingPathComponent(item.name + ".jpeg"))
            XCTAssertEqual(image.width, item.width, item.name)
            XCTAssertEqual(image.height, item.height, item.name)
            let rgb = image.rgba8.enumerated().compactMap { $0.offset % 4 == 3 ? nil : $0.element }
            XCTAssertEqual(Data(rgb), try Data(contentsOf: fixtures.appendingPathComponent(item.name + ".rgb")), item.name)
            XCTAssertTrue(stride(from: 3, to: image.rgba8.count, by: 4).allSatisfy { image.rgba8[$0] == 255 }, item.name)
        }
    }

    func testNonJPEGInputsRetainThePlatformDecoder() throws {
        let url = fixtures.appendingPathComponent("rgb.png")
        XCTAssertEqual(try MediaImageIO.decodeWithPillowJPEG(url), try MediaImageIO.decode(url))
    }

    func testMalformedAndTruncatedJPEGThrowWithoutAborting() throws {
        let original = try Data(contentsOf: fixtures.appendingPathComponent("rgb-2.jpeg"))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".jpeg")
        defer { try? FileManager.default.removeItem(at: url) }
        for bytes in [Data([0xff, 0xd8, 0xff]), Data(original.prefix(original.count / 2)), Data(original.dropLast(2))] {
            try bytes.write(to: url)
            XCTAssertThrowsError(try MediaImageIO.decodeWithPillowJPEG(url)) { error in
                XCTAssertEqual(error as? MediaIOError, .imageDecodeFailed(url))
            }
        }
    }

    func testJPEGSignatureSelectsTheDecoderRegardlessOfFilename() throws {
        let source = fixtures.appendingPathComponent("progressive-420.jpeg")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(contentsOf: source).write(to: url)
        XCTAssertEqual(try MediaImageIO.decodeWithPillowJPEG(url), try MediaImageIO.decodeWithPillowJPEG(source))
    }

    func testConcurrentDecodesHaveIndependentErrorAndPixelBuffers() async throws {
        let url = fixtures.appendingPathComponent("progressive-420.jpeg")
        let expected = try MediaImageIO.decodeWithPillowJPEG(url)
        try await withThrowingTaskGroup(of: MediaImage.self) { group in
            for _ in 0..<16 { group.addTask { try MediaImageIO.decodeWithPillowJPEG(url) } }
            for try await image in group { XCTAssertEqual(image, expected) }
        }
    }
}
