import Foundation
import MediaIO
import XCTest
@testable import MereRunCore

final class ClefMediaParityTests: XCTestCase {
    private struct Fixture: Decodable {
        let cases: [Case]
        let normalized_8bit_lut: [Float]
        let video_geometry: [VideoGeometry]
        struct Case: Decodable {
            let kind: String
            let inputs: [String]
            let expected: [String]
            let grid: [Int]
        }
        struct VideoGeometry: Decodable {
            let frames: Int
            let width: Int
            let height: Int
            let minPixels: Int
            let maxPixels: Int
            let expectedWidth: Int
            let expectedHeight: Int
        }
    }

    private struct Request: Encodable {
        struct Question: Encodable {
            let type = "noul"
            let instructions = "Is the media predominantly red?"
        }
        let state = "Inspect the supplied media."
        let questions = ["red": Question()]
        let images: [String]?
        let videos: [[String]]?
    }

    func testImageAndVideoResizeMatchIndependentPillowPixels() throws {
        let root = Bundle.module.resourceURL!.appending(path: "Fixtures/Clef")
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: root.appending(path: "media-reference.json")))
        let processor = try JSONDecoder().decode(
            ClefProcessorConfiguration.self, from: Data(contentsOf: root.appending(path: "processor_config.json")))
        XCTAssertEqual(fixture.cases.map(\.kind), ["image", "video"])
        for item in fixture.cases {
            let paths = item.inputs.map { root.appending(path: $0).path }
            let input = Request(images: item.kind == "image" ? paths : nil, videos: item.kind == "video" ? [paths] : nil)
            let request = try ClefDecisionRequest.decode(JSONEncoder().encode(input))
            let media = try ClefPreparedMedia.prepare(request, processor: processor)
            XCTAssertEqual(media.items.count, 1)
            let actual = try XCTUnwrap(media.items.first)
            XCTAssertEqual([actual.grid.0, actual.grid.1, actual.grid.2], item.grid)
            XCTAssertEqual(actual.frameCount, item.expected.count)
            var normalized: [Float] = []
            for (sourcePath, referencePath) in zip(item.inputs, item.expected) {
                let source = try MediaImageIO.decode(root.appending(path: sourcePath))
                let reference = try MediaImageIO.decode(root.appending(path: referencePath))
                let resized = try MediaImageIO.bicubicResizedRGB(source, width: reference.width, height: reference.height)
                let actualRGB = resized.rgba8.enumerated().compactMap { $0.offset % 4 < 3 ? $0.element : nil }
                let expectedRGB = reference.rgba8.enumerated().compactMap { $0.offset % 4 < 3 ? $0.element : nil }
                XCTAssertEqual(actualRGB, expectedRGB, sourcePath)
                XCTAssertEqual(actual.width, reference.width)
                XCTAssertEqual(actual.height, reference.height)
                // Independently exported numpy FP32 rescale, then mean/std.
                for channel in 0..<3 {
                    for pixel in 0..<(reference.width * reference.height) {
                        normalized.append(fixture.normalized_8bit_lut[Int(reference.rgba8[pixel * 4 + channel])])
                    }
                }
            }
            XCTAssertEqual(actual.pixels.count, normalized.count)
            let maximumError = zip(actual.pixels, normalized).map { abs($0 - $1) }.max() ?? 0
            XCTAssertEqual(maximumError, 0, item.kind)
        }
    }

    func testOddFrameResizeBudgetIncludesPaddedTemporalPair() throws {
        let root = Bundle.module.resourceURL!.appending(path: "Fixtures/Clef")
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: root.appending(path: "media-reference.json")))
        XCTAssertEqual(fixture.video_geometry.count, 2)
        for item in fixture.video_geometry {
            let size = try ClefPreparedMedia.videoSize(width: item.width, height: item.height, frames: item.frames,
                                                       minPixels: item.minPixels, maxPixels: item.maxPixels)
            XCTAssertEqual(size.width, item.expectedWidth)
            XCTAssertEqual(size.height, item.expectedHeight)
        }
    }

    func testRescaleMatchesReferenceForEveryByteIncludingBF16RoundingBoundary() throws {
        let root = Bundle.module.resourceURL!.appending(path: "Fixtures/Clef")
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: root.appending(path: "media-reference.json")))
        let rgba = (0...255).flatMap { [UInt8($0), UInt8($0), UInt8($0), UInt8(255)] }
        let input = try MediaImage(width: 256, height: 1, rgba8: rgba)
        let actual = MediaImageIO.rescaledRGBCHWFloat(input, rescaleFactor: Float(1 / 255.0), normalizedToMinusOneToOne: true)
        XCTAssertEqual(fixture.normalized_8bit_lut.count, 256)
        XCTAssertEqual(actual, Array(repeating: fixture.normalized_8bit_lut, count: 3).flatMap { $0 })
    }
}
