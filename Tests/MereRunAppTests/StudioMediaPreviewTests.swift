@testable import MereRunApp
import XCTest

final class StudioMediaPreviewTests: XCTestCase {
    func testMovieOutputsAreNotReadAsTextPreviews() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mp4")
        try Data("not really a movie, but still not a text preview".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertEqual(StudioOutputFileKind.classify(url), .other)
        XCTAssertNil(StudioTextPreviewReader.previewText(from: url))
    }

    func testTextPreviewsAreCapped() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("log")
        let text = String(repeating: "a", count: StudioTextPreviewReader.maxPreviewBytes + 20)
        try Data(text.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let preview = try XCTUnwrap(StudioTextPreviewReader.previewText(from: url))
        XCTAssertLessThanOrEqual(
            preview.utf8.count,
            StudioTextPreviewReader.maxPreviewBytes + "\n\n[Preview truncated.]".utf8.count
        )
        XCTAssertTrue(preview.hasSuffix("[Preview truncated.]"))
    }

    func testImagesAreClassifiedWithoutLoadingTheFile() {
        let url = URL(fileURLWithPath: "/tmp/example.png")
        XCTAssertEqual(StudioOutputFileKind.classify(url), .image)
    }
}
