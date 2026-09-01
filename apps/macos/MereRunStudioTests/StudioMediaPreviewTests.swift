import AppKit
@testable import MereRunApp
import SwiftUI
import XCTest

final class StudioMediaPreviewTests: XCTestCase {
    @MainActor
    func testVideoPlayerViewCanBeHosted() {
        let url = URL(fileURLWithPath: "/tmp/mere-run-video-player-linkage-smoke.mp4")
        let hostingView = NSHostingView(rootView: StudioVideoPlayerView(url: url))
        hostingView.frame = NSRect(x: 0, y: 0, width: 320, height: 180)

        hostingView.layoutSubtreeIfNeeded()
        hostingView.displayIfNeeded()

        XCTAssertEqual(hostingView.frame.size, NSSize(width: 320, height: 180))
    }

    func testMovieOutputsAreNotReadAsTextPreviews() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mp4")
        try Data("not really a movie, but still not a text preview".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertEqual(StudioOutputFileKind.classify(url), .video)
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

    func testAudioAndVideoOutputsAreClassifiedForPlayback() {
        XCTAssertEqual(StudioOutputFileKind.classify(URL(fileURLWithPath: "/tmp/a.wav")), .audio)
        XCTAssertEqual(StudioOutputFileKind.classify(URL(fileURLWithPath: "/tmp/a.mp3")), .audio)
        XCTAssertEqual(StudioOutputFileKind.classify(URL(fileURLWithPath: "/tmp/a.mov")), .video)
    }

    func test3DAssetsAreClassifiedForEmbeddedQuickLook() {
        for pathExtension in ["glb", "gltf", "obj", "ply", "stl", "usdz"] {
            XCTAssertEqual(
                StudioOutputFileKind.classify(URL(fileURLWithPath: "/tmp/asset.\(pathExtension)")),
                .model3D
            )
        }
    }
}
