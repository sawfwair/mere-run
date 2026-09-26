@testable import StudioUI
import AppKit
import CoreTransferable
import UniformTypeIdentifiers
import XCTest

/// A Library row or output tile drags as a file: Finder gets the file's own type, and a Studio
/// well — whose `dropDestination(for: URL.self)` decodes the same way — gets the original path,
/// never a copy, so the slot names the file the Library shows.
@MainActor
final class StudioLibraryInputDragTests: XCTestCase {
    func testADraggedOutputArrivesAsItsOwnFileURL() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("drag-\(UUID().uuidString)")
            .appendingPathExtension("wav")
        try Data([0x52, 0x49, 0x46, 0x46]).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let provider = StudioFileDragSource.provider(for: url)

        XCTAssertTrue(provider.registeredTypeIdentifiers.contains(UTType.fileURL.identifier))
        XCTAssertTrue(provider.registeredTypeIdentifiers.contains(UTType.wav.identifier),
                      "Finder and other apps receive the file itself")
        XCTAssertEqual(provider.suggestedName, url.lastPathComponent)

        let received: URL = try await withCheckedThrowingContinuation { continuation in
            _ = provider.loadTransferable(type: URL.self) { continuation.resume(with: $0) }
        }
        XCTAssertEqual(received.standardizedFileURL.path, url.standardizedFileURL.path)
    }
}
