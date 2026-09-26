@testable import StudioKit
import AppKit
import UniformTypeIdentifiers
import XCTest

/// ⌘V into the composer or a well: what the pasteboard is read as, and that a paste writes a
/// picture or sound only where a slot takes it. Every test uses a private pasteboard, never the
/// user's clipboard.
final class StudioAttachmentPasteboardTests: XCTestCase {
    private var pasteboard: NSPasteboard!
    private var directory: URL!
    private let now = Date(timeIntervalSince1970: 1_790_428_991) // 2026-09-26 at 14.03.11 UTC

    override func setUpWithError() throws {
        pasteboard = NSPasteboard(name: NSPasteboard.Name("studio-paste-test-\(UUID().uuidString)"))
        pasteboard.clearContents()
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("studio-paste-\(UUID().uuidString)")
    }

    override func tearDownWithError() throws {
        pasteboard.releaseGlobally()
        if FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.removeItem(at: directory)
        }
    }

    private func pngData() throws -> Data {
        let bitmap = try XCTUnwrap(NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: 2, pixelsHigh: 2, bitsPerSample: 8, samplesPerPixel: 4,
            hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
        ))
        return try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
    }

    private func type(_ uti: UTType) -> NSPasteboard.PasteboardType {
        NSPasteboard.PasteboardType(uti.identifier)
    }

    // MARK: Reading the pasteboard

    /// A Finder copy is the files, whatever else rides along (Finder adds the name as text).
    func testAFinderCopyIsItsFiles() throws {
        let song = URL(fileURLWithPath: "/Users/me/Music/harbor.wav")
        let mug = URL(fileURLWithPath: "/Users/me/Pictures/mug.png")
        pasteboard.writeObjects([song as NSURL, mug as NSURL])
        pasteboard.addTypes([.string], owner: nil)
        pasteboard.setString("harbor.wav", forType: .string)
        XCTAssertEqual(StudioAttachmentPasteboard.content(of: pasteboard), .files([song, mug]))
    }

    /// A screenshot is its picture: PNG as copied, a TIFF redrawn as PNG.
    func testAScreenshotIsItsPicture() throws {
        let png = try pngData()
        pasteboard.setData(png, forType: type(.png))
        XCTAssertEqual(StudioAttachmentPasteboard.content(of: pasteboard), .media(png, .png))

        pasteboard.clearContents()
        let tiff = try XCTUnwrap(NSImage(data: png)?.tiffRepresentation)
        pasteboard.setData(tiff, forType: .tiff)
        guard case .media(let data, let written) = StudioAttachmentPasteboard.content(of: pasteboard) else {
            return XCTFail("a TIFF screenshot is a picture")
        }
        XCTAssertEqual(written, .png)
        XCTAssertEqual(NSBitmapImageRep(data: data)?.pixelsWide, 2)
    }

    /// Copied sound is its bytes in the type it was copied as.
    func testCopiedSoundIsItsAudio() {
        let wav = Data("RIFF....WAVEfmt ".utf8)
        pasteboard.setData(wav, forType: type(.wav))
        XCTAssertEqual(StudioAttachmentPasteboard.content(of: pasteboard), .media(wav, .wav))
    }

    /// Words are text, and so is a rich-text selection that carries a picture of itself.
    func testWordsAreTextEvenWithAPictureOfThem() throws {
        pasteboard.setString("a ceramic mug in morning light", forType: .string)
        XCTAssertEqual(StudioAttachmentPasteboard.content(of: pasteboard), .text)

        pasteboard.clearContents()
        pasteboard.declareTypes([.rtf, .string, type(.png)], owner: nil)
        pasteboard.setData(Data("{\\rtf1 mug}".utf8), forType: .rtf)
        pasteboard.setString("mug", forType: .string)
        pasteboard.setData(try pngData(), forType: type(.png))
        XCTAssertEqual(StudioAttachmentPasteboard.content(of: pasteboard), .text)
    }

    /// Copy Image in a browser is rich text with no words — an `<img>` beside the picture — and
    /// reads as the picture.
    func testCopyImageFromABrowserIsItsPicture() throws {
        let png = try pngData()
        pasteboard.declareTypes([.html, type(.png)], owner: nil)
        pasteboard.setString("<img src=\"https://example.com/mug.png\">", forType: .html)
        pasteboard.setData(png, forType: type(.png))
        XCTAssertEqual(StudioAttachmentPasteboard.content(of: pasteboard), .media(png, .png))
    }

    func testAnEmptyOrForeignPasteboardIsUnsupported() {
        XCTAssertEqual(StudioAttachmentPasteboard.content(of: pasteboard), .unsupported)
        pasteboard.setData(Data([1, 2, 3]), forType: NSPasteboard.PasteboardType("com.example.private"))
        XCTAssertEqual(StudioAttachmentPasteboard.content(of: pasteboard), .unsupported)
    }

    // MARK: Taking the paste

    /// Files go through as they are, only those the slot takes; none taken is a refusal.
    func testFilesAttachWhenTakenAndAreRefusedOtherwise() throws {
        let song = URL(fileURLWithPath: "/Users/me/Music/harbor.wav")
        let notes = URL(fileURLWithPath: "/Users/me/notes.pdf")
        let audioSlot = StudioMode.listen.attachmentSlots[0]
        XCTAssertEqual(
            try StudioAttachmentPasteboard.intake(.files([notes, song]), into: directory, now: now, accepts: audioSlot.accepts),
            .attach([song])
        )
        XCTAssertEqual(
            try StudioAttachmentPasteboard.intake(.files([notes]), into: directory, now: now, accepts: audioSlot.accepts),
            .refused
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path), "files are never copied")
    }

    /// A picture a slot takes is written under the given folder, named like a screenshot; a
    /// second in the same second gets its own name. One no slot takes writes nothing.
    func testAPictureIsWrittenOnlyWhereASlotTakesIt() throws {
        let png = try pngData()
        let imageSlot = StudioMode.findObjects.attachmentSlots[0]
        let audioSlot = StudioMode.listen.attachmentSlots[0]

        XCTAssertEqual(
            try StudioAttachmentPasteboard.intake(.media(png, .png), into: directory, now: now, accepts: audioSlot.accepts),
            .refused
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))

        guard case .attach(let first) = try StudioAttachmentPasteboard.intake(
            .media(png, .png), into: directory, now: now, accepts: imageSlot.accepts
        ), case .attach(let second) = try StudioAttachmentPasteboard.intake(
            .media(png, .png), into: directory, now: now, accepts: imageSlot.accepts
        ) else {
            return XCTFail("an image slot takes a pasted picture")
        }
        XCTAssertEqual(first.map { $0.deletingLastPathComponent().standardizedFileURL }, [directory.standardizedFileURL])
        XCTAssertTrue(first[0].lastPathComponent.hasPrefix("Pasted image "))
        XCTAssertEqual(first[0].pathExtension, "png")
        XCTAssertNotEqual(first, second)
        XCTAssertEqual(try Data(contentsOf: first[0]), png)
        XCTAssertTrue(imageSlot.accepts(first[0]))
    }

    /// Copied sound lands as an audio file an audio slot takes.
    func testCopiedSoundIsWrittenAsAudio() throws {
        let wav = Data("RIFF....WAVEfmt ".utf8)
        let audioSlot = StudioMode.listen.attachmentSlots[0]
        guard case .attach(let urls) = try StudioAttachmentPasteboard.intake(
            .media(wav, .wav), into: directory, now: now, accepts: audioSlot.accepts
        ) else {
            return XCTFail("an audio slot takes pasted sound")
        }
        XCTAssertTrue(urls[0].lastPathComponent.hasPrefix("Pasted audio "))
        XCTAssertEqual(try Data(contentsOf: urls[0]), wav)
    }

    /// Words are left to the text field; nothing else is.
    func testWordsAreLeftToTheTextField() throws {
        XCTAssertEqual(try StudioAttachmentPasteboard.intake(.text, into: directory, accepts: { _ in true }), .text)
        XCTAssertEqual(try StudioAttachmentPasteboard.intake(.unsupported, into: directory, accepts: { _ in true }), .refused)
    }

    /// Pasted files live in the app's own support folder, never a folder of the user's.
    func testPastedFilesLiveInTheAppSupportFolder() {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        XCTAssertEqual(
            StudioAttachmentPasteboard.pastedFilesDirectory.standardizedFileURL,
            support.appendingPathComponent("MereRun/Pasted", isDirectory: true).standardizedFileURL
        )
    }
}
