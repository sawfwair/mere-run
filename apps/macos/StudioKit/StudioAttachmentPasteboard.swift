import AppKit
import UniformTypeIdentifiers

// ⌘V into the composer or a well. The pasteboard is read into files first — a Finder copy as the
// files themselves, a screenshot or copied sound written under the app's own support folder —
// and those files then take the drop path (`attach(dropped:slots:)`, or a well's own `attach`),
// so a paste lands exactly where dropping the same file would.

/// What the pasteboard holds, as the composer and the wells read it.
package enum StudioPasteContent: Equatable {
    /// Files copied in Finder or another app, as their URLs.
    case files([URL])
    /// Picture or sound bytes with no file behind them (a screenshot, Copy Image, a clip copied
    /// from an audio editor), in the type they will be written as.
    case media(Data, UTType)
    /// Words, which paste into the prompt as text.
    case text
    /// Nothing an attachment or the prompt can take.
    case unsupported
}

/// What a paste does once the pasteboard is read.
package enum StudioPasteIntake: Equatable {
    /// Attach these files, through the same routing a drop of them takes.
    case attach([URL])
    /// Let the focused text field paste the words.
    case text
    /// Nothing here takes what was copied; the paste does nothing.
    case refused
}

package enum StudioAttachmentPasteboard {
    /// Where pasted pictures and sounds are written: the app's own support folder, never a
    /// folder of the user's.
    package static var pastedFilesDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MereRun", isDirectory: true)
            .appendingPathComponent("Pasted", isDirectory: true)
    }

    /// Reads `pasteboard`: file URLs first; then a picture or a sound, unless the copy is rich
    /// text with words (a Word or web-page selection carries a picture of itself beside its
    /// words, and those words are what the user copied — while Copy Image in a browser or a
    /// picture copied from Notes is rich text with no words); then plain text.
    package static func content(of pasteboard: NSPasteboard) -> StudioPasteContent {
        let files = pasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? []
        if !files.isEmpty { return .files(files) }
        let types = pasteboard.types ?? []
        let isRichText = types.contains { [.rtf, .rtfd, .html].contains($0) }
        if !(isRichText && types.contains(.string)) {
            if let media = image(on: pasteboard, types: types) ?? audio(on: pasteboard, types: types) {
                return .media(media.data, media.type)
            }
        }
        return types.contains(.string) ? .text : .unsupported
    }

    /// Decides what pasting `content` does where `accepts` says which files are taken: files it
    /// takes are attached, a picture or sound it takes is first written into `directory`, words
    /// are left to the text field, and anything else is refused. Nothing is written for a paste
    /// that is refused.
    package static func intake(
        _ content: StudioPasteContent,
        into directory: URL = pastedFilesDirectory,
        now: Date = Date(),
        accepts: (URL) -> Bool
    ) throws -> StudioPasteIntake {
        switch content {
        case .files(let urls):
            let taken = urls.filter(accepts)
            return taken.isEmpty ? .refused : .attach(taken)
        case .media(let data, let type):
            let noun = type.conforms(to: .audio) ? "audio" : "image"
            let name = "Pasted \(noun) \(timestamp.string(from: now))"
            let ext = type.preferredFilenameExtension ?? "dat"
            guard accepts(directory.appendingPathComponent("\(name).\(ext)")) else { return .refused }
            return .attach([try write(data, named: name, extension: ext, into: directory)])
        case .text:
            return .text
        case .unsupported:
            return .refused
        }
    }

    /// The pasteboard's picture: PNG or JPEG bytes as copied, any other picture (a TIFF
    /// screenshot) redrawn as PNG.
    private static func image(on pasteboard: NSPasteboard, types: [NSPasteboard.PasteboardType]) -> (data: Data, type: UTType)? {
        for type in [UTType.png, .jpeg] {
            if let data = pasteboard.data(forType: NSPasteboard.PasteboardType(type.identifier)) { return (data, type) }
        }
        guard types.contains(where: { UTType($0.rawValue)?.conforms(to: .image) == true }),
              let image = NSImage(pasteboard: pasteboard),
              let tiff = image.tiffRepresentation,
              let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) else {
            return nil
        }
        return (png, .png)
    }

    /// The pasteboard's sound, in the first audio type it was copied as.
    private static func audio(on pasteboard: NSPasteboard, types: [NSPasteboard.PasteboardType]) -> (data: Data, type: UTType)? {
        for pasteboardType in types {
            guard let type = UTType(pasteboardType.rawValue), type.conforms(to: .audio),
                  let data = pasteboard.data(forType: pasteboardType) else { continue }
            return (data, type)
        }
        return nil
    }

    /// Writes `data` as "<name>.<ext>", or "<name> 2.<ext>" and on when a paste in the same
    /// second already took the name.
    private static func write(_ data: Data, named name: String, extension ext: String, into directory: URL) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var url = directory.appendingPathComponent("\(name).\(ext)")
        var counter = 2
        while FileManager.default.fileExists(atPath: url.path) {
            url = directory.appendingPathComponent("\(name) \(counter).\(ext)")
            counter += 1
        }
        try data.write(to: url, options: .withoutOverwriting)
        return url
    }

    /// "2026-09-26 at 14.03.11", the way macOS names a screenshot.
    private static let timestamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd 'at' HH.mm.ss"
        return formatter
    }()
}
