@testable import StudioKit
import Foundation
import UniformTypeIdentifiers
import XCTest

/// "From Library…": which finished runs an attachment entry point offers, and that a pick fills
/// the slot exactly as a disk pick does.
final class StudioLibraryInputsTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_788_527_400)

    private func run(
        _ title: String,
        mode: StudioMode = .music,
        outputs: [String],
        status: StudioLibraryStatus = .completed,
        minutesAgo: Double,
        roles: [String: String]? = nil
    ) -> StudioLibraryItem {
        let urls = outputs.map { URL(fileURLWithPath: "/Library/Outputs/\($0)") }
        var item = StudioLibraryItem(
            id: UUID(),
            mode: mode,
            prompt: title,
            inputURL: nil,
            outputURL: urls.first,
            createdAt: now.addingTimeInterval(-minutesAgo * 60),
            updatedAt: now,
            status: status,
            exitCode: status == .completed ? 0 : 1,
            commandPreview: "mere.run",
            outputText: nil
        )
        item.artifactURLs = urls.count > 1 ? Array(urls.dropFirst()) : nil
        item.artifactRoles = roles
        return item
    }

    private var audioSlot: StudioAttachmentSlot {
        StudioMode.listen.attachmentSlots[0]
    }

    func testAnAudioSlotOffersFinishedAudioNewestFirstWithEachStem() {
        let song = run("harbor at dawn", outputs: ["song.wav"], minutesAgo: 30)
        let stems = run(
            "split harbor", outputs: ["vocals.wav", "drums.wav", "stems.json"], minutesAgo: 5,
            roles: ["/Library/Outputs/drums.wav": "stem"]
        )
        let picture = run("a mug", mode: .createImage, outputs: ["mug.png"], minutesAgo: 1)
        let failed = run("broken take", outputs: ["broken.wav"], status: .failed, minutesAgo: 2)
        let running = run("still going", outputs: ["partial.wav"], status: .running, minutesAgo: 3)
        let gone = run("deleted since", outputs: ["gone.wav"], minutesAgo: 4)
        var thread = run("chat", mode: .chat, outputs: ["reply.wav"], minutesAgo: 6)
        thread.messages = []

        let groups = StudioLibraryInputs.groups(
            in: [song, stems, picture, failed, running, gone, thread],
            for: StudioAttachmentRequirement(slot: audioSlot),
            fileExists: { $0.lastPathComponent != "gone.wav" }
        )

        XCTAssertEqual(groups.map(\.id), [stems.id, song.id], "newest first; only finished, present audio")
        XCTAssertEqual(groups[0].files.map(\.lastPathComponent), ["vocals.wav", "drums.wav"],
                       "every stem is offered in the run's order; the JSON sidecar is not audio")
        XCTAssertEqual(groups[1].files.map(\.lastPathComponent), ["song.wav"])
        XCTAssertEqual(
            StudioLibraryInputs.choices(in: groups).map(\.url.lastPathComponent),
            ["vocals.wav", "drums.wav", "song.wav"],
            "the keyboard walks every file in list order"
        )
    }

    func testImageAndVideoSlotsTakeOnlyTheirMedia() {
        let batch = run("four mugs", mode: .createImage, outputs: ["a.png", "b.jpg", "c.heic", "grid.json"], minutesAgo: 1)
        let clip = run("a walk", mode: .video, outputs: ["walk.mp4", "walk.wav"], minutesAgo: 2)
        let items = [batch, clip]

        let image = StudioAttachmentRequirement(slot: StudioMode.findObjects.attachmentSlots[0])
        let images = StudioLibraryInputs.groups(in: items, for: image, fileExists: { _ in true })
        XCTAssertEqual(images.map(\.id), [batch.id])
        XCTAssertEqual(images[0].files.map(\.lastPathComponent), ["a.png", "b.jpg", "c.heic"])

        let video = StudioAttachmentRequirement(slot: StudioMode.track.attachmentSlots[0])
        let videos = StudioLibraryInputs.groups(in: items, for: video, fileExists: { _ in true })
        XCTAssertEqual(videos.map(\.id), [clip.id])
        XCTAssertEqual(videos[0].files.map(\.lastPathComponent), ["walk.mp4"])
        XCTAssertEqual(image.mediaNoun, "images")
        XCTAssertEqual(video.mediaNoun, "videos")
        XCTAssertEqual(StudioAttachmentRequirement(slot: audioSlot).mediaNoun, "audio")
    }

    func testNoCompatibleRunMeansTheDiskAlone() {
        let picture = run("a mug", mode: .createImage, outputs: ["mug.png"], minutesAgo: 1)
        let failedSong = run("broken", outputs: ["broken.wav"], status: .failed, minutesAgo: 2)
        let requirement = StudioAttachmentRequirement(slot: audioSlot)

        XCTAssertFalse(StudioLibraryInputs.hasCandidates(in: [picture, failedSong], for: requirement, fileExists: { _ in true }))
        XCTAssertFalse(StudioLibraryInputs.hasCandidates(in: [], for: requirement, fileExists: { _ in true }))
        let song = run("harbor", outputs: ["song.m4a"], minutesAgo: 3)
        XCTAssertTrue(StudioLibraryInputs.hasCandidates(in: [picture, song], for: requirement, fileExists: { _ in true }))
        XCTAssertFalse(StudioLibraryInputs.hasCandidates(in: [song], for: requirement, fileExists: { _ in false }),
                       "a file removed from disk is not an input")
    }

    func testSearchMatchesTheRunOrAFileName() {
        let song = run("harbor at dawn", outputs: ["song.wav"], minutesAgo: 1)
        let stems = run("split", outputs: ["vocals.wav", "drums.wav"], minutesAgo: 2)
        let requirement = StudioAttachmentRequirement(slot: audioSlot)

        func titles(_ query: String) -> [UUID] {
            StudioLibraryInputs.groups(in: [song, stems], for: requirement, query: query, fileExists: { _ in true }).map(\.id)
        }
        XCTAssertEqual(titles("HARBOR"), [song.id])
        XCTAssertEqual(titles("drums"), [stems.id], "a stem's file name finds its run")
        XCTAssertEqual(titles("  "), [song.id, stems.id])
        XCTAssertEqual(titles("nothing like this"), [])
    }

    /// Every well slot in Studio — each prompt mode's and each task template's, whatever it stores
    /// into — offers a Library file its types accept, and picking that file fills the slot just as
    /// the open panel would: the path in the draft field, flag, or positional the slot names.
    func testEverySlotOffersItsMediaAndAPickFillsIt() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("library-inputs-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        var storages = Set<String>()

        func check<Draft: StudioAttachmentDraft>(_ slot: StudioAttachmentSlot, in draft: Draft, context: String) throws {
            let url = try libraryFile(for: slot, folder: folder)
            var item = run("made for \(slot.id)", outputs: [], minutesAgo: 1)
            item.outputURL = url
            let requirement = StudioAttachmentRequirement(slot: slot)
            let groups = StudioLibraryInputs.groups(in: [item], for: requirement, fileExists: { _ in true })
            XCTAssertEqual(groups.first?.files, [url], "\(context) \(slot.id) offers \(url.lastPathComponent)")

            var filled = draft
            slot.attach([url], to: &filled)
            XCTAssertEqual(slot.paths(in: filled), [url.path], "\(context) \(slot.id) holds the picked path")
            storages.insert(Self.storageKind(slot.storage))
        }

        for mode in StudioMode.allCases {
            var draft = StudioDraft()
            draft.reset(for: mode)
            for slot in mode.attachmentSlots {
                try check(slot, in: draft, context: mode.rawValue)
            }
        }
        for templateID in CommandTemplateID.allCases {
            let draft = StudioTaskDraft(templateID: templateID)
            for slot in StudioTaskSchema.slots(for: templateID) {
                try check(slot, in: draft, context: templateID.rawValue)
            }
        }
        XCTAssertEqual(storages, ["path", "pathList", "flag", "flagList", "argument", "argumentList"],
                       "the sweep reaches every way a slot stores a file")
    }

    func testAPickFillsTheTaskDraftFlagTheSlotNames() throws {
        var found: (templateID: CommandTemplateID, slot: StudioAttachmentSlot, flag: String)?
        for templateID in CommandTemplateID.allCases where found == nil {
            for slot in StudioTaskSchema.slots(for: templateID) {
                if case .flag(let flag) = slot.storage, slot.acceptedTypes.contains(.audio) {
                    found = (templateID, slot, flag)
                    break
                }
            }
        }
        let target = try XCTUnwrap(found, "some task takes a reference audio file as a flag")
        let stem = URL(fileURLWithPath: "/Library/Outputs/vocals.wav")
        var item = run("split harbor", outputs: ["vocals.wav", "drums.wav"], minutesAgo: 1)
        item.artifactRoles = nil
        let groups = StudioLibraryInputs.groups(
            in: [item], for: StudioAttachmentRequirement(slot: target.slot), fileExists: { _ in true }
        )
        let picked = try XCTUnwrap(groups.first?.files.first)
        XCTAssertEqual(picked, stem)

        var draft = StudioTaskDraft(templateID: target.templateID)
        target.slot.attach([picked], to: &draft)

        XCTAssertEqual(draft.form.text(target.flag), stem.path,
                       "\(target.templateID.rawValue) \(target.flag) holds the stem's path")
    }

    private func libraryFile(for slot: StudioAttachmentSlot, folder: URL) throws -> URL {
        if slot.acceptedTypes.contains(.folder) {
            let directory = folder.appendingPathComponent("run-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            return directory
        }
        // Accepted types are often abstract (`.image`, `.audio`), so take the first common output
        // extension one of them covers.
        let fileExtension = try XCTUnwrap(
            ["png", "wav", "mp4", "txt", "json", "bin"].first { candidate in
                UTType(filenameExtension: candidate).map { type in slot.acceptedTypes.contains { type.conforms(to: $0) } } ?? false
            },
            "\(slot.id) takes none of the common output kinds"
        )
        return URL(fileURLWithPath: "/Library/Outputs/\(slot.id.trimmingCharacters(in: CharacterSet(charactersIn: "-"))).\(fileExtension)")
    }

    private static func storageKind(_ storage: StudioAttachmentSlot.Storage) -> String {
        switch storage {
        case .path: return "path"
        case .pathList: return "pathList"
        case .flag: return "flag"
        case .flagList: return "flagList"
        case .argument: return "argument"
        case .argumentList: return "argumentList"
        }
    }
}
