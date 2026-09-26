@testable import StudioKit
import Foundation
import UniformTypeIdentifiers
import XCTest

/// "Send to…": which pages and slots an output is offered to, read from the wells' own slot
/// schema, and that choosing one fills the slot the way a drop would.
@MainActor
final class StudioSendDestinationsTests: XCTestCase {
    /// Files an output can be, by the extensions the runs write, one or more per family the
    /// wells name.
    private static let samples = [
        "png", "jpg", "tiff", "heic", "gif",
        "wav", "mp3", "m4a", "aiff", "flac",
        "mp4", "mov",
        "json", "txt", "csv", "pdf",
    ]

    private func url(_ ext: String) -> URL {
        URL(fileURLWithPath: "/Outputs/sample.\(ext)")
    }

    private func key(_ destination: StudioSendDestination) -> String {
        "\(destination.task.rawValue) · \(destination.slot.label)"
    }

    /// The sweep: for every sample type, the offered destinations are exactly the slots whose
    /// declared types that file conforms to (one per task and slot name), and every slot that
    /// declares a type of its own is reached by some sample, so no well is left out of the menu.
    func testEverySlotIsOfferedForTheTypesItTakesAndNoOthers() throws {
        var reached: Set<String> = []
        for ext in Self.samples {
            let file = url(ext)
            let type = try XCTUnwrap(UTType(filenameExtension: ext))
            var expected: [String] = []
            for destination in StudioSendDestinations.all {
                let declared = destination.slot.acceptedTypes.filter {
                    !StudioSendDestinations.catchAllTypes.contains($0) && $0 != .folder
                }
                guard !declared.isEmpty, !destination.slot.acceptedTypes.contains(.folder),
                      declared.contains(where: { type.conforms(to: $0) }) else { continue }
                if !expected.contains(key(destination)) { expected.append(key(destination)) }
                reached.insert(destination.id)
            }
            let offered = StudioSendDestinations.destinations(for: file).map(key)
            XCTAssertEqual(offered, expected, ext)
        }
        for destination in StudioSendDestinations.all {
            let declared = destination.slot.acceptedTypes.filter { !StudioSendDestinations.catchAllTypes.contains($0) }
            guard !declared.isEmpty, !declared.contains(.folder) else { continue }
            XCTAssertTrue(
                reached.contains(destination.id)
                    || StudioSendDestinations.all.contains { reached.contains($0.id) && key($0) == key(destination) },
                "\(destination.id) takes \(declared.map(\.identifier)) but no sample reaches it"
            )
        }
    }

    /// Only files a slot names are sent to it: a slot that takes any file (`.data`) or a folder
    /// never makes a destination, whatever the file.
    func testCatchAllAndFolderSlotsAreNeverDestinations() {
        let catchAll = StudioSendDestinations.all.filter { destination in
            destination.slot.acceptedTypes.allSatisfy(StudioSendDestinations.catchAllTypes.contains)
                || destination.slot.acceptedTypes.contains(.folder)
        }
        XCTAssertFalse(catchAll.isEmpty, "the schema declares catch-all and folder slots")
        for ext in Self.samples {
            let offered = Set(StudioSendDestinations.destinations(for: url(ext)).map(\.id))
            for destination in catchAll { XCTAssertFalse(offered.contains(destination.id), destination.id) }
        }
        XCTAssertTrue(StudioSendDestinations.destinations(for: URL(fileURLWithPath: "/Outputs/features.npy")).isEmpty)
        XCTAssertTrue(StudioSendDestinations.destinations(for: URL(fileURLWithPath: "/Outputs/stems")).isEmpty)
    }

    /// The journeys the menu is for: a picture to a video's start frame, Vision, 3D, and an edit;
    /// a song or a stem to Separate, Transcribe, Analyze, and Enhance; a narration to Voices and
    /// Foley; a clip to Track and Foley. Each file reaches no page of another medium.
    func testOutputsReachThePagesOfTheirMedium() {
        func offered(_ name: String) -> Set<String> {
            Set(StudioSendDestinations.destinations(for: URL(fileURLWithPath: "/Outputs/\(name)")).map(key))
        }
        let picture = offered("mug.png")
        for expected in [
            "video.generate · Start frame", "vision.read · Image", "vision.find · Image", "vision.segment · Image",
            "vision.depth · Image", "threeD.fromImage · Input image", "image.generate · Input", "chat.chat · Image",
        ] {
            XCTAssertTrue(picture.contains(expected), expected)
        }
        XCTAssertFalse(picture.contains { $0.hasPrefix("audio.") || $0.hasPrefix("music.") || $0.hasPrefix("vision.track") })

        let song = offered("harbor.wav")
        for expected in [
            "music.separate · Audio", "music.transcribe · Audio", "music.analyze · Audio", "audio.enhance · Audio",
            "audio.transcribe · Audio", "voice.voices · Reference audio", "sound.foley · Video or features", "voice.speak · Reference audio",
        ] {
            XCTAssertTrue(song.contains(expected), expected)
        }
        XCTAssertFalse(song.contains { $0.hasPrefix("vision.") || $0.hasPrefix("image.") || $0.hasPrefix("threeD.") })

        let clip = offered("take.mp4")
        XCTAssertEqual(clip, ["vision.track · Video", "vision.depth · Video", "sound.foley · Video or features"])
    }

    /// The page the output is shown on is its "Use as input", never a Send to item.
    func testThePageShowingTheOutputIsLeftOut() {
        let song = URL(fileURLWithPath: "/Outputs/harbor.wav")
        XCTAssertTrue(StudioSendDestinations.destinations(for: song).contains { $0.task == .musicSeparate })
        XCTAssertFalse(StudioSendDestinations.destinations(for: song, excluding: .musicSeparate).contains { $0.task == .musicSeparate })
    }

    /// Sections follow the sidebar, headed by the domain; an item names its slot only when its
    /// task takes the file in more than one.
    func testSectionsFollowTheSidebarAndNameTheSlotOnlyWhenNeeded() {
        let sections = StudioSendDestinations.sections(
            StudioSendDestinations.destinations(for: URL(fileURLWithPath: "/Outputs/mug.png"), excluding: .imageGenerate)
        )
        XCTAssertEqual(sections.map(\.domain), StudioDomain.allCases.filter { domain in sections.contains { $0.domain == domain } })
        XCTAssertEqual(sections.first?.title, "Video")
        let video = sections.first { $0.domain == .video }?.items.map(\.title)
        XCTAssertEqual(video, ["Generate · Start frame", "Generate · End frame"])
        let vision = sections.first { $0.domain == .vision }?.items.map(\.title) ?? []
        XCTAssertTrue(vision.contains("Read"))
        XCTAssertTrue(vision.contains("Flow · From"))
        XCTAssertFalse(sections.contains { $0.items.contains { $0.destination.task == .imageGenerate } })
    }

    /// A prompt mode's slot fills through its own `attach`, with the settings that follow it: a
    /// reference clip puts Speak in clone mode.
    func testAPromptModeDestinationFillsItsSlotLikeADrop() throws {
        let clip = URL(fileURLWithPath: "/Outputs/narration.wav")
        let reference = try XCTUnwrap(StudioSendDestinations.destinations(for: clip).first { $0.task == .voiceSpeak })
        var draft = StudioDraft()
        draft.reset(for: .speak)
        draft.prompt = "Keep this line"
        reference.attach(clip, to: &draft)
        XCTAssertEqual(draft.refAudioPath, clip.path)
        XCTAssertEqual(draft.voiceMode, "clone")
        XCTAssertEqual(draft.prompt, "Keep this line")
    }

    /// A task draft stays on its variant when that variant takes the file under the same slot
    /// name, and switches to the declaring variant when it does not: a clip sent to Depth moves
    /// a picture draft to Depth's video variant, and a picture moves it back.
    func testATaskDestinationSwitchesToTheVariantThatTakesTheFile() throws {
        let clip = URL(fileURLWithPath: "/Outputs/take.mp4")
        let picture = URL(fileURLWithPath: "/Outputs/mug.png")
        let toVideo = try XCTUnwrap(StudioSendDestinations.destinations(for: clip).first { $0.task == .visionDepth })
        let toPicture = try XCTUnwrap(StudioSendDestinations.destinations(for: picture).first { $0.task == .visionDepth })

        var draft = try XCTUnwrap(StudioTaskDraft(task: .visionDepth))
        XCTAssertEqual(draft.templateID, .visionDepth)
        toVideo.attach(clip, to: &draft)
        XCTAssertEqual(draft.templateID, .visionDepthVideo)
        XCTAssertEqual(draft.primaryInputPath, clip.path)

        toPicture.attach(picture, to: &draft)
        XCTAssertEqual(draft.templateID, .visionDepth)
        XCTAssertEqual(draft.primaryInputPath, picture.path)

        // Already on a variant whose well takes the file under that name: no switch.
        var compare = StudioTaskDraft(templateID: .visionFaceCompare)
        let candidate = try XCTUnwrap(StudioSendDestinations.destinations(for: picture).first {
            $0.task == .visionFaces && $0.slot.label == "Candidate"
        })
        candidate.attach(picture, to: &compare)
        XCTAssertEqual(compare.templateID, .visionFaceCompare)
        XCTAssertEqual(candidate.slot.paths(in: compare), [picture.path])
    }
}
