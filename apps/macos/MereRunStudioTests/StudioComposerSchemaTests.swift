@testable import MereRunApp
import Foundation
import MereRunContract
import UniformTypeIdentifiers
import XCTest

final class StudioComposerSchemaTests: XCTestCase {
    // MARK: - Slots

    func testEveryPromptModeDeclaresItsSlotsAndTheyMapToRealDraftFields() {
        let expectedSlotIDs: [StudioMode: [String]] = [
            .createImage: ["input", "references"],
            .video: ["startFrame", "endFrame", "audio"],
            .music: ["source", "timbre"],
            .speak: ["referenceAudio"],
            .chat: ["image"],
            .code: [],
            .readImage: ["input"],
            .findObjects: ["input"],
            .segment: ["input"],
            .track: ["input"],
            .listen: ["input"],
            .sfx: [],
        ]
        for mode in StudioMode.allCases {
            let slots = mode.attachmentSlots
            XCTAssertEqual(slots.map(\.id), expectedSlotIDs[mode], "\(mode) slot declaration drifted")
            XCTAssertEqual(Set(slots.map(\.id)).count, slots.count, "\(mode) declares a duplicate slot id")
            for slot in slots {
                XCTAssertFalse(slot.acceptedTypes.isEmpty, "\(mode).\(slot.id) accepts nothing")
                var draft = StudioDraft()
                draft.reset(for: mode)
                XCTAssertFalse(slot.isFilled(in: draft), "\(mode).\(slot.id) starts filled")
                let url = URL(fileURLWithPath: "/tmp/sample.\(Self.sampleExtension(for: slot))")
                slot.attach([url], to: &draft)
                XCTAssertEqual(slot.paths(in: draft), [url.path], "\(mode).\(slot.id) did not store the attachment")
                XCTAssertEqual(slot.caption(in: draft), url.lastPathComponent)
                slot.clear(in: &draft)
                XCTAssertFalse(slot.isFilled(in: draft), "\(mode).\(slot.id) did not clear")
            }
        }
    }

    func testRequiredSlotsMatchTheModesThatRequireAnAttachment() {
        for mode in StudioMode.allCases {
            let hasRequiredSlot = mode.attachmentSlots.contains(where: \.isRequired)
            XCTAssertEqual(hasRequiredSlot, mode.requiresAttachment, "\(mode) required-slot flag drifted")
        }
    }

    func testSlotsFeedTheCommandTheModeBuilds() throws {
        var image = StudioDraft()
        image.reset(for: .createImage)
        image.prompt = "a mug"
        let slots = StudioMode.createImage.attachmentSlots
        slots[0].attach([URL(fileURLWithPath: "/tmp/in.png")], to: &image)
        slots[1].attach([URL(fileURLWithPath: "/tmp/ref-a.png"), URL(fileURLWithPath: "/tmp/ref-b.jpg")], to: &image)
        let request = try StudioCommandAdapter.makeRequest(mode: .createImage, draft: image)
        let arguments = request.template.arguments(from: request.draft)
        XCTAssertTrue(arguments.contains("/tmp/in.png"))
        XCTAssertTrue(arguments.contains("/tmp/ref-a.png"))
        XCTAssertTrue(arguments.contains("/tmp/ref-b.jpg"))

        var find = StudioDraft()
        find.reset(for: .findObjects)
        find.prompt = "every coffee cup"
        StudioMode.findObjects.attachmentSlots[0].attach([URL(fileURLWithPath: "/tmp/mug.png")], to: &find)
        let findRequest = try StudioCommandAdapter.makeRequest(mode: .findObjects, draft: find)
        let findArguments = findRequest.template.arguments(from: findRequest.draft)
        XCTAssertTrue(findArguments.contains("/tmp/mug.png"))
        XCTAssertFalse(findArguments.contains("--threshold"), "vision ground has no threshold flag")

        var segment = StudioDraft()
        segment.reset(for: .segment)
        segment.prompt = "the cup"
        segment.visionThreshold = 0.3
        StudioMode.segment.attachmentSlots[0].attach([URL(fileURLWithPath: "/tmp/mug.png")], to: &segment)
        let segmentRequest = try StudioCommandAdapter.makeRequest(mode: .segment, draft: segment)
        let segmentArguments = segmentRequest.template.arguments(from: segmentRequest.draft)
        let thresholdIndex = try XCTUnwrap(segmentArguments.firstIndex(of: "--threshold"))
        XCTAssertEqual(segmentArguments[thresholdIndex + 1], "0.3")
    }

    func testListSlotAppendsWithoutDuplicatesAndSingleSlotReplaces() {
        var draft = StudioDraft()
        draft.reset(for: .music)
        let slots = StudioMode.music.attachmentSlots
        let source = slots[0]
        let timbre = slots[1]
        source.attach([URL(fileURLWithPath: "/tmp/a.wav")], to: &draft)
        source.attach([URL(fileURLWithPath: "/tmp/b.wav")], to: &draft)
        XCTAssertEqual(draft.musicSourceAudio, "/tmp/b.wav")
        timbre.attach([URL(fileURLWithPath: "/tmp/t1.wav")], to: &draft)
        timbre.attach([URL(fileURLWithPath: "/tmp/t1.wav"), URL(fileURLWithPath: "/tmp/t2.mp3")], to: &draft)
        XCTAssertEqual(timbre.paths(in: draft), ["/tmp/t1.wav", "/tmp/t2.mp3"])
        XCTAssertEqual(timbre.caption(in: draft), "t1.wav +1")
    }

    func testSlotsRejectFilesOfTheWrongKind() {
        var draft = StudioDraft()
        draft.reset(for: .video)
        let audio = StudioMode.video.attachmentSlots[2]
        audio.attach([URL(fileURLWithPath: "/tmp/frame.png")], to: &draft)
        XCTAssertTrue(draft.audioPath.isEmpty)
        audio.attach([URL(fileURLWithPath: "/tmp/track.wav")], to: &draft)
        XCTAssertEqual(draft.audioPath, "/tmp/track.wav")
        // Attaching LTX audio keeps the run producing audio + video, as the popover picker did.
        XCTAssertEqual(draft.videoOutputMode, .audioVideo)
        XCTAssertEqual(draft.videoQuality, .final)
    }

    func testReferenceVoiceSwitchesSpeakToCloneMode() {
        var draft = StudioDraft()
        draft.reset(for: .speak)
        XCTAssertEqual(draft.voiceMode, "style")
        StudioMode.speak.attachmentSlots[0].attach([URL(fileURLWithPath: "/tmp/me.wav")], to: &draft)
        XCTAssertEqual(draft.voiceMode, "clone")
        XCTAssertEqual(draft.refAudioPath, "/tmp/me.wav")
    }

    func testCanvasDropRoutesToTheFirstEmptySlotThatAcceptsTheFile() {
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        XCTAssertTrue(draft.attach(dropped: [URL(fileURLWithPath: "/tmp/one.png")], for: .createImage))
        XCTAssertEqual(draft.inputPath, "/tmp/one.png")
        XCTAssertTrue(draft.attach(dropped: [URL(fileURLWithPath: "/tmp/two.png")], for: .createImage))
        XCTAssertEqual(draft.inputPath, "/tmp/one.png")
        XCTAssertEqual(draft.referenceImagePaths, "/tmp/two.png")
        XCTAssertFalse(draft.attach(dropped: [URL(fileURLWithPath: "/tmp/song.wav")], for: .createImage))

        var code = StudioDraft()
        code.reset(for: .code)
        XCTAssertFalse(code.attach(dropped: [URL(fileURLWithPath: "/tmp/one.png")], for: .code))
    }

    func testChatImageSlotCollapsesToThePaperclipUntilFilled() {
        var draft = StudioDraft()
        draft.reset(for: .chat)
        XCTAssertFalse(StudioMode.chat.showsAttachmentWell(for: draft))
        StudioMode.chat.attachmentSlots[0].attach([URL(fileURLWithPath: "/tmp/photo.png")], to: &draft)
        XCTAssertTrue(StudioMode.chat.showsAttachmentWell(for: draft))

        var image = StudioDraft()
        image.reset(for: .createImage)
        XCTAssertTrue(StudioMode.createImage.showsAttachmentWell(for: image))
        var sfx = StudioDraft()
        sfx.reset(for: .sfx)
        XCTAssertFalse(StudioMode.sfx.showsAttachmentWell(for: sfx))
    }

    // MARK: - Chips

    func testEveryPromptModeShowsTwoToFourChipsEndingWithOrIncludingTheModel() {
        for mode in StudioMode.allCases {
            let chips = mode.composerChips
            XCTAssertTrue((1...4).contains(chips.count), "\(mode) shows \(chips.count) chips")
            XCTAssertTrue(chips.contains(.model), "\(mode) has no model chip")
            XCTAssertEqual(Set(chips).count, chips.count, "\(mode) repeats a chip")
        }
        XCTAssertEqual(StudioMode.createImage.composerChips, [.dimensions, .steps, .seed, .model])
        XCTAssertEqual(StudioMode.chat.composerChips, [.model, .thinking])
        XCTAssertEqual(StudioMode.findObjects.composerChips, [.model], "vision ground has no threshold")
        XCTAssertEqual(StudioMode.segment.composerChips, [.threshold, .model])
    }

    func testAspectPresetsRoundTripThroughTheDraft() throws {
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        let presets = StudioAspectPreset.presets(for: .createImage)
        let square = try XCTUnwrap(presets.first { $0.label == "1:1" })
        XCTAssertTrue(square.matches(draft))

        let wide = try XCTUnwrap(presets.first { $0.label == "16:9" })
        wide.apply(to: &draft)
        XCTAssertEqual(draft.width, 1344)
        XCTAssertEqual(draft.height, 768)
        XCTAssertTrue(wide.matches(draft))
        XCTAssertEqual(StudioComposerPresets.dimensionsTitle(draft), "1344 × 768")
        XCTAssertFalse(square.matches(draft))

        for preset in StudioAspectPreset.presets(for: .video) {
            XCTAssertEqual(preset.width % 32, 0, "\(preset.label) video width is not a multiple of 32")
            XCTAssertEqual(preset.height % 32, 0, "\(preset.label) video height is not a multiple of 32")
        }
        var video = StudioDraft()
        video.reset(for: .video)
        XCTAssertTrue(StudioAspectPreset.presets(for: .video).contains { $0.matches(video) })
    }

    func testSeedModesReadAndWriteTheDraftSeed() {
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        XCTAssertEqual(StudioSeedMode(draft: draft), .random)
        XCTAssertEqual(StudioSeedMode(draft: draft).chipTitle, "Seed random")

        StudioSeedMode.fixed(8812).apply(to: &draft)
        XCTAssertEqual(draft.seed, "8812")
        XCTAssertEqual(StudioSeedMode(draft: draft), .fixed(8812))
        XCTAssertEqual(StudioSeedMode(draft: draft).chipTitle, "Seed 8812")

        draft.seed = " 42 "
        XCTAssertEqual(StudioSeedMode(draft: draft), .fixed(42))
        draft.seed = "random"
        XCTAssertEqual(StudioSeedMode(draft: draft), .random)

        StudioSeedMode.random.apply(to: &draft)
        XCTAssertEqual(draft.seed, "")
    }

    func testChipTitlesFollowTheDraft() {
        var image = StudioDraft()
        image.reset(for: .createImage)
        XCTAssertEqual(StudioComposerPresets.stepsTitle(image, mode: .createImage), "4 steps")
        image.steps = 1
        XCTAssertEqual(StudioComposerPresets.stepsTitle(image, mode: .createImage), "1 step")

        var music = StudioDraft()
        music.reset(for: .music)
        XCTAssertEqual(StudioComposerPresets.stepsTitle(music, mode: .music), "Preset steps")
        XCTAssertEqual(StudioComposerPresets.durationTitle(music, mode: .music), "Preset length")
        music.useDuration = true
        music.durationSeconds = 90
        XCTAssertEqual(StudioComposerPresets.durationTitle(music, mode: .music), "90 s")

        var video = StudioDraft()
        video.reset(for: .video)
        XCTAssertEqual(StudioComposerPresets.durationTitle(video, mode: .video), "65 frames")

        var sfx = StudioDraft()
        sfx.reset(for: .sfx)
        XCTAssertEqual(StudioComposerPresets.durationTitle(sfx, mode: .sfx), "10 s")
        sfx.durationSeconds = 2.5
        XCTAssertEqual(StudioComposerPresets.durationTitle(sfx, mode: .sfx), "2.5 s")

        var segment = StudioDraft()
        segment.reset(for: .segment)
        XCTAssertEqual(StudioComposerPresets.thresholdTitle(segment), "Threshold 0.05")
        segment.visionThreshold = 0.3
        XCTAssertEqual(StudioComposerPresets.thresholdTitle(segment), "Threshold 0.3")

        XCTAssertEqual(StudioComposerPresets.thinkingTitle(.hide), "Thinking off")
        XCTAssertEqual(StudioComposerPresets.thinkingTitle(.automatic), "Thinking auto")
    }

    // MARK: - Model chip

    func testModelChoicesAreFilteredToTheModeCategoryWithInstalledFirst() {
        let inventory = [
            row("image-zimage-nano", category: "image", status: "installed"),
            row("image-flux2-klein", category: "image", status: "missing"),
            row("text-chat-qwen3.6-4b", category: "text-chat", status: "installed"),
            row("vision-chat-qwen3.6-vl-4b", category: "vision-chat", status: "installed"),
            row("vision-ground-falcon-perception", category: "vision-ground", status: "missing"),
            row("speech-tts-qwen3-nano", category: "speech-tts", status: "installed"),
            row("music-acestep", category: "music", status: "installed"),
        ]

        XCTAssertEqual(
            StudioMode.createImage.modelChoices(from: inventory).map(\.id),
            ["image-zimage-nano", "image-flux2-klein"]
        )
        XCTAssertEqual(
            StudioMode.chat.modelChoices(from: inventory).map(\.id),
            ["text-chat-qwen3.6-4b", "vision-chat-qwen3.6-vl-4b"]
        )
        XCTAssertEqual(StudioMode.findObjects.modelChoices(from: inventory).map(\.id), ["vision-ground-falcon-perception"])
        XCTAssertEqual(StudioMode.speak.modelChoices(from: inventory).map(\.id), ["speech-tts-qwen3-nano"])
        XCTAssertEqual(StudioMode.music.modelChoices(from: inventory).map(\.id), ["music-acestep"])
        XCTAssertTrue(StudioMode.video.modelChoices(from: inventory).isEmpty)
    }

    func testEveryModeDefaultModelFallsInsideItsOwnCategories() throws {
        for mode in StudioMode.allCases {
            let template = try XCTUnwrap(CommandCatalog.template(id: mode.defaultTemplateID))
            guard !template.defaultModel.isEmpty else { continue }
            // Managed model ids start with their `model list` category ("image-", "vision-ground-").
            XCTAssertTrue(
                mode.modelCategories.contains { template.defaultModel.hasPrefix($0 + "-") },
                "\(mode) default model \(template.defaultModel) is outside its chip categories"
            )
        }
    }

    func testDisplayModelNameDropsCategoryPrefixes() {
        XCTAssertEqual(StudioComposer.displayModelName("image-zimage-nano"), "Zimage Nano")
        XCTAssertEqual(StudioComposer.displayModelName("vision-ground-falcon-perception"), "Falcon Perception")
        XCTAssertEqual(StudioComposer.displayModelName("speech-tts-qwen3-nano"), "Qwen3 Nano")
        XCTAssertEqual(StudioComposer.displayModelName("text-agent-deepseek-v4-flash"), "Deepseek V4 Flash")
    }

    // MARK: - Helpers

    private static func sampleExtension(for slot: StudioAttachmentSlot) -> String {
        if slot.acceptedTypes.contains(where: { UTType.png.conforms(to: $0) }) { return "png" }
        if slot.acceptedTypes.contains(where: { UTType.wav.conforms(to: $0) }) { return "wav" }
        return "mp4"
    }

    private func row(_ id: String, category: String, status: String) -> StudioModelInventoryRow {
        StudioModelInventoryRow(id: id, category: category, status: status, size: "1 GB", usageTerms: nil)
    }
}
