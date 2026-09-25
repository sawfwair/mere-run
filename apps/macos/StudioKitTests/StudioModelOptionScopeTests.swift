@testable import StudioKit
import MereRunContract
import XCTest

final class StudioModelOptionScopeTests: XCTestCase {
    func testMusicModelChangesTheControlsAndComposerAttachments() throws {
        let capability = try XCTUnwrap(CommandTemplateID.musicGenerate.capability)
        func flags(_ model: String) -> Set<String> {
            Set(StudioModelOptionScope.options(for: capability, model: model).map(\.flag))
        }
        XCTAssertTrue(flags("music-yue2").contains("--score-mode"))
        XCTAssertFalse(flags("music-yue2").contains("--quality"))
        XCTAssertFalse(flags("music-yue2").contains("--source-audio"))
        XCTAssertTrue(flags("music-acestep").contains("--quality"))
        XCTAssertFalse(flags("music-acestep").contains("--score-mode"))
        XCTAssertTrue(flags("music-minimax-music3").contains("--compose"))
        XCTAssertFalse(flags("music-minimax-music3").contains("--quality"))
        XCTAssertTrue(flags("music-magenta-rt2-small").contains("--seed-rotation"))
        XCTAssertFalse(flags("music-magenta-rt2-small").contains("--steps"))

        var draft = StudioDraft()
        draft.model = "music-yue2"
        XCTAssertTrue(StudioMode.music.attachmentSlots(for: draft).isEmpty)
        let inspectorFlags = Set(StudioContractSchema.fields(for: .music, draft: draft).map(\.flag))
        XCTAssertFalse(inspectorFlags.contains("--quality"))
        draft.model = "music-acestep"
        XCTAssertEqual(StudioMode.music.attachmentSlots(for: draft).map(\.id), ["source", "timbre"])
        XCTAssertTrue(StudioContractSchema.fields(for: .music, draft: draft).contains { $0.flag == "--quality" })
    }

    func testSwitchingModelsKeepsOldValuesButDoesNotLaunchThem() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .musicGenerate))
        let seed = template.defaultDraft()
        var form = StudioConsoleCommand.seed(template: template, draft: seed)
        form["--model"] = .text("music-yue2")
        form["--quality"] = .text("song")
        form["--source-audio"] = .text("/tmp/source.wav")
        form["--score-mode"] = .text("full")

        let groups = StudioConsoleCommand.groups(for: try XCTUnwrap(template.id.capability), model: "music-yue2")
        let visible = Set(groups.flatMap(\.fields).map(\.flag))
        XCTAssertTrue(visible.contains("--score-mode"))
        XCTAssertFalse(visible.contains("--quality"))
        XCTAssertFalse(visible.contains("--source-audio"))

        let run = try XCTUnwrap(StudioConsoleRun(template: template, draft: form, seed: seed))
        XCTAssertTrue(run.arguments.contains("--score-mode"))
        XCTAssertFalse(run.arguments.contains("--quality"))
        XCTAssertFalse(run.arguments.contains("--source-audio"))
        XCTAssertEqual(form.text("--source-audio"), "/tmp/source.wav")
    }

    func testOtherModelFamiliesScopeTaskFormsAndCommands() throws {
        func flags(_ id: String, _ model: String) throws -> Set<String> {
            let capability = try XCTUnwrap(MereRunCapabilityCatalog.command(id: id))
            return Set(StudioModelOptionScope.options(for: capability, model: model).map(\.flag))
        }
        XCTAssertTrue(try flags("video.generate", "video-wan22-ti2v-5b-mlx").contains("--shift"))
        XCTAssertFalse(try flags("video.generate", "video-wan22-ti2v-5b-mlx").contains("--ltx-preset"))
        XCTAssertTrue(try flags("video.generate", "video-minimax-h3-fl2va-mlx").contains("--h3-weight-mode"))
        XCTAssertFalse(try flags("video.generate", "video-minimax-h3-fl2va-mlx").contains("--quality"))
        XCTAssertTrue(try flags("video.generate", "video-ltx25-full-bf16").contains("--ltx-preset"))
        XCTAssertFalse(try flags("video.generate", "video-ltx25-full-bf16").contains("--h3-frame"))
        XCTAssertFalse(try flags("image.generate", "image-zimage-nano").contains("--krea-conditioning-multiplier"))
        XCTAssertTrue(try flags("image.generate", "image-krea2-turbo").contains("--krea-conditioning-multiplier"))
        XCTAssertFalse(try flags("sfx.generate", "sfx-woosh-dflow").contains("--negative-prompt"))
        XCTAssertFalse(try flags("sfx.generate", "sfx-mmaudio-large-44k-v2").contains("--renoise"))
        XCTAssertFalse(try flags("audio.enhance", "audio-enhance-ap-bwe-16kto48k").contains("--ode-method"))
        XCTAssertTrue(try flags("audio.enhance", "audio-enhance-ap-bwe-16kto48k").contains("--input-rate"))
        XCTAssertTrue(try flags("audio.enhance", "audio-enhance-ap-bwe-16kto48k").contains("--dtype"))
        XCTAssertFalse(try flags("audio.enhance", "audio-enhance-universr").contains("--overlap"))
        XCTAssertFalse(try flags("speech.transcribe", "speech-asr-qwen3").contains("--provider"))

        var task = StudioTaskDraft(templateID: .audioEnhance)
        task.model = "audio-enhance-ap-bwe-16kto48k"
        XCTAssertFalse(StudioTaskSchema.fields(for: .audioEnhance, draft: task).contains { $0.flag == "--ode-method" })
    }
}
