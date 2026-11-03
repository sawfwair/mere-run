@testable import StudioKit
import Foundation
import MereRunContract
import XCTest

final class StudioSamplingDefaultsTests: XCTestCase {
    func testFreshTemplatesLeaveSamplingToTheCLI() {
        for template in CommandCatalog.templates {
            guard let capability = template.id.capability else { continue }
            let argv = template.arguments(from: template.defaultDraft(), source: .contract)
            for option in capability.options where StudioSamplingDefaults.applies(to: option.flag, in: capability) {
                XCTAssertFalse(argv.contains(option.flag), "\(template.id): inherited \(option.flag)")
            }
        }
    }

    func testEveryManagedImageModelUsesItsOwnRecipeAfterSwitching() throws {
        var draft = StudioDraft.baseline(for: .createImage)
        draft.prompt = "A red ceramic mug"
        let models = try XCTUnwrap(MereRunCapabilityCatalog.imageGenerate.routing).families.flatMap(\.models)
        XCTAssertTrue(models.contains("image-qwen-edit-2511"))
        XCTAssertTrue(models.contains("image-qwen-edit-2511-lightning"))
        for model in models {
            draft.model = model
            let argv = try arguments(draft, mode: .createImage)
            XCTAssertFalse(argv.contains("--steps"), model)
            XCTAssertFalse(argv.contains("--cfg"), model)
            XCTAssertEqual(StudioComposerPresets.stepsTitle(draft, mode: .createImage), "Model steps")
        }
    }

    func testModelAndQualitySwitchesDoNotIntroduceInheritedSamplingArguments() {
        for template in CommandCatalog.templates {
            guard let capability = template.id.capability else { continue }
            let models = capability.routing?.families.flatMap(\.models) ?? [template.defaultModel]
            for model in models {
                var draft = template.defaultDraft()
                draft.model = model
                for profile in [draft.renderProfile, "quality"] {
                    draft.renderProfile = profile
                    let argv = template.arguments(from: draft, source: .contract)
                    for option in capability.options where StudioSamplingDefaults.applies(to: option.flag, in: capability) {
                        XCTAssertFalse(argv.contains(option.flag), "\(template.id) \(model) \(profile): \(option.flag)")
                    }
                }
            }
        }
    }

    func testExplicitFourStepsAndCFGOneSurviveSwitchesAndSerialization() throws {
        var draft = StudioDraft.baseline(for: .createImage)
        let bindings = StudioContractBindings.bindings(for: .createImage)
        bindings["--steps"]?.write(&draft, .integer(4))
        bindings["--cfg"]?.write(&draft, .number(1))
        draft = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(draft))
        for model in ["image-qwen-edit-2511", "image-qwen-edit-2511-lightning", "image-klein-nano"] {
            draft.model = model
            let argv = try arguments(draft, mode: .createImage)
            XCTAssertEqual(value("--steps", in: argv), "4", model)
            if model == "image-qwen-edit-2511" { XCTAssertEqual(value("--cfg", in: argv), "1", model) }
        }
        bindings["--steps"]?.write(&draft, .unset)
        bindings["--cfg"]?.write(&draft, .unset)
        draft.model = "image-qwen-edit-2511"
        let automatic = try arguments(draft, mode: .createImage)
        XCTAssertFalse(automatic.contains("--steps"))
        XCTAssertFalse(automatic.contains("--cfg"))
        bindings["--steps"]?.write(&draft, .integer(4))
        draft.reset(for: .createImage)
        XCTAssertEqual(bindings["--steps"]?.read(draft), .unset)
    }

    func testCustomValuesAndSavedPageDefaultsRemainExplicit() throws {
        var draft = StudioDraft.baseline(for: .createImage)
        draft.model = "image-qwen-edit-2511"
        draft.steps = 28
        draft.cfgScale = 3.5
        let defaults = StudioPageDefaults(capturing: draft, mode: .createImage, source: .contract)
        var restored = StudioDraft.baseline(for: .createImage)
        defaults.apply(to: &restored, mode: .createImage)
        restored.model = "image-qwen-edit-2511"
        let argv = try arguments(restored, mode: .createImage)
        XCTAssertEqual(value("--steps", in: argv), "28")
        XCTAssertEqual(value("--cfg", in: argv), "3.5")
    }

    func testRecordedCommandsKeepExplicitValuesEqualToOldDefaults() throws {
        let capability = MereRunCapabilityCatalog.imageGenerate
        let argv = ["image", "generate", "--prompt", "mug", "--model", "image-qwen-edit-2511", "--steps", "4", "--cfg", "1"]
        let form = StudioConsoleCommand.seed(capability: capability, arguments: argv)
        let rebuilt = StudioConsoleCommand.arguments(for: capability, draft: form)
        XCTAssertEqual(value("--steps", in: rebuilt), "4")
        XCTAssertEqual(value("--cfg", in: rebuilt), "1")
        var restored = StudioDraft.baseline(for: .createImage)
        form.applyingChanges(from: StudioConsoleDraft(), to: &restored, mode: .createImage, templateID: .imageGenerate)
        let prompt = try arguments(restored, mode: .createImage)
        XCTAssertEqual(value("--steps", in: prompt), "4")
        XCTAssertEqual(value("--cfg", in: prompt), "1")
    }

    func testLegacyFormsMigrateInheritedValuesButKeepCustomValues() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        let capability = MereRunCapabilityCatalog.imageGenerate
        var form = StudioConsoleCommand.seed(capability: capability,
            arguments: CommandArguments.build(for: .imageGenerate, draft: template.defaultDraft(), source: .contract))
        form.modelSamplingDefaults = nil
        form["--model"] = .text("image-qwen-edit-2511")
        var migrated = StudioSamplingDefaults.migrating(form, capability: capability)
        XCTAssertEqual(migrated["--steps"], .unset)
        form["--steps"] = .integer(28)
        migrated = StudioSamplingDefaults.migrating(form, capability: capability)
        XCTAssertEqual(migrated["--steps"], .integer(28))
    }

    func testLegacyLibraryFallbackPreservesWhatTheOldBuilderEmitted() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var recorded = template.defaultDraft()
        recorded.model = "image-qwen-edit-2511"
        recorded.steps = 4
        recorded.cfgScale = 1
        recorded.samplingOverrides = nil
        let argv = StudioLibraryDraftRestoration.recordedArguments(recorded, template: template, source: .contract)
        XCTAssertEqual(value("--steps", in: argv), "4")
        XCTAssertNil(value("--cfg", in: argv))
    }

    func testAutomaticCommandSnapshotsDoNotReintroduceZeroStepPlaceholders() throws {
        for id in [CommandTemplateID.imageGenerate, .videoGenerate, .sfxGenerate] {
            let template = try XCTUnwrap(CommandCatalog.template(id: id))
            let capability = try XCTUnwrap(id.capability)
            var form = StudioConsoleCommand.seed(template: template, draft: template.defaultDraft(), source: .contract)
            if id == .imageGenerate { form["--model"] = .text("image-qwen-edit-2511") }
            if id == .videoGenerate { form["--model"] = .text("video-wan22-ti2v-5b-mlx") }
            let command = StudioConsoleCommand.commandDraft(seed: template.defaultDraft(), template: template, capability: capability, draft: form)
            let argv = StudioLibraryDraftRestoration.recordedArguments(command, template: template, source: .contract)
            XCTAssertFalse(argv.contains("--steps"), "\(id)")
            XCTAssertFalse(argv.contains("--cfg"), "\(id)")
            XCTAssertFalse(argv.contains("--guidance-scale"), "\(id)")
        }
    }

    func testLegacyPromptDraftTreatsInheritedFourStepsAsAutomatic() throws {
        var draft = StudioDraft.baseline(for: .createImage)
        draft.model = "image-qwen-edit-2511"
        draft.samplingOverrides = nil
        draft = try JSONDecoder().decode(StudioDraft.self, from: JSONEncoder().encode(draft))
        XCTAssertNil(draft.samplingOverrides)
        XCTAssertFalse(try arguments(draft, mode: .createImage).contains("--steps"))
        draft.steps = 28
        XCTAssertEqual(value("--steps", in: try arguments(draft, mode: .createImage)), "28")
    }

    func testMusicPresetOverrideAndChatTemperatureAreExplicitWhenChosen() throws {
        var music = StudioDraft.baseline(for: .music)
        music.model = "music-acestep-xl-sft"
        music.musicOverrideSteps = true
        XCTAssertEqual(StudioComposerPresets.stepsTitle(music, mode: .music), "8 steps")
        XCTAssertEqual(value("--steps", in: try arguments(music, mode: .music)), String(music.steps))
        var chat = StudioDraft.baseline(for: .chat)
        let binding = try XCTUnwrap(StudioContractBindings.bindings(for: .chat)["--temperature"])
        XCTAssertEqual(binding.read(chat), .unset)
        binding.write(&chat, .number(chat.temperature))
        XCTAssertNotNil(value("--temperature", in: try arguments(chat, mode: .chat)))
        binding.write(&chat, .number(0))
        XCTAssertEqual(value("--temperature", in: try arguments(chat, mode: .chat)), "0")
    }

    func testOptionalSamplingValuesAreNotMistakenForInheritedDefaults() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .audioEnhance))
        var draft = template.defaultDraft()
        draft.model = "/tmp/custom-enhancer"
        draft.audioGuidanceScale = 2
        draft.audioODESteps = 5
        let argv = template.arguments(from: draft, source: .contract)
        XCTAssertEqual(value("--guidance-scale", in: argv), "2")
        XCTAssertEqual(value("--ode-steps", in: argv), "5")
    }

    func testTrainingBudgetsAndExplicitMusicShiftArePreserved() throws {
        let training = try XCTUnwrap(CommandCatalog.template(id: .musicTrainAdapter))
        let trainingArgv = training.arguments(from: training.defaultDraft(), source: .contract)
        XCTAssertEqual(value("--steps", in: trainingArgv), "1000")
        let music = try XCTUnwrap(CommandCatalog.template(id: .musicGenerate))
        var draft = music.defaultDraft()
        draft.model = "music-acestep-xl-sft"
        draft.musicShift = "2"
        XCTAssertEqual(value("--shift", in: music.arguments(from: draft, source: .contract)), "2")
    }

    private func arguments(_ draft: StudioDraft, mode: StudioMode) throws -> [String] {
        let template = try XCTUnwrap(CommandCatalog.template(id: mode.defaultTemplateID))
        let command = StudioCommandAdapter.commandDraft(mode: mode, draft: draft, template: template)
        return template.arguments(from: command, source: .contract)
    }

    private func value(_ flag: String, in argv: [String]) -> String? {
        guard let index = argv.firstIndex(of: flag), index + 1 < argv.count else { return nil }
        return argv[index + 1]
    }
}
