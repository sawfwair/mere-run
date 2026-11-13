@testable import StudioKit
import MereRunContract
import XCTest

final class StudioSelectionSchemaTests: XCTestCase {
    func testEveryTaskVariantIsReachableThroughItsOperationField() throws {
        for task in StudioTask.taskDraftTasks where task.variantTemplates.count > 1 {
            let draft = try XCTUnwrap(StudioTaskDraft(task: task))
            let fields = StudioTaskSchema.operationFields(for: task, draft: draft, source: .contract)
            let operation = try XCTUnwrap(fields.first { $0.overrideID == .variant }, "\(task)")
            XCTAssertEqual(operation.option.choices, task.variantTemplates.map(\.id.rawValue))
            for template in task.variantTemplates {
                var switched = draft
                operation.write(.text(template.id.rawValue), to: &switched)
                XCTAssertEqual(switched.templateID, template.id)
                let path = try XCTUnwrap(template.id.capability).command
                XCTAssertEqual(Array(switched.arguments(source: .contract).prefix(path.count)), path)
            }
        }
    }

    func testOCRModelPickerFollowsBackendAndWritesItsActualArgument() throws {
        let cases = [
            (backend: "lighton", flag: "--model", model: "vision-ocr-lighton-3-0.8b"),
            (backend: "infinity", flag: "--infinity-model", model: "vision-ocr-infinity-pro"),
            (backend: "unlimited", flag: "--unlimited-model", model: "vision-ocr-unlimited-2"),
        ]
        for item in cases {
            var draft = StudioTaskDraft(templateID: .visionOCR)
            draft.form["--backend"] = .text(item.backend)
            draft.form["--infinity-runtime"] = .text("native")
            XCTAssertEqual(StudioTaskSchema.modelPickerFlag(for: draft, source: .contract), item.flag)
            let fields = StudioTaskSchema.fields(for: .visionDocument, draft: draft, source: .contract)
            let picker = try XCTUnwrap(fields.first { $0.overrideID == .model })
            XCTAssertEqual(picker.flag, item.flag)
            XCTAssertNotEqual(picker.tier, .expert, "the active model must not require Advanced")
            picker.write(.text(item.model), to: &draft)
            XCTAssertEqual(draft.text(item.flag), item.model)
            XCTAssertEqual(StudioTaskSchema.modelID(for: draft, source: .contract), item.model)
            let models = try XCTUnwrap(StudioTaskSchema.modelScope(for: draft, source: .contract).runnableModels)
            XCTAssertTrue(models.contains(item.model))
            if item.backend != "lighton" { XCTAssertFalse(models.contains("vision-ocr-lighton")) }
            let invocation = MereRunCommandInvocation(capability: try XCTUnwrap(draft.capability),
                                                     arguments: Array(draft.arguments(source: .contract).dropFirst(2)))
            XCTAssertEqual(invocation.value(item.flag), item.model)
        }
    }

    func testExternalOCRAndModelFreeCommandsHaveNoManagedModelMenu() {
        var draft = StudioTaskDraft(templateID: .visionOCR)
        draft.form["--backend"] = .text("glm")
        draft.model = "vision-ocr-lighton"
        XCTAssertNil(StudioTaskSchema.modelPickerFlag(for: draft, source: .contract))
        XCTAssertEqual(StudioTaskSchema.modelID(for: draft, source: .contract), "")
        XCTAssertNil(StudioTaskSchema.requirement(for: draft, source: .contract))
        for template in [CommandTemplateID.imageDatasetDiscover, .imageRunPlan, .visionImageTo3DIdol] {
            XCTAssertNil(StudioTaskSchema.modelPickerFlag(for: StudioTaskDraft(templateID: template), source: .contract))
        }
    }

    func testOCRBackendAndRuntimeRemainVisibleAsOperationChoices() {
        var draft = StudioTaskDraft(templateID: .visionOCR)
        let initial = StudioTaskSchema.operationFields(for: .visionDocument, draft: draft, source: .contract)
        XCTAssertTrue(initial.contains { $0.flag == "--backend" })
        XCTAssertFalse(initial.contains { $0.flag == "--infinity-runtime" })
        draft.form["--backend"] = .text("infinity")
        let infinity = StudioTaskSchema.operationFields(for: .visionDocument, draft: draft, source: .contract)
        XCTAssertTrue(infinity.contains { $0.flag == "--infinity-runtime" }, "runtime choice is visible without Advanced")
    }

    func testPromptModelsFollowVoiceModeAndForcedTranscriptionBackend() throws {
        var voice = StudioDraft()
        voice.reset(for: .speak)
        voice.voiceMode = "style"
        voice.model = "speech-tts-qwen3-customvoice"
        let style = StudioModelScope(mode: .speak, draft: voice, source: .contract)
        XCTAssertTrue(style.runnableModels?.contains("speech-tts-qwen3-customvoice") == true)
        XCTAssertEqual(StudioScopeSource.contract.scope(mode: .speak, draft: voice)?.family?.id, "custom-voice")
        voice.voiceMode = "clone"
        let clone = StudioModelScope(mode: .speak, draft: voice, source: .contract)
        XCTAssertTrue(clone.runnableModels?.contains("speech-tts-qwen3-customvoice") == true)
        XCTAssertEqual(StudioScopeSource.contract.scope(mode: .speak, draft: voice)?.family?.id, "clone")

        var speech = StudioDraft()
        speech.reset(for: .listen)
        speech.backend = "qwen"
        let forced = try XCTUnwrap(StudioModelScope(mode: .listen, draft: speech, source: .contract).runnableModels)
        XCTAssertTrue(forced.contains("speech-asr-qwen3"))
        XCTAssertFalse(forced.contains("speech-asr-parakeet-tdt"))
    }

    func testChangingBackendDropsAnIgnoredModelAndNamesTheNewDefault() {
        var previous = StudioDraft()
        previous.reset(for: .listen)
        previous.backend = "auto"
        previous.model = "speech-asr-parakeet-tdt"
        var next = previous
        next.backend = "qwen"
        next.reconcileModelAfterOperationChange(from: previous, mode: .listen, source: .contract)
        XCTAssertEqual(next.model, "")
        XCTAssertEqual(StudioModelScope(mode: .listen, draft: next, source: .contract).defaultModelID, "speech-asr-qwen3")
        previous.model = "speech-asr-qwen3"
        next = previous
        next.backend = "qwen"
        next.reconcileModelAfterOperationChange(from: previous, mode: .listen, source: .contract)
        XCTAssertEqual(next.model, "speech-asr-qwen3", "a compatible explicit choice stays selected")
        previous.model = "/tmp/local-model"
        next = previous
        next.backend = "qwen"
        next.reconcileModelAfterOperationChange(from: previous, mode: .listen, source: .contract)
        XCTAssertEqual(next.model, previous.model, "local models are identified by the CLI")
    }
}
