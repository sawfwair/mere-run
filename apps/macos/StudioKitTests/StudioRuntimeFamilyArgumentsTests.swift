@testable import StudioKit
import MereRunContract
import XCTest

/// The audio, sound-effect, and OCR builders send only the options the selected runtime reads,
/// as the capability contract scopes them, and the command lines they build pass the CLI's gate.
final class StudioRuntimeFamilyArgumentsTests: XCTestCase {
    private func flags(_ id: CommandTemplateID, _ edit: (inout CommandDraft) -> Void) throws -> Set<String> {
        let template = try XCTUnwrap(CommandCatalog.template(id: id))
        var draft = template.defaultDraft()
        edit(&draft)
        let arguments = CommandArguments.build(for: id, draft: draft)
        let capability = try XCTUnwrap(id.capability)
        let invocation = MereRunCommandInvocation(capability: capability, arguments: Array(arguments.dropFirst(capability.command.count)))
        let report = capability.resolutionReport(invocation)
        XCTAssertEqual(report.violations + report.warnings, [], "\(id): \(arguments)")
        return Set(invocation.values.keys)
    }

    func testSoundEffectBuildersSendWhatEachRuntimeReads() throws {
        let woosh = try flags(.sfxGenerate) { draft in
            draft.secondaryText = "music"
            draft.sfxRenoise = "0.5"
        }
        XCTAssertTrue(woosh.contains("--renoise") && !woosh.contains("--negative-prompt"))
        let mmaudio = try flags(.sfxGenerate) { draft in
            draft.model = "sfx-mmaudio-large-44k-v2"
            draft.secondaryText = "music"
            draft.sfxRenoise = "0.5"
        }
        XCTAssertTrue(mmaudio.contains("--negative-prompt") && !mmaudio.contains("--renoise"))

        let foley = try flags(.sfxVideo) { draft in
            draft.model = "sfx-mmaudio-large-44k-v2"
            draft.secondaryText = "music"
            draft.sfxSynchformerModel = "sfx-woosh-synchformer"
        }
        XCTAssertTrue(foley.isSuperset(of: ["--negative-prompt", "--clip-batch-size"]))
        XCTAssertFalse(foley.contains("--synchformer-model"))
    }

    func testAudioBuildersLeaveOffWhatTheRuntimeFixesOrIgnores() throws {
        let apBWE = try flags(.audioEnhance) { draft in
            draft.audioInputRate = 16_000
            draft.audioODESteps = 6
            draft.seed = "7"
        }
        XCTAssertFalse(apBWE.contains { ["--input-rate", "--ode-steps", "--seed"].contains($0) })
        let univerSR = try flags(.audioEnhance) { draft in
            draft.model = "audio-enhance-universr-audio"
            draft.audioOverlap = 4
            draft.audioInputRate = 12_000
        }
        XCTAssertTrue(univerSR.contains("--input-rate") && !univerSR.contains("--overlap"))

        let flash = try flags(.audioEdit) { draft in
            draft.model = "audio-auk-flash"
            draft.useDuration = true
            draft.audioGuidanceScale = 3
        }
        XCTAssertFalse(flash.contains("--steps") || flash.contains("--guidance"))
    }

    func testOCRSendsTheSelectedBackendsOptions() throws {
        let glm = try flags(.visionOCR) { draft in
            draft.backend = "glm"
            draft.visionGLMConfig = "/tmp/glm.yaml"
        }
        XCTAssertTrue(glm.contains("--glm-config"))
        XCTAssertTrue(glm.isDisjoint(with: ["--model", "--max-tokens", "--temperature", "--infinity-model"]))

        let external = try flags(.visionOCR) { draft in
            draft.backend = "infinity"
            draft.visionInfinityRuntime = "external"
            draft.all = true
        }
        XCTAssertTrue(external.isSuperset(of: ["--model", "--infinity-api-url", "--infinity-model", "--compare"]))
        XCTAssertFalse(external.contains("--glmocr-cli"))
    }
}
