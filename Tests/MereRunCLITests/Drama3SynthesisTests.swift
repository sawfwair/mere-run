import AudioTTS
import Foundation
import MereRunCore
import XCTest
@testable import MereRunCLI

final class Drama3SynthesisTests: XCTestCase {
    func testManagedAndLocalRoutingAndLocalOnlyAPI() throws {
        let selection = try SpeechSynthesisModelSelection.resolve("speech-tts-drama-3")
        XCTAssertEqual(selection.backend, .drama3)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(#"{"model_type":"fish_omni"}"#.utf8).write(to: root.appendingPathComponent("config.json"))
        XCTAssertEqual(try SpeechSynthesisModelSelection.resolve(root.path).backend, .drama3)
        for model in ["speech-tts-drama-3", root.path] {
            let plan = try APIServerContract.speechPlan(from: OpenAIAudioSpeechRequest(model: model, input: "Hello"))
            XCTAssertThrowsError(try plan.modelSelection())
            let report = try XCTUnwrap(CLICapabilityGate.evaluate(commandLine: [
                "speech", "synthesize", "<|speaker:0|> [laughs] Hello", "--model", model,
                "--output", "scene.wav", "--seed", "42", "--stream"
            ])).report
            XCTAssertEqual(report.family, "drama3")
            XCTAssertTrue(report.violations.isEmpty)
            let clone = try XCTUnwrap(CLICapabilityGate.evaluate(commandLine: [
                "speech", "synthesize", "Hello", "--model", model, "--mode", "clone",
                "--ref-audio", "reference.wav", "--ref-text", "Exact words", "--output", "clone.wav"
            ])).report
            XCTAssertEqual(clone.family, "drama3-clone")
            XCTAssertTrue(clone.violations.isEmpty)
        }
    }

    func testPinnedLicensedCatalogAndMemoryAdmission() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "speech-tts-drama-3"))
        XCTAssertEqual(spec.upstreamRevision, "dd64a85cd82e9521f7202610ede97b367fc83946")
        XCTAssertEqual(spec.validationKind, .drama3TTS)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertNotNil(spec.usageRestriction)
        XCTAssertEqual(spec.apiAvailability, .cliOnly)
        XCTAssertEqual(ManagedModelCapabilityCatalog.descriptor(for: spec.id)?.minimumUnifiedMemoryGB, 32)
        XCTAssertEqual(ManagedModelCapabilityCatalog.descriptor(for: spec.id)?.recommendedUnifiedMemoryGB, 36)
        let manifest = MereRunModelManifest.template(for: .drama3TTS)
        XCTAssertEqual(manifest.engine, .drama3TTS)
        XCTAssertEqual(CLIInferenceAdmissionClassifier.request(arguments: [
            "mere.run", "speech", "synthesize", "Hello", "--model", spec.id, "--output", "out.wav"
        ]), MachineInferenceRequest(label: "speech synthesize", resourceClass: .standard))
    }
}
