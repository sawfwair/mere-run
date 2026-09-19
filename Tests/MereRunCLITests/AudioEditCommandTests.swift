import XCTest
import MereRunCore
@testable import MereRunCLI

final class AudioEditCommandTests: XCTestCase {
    func testTextOnlyRequiresDurationAndReferenceCanInferIt() throws {
        XCTAssertThrowsError(try AudioEdit.parse(["Say hello"]))
        let text = try AudioEdit.parse(["Say hello", "--duration", "3"])
        XCTAssertEqual(text.options.variant, .base)
        XCTAssertEqual(text.duration, 3)
        let reference = try AudioEdit.parse(["Remove noise", "--audio", "input.wav", "--model", "audio-auk-flash"])
        XCTAssertEqual(reference.options.variant, .flash)
        XCTAssertNil(reference.duration)
    }

    func testRejectsInvalidNumbersModelsAndInputOverwrite() {
        for arguments in [
            ["Say hello", "--duration", "0"], ["Say hello", "--duration", "nan"],
            ["Say hello", "--duration", "301"], ["Say hello", "--duration", "1", "--steps", "0"],
            ["Say hello", "--duration", "1", "--guidance", "-1"],
            ["Say hello", "--duration", "1", "--model", "unknown"],
            ["Remove noise", "--audio", "input.wav", "--output", "input.wav"],
            ["  ", "--duration", "1"]
        ] { XCTAssertThrowsError(try AudioEdit.parse(arguments), arguments.joined(separator: " ")) }
    }

    func testAuKAdmissionIsStandardForDefaultAndLocalCheckpoints() {
        for arguments in [
            ["mere.run", "audio", "edit", "Say hello", "--duration", "1"],
            ["mere.run", "audio", "edit", "Say hello", "--duration", "1", "--model-path", "/tmp/auk"]
        ] {
            XCTAssertEqual(CLIInferenceAdmissionClassifier.request(arguments: arguments)?.resourceClass, .standard)
        }
    }

    func testPromptMatchesUpstreamMarkers() {
        let plain = AuKGenerator.prompt(instruction: "hello", hasAudio: false)
        XCTAssertTrue(plain.contains("hello|<no_prompt_audio>|<|im_end|>"))
        XCTAssertTrue(plain.hasSuffix("<|im_start|>assistant\n"))
        let audio = AuKGenerator.prompt(instruction: "hello", hasAudio: true)
        XCTAssertTrue(audio.contains("hello<|audio_bos|><|AUDIO|><|audio_eos|>"))
        XCTAssertFalse(audio.contains("no_prompt_audio"))
    }

    func testManagedModelsHavePinnedDownloadsAndHandbook() throws {
        for model in ["audio-auk-base", "audio-auk-flash", "audio-auk-thinker"] {
            let spec = try XCTUnwrap(ManagedModelCatalog.allSpecs.first { $0.id == model })
            XCTAssertNotNil(spec.hubFallback)
            XCTAssertEqual(spec.apiAvailability, .cliOnly)
            XCTAssertGreaterThan(spec.estimatedDownloadBytes ?? 0, 0)
            if model != "audio-auk-thinker" { XCTAssertEqual(spec.companionModelIDs, ["audio-auk-thinker"]) }
            if model == "audio-auk-thinker" {
                XCTAssertEqual(spec.hubFallback?.patterns,
                               ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "*.json", "LICENSE"])
                XCTAssertEqual(spec.estimatedDownloadBytes, 10_009_109_352)
            }
            XCTAssertEqual(try ModelGuideRegistry.guide(for: model).topic, "handbook-auk")
        }
    }
}
