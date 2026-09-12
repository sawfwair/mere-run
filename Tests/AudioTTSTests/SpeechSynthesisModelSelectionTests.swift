import AudioTTS
import Foundation
import XCTest

final class SpeechSynthesisModelSelectionTests: XCTestCase {
    func testManagedDefaultsAndSupportedModelsResolveWithoutLoading() throws {
        XCTAssertEqual(try SpeechSynthesisModelSelection.resolve("  ").modelID, "speech-tts-qwen3-nano")
        for id in ["speech-tts-qwen3-nano", "speech-tts-qwen3-customvoice"] {
            let selected = try SpeechSynthesisModelSelection.resolve("  \(id)  ")
            XCTAssertEqual(selected.modelID, id)
            XCTAssertNil(selected.modelPath)
        }
    }

    func testLocalPathsRetainTheSharedDefaultIDAndNormalizedPath() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let selected = try SpeechSynthesisModelSelection.resolve(directory.path + "/./")
        XCTAssertEqual(selected.modelID, "speech-tts-qwen3-nano")
        XCTAssertEqual(selected.modelPath, directory.standardizedFileURL.path)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
    }

    func testNonTTSModelsAndWireOnlyAliasesDoNotReachTheNativeRuntime() {
        for selector in ["tts-1", "speech-asr-qwen3", "unknown-speech-fixture"] {
            XCTAssertThrowsError(try SpeechSynthesisModelSelection.resolve(selector))
        }
    }
}
