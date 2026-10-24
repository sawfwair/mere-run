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

    func testNativeFamiliesRemainDistinctForManagedAndLocalSelectors() throws {
        struct Config: Encodable { let model_type: String }
        let cases: [(id: String, type: String, backend: SpeechSynthesisModelSelection.Backend)] = [
            ("speech-tts-drama-3", "fish_omni", .drama3),
            ("speech-tts-phonon-31m", "gradium_phonon_tts", .phonon),
            ("speech-tts-breeze-2", "breeze", .breeze)
        ]
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        for fixture in cases {
            let managed = try SpeechSynthesisModelSelection.resolve(fixture.id)
            XCTAssertEqual(managed.backend, fixture.backend)
            XCTAssertEqual(managed.modelID, fixture.id)
            XCTAssertNil(managed.modelPath)
            let local = root.appendingPathComponent(fixture.id, isDirectory: true)
            try FileManager.default.createDirectory(at: local, withIntermediateDirectories: true)
            try JSONEncoder().encode(Config(model_type: fixture.type)).write(to: local.appendingPathComponent("config.json"))
            let selected = try SpeechSynthesisModelSelection.resolve(local.path)
            XCTAssertEqual(selected.backend, fixture.backend)
            XCTAssertEqual(selected.modelID, fixture.id)
            XCTAssertEqual(selected.modelPath, local.path)
        }
    }

    func testLocalPathsRetainTheSharedDefaultIDAndNormalizedPath() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"tts_model_type":"base"}"#.utf8).write(to: directory.appendingPathComponent("config.json"))
        let selected = try SpeechSynthesisModelSelection.resolve(directory.path + "/./")
        XCTAssertEqual(selected.modelID, "speech-tts-qwen3-nano")
        XCTAssertEqual(selected.modelPath, directory.standardizedFileURL.path)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), ["config.json"])
    }

    func testNonTTSModelsAndWireOnlyAliasesDoNotReachTheNativeRuntime() {
        for selector in ["tts-1", "speech-asr-qwen3", "unknown-speech-fixture"] {
            XCTAssertThrowsError(try SpeechSynthesisModelSelection.resolve(selector))
        }
    }
}
