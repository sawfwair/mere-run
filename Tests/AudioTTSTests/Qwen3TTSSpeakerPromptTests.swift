import AudioQwen3TTSModel
@testable import AudioTTS
import Foundation
import XCTest

/// The CustomVoice checkpoint's own speaker and language tables, trimmed to what the prompt reads.
private let customVoiceTalker = #"""
{
  "codec_think_id": 2154, "codec_nothink_id": 2155, "codec_think_bos_id": 2156, "codec_think_eos_id": 2157,
  "codec_language_id": {"chinese": 2055, "english": 2050, "beijing_dialect": 2074, "sichuan_dialect": 2062},
  "spk_id": {"ryan": 3061, "vivian": 3065, "dylan": 2878, "eric": 2875},
  "spk_is_dialect": {"ryan": false, "vivian": false, "dylan": "beijing_dialect", "eric": "sichuan_dialect"}
}
"""#

final class Qwen3TTSSpeakerPromptTests: XCTestCase {
    private func talker() throws -> Qwen3TTSTalkerConfig {
        try JSONDecoder().decode(Qwen3TTSTalkerConfig.self, from: Data(customVoiceTalker.utf8))
    }

    /// The speaker's codec id follows the language tag, as upstream's `generate_custom_voice`
    /// places the speaker embedding.
    func testANamedSpeakerFollowsTheLanguageTag() throws {
        let config = try talker()
        XCTAssertEqual(Qwen3TTSGenerator.styleCodecPrefill(language: "english", speaker: "ryan", talkerConfig: config),
                       [2154, 2156, 2050, 2157, 3061])
        XCTAssertEqual(Qwen3TTSGenerator.styleCodecPrefill(language: "auto", speaker: "vivian", talkerConfig: config),
                       [2155, 2156, 2157, 3065])
        XCTAssertEqual(Qwen3TTSGenerator.styleCodecPrefill(language: "English", speaker: nil, talkerConfig: config),
                       [2154, 2156, 2050, 2157])
    }

    /// A dialect speaker takes its dialect's tag for Chinese or unmarked text only.
    func testADialectSpeakerTakesItsDialectForChineseText() throws {
        let config = try talker()
        XCTAssertEqual(Qwen3TTSGenerator.styleCodecPrefill(language: "chinese", speaker: "dylan", talkerConfig: config),
                       [2154, 2156, 2074, 2157, 2878])
        XCTAssertEqual(Qwen3TTSGenerator.styleCodecPrefill(language: "auto", speaker: "eric", talkerConfig: config),
                       [2154, 2156, 2062, 2157, 2875])
        XCTAssertEqual(Qwen3TTSGenerator.styleCodecPrefill(language: "english", speaker: "dylan", talkerConfig: config),
                       [2154, 2156, 2050, 2157, 2878])
    }

    /// The runtime checks a name against the checkpoint that runs, without case, before loading.
    func testASpeakerIsCheckedAgainstTheCheckpointConfig() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let config = root.appendingPathComponent("config.json")
        try Data(#"{"talker_config": \#(customVoiceTalker)}"#.utf8).write(to: config)
        let resources = Qwen3TTSResources(rootURL: root)

        XCTAssertEqual(try resources.speakers(), ["dylan", "eric", "ryan", "vivian"])
        XCTAssertEqual(try resources.speaker(named: " Ryan "), "ryan")
        XCTAssertNil(try resources.speaker(named: nil))
        XCTAssertThrowsError(try resources.speaker(named: "alloy")) { error in
            XCTAssertEqual(
                error.localizedDescription,
                "Unknown speaker alloy. This checkpoint's speakers are dylan, eric, ryan, vivian."
            )
        }

        // A checkpoint without speakers (VoiceDesign, Base) ignores the name.
        try Data(#"{"talker_config": {"spk_id": {}}}"#.utf8).write(to: config)
        XCTAssertEqual(try resources.speakers(), [])
        XCTAssertNil(try resources.speaker(named: "ryan"))
    }
}
