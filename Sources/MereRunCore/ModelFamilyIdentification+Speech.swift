import Foundation
import MereRunContract

extension ModelFamilyIdentifier {
    /// Speech transcribe has no probe: its router tells a local Parakeet folder apart with the
    /// config decoder in AudioParakeetModel, which Core does not depend on, so a local ASR folder
    /// stays unidentified and the command checks it.
    static let speechProbes: [String: Probe] = [
        "speech.diarize": speechDiarizeProbe,
        "speech.synthesize": speechSynthesizeProbe
    ]

    /// `speech diarize` loads any local folder, and runs Nemotron 3 on one that holds its NeMo
    /// archive and Sortformer on the rest (`Nemotron3DiarizationResources.isNemotron3`). Anything
    /// that is not a folder the command refuses on its own.
    static let speechDiarizeProbe: Probe = { model, _ in
        let root = URL(fileURLWithPath: model).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return nil
        }
        return .family(Nemotron3DiarizationResources.isNemotron3(model: model, root: root) ? "nemotron3" : "sortformer")
    }

    /// `speech synthesize` runs any local Qwen3-TTS folder in either mode. In style mode a folder
    /// whose config names speakers (`talker_config.spk_id`, a CustomVoice checkpoint) takes
    /// `--speaker`, as `Qwen3TTSResources.speaker(named:)` reads it. A folder without a readable
    /// config is left to the command, which refuses it.
    static let speechSynthesizeProbe: Probe = { model, invocation in
        let config = URL(fileURLWithPath: model).standardizedFileURL.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: config),
              let modelType = try? JSONDecoder().decode(SpeechModelType.self, from: data),
              let speakers = try? JSONDecoder().decode(Qwen3TTSSpeakerTable.self, from: data) else {
            return nil
        }
        if invocation.value("--mode") == "clone" {
            return .family(modelType.modelType == "breeze" ? "breeze-clone" : "clone")
        }
        if modelType.modelType == "breeze" { return .family("breeze") }
        return .family(speakers.names.isEmpty ? "style" : "custom-voice")
    }
}

private struct SpeechModelType: Decodable {
    let modelType: String?
    enum CodingKeys: String, CodingKey { case modelType = "model_type" }
}

/// The speaker names in a Qwen3-TTS `config.json`; each id's shape is the runtime's concern.
private struct Qwen3TTSSpeakerTable: Decodable {
    let names: [String]

    private enum CodingKeys: String, CodingKey { case talkerConfig = "talker_config" }
    private enum TalkerKeys: String, CodingKey { case spkId = "spk_id" }

    private struct SpeakerID: Decodable {
        init(from decoder: Decoder) throws {}
    }

    /// Both tables may be absent, as `Qwen3TTSModelConfig` reads them: then there are no speakers.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        guard container.contains(.talkerConfig) else {
            names = []
            return
        }
        let talker = try container.nestedContainer(keyedBy: TalkerKeys.self, forKey: .talkerConfig)
        names = try talker.decodeIfPresent([String: SpeakerID].self, forKey: .spkId)?.keys.sorted() ?? []
    }
}
