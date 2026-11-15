import Foundation
import AudioCore

struct SOVTranscribePreprocessing: Decodable {
    let sampling_rate: Int
    let feature_size: Int
    let n_fft: Int
    let hop_length: Int
    let n_samples: Int
    let nb_max_frames: Int

    static func validate(root: URL, downsampleFactor: Int) throws {
        struct Processor: Decodable {
            let downsample_factor: Int
            let audio_token: String
            let audio_begin_token: String
            let audio_end_token: String
        }
        let decoder = JSONDecoder()
        let features = try decoder.decode(Self.self, from: Data(contentsOf: root.appendingPathComponent("preprocessor_config.json")))
        let processor = try decoder.decode(Processor.self, from: Data(contentsOf: root.appendingPathComponent("processor_config.json")))
        guard features.sampling_rate == 16_000, features.feature_size == 128, features.n_fft == 400,
              features.hop_length == 160, features.n_samples == 480_000, features.nb_max_frames == 3000,
              processor.downsample_factor == downsampleFactor, processor.audio_token == "<|audio|>",
              processor.audio_begin_token == "<|audio>", processor.audio_end_token == "<audio|>" else {
            throw SpeechTranscriptionIssue("invalid_processor", "SOV-Transcribe-1 requires its released Whisper frontend and audio markers.")
        }
    }
}
