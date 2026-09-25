import AudioCore
import AudioSTT
import Foundation
import MereRunContract

extension CLIFamilyRouters {
    static let speechRouters: [String: Router] = [
        "speech.transcribe": speechTranscribe
    ]

    /// `speech transcribe` routes files and streams through `SpeechTranscriptionResolver.route`:
    /// translation, and a language Parakeet's router does not recognize, run Qwen3-ASR. The
    /// language half depends on normalization and the installed checkpoint, which the contract
    /// cannot declare. A route the resolver refuses (a local folder of the other backend, Core ML
    /// on Qwen) answers `nil`, and the command refuses it before loading.
    static let speechTranscribe: Router = { invocation in
        guard let backend = ASRBackend(rawValue: invocation.value("--backend") ?? ASRBackend.auto.rawValue),
              let task = ASRTask(rawValue: invocation.value("--task") ?? ASRTask.transcribe.rawValue),
              let route = try? SpeechTranscriptionResolver.route(
                  task: task, language: invocation.value("--language"), preferredBackend: backend,
                  modelOverride: invocation.value("--model")
              ) else {
            return nil
        }
        return switch route.decision.backend {
        case .parakeet: "parakeet"
        case .qwen: "qwen3-asr"
        }
    }
}
