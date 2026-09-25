import Foundation
import MereRunContract

extension ModelFamilyIdentifier {
    /// Speech transcribe has no probe: its router tells a local Parakeet folder apart with the
    /// config decoder in AudioParakeetModel, which Core does not depend on, so a local ASR folder
    /// stays unidentified and the command checks it. Speech synthesize routes by `--mode` alone.
    static let speechProbes: [String: Probe] = [
        "speech.diarize": speechDiarizeProbe
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
}
