import Foundation
import MereRunContract

/// The image commands pick their runtime from `mererun_model.json`: `image generate` through
/// `ImageGenerationBackend(manifest:)`, `image train-lora` through the manifest family. These
/// probes read the same manifest, a local folder's own or the one a managed id installs with,
/// and name the contract family that runtime is.
extension ModelFamilyIdentifier {
    static let imageProbes: [String: Probe] = [
        "image.generate": { model, _ in imageManifest(for: model).flatMap(imageGenerateFamily).map { .family($0) } },
        "image.train-lora": { model, _ in imageManifest(for: model).flatMap(imageTrainLoRAFamily).map { .family($0) } },
    ]

    /// The manifest both image commands load for `model`: an existing path first, then a
    /// managed id, as `ImageGenerationModelSelection` orders them. A folder without a readable
    /// manifest is unidentified; the command reports it when it runs.
    static func imageManifest(for model: String) -> MereRunModelManifest? {
        switch ImageGenerationModelSelection(model) {
        case .local(let root): return try? MereRunModelManifest.loadRequired(from: root)
        case .managed(let id): return MereRunModelManifest.template(for: id)
        case .unknown: return nil
        }
    }

    /// The `image generate` family of a manifest. FLUX.2-dev (the `standard` variant) runs the
    /// Klein engine with embedded guidance, so it ignores negative prompts; a Klein manifest with
    /// no variant is the shared-components root, which no generator runs. Qwen-Image-Edit
    /// Lightning is the edit engine under the Lightning manifest id, as the generator checks it.
    static func imageGenerateFamily(_ manifest: MereRunModelManifest) -> String? {
        switch try? ImageGenerationBackend(manifest: manifest) {
        case .flux1: "flux1"
        case .flux2Klein:
            switch manifest.variant {
            case .standard: "flux2-dev"
            case nil: nil
            default: "klein"
            }
        case .zImageTurbo: "zimage"
        case .hiDreamO1: "hidream"
        case .senseNovaU15: "sensenova"
        case .krea2: "krea"
        case .ideogram4: "ideogram"
        case .qwenImage21: "qwen-21"
        case .qwenImageEdit: manifest.id == QwenImageEditRepository.lightning2511Id ? "qwen-edit-lightning" : "qwen-edit"
        case nil: nil
        }
    }

    /// The `image train-lora` family of a manifest, as `ImageLoRATrainingPlan` dispatches on it.
    /// Every Klein manifest goes to the Klein trainer; only the Krea 2 base trains, and a
    /// distilled Krea 2 folder is left to the trainer, which refuses it.
    static func imageTrainLoRAFamily(_ manifest: MereRunModelManifest) -> String? {
        switch manifest.family {
        case .klein: "klein"
        case .krea: manifest.variant == .base ? "krea" : nil
        default: nil
        }
    }
}
