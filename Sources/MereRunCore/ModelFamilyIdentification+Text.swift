import Foundation
import MereRunContract

extension ModelFamilyIdentifier {
    /// `text chat`'s family for a model id the contract does not list: an upstream repository, a
    /// local folder name, or an id that only a runtime's substring match claims. Checked in the
    /// order the command always used. Where the id can't say whether the checkpoint ships a vision
    /// tower, the answer is the vision-capable family, which takes every option its text-only
    /// sibling does. Anything no runtime claims falls through to the Qwen family, as it always has.
    public static func textChatFamily(matching model: String) -> MereRunCapabilityCatalog.TextChatFamily {
        if model == Psi3ChatResources.defaultModelId {
            return .psi
        }
        if model == DiffusionGemmaResources.modelID {
            return .diffusionGemma
        }
        if Gemma4Resources.handles(modelSpec: model) {
            return .gemma4Unified
        }
        if LagunaResources.handles(modelSpec: model) {
            return .laguna
        }
        if ManagedModelCatalog.spec(for: model)?.validationKind == .codegenGGUF {
            return .gguf
        }
        if InklingResources.handles(modelSpec: model) {
            return .inkling
        }
        if MuseGlimmerResources.handles(modelSpec: model) {
            return .museGlimmer
        }
        if NemotronOmniResources.handles(modelSpec: model) {
            return .nemotronOmni
        }
        if NemotronHResources.handles(modelSpec: model) {
            return .nemotronH
        }
        if LFM2Resources.handles(modelSpec: model) {
            return .lfm2VL
        }
        return Q35Resources.isQ38ModelId(model) ? .q38 : .q35VL
    }

    static let textProbes: [String: Probe] = [
        "text.chat": textChatProbe,
        "text.train-lora": textTrainLoRAProbe,
    ]

    static let textDefaultChoosers: [String: DefaultChooser] = [
        // Linux picks among Qwen3.6 A3B (GGUF or MLX) and Gemma 4 by memory and accelerator.
        "text.chat": { _ in TextChatDefaultModel.current }
    ]

    /// The identifier's `text chat` probe. The Qwen fallthrough runs only a model with a Qwen
    /// profile, so any other id is unidentified and the command reports it.
    static func textChatProbe(model: String, invocation: MereRunCommandInvocation) -> MereRunModelIdentification? {
        // The command reads its model id trimmed and lowercased.
        let model = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let family = textChatFamily(matching: model)
        switch family {
        case .q35, .q35VL, .q38:
            return Q35Resources.profile(for: model) == nil ? nil : .family(family.rawValue)
        case .lfm2, .lfm2A1B, .lfm2VL:
            return lfm2Family(folder: model).map { .family($0.rawValue) }
        default:
            return .family(family.rawValue)
        }
    }

    /// The LFM2.5 family of a local checkpoint, read from its config the way the runtime reads it:
    /// a vision tower, the 8-bit A1B mixture of experts (the one text LoRA adapters load on), or
    /// the rest. An id that is not a folder can't say, so the command decides when it runs.
    static func lfm2Family(folder model: String) -> MereRunCapabilityCatalog.TextChatFamily? {
        let root = LFM2Resources.normalizedRootURL(URL(fileURLWithPath: model).standardizedFileURL)
        guard let data = try? Data(contentsOf: root.appendingPathComponent("config.json")),
              let modelType = try? JSONDecoder().decode(LFM2ModelTypeEnvelope.self, from: data).modelType else {
            return nil
        }
        if modelType == "lfm2_vl" { return .lfm2VL }
        guard let config = try? JSONDecoder().decode(LFM2Config.self, from: data) else { return nil }
        return config.modelType == "lfm2_moe" && config.quantization?.bits == 8 ? .lfm2A1B : .lfm2
    }

    /// The identifier's `text train-lora` probe: the trainer the command itself selects.
    static func textTrainLoRAProbe(model: String, invocation: MereRunCommandInvocation) -> MereRunModelIdentification? {
        let options = TextLoRATrainingOptions(data: "", output: "", model: model)
        return (try? options.resolvedTrainingFamily()).map { .family($0.contractFamily.rawValue) }
    }
}
