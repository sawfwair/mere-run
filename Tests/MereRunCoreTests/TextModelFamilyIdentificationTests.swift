import Foundation
import MereRunContract
import Testing

@testable import MereRunCore

private typealias ChatFamily = MereRunCapabilityCatalog.TextChatFamily

/// The runtime a text chat family loads; vision and text-only siblings share one.
private func runtime(_ family: ChatFamily) -> String {
    switch family {
    case .gemma4, .gemma4Unified: "gemma4"
    case .lfm2, .lfm2A1B, .lfm2VL: "lfm2"
    case .q35, .q35VL, .q38: "qwen"
    default: family.rawValue
    }
}

private func routing(_ capability: MereRunCommandCapability) throws -> MereRunCapabilityRouting {
    try #require(capability.routing)
}

/// Router agreement: the substring detectors the command used before the contract listed exact
/// ids pick the same runtime for every listed model.
@Test func textChatDetectorsAgreeWithTheContractForEveryListedModel() throws {
    for family in try routing(MereRunCapabilityCatalog.textChat).families {
        let contract = try #require(ChatFamily(rawValue: family.id))
        for model in family.models {
            #expect(ChatFamily(managedModel: model) == contract, "\(model)")
            let detected = ModelFamilyIdentifier.textChatFamily(matching: model)
            #expect(runtime(detected) == runtime(contract), "\(model): detected \(detected), listed \(contract)")
            #expect(NativeChatRuntime.commandFamily(modelID: model) == contract, "\(model)")
            switch contract {
            case .gemma4, .gemma4Unified:
                #expect(Gemma4Resources.supportsVision(modelSpec: model) == (contract == .gemma4Unified), "\(model)")
            case .lfm2, .lfm2A1B, .lfm2VL:
                let vision = [LFM2Resources.visionModelId, LFM2Resources.visionBF16ModelId].contains(model)
                #expect(vision == (contract == .lfm2VL), "\(model)")
                // The one checkpoint the runtime loads a text adapter on.
                #expect((model == LFM2Resources.defaultModelId) == (contract == .lfm2A1B), "\(model)")
            case .q35, .q35VL, .q38:
                #expect(Q35Resources.profile(for: model) != nil, "\(model) has no Qwen profile")
                #expect(Q35Resources.isQ38ModelId(model) == (contract == .q38), "\(model)")
            default:
                break
            }
        }
    }
}

/// Probe fixtures: ids the contract does not list resolve the way the command reads them.
@Test func textChatProbeIdentifiesUnlistedModels() {
    let identify = { (model: String) in
        let invocation = MereRunCommandInvocation(capability: MereRunCapabilityCatalog.textChat, arguments: ["--model", model])
        return ModelFamilyIdentifier.identify(capabilityID: "text.chat", model: model, invocation: invocation)
    }
    let cases: [(String, MereRunModelIdentification?)] = [
        ("TEXT-CHAT-GEMMA4-NANO", .managedModel("text-chat-gemma4-nano")),
        // A Gemma 4 id the contract does not list may be the unified vision checkpoint.
        ("google/gemma-4-custom", .family("gemma4-unified")),
        ("acme/Inkling-Small-SFT", .family("inkling")),
        ("my-muse-glimmer-finetune", .family("muse-glimmer")),
        ("/models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16-MLX-Native", .family("nemotron-omni")),
        // An LFM2.5 id that is not a folder can't say which checkpoint it is; the run decides.
        ("liquidai/lfm2-custom", nil),
        // Managed models that list another command still run where text chat sends them.
        ("text-code-qwen3", .family("gguf")),
        ("vision-ocr-infinity-pro", .family("q35-vl")),
        // Nothing claims it, and the Qwen fallthrough has no profile for it: the command fails.
        ("unknown-model", nil)
    ]
    for (model, expected) in cases {
        #expect(identify(model) == expected, "\(model)")
    }
}

/// Router agreement for training: the trainer the command selects, and its default targets.
@Test func textTrainLoRADetectorsAgreeWithTheContractForEveryListedModel() throws {
    let capability = MereRunCapabilityCatalog.textTrainLoRA
    let targets = try #require(capability.options.first { $0.flag == "--target-modules" })
    for family in try routing(capability).families {
        let defaultTargets = targets.familyRules.first { $0.family == family.id }?.defaultValue
        for model in family.models {
            let options = TextLoRATrainingOptions(data: "", output: "", model: model)
            #expect(try options.resolvedTrainingFamily().contractFamily.rawValue == family.id, "\(model)")
            #expect(TextLoRATrainingOptions.defaultTargetModules(for: model).joined(separator: ",") == defaultTargets, "\(model)")
        }
    }
    for excluded in try routing(capability).excludedModels where excluded.id.contains("laguna") {
        #expect(throws: TextLoRATrainingIssue.self) {
            try TextLoRATrainingOptions(data: "", output: "", model: excluded.id).resolvedTrainingFamily()
        }
    }
}

@Test func textTrainLoRAProbeIdentifiesUnlistedModels() {
    let identify = { (model: String) in
        let invocation = MereRunCommandInvocation(capability: MereRunCapabilityCatalog.textTrainLoRA, arguments: ["--model", model])
        return ModelFamilyIdentifier.identify(capabilityID: "text.train-lora", model: model, invocation: invocation)
    }
    #expect(identify("acme/inkling-small-sft") == .family("inkling"))
    #expect(identify("google/gemma-4-custom") == .family("gemma4"))
    #expect(identify("text-chat-q36-nano") == nil, "no trainer takes it; the command says so")
}

/// A local LFM2.5 folder is the family its config says, as the runtime reads it: a vision tower,
/// the 8-bit A1B mixture of experts that takes text adapters, or the rest.
@Test func localLFM2FoldersIdentifyByTheirConfig() throws {
    func folder(_ config: [String: Any]) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("lfm2-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: config).write(to: root.appendingPathComponent("config.json"))
        return root
    }
    let text: [String: Any] = [
        "vocab_size": 64, "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 2, "max_position_embeddings": 128, "norm_eps": 0.00001,
        "conv_L_cache": 2, "layer_types": ["conv", "full_attention"]
    ]
    let eightBit: [String: Any] = ["group_size": 64, "bits": 8, "mode": "affine"]
    let moe8 = try folder(text.merging(["model_type": "lfm2_moe", "quantization": eightBit]) { $1 })
    let moeBF16 = try folder(text.merging(["model_type": "lfm2_moe"]) { $1 })
    let dense8 = try folder(text.merging(["model_type": "lfm2", "quantization": eightBit]) { $1 })
    let vision = try folder(["model_type": "lfm2_vl", "text_config": text.merging(["model_type": "lfm2"]) { $1 }])
    defer { for root in [moe8, moeBF16, dense8, vision] { try? FileManager.default.removeItem(at: root) } }
    let cases: [(URL, String)] = [(moe8, "lfm2-a1b"), (moeBF16, "lfm2"), (dense8, "lfm2"), (vision, "lfm2-vl")]
    for (root, family) in cases {
        // The folder's name carries "lfm2", which is how the command's detector claims it.
        let invocation = MereRunCommandInvocation(capability: MereRunCapabilityCatalog.textChat, arguments: ["--model", root.path])
        #expect(ModelFamilyIdentifier.identify(capabilityID: "text.chat", model: root.path, invocation: invocation)
            == .family(family), "\(family)")
    }
}

/// Linux picks its chat default by memory and accelerator among candidates that span families;
/// Core's chooser answers the gate with one of the contract's candidates on every machine.
@Test func theTextChatDefaultIsOneOfTheContractsCandidatesOnEveryMachine() throws {
    let rules = try routing(MereRunCapabilityCatalog.textChat).defaultModels
    for gigabytes in [8, 16, 24, 32, 64, 128] {
        let memory = UInt64(gigabytes) * 1_073_741_824
        for (isLinux, cuda) in [(false, false), (true, false), (true, true)] {
            let machine = MereRunMachineProfile(
                physicalMemoryBytes: memory, processorName: "test", isAppleSiliconMac: !isLinux, isLinux: isLinux
            )
            let platform = isLinux ? "linux" : "macos"
            let chosen = TextChatDefaultModel.id(on: machine, linuxCUDA: cuda)
            let rule = try #require(rules.first { $0.applies(on: platform) })
            #expect(rule.models.contains(chosen), "\(platform) \(gigabytes) GB cuda=\(cuda): \(chosen)")
        }
    }
    let linux = try #require(rules.first { $0.applies(on: "linux") })
    #expect(ModelFamilyIdentifier.machineDefault(capabilityID: "text.chat", candidates: linux.models)
        .map(linux.models.contains) == true)
    let report = MereRunCapabilityCatalog.textChat.resolutionReport(
        MereRunCommandInvocation(capability: MereRunCapabilityCatalog.textChat, arguments: ["--prompt", "hi"]),
        platform: "linux",
        chooseDefault: { ModelFamilyIdentifier.machineDefault(capabilityID: "text.chat", candidates: $0) }
    )
    #expect(report.family != nil && report.source == .defaultModel, "\(report)")
}
