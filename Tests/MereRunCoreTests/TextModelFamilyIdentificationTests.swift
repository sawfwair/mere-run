import Foundation
import MereRunContract
import Testing

@testable import MereRunCore

private typealias ChatFamily = MereRunCapabilityCatalog.TextChatFamily

/// The runtime a text chat family loads; vision and text-only siblings share one.
private func runtime(_ family: ChatFamily) -> String {
    switch family {
    case .gemma4, .gemma4Unified: "gemma4"
    case .lfm2, .lfm2VL: "lfm2"
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
            case .lfm2, .lfm2VL:
                let vision = [LFM2Resources.visionModelId, LFM2Resources.visionBF16ModelId].contains(model)
                #expect(vision == (contract == .lfm2VL), "\(model)")
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
        ("liquidai/lfm2-custom", .family("lfm2-vl")),
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
