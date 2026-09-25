#if os(macOS) || os(Linux)
import Foundation
import MereRunContract

extension NativeChatRuntime {
    /// A command uses generator defaults; a server can explicitly override
    /// cache and batching policy when creating a resident runtime.
    public static func make(
        engine: RuntimeServingEngine, modelID: String, modelPath: String?,
        gemma4KVCacheQuantization: Gemma4KVCacheQuantization = Gemma4KVCacheQuantization(),
        prefixKVCacheEnabled: Bool? = nil, continuousBatchingEnabled: Bool? = nil
    ) -> Self {
        func option(_ value: Bool?, _ key: String) -> Bool {
            value ?? (ProcessInfo.processInfo.environment[key] == "1")
        }
        switch engine {
        case .textCode:
            return .textCode(CodeGenGenerator(modelId: modelID), modelPath: modelPath)
        case .textChatKlein:
            return .textChatKlein(
                Flux2KleinGenerator(), modelPath: modelPath,
                useStandalone: modelID == ModelResolver.ModelID.mebot.rawValue
                    && modelPath == MeBotModelCatalog.resolveModelPath()
            )
        case .textChatGemma4:
            return .textChatGemma4(
                Gemma4Generator(
                    modelId: modelID, kvCacheQuantization: gemma4KVCacheQuantization,
                    prefixKVCacheEnabled: option(prefixKVCacheEnabled, "MERERUN_GEMMA4_PREFIX_KV_CACHE"),
                    continuousBatchingEnabled: option(continuousBatchingEnabled, "MERERUN_GEMMA4_CONTINUOUS_BATCHING")
                ), modelPath: modelPath
            )
        case .textChatDiffusionGemma:
            return .textChatDiffusionGemma(DiffusionGemmaGenerator(modelID: modelID), modelPath: modelPath)
        case .textChatLaguna:
            return .textChatLaguna(
                LagunaGenerator(
                    continuousBatchingEnabled: option(continuousBatchingEnabled, "MERERUN_LAGUNA_CONTINUOUS_BATCHING"),
                    dflashModelPath: LagunaResources.installedDFlashPath(for: modelID)
                ), modelPath: modelPath
            )
        case .textChatQ35, .textChatQ36:
            return .textChatQ35(
                Q35Generator(
                    modelId: modelID,
                    prefixKVCacheEnabled: option(prefixKVCacheEnabled, "MERERUN_Q35_PREFIX_KV_CACHE"),
                    continuousBatchingEnabled: option(continuousBatchingEnabled, "MERERUN_Q35_CONTINUOUS_BATCHING")
                ), modelPath: modelPath
            )
        case .textChatLFM2:
            return .textChatLFM2(
                LFM2Generator(
                    modelId: modelID,
                    prefixKVCacheEnabled: option(prefixKVCacheEnabled, "MERERUN_LFM2_PREFIX_KV_CACHE"),
                    continuousBatchingEnabled: option(continuousBatchingEnabled, "MERERUN_LFM2_CONTINUOUS_BATCHING")
                ), modelPath: modelPath
            )
        case .textChatDeepseekV4Flash:
            return .textChatDeepseekV4Flash(DeepseekV4FlashGenerator(modelId: modelID), modelPath: modelPath)
        case .textChatMuseGlimmer:
            return .textChatMuseGlimmer(MuseGlimmerGenerator(modelID: modelID), modelPath: modelPath)
        case .textChatNemotronH:
            return .textChatNemotronH(NemotronHGenerator(), modelPath: modelPath)
        case .textChatNemotronOmni:
            return .textChatNemotronOmni(NemotronOmniGenerator(), modelPath: modelPath)
        }
    }

    /// The `text chat` family that runs `modelID`: the contract's exact managed ids first, then
    /// the identifier's reading of any other id.
    public static func commandFamily(modelID: String) -> MereRunCapabilityCatalog.TextChatFamily {
        MereRunCapabilityCatalog.TextChatFamily(managedModel: modelID)
            ?? ModelFamilyIdentifier.textChatFamily(matching: modelID)
    }

    /// Preserves command family selection, including command-only Psi and
    /// Inkling runtimes. API selection uses its managed serving profile.
    public static func command(
        modelID: String, modelPath: String?,
        gemma4KVCacheQuantization: Gemma4KVCacheQuantization = Gemma4KVCacheQuantization()
    ) throws -> Self {
        let engine: RuntimeServingEngine
        switch commandFamily(modelID: modelID) {
        case .psi:
            return .textChatPsi(Psi3ChatGenerator(modelId: modelID), modelPath: modelPath)
        case .inkling:
            return .textChatInkling(InklingGenerator(modelID: modelID), modelPath: modelPath)
        case .diffusionGemma:
            engine = .textChatDiffusionGemma
        case .gemma4, .gemma4Unified:
            engine = .textChatGemma4
        case .laguna:
            guard modelPath != nil else {
                let id = LagunaResources.managedModelID(for: modelID) ?? modelID
                throw ChatRequestIssue("model", "'\(id)' is not installed. Run 'mere.run model pull \(id)' first.")
            }
            engine = .textChatLaguna
        case .gguf:
            engine = .textCode
        case .museGlimmer:
            engine = .textChatMuseGlimmer
        case .nemotronOmni:
            engine = .textChatNemotronOmni
        case .nemotronH:
            engine = .textChatNemotronH
        case .lfm2, .lfm2A1B, .lfm2VL:
            engine = .textChatLFM2
        case .q35, .q35VL, .q38:
            engine = .textChatQ35
        }
        // Gemma4 claims an empty spec, so an empty ID must fall back to the selected
        // engine's own default rather than a Q35 ID the Gemma4 generator cannot use.
        let effectiveModelID: String
        if modelID.isEmpty {
            effectiveModelID = engine == .textChatGemma4
                ? Gemma4Resources.defaultModelId : Q35Resources.defaultModelId
        } else {
            effectiveModelID = modelID
        }
        return make(
            engine: engine, modelID: effectiveModelID, modelPath: modelPath,
            gemma4KVCacheQuantization: gemma4KVCacheQuantization
        )
    }
}
#endif
