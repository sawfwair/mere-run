#if os(macOS) || os(Linux)
import Foundation

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

    /// Preserves command family selection, including command-only Psi and
    /// Inkling runtimes. API selection uses its managed serving profile.
    public static func command(
        modelID: String, modelPath: String?,
        gemma4KVCacheQuantization: Gemma4KVCacheQuantization = Gemma4KVCacheQuantization()
    ) throws -> Self {
        let engine: RuntimeServingEngine
        if modelID == Psi3ChatResources.defaultModelId {
            return .textChatPsi(Psi3ChatGenerator(modelId: modelID), modelPath: modelPath)
        } else if modelID == DiffusionGemmaResources.modelID {
            engine = .textChatDiffusionGemma
        } else if Gemma4Resources.handles(modelSpec: modelID) {
            engine = .textChatGemma4
        } else if LagunaResources.handles(modelSpec: modelID) {
            guard modelPath != nil else {
                let id = LagunaResources.managedModelID(for: modelID) ?? modelID
                throw ChatRequestIssue("model", "'\(id)' is not installed. Run 'mere.run model pull \(id)' first.")
            }
            engine = .textChatLaguna
        } else if ManagedModelCatalog.spec(for: modelID)?.validationKind == .codegenGGUF {
            engine = .textCode
        } else if InklingResources.handles(modelSpec: modelID) {
            return .textChatInkling(InklingGenerator(modelID: modelID), modelPath: modelPath)
        } else if MuseGlimmerResources.handles(modelSpec: modelID) {
            engine = .textChatMuseGlimmer
        } else if NemotronOmniResources.handles(modelSpec: modelID) {
            engine = .textChatNemotronOmni
        } else if NemotronHResources.handles(modelSpec: modelID) {
            engine = .textChatNemotronH
        } else if LFM2Resources.handles(modelSpec: modelID) {
            engine = .textChatLFM2
        } else {
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
