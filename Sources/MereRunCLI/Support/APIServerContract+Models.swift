import Foundation
import MereRunCore

struct APIEngineCapabilities: Equatable, Sendable {
    var supportsRawProxy: Bool = false
    var supportsTools: Bool = false
    var usesNativeToolHistory: Bool = false
    var supportsToolChoice: Bool = false
    var supportsDeveloperRole: Bool = true
    var supportsStructuredOutputs: Bool = false
    var supportsReasoningEffort: Bool = false
    var supportsMaxCompletionTokens: Bool = true
    var supportsUsageInStreaming: Bool = true
    var supportsVisionContentParts: Bool = false
    var supportsAudioContentParts: Bool = false
    var supportsVideoContentParts: Bool = false
    var supportsStrictMode: Bool = false
    var supportsStopSequences: Bool = false
    var supportsSeed: Bool = false
    var supportsPenalties: Bool = false
    var supportsTopK: Bool = false
    var supportsRepetitionPenalty: Bool = false
    var supportsLogprobs: Bool = false
    var supportsProviderThinkingControls: Bool = false

    static func catalog(_ profile: ManagedModelAPIProfile) -> APIEngineCapabilities {
        APIEngineCapabilities(
            supportsRawProxy: profile.supportsRawProxy,
            supportsTools: profile.toolCall,
            usesNativeToolHistory: [.textChatQ36, .textChatLaguna, .textChatGemma4, .textChatMuseGlimmer]
                .contains(profile.servingEngine),
            supportsToolChoice: profile.supportsToolChoice,
            supportsDeveloperRole: profile.compatibility.supportsDeveloperRole,
            supportsStructuredOutputs: profile.structuredOutput,
            supportsReasoningEffort: profile.compatibility.supportsReasoningEffort,
            supportsMaxCompletionTokens: profile.compatibility.maxTokensField == .maxCompletionTokens,
            supportsUsageInStreaming: profile.compatibility.supportsUsageInStreaming,
            supportsVisionContentParts: profile.inputModalities.contains(.image),
            supportsAudioContentParts: profile.inputModalities.contains(.audio),
            supportsVideoContentParts: profile.inputModalities.contains(.video),
            supportsStrictMode: profile.compatibility.supportsStrictMode,
            supportsStopSequences: profile.supportsStopSequences,
            supportsSeed: profile.supportsSeed,
            supportsPenalties: profile.supportsPenalties,
            supportsTopK: [.textChatQ35, .textChatQ36, .textChatLaguna, .textChatLFM2,
                          .textChatMuseGlimmer, .textChatNemotronH, .textChatNemotronOmni,
                          .textChatDiffusionGemma].contains(profile.servingEngine),
            supportsRepetitionPenalty: [.textChatQ35, .textChatQ36].contains(profile.servingEngine),
            supportsLogprobs: profile.supportsLogprobs,
            supportsProviderThinkingControls: profile.supportsProviderThinkingControls
        )
    }

    static let localText = APIEngineCapabilities()

    static let localTextWithStructuredJSON = APIEngineCapabilities(
        supportsStructuredOutputs: true
    )

    static let localTextWithTools = APIEngineCapabilities(
        supportsTools: true,
        supportsToolChoice: true
    )

    static let localTextWithToolsAndVision = APIEngineCapabilities(
        supportsTools: true,
        supportsToolChoice: true,
        supportsVisionContentParts: true
    )
}

extension APIServerContract {
    static func modelsResponse(modelId: String, createdAt: Date = Date()) -> OpenAIModelsResponse {
        modelsResponse(modelIds: [modelId], createdAt: createdAt)
    }

    static func modelsResponse(modelIds: [String], createdAt: Date = Date()) -> OpenAIModelsResponse {
        OpenAIModelsResponse(
            object: "list",
            data: modelIds.map {
                OpenAIModel(
                    id: $0,
                    object: "model",
                    created: Int(createdAt.timeIntervalSince1970),
                    owned_by: "mere.run"
                )
            }
        )
    }

    static func chatModel(
        id: String,
        name: String,
        profile: ManagedModelAPIProfile,
        contextWindow: Int,
        maximumOutputTokens: Int,
        createdAt: Date = Date()
    ) -> OpenAIModel {
        let compatibility = profile.compatibility
        let thinkingLevels = profile.thinkingLevels.isEmpty
            ? nil
            : profile.thinkingLevels.map(\.rawValue)
        let thinkingLevelMap = profile.thinkingLevelMap.isEmpty
            ? nil
            : Dictionary(uniqueKeysWithValues: profile.thinkingLevelMap.map {
                ($0.key.rawValue, $0.value.rawValue)
            })

        return OpenAIModel(
            id: id,
            object: "model",
            created: Int(createdAt.timeIntervalSince1970),
            owned_by: "mere.run",
            name: name,
            task: profile.task.rawValue,
            reasoning: profile.reasoning,
            thinking_levels: thinkingLevels,
            tool_call: profile.toolCall,
            structured_output: profile.structuredOutput,
            modalities: OpenAIModelModalities(
                input: profile.inputModalities.map(\.rawValue),
                output: profile.outputModalities.map(\.rawValue)
            ),
            limit: OpenAIModelLimit(context: contextWindow, output: maximumOutputTokens),
            openai_compat: OpenAIModelCompatibility(
                supports_store: compatibility.supportsStore,
                supports_developer_role: compatibility.supportsDeveloperRole,
                supports_reasoning_effort: compatibility.supportsReasoningEffort,
                supports_usage_in_streaming: compatibility.supportsUsageInStreaming,
                supports_finish_reason: compatibility.supportsFinishReason,
                max_tokens_field: compatibility.maxTokensField.rawValue,
                supports_strict_mode: compatibility.supportsStrictMode,
                thinking_format: compatibility.thinkingFormat?.rawValue,
                thinking_level_map: thinkingLevelMap,
                requires_reasoning_content_on_assistant_messages: compatibility
                    .requiresReasoningContentOnAssistantMessages
            )
        )
    }

    static func companionModel(
        id: String,
        profile: ManagedModelAPIProfile,
        createdAt: Date = Date()
    ) -> OpenAIModel {
        return OpenAIModel(
            id: id,
            object: "model",
            created: Int(createdAt.timeIntervalSince1970),
            owned_by: "mere.run",
            name: id,
            task: profile.task.rawValue,
            reasoning: profile.reasoning,
            tool_call: profile.toolCall,
            structured_output: profile.structuredOutput,
            modalities: OpenAIModelModalities(
                input: profile.inputModalities.map(\.rawValue),
                output: profile.outputModalities.map(\.rawValue)
            )
        )
    }

    static func companionModelIDs(
        fileManager: FileManager = .default,
        installedModelIDs: Set<String>? = nil,
        includeLoopbackArtifactModels: Bool = true
    ) -> [String] {
        let categories: Set<ManagedModelCategory> = [
            .image, .image3D, .speechTTS, .speechASR, .textEmbed, .visionGeometry, .visionDepth,
        ]
        let ids = ManagedModelCatalog.allSpecs
            .filter { categories.contains($0.category) }
            .filter {
                includeLoopbackArtifactModels
                    || !APIVFXArtifactRoutePolicy.modelIDs.contains($0.id)
            }
            .filter { isCompanionModelInstalled($0, fileManager: fileManager, installedModelIDs: installedModelIDs) }
            .map(\.id)
        var uniqueIDs = Set(ids)
        if isQwenImageEditInstalled(fileManager: fileManager, installedModelIDs: installedModelIDs) {
            uniqueIDs.insert(QwenImageEditRepository.modelId)
        }
        return Array(uniqueIDs).sorted()
    }

    private static func isCompanionModelInstalled(
        _ spec: ManagedModelSpec,
        fileManager: FileManager,
        installedModelIDs: Set<String>?
    ) -> Bool {
        if let installedModelIDs {
            return installedModelIDs.contains(spec.id)
        }
        return spec.managedRuntimeURL(fileManager: fileManager) != nil
    }

    private static func isQwenImageEditInstalled(
        fileManager: FileManager,
        installedModelIDs: Set<String>?
    ) -> Bool {
        if let installedModelIDs {
            return installedModelIDs.contains(QwenImageEditRepository.modelId)
        }
        return QwenImageEditRepository.resolveInstalledModelRoot(fileManager: fileManager) != nil
    }
}
