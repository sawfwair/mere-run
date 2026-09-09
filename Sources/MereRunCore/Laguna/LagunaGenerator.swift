import Foundation
import MLX

public actor LagunaGenerator: ChatGenerator {
    static let prefillChunkSize = 512
    static let prefillChunkingThreshold = 4_096

    var model: LagunaCausalLM?
    var tokenizerAndTemplate: LagunaTokenizerAndTemplate?
    var config: LagunaConfig?
    var dflashModel: LagunaDFlashModel?
    var dflashConfig: LagunaDFlashConfig?
    var loadedModelPath: String?
    var loadedDFlashPath: String?
    var loadedTextLoRASignature: String?
    let continuousBatchingEnabled: Bool
    let configuredDFlashPath: String?
    let dflashSpeculativeTokens: Int
    let dflashMinimumOutputTokens: Int

    var decodeQueue: [LagunaBatchedDecodeRow] = []
    var activeDecodeRows: [LagunaBatchedDecodeRow] = []
    var decodeLoopRunning = false
    var dflashDecodeQueue: [LagunaDFlashBatchedDecodeRow] = []
    var activeDFlashDecodeRows: [LagunaDFlashBatchedDecodeRow] = []
    var dflashDecodeLoopRunning = false
    var batchedDecodeSteps = 0
    var samePositionBatchedSteps = 0
    var variablePositionBatchedSteps = 0
    var singleDecodeSteps = 0
    var totalBatchedRows = 0
    var maxObservedBatchSize = 0
    var dflashRounds = 0
    var dflashDraftedTokens = 0
    var dflashAcceptedDraftTokens = 0
    var dflashRejectedDraftTokens = 0
    var dflashFullAcceptanceRounds = 0
    var dflashTargetVerificationForwards = 0
    var dflashTargetRecoveryForwards = 0
    var dflashTargetFallbackForwards = 0
    var dflashAdaptiveFallbacks = 0
    var dflashRoutedRequests = 0
    var dflashBypassedRequests = 0

    public init(
        continuousBatchingEnabled: Bool =
            ProcessInfo.processInfo.environment["MERERUN_LAGUNA_CONTINUOUS_BATCHING"] == "1",
        dflashModelPath: String? =
            ProcessInfo.processInfo.environment["MERERUN_LAGUNA_DFLASH_PATH"],
        dflashSpeculativeTokens: Int =
            Int(ProcessInfo.processInfo.environment["MERERUN_LAGUNA_DFLASH_TOKENS"] ?? "")
                ?? LagunaDFlashRouting.defaultSpeculativeTokens,
        dflashMinimumOutputTokens: Int =
            Int(ProcessInfo.processInfo.environment[
                "MERERUN_LAGUNA_DFLASH_MIN_TOKENS"
            ] ?? "") ?? LagunaDFlashRouting.defaultMinimumOutputTokens
    ) {
        self.continuousBatchingEnabled = continuousBatchingEnabled
        self.configuredDFlashPath = dflashModelPath
        self.dflashSpeculativeTokens = max(1, dflashSpeculativeTokens)
        self.dflashMinimumOutputTokens = max(1, dflashMinimumOutputTokens)
    }

    public func chat(
        _ request: ChatRequest,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> ChatResponse {
        throw LagunaError.modelPathRequired
    }

    public func chat(
        _ request: ChatRequest,
        modelPath: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> ChatResponse {
        try await chat(
            request,
            modelPath: modelPath,
            dflashRouting: .automatic,
            progressHandler: progressHandler
        )
    }

    public func chat(
        _ request: ChatRequest,
        modelPath: String,
        dflashRouting: LagunaDFlashRoutingMode,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws -> ChatResponse {
        try await Stream.withNewDefaultStream {
            let rootURL = URL(fileURLWithPath: modelPath).standardizedFileURL
            let loadStart = Date()
            let requestedLoRASignature = Self.loraSignature(request.lora)
            if loadedTextLoRASignature != requestedLoRASignature {
                guard !hasActiveGeneration else {
                    throw LagunaError.adapterSwitchDuringActiveGeneration
                }
                resetLoadedModel()
            }
            try await ensureLoaded(rootURL: rootURL, progressHandler: progressHandler)
            try await applyTextLoRAIfNeeded(
                request.lora,
                progressHandler: progressHandler
            )
            let loadSeconds = Date().timeIntervalSince(loadStart)

            var response = try await generate(
                request,
                dflashRouting: dflashRouting,
                progressHandler: progressHandler
            )
            if var timing = response.timing {
                timing.loadSeconds = loadSeconds
                response.timing = timing
            } else {
                response.timing = ChatTiming(loadSeconds: loadSeconds)
            }
            return response
        }
    }

    public func prepare(
        modelPath: String,
        progressHandler: (@Sendable (ChatProgress) -> Void)? = nil
    ) async throws {
        try await Stream.withNewDefaultStream {
            try await ensureLoaded(
                rootURL: URL(fileURLWithPath: modelPath).standardizedFileURL,
                progressHandler: progressHandler
            )
            guard let model else {
                throw LagunaError.modelNotLoaded
            }
            progressHandler?(ChatProgress(
                stage: .loadingModel,
                message: "Warming Laguna inference"
            ))
            warmUp(model: model, dflash: dflashModel)
        }
    }

    public func unload() {
        guard !hasActiveGeneration else { return }
        resetLoadedModel()
    }

    func resetLoadedModel() {
        model = nil
        tokenizerAndTemplate = nil
        config = nil
        dflashModel = nil
        dflashConfig = nil
        loadedModelPath = nil
        loadedDFlashPath = nil
        loadedTextLoRASignature = nil
        Memory.clearCache()
    }

    var hasActiveGeneration: Bool {
        decodeLoopRunning
            || dflashDecodeLoopRunning
            || !decodeQueue.isEmpty
            || !activeDecodeRows.isEmpty
            || !dflashDecodeQueue.isEmpty
            || !activeDFlashDecodeRows.isEmpty
    }

    public func continuousBatchingStats() -> LagunaContinuousBatchingStats {
        LagunaContinuousBatchingStats(
            enabled: continuousBatchingEnabled,
            activeRows: activeDecodeRows.count + activeDFlashDecodeRows.count,
            queuedRows: decodeQueue.count + dflashDecodeQueue.count,
            batchedDecodeSteps: batchedDecodeSteps,
            samePositionBatchedSteps: samePositionBatchedSteps,
            variablePositionBatchedSteps: variablePositionBatchedSteps,
            singleDecodeSteps: singleDecodeSteps,
            totalBatchedRows: totalBatchedRows,
            maxBatchSize: maxObservedBatchSize
        )
    }

    public func dflashStats() -> LagunaDFlashStats {
        LagunaDFlashStats(
            enabled: dflashModel != nil,
            speculativeTokens: dflashSpeculativeTokens,
            minimumOutputTokens: dflashMinimumOutputTokens,
            routedRequests: dflashRoutedRequests,
            bypassedRequests: dflashBypassedRequests,
            rounds: dflashRounds,
            draftedTokens: dflashDraftedTokens,
            acceptedDraftTokens: dflashAcceptedDraftTokens,
            rejectedDraftTokens: dflashRejectedDraftTokens,
            fullAcceptanceRounds: dflashFullAcceptanceRounds,
            targetVerificationForwards: dflashTargetVerificationForwards,
            targetRecoveryForwards: dflashTargetRecoveryForwards,
            targetFallbackForwards: dflashTargetFallbackForwards,
            adaptiveFallbacks: dflashAdaptiveFallbacks
        )
    }

}
