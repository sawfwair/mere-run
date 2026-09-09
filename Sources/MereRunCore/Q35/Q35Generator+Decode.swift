import Foundation
import MediaIO
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

extension Q35Generator {
    static func finishReason(
        generatedTokenCount: Int,
        tokenBudget: Int,
        matchedStopSequence: Bool
    ) -> ChatFinishReason {
        if matchedStopSequence { return .stopSequence }
        return generatedTokenCount >= tokenBudget ? .length : .stop
    }

    static func decodePath(
        jsonConstrained: Bool,
        continuousBatchingEnabled: Bool,
        mtpSpeculationEnabled: Bool,
        schedulerContended: Bool = false,
        stopAtCompletedToolCall: Bool = false
    ) -> Q35DecodePath {
        if stopAtCompletedToolCall { return .pipelined }
        if jsonConstrained { return .jsonConstrainedSerial }
        if mtpSpeculationEnabled, !schedulerContended { return .mtpSpeculativeSerial }
        if continuousBatchingEnabled { return .continuousBatched }
        if mtpSpeculationEnabled { return .mtpSpeculativeSerial }
        return .pipelined
    }

    /// Decide whether to use MTP speculative decode for a request.
    ///
    /// Select the model-specific MTP break-even point. Qwen3.6 hybrid MoE uses
    /// the measured long-context threshold (~20-token context -31%, ~4K -22%,
    /// ~12K +1.5-2.5x on M4 Max). Ornith 1.5 uses its validated MTP companion
    /// from short prompts. Quantized Qwen3.8 uses serial-exact small-batch Q4
    /// verification from short prompts; its BF16 sibling remains opt-in.
    /// Qwen4Exp uses its exact verified inline head from short prompts.
    /// MERERUN_Q35_MTP_SPECULATION can enable/disable the path and
    /// MERERUN_Q35_MTP_MIN_PROMPT_TOKENS overrides either default.
    static func shouldSpeculate(
        promptTokenCount: Int,
        maxContextTokens: Int? = nil,
        defaultMinimumPromptTokens: Int = 6144,
        enabledByDefault: Bool = true
    ) -> Bool {
        shouldSpeculate(
            promptTokenCount: promptTokenCount,
            maxContextTokens: maxContextTokens,
            defaultMinimumPromptTokens: defaultMinimumPromptTokens,
            enabledByDefault: enabledByDefault,
            environment: ProcessInfo.processInfo.environment
        )
    }

    static func shouldSpeculate(
        promptTokenCount: Int,
        maxContextTokens: Int?,
        defaultMinimumPromptTokens: Int = 6144,
        enabledByDefault: Bool = true,
        environment env: [String: String]
    ) -> Bool {
        let rawPolicy = env["MERERUN_Q35_MTP_SPECULATION"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if rawPolicy == "0" || rawPolicy == "false" || rawPolicy == "no" {
            return false
        }
        let explicitlyEnabled = rawPolicy == "1" || rawPolicy == "true"
            || rawPolicy == "yes" || rawPolicy == "on"
        if !enabledByDefault, !explicitlyEnabled {
            return false
        }

        let threshold = env["MERERUN_Q35_MTP_MIN_PROMPT_TOKENS"].flatMap { Int($0) }
            ?? defaultMinimumPromptTokens
        let contextAllowsSpeculation = maxContextTokens.map { $0 >= threshold } ?? true
        if explicitlyEnabled {
            return contextAllowsSpeculation
        }
        if !contextAllowsSpeculation {
            return false
        }
        return promptTokenCount >= threshold
    }

    static func defaultMTPMinimumPromptTokens(modelId: String, usesMoE: Bool) -> Int {
        if Q35Resources.isOrnith35BModelId(modelId) {
            return 0
        }
        if modelId == Q35Resources.q38FlashNextMixedModelId
            || modelId == Q35Resources.q38FlashNext3BitModelId
            || modelId == Q35Resources.q38FlashNext3BitNativePLEModelId
            || modelId == Q35Resources.q38FlashNext4BitModelId {
            return 0
        }
        return usesMoE ? 6144 : 0
    }

    static func mtpBlockSize(
        modelId: String = Q35Resources.q36NanoModelId,
        environment env: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        let denseQ38 = modelId == Q35Resources.q38TwentySevenBModelId
            || modelId == Q35Resources.q38TwentySevenB4BitModelId
        let modelDefault = denseQ38
            ? q38MTPBlockSize
            : defaultMTPBlockSize
        guard let raw = env["MERERUN_Q35_MTP_BLOCK_SIZE"],
              let value = Int(raw), value >= 2 else {
            return modelDefault
        }
        let maximum = modelId == Q35Resources.q38TwentySevenB4BitModelId ? 9 : 16
        return min(maximum, value)
    }

    func decodeTokens(
        model: Q35Model,
        tokenizerAndTemplate: Q35TokenizerAndTemplate,
        initialLogits: MLXArray,
        initialHidden: MLXArray?,
        prefillMTPHidden: MLXArray?,
        prefillMTPSession: Q35MTPDraftSession?,
        layerCaches: [Q35LayerCache?],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        prefillTokenCount: Int,
        mropeRopeDelta: Int?,
        promptTokens: [Int],
        maxContextTokens: Int,
        jsonConstrained: Bool,
        stopAtCompletedToolCall: Bool,
        logprobCapture: ChatLogprobCapture,
        logprobRegion: ChatLogprobRegion,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35BatchedDecodeResult {
        guard tokenBudget > 0 else {
            return Q35BatchedDecodeResult(generatedTokens: [], decodeSeconds: 0)
        }
        let speculationMTP = !logprobCapture.isEnabled
            && !generationConfig.hasActivePenalties
            && !jsonConstrained && !stopAtCompletedToolCall && Self.shouldSpeculate(
            modelId: modelId,
            usesMoE: model.config.textConfig.usesMoE,
            promptTokenCount: promptTokens.count,
            maxContextTokens: maxContextTokens
        ) && mropeRopeDelta == nil ? mtpModel : nil
        let decodePath = Self.decodePath(
            jsonConstrained: jsonConstrained,
            continuousBatchingEnabled: continuousBatchingEnabled && !logprobCapture.isEnabled,
            mtpSpeculationEnabled: speculationMTP != nil,
            schedulerContended: activeChatRequestCount > 1
                || !decodeQueue.isEmpty
                || !activeDecodeRows.isEmpty,
            stopAtCompletedToolCall: stopAtCompletedToolCall
        )

        // JSON mode owns mutable prefix-grammar state and must validate every
        // token before it is streamed. Its plan therefore cannot select continuous
        // batching, MTP speculation, or the shared pipelined decoder.
        if decodePath != .continuousBatched {
            return try await decodeTokensSerially(
                model: model,
                tokenizerAndTemplate: tokenizerAndTemplate,
                initialLogits: initialLogits,
                initialHidden: initialHidden,
                prefillMTPHidden: prefillMTPHidden,
                prefillMTPSession: prefillMTPSession,
                mtpModel: decodePath == .mtpSpeculativeSerial ? speculationMTP : nil,
                layerCaches: layerCaches,
                eosSet: eosSet,
                generationConfig: generationConfig,
                tokenBudget: tokenBudget,
                prefillTokenCount: prefillTokenCount,
                mropeRopeDelta: mropeRopeDelta,
                promptTokens: promptTokens,
                jsonConstrained: decodePath == .jsonConstrainedSerial,
                stopAtCompletedToolCall: stopAtCompletedToolCall,
                logprobCapture: logprobCapture,
                logprobRegion: logprobRegion,
                progressHandler: progressHandler
            )
        }

        let rowID = UUID()
        let initialLogitsBox = RuntimeUncheckedSendable(initialLogits)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let row = Q35BatchedDecodeRow(
                    id: rowID,
                    logits: initialLogitsBox.value,
                    layerCaches: layerCaches,
                    eosSet: eosSet,
                    generationConfig: generationConfig,
                    tokenBudget: tokenBudget,
                    prefillTokenCount: prefillTokenCount,
                    mropeRopeDelta: mropeRopeDelta,
                    repetitionHistory: promptTokens,
                    progressHandler: progressHandler,
                    continuation: continuation
                )
                enqueueDecodeRow(row, model: model, tokenizerAndTemplate: tokenizerAndTemplate)
            }
        } onCancel: { [weak self] in
            guard let self else { return }
            Task {
                await self.cancelDecodeRow(id: rowID)
            }
        }
    }
    func decodeTokensPipelined(
        model: Q35Model,
        tokenizerAndTemplate: Q35TokenizerAndTemplate,
        initialLogits: MLXArray,
        layerCaches: [Q35LayerCache?],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        mropeRopeDelta: Int?,
        promptTokens: [Int],
        stopAtCompletedToolCall: Bool,
        logprobCapture: ChatLogprobCapture,
        logprobRegion: ChatLogprobRegion,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35BatchedDecodeResult {
        let layerCaches = layerCaches
        let banMask = tokenBanMask(
            vocabularySize: initialLogits.dim(-1),
            dtype: initialLogits.dtype,
            tokens: generationConfig.bannedTokens
        )

        // Shared pipelined decode; mRoPE positions derive from the cache
        // offset, so the step closure computes them per forward.
        var toolCallCompletionDetector = Q35ToolParser.StreamingCompletionDetector()
        let result = try AutoregressiveDecodeEngine.decode(
            AutoregressiveDecodeRequest(
                initialLogits: initialLogits,
                generationConfig: generationConfig,
                eosTokens: eosSet,
                tokenBudget: tokenBudget,
                historySeedTokens: promptTokens,
                banMask: banMask,
                logprobCapture: logprobCapture,
                logprobRegion: logprobRegion
            ),
            stepForward: { token in
                if model.config.textConfig.isQwen4Exp {
                    clearMLXCacheUnderPressureIfNeeded()
                }
                return model(
                    token,
                    cache: layerCaches,
                    positionIds: decodePositionIds(layerCaches: layerCaches, tokenCount: 1, ropeDelta: mropeRopeDelta)
                )
            },
            decodeToken: { tokenizerAndTemplate.decode(token: $0) },
            emitPiece: { _, piece in
                progressHandler?(ChatProgress(stage: .generating, message: piece))
            },
            shouldContinue: { _, piece in
                guard stopAtCompletedToolCall else { return true }
                return !toolCallCompletionDetector.feed(piece)
            },
            checkCancellation: { try Task.checkCancellation() }
        )

        if Gemma4DecodeTrace.enabled, !result.generatedTokens.isEmpty {
            let count = Double(result.generatedTokens.count)
            Gemma4DecodeTrace.emit(String(
                format: "[q35-decode-trace] mode=pipelined temp=\(generationConfig.temperature) tokens=%d build=%.2fms/tok wait=%.2fms/tok wall=%.2fms/tok",
                result.generatedTokens.count,
                result.buildSeconds / count * 1000,
                result.waitSeconds / count * 1000,
                result.decodeSeconds / count * 1000
            ))
        }

        return Q35BatchedDecodeResult(
            generatedTokens: result.generatedTokens,
            decodeSeconds: result.decodeSeconds,
            firstTokenSeconds: result.firstTokenSeconds,
            logprobs: result.logprobs,
            acceleration: ChatAccelerationDiagnostics(route: "final-target-pipelined")
        )
    }
    func lastTokenHidden(_ hidden: MLXArray) -> MLXArray {
        let start = max(0, hidden.dim(1) - 1)
        return hidden[0..., start..<(start + 1), 0...]
    }

    func lastTokenLogits(_ logits: MLXArray) -> MLXArray {
        let start = max(0, logits.dim(1) - 1)
        return logits[0..., start..<(start + 1), 0...]
    }

    private func greedyTokenArray(from logits: MLXArray) -> MLXArray {
        argMax(logits[0, -1, 0...], axis: -1).asType(.int32)
    }
}
