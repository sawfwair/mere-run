import Foundation
import MediaIO
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

extension Q35Generator {
    func decodeTokensSerially(
        model: Q35Model,
        tokenizerAndTemplate: Q35TokenizerAndTemplate,
        initialLogits: MLXArray,
        initialHidden: MLXArray?,
        prefillMTPHidden: MLXArray?,
        prefillMTPSession: Q35MTPDraftSession?,
        mtpModel: (any Q35MTPDraftModel)?,
        layerCaches: [Q35LayerCache?],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        prefillTokenCount: Int,
        mropeRopeDelta: Int?,
        promptTokens: [Int],
        jsonConstrained: Bool,
        stopAtCompletedToolCall: Bool,
        logprobCapture: ChatLogprobCapture,
        logprobRegion: ChatLogprobRegion,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35BatchedDecodeResult {
        if mtpModel == nil, !jsonConstrained {
            return try await decodeTokensPipelined(
                model: model,
                tokenizerAndTemplate: tokenizerAndTemplate,
                initialLogits: initialLogits,
                layerCaches: layerCaches,
                eosSet: eosSet,
                generationConfig: generationConfig,
                tokenBudget: tokenBudget,
                mropeRopeDelta: mropeRopeDelta,
                promptTokens: promptTokens,
                stopAtCompletedToolCall: stopAtCompletedToolCall,
                logprobCapture: logprobCapture,
                logprobRegion: logprobRegion,
                progressHandler: progressHandler
            )
        }

        var logits = initialLogits
        var layerCaches = layerCaches
        let retainHidden = mtpModel != nil
        var previousHidden = retainHidden ? initialHidden.map(lastTokenHidden) : nil
        var generated: [Int] = []
        generated.reserveCapacity(tokenBudget)
        var repetitionHistory = promptTokens
        var pendingProgressWhitespace = ""
        var streamedJSONText = ""
        var firstTokenSeconds: Double?
        var jsonGrammar = JSONObjectPrefixGrammar()
        var mtpDraftedTokens = 0
        var mtpAcceptedTokens = 0
        var mtpVerificationPasses = 0
        var mtpReplacementPasses = 0
        var mtpNonDraftingRounds = 0
        var usePipelinedFallback = false
        let supportsPipelinedFallback = modelId == Q35Resources.q38TwentySevenB4BitModelId
            || modelId == Q35Resources.ornith35BMLX4BitModelId
        let pipelinedFallbackEnabled = Q35RuntimeTuning.isEnabled(.pipelinedFallback, modelID: modelId)
        let mtpBlockSize = Self.mtpBlockSize(modelId: modelId)
        var mtpAdaptivePolicy = Q35MTPAdaptivePolicy(
            maxDraftDepth: mtpBlockSize - 1, headStepCostRatio: Q35MTPAdaptivePolicy.configuredCostRatio()
        )
        let mtpDraftSession = prefillMTPSession ?? Q35MTPDraftSession(
            promptTokens: promptTokens,
            promptHidden: prefillMTPHidden,
            historyCache: model.config.textConfig.isQwen4Exp ? Q38QSACache() : KVCacheSimple()
        )
        let draftHistoryTokens = mtpDraftSession.committedHistoryCount
        let mtpProfile = generationConfig.temperature == 0 && mtpModel != nil ? Q35MTPProfile.make() : nil
        let decodeStart = Date()

        func emit(_ token: Int) {
            generated.append(token)
            repetitionHistory.append(token)
            if firstTokenSeconds == nil {
                firstTokenSeconds = Date().timeIntervalSince(decodeStart)
            }
            guard let progressHandler else { return }
            if jsonConstrained {
                // Byte-fallback BPE tokens can decode individually as U+FFFD even
                // though the cumulative token sequence decodes to valid Unicode.
                // Stream only the stable cumulative prefix and wait for trailing
                // replacement scalars to resolve before exposing them.
                var stableText = tokenizerAndTemplate.decode(tokens: generated)
                while stableText.last == "\u{FFFD}" {
                    stableText.removeLast()
                }
                guard stableText.hasPrefix(streamedJSONText) else { return }
                let deltaStart = stableText.index(
                    stableText.startIndex,
                    offsetBy: streamedJSONText.count
                )
                let delta = String(stableText[deltaStart...])
                streamedJSONText = stableText
                if !delta.isEmpty {
                    progressHandler(ChatProgress(stage: .generating, message: delta))
                }
                return
            }
            let piece = tokenizerAndTemplate.decode(token: token)
            guard !piece.isEmpty else { return }
            if piece.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                pendingProgressWhitespace += piece
                return
            }

            let visiblePiece: String
            if pendingProgressWhitespace.isEmpty {
                visiblePiece = piece
            } else {
                visiblePiece = pendingProgressWhitespace + piece
                pendingProgressWhitespace = ""
            }
            progressHandler(ChatProgress(stage: .generating, message: visiblePiece))
        }

        while generated.count < tokenBudget {
            try Task.checkCancellation()
            if model.config.textConfig.isQwen4Exp {
                clearMLXCacheUnderPressureIfNeeded()
            }
            let samplingStart = mtpProfile?.clock()
            var next = sampleToken(
                logits: logits[0, -1, 0...],
                config: generationConfig,
                previousTokens: repetitionHistory
            )

            mtpProfile?.recordSampling(since: samplingStart)
            if jsonConstrained {
                guard let constrained = jsonConstrainedToken(
                    initial: next,
                    logits: applyingSamplingPenalties(
                        logits[0, -1, 0...], config: generationConfig,
                        history: repetitionHistoryArray(promptTokens: repetitionHistory, config: generationConfig)
                    ),
                    config: generationConfig,
                    eosSet: eosSet,
                    grammar: &jsonGrammar,
                    decode: { tokenizerAndTemplate.decode(token: $0) }
                ) else {
                    break
                }
                next = constrained
            }

            if eosSet.contains(next) {
                break
            }

            emit(next)

            if jsonConstrained, jsonGrammar.isComplete {
                break
            }

            guard generated.count < tokenBudget else {
                break
            }

            if let mtpModel, let hidden = previousHidden {
                let positionOffset = prefillTokenCount + generated.count - 1
                if generationConfig.temperature == 0 {
                    let offeredDepth = min(
                        mtpBlockSize - 1,
                        tokenBudget - generated.count
                    )
                    let draftDepth = mtpAdaptivePolicy.draftDepth(offeredDepth: offeredDepth)
                    if draftDepth > 0 {
                        mtpProfile?.beginRound(depth: draftDepth)
                        defer { mtpProfile?.finishRound() }
                        let draftBlock = mtpModel.draftBlock(
                            lastToken: next,
                            hidden: hidden,
                            blockSize: draftDepth + 1,
                            session: mtpDraftSession,
                            baseModel: model
                        )
                        mtpProfile?.finishDraft(tokens: draftBlock.tokenIDs)
                        mtpDraftedTokens += draftBlock.count

                        let candidateCaches = forkLayerCaches(layerCaches)
                        let nextToken = MLXArray([Int32(next)]).reshaped(1, 1)
                        let candidateInput = MLX.concatenated(
                            [nextToken, draftBlock.tokenIDs],
                            axis: 1
                        )
                        let candidate = model.forward(
                            candidateInput,
                            cache: candidateCaches,
                            targetVerify: true
                        )
                        let candidateMTPHidden = candidate.mtpHidden ?? candidate.hidden
                        mtpProfile?.verificationSubmitted()
                        MLX.eval(candidate.logits, candidateMTPHidden, draftBlock.tokenIDs)
                        mtpProfile?.verificationCompleted()
                        mtpVerificationPasses += 1
                        let draftTokens = draftBlock.tokens

                        var accepted = 0
                        var verificationHistory = repetitionHistory
                        var replacement: Int?
                        for (index, draftToken) in draftTokens.enumerated() {
                            let targetToken = sampleToken(
                                logits: candidate.logits[0, index, 0...],
                                config: generationConfig,
                                previousTokens: verificationHistory
                            )
                            guard targetToken == draftToken else {
                                replacement = targetToken
                                break
                            }
                            accepted += 1
                            verificationHistory.append(draftToken)
                        }
                        mtpProfile?.accepted(accepted)
                        mtpAdaptivePolicy.record(
                            acceptedDrafts: accepted,
                            drafted: draftTokens.count
                        )

                        if accepted == draftTokens.count {
                            mtpAcceptedTokens += accepted
                            mtpDraftSession.recordCommittedTransitions(
                                hiddenStates: candidateMTPHidden,
                                nextTokens: draftTokens
                            )
                            var hitEOS = false
                            for token in draftTokens {
                                if eosSet.contains(token) {
                                    hitEOS = true
                                    break
                                }
                                emit(token)
                            }
                            commitVerificationCaches(candidateCaches)
                            layerCaches = candidateCaches
                            logits = lastTokenLogits(candidate.logits)
                            previousHidden = lastTokenHidden(candidateMTPHidden)
                            if hitEOS || generated.count >= tokenBudget {
                                break
                            }
                            continue
                        }

                        let acceptedPrefix = Array(draftTokens.prefix(accepted))
                        mtpAcceptedTokens += accepted
                        var hitEOS = false
                        for token in acceptedPrefix {
                            if eosSet.contains(token) {
                                hitEOS = true
                                break
                            }
                            emit(token)
                        }
                        if hitEOS || generated.count >= tokenBudget {
                            break
                        }

                        guard let replacement else {
                            continue
                        }
                        if eosSet.contains(replacement) {
                            break
                        }

                        let committedTokenCount = accepted + 1
                        if restoreVerificationCaches(
                            candidateCaches,
                            totalTokens: draftTokens.count + 1,
                            tokenCount: committedTokenCount
                        ) {
                            let restored = mtpDraftSession.restoredVerificationState(
                                from: candidate,
                                acceptedTokens: acceptedPrefix
                            )
                            layerCaches = candidateCaches
                            logits = restored.logits
                            previousHidden = restored.hidden
                            continue
                        }

                        let replacementCaches = forkLayerCaches(layerCaches)
                        let replacementTokens = [next] + acceptedPrefix + [replacement]
                        let replacementInputValues = replacementTokens.map { Int32($0) }
                        let replacementInput = MLXArray(replacementInputValues)
                            .reshaped(1, acceptedPrefix.count + 2)
                        let replacementForward = model.forward(replacementInput, cache: replacementCaches)
                        let replacementMTPHidden = replacementForward.mtpHidden ?? replacementForward.hidden
                        MLX.eval(replacementForward.logits)
                        MLX.eval(replacementMTPHidden)
                        mtpReplacementPasses += 1
                        mtpDraftSession.recordCommittedTransitions(
                            hiddenStates: replacementMTPHidden,
                            nextTokens: acceptedPrefix + [replacement]
                        )
                        emit(replacement)
                        layerCaches = replacementCaches
                        logits = lastTokenLogits(replacementForward.logits)
                        previousHidden = lastTokenHidden(replacementMTPHidden)
                        continue
                    }

                    mtpNonDraftingRounds += 1
                    // Once the policy reaches zero, no more acceptance evidence
                    // can change it. Finish through the existing pipelined
                    // target decoder after consuming this emitted token.
                    usePipelinedFallback = supportsPipelinedFallback && pipelinedFallbackEnabled
                    mtpDraftSession.recordCommittedTransitions(
                        hiddenStates: hidden,
                        nextTokens: [next]
                    )
                } else {
                    let draftLogits = mtpModel.draftLogits(
                        token: next,
                        previousHidden: hidden,
                        positionOffset: positionOffset,
                        baseModel: model
                    )
                    MLX.eval(draftLogits)
                    mtpDraftedTokens += 1

                    let draftProbs = samplingProbabilities(
                        logits: draftLogits[0, -1, 0...],
                        config: generationConfig,
                        previousTokens: repetitionHistory
                    )
                    let draft = sampleToken(probabilities: draftProbs)

                    let candidateCaches = forkLayerCaches(layerCaches)
                    let candidateInput = MLXArray([Int32(next), Int32(draft)]).reshaped(1, 2)
                    let candidate = model.forward(
                        candidateInput,
                        cache: candidateCaches,
                        targetVerify: true
                    )
                    let candidateMTPHidden = candidate.mtpHidden ?? candidate.hidden
                    MLX.eval(candidate.logits)
                    MLX.eval(candidateMTPHidden)
                    mtpVerificationPasses += 1

                    let targetProbs = samplingProbabilities(
                        logits: candidate.logits[0, 0, 0...],
                        config: generationConfig,
                        previousTokens: repetitionHistory
                    )
                    let draftProb = max(draftProbs[draft].item(Float.self), Float.leastNonzeroMagnitude)
                    let targetProb = targetProbs[draft].item(Float.self)
                    let acceptProbability = min(1.0, targetProb / draftProb)

                    if Q35Sampling.acceptsDraft(probability: acceptProbability) {
                        mtpAcceptedTokens += 1
                        if eosSet.contains(draft) {
                            break
                        }
                        emit(draft)
                        commitVerificationCaches(candidateCaches)
                        layerCaches = candidateCaches
                        logits = lastTokenLogits(candidate.logits)
                        previousHidden = lastTokenHidden(candidateMTPHidden)
                        continue
                    }

                    let residualProbs = MLX.maximum(targetProbs - draftProbs, MLXArray(0.0))
                    let residualMass = residualProbs.sum().item(Float.self)
                    let replacement = residualMass > 1e-6
                        ? sampleToken(probabilities: residualProbs / residualProbs.sum())
                        : sampleToken(probabilities: targetProbs)
                    if eosSet.contains(replacement) {
                        break
                    }

                    let replacementCaches = forkLayerCaches(layerCaches)
                    let replacementInput = MLXArray([Int32(next), Int32(replacement)]).reshaped(1, 2)
                    let replacementForward = model.forward(replacementInput, cache: replacementCaches)
                    let replacementMTPHidden = replacementForward.mtpHidden ?? replacementForward.hidden
                    MLX.eval(replacementForward.logits)
                    MLX.eval(replacementMTPHidden)
                    mtpReplacementPasses += 1
                    emit(replacement)
                    layerCaches = replacementCaches
                    logits = lastTokenLogits(replacementForward.logits)
                    previousHidden = lastTokenHidden(replacementMTPHidden)
                    continue
                }
            }

            let serialStart = mtpProfile?.clock()
            let nextInput = MLXArray([Int32(next)]).reshaped(1, 1)
            let positionIds = decodePositionIds(layerCaches: layerCaches, tokenCount: 1, ropeDelta: mropeRopeDelta)
            if retainHidden {
                let output = model.forward(
                    nextInput,
                    cache: layerCaches,
                    positionIds: positionIds
                )
                let outputMTPHidden = output.mtpHidden ?? output.hidden
                logits = output.logits
                previousHidden = lastTokenHidden(outputMTPHidden)
                MLX.eval(logits)
                MLX.eval(previousHidden!)
            } else {
                logits = model(
                    nextInput,
                    cache: layerCaches,
                    positionIds: positionIds
                )
                MLX.eval(logits)
            }
            mtpProfile?.recordSerial(since: serialStart)
            if usePipelinedFallback {
                if !pendingProgressWhitespace.isEmpty {
                    progressHandler?(ChatProgress(stage: .generating, message: pendingProgressWhitespace))
                }
                let tail = try await decodeTokensPipelined(
                    model: model, tokenizerAndTemplate: tokenizerAndTemplate,
                    initialLogits: logits, layerCaches: layerCaches, eosSet: eosSet,
                    generationConfig: generationConfig, tokenBudget: tokenBudget - generated.count,
                    mropeRopeDelta: mropeRopeDelta, promptTokens: repetitionHistory,
                    stopAtCompletedToolCall: false, logprobCapture: logprobCapture,
                    logprobRegion: logprobRegion, progressHandler: progressHandler
                )
                generated.append(contentsOf: tail.generatedTokens)
                mtpNonDraftingRounds += tail.generatedTokens.count
                mtpProfile?.recordPipeline(seconds: tail.decodeSeconds, tokens: tail.generatedTokens.count)
                break
            }
        }

        let decodeSeconds = Date().timeIntervalSince(decodeStart)
        if Gemma4DecodeTrace.enabled, mtpModel != nil {
            let acceptance = mtpDraftedTokens > 0
                ? Double(mtpAcceptedTokens) / Double(mtpDraftedTokens) * 100
                : 0
            Gemma4DecodeTrace.emit(String(
                format: "[q35-decode-trace] mode=mtp tokens=%d drafted=%d accepted=%d acceptance=%.1f%% verify=%d replacement=%d serial=%d wall=%.2fms/tok",
                generated.count,
                mtpDraftedTokens,
                mtpAcceptedTokens,
                acceptance,
                mtpVerificationPasses,
                mtpReplacementPasses,
                mtpNonDraftingRounds,
                decodeSeconds / Double(max(1, generated.count)) * 1000
            ))
        }
        return Q35BatchedDecodeResult(
            generatedTokens: generated,
            decodeSeconds: decodeSeconds,
            firstTokenSeconds: firstTokenSeconds,
            acceleration: mtpModel.map { _ in
                ChatAccelerationDiagnostics(
                    route: "mtp-speculative",
                    draftModel: mtpModel?.diagnosticsID ?? "qwen-mtp",
                    rounds: mtpVerificationPasses,
                    draftedTokens: mtpDraftedTokens,
                    acceptedDraftTokens: mtpAcceptedTokens,
                    draftHistoryTokens: draftHistoryTokens,
                    speculationProfile: mtpProfile?.result
                )
            } ?? ChatAccelerationDiagnostics(
                route: jsonConstrained ? "json-constrained-serial" : "serial"
            )
        )
    }
}
