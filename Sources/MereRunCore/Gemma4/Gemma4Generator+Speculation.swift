import Foundation
import MLX
import MLXNN

extension Gemma4Generator {
    func decodeTokensSerially(
        model: any Gemma4CausalModel,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate,
        initialLogits: MLXArray,
        layerCaches: [Gemma4AttentionCache],
        eosSet: Set<Int>,
        generationConfig: GenerationConfig,
        tokenBudget: Int,
        promptTokens: [Int],
        mtpModel: Gemma4AssistantDraftModel?,
        prefillHidden: MLXArray?,
        sharedKVStates: [String: Gemma4SharedKVState],
        prefillTokenCount: Int,
        mtpStatsTemplate: Gemma4MTPStats,
        jsonConstrained: Bool = false,
        noRepeatNgramSize: Int? = nil,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Gemma4BatchedDecodeResult {
        if noRepeatNgramSize == nil, canUsePipelinedDecode(
            generationConfig,
            mtpModel: mtpModel,
            jsonConstrained: jsonConstrained
        ) {
            return try await decodeTokensPipelined(
                model: model,
                tokenizerAndTemplate: tokenizerAndTemplate,
                initialLogits: initialLogits,
                layerCaches: layerCaches,
                eosSet: eosSet,
                generationConfig: generationConfig,
                tokenBudget: tokenBudget,
                promptTokens: promptTokens,
                mtpStatsTemplate: mtpStatsTemplate,
                progressHandler: progressHandler
            )
        }

        var logits = initialLogits
        var layerCaches = layerCaches
        var generated: [Int] = []
        generated.reserveCapacity(tokenBudget)
        var repetitionHistory = promptTokens
        var previousHidden = prefillHidden.map { model.speculativeDraftHidden(lastTokenHidden($0)) }
        var currentSharedKVStates = sharedKVStates
        var mtpStats = mtpStatsTemplate
        var firstTokenSeconds: Double?
        var pendingSampledToken: Int?
        var jsonGrammar = JSONObjectPrefixGrammar()
        let decodeStart = Date()
        let traceEnabled = Gemma4DecodeTrace.enabled
        var traceSampleSeconds = 0.0
        var traceForwardSeconds = 0.0

        while generated.count < tokenBudget {
            try Task.checkCancellation()
            var next: Int
            if let pending = pendingSampledToken {
                next = pending
                pendingSampledToken = nil
            } else {
                let sampleStart = CFAbsoluteTimeGetCurrent()
                var samplingConfig = generationConfig
                if let noRepeatNgramSize {
                    samplingConfig.bannedTokens.append(contentsOf:
                        Self.noRepeatNgramBannedTokens(
                            history: repetitionHistory,
                            size: noRepeatNgramSize
                        )
                    )
                }
                next = sampleToken(
                    logits: logits[0, -1, 0...],
                    config: samplingConfig,
                    previousTokens: repetitionHistory
                )
                if traceEnabled {
                    traceSampleSeconds += CFAbsoluteTimeGetCurrent() - sampleStart
                }
            }
            if jsonConstrained {
                guard let constrained = jsonConstrainedToken(
                    initial: next,
                    logits: logits[0, -1, 0...],
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

            generated.append(next)
            if firstTokenSeconds == nil {
                firstTokenSeconds = Date().timeIntervalSince(decodeStart)
            }
            repetitionHistory.append(next)
            if let progressHandler {
                let piece = tokenizerAndTemplate.decode(token: next)
                if !piece.isEmpty {
                    progressHandler(ChatProgress(stage: .generating, message: piece))
                }
            }

            if jsonConstrained, jsonGrammar.isComplete {
                break
            }

            guard generated.count < tokenBudget else {
                break
            }

            var draftTokens: [Int] = []
            var speculationBaseCaches: [Gemma4AttentionCache]?
            if let mtpModel, let hidden = previousHidden, !jsonConstrained, !currentSharedKVStates.isEmpty {
                let blockSize = min(
                    mtpStats.blockSize,
                    max(2, tokenBudget - generated.count + 1)
                )
                let positionOffset = prefillTokenCount + generated.count - 1
                // For sampled requests the drafts are generated greedily: the
                // verify loop samples the target either way, so correctness is
                // unaffected, and matching the target's argmax maximizes the
                // acceptance rate (sampled drafts collapse it to sum(p*q)).
                var draftConfig = generationConfig
                draftConfig.temperature = 0
                let draft = try mtpModel.draftBlock(
                    lastToken: next,
                    hidden: hidden,
                    sharedKVStates: currentSharedKVStates,
                    positionOffset: positionOffset,
                    blockSize: blockSize,
                    baseModel: model,
                    generationConfig: draftConfig,
                    repetitionHistory: repetitionHistory
                )
                draftTokens = draft.tokens
                speculationBaseCaches = layerCaches
            }
            if let baseCaches = speculationBaseCaches, !draftTokens.isEmpty {
                do {
                    mtpStats.rounds += 1
                    mtpStats.draftedTokens += draftTokens.count
                    let candidateCaches = forkLayerCaches(baseCaches)
                    let candidateInput = MLXArray(([next] + draftTokens).map(Int32.init))
                        .reshaped(1, draftTokens.count + 1)
                    let candidate = model.forwardForSpeculation(
                        inputIds: candidateInput,
                        cache: candidateCaches
                    )

                    // Sample every verify position — the draft checks plus the
                    // bonus token after a full accept — in one graph with one
                    // readback, instead of a blocking sample per position.
                    // Histories are prospective: position i verifies against
                    // history + draft[0..<i], which matches the serial loop for
                    // every position at or before the first mismatch (later
                    // samples go unused).
                    let verifyBanMask = tokenBanMask(
                        vocabularySize: candidate.logits.dim(-1),
                        dtype: candidate.logits.dtype,
                        tokens: generationConfig.bannedTokens
                    )
                    var verifySampleArrays: [MLXArray] = []
                    verifySampleArrays.reserveCapacity(draftTokens.count + 1)
                    var prospectiveHistory = repetitionHistory
                    for index in 0...draftTokens.count {
                        verifySampleArrays.append(sampledTokenArray(
                            logits: candidate.logits[0, index, 0...],
                            config: generationConfig,
                            previousTokenIndices: repetitionHistoryArray(
                                promptTokens: prospectiveHistory,
                                config: generationConfig
                            ),
                            banMask: verifyBanMask
                        ))
                        if index < draftTokens.count {
                            prospectiveHistory.append(draftTokens[index])
                        }
                    }
                    let stackedVerify = MLX.stacked(verifySampleArrays)
                    MLX.eval(stackedVerify, candidate.hidden)
                    let verifySamples = stackedVerify.asArray(Int32.self).map(Int.init)

                    var accepted = 0
                    var replacement: Int?
                    for (index, draftToken) in draftTokens.enumerated() {
                        guard verifySamples[index] == draftToken else {
                            replacement = verifySamples[index]
                            mtpStats.rejectedTokens += 1
                            break
                        }
                        accepted += 1
                    }

                    if accepted == draftTokens.count {
                        var hitEOS = false
                        for token in draftTokens {
                            if eosSet.contains(token) {
                                hitEOS = true
                                break
                            }
                            generated.append(token)
                            repetitionHistory.append(token)
                            mtpStats.acceptedTokens += 1
                            if let progressHandler {
                                let tokenPiece = tokenizerAndTemplate.decode(token: token)
                                if !tokenPiece.isEmpty {
                                    progressHandler(ChatProgress(stage: .generating, message: tokenPiece))
                                }
                            }
                        }
                        layerCaches = candidateCaches
                        logits = lastTokenLogits(candidate.logits)
                        previousHidden = model.speculativeDraftHidden(lastTokenHidden(candidate.hidden))
                        currentSharedKVStates = candidate.sharedKVStates
                        if hitEOS || generated.count >= tokenBudget {
                            break
                        }
                        // The bonus token was already sampled in the batched
                        // verify pass (last position, full-draft history).
                        pendingSampledToken = verifySamples[draftTokens.count]
                        if let previousHidden {
                            MLX.eval(previousHidden)
                        }
                        continue
                    }

                    let acceptedPrefix = Array(draftTokens.prefix(accepted))
                    var hitEOS = false
                    for token in acceptedPrefix {
                        if eosSet.contains(token) {
                            hitEOS = true
                            break
                        }
                        generated.append(token)
                        repetitionHistory.append(token)
                        mtpStats.acceptedTokens += 1
                        if let progressHandler {
                            let tokenPiece = tokenizerAndTemplate.decode(token: token)
                            if !tokenPiece.isEmpty {
                                progressHandler(ChatProgress(stage: .generating, message: tokenPiece))
                            }
                        }
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
                    generated.append(replacement)
                    repetitionHistory.append(replacement)
                    if let progressHandler {
                        let replacementPiece = tokenizerAndTemplate.decode(token: replacement)
                        if !replacementPiece.isEmpty {
                            progressHandler(ChatProgress(stage: .generating, message: replacementPiece))
                        }
                    }

                    let replacementCaches = forkLayerCaches(baseCaches)
                    let replacementTokens = [next] + acceptedPrefix + [replacement]
                    let replacementInputValues = replacementTokens.map { Int32($0) }
                    let replacementInput = MLXArray(replacementInputValues)
                        .reshaped(1, acceptedPrefix.count + 2)
                    let replacementForward = model.forwardForSpeculation(
                        inputIds: replacementInput,
                        cache: replacementCaches
                    )
                    MLX.eval(replacementForward.logits, replacementForward.hidden)
                    layerCaches = replacementCaches
                    logits = lastTokenLogits(replacementForward.logits)
                    previousHidden = model.speculativeDraftHidden(lastTokenHidden(replacementForward.hidden))
                    currentSharedKVStates = replacementForward.sharedKVStates
                    if let previousHidden {
                        MLX.eval(logits, previousHidden)
                    }
                    continue
                }
            }

            let forwardStart = CFAbsoluteTimeGetCurrent()
            let nextInput = MLXArray([Int32(next)]).reshaped(1, 1)
            if mtpModel != nil {
                let output = model.forwardForSpeculation(inputIds: nextInput, cache: layerCaches)
                logits = output.logits
                previousHidden = model.speculativeDraftHidden(lastTokenHidden(output.hidden))
                currentSharedKVStates = output.sharedKVStates
                MLX.eval(logits, previousHidden!)
            } else {
                logits = model.forward(inputIds: nextInput, cache: layerCaches)
                MLX.eval(logits)
            }
            if traceEnabled {
                traceForwardSeconds += CFAbsoluteTimeGetCurrent() - forwardStart
            }
        }

        if traceEnabled, !generated.isEmpty {
            let count = Double(generated.count)
            Gemma4DecodeTrace.emit(String(
                format: "[gemma4-decode-trace] mode=serial mtp=%d tokens=%d sample=%.2fms/tok forward=%.2fms/tok wall=%.2fms/tok",
                mtpModel == nil ? 0 : 1,
                generated.count,
                traceSampleSeconds / count * 1000,
                traceForwardSeconds / count * 1000,
                Date().timeIntervalSince(decodeStart) / count * 1000
            ))
        }

        return Gemma4BatchedDecodeResult(
            generatedTokens: generated,
            decodeSeconds: Date().timeIntervalSince(decodeStart),
            firstTokenSeconds: firstTokenSeconds,
            mtpStats: mtpStats
        )
    }
}
