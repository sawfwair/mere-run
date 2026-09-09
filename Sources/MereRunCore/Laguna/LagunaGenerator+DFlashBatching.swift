import Foundation
import MLX

extension LagunaGenerator {
    func enqueueDFlashDecodeRow(
        _ row: LagunaDFlashBatchedDecodeRow,
        model: LagunaCausalLM,
        dflash: LagunaDFlashModel,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) {
        dflashDecodeQueue.append(row)
        startDFlashDecodeLoopIfNeeded(
            model: model,
            dflash: dflash,
            tokenizerAndTemplate: tokenizerAndTemplate
        )
    }

    func cancelDFlashDecodeRow(id: UUID) {
        if let index = dflashDecodeQueue.firstIndex(where: { $0.id == id }) {
            dflashDecodeQueue.remove(at: index).fail(CancellationError())
            return
        }
        if let index = activeDFlashDecodeRows.firstIndex(where: { $0.id == id }) {
            activeDFlashDecodeRows.remove(at: index).fail(CancellationError())
        }
    }

    func startDFlashDecodeLoopIfNeeded(
        model: LagunaCausalLM,
        dflash: LagunaDFlashModel,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) {
        guard !dflashDecodeLoopRunning else { return }
        dflashDecodeLoopRunning = true
        Task {
            await runDFlashDecodeLoop(
                model: model,
                dflash: dflash,
                tokenizerAndTemplate: tokenizerAndTemplate
            )
        }
    }

    func runDFlashDecodeLoop(
        model: LagunaCausalLM,
        dflash: LagunaDFlashModel,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) async {
        defer {
            dflashDecodeLoopRunning = false
            if !dflashDecodeQueue.isEmpty || !activeDFlashDecodeRows.isEmpty {
                startDFlashDecodeLoopIfNeeded(
                    model: model,
                    dflash: dflash,
                    tokenizerAndTemplate: tokenizerAndTemplate
                )
            }
        }

        while !dflashDecodeQueue.isEmpty || !activeDFlashDecodeRows.isEmpty {
            if activeDFlashDecodeRows.isEmpty {
                try? await Task.sleep(nanoseconds: 1_000_000)
            }
            activeDFlashDecodeRows.append(contentsOf: dflashDecodeQueue)
            dflashDecodeQueue.removeAll(keepingCapacity: true)
            let rows = activeDFlashDecodeRows.filter(\.needsDecodeRound)
            guard !rows.isEmpty else {
                finishCompletedDFlashDecodeRows()
                continue
            }
            do {
                try decodeOneDFlashRound(
                    rows: rows,
                    model: model,
                    dflash: dflash,
                    tokenizerAndTemplate: tokenizerAndTemplate
                )
            } catch {
                failDFlashRows(rows, with: error)
            }
            finishCompletedDFlashDecodeRows()
            await Task.yield()
        }
    }

    func decodeOneDFlashRound(
        rows: [LagunaDFlashBatchedDecodeRow],
        model: LagunaCausalLM,
        dflash: LagunaDFlashModel,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) throws {
        for row in rows {
            let anchor = sampleToken(
                logits: row.logits[0, -1, 0...],
                config: row.generationConfig,
                previousTokens: row.repetitionHistory
            )
            if row.eosTokens.contains(anchor) {
                row.stopped = true
            } else {
                emitDFlashToken(
                    anchor,
                    row: row,
                    tokenizerAndTemplate: tokenizerAndTemplate
                )
            }
        }

        let continuingRows = rows.filter(\.needsDecodeRound)
        guard !continuingRows.isEmpty else { return }
        let draftCount = min(
            dflashSpeculativeTokens,
            min(
                dflash.config.dflash.blockSize - 1,
                continuingRows.map {
                    $0.tokenBudget - $0.generatedTokens.count
                }.min()!
            )
        )
        precondition(draftCount > 0)

        let draftCandidateRows = continuingRows.map {
            $0.draftCaches.map { $0.fork() }
        }
        guard let batchedDraftCaches = makeBatchedCaches(draftCandidateRows) else {
            throw LagunaError.generationFailed(
                "Laguna could not batch DFlash draft caches."
            )
        }
        let anchorTokens = MLXArray(
            continuingRows.map { Int32($0.generatedTokens.last!) }
        ).reshaped(continuingRows.count, 1)
        let draftLogits = dflash.draftLogits(
            anchorTokens: anchorTokens,
            speculativeTokenCount: draftCount,
            cache: batchedDraftCaches,
            target: model
        )
        MLX.eval(draftLogits)

        var proposals: [[Int]] = []
        var proposalProbabilities: [[MLXArray]] = []
        proposals.reserveCapacity(continuingRows.count)
        proposalProbabilities.reserveCapacity(continuingRows.count)
        for (rowIndex, row) in continuingRows.enumerated() {
            var rowProposals: [Int] = []
            var rowProbabilities: [MLXArray] = []
            var history = row.repetitionHistory
            for draftIndex in 0..<draftCount {
                let proposalLogits = draftLogits[rowIndex, draftIndex, 0...]
                if row.generationConfig.temperature == 0 {
                    let token = sampleToken(
                        logits: proposalLogits,
                        config: row.generationConfig,
                        previousTokens: history
                    )
                    rowProposals.append(token)
                    history.append(token)
                } else {
                    let probabilities = samplingProbabilities(
                        logits: proposalLogits,
                        config: row.generationConfig,
                        previousTokens: history
                    )
                    let token = sampleToken(probabilities: probabilities)
                    rowProposals.append(token)
                    rowProbabilities.append(probabilities)
                    history.append(token)
                }
            }
            proposals.append(rowProposals)
            proposalProbabilities.append(rowProbabilities)
        }

        let targetCandidateRows = continuingRows.map {
            $0.targetCaches.map { $0.fork() }
        }
        guard let batchedTargetCaches = makeBatchedCaches(targetCandidateRows) else {
            throw LagunaError.generationFailed(
                "Laguna could not batch DFlash verification caches."
            )
        }
        let candidateInput = MLXArray(
            zip(continuingRows, proposals).flatMap { row, rowProposals in
                [Int32(row.generatedTokens.last!)] + rowProposals.map(Int32.init)
            }
        ).reshaped(continuingRows.count, draftCount + 1)
        let candidate = model.forward(
            candidateInput,
            cache: batchedTargetCaches,
            captureLayerIndices: Set(dflash.config.dflash.targetLayerIDs)
        )
        MLX.eval([candidate.logits] + Array(candidate.capturedHiddenStates.values))
        guard let candidateCaches = splitBatchedCaches(
            batchedTargetCaches,
            rowCount: continuingRows.count
        ) else {
            throw LagunaError.generationFailed(
                "Laguna could not split DFlash verification caches."
            )
        }

        dflashRounds += continuingRows.count
        dflashDraftedTokens += continuingRows.count * draftCount
        dflashTargetVerificationForwards += 1
        recordBatchedForward(
            positions: continuingRows.map {
                $0.targetCaches.map(\.offset).min() ?? 0
            }
        )

        var recoveries: [LagunaDFlashRecovery] = []
        for (rowIndex, row) in continuingRows.enumerated() {
            let rowProposals = proposals[rowIndex]
            var accepted = 0
            var replacement: Int?
            var history = row.repetitionHistory
            for (draftIndex, proposal) in rowProposals.enumerated() {
                let targetLogits = candidate.logits[rowIndex, draftIndex, 0...]
                if row.generationConfig.temperature == 0 {
                    let targetToken = sampleToken(
                        logits: targetLogits,
                        config: row.generationConfig,
                        previousTokens: history
                    )
                    guard targetToken == proposal else {
                        replacement = targetToken
                        break
                    }
                } else {
                    let targetProbabilities = samplingProbabilities(
                        logits: targetLogits,
                        config: row.generationConfig,
                        previousTokens: history
                    )
                    let draftProbabilities = proposalProbabilities[rowIndex][draftIndex]
                    let draftProbability = max(
                        draftProbabilities[proposal].item(Float.self),
                        Float.leastNonzeroMagnitude
                    )
                    let targetProbability = targetProbabilities[proposal]
                        .item(Float.self)
                    guard Float.random(in: 0..<1)
                        <= min(1, targetProbability / draftProbability) else {
                        replacement = sampleToken(probabilities:
                            LagunaDFlashDecoder.rejectionDistribution(
                                target: targetProbabilities,
                                draft: draftProbabilities
                            )
                        )
                        break
                    }
                }
                accepted += 1
                history.append(proposal)
            }
            dflashAcceptedDraftTokens += accepted

            let rowHiddenStates = Dictionary(
                uniqueKeysWithValues: dflash.config.dflash.targetLayerIDs.map { layerID in
                    (
                        layerID,
                        candidate.capturedHiddenStates[layerID]![
                            rowIndex..<(rowIndex + 1),
                            0...,
                            0...
                        ]
                    )
                }
            )
            if accepted == rowProposals.count {
                dflashFullAcceptanceRounds += 1
                for proposal in rowProposals {
                    if row.eosTokens.contains(proposal) {
                        row.stopped = true
                        break
                    }
                    emitDFlashToken(
                        proposal,
                        row: row,
                        tokenizerAndTemplate: tokenizerAndTemplate
                    )
                    if !row.needsDecodeRound {
                        break
                    }
                }
                guard row.needsDecodeRound else { continue }
                row.targetCaches = candidateCaches[rowIndex]
                row.logits = candidate.logits[
                    rowIndex..<(rowIndex + 1),
                    (candidate.logits.dim(1) - 1)...,
                    0...
                ]
                dflash.appendTargetContext(
                    dflash.combineTargetHiddenStates(rowHiddenStates),
                    cache: row.draftCaches
                )
                evaluateGemma4CacheStorage(row.draftCaches)
                continue
            }

            dflashRejectedDraftTokens += 1
            for proposal in rowProposals.prefix(accepted) {
                if row.eosTokens.contains(proposal) {
                    row.stopped = true
                    break
                }
                emitDFlashToken(
                    proposal,
                    row: row,
                    tokenizerAndTemplate: tokenizerAndTemplate
                )
                if !row.needsDecodeRound {
                    break
                }
            }
            guard row.needsDecodeRound,
                  let replacement,
                  !row.eosTokens.contains(replacement) else {
                row.stopped = row.stopped || replacement.map(row.eosTokens.contains) == true
                continue
            }
            let committedCandidateTokenCount = accepted + 1
            recoveries.append(LagunaDFlashRecovery(
                row: row,
                candidateHiddenStates: rowHiddenStates,
                committedCandidateTokenCount: committedCandidateTokenCount,
                replacement: replacement,
                recoveryCache: LagunaDFlashDecoder.commitCandidatePrefix(
                    base: row.targetCaches,
                    candidate: candidateCaches[rowIndex],
                    tokenCount: committedCandidateTokenCount
                )
            ))
        }

        guard !recoveries.isEmpty else { return }
        guard let batchedRecoveryCaches = makeBatchedCaches(
            recoveries.map(\.recoveryCache)
        ) else {
            throw LagunaError.generationFailed(
                "Laguna could not batch DFlash recovery caches."
            )
        }
        let recoveryInput = MLXArray(
            recoveries.map { Int32($0.replacement) }
        ).reshaped(recoveries.count, 1)
        let recovery = model.forward(
            recoveryInput,
            cache: batchedRecoveryCaches,
            captureLayerIndices: Set(dflash.config.dflash.targetLayerIDs)
        )
        MLX.eval([recovery.logits] + Array(recovery.capturedHiddenStates.values))
        guard let recoveryCaches = splitBatchedCaches(
            batchedRecoveryCaches,
            rowCount: recoveries.count
        ) else {
            throw LagunaError.generationFailed(
                "Laguna could not split DFlash recovery caches."
            )
        }
        dflashTargetRecoveryForwards += 1
        recordBatchedForward(
            positions: recoveries.map {
                $0.recoveryCache.map(\.offset).min() ?? 0
            }
        )

        for (recoveryIndex, pending) in recoveries.enumerated() {
            let row = pending.row
            emitDFlashToken(
                pending.replacement,
                row: row,
                tokenizerAndTemplate: tokenizerAndTemplate
            )
            row.targetCaches = recoveryCaches[recoveryIndex]
            row.logits = recovery.logits[
                recoveryIndex..<(recoveryIndex + 1),
                (recovery.logits.dim(1) - 1)...,
                0...
            ]
            let committedHiddenStates = Dictionary(
                uniqueKeysWithValues: dflash.config.dflash.targetLayerIDs.map { layerID in
                    let candidatePrefix = pending.candidateHiddenStates[layerID]![
                        0...,
                        ..<pending.committedCandidateTokenCount,
                        0...
                    ]
                    return (
                        layerID,
                        concatenated(
                            [
                                candidatePrefix,
                                recovery.capturedHiddenStates[layerID]![
                                    recoveryIndex..<(recoveryIndex + 1),
                                    0...,
                                    0...
                                ],
                            ],
                            axis: 1
                        )
                    )
                }
            )
            dflash.appendTargetContext(
                dflash.combineTargetHiddenStates(committedHiddenStates),
                cache: row.draftCaches
            )
            evaluateGemma4CacheStorage(row.draftCaches)
        }
    }

    func emitDFlashToken(
        _ token: Int,
        row: LagunaDFlashBatchedDecodeRow,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) {
        row.generatedTokens.append(token)
        row.repetitionHistory.append(token)
        if row.firstTokenSeconds == nil {
            row.firstTokenSeconds = Date().timeIntervalSince(row.decodeStart)
        }
        guard let progressHandler = row.progressHandler else { return }
        let piece = row.progressDecoder.append(
            decodedText: tokenizerAndTemplate.decode(tokens: row.generatedTokens)
        )
        if piece.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            row.pendingProgressWhitespace += piece
        } else if !piece.isEmpty {
            progressHandler(ChatProgress(
                stage: .generating,
                message: row.pendingProgressWhitespace + piece
            ))
            row.pendingProgressWhitespace = ""
        }
    }

    func recordBatchedForward(positions: [Int]) {
        if positions.count > 1 {
            batchedDecodeSteps += 1
            if RuntimeDecodeBatchPositionKind.variablePositionBatchCount(positions) > 0 {
                variablePositionBatchedSteps += 1
            } else {
                samePositionBatchedSteps += 1
            }
            totalBatchedRows += positions.count
            maxObservedBatchSize = max(maxObservedBatchSize, positions.count)
        } else {
            singleDecodeSteps += 1
        }
    }

    func finishCompletedDFlashDecodeRows() {
        var remaining: [LagunaDFlashBatchedDecodeRow] = []
        for row in activeDFlashDecodeRows {
            if row.needsDecodeRound {
                remaining.append(row)
            } else {
                row.finish()
            }
        }
        activeDFlashDecodeRows = remaining
    }

    func failDFlashRows(
        _ rows: [LagunaDFlashBatchedDecodeRow],
        with error: Error
    ) {
        let failedIDs = Set(rows.map(\.id))
        for row in rows {
            row.fail(error)
        }
        activeDFlashDecodeRows.removeAll { failedIDs.contains($0.id) }
    }

}
