import Foundation
import MLX

extension LagunaGenerator {
    func enqueueDecodeRow(
        _ row: LagunaBatchedDecodeRow,
        model: LagunaCausalLM,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) {
        decodeQueue.append(row)
        startDecodeLoopIfNeeded(model: model, tokenizerAndTemplate: tokenizerAndTemplate)
    }

    func cancelDecodeRow(id: UUID) {
        if let index = decodeQueue.firstIndex(where: { $0.id == id }) {
            decodeQueue.remove(at: index).fail(CancellationError())
            return
        }
        if let index = activeDecodeRows.firstIndex(where: { $0.id == id }) {
            activeDecodeRows.remove(at: index).fail(CancellationError())
        }
    }

    func startDecodeLoopIfNeeded(
        model: LagunaCausalLM,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) {
        guard !decodeLoopRunning else { return }
        decodeLoopRunning = true
        Task {
            await runDecodeLoop(model: model, tokenizerAndTemplate: tokenizerAndTemplate)
        }
    }

    func runDecodeLoop(
        model: LagunaCausalLM,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) async {
        defer {
            decodeLoopRunning = false
            if !decodeQueue.isEmpty || !activeDecodeRows.isEmpty {
                startDecodeLoopIfNeeded(
                    model: model,
                    tokenizerAndTemplate: tokenizerAndTemplate
                )
            }
        }

        while !decodeQueue.isEmpty || !activeDecodeRows.isEmpty {
            if activeDecodeRows.isEmpty {
                try? await Task.sleep(nanoseconds: 1_000_000)
            }
            activateQueuedDecodeRows()
            guard !activeDecodeRows.isEmpty else { continue }

            let rows = selectDecodeRows()
            do {
                try decodeOneStep(
                    rows: rows,
                    model: model,
                    tokenizerAndTemplate: tokenizerAndTemplate
                )
            } catch {
                failRows(rows, with: error)
            }
            finishCompletedDecodeRows()
            await Task.yield()
        }
    }

    func activateQueuedDecodeRows() {
        activeDecodeRows.append(contentsOf: decodeQueue)
        decodeQueue.removeAll(keepingCapacity: true)
    }

    func selectDecodeRows() -> [LagunaBatchedDecodeRow] {
        let eligible = activeDecodeRows.filter(\.needsDecodeStep)
        let selectedIDs = Set(RuntimeDecodeBatchPlanner.selectRows(
            eligible.map { row in
                RuntimeDecodeBatchRowMetadata(
                    row: row.id,
                    signature: row.caches
                        .map { String(describing: type(of: $0)) }
                        .joined(separator: "|"),
                    position: row.caches.map(\.offset).min() ?? 0
                )
            }
        ))
        return eligible.filter { selectedIDs.contains($0.id) }
    }

    func decodeOneStep(
        rows: [LagunaBatchedDecodeRow],
        model: LagunaCausalLM,
        tokenizerAndTemplate: LagunaTokenizerAndTemplate
    ) throws {
        let sampledRows = rows.filter(\.needsDecodeStep)
        guard !sampledRows.isEmpty else { return }

        for row in sampledRows {
            let token = sampleToken(
                logits: row.logits[0, -1, 0...],
                config: row.generationConfig,
                previousTokens: row.repetitionHistory
            )
            guard !row.eosTokens.contains(token) else {
                row.stopped = true
                continue
            }

            row.generatedTokens.append(token)
            row.repetitionHistory.append(token)
            if row.firstTokenSeconds == nil {
                row.firstTokenSeconds = Date().timeIntervalSince(row.decodeStart)
            }
            if let progressHandler = row.progressHandler {
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
        }

        let continuingRows = sampledRows.filter(\.needsDecodeStep)
        guard !continuingRows.isEmpty else { return }

        if continuingRows.count > 1,
           let batchedCaches = makeBatchedCaches(continuingRows.map(\.caches)) {
            let positions = continuingRows.map { $0.caches.map(\.offset).min() ?? 0 }
            let input = MLXArray(
                continuingRows.compactMap(\.generatedTokens.last).map(Int32.init)
            ).reshaped(continuingRows.count, 1)
            let logits = model.lastPositionLogits(input, cache: batchedCaches)
            MLX.eval(logits)
            guard let splitCaches = splitBatchedCaches(
                batchedCaches,
                rowCount: continuingRows.count
            ) else {
                throw LagunaError.generationFailed(
                    "Laguna could not split ragged decode cache rows."
                )
            }
            for (index, row) in continuingRows.enumerated() {
                row.caches = splitCaches[index]
                row.logits = logits[index..<(index + 1), 0..., 0...]
            }
            batchedDecodeSteps += 1
            if RuntimeDecodeBatchPositionKind.variablePositionBatchCount(positions) > 0 {
                variablePositionBatchedSteps += 1
            } else {
                samePositionBatchedSteps += 1
            }
            totalBatchedRows += continuingRows.count
            maxObservedBatchSize = max(maxObservedBatchSize, continuingRows.count)
            return
        }

        for row in continuingRows {
            guard let token = row.generatedTokens.last else { continue }
            row.logits = model.lastPositionLogits(
                MLXArray([Int32(token)]).reshaped(1, 1),
                cache: row.caches
            )
            MLX.eval(row.logits)
            singleDecodeSteps += 1
        }
    }

    func makeBatchedCaches(
        _ rowCaches: [[Gemma4AttentionCache]]
    ) -> [Gemma4AttentionCache]? {
        guard let first = rowCaches.first, !first.isEmpty else { return nil }
        guard rowCaches.allSatisfy({ $0.count == first.count }) else { return nil }

        var result: [Gemma4AttentionCache] = []
        result.reserveCapacity(first.count)
        for layerIndex in first.indices {
            guard let cache = LagunaRaggedKVCache(
                rows: rowCaches.map { $0[layerIndex] }
            ) else {
                return nil
            }
            result.append(cache)
        }
        return result
    }

    func splitBatchedCaches(
        _ caches: [Gemma4AttentionCache],
        rowCount: Int
    ) -> [[Gemma4AttentionCache]]? {
        var rows = Array(repeating: [Gemma4AttentionCache](), count: rowCount)
        for cache in caches {
            guard let split = cache.unbatchedRows(count: rowCount) else {
                return nil
            }
            for index in 0..<rowCount {
                rows[index].append(split[index])
            }
        }
        return rows
    }

    func finishCompletedDecodeRows() {
        var remaining: [LagunaBatchedDecodeRow] = []
        for row in activeDecodeRows {
            if row.needsDecodeStep {
                remaining.append(row)
            } else {
                row.finish()
            }
        }
        activeDecodeRows = remaining
    }

    func failRows(_ rows: [LagunaBatchedDecodeRow], with error: Error) {
        let failedIDs = Set(rows.map(\.id))
        for row in rows {
            row.fail(error)
        }
        activeDecodeRows.removeAll { failedIDs.contains($0.id) }
    }

}
