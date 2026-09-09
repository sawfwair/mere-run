import Foundation
import MLX
import MLXNN

extension Gemma4Generator {
    func enqueueDecodeRow(
        _ row: Gemma4BatchedDecodeRow,
        model: any Gemma4CausalModel,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate
    ) {
        decodeQueue.append(row)
        startDecodeLoopIfNeeded(model: model, tokenizerAndTemplate: tokenizerAndTemplate)
    }

    func cancelDecodeRow(id: UUID) {
        if let index = decodeQueue.firstIndex(where: { $0.id == id }) {
            let row = decodeQueue.remove(at: index)
            row.fail(CancellationError())
            return
        }
        if let index = activeDecodeRows.firstIndex(where: { $0.id == id }) {
            let row = activeDecodeRows.remove(at: index)
            row.fail(CancellationError())
        }
    }

    func startDecodeLoopIfNeeded(
        model: any Gemma4CausalModel,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate
    ) {
        guard !decodeLoopRunning else { return }
        decodeLoopRunning = true
        Task {
            await runDecodeLoop(model: model, tokenizerAndTemplate: tokenizerAndTemplate)
        }
    }

    func runDecodeLoop(
        model: any Gemma4CausalModel,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate
    ) async {
        defer {
            decodeLoopRunning = false
            if !decodeQueue.isEmpty || !activeDecodeRows.isEmpty {
                startDecodeLoopIfNeeded(model: model, tokenizerAndTemplate: tokenizerAndTemplate)
            }
        }

        while !decodeQueue.isEmpty || !activeDecodeRows.isEmpty {
            if activeDecodeRows.isEmpty {
                try? await Task.sleep(nanoseconds: 1_000_000)
            }
            activateQueuedDecodeRows()
            guard !activeDecodeRows.isEmpty else {
                continue
            }

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
        guard !decodeQueue.isEmpty else { return }
        activeDecodeRows.append(contentsOf: decodeQueue)
        decodeQueue.removeAll(keepingCapacity: true)
    }

    func selectDecodeRows() -> [Gemma4BatchedDecodeRow] {
        let eligible = activeDecodeRows.filter(\.needsDecodeStep)
        let selectedIDs = Set(RuntimeDecodeBatchPlanner.selectRows(
            eligible.map { row in
                RuntimeDecodeBatchRowMetadata(
                    row: row.id,
                    signature: decodeBatchSignature(for: row),
                    position: decodePosition(row)
                )
            }
        ))
        return eligible.filter { selectedIDs.contains($0.id) }
    }

    func decodeBatchSignature(for row: Gemma4BatchedDecodeRow) -> String {
        row.layerCaches
            .map { "\(String(describing: type(of: $0))):\($0.offset)" }
            .joined(separator: "|")
    }

    func decodePosition(_ row: Gemma4BatchedDecodeRow) -> Int {
        row.layerCaches.map(\.offset).min() ?? 0
    }

    func decodeOneStep(
        rows: [Gemma4BatchedDecodeRow],
        model: any Gemma4CausalModel,
        tokenizerAndTemplate: Gemma4TokenizerAndTemplate
    ) throws {
        let sampledRows = rows.filter(\.needsDecodeStep)
        guard !sampledRows.isEmpty else { return }

        for row in sampledRows {
            let next = sampleToken(
                logits: row.logits[0, -1, 0...],
                config: row.generationConfig,
                previousTokens: row.repetitionHistory
            )
            guard !row.eosSet.contains(next) else {
                row.stopped = true
                continue
            }
            row.generatedTokens.append(next)
            if row.firstTokenSeconds == nil {
                row.firstTokenSeconds = Date().timeIntervalSince(row.decodeStart)
            }
            row.repetitionHistory.append(next)
            if let progressHandler = row.progressHandler {
                let piece = tokenizerAndTemplate.decode(token: next)
                if !piece.isEmpty {
                    progressHandler(ChatProgress(stage: .generating, message: piece))
                }
            }
        }

        let continuingRows = sampledRows.filter(\.needsDecodeStep)
        guard !continuingRows.isEmpty else { return }

        if continuingRows.count > 1,
           let batchedCaches = makeBatchedLayerCaches(continuingRows.map(\.layerCaches)) {
            let nextInput = MLXArray(continuingRows.compactMap { $0.generatedTokens.last }.map(Int32.init))
                .reshaped(continuingRows.count, 1)
            let batchedLogits = model.forward(inputIds: nextInput, cache: batchedCaches)
            MLX.eval(batchedLogits)
            guard let splitCaches = splitBatchedLayerCaches(batchedCaches, rowCount: continuingRows.count) else {
                throw Gemma4Error.unsupportedConfiguration("Gemma4 batched decode could not split merged KV cache rows.")
            }
            for (index, row) in continuingRows.enumerated() {
                row.layerCaches = splitCaches[index]
                row.logits = batchedLogits[index..<(index + 1), 0..., 0...]
            }
            batchedDecodeSteps += 1
            if RuntimeDecodeBatchPositionKind.variablePositionBatchCount(continuingRows.map(decodePosition)) > 0 {
                variablePositionBatchedSteps += 1
            } else {
                samePositionBatchedSteps += 1
            }
            totalBatchedRows += continuingRows.count
            maxObservedBatchSize = max(maxObservedBatchSize, continuingRows.count)
            return
        }

        for row in continuingRows {
            guard let next = row.generatedTokens.last else { continue }
            let nextInput = MLXArray([Int32(next)]).reshaped(1, 1)
            row.logits = model.forward(inputIds: nextInput, cache: row.layerCaches)
            MLX.eval(row.logits)
            singleDecodeSteps += 1
        }
    }

    func makeBatchedLayerCaches(_ rowCaches: [[Gemma4AttentionCache]]) -> [Gemma4AttentionCache]? {
        guard let first = rowCaches.first, !first.isEmpty else { return nil }
        guard rowCaches.allSatisfy({ $0.count == first.count }) else { return nil }

        var result: [Gemma4AttentionCache] = []
        result.reserveCapacity(first.count)
        for layerIndex in first.indices {
            let layerCaches = rowCaches.map { $0[layerIndex] }
            guard let batched = layerCaches[0].batched(with: layerCaches) else {
                return nil
            }
            result.append(batched)
        }
        return result
    }

    func splitBatchedLayerCaches(
        _ caches: [Gemma4AttentionCache],
        rowCount: Int
    ) -> [[Gemma4AttentionCache]]? {
        guard rowCount > 0 else { return nil }
        var rows = Array(repeating: [Gemma4AttentionCache](), count: rowCount)
        for cache in caches {
            guard let split = cache.unbatchedRows(count: rowCount), split.count == rowCount else {
                return nil
            }
            for index in 0..<rowCount {
                rows[index].append(split[index])
            }
        }
        return rows
    }

    func finishCompletedDecodeRows() {
        var remaining: [Gemma4BatchedDecodeRow] = []
        remaining.reserveCapacity(activeDecodeRows.count)
        for row in activeDecodeRows {
            if row.needsDecodeStep {
                remaining.append(row)
            } else {
                row.finish()
            }
        }
        activeDecodeRows = remaining
    }

    func failRows(_ rows: [Gemma4BatchedDecodeRow], with error: Error) {
        let ids = Set(rows.map(\.id))
        activeDecodeRows.removeAll { row in
            guard ids.contains(row.id) else { return false }
            row.fail(error)
            return true
        }
    }

    func failQueuedDecodeRows(_ error: Error) {
        for row in decodeQueue {
            row.fail(error)
        }
        for row in activeDecodeRows {
            row.fail(error)
        }
        decodeQueue.removeAll()
        activeDecodeRows.removeAll()
    }
}
