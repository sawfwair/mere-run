import Foundation
import MediaIO
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

extension Q35Generator {
    func enqueueDecodeRow(
        _ row: Q35BatchedDecodeRow,
        model: Q35Model,
        tokenizerAndTemplate: Q35TokenizerAndTemplate
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

    private func startDecodeLoopIfNeeded(
        model: Q35Model,
        tokenizerAndTemplate: Q35TokenizerAndTemplate
    ) {
        guard !decodeLoopRunning else { return }
        decodeLoopRunning = true
        Task {
            await runDecodeLoop(model: model, tokenizerAndTemplate: tokenizerAndTemplate)
        }
    }

    private func runDecodeLoop(
        model: Q35Model,
        tokenizerAndTemplate: Q35TokenizerAndTemplate
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
            if model.config.textConfig.isQwen4Exp {
                clearMLXCacheUnderPressureIfNeeded()
            }
            await Task.yield()
        }
    }

    private func activateQueuedDecodeRows() {
        guard !decodeQueue.isEmpty else { return }
        activeDecodeRows.append(contentsOf: decodeQueue)
        decodeQueue.removeAll(keepingCapacity: true)
    }

    private func selectDecodeRows() -> [Q35BatchedDecodeRow] {
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

    private func decodeBatchSignature(for row: Q35BatchedDecodeRow) -> String {
        row.layerCaches
            .map { cache in
                switch cache {
                case .full(let kv):
                    if kv.supportsVariablePositionBatching {
                        return "full:variable"
                    }
                    return "full:\(kv.offset)"
                case .linear(let linear):
                    return "linear:\(Q35Generator.linearCacheSignature(linear))"
                case nil:
                    return "nil"
                }
            }
            .joined(separator: "|")
    }

    private static func linearCacheSignature(_ cache: Q35LinearCache) -> String {
        let convShape = cache.convState?.shape.map(String.init).joined(separator: "x") ?? "nil"
        let recurrentShape = cache.recurrentState?.shape.map(String.init).joined(separator: "x") ?? "nil"
        return "\(convShape):\(recurrentShape)"
    }

    private func decodePosition(_ row: Q35BatchedDecodeRow) -> Int {
        row.layerCaches.compactMap { cache in
            if case .full(let kv)? = cache {
                return kv.offset
            }
            return nil
        }.min() ?? (row.prefillTokenCount + row.generatedTokens.count)
    }

    func decodePositionIds(
        layerCaches: [Q35LayerCache?],
        tokenCount: Int,
        ropeDelta: Int?
    ) -> MLXArray? {
        guard let ropeDelta, tokenCount > 0 else { return nil }
        let offset = layerCaches.compactMap { cache in
            if case .full(let kv)? = cache {
                return kv.offset
            }
            return nil
        }.min() ?? 0
        let positions = (0..<tokenCount).map { Int32(offset + ropeDelta + $0) }
        let values = positions + positions + positions
        return MLXArray(values, [3, 1, tokenCount])
    }

    private func batchedDecodePositionIds(rows: [Q35BatchedDecodeRow]) -> MLXArray? {
        guard rows.contains(where: { $0.mropeRopeDelta != nil }) else { return nil }
        var values: [Int32] = []
        values.reserveCapacity(rows.count * 3)
        for _ in 0..<3 {
            for row in rows {
                values.append(Int32(decodePosition(row) + (row.mropeRopeDelta ?? 0)))
            }
        }
        return MLXArray(values, [3, rows.count, 1])
    }

    private func decodeOneStep(
        rows: [Q35BatchedDecodeRow],
        model: Q35Model,
        tokenizerAndTemplate: Q35TokenizerAndTemplate
    ) throws {
        let sampledRows = rows.filter(\.needsDecodeStep)
        guard !sampledRows.isEmpty else { return }

        if Self.batchedGPUSamplingEnabled {
            // Sample every row on GPU (same sampler the serial pipelined
            // decode uses), then read the whole batch back in one sync —
            // the legacy path performed one blocking readback per row per
            // step, which scales linearly with serve concurrency.
            var tokenArrays: [MLXArray] = []
            tokenArrays.reserveCapacity(sampledRows.count)
            for row in sampledRows {
                if !row.repetitionHistoryGPUSeeded {
                    row.repetitionHistoryGPU = repetitionHistoryArray(
                        promptTokens: row.repetitionHistory,
                        config: row.generationConfig
                    )
                    row.repetitionHistoryGPUSeeded = true
                }
                let tokenArray = sampledTokenArray(
                    logits: row.logits[0, -1, 0...],
                    config: row.generationConfig,
                    previousTokenIndices: row.repetitionHistoryGPU,
                    banMask: nil
                )
                row.repetitionHistoryGPU = appendingRepetitionHistory(
                    row.repetitionHistoryGPU,
                    token: tokenArray,
                    config: row.generationConfig
                )
                tokenArrays.append(tokenArray.reshaped(1))
            }
            let values = MLX.concatenated(tokenArrays, axis: 0).asArray(Int32.self)
            for (index, row) in sampledRows.enumerated() {
                let next = Int(values[index])
                guard !row.eosSet.contains(next) else {
                    row.stopped = true
                    continue
                }
                row.generatedTokens.append(next)
                row.repetitionHistory.append(next)
                if row.firstTokenSeconds == nil {
                    row.firstTokenSeconds = Date().timeIntervalSince(row.decodeStart)
                }
                if let progressHandler = row.progressHandler {
                    let piece = tokenizerAndTemplate.decode(token: next)
                    if !piece.isEmpty {
                        progressHandler(ChatProgress(stage: .generating, message: piece))
                    }
                }
            }
        } else {
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
                row.repetitionHistory.append(next)
                if row.firstTokenSeconds == nil {
                    row.firstTokenSeconds = Date().timeIntervalSince(row.decodeStart)
                }
                if let progressHandler = row.progressHandler {
                    let piece = tokenizerAndTemplate.decode(token: next)
                    if !piece.isEmpty {
                        progressHandler(ChatProgress(stage: .generating, message: piece))
                    }
                }
            }
        }

        let continuingRows = sampledRows.filter(\.needsDecodeStep)
        guard !continuingRows.isEmpty else { return }

        if continuingRows.count > 1,
           let batchedCaches = makeBatchedLayerCaches(continuingRows.map(\.layerCaches)) {
            let nextInput = MLXArray(continuingRows.compactMap { $0.generatedTokens.last }.map(Int32.init))
                .reshaped(continuingRows.count, 1)
            let batchedLogits = model(
                nextInput,
                cache: batchedCaches,
                positionIds: batchedDecodePositionIds(rows: continuingRows)
            )
            MLX.eval(batchedLogits)
            guard let splitCaches = splitBatchedLayerCaches(batchedCaches, rowCount: continuingRows.count) else {
                throw Q35Error.generationFailed("Qwen-family batched decode could not split merged cache rows.")
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
            row.logits = model(
                nextInput,
                cache: row.layerCaches,
                positionIds: decodePositionIds(
                    layerCaches: row.layerCaches,
                    tokenCount: 1,
                    ropeDelta: row.mropeRopeDelta
                )
            )
            MLX.eval(row.logits)
            singleDecodeSteps += 1
        }
    }

    private func makeBatchedLayerCaches(_ rowCaches: [[Q35LayerCache?]]) -> [Q35LayerCache?]? {
        guard let first = rowCaches.first, !first.isEmpty else { return nil }
        guard rowCaches.allSatisfy({ $0.count == first.count }) else { return nil }

        var result: [Q35LayerCache?] = []
        result.reserveCapacity(first.count)
        for layerIndex in first.indices {
            let layerCaches = rowCaches.map { $0[layerIndex] }
            if layerCaches.allSatisfy({ $0 == nil }) {
                result.append(nil)
                continue
            }
            let nonNil = layerCaches.compactMap { $0 }
            guard nonNil.count == layerCaches.count,
                  let batched = nonNil[0].batched(with: nonNil) else {
                return nil
            }
            result.append(batched)
        }
        return result
    }

    private func splitBatchedLayerCaches(
        _ caches: [Q35LayerCache?],
        rowCount: Int
    ) -> [[Q35LayerCache?]]? {
        guard rowCount > 0 else { return nil }
        var rows = Array(repeating: [Q35LayerCache?](), count: rowCount)
        for cache in caches {
            guard let cache else {
                for index in 0..<rowCount {
                    rows[index].append(nil)
                }
                continue
            }
            guard let split = cache.unbatchedRows(count: rowCount), split.count == rowCount else {
                return nil
            }
            for index in 0..<rowCount {
                rows[index].append(split[index])
            }
        }
        return rows
    }

    private func finishCompletedDecodeRows() {
        var remaining: [Q35BatchedDecodeRow] = []
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

    private func failRows(_ rows: [Q35BatchedDecodeRow], with error: Error) {
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
