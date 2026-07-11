import AudioCodecs
import Foundation
import MLX

public struct MuScriptorTranscriptionOptions: Hashable, Sendable {
    public var useSampling: Bool
    public var temperature: Float
    public var maxTokensPerChunk: Int
    public var strictEOS: Bool
    public var beamSize: Int
    public var chunkBatchSize: Int
    public var instruments: [String]?

    public init(
        useSampling: Bool = false,
        temperature: Float = 1,
        maxTokensPerChunk: Int = 2_000,
        strictEOS: Bool = false,
        beamSize: Int = 1,
        chunkBatchSize: Int = 4,
        instruments: [String]? = nil
    ) {
        self.useSampling = useSampling
        self.temperature = temperature
        self.maxTokensPerChunk = maxTokensPerChunk
        self.strictEOS = strictEOS
        self.beamSize = beamSize
        self.chunkBatchSize = chunkBatchSize
        self.instruments = instruments
    }

    public func validate() throws {
        guard maxTokensPerChunk > 0 else {
            throw MuScriptorError.malformedTokenStream("max token count must be positive")
        }
        guard !useSampling || temperature > 0 else {
            throw MuScriptorError.malformedTokenStream("sampling temperature must be positive")
        }
        guard beamSize > 0 else {
            throw MuScriptorError.malformedTokenStream("beam size must be positive")
        }
        guard chunkBatchSize > 0 else {
            throw MuScriptorError.malformedTokenStream("chunk batch size must be positive")
        }
        guard beamSize == 1 || !useSampling else {
            throw MuScriptorError.malformedTokenStream("beam search cannot be combined with sampling")
        }
    }
}

public final class MuScriptorTranscriber {
    public typealias ProgressHandler = @Sendable (_ completedChunks: Int, _ totalChunks: Int) -> Void

    private let model: MuScriptorModel
    private let melSpectrogram: MuScriptorMelSpectrogram
    private let memoryScale: Int

    public init(model: MuScriptorModel, memoryScale: Int = 1) {
        self.model = model
        self.melSpectrogram = MuScriptorMelSpectrogram()
        self.memoryScale = max(1, memoryScale)
    }

    public static func load(
        rootURL: URL,
        variant: MuScriptorVariant,
        dtype: DType = .bfloat16
    ) throws -> MuScriptorTranscriber {
        let resources = MuScriptorResources(rootURL: rootURL)
        let model = try MuScriptorModel.load(
            resources: resources,
            variant: variant,
            dtype: dtype
        )
        return MuScriptorTranscriber(
            model: model,
            memoryScale: memoryScale(for: dtype)
        )
    }

    static func memoryScale(for dtype: DType) -> Int {
        max(1, (dtype.size + 1) / 2)
    }

    public func transcribe(
        samples: [Float],
        options: MuScriptorTranscriptionOptions = .init(),
        progress: ProgressHandler? = nil
    ) throws -> MuScriptorTranscription {
        try options.validate()
        guard !samples.isEmpty else {
            return MuScriptorTranscription(notes: [], events: [], chunkCount: 0)
        }
        let chunkSize = MuScriptorMelSpectrogram.chunkSampleCount
        let chunkCount = (samples.count + chunkSize - 1) / chunkSize
        var chunkTokens: [[Int]] = []
        chunkTokens.reserveCapacity(chunkCount)
        progress?(0, chunkCount)
        let machine = MereRunMachineProfile.current
        let effectiveChunkBatchSize = MuScriptorBatchPolicy.effectiveChunkBatchSize(
            requested: options.chunkBatchSize,
            configuration: model.configuration,
            beamSize: options.beamSize,
            physicalMemoryBytes: machine.isAppleSiliconMac ? machine.physicalMemoryBytes : nil,
            activeMemoryBytes: Memory.activeMemory,
            cacheMemoryBytes: Memory.cacheMemory,
            memoryScale: memoryScale
        )

        if options.beamSize > 1 {
            for batchStart in stride(from: 0, to: chunkCount, by: effectiveChunkBatchSize) {
                let batchEnd = min(chunkCount, batchStart + effectiveChunkBatchSize)
                let indices = Array(batchStart..<batchEnd)
                let prefixes = try indices.map { chunkIndex in
                    try conditioningPrefix(
                        samples: samples,
                        chunkIndex: chunkIndex,
                        chunkSize: chunkSize,
                        instruments: options.instruments
                    )
                }
                let generated = try generateChunksWithBeamSearch(
                    indices: indices,
                    prefix: MLX.concatenated(prefixes, axis: 0),
                    options: options
                )
                for tokens in generated {
                    chunkTokens.append(tokens)
                    progress?(chunkTokens.count, chunkCount)
                }
            }
        } else {
            for batchStart in stride(from: 0, to: chunkCount, by: effectiveChunkBatchSize) {
                let batchEnd = min(chunkCount, batchStart + effectiveChunkBatchSize)
                let indices = Array(batchStart..<batchEnd)
                let prefixes = try indices.map { chunkIndex in
                    try conditioningPrefix(
                        samples: samples,
                        chunkIndex: chunkIndex,
                        chunkSize: chunkSize,
                        instruments: options.instruments
                    )
                }
                let batchPrefix = MLX.concatenated(prefixes, axis: 0)
                let generated = try generateChunks(
                    indices: indices,
                    prefix: batchPrefix,
                    options: options
                )
                for tokens in generated {
                    chunkTokens.append(tokens)
                    progress?(chunkTokens.count, chunkCount)
                }
            }
        }
        return try MuScriptorTokenDecoder().decode(chunks: chunkTokens)
    }

    private func conditioningPrefix(
        samples: [Float],
        chunkIndex: Int,
        chunkSize: Int,
        instruments: [String]?
    ) throws -> MLXArray {
        let start = chunkIndex * chunkSize
        let end = min(samples.count, start + chunkSize)
        let mel = melSpectrogram.extract(from: Array(samples[start..<end]))
        return try model.conditioningPrefix(mel: mel, instruments: instruments)
    }

    private func generateChunks(
        indices: [Int],
        prefix: MLXArray,
        options: MuScriptorTranscriptionOptions
    ) throws -> [[Int]] {
        let batchSize = indices.count
        var caches = model.makeCaches()
        let initialTokens = MLXArray(
            [Int32](repeating: Int32(model.configuration.card), count: batchSize)
        ).reshaped(batchSize, 1)
        var logits = model.batchedLogits(
            tokenIDs: initialTokens,
            prefix: prefix,
            caches: caches
        )
        MLX.eval(logits)

        var generated = [[Int]](repeating: [], count: batchSize)
        var activeRows = Array(0..<batchSize)
        var sampledEnded = [Bool](repeating: false, count: batchSize)

        for step in 0..<options.maxTokensPerChunk {
            let sampled: MLXArray
            if !MuScriptorBatchCompaction.isEnabled(useSampling: options.useSampling) {
                sampled = MLX.categorical(logits / options.temperature).asType(.int32)
            } else {
                sampled = MLX.argMax(logits, axis: -1).asType(.int32)
            }
            let nextTokens: MLXArray
            if !MuScriptorBatchCompaction.isEnabled(useSampling: options.useSampling) {
                // Preserve seeded sampling semantics: keeping ended rows in
                // the categorical batch means sibling EOS timing cannot
                // reassign the global RNG stream for surviving rows.
                let endedMask = MLXArray(sampledEnded.map { $0 ? Int32(1) : Int32(0) }) .> 0
                nextTokens = MLX.where(endedMask, MLXArray(Int32(1)), sampled)
            } else {
                nextTokens = sampled
            }

            let nextLogits: MLXArray?
            if step + 1 < options.maxTokensPerChunk {
                let value = model.batchedLogits(
                    tokenIDs: nextTokens.reshaped(activeRows.count, 1),
                    prefix: nil,
                    caches: caches
                )
                asyncEval([value, nextTokens])
                nextLogits = value
            } else {
                asyncEval(nextTokens)
                nextLogits = nil
            }

            let values = nextTokens.asArray(Int32.self)
            if !MuScriptorBatchCompaction.isEnabled(useSampling: options.useSampling) {
                for row in 0..<batchSize where !sampledEnded[row] {
                    if values[row] == 1 {
                        sampledEnded[row] = true
                    } else {
                        generated[row].append(Int(values[row]))
                    }
                }
                if sampledEnded.allSatisfy({ $0 }) {
                    return generated
                }
                if let nextLogits {
                    logits = nextLogits
                }
                continue
            }

            let survivorLocalRows = MuScriptorBatchCompaction.survivingRows(tokenValues: values)
            for (localRow, originalRow) in activeRows.enumerated() where values[localRow] != 1 {
                generated[originalRow].append(Int(values[localRow]))
            }
            if survivorLocalRows.isEmpty {
                return generated
            }
            let survivingOriginalRows = survivorLocalRows.map { activeRows[$0] }
            if let nextLogits {
                if survivorLocalRows.count == activeRows.count {
                    logits = nextLogits
                } else {
                    let rowIndices = MLXArray(survivorLocalRows.map(Int32.init))
                    logits = MLX.take(nextLogits, rowIndices, axis: 0)
                    guard let compacted = MuScriptorBatchCompaction.compact(
                        caches: caches,
                        rowCount: activeRows.count,
                        keeping: survivorLocalRows
                    ) else {
                        preconditionFailure("MuScriptor KV cache rows could not be compacted")
                    }
                    caches = compacted
                }
            }
            activeRows = survivingOriginalRows
        }

        let unfinishedRow = MuScriptorBatchCompaction.isEnabled(useSampling: options.useSampling)
            ? activeRows.first
            : sampledEnded.firstIndex(of: false)
        if options.strictEOS, let row = unfinishedRow {
            throw MuScriptorError.generationLimit(
                chunk: indices[row],
                limit: options.maxTokensPerChunk
            )
        }
        return generated
    }

    private struct Beam {
        var tokens: [Int]
        var currentToken: Int
        var score: Float
        var caches: [KVCacheSimple]
        var ended: Bool

        var rankedScore: Float {
            score / pow(Float(max(tokens.count, 1)), 0.75)
        }
    }

    func generateChunksWithBeamSearch(
        indices: [Int],
        prefix: MLXArray,
        options: MuScriptorTranscriptionOptions,
        maximumLiveLanes: Int? = nil
    ) throws -> [[Int]] {
        guard !indices.isEmpty else { return [] }
        let width = options.beamSize
        let initialCaches = model.makeCaches()
        let initialTokens = MLXArray(
            [Int32](repeating: Int32(model.configuration.card), count: indices.count)
        ).reshaped(indices.count, 1)
        let initialLogits = model.batchedLogits(
            tokenIDs: initialTokens,
            prefix: prefix,
            caches: initialCaches
        )
        MLX.eval(initialLogits)
        guard let initialCacheRows = MuScriptorBeamBatching.split(
            caches: initialCaches,
            rowCount: indices.count
        ) else {
            preconditionFailure("MuScriptor initial beam cache rows could not be split")
        }
        let initialCandidates = topLogProbabilitiesByRow(initialLogits, count: width)
        var chunkBeams = initialCandidates.enumerated().map { chunkRow, candidates in
            candidates.map { candidate in
                Beam(
                    tokens: candidate.token == 1 ? [] : [candidate.token],
                    currentToken: candidate.token,
                    score: candidate.logProbability,
                    caches: fork(initialCacheRows[chunkRow]),
                    ended: candidate.token == 1
                )
            }
        }

        if options.maxTokensPerChunk > 1 {
            for _ in 1..<options.maxTokensPerChunk {
                let active = chunkBeams.enumerated().flatMap { chunkRow, beams in
                    beams.filter { !$0.ended }.map { (chunkRow: chunkRow, beam: $0) }
                }
                if active.isEmpty { break }

                var candidatesByChunk = chunkBeams.map { $0.filter(\.ended) }
                let liveLaneLimit = max(
                    1,
                    maximumLiveLanes
                        ?? Int(MuScriptorBatchPolicy.maximumLiveLanes(configuration: model.configuration))
                )
                for range in MuScriptorBeamBatching.microbatchRanges(
                    rowCount: active.count,
                    maximumRows: liveLaneLimit
                ) {
                    let microbatch = active[range]
                    guard let batchedCaches = MuScriptorBeamBatching.batch(
                        microbatch.map { $0.beam.caches }
                    ) else {
                        preconditionFailure("MuScriptor active beam caches could not be batched")
                    }
                    let activeTokens = MLXArray(microbatch.map { Int32($0.beam.currentToken) })
                        .reshaped(microbatch.count, 1)
                    let logits = model.batchedLogits(
                        tokenIDs: activeTokens,
                        prefix: nil,
                        caches: batchedCaches
                    )
                    MLX.eval(logits)
                    guard let advancedCacheRows = MuScriptorBeamBatching.split(
                        caches: batchedCaches,
                        rowCount: microbatch.count
                    ) else {
                        preconditionFailure("MuScriptor advanced beam cache rows could not be split")
                    }
                    let nextCandidates = topLogProbabilitiesByRow(logits, count: width)
                    for (activeRow, entry) in microbatch.enumerated() {
                        for next in nextCandidates[activeRow] {
                            var tokens = entry.beam.tokens
                            if next.token != 1 { tokens.append(next.token) }
                            candidatesByChunk[entry.chunkRow].append(Beam(
                                tokens: tokens,
                                currentToken: next.token,
                                score: entry.beam.score + next.logProbability,
                                caches: fork(advancedCacheRows[activeRow]),
                                ended: next.token == 1
                            ))
                        }
                    }
                }
                chunkBeams = candidatesByChunk.map { candidates in
                    Array(candidates.sorted { $0.rankedScore > $1.rankedScore }.prefix(width))
                }
            }
        }

        return try chunkBeams.enumerated().map { chunkRow, beams in
            let ended = beams.filter(\.ended)
            guard let best = (ended.isEmpty ? beams : ended).max(by: { $0.rankedScore < $1.rankedScore }) else {
                return []
            }
            if !best.ended, options.strictEOS {
                throw MuScriptorError.generationLimit(
                    chunk: indices[chunkRow],
                    limit: options.maxTokensPerChunk
                )
            }
            return best.tokens
        }
    }

    private func fork(_ caches: [KVCacheSimple]) -> [KVCacheSimple] {
        caches.map { cache in
            guard let copy = cache.fork() as? KVCacheSimple else {
                preconditionFailure("MuScriptor requires KVCacheSimple beam copies")
            }
            return copy
        }
    }

    private func topLogProbabilitiesByRow(
        _ logits: MLXArray,
        count: Int
    ) -> [[(token: Int, logProbability: Float)]] {
        precondition(logits.ndim == 2)
        let rowCount = logits.dim(0)
        let vocabularySize = logits.dim(1)
        let values = logits.asArray(Float.self)
        return (0..<rowCount).map { row in
            let start = row * vocabularySize
            return topLogProbabilities(
                Array(values[start..<(start + vocabularySize)]),
                count: count
            )
        }
    }

    private func topLogProbabilities(
        _ values: [Float],
        count: Int
    ) -> [(token: Int, logProbability: Float)] {
        let maximum = values.max() ?? 0
        let logDenominator = maximum + log(values.reduce(Float(0)) { partial, value in
            partial + exp(value - maximum)
        })
        return values.indices
            .sorted { values[$0] > values[$1] }
            .prefix(count)
            .map { ($0, values[$0] - logDenominator) }
    }
}

enum MuScriptorBeamBatching {
    /// Structural scheduler evidence: legacy beam search issued one model
    /// forward per live beam; the batched path issues one per non-empty step.
    static func modelDispatchCounts(activeRowsPerStep: [Int]) -> (legacy: Int, batched: Int) {
        (
            legacy: activeRowsPerStep.reduce(0) { $0 + max(0, $1) },
            batched: activeRowsPerStep.filter { $0 > 0 }.count
        )
    }

    static func microbatchRanges(rowCount: Int, maximumRows: Int) -> [Range<Int>] {
        guard rowCount > 0 else { return [] }
        let maximumRows = max(1, maximumRows)
        return stride(from: 0, to: rowCount, by: maximumRows).map { start in
            start..<min(rowCount, start + maximumRows)
        }
    }

    static func batch(_ rowCaches: [[KVCacheSimple]]) -> [KVCacheSimple]? {
        guard let first = rowCaches.first,
              !first.isEmpty,
              rowCaches.allSatisfy({ $0.count == first.count }) else {
            return nil
        }
        var result: [KVCacheSimple] = []
        result.reserveCapacity(first.count)
        for layerIndex in first.indices {
            let layerRows = rowCaches.map { $0[layerIndex] }
            guard let batched = layerRows[0].batched(with: layerRows) as? KVCacheSimple else {
                return nil
            }
            result.append(batched)
        }
        return result
    }

    static func split(
        caches: [KVCacheSimple],
        rowCount: Int
    ) -> [[KVCacheSimple]]? {
        guard rowCount > 0 else { return nil }
        var rows = Array(repeating: [KVCacheSimple](), count: rowCount)
        for cache in caches {
            guard let split = cache.unbatchedRows(count: rowCount) as? [KVCacheSimple],
                  split.count == rowCount else {
                return nil
            }
            for row in rows.indices {
                rows[row].append(split[row])
            }
        }
        return rows
    }
}

enum MuScriptorBatchCompaction {
    static func isEnabled(useSampling: Bool) -> Bool {
        !useSampling
    }

    static func survivingRows(tokenValues: [Int32], eosToken: Int32 = 1) -> [Int] {
        tokenValues.indices.filter { tokenValues[$0] != eosToken }
    }

    static func compact(
        caches: [KVCacheSimple],
        rowCount: Int,
        keeping rowsToKeep: [Int]
    ) -> [KVCacheSimple]? {
        guard !rowsToKeep.isEmpty else { return [] }
        var compacted: [KVCacheSimple] = []
        compacted.reserveCapacity(caches.count)
        for cache in caches {
            guard let rows = cache.unbatchedRows(count: rowCount) as? [KVCacheSimple] else {
                return nil
            }
            let selected = rowsToKeep.map { rows[$0] }
            guard let first = selected.first,
                  let batched = first.batched(with: selected) as? KVCacheSimple else {
                return nil
            }
            compacted.append(batched)
        }
        return compacted
    }
}
