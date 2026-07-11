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

    public init(model: MuScriptorModel) {
        self.model = model
        self.melSpectrogram = MuScriptorMelSpectrogram()
    }

    public static func load(
        rootURL: URL,
        variant: MuScriptorVariant,
        dtype: DType = .bfloat16
    ) throws -> MuScriptorTranscriber {
        let resources = MuScriptorResources(rootURL: rootURL)
        return try MuScriptorTranscriber(model: .load(
            resources: resources,
            variant: variant,
            dtype: dtype
        ))
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

        if options.beamSize > 1 {
            for chunkIndex in 0..<chunkCount {
                let prefix = try conditioningPrefix(
                    samples: samples,
                    chunkIndex: chunkIndex,
                    chunkSize: chunkSize,
                    instruments: options.instruments
                )
                let tokens = try generateChunkWithBeamSearch(
                    index: chunkIndex,
                    prefix: prefix,
                    options: options
                )
                chunkTokens.append(tokens)
                progress?(chunkIndex + 1, chunkCount)
            }
        } else {
            for batchStart in stride(from: 0, to: chunkCount, by: options.chunkBatchSize) {
                let batchEnd = min(chunkCount, batchStart + options.chunkBatchSize)
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

    private func generateChunkWithBeamSearch(
        index: Int,
        prefix: MLXArray,
        options: MuScriptorTranscriptionOptions
    ) throws -> [Int] {
        let width = options.beamSize
        let initialCaches = model.makeCaches()
        let initialLogits = model.logits(
            tokenID: model.configuration.card,
            prefix: prefix,
            caches: initialCaches
        )
        var beams = topLogProbabilities(initialLogits, count: width).map { candidate in
            Beam(
                tokens: candidate.token == 1 ? [] : [candidate.token],
                currentToken: candidate.token,
                score: candidate.logProbability,
                caches: fork(initialCaches),
                ended: candidate.token == 1
            )
        }

        if options.maxTokensPerChunk > 1 {
            for _ in 1..<options.maxTokensPerChunk {
                if beams.allSatisfy(\.ended) { break }
                var candidates: [Beam] = beams.filter(\.ended)
                for beam in beams where !beam.ended {
                    let logits = model.logits(
                        tokenID: beam.currentToken,
                        prefix: nil,
                        caches: beam.caches
                    )
                    for next in topLogProbabilities(logits, count: width) {
                        var tokens = beam.tokens
                        if next.token != 1 { tokens.append(next.token) }
                        candidates.append(Beam(
                            tokens: tokens,
                            currentToken: next.token,
                            score: beam.score + next.logProbability,
                            caches: fork(beam.caches),
                            ended: next.token == 1
                        ))
                    }
                }
                beams = Array(candidates.sorted { $0.rankedScore > $1.rankedScore }.prefix(width))
            }
        }

        let ended = beams.filter(\.ended)
        if let best = (ended.isEmpty ? beams : ended).max(by: { $0.rankedScore < $1.rankedScore }) {
            if !best.ended, options.strictEOS {
                throw MuScriptorError.generationLimit(chunk: index, limit: options.maxTokensPerChunk)
            }
            return best.tokens
        }
        return []
    }

    private func fork(_ caches: [KVCacheSimple]) -> [KVCacheSimple] {
        caches.map { cache in
            guard let copy = cache.fork() as? KVCacheSimple else {
                preconditionFailure("MuScriptor requires KVCacheSimple beam copies")
            }
            return copy
        }
    }

    private func topLogProbabilities(
        _ logits: MLXArray,
        count: Int
    ) -> [(token: Int, logProbability: Float)] {
        let values = logits.asArray(Float.self)
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
