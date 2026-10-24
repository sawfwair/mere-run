// Packed inference adapted from FluidInference/FluidUse@d278e17df85e39e8c21bb64a467da81c39576fcd.
// Licensed under Apache-2.0; see LICENSE and NOTICE in this directory.
#if canImport(CoreML) && os(macOS) && arch(arm64)
@preconcurrency import CoreML
import Foundation

extension EmbeddingGemma2 {
    /// Packs text records and overlaps up to eight short ANE predictions. Larger GPU
    /// sequences run individually to bound attention memory. Results retain input order.
    public func embed(sequences: [MultimodalSequence]) async throws -> [[Float]] {
        let groups = CoreMLTextBatching.groups(for: sequences)
        var result = [[Float]](repeating: [], count: sequences.count)
        var cursor = 0
        while cursor < groups.count {
            try Task.checkCancellation()
            let first = groups[cursor]
            if first.count == 1, sequences[first[0]].ids.count > 512 {
                result[first[0]] = try await embed(sequence: sequences[first[0]])
                cursor += 1
                continue
            }
            var jobs: [(indices: [Int], prediction: EmbeddingPrediction)] = []
            while cursor < groups.count, jobs.count < 8 {
                let indices = groups[cursor]
                if indices.count == 1, sequences[indices[0]].ids.count > 512 { break }
                let selected = indices.map { sequences[$0] }
                let prediction: EmbeddingPrediction
                if selected.count > 1 { prediction = try await packedPrediction(selected) }
                else { prediction = try await singlePrediction(selected[0]) }
                jobs.append((indices, prediction))
                cursor += 1
            }
            try await withThrowingTaskGroup(of: (indices: [Int], vectors: [[Float]]).self) { group in
                for job in jobs {
                    group.addTask {
                        try Task.checkCancellation()
                        return (job.indices, try await job.prediction.run())
                    }
                }
                for try await output in group {
                    for (index, vector) in zip(output.indices, output.vectors) { result[index] = vector }
                }
            }
        }
        return result
    }

    // Implements the published pack_256 contract: block-diagonal attention, reset
    // rotary positions, and independent mean pooling, including BOS/EOS per record.
    private func packedPrediction(_ sequences: [MultimodalSequence]) async throws -> EmbeddingPrediction {
        let length = CoreMLTextBatching.tokenCapacity
        var ids: [Int32] = []
        var bias = [Float16](repeating: -10_000, count: length * length)
        var positions = [Float16](repeating: 0, count: length)
        var pool = [Float16](repeating: 0, count: 8 * length)
        for (slot, sequence) in sequences.enumerated() {
            let offset = ids.count, count = sequence.ids.count
            ids += sequence.ids
            for row in offset..<(offset + count) {
                for column in offset..<(offset + count) { bias[row * length + column] = 0 }
                positions[row] = Float16(row - offset)
                pool[slot * length + row] = Float16(1 / Float(count))
            }
        }
        for row in ids.count..<length { bias[row * length + row] = 0 }
        ids += repeatElement(config.pad_token_id, count: length - ids.count)
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "inputs_embeds": MLFeatureValue(multiArray: array(table.embeddings(ids: ids), shape: [1, length, 512])),
            "attention_bias": MLFeatureValue(multiArray: array(bias, shape: [1, 1, length, length])),
            "positions": MLFeatureValue(multiArray: array(positions, shape: [length, 1])),
            "pool": MLFeatureValue(multiArray: array(pool, shape: [8, length]))
        ])
        let loaded = try await model(package: "EmbeddingGemma2Text.mlpackage", function: "pack_256", compute: .cpuAndNeuralEngine)
        return EmbeddingPrediction(model: loaded, features: features, rows: sequences.count, outputRows: 8)
    }
}

/// Immutable prediction inputs. Core ML supports concurrent async predictions on a
/// loaded MLModel; each job owns its feature arrays and no caller mutates them.
struct EmbeddingPrediction: @unchecked Sendable {
    let model: MLModel
    let features: MLDictionaryFeatureProvider
    let rows: Int
    let outputRows: Int

    func run() async throws -> [[Float]] {
        let result = try await model.prediction(from: features)
        guard let output = result.featureValue(for: "embedding")?.multiArrayValue,
              output.shape.map(\.intValue) == [outputRows, 768], output.dataType == .float32 else {
            throw EmbedderError(description: "Core ML returned an invalid embedding output.")
        }
        let stride = output.strides[0].intValue
        let pointer = output.dataPointer.assumingMemoryBound(to: Float.self)
        return (0..<rows).map { row in (0..<768).map { pointer[row * stride + $0] } }
    }
}
#endif
