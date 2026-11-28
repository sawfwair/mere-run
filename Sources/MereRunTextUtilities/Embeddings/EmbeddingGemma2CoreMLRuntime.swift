import Foundation
import MereRunCore

public enum EmbeddingGemma2CoreMLRuntime {
    public static var isAvailable: Bool {
        #if canImport(CoreML) && os(macOS) && arch(arm64)
        if #available(macOS 15.0, *) { return true }
        #endif
        return false
    }

    public static func requireAvailable() throws {
        guard isAvailable else {
            throw EmbeddingGemma2Error.invalidConfiguration(
                "EmbeddingGemma 2 Core ML requires Apple Silicon macOS 15 or later and Core ML support in this build."
            )
        }
    }

    public static func validateInputs(_ inputs: [EmbeddingGemma2Input], task: EmbeddingGemma2Task,
                                      title: String?, dimensions: Int, maxTokens: Int?) throws {
        guard !inputs.isEmpty, inputs.allSatisfy({ !$0.content.isEmpty }),
              [128, 256, 512, 768].contains(dimensions), title == nil || task == .document,
              maxTokens == nil || maxTokens! >= 2 else {
            throw EmbeddingGemma2Error.invalidInput("Provide nonempty inputs, 128/256/512/768 dimensions, and at least two tokens. A title requires --task document.")
        }
        for part in inputs.flatMap(\.content) {
            switch part {
            case .video, .videoFrames:
                throw EmbeddingGemma2Error.invalidInput("The pinned Core ML export supports text, images, and audio. Use text-embed-embeddinggemma2 for video.")
            case .image(let url), .audio(let url):
                guard url.isFileURL else { throw EmbeddingGemma2Error.invalidInput("Media inputs require local file URLs.") }
            case .text: break
            }
        }
    }

    public static func embed(root: URL, inputs: [EmbeddingGemma2Input], task: EmbeddingGemma2Task = .raw,
                             title: String? = nil, dimensions: Int = 768, maxTokens: Int? = nil) async throws
        -> (embeddings: [[Float]], tokenCounts: [Int]) {
        try validateInputs(inputs, task: task, title: title, dimensions: dimensions, maxTokens: maxTokens)
        try requireAvailable()
        #if canImport(CoreML) && os(macOS) && arch(arm64)
        if #available(macOS 15.0, *) {
            return try await execute(root: EmbeddingGemma2CoreMLCatalog.bundleRoot(root), inputs: inputs,
                                     task: task, title: title, dimensions: dimensions, maxTokens: maxTokens)
        }
        #endif
        throw EmbeddingGemma2Error.invalidConfiguration("Core ML is unavailable in this build.")
    }
}

#if canImport(CoreML) && os(macOS) && arch(arm64)
import MereRunCoreMLModel

@available(macOS 15.0, *)
extension EmbeddingGemma2CoreMLRuntime {
    static func execute(root: URL, inputs: [EmbeddingGemma2Input], task: EmbeddingGemma2Task,
                        title: String?, dimensions: Int, maxTokens: Int?) async throws
        -> (embeddings: [[Float]], tokenCounts: [Int]) {
        let session = try EmbeddingGemma2CoreMLSession(root: root)
        return try await session.embed(inputs: inputs, task: task, title: title, dimensions: dimensions, maxTokens: maxTokens)
    }

    static func execute(model: EmbeddingGemma2, inputs: [EmbeddingGemma2Input], task: EmbeddingGemma2Task,
                        title: String?, dimensions: Int, maxTokens: Int?) async throws
        -> (embeddings: [[Float]], tokenCounts: [Int]) {
        let limit = min(maxTokens ?? model.config.maximumTokens, model.config.maximumTokens)
        var vectors: [[Float]] = [], counts: [Int] = []
        // Keep media preparation and long GPU records bounded. Text batches use the
        // packed ANE path; no model instances survive beyond their session owner.
        var pending: [MultimodalSequence] = []
        func append(_ sequences: [MultimodalSequence]) async throws {
            let embeddings = try await model.embed(sequences: sequences)
            for (sequence, vector) in zip(sequences, embeddings) {
                var prefix = Array(vector.prefix(dimensions))
                let norm = sqrt(prefix.reduce(Float(0)) { $0 + $1 * $1 })
                guard norm.isFinite, norm > 0, prefix.count == dimensions, prefix.allSatisfy(\.isFinite) else {
                    throw EmbeddingGemma2Error.invalidConfiguration("Core ML returned an invalid embedding.")
                }
                for index in prefix.indices { prefix[index] /= norm }
                vectors.append(prefix)
                counts.append(sequence.ids.count)
            }
        }
        for input in inputs {
            try Task.checkCancellation()
            let media = input.content.contains { if case .text = $0 { return false }; return true }
            if media, !pending.isEmpty { try await append(pending); pending.removeAll(keepingCapacity: true) }
            let sequence = try await prepare(input, model: model, task: task, title: title, limit: limit)
            if media { try await append([sequence]) }
            else {
                pending.append(sequence)
                if pending.count == 64 { try await append(pending); pending.removeAll(keepingCapacity: true) }
            }
        }
        if !pending.isEmpty { try await append(pending) }
        return (vectors, counts)
    }

    private static func prepare(_ input: EmbeddingGemma2Input, model: EmbeddingGemma2,
                                task: EmbeddingGemma2Task, title: String?, limit: Int) async throws -> MultimodalSequence {
        let hasMedia = input.content.contains { if case .text = $0 { return false }; return true }
        if !hasMedia {
            let text = input.content.map { if case .text(let text) = $0 { return text }; return "" }.joined()
            let body = model.tokenizer.encode(task.format(text, title: title))
            return MultimodalSequence(ids: [model.config.bos_token_id] + body.prefix(limit - 2) + [model.config.eos_token_id])
        }
        let reserved: Set<Int32> = [255999, 256000, 258880, 258881, 258882, 258883, 258884]
        func textIDs(_ text: String) throws -> [Int32] {
            let ids = model.tokenizer.encode(text)
            guard reserved.isDisjoint(with: ids) else {
                throw EmbeddingGemma2Error.invalidInput("Text and titles in mixed records cannot contain reserved media markers.")
            }
            return ids
        }
        var seq = MultimodalSequence(ids: [model.config.bos_token_id])
        var pendingText = task.format("", title: title)
        func flushText() throws {
            seq.ids += try textIDs(pendingText)
            pendingText = ""
        }
        for part in input.content {
            switch part {
            case .text(let text): pendingText += text
            case .image(let url):
                try flushText()
                seq.appendMedia(try await model.imageSequence(url: url))
            case .audio(let url):
                try flushText()
                seq.appendMedia(try await model.audioSequence(url: url))
            case .video, .videoFrames:
                throw EmbeddingGemma2Error.invalidInput("Video is unsupported by the pinned Core ML export.")
            }
            guard seq.ids.count + 1 <= limit else {
                throw EmbeddingGemma2Error.invalidInput("Mixed record exceeds the \(limit)-token Core ML budget; media blocks cannot be truncated.")
            }
        }
        try flushText()
        guard seq.ids.count + 1 <= limit else {
            throw EmbeddingGemma2Error.invalidInput("Mixed record exceeds the \(limit)-token Core ML budget; media blocks cannot be truncated.")
        }
        seq.ids.append(model.config.eos_token_id)
        return seq
    }

}
#endif
