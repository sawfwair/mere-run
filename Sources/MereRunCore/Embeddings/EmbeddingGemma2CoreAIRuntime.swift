#if canImport(CoreAI) && os(macOS) && arch(arm64)
import Foundation
import MereRunCoreAIModel

@available(macOS 27.0, *)
extension EmbeddingGemma2CoreAIRuntime {
    static func execute(root: URL, inputs: [EmbeddingGemma2Input], task: EmbeddingGemma2Task,
                        title: String?, dimensions: Int, maxTokens: Int?) async throws
        -> (embeddings: [[Float]], tokenCounts: [Int]) {
        let parts = inputs.flatMap(\.content)
        let loadVision = parts.contains { if case .image = $0 { return true }; return false }
        let loadAudio = parts.contains { if case .audio = $0 { return true }; return false }
        let model = try await EmbeddingGemma2(bundle: root, loadVision: loadVision, loadAudio: loadAudio)
        let limit = min(maxTokens ?? model.config.max_tokens, model.config.max_tokens)
        var vectors: [[Float]] = [], counts: [Int] = []
        for input in inputs {
            let sequence = try await prepare(input, model: model, task: task, title: title, limit: limit)
            let vector = try await model.embed(sequence: sequence)
            var prefix = Array(vector.prefix(dimensions))
            let norm = sqrt(prefix.reduce(Float(0)) { $0 + $1 * $1 })
            guard norm.isFinite, norm > 0, prefix.count == dimensions, prefix.allSatisfy(\.isFinite) else {
                throw EmbeddingGemma2Error.invalidConfiguration("Core AI returned an invalid embedding.")
            }
            for index in prefix.indices { prefix[index] /= norm }
            vectors.append(prefix)
            counts.append(sequence.ids.count)
        }
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
                let soft = try await model.softTokens(ImagePreprocessor.prepare(url: url))
                seq.appendMedia(model.imageSequence(softTokens: soft))
            case .audio(let url):
                try flushText()
                let prepared = try AudioFrontend.prepare(wav16k: AudioFrontend.loadMono16k(url: url))
                let soft = try await model.audioSoftTokens(prepared)
                seq.appendMedia(model.audioSequence(softTokens: soft))
            case .video, .videoFrames:
                throw EmbeddingGemma2Error.invalidInput("Video is unsupported by the pinned Core AI export.")
            }
            guard seq.ids.count + 1 <= limit else {
                throw EmbeddingGemma2Error.invalidInput("Mixed record exceeds the \(limit)-token Core AI budget; media blocks cannot be truncated.")
            }
        }
        try flushText()
        guard seq.ids.count + 1 <= limit else {
            throw EmbeddingGemma2Error.invalidInput("Mixed record exceeds the \(limit)-token Core AI budget; media blocks cannot be truncated.")
        }
        seq.ids.append(model.config.eos_token_id)
        return seq
    }

}
#endif
