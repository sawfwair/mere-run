import Foundation

public struct YuE2RequestPreflight: Codable, Sendable {
    public let runtime: String
    public let sourceRevision: String
    public let request: YuE2GenerationPlan
    public let nextStage: YuE2SavedStage.Stage
    public let promptTokens: Int
    public let generationTokenBudget: Int
    public let contextTokens: Int
    public let scoreTokens: Int?
    public let maximumMusicFrames: Int
    public let maximumDecodedSamples: Int
    public let acousticChunkFrames: [Int]?
}

extension YuE2GenerationPlan {
    /// Tokenizer-only preflight. Generated scores need another context check after planning.
    public func preflight(tokenizerURL: URL) throws -> YuE2RequestPreflight {
        try preflight(tokenizer: YuE2Tokenizer(url: tokenizerURL))
    }

    func preflight(tokenizer: YuE2Tokenizer) throws -> YuE2RequestPreflight {
        try validate()
        var score = try abc.map { value in
            let text = instrumental == true
                ? try YuE2Score(abc: value).instrumentalized(keepChords: planning == .full).abc : value
            return try tokenizer.encode(text)
        }
        if planning == .off { score = [] }
        let prefix = try YuE2Protocol.prefix(plan: self, tokenizer: tokenizer, score: score)
        let nextStage: YuE2SavedStage.Stage = score == nil ? .score : .semantic
        let budget = nextStage == .score ? scoreSampling.maximumTokens : semanticSampling.maximumTokens
        guard prefix.count + budget <= YuE2Protocol.context else {
            throw YuE2Error.invalidRequest("Prompt plus generation budget exceeds 24576 tokens; shorten the input.")
        }
        let chunks = try score.map { _ in
            try YuE2Protocol.chunkRanges(frames: semanticSampling.maximumTokens, prefixTokens: prefix.count).map(\.count)
        }
        return YuE2RequestPreflight(runtime: "native-swift-mlx-request-local-v1", sourceRevision: YuE2Resources.sourceRevision,
            request: self, nextStage: nextStage, promptTokens: prefix.count, generationTokenBudget: budget,
            contextTokens: YuE2Protocol.context, scoreTokens: score?.count, maximumMusicFrames: semanticSampling.maximumTokens,
            maximumDecodedSamples: semanticSampling.maximumTokens * 1920 - 64, acousticChunkFrames: chunks)
    }
}
