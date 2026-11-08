import Foundation
import MLX

struct MuseGlimmerCachedPrefix {
    let cache: [Gemma4AttentionCache]
    let logits: MLXArray
}

extension MuseGlimmerGenerator {
    func prefixIdentity(_ request: ChatRequest) -> String {
        // Cached state ends before the first image; later images cannot affect causal text KV.
        let media = request.messages.flatMap { message in
            (message.contentBlocks ?? []).compactMap { block -> String? in
                if case .document(let document) = block { return document.identity }
                return nil
            }
        }.joined(separator: "\u{0}")
        return [request.prefixCachePolicy?.scope ?? "", loadedModelPath ?? "", "atem-v1", String(request.showThinking),
         String(request.reasoningEffort ?? -1), String(request.maxContextTokens ?? 0), request.kvCacheMode?.rawValue ?? "default", media].joined(separator: "\u{0}")
    }

    func prefixCheckpoints(_ request: ChatRequest, tokens: [Int]) throws -> [Int: Int] {
        guard let policy = request.prefixCachePolicy, let tokenizerAndTemplate else { return [:] }
        // Never retain image-derived KV. A marked text prefix before the first
        // image remains reusable when later screenshots change.
        let firstImage = [tokenizerAndTemplate.imageStartTokenId, tokenizerAndTemplate.imageTokenId]
            .compactMap { tokens.firstIndex(of: $0) }.min() ?? tokens.count
        let maximumPrefix = min(Self.maximumCachedPrefixTokens, firstImage)
        var boundaries = policy.boundaries
        if let ttl = policy.wholePromptTTL, let last = request.messages.indices.last {
            boundaries.append(.init(messageIndex: last, blockCount: request.messages[last].contentBlocks?.count ?? 1, ttlSeconds: ttl))
        }
        let formatter = DateFormatter(); formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX"); formatter.timeZone = .current; formatter.dateFormat = "yyyy-MM-dd"
        var result: [Int: Int] = [:]
        if let ttl = policy.toolPrefixTTL {
            let toolPrefix = try tokenizerAndTemplate.encodeForGeneration(messages: [], tools: request.tools,
                reasoningStrength: Self.reasoningStrength(request.reasoningEffort), currentDate: formatter.string(from: Date()),
                maxLength: Int.max, truncate: false, addGenerationPrompt: false)
            let count = Self.cacheBoundaryTokenCount(prepared: tokens, renderedBoundary: toolPrefix)
            if count > 0, count <= maximumPrefix { result[count] = ttl }
        }
        for boundary in policy.toolBoundaries {
            guard let tools = request.tools, let index = tools.firstIndex(where: { $0.name == boundary.throughTool }) else { continue }
            let prefix = try tokenizerAndTemplate.encodeForGeneration(messages: [], tools: Array(tools.prefix(index + 1)),
                reasoningStrength: Self.reasoningStrength(request.reasoningEffort), currentDate: formatter.string(from: Date()),
                maxLength: Int.max, truncate: false, addGenerationPrompt: false)
            let count = Self.cacheBoundaryTokenCount(prepared: tokens, renderedBoundary: prefix)
            if count > 0, count <= maximumPrefix { result[count] = max(result[count] ?? 0, boundary.ttlSeconds) }
        }
        for boundary in boundaries {
            guard request.messages.indices.contains(boundary.messageIndex), boundary.blockCount > 0 else { continue }
            var message = request.messages[boundary.messageIndex]
            if let blocks = message.contentBlocks {
                guard boundary.blockCount <= blocks.count else { continue }
                let shortened = ChatMessage(role: message.role, blocks: Array(blocks.prefix(boundary.blockCount)))
                message.content = shortened.content; message.contentBlocks = shortened.contentBlocks
                message.reasoningContent = shortened.reasoningContent; message.toolCalls = shortened.toolCalls
            }
            var prefixRequest = request
            prefixRequest.messages = Array(request.messages.prefix(boundary.messageIndex)) + [message]
            let prefixMessages = MuseGlimmerTokenizerAndTemplate.requestMessages(prefixRequest)
            let prefix = try tokenizerAndTemplate.encodeForGeneration(messages: prefixMessages, tools: request.tools,
                reasoningStrength: Self.reasoningStrength(request.reasoningEffort), currentDate: formatter.string(from: Date()),
                maxLength: Int.max, truncate: false, addGenerationPrompt: false)
            guard !prefix.contains(tokenizerAndTemplate.imageStartTokenId),
                  !prefix.contains(tokenizerAndTemplate.imageTokenId) else { continue }
            let count = Self.cacheBoundaryTokenCount(prepared: tokens, renderedBoundary: prefix)
            if count > 0, count <= maximumPrefix {
                result[count] = max(result[count] ?? 0, boundary.ttlSeconds)
            }
        }
        return result
    }

    static func cacheBoundaryTokenCount(prepared: [Int], renderedBoundary: [Int]) -> Int {
        // Rendering a partial message adds its closing delimiter. Cache only
        // tokens shared with the real prompt, before that synthetic delimiter
        // or a BPE token that merges across the marked content boundary.
        zip(prepared, renderedBoundary).prefix(while: { $0.0 == $0.1 }).count
    }
}
