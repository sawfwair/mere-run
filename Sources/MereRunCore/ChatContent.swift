import Foundation

/// Runtime content has no provider-specific wire fields. Adapters own rendering.
public indirect enum ChatContentBlock: Codable, Hashable, Sendable {
    case text(String)
    case image(reference: String, identity: String)
    case reasoning(text: String, continuity: String?)
    case toolUse(ChatMessageToolCall)
    case toolResult(id: String, content: [ChatContentBlock], isError: Bool)
    case document(ChatDocument)
    case toolReference(String)
}

public struct ChatDocument: Codable, Hashable, Sendable {
    public let identity: String
    public let title: String?
    public let context: String?
    public let pages: [String]

    public init(identity: String, title: String? = nil, context: String? = nil, pages: [String]) {
        self.identity = identity
        self.title = title
        self.context = context
        self.pages = pages
    }
}

public enum ChatGenerationEvent: Hashable, Sendable {
    case text(String)
    case reasoning(String)
    case catalogSearchStart(ChatMessageToolCall)
    case catalogSearchResult(id: String, names: [String], error: ChatToolSearchFailure?)
    /// Only complete, validated calls can be emitted as executable tool uses.
    case toolUse(ChatMessageToolCall)
    case usage(input: Int, output: Int, cacheRead: Int, cacheWrite: Int)
}

public struct ChatPreparedPrompt: Sendable, Hashable {
    public let tokens: [Int]
    public let contextLimit: Int

    public init(tokens: [Int], contextLimit: Int) throws {
        guard !tokens.isEmpty, contextLimit > 0 else {
            throw ChatRequestIssue("messages", "prepared prompt exceeds the model context limit or is empty")
        }
        self.tokens = tokens
        self.contextLimit = contextLimit
    }
}

extension ChatJSONValue {
    public var jsonString: String {
        // All cases are finite JSON values after decoding. Use a throwing encoder
        // at external boundaries that can construct non-finite numeric values.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return (try? encoder.encode(self)).map { String(decoding: $0, as: UTF8.self) } ?? "null"
    }
}

extension ChatMessage {
    public init(role: Role, blocks: [ChatContentBlock]) {
        self.init(role: role, content: blocks.compactMap { block in
            switch block {
            case .text(let text): return text
            case .document(let document): return ([document.title, document.context].compactMap { $0 } + document.pages.enumerated().map {
                "<page number=\"\($0.offset + 1)\">\n\($0.element)\n</page>"
            }).joined(separator: "\n")
            default: return nil
            }
        }.joined(separator: "\n"), contentBlocks: blocks)
        self.imageUrl = blocks.compactMap { if case .image(let reference, _) = $0 { return reference }; return nil }.first
        self.reasoningContent = blocks.compactMap {
            if case .reasoning(let text, _) = $0 { return text }; return nil
        }.joined(separator: "\n")
        let calls = blocks.compactMap { if case .toolUse(let call) = $0 { return call }; return nil }
        self.toolCalls = calls.isEmpty ? nil : calls
    }
}

enum NativeChatTextContent {
    static func render(_ message: ChatMessage) -> String {
        guard let blocks = message.contentBlocks else { return message.content }
        return blocks.compactMap(render).joined(separator: "\n")
    }
    private static func render(_ block: ChatContentBlock) -> String? {
        switch block {
        case .text(let text): return text
        case .document(let document): return ([document.title, document.context].compactMap { $0 } + document.pages.enumerated().map {
            "<page number=\"\($0.offset + 1)\">\n\($0.element)\n</page>"
        }).joined(separator: "\n")
        case .toolReference(let name): return "Discovered tool: " + name
        case .toolResult(let id, let content, let error):
            return "Tool result \(id)\(error ? " (error)" : ""):\n" + content.compactMap(render).joined(separator: "\n")
        default: return nil
        }
    }
}
