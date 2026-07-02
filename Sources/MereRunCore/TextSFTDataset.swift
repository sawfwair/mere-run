import CryptoKit
import Foundation

public struct TextSFTExample: Sendable, Hashable, Codable {
    public let id: String?
    public let sources: [String]
    public let messages: [ChatMessage]

    public init(id: String?, sources: [String], messages: [ChatMessage]) {
        self.id = id
        self.sources = sources
        self.messages = messages
    }
}
public struct TextSFTDatasetSummary: Sendable, Hashable, Codable {
    public let exampleCount: Int
    public let messageCount: Int
    public let sourceCount: Int
    public let fingerprint: String
    public let averageAssistantCharacters: Double
    public let maxAssistantCharacters: Int

    public init(
        exampleCount: Int,
        messageCount: Int,
        sourceCount: Int,
        fingerprint: String,
        averageAssistantCharacters: Double,
        maxAssistantCharacters: Int
    ) {
        self.exampleCount = exampleCount
        self.messageCount = messageCount
        self.sourceCount = sourceCount
        self.fingerprint = fingerprint
        self.averageAssistantCharacters = averageAssistantCharacters
        self.maxAssistantCharacters = maxAssistantCharacters
    }
}

public enum TextSFTDataset {
    public static func load(from url: URL) throws -> [TextSFTExample] {
        let data = try Data(contentsOf: url)
        let text = String(decoding: data, as: UTF8.self)
        var examples: [TextSFTExample] = []
        for (index, rawLine) in text.split(whereSeparator: \.isNewline).enumerated() {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !line.isEmpty else { continue }
            let lineData = Data(line.utf8)
            do {
                examples.append(try JSONDecoder().decode(TextSFTExample.self, from: lineData))
            } catch {
                throw TextSFTDatasetError.invalidJSONLine(index + 1, error.localizedDescription)
            }
        }
        try validate(examples)
        return examples
    }

    public static func validate(_ examples: [TextSFTExample]) throws {
        guard !examples.isEmpty else {
            throw TextSFTDatasetError.emptyDataset
        }

        var ids = Set<String>()
        var prompts = Set<String>()
        for (index, example) in examples.enumerated() {
            let line = index + 1
            if let id = example.id, !id.isEmpty {
                guard ids.insert(id).inserted else {
                    throw TextSFTDatasetError.duplicateID(id)
                }
            }

            guard example.messages.count >= 3 else {
                throw TextSFTDatasetError.tooFewMessages(line)
            }
            guard example.messages.first?.role == .system else {
                throw TextSFTDatasetError.invalidRoleOrder(line, "first message must be system")
            }
            guard example.messages.contains(where: { $0.role == .user }) else {
                throw TextSFTDatasetError.invalidRoleOrder(line, "example must include a user message")
            }
            guard example.messages.last?.role == .assistant else {
                throw TextSFTDatasetError.invalidRoleOrder(line, "last message must be assistant")
            }
            guard !example.sources.isEmpty else {
                throw TextSFTDatasetError.missingSources(line)
            }

            for message in example.messages {
                let trimmed = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
                guard trimmed.count >= 4 else {
                    throw TextSFTDatasetError.emptyMessage(line)
                }
            }

            if let prompt = example.messages.first(where: { $0.role == .user })?.content {
                let normalized = prompt.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                guard prompts.insert(normalized).inserted else {
                    throw TextSFTDatasetError.duplicatePrompt(line)
                }
            }
        }
    }

    public static func summarize(_ examples: [TextSFTExample]) -> TextSFTDatasetSummary {
        let messageCount = examples.reduce(0) { $0 + $1.messages.count }
        let sources = Set(examples.flatMap(\.sources))
        let assistantLengths = examples.map { example in
            example.messages.last(where: { $0.role == .assistant })?.content.count ?? 0
        }
        let totalAssistantCharacters = assistantLengths.reduce(0, +)
        let average = examples.isEmpty ? 0 : Double(totalAssistantCharacters) / Double(examples.count)
        return TextSFTDatasetSummary(
            exampleCount: examples.count,
            messageCount: messageCount,
            sourceCount: sources.count,
            fingerprint: fingerprint(examples),
            averageAssistantCharacters: average,
            maxAssistantCharacters: assistantLengths.max() ?? 0
        )
    }

    public static func fingerprint(_ examples: [TextSFTExample]) -> String {
        var hasher = SHA256()
        for example in examples {
            if let id = example.id {
                hasher.update(data: Data(id.utf8))
            }
            for source in example.sources.sorted() {
                hasher.update(data: Data(source.utf8))
            }
            for message in example.messages {
                hasher.update(data: Data(message.role.rawValue.utf8))
                hasher.update(data: Data(message.content.utf8))
                if let imageUrl = message.imageUrl {
                    hasher.update(data: Data(imageUrl.utf8))
                }
            }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

public enum TextSFTDatasetError: Error, LocalizedError, Sendable {
    case emptyDataset
    case invalidJSONLine(Int, String)
    case duplicateID(String)
    case duplicatePrompt(Int)
    case tooFewMessages(Int)
    case invalidRoleOrder(Int, String)
    case missingSources(Int)
    case emptyMessage(Int)

    public var errorDescription: String? {
        switch self {
        case .emptyDataset:
            return "Text SFT dataset is empty."
        case .invalidJSONLine(let line, let detail):
            return "Invalid JSONL record at line \(line): \(detail)"
        case .duplicateID(let id):
            return "Duplicate text SFT example id: \(id)"
        case .duplicatePrompt(let line):
            return "Duplicate user prompt in text SFT example at line \(line)."
        case .tooFewMessages(let line):
            return "Text SFT example at line \(line) must include at least system, user, and assistant messages."
        case .invalidRoleOrder(let line, let detail):
            return "Invalid text SFT message order at line \(line): \(detail)."
        case .missingSources(let line):
            return "Text SFT example at line \(line) must include at least one source id."
        case .emptyMessage(let line):
            return "Text SFT example at line \(line) contains an empty message."
        }
    }
}
