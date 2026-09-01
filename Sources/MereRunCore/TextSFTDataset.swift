#if canImport(CryptoKit)
import CryptoKit
#else
import Crypto
#endif
import Foundation
import MediaIO

public struct TextSFTExample: Sendable, Hashable, Codable {
    public let id: String?
    public let sources: [String]
    public let messages: [ChatMessage]
    public let tools: [ToolDefinition]?

    public init(
        id: String?,
        sources: [String],
        messages: [ChatMessage],
        tools: [ToolDefinition]? = nil
    ) {
        self.id = id
        self.sources = sources
        self.messages = messages
        self.tools = tools
    }
}
public struct TextSFTDatasetSummary: Sendable, Hashable, Codable {
    public let exampleCount: Int
    public let messageCount: Int
    public let sourceCount: Int
    public let fingerprint: String
    public let averageAssistantCharacters: Double
    public let maxAssistantCharacters: Int
    public let imageReferenceCount: Int?
    public let uniqueImageCount: Int?
    public let imageByteCount: Int64?
    public let imageFingerprint: String?

    public init(
        exampleCount: Int,
        messageCount: Int,
        sourceCount: Int,
        fingerprint: String,
        averageAssistantCharacters: Double,
        maxAssistantCharacters: Int,
        imageReferenceCount: Int? = nil,
        uniqueImageCount: Int? = nil,
        imageByteCount: Int64? = nil,
        imageFingerprint: String? = nil
    ) {
        self.exampleCount = exampleCount
        self.messageCount = messageCount
        self.sourceCount = sourceCount
        self.fingerprint = fingerprint
        self.averageAssistantCharacters = averageAssistantCharacters
        self.maxAssistantCharacters = maxAssistantCharacters
        self.imageReferenceCount = imageReferenceCount
        self.uniqueImageCount = uniqueImageCount
        self.imageByteCount = imageByteCount
        self.imageFingerprint = imageFingerprint
    }
}

public enum TextSFTMediaPolicy: Sendable, Hashable {
    case forbid
    case requireSingleLocalImage
}

public struct TextSFTPreparedDataset: Sendable, Hashable {
    public let examples: [TextSFTExample]
    public let summary: TextSFTDatasetSummary
    public let imageDigestsByPath: [String: String]

    public init(
        examples: [TextSFTExample],
        summary: TextSFTDatasetSummary,
        imageDigestsByPath: [String: String] = [:]
    ) {
        self.examples = examples
        self.summary = summary
        self.imageDigestsByPath = imageDigestsByPath
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

    public static func loadForTraining(
        from url: URL,
        mediaPolicy: TextSFTMediaPolicy
    ) throws -> TextSFTPreparedDataset {
        let examples = try load(from: url)
        switch mediaPolicy {
        case .forbid:
            try validateNoMedia(examples)
            return TextSFTPreparedDataset(
                examples: examples,
                summary: summarize(examples)
            )
        case .requireSingleLocalImage:
            return try prepareSingleLocalImages(
                examples,
                datasetURL: url.standardizedFileURL
            )
        }
    }

    public static func validate(_ examples: [TextSFTExample]) throws {
        guard !examples.isEmpty else {
            throw TextSFTDatasetError.emptyDataset
        }

        var ids = Set<String>()
        var controllerStates = Set<String>()
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
                let hasToolCalls = message.role == .assistant && message.toolCalls?.isEmpty == false
                let hasToolResult = message.role == .tool && !trimmed.isEmpty
                guard trimmed.count >= 4 || hasToolCalls || hasToolResult else {
                    throw TextSFTDatasetError.emptyMessage(line)
                }
            }

            let state = controllerStateFingerprint(example)
            guard controllerStates.insert(state).inserted else {
                throw TextSFTDatasetError.duplicatePrompt(line)
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
        digest(canonicalData(examples.map(CanonicalExample.init)))
    }

    private static func validateNoMedia(_ examples: [TextSFTExample]) throws {
        for (exampleIndex, example) in examples.enumerated() {
            for message in example.messages where message.imageUrl != nil
                || message.audioUrl != nil
                || message.videoUrl != nil {
                throw TextSFTDatasetError.mediaNotSupported(exampleIndex + 1)
            }
        }
    }

    private static func prepareSingleLocalImages(
        _ examples: [TextSFTExample],
        datasetURL: URL
    ) throws -> TextSFTPreparedDataset {
        let datasetRoot = datasetURL.deletingLastPathComponent().standardizedFileURL
        let fileManager = FileManager.default
        var rewrittenExamples: [TextSFTExample] = []
        rewrittenExamples.reserveCapacity(examples.count)
        var imageReferences: [(relativePath: String, url: URL)] = []

        for (exampleIndex, example) in examples.enumerated() {
            guard !example.messages.contains(where: { $0.audioUrl != nil || $0.videoUrl != nil }) else {
                throw TextSFTDatasetError.mediaNotSupported(exampleIndex + 1)
            }

            let referencedMessages = example.messages.indices.filter { index in
                let value = example.messages[index].imageUrl?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                return value?.isEmpty == false
            }
            guard referencedMessages.count == 1,
                  let messageIndex = referencedMessages.first,
                  example.messages[messageIndex].role == .user,
                  let rawReference = example.messages[messageIndex].imageUrl else {
                throw TextSFTDatasetError.singleUserImageRequired(exampleIndex + 1)
            }

            let resolved = try resolveDatasetImage(
                rawReference,
                datasetRoot: datasetRoot,
                exampleLine: exampleIndex + 1,
                fileManager: fileManager
            )
            var messages = example.messages
            messages[messageIndex].imageUrl = resolved.url.path
            rewrittenExamples.append(TextSFTExample(
                id: example.id,
                sources: example.sources,
                messages: messages,
                tools: example.tools
            ))
            imageReferences.append((resolved.relativePath, resolved.url))
        }

        let uniqueImages = Dictionary(
            imageReferences.map { ($0.relativePath, $0.url) },
            uniquingKeysWith: { first, _ in first }
        )
        var totalBytes: Int64 = 0
        var imageRows: [CanonicalImage] = []
        var imageDigestsByPath: [String: String] = [:]
        imageRows.reserveCapacity(uniqueImages.count)
        for (relativePath, imageURL) in uniqueImages.sorted(by: { $0.key < $1.key }) {
            let values = try imageURL.resourceValues(forKeys: [.fileSizeKey])
            let byteCount = Int64(values.fileSize ?? 0)
            let sha256 = try fileDigest(imageURL)
            totalBytes += byteCount
            imageDigestsByPath[imageURL.path] = sha256
            imageRows.append(CanonicalImage(
                relativePath: relativePath,
                byteCount: byteCount,
                sha256: sha256
            ))
        }

        let imageFingerprint = digest(canonicalData(imageRows))
        let baseSummary = summarize(examples)
        let combinedFingerprint = digest(Data(
            "\(baseSummary.fingerprint)\n\(imageFingerprint)".utf8
        ))
        let summary = TextSFTDatasetSummary(
            exampleCount: baseSummary.exampleCount,
            messageCount: baseSummary.messageCount,
            sourceCount: baseSummary.sourceCount,
            fingerprint: combinedFingerprint,
            averageAssistantCharacters: baseSummary.averageAssistantCharacters,
            maxAssistantCharacters: baseSummary.maxAssistantCharacters,
            imageReferenceCount: imageReferences.count,
            uniqueImageCount: uniqueImages.count,
            imageByteCount: totalBytes,
            imageFingerprint: imageFingerprint
        )
        return TextSFTPreparedDataset(
            examples: rewrittenExamples,
            summary: summary,
            imageDigestsByPath: imageDigestsByPath
        )
    }

    private static func resolveDatasetImage(
        _ rawReference: String,
        datasetRoot: URL,
        exampleLine: Int,
        fileManager: FileManager
    ) throws -> (relativePath: String, url: URL) {
        let trimmed = rawReference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              URL(string: trimmed)?.scheme == nil,
              !trimmed.hasPrefix("/") else {
            throw TextSFTDatasetError.invalidImageReference(exampleLine, rawReference)
        }

        let imageURL = datasetRoot.appendingPathComponent(trimmed).standardizedFileURL
        let rootPath = datasetRoot.path.hasSuffix("/") ? datasetRoot.path : datasetRoot.path + "/"
        guard imageURL.path.hasPrefix(rootPath), imageURL.path != datasetRoot.path else {
            throw TextSFTDatasetError.imageOutsideDataset(exampleLine, rawReference)
        }

        var componentURL = datasetRoot
        for component in trimmed.split(separator: "/").map(String.init)
            where component != "." && component != ".." {
            componentURL.appendPathComponent(component)
            let componentValues = try componentURL.resourceValues(forKeys: [.isSymbolicLinkKey])
            guard componentValues.isSymbolicLink != true else {
                throw TextSFTDatasetError.imageSymlinkNotAllowed(exampleLine, trimmed)
            }
        }

        let attributes = try imageURL.resourceValues(forKeys: [
            .isRegularFileKey,
        ])
        guard attributes.isRegularFile == true else {
            throw TextSFTDatasetError.imageMissing(exampleLine, trimmed)
        }
        _ = try MediaImageIO.decode(imageURL)

        let relativePath = String(imageURL.path.dropFirst(rootPath.count))
        guard !relativePath.isEmpty, fileManager.fileExists(atPath: imageURL.path) else {
            throw TextSFTDatasetError.imageMissing(exampleLine, trimmed)
        }
        return (relativePath, imageURL)
    }

    static func fileDigest(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func controllerStateFingerprint(_ example: TextSFTExample) -> String {
        digest(canonicalData(CanonicalControllerState(
            messages: example.messages.dropLast().map { message in
                var normalized = message
                normalized.content = message.content
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                return normalized
            },
            tools: canonicalTools(example.tools)
        )))
    }

    private static func canonicalData<Value: Encodable>(_ value: Value) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        do {
            return try encoder.encode(value)
        } catch {
            preconditionFailure("Canonical Text SFT state encoding failed: \(error)")
        }
    }

    private static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    fileprivate static func canonicalTools(_ tools: [ToolDefinition]?) -> [ToolDefinition]? {
        tools?.map { tool in
            ToolDefinition(
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
                required: tool.required.sorted()
            )
        }.sorted { lhs, rhs in
            if lhs.name == rhs.name {
                return lhs.description < rhs.description
            }
            return lhs.name < rhs.name
        }
    }
}

private struct CanonicalImage: Encodable {
    let relativePath: String
    let byteCount: Int64
    let sha256: String
}

private struct CanonicalControllerState: Encodable {
    let messages: [ChatMessage]
    let tools: [ToolDefinition]?
}

private struct CanonicalExample: Encodable {
    let id: String?
    let sources: [String]
    let messages: [ChatMessage]
    let tools: [ToolDefinition]?

    init(_ example: TextSFTExample) {
        id = example.id
        sources = example.sources.sorted()
        messages = example.messages
        tools = example.tools
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
    case mediaNotSupported(Int)
    case singleUserImageRequired(Int)
    case invalidImageReference(Int, String)
    case imageOutsideDataset(Int, String)
    case imageMissing(Int, String)
    case imageSymlinkNotAllowed(Int, String)

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
        case .mediaNotSupported(let line):
            return "Text SFT example at line \(line) includes media, but the selected model is text-only."
        case .singleUserImageRequired(let line):
            return "VLM SFT example at line \(line) must include exactly one imageUrl on a user message."
        case .invalidImageReference(let line, let reference):
            return "VLM SFT example at line \(line) must use a dataset-relative image path, not \(reference)."
        case .imageOutsideDataset(let line, let reference):
            return "VLM SFT image at line \(line) escapes the dataset directory: \(reference)."
        case .imageMissing(let line, let reference):
            return "VLM SFT image at line \(line) is missing or not a regular file: \(reference)."
        case .imageSymlinkNotAllowed(let line, let reference):
            return "VLM SFT image at line \(line) must not be a symbolic link: \(reference)."
        }
    }
}
