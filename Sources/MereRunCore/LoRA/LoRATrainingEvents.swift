import Foundation

public struct LoRATrainingRunEvent: Codable, Hashable, Sendable {
    public static let filenameSuffix = ".events.jsonl"

    public let sequence: Int
    public let createdAt: Date
    public let type: String
    public let stage: String?
    public let message: String?
    public let step: Int?
    public let totalSteps: Int?
    public let loss: Float?
    public let fraction: Float?
    public let path: String?
    public let metadata: [String: String]

    enum CodingKeys: String, CodingKey {
        case sequence
        case createdAt = "created_at"
        case type
        case stage
        case message
        case step
        case totalSteps = "total_steps"
        case loss
        case fraction
        case path
        case metadata
    }

    public init(
        sequence: Int,
        createdAt: Date = Date(),
        type: String,
        stage: String? = nil,
        message: String? = nil,
        step: Int? = nil,
        totalSteps: Int? = nil,
        loss: Float? = nil,
        fraction: Float? = nil,
        path: String? = nil,
        metadata: [String: String] = [:]
    ) {
        self.sequence = sequence
        self.createdAt = createdAt
        self.type = type
        self.stage = stage
        self.message = message
        self.step = step
        self.totalSteps = totalSteps
        self.loss = loss
        self.fraction = fraction
        self.path = path
        self.metadata = metadata
    }

    public static func url(nextTo outputURL: URL) -> URL {
        let base = outputURL.deletingPathExtension().lastPathComponent
        return outputURL
            .deletingLastPathComponent()
            .appendingPathComponent("\(base)\(filenameSuffix)", isDirectory: false)
    }

    public static func load(from eventURL: URL) throws -> [LoRATrainingRunEvent] {
        guard FileManager.default.fileExists(atPath: eventURL.path) else {
            return []
        }

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let text = try String(contentsOf: eventURL, encoding: .utf8)
        var events: [LoRATrainingRunEvent] = []
        for line in text.split(whereSeparator: \.isNewline) {
            let row = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !row.isEmpty, let data = row.data(using: .utf8) else { continue }
            if let event = try? decoder.decode(LoRATrainingRunEvent.self, from: data) {
                events.append(event)
            }
        }
        return events.sorted { lhs, rhs in
            if lhs.sequence == rhs.sequence {
                return lhs.createdAt < rhs.createdAt
            }
            return lhs.sequence < rhs.sequence
        }
    }
}

public final class LoRATrainingEventLogger: @unchecked Sendable {
    public let eventURL: URL

    private let lock = NSLock()
    private var nextSequence: Int

    public init(baseOutputURL: URL, resumeExisting: Bool = false) throws {
        self.eventURL = LoRATrainingRunEvent.url(nextTo: baseOutputURL)
        try FileManager.default.createDirectory(
            at: eventURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        if resumeExisting {
            let existing = try LoRATrainingRunEvent.load(from: eventURL)
            self.nextSequence = (existing.map(\.sequence).max() ?? -1) + 1
        } else {
            self.nextSequence = 0
            try Data().write(to: eventURL, options: [.atomic])
        }
    }

    public func record(
        type: String,
        stage: String? = nil,
        message: String? = nil,
        step: Int? = nil,
        totalSteps: Int? = nil,
        loss: Float? = nil,
        fraction: Float? = nil,
        path: String? = nil,
        metadata: [String: String] = [:]
    ) throws {
        lock.lock()
        defer { lock.unlock() }

        let event = LoRATrainingRunEvent(
            sequence: nextSequence,
            type: type,
            stage: stage,
            message: message,
            step: step,
            totalSteps: totalSteps,
            loss: loss,
            fraction: fraction,
            path: path,
            metadata: metadata
        )
        nextSequence += 1

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        var data = try encoder.encode(event)
        data.append(contentsOf: "\n".utf8)

        if let handle = try? FileHandle(forWritingTo: eventURL) {
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
            try handle.close()
        } else {
            try data.write(to: eventURL, options: [.atomic])
        }
    }
}
