import Foundation

public struct LoRATrainingRunManifest: Codable, Sendable, Hashable {
    public static let schemaVersion: Int = 1
    public static let filename = "run.json"

    public struct DataFingerprint: Codable, Sendable, Hashable {
        public let count: Int
        public let images: [String]
        public let inputImages: [String]
        public let isEdit: Bool

        enum CodingKeys: String, CodingKey {
            case count
            case images
            case inputImages = "input_images"
            case isEdit = "is_edit"
        }

        public init(
            count: Int,
            images: [String],
            inputImages: [String],
            isEdit: Bool
        ) {
            self.count = count
            self.images = images.sorted()
            self.inputImages = inputImages.sorted()
            self.isEdit = isEdit
        }
    }

    public let version: Int
    public let createdAt: Date
    public let format: String
    public let model: String
    public let isEdit: Bool
    public let dataRoot: String?
    public let dataRootRelative: String?
    public let dataFingerprint: DataFingerprint?
    public let checkpointFiles: [String: String]
    public let step: Int
    public let totalSteps: Int
    public let seed: UInt64
    public let rngState: UInt64?
    public let datasetFingerprint: String?
    public let configFingerprint: String?
    public let phaseSchedule: [LoRATrainingCheckpointState.Phase]?
    public let phaseCursor: LoRATrainingCheckpointState.Cursor?
    public let configSnapshot: [String: String]?

    public init(
        createdAt: Date = Date(),
        format: String,
        model: String,
        isEdit: Bool,
        dataRoot: String?,
        dataRootRelative: String?,
        dataFingerprint: DataFingerprint?,
        checkpointFiles: [String: String],
        step: Int,
        totalSteps: Int,
        seed: UInt64,
        rngState: UInt64?,
        datasetFingerprint: String?,
        configFingerprint: String?,
        phaseSchedule: [LoRATrainingCheckpointState.Phase]? = nil,
        phaseCursor: LoRATrainingCheckpointState.Cursor? = nil,
        configSnapshot: [String: String]?
    ) {
        self.version = Self.schemaVersion
        self.createdAt = createdAt
        self.format = format
        self.model = model
        self.isEdit = isEdit
        self.dataRoot = dataRoot
        self.dataRootRelative = dataRootRelative
        self.dataFingerprint = dataFingerprint
        self.checkpointFiles = checkpointFiles
        self.step = step
        self.totalSteps = totalSteps
        self.seed = seed
        self.rngState = rngState
        self.datasetFingerprint = datasetFingerprint
        self.configFingerprint = configFingerprint
        self.phaseSchedule = phaseSchedule
        self.phaseCursor = phaseCursor
        self.configSnapshot = configSnapshot
    }

    enum CodingKeys: String, CodingKey {
        case version
        case createdAt = "created_at"
        case format
        case model
        case isEdit = "is_edit"
        case dataRoot = "data_root"
        case dataRootRelative = "data_root_relative"
        case dataFingerprint = "data_fingerprint"
        case checkpointFiles = "checkpoint_files"
        case step
        case totalSteps = "total_steps"
        case seed
        case rngState = "rng_state"
        case datasetFingerprint = "dataset_fingerprint"
        case configFingerprint = "config_fingerprint"
        case phaseSchedule = "phase_schedule"
        case phaseCursor = "phase_cursor"
        case configSnapshot = "config_snapshot"
    }

    public static func url(nextTo checkpointURL: URL) -> URL {
        checkpointURL.deletingLastPathComponent().appendingPathComponent(filename, isDirectory: false)
    }

    public func write(nextTo checkpointURL: URL) throws {
        let outputURL = Self.url(nextTo: checkpointURL)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(self)
        try data.write(to: outputURL, options: [.atomic])
    }

    public static func load(nextTo checkpointURL: URL) throws -> LoRATrainingRunManifest? {
        let inputURL = Self.url(nextTo: checkpointURL)
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            return nil
        }
        let data = try Data(contentsOf: inputURL)
        return try decode(from: data)
    }

    public static func decode(from data: Data) throws -> LoRATrainingRunManifest {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { container in
            let value = try container.singleValueContainer().decode(String.self)
            let iso8601 = ISO8601DateFormatter()
            if let parsed = iso8601.date(from: value) {
                return parsed
            }

            let fallback = DateFormatter()
            fallback.locale = Locale(identifier: "en_US_POSIX")
            fallback.timeZone = TimeZone(secondsFromGMT: 0)
            fallback.dateFormat = "yyyy-MM-dd HH:mm:ss"
            if let parsed = fallback.date(from: value) {
                return parsed
            }

            throw DecodingError.dataCorruptedError(
                in: try container.singleValueContainer(),
                debugDescription: "Unsupported run manifest date format: \(value)"
            )
        }
        return try decoder.decode(LoRATrainingRunManifest.self, from: data)
    }

    public static func relativePath(
        from baseDirectoryURL: URL,
        to targetPath: String?
    ) -> String? {
        guard let targetPath else { return nil }
        let trimmed = targetPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let baseComponents = baseDirectoryURL.standardizedFileURL.pathComponents
        let targetComponents = URL(fileURLWithPath: trimmed).standardizedFileURL.pathComponents

        var commonPrefixCount = 0
        while commonPrefixCount < min(baseComponents.count, targetComponents.count),
              baseComponents[commonPrefixCount] == targetComponents[commonPrefixCount] {
            commonPrefixCount += 1
        }

        let upComponents = Array(repeating: "..", count: max(baseComponents.count - commonPrefixCount, 0))
        let downComponents = Array(targetComponents.dropFirst(commonPrefixCount))
        let relativeComponents = upComponents + downComponents
        if relativeComponents.isEmpty {
            return "."
        }
        return (relativeComponents as NSArray).componentsJoined(by: "/")
    }

    public static func dataPath(
        for assetURL: URL,
        dataRootPath: String?
    ) -> String {
        let standardizedAsset = assetURL.standardizedFileURL

        guard let dataRootPath else {
            return standardizedAsset.lastPathComponent
        }
        let trimmedRoot = dataRootPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedRoot.isEmpty else {
            return standardizedAsset.lastPathComponent
        }

        let rootComponents = URL(fileURLWithPath: trimmedRoot).standardizedFileURL.pathComponents
        let assetComponents = standardizedAsset.pathComponents
        guard assetComponents.count >= rootComponents.count else {
            return standardizedAsset.lastPathComponent
        }
        for index in rootComponents.indices {
            guard assetComponents[index] == rootComponents[index] else {
                return standardizedAsset.lastPathComponent
            }
        }
        let relativeComponents = Array(assetComponents.dropFirst(rootComponents.count))
        if relativeComponents.isEmpty {
            return standardizedAsset.lastPathComponent
        }
        return (relativeComponents as NSArray).componentsJoined(by: "/")
    }
}
