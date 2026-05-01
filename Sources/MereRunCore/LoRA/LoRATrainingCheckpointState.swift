import Foundation

public struct LoRATrainingCheckpointState: Codable, Sendable, Hashable {
    public static let schemaVersion: Int = 2

    public struct Phase: Codable, Sendable, Hashable {
        public let width: Int
        public let height: Int
        public let steps: Int
        public let sampleCount: Int

        public init(width: Int, height: Int, steps: Int, sampleCount: Int) {
            self.width = width
            self.height = height
            self.steps = steps
            self.sampleCount = sampleCount
        }
    }

    public struct Cursor: Codable, Sendable, Hashable {
        public let phaseIndex: Int
        public let stepInPhase: Int
        public let phaseSteps: Int

        public init(phaseIndex: Int, stepInPhase: Int, phaseSteps: Int) {
            self.phaseIndex = phaseIndex
            self.stepInPhase = stepInPhase
            self.phaseSteps = phaseSteps
        }
    }

    public let schemaVersion: Int
    public let createdAt: Date
    public let format: String
    public let baseModel: String
    public let checkpointFile: String
    public let step: Int
    public let totalSteps: Int
    public let seed: UInt64
    public let rngState: UInt64?
    public let datasetFingerprint: String?
    public let configFingerprint: String?
    public let phaseSchedule: [Phase]?
    public let phaseCursor: Cursor?
    public let configSnapshot: [String: String]?
    public let lossCSVFile: String?
    public let lossHTMLFile: String?
    public let manifestFile: String?

    public init(
        createdAt: Date = Date(),
        format: String,
        baseModel: String,
        checkpointFile: String,
        step: Int,
        totalSteps: Int,
        seed: UInt64,
        rngState: UInt64?,
        datasetFingerprint: String?,
        configFingerprint: String?,
        phaseSchedule: [Phase]? = nil,
        phaseCursor: Cursor? = nil,
        configSnapshot: [String: String]? = nil,
        lossCSVFile: String? = nil,
        lossHTMLFile: String? = nil,
        manifestFile: String? = nil
    ) {
        self.schemaVersion = Self.schemaVersion
        self.createdAt = createdAt
        self.format = format
        self.baseModel = baseModel
        self.checkpointFile = checkpointFile
        self.step = step
        self.totalSteps = totalSteps
        self.seed = seed
        self.rngState = rngState
        self.datasetFingerprint = datasetFingerprint
        self.configFingerprint = configFingerprint
        self.phaseSchedule = phaseSchedule
        self.phaseCursor = phaseCursor
        self.configSnapshot = configSnapshot
        self.lossCSVFile = lossCSVFile
        self.lossHTMLFile = lossHTMLFile
        self.manifestFile = manifestFile
    }

    public static func url(nextTo checkpointURL: URL) -> URL {
        checkpointURL
            .deletingPathExtension()
            .appendingPathExtension("checkpoint")
            .appendingPathExtension("json")
    }

    public func write(nextTo checkpointURL: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(self)
        try data.write(to: Self.url(nextTo: checkpointURL), options: [.atomic])
    }

    public static func load(nextTo checkpointURL: URL) throws -> LoRATrainingCheckpointState? {
        let url = Self.url(nextTo: checkpointURL)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(LoRATrainingCheckpointState.self, from: data)
    }

    public static func cursor(
        forCompletedStep completedStep: Int,
        phaseSchedule: [Phase]
    ) -> Cursor? {
        guard completedStep > 0, !phaseSchedule.isEmpty else { return nil }
        var remaining = completedStep
        for (index, phase) in phaseSchedule.enumerated() {
            guard phase.steps > 0 else { continue }
            if remaining <= phase.steps {
                return Cursor(
                    phaseIndex: index,
                    stepInPhase: remaining,
                    phaseSteps: phase.steps
                )
            }
            remaining -= phase.steps
        }

        guard let lastIndex = phaseSchedule.indices.last else { return nil }
        let last = phaseSchedule[lastIndex]
        guard last.steps > 0 else { return nil }
        return Cursor(
            phaseIndex: lastIndex,
            stepInPhase: last.steps,
            phaseSteps: last.steps
        )
    }

    public static func scheduleMatches(
        expected: [Phase],
        actual: [Phase]
    ) -> Bool {
        expected == actual
    }
}
