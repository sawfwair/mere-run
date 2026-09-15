import Foundation

/// Validated manifest metadata; decoding audio and loading weights occur only
/// when the operation executes.
public struct ACEStepAdapterTrainingPlan: Sendable {
    public let options: ACEStepAdapterTrainingOptions
    public let manifestURL: URL
    public let outputURL: URL
    public let records: [ManifestRecord]

    public static func resolve(_ options: ACEStepAdapterTrainingOptions) throws -> Self {
        try options.validate()
        let manifestURL = ACEStepRuntimePreparation.resolveUserPath(options.dataset)
        return try Self(options: options, manifestURL: manifestURL,
                        outputURL: ACEStepRuntimePreparation.resolveUserPath(options.output),
                        records: loadManifest(from: manifestURL))
    }

    public func audioURL(for record: ManifestRecord) -> URL {
        if record.audio.hasPrefix("/") || record.audio.hasPrefix("~") {
            return ACEStepRuntimePreparation.resolveUserPath(record.audio)
        }
        return manifestURL.deletingLastPathComponent()
            .appendingPathComponent(record.audio).standardizedFileURL
    }

    public struct ManifestRecord: Codable, Sendable {
        public var audio: String
        public var caption: String
        public var lyrics: String?
    }

    public static func loadManifest(from url: URL) throws -> [ManifestRecord] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ACEStepPreparationIssue("Dataset manifest not found: \(url.path)")
        }
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let records: [ManifestRecord]
        if let decoded = try? decoder.decode([ManifestRecord].self, from: data) {
            records = decoded
        } else {
            let text = String(decoding: data, as: UTF8.self)
            records = try text.split(whereSeparator: \.isNewline)
                .enumerated()
                .map { lineNumber, line in
                    do {
                        return try decoder.decode(
                            ManifestRecord.self,
                            from: Data(line.utf8)
                        )
                    } catch {
                        throw ACEStepPreparationIssue(
                            "Invalid dataset JSONL line \(lineNumber + 1): "
                                + error.localizedDescription
                        )
                    }
                }
        }
        guard !records.isEmpty else {
            throw ACEStepPreparationIssue("Dataset manifest contains no examples.")
        }
        for (index, record) in records.enumerated() {
            if record.audio.trimmingCharacters(
                in: .whitespacesAndNewlines
            ).isEmpty {
                throw ACEStepPreparationIssue(
                    "Dataset record \(index + 1) has an empty audio path."
                )
            }
            if record.caption.trimmingCharacters(
                in: .whitespacesAndNewlines
            ).isEmpty {
                throw ACEStepPreparationIssue(
                    "Dataset record \(index + 1) has an empty caption."
                )
            }
        }
        return records
    }

}
