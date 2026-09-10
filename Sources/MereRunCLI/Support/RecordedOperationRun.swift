import AudioCore
import Foundation
import MereRunCore
import MereRunExecution

/// Presents typed operation records to local run inspection and listing.
/// Wire responses retain each family's versioned record and compatibility key.
enum RecordedOperationRun {
    case image(ImageRunRecord)
    case transcription(SpeechTranscriptionRunRecord)

    enum Kind: String {
        case image = "image_run"
        case transcription = "transcription_run"

        var filename: String {
            switch self {
            case .image: return ImageRunRecord.filename
            case .transcription: return SpeechTranscriptionRunRecord.filename
            }
        }

        var label: String { self == .image ? "Image" : "Transcription" }
        var command: [String] { self == .image ? ["image", "generate"] : ["speech", "transcribe"] }
        var retryActionID: String { self == .image ? "retry-image-run" : "retry-transcription-run" }
    }

    static let kinds: [Kind] = [.image, .transcription]

    static func location(at url: URL, fileManager: FileManager = .default) -> (kind: Kind, url: URL)? {
        for kind in kinds {
            let recordURL = url.lastPathComponent == kind.filename ? url : url.appendingPathComponent(kind.filename)
            if fileManager.fileExists(atPath: recordURL.path) { return (kind, recordURL) }
        }
        return nil
    }

    static func inspect(kind: Kind, at url: URL) throws -> Self {
        switch kind {
        case .image: return .image(try ImageRunRecord.inspect(at: url))
        case .transcription: return .transcription(try SpeechTranscriptionRunRecord.inspect(at: url))
        }
    }

    var kind: Kind {
        switch self {
        case .image: return .image
        case .transcription: return .transcription
        }
    }

    var id: UUID {
        switch self {
        case .image(let record): return record.id
        case .transcription(let record): return record.id
        }
    }

    var state: RunState {
        switch self {
        case .image(let record): return record.state
        case .transcription(let record): return record.state
        }
    }

    var createdAt: Date {
        switch self {
        case .image(let record): return record.createdAt
        case .transcription(let record): return record.createdAt
        }
    }

    var updatedAt: Date {
        switch self {
        case .image(let record): return record.updatedAt
        case .transcription(let record): return record.updatedAt
        }
    }

    var artifacts: [RunArtifact] {
        switch self {
        case .image(let record): return record.artifacts
        case .transcription(let record): return record.artifacts
        }
    }

    var canRetry: Bool {
        switch self {
        case .image(let record): return record.state.isTerminal && record.resolvedOptions != nil
        case .transcription(let record): return record.canRetry
        }
    }

    var summary: String {
        "\(kind.label) run \(id.uuidString.lowercased()): \(state.rawValue), \(artifacts.count) artifact(s)."
    }

    func inspectionResult(path: String) -> RunInspectionResult {
        var result = RunInspectionResult(kind: kind.rawValue, path: path, runDirectory: nil, report: nil, plan: nil)
        switch self {
        case .image(let record): result.imageRun = record
        case .transcription(let record): result.transcriptionRun = record
        }
        return result
    }

    func retryAction(path: String, cwd: String) -> DeclarativeAction {
        DeclarativeAction(
            id: kind.retryActionID, label: "Retry \(kind.label.lowercased()) run", kind: .command, style: .primary,
            enabled: canRetry,
            command: DeclarativeCommand(argv: ["mere.run", "run", "retry", path], cwd: cwd, commandPath: ["run", "retry"])
        )
    }
}

extension RunInspectionResult {
    var recordedOperation: RecordedOperationRun? {
        if let imageRun { return .image(imageRun) }
        if let transcriptionRun { return .transcription(transcriptionRun) }
        return nil
    }
}
