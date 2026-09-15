import Foundation
import MereRunExecution

/// Owns a file-transcription record, retained audio, and terminal result files.
/// The process lease remains held until a terminal record is written atomically.
public final class SpeechTranscriptionRunSession: @unchecked Sendable {
    public let directory: URL
    public let id: UUID
    private let mutex = NSLock()
    private let lease: RunDirectoryLease
    private var record: SpeechTranscriptionRunRecord

    public init(directory: URL, requested: SpeechTranscriptionRunOptions, parentID: UUID? = nil) throws {
        self.directory = directory.standardizedFileURL
        self.id = UUID()
        try FileManager.default.createDirectory(at: self.directory.deletingLastPathComponent(), withIntermediateDirectories: true)
        try RunDirectoryLease.createDirectory(self.directory)
        guard let lease = try RunDirectoryLease.acquire(in: self.directory, filename: SpeechTranscriptionRunRecord.lockFilename) else {
            throw SpeechTranscriptionIssue("run_active", "The transcription run directory is already in use.")
        }
        self.lease = lease
        let now = RunRecordCodec.timestamp()
        record = SpeechTranscriptionRunRecord(
            schemaVersion: 1, id: id, parentID: parentID, createdAt: now, updatedAt: now,
            state: .preparing, requested: requested, artifacts: []
        )
        try save()
    }

    func prepare(_ plan: SpeechTranscriptionPlan) throws -> SpeechTranscriptionPlan {
        try mutex.withLock {
            guard record.state == .preparing else {
                throw SpeechTranscriptionIssue("run_state_invalid", "This transcription run has already started.")
            }
            try plan.validate()
            for file in plan.modelMetadata where try !file.isUnchanged() {
                throw SpeechTranscriptionIssue("run_model_changed", "Model metadata changed before transcription: \(file.url.path).")
            }
            let inputs = directory.appendingPathComponent("inputs")
            try FileManager.default.createDirectory(at: inputs, withIntermediateDirectories: false)
            let target = inputs.appendingPathComponent("audio").appendingPathExtension(plan.request.audioURL.pathExtension)
            try FileManager.default.copyItem(at: plan.request.audioURL, to: target)
            record.input = try RunArtifact.read(target)
            var request = plan.request
            request.audioURL = target
            let effective = SpeechTranscriptionPlan(
                request: request, decision: plan.decision, modelID: plan.modelID, modelPath: plan.modelPath,
                provider: plan.provider, modelMetadata: plan.modelMetadata
            )
            record.effective = effective
            record.state = .running
            try save()
            return effective
        }
    }

    func succeed(_ outcome: SpeechTranscriptionOutcome) throws {
        try mutex.withLock {
            guard record.state == .running else {
                throw SpeechTranscriptionIssue("run_state_invalid", "The transcription run is not running.")
            }
            let resultURL = directory.appendingPathComponent("result.json")
            let transcriptURL = directory.appendingPathComponent("transcript.txt")
            let resultArtifact = try RunRecordCodec.writeArtifact(outcome.result, to: resultURL)
            try outcome.result.text.write(to: transcriptURL, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: transcriptURL.path)
            record.artifacts = [resultArtifact, try RunArtifact.read(transcriptURL)]
            try Task.checkCancellation()
            record.state = .succeeded
            do { try save() } catch {
                record.state = .running
                throw error
            }
            lease.release()
        }
    }

    public func fail(_ error: Error) throws {
        try mutex.withLock {
            guard !record.state.isTerminal else { return }
            let previousState = record.state
            record.state = error is CancellationError ? .cancelled : .failed
            record.issue = error as? SpeechTranscriptionIssue ?? SpeechTranscriptionIssue(
                error is CancellationError ? "cancelled" : "transcription_failed", error.localizedDescription
            )
            do { try save() } catch {
                record.state = previousState
                throw error
            }
            lease.release()
        }
    }

    /// Keeps optional CLI presentation files out of the owned run directory.
    public static func validateOutput(_ output: URL, in directory: URL) throws {
        let root = directory.standardizedFileURL.resolvingSymlinksInPath()
        let target = output.standardizedFileURL.resolvingSymlinksInPath()
        guard target.path != root.path, !target.path.hasPrefix(root.path + "/") else {
            throw SpeechTranscriptionIssue("run_output_conflict", "Choose an --output path outside --run-dir. The run already retains transcript.txt and result.json.")
        }
    }

    /// Starts a new sibling run with saved settings. It does not resume a
    /// partial decoder state, reroute the backend, or modify its parent.
    public static func retryPlan(at url: URL) throws -> (session: SpeechTranscriptionRunSession, plan: SpeechTranscriptionPlan) {
        let parent = try SpeechTranscriptionRunRecord.inspect(at: url)
        guard parent.state.isTerminal else {
            throw SpeechTranscriptionIssue("run_active", "Wait for the transcription run to stop before retrying.")
        }
        guard parent.canRetry, let plan = parent.effective, let input = parent.input else {
            throw SpeechTranscriptionIssue("run_not_prepared", "This run has no retained input or local model metadata. Repeat the original transcription command.")
        }
        guard try RunArtifact.read(input.url) == input else {
            throw SpeechTranscriptionIssue("run_input_changed", "The recorded audio input changed: \(input.url.path).")
        }
        for file in plan.modelMetadata where try !file.isUnchanged() {
            throw SpeechTranscriptionIssue("run_model_changed", "Recorded model metadata changed: \(file.url.path). Start a new transcription command to use it.")
        }
        try plan.validate()
        let parentDirectory = SpeechTranscriptionRunRecord.recordURL(at: url).deletingLastPathComponent()
        let directory = parentDirectory.deletingLastPathComponent().appendingPathComponent("transcription-\(UUID().uuidString.lowercased())")
        let requested = SpeechTranscriptionRunOptions(
            request: plan.request, preferredBackend: parent.requested.preferredBackend,
            modelOverride: parent.requested.modelOverride, provider: plan.provider
        )
        let session = try SpeechTranscriptionRunSession(directory: directory, requested: requested, parentID: parent.id)
        return (session, plan)
    }

    private func save() throws {
        record.updatedAt = RunRecordCodec.timestamp()
        try RunRecordCodec.write(record, to: directory.appendingPathComponent(SpeechTranscriptionRunRecord.filename))
    }
}
