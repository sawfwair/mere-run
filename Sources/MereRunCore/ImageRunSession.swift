import Foundation
import MereRunExecution

/// Owns one run directory and its process lease. All mutations are serialized;
/// terminal records release the lease only after their atomic write succeeds.
public final class ImageRunSession: @unchecked Sendable {
    public let directory: URL
    public let id: UUID
    private let mutex = NSLock()
    private let lease: RunDirectoryLease
    private var record: ImageRunRecord

    public init(directory: URL, requested: ImageGenerationOptions, modelSelector: String, parentID: UUID? = nil) throws {
        self.directory = directory.standardizedFileURL
        self.id = UUID()
        try Self.validateOutput(requested.outputURL, in: self.directory)
        let manager = FileManager.default
        try manager.createDirectory(at: self.directory.deletingLastPathComponent(), withIntermediateDirectories: true)
        // Refuse existing directories, including earlier runs and unrelated user files.
        try RunDirectoryLease.createDirectory(self.directory)
        guard let lease = try RunDirectoryLease.acquire(in: self.directory, filename: ".image-run.lock") else {
            throw ImageGenerationIssue("run_active", "The image run directory is already in use.")
        }
        self.lease = lease
        let now = ImageRunRecord.timestamp()
        record = ImageRunRecord(
            schemaVersion: 1, id: id, parentID: parentID, createdAt: now, updatedAt: now,
            state: .preparing, modelSelector: modelSelector, requested: requested, inputs: [], artifacts: []
        )
        try record.write(to: self.directory.appendingPathComponent(ImageRunRecord.filename))
    }

    /// Snapshot edit inputs before an HTTP upload directory can be cleaned up.
    /// Local adapters stay in place and are fingerprinted for retry validation.
    func prepare(_ plan: ImageGenerationPlan) throws -> ImageGenerationPlan {
        try mutex.withLock {
            guard record.state == .preparing else {
                throw ImageGenerationIssue("run_state_invalid", "This image run has already started.")
            }
            try Self.validateOutput(plan.options.outputURL, in: directory)
            var options = plan.options
            let inputDirectory = directory.appendingPathComponent("inputs")
            try FileManager.default.createDirectory(at: inputDirectory, withIntermediateDirectories: false)
            func snapshot(_ source: URL, name: String) throws -> URL {
                let target = inputDirectory.appendingPathComponent(name).appendingPathExtension(source.pathExtension)
                try FileManager.default.copyItem(at: source, to: target)
                record.inputs.append(try .read(target))
                return target
            }
            options.inputImage = try options.inputImage.map { try snapshot($0, name: "input") }
            options.mask = try options.mask.map { try snapshot($0, name: "mask") }
            options.referenceImages = try options.referenceImages.enumerated().map { try snapshot($0.element, name: "reference-\($0.offset)") }
            options.seed = ImageGenerationSeed.resolve(plan.request.seed, prompt: plan.request.prompt, backend: plan.backend)
            options.steps = plan.request.steps
            options.guidanceScale = plan.request.guidanceScale
            options.sigmaShift = plan.request.sigmaShift
            options.sigmas = plan.request.sigmas
            options.loras = try plan.request.loras.map { adapter in
                switch adapter {
                case .local(let path, let scale):
                    record.inputs.append(try .read(URL(fileURLWithPath: path)))
                    return ImageLoRAReference(raw: path, reference: path, scale: scale)
                case .remote:
                    throw ImageGenerationIssue("run_adapter_unresolved", "Recorded image runs require locally resolved adapters.")
                }
            }
            let resolved = try ImageGenerationPlan.resolve(options, modelRoot: plan.modelRoot, manifest: plan.manifest, policy: plan.policy)
            record.resolvedOptions = options
            record.effective = resolved.request
            record.policy = plan.policy
            record.modelRoot = plan.modelRoot
            record.modelManifest = plan.manifest
            record.manifestDigest = try ImageRunRecord.digest(plan.manifest)
            record.installedManifestDigest = try MereRunModelManifest.loadIfPresent(from: plan.modelRoot).map(ImageRunRecord.digest)
            record.backend = plan.backend
            record.state = .running
            try save()
            return resolved
        }
    }

    func succeed(_ outcome: ImageGenerationOutcome) throws {
        try mutex.withLock {
            guard record.state == .running else { throw ImageGenerationIssue("run_state_invalid", "The image run is not running.") }
            let target = directory.appendingPathComponent("output.png")
            if outcome.result.outputURL.standardizedFileURL != target.standardizedFileURL {
                try FileManager.default.copyItem(at: outcome.result.outputURL, to: target)
            }
            record.artifacts = [try .read(target)]
            record.effective = outcome.effectiveRequest
            try Task.checkCancellation()
            record.state = .succeeded
            do {
                try save()
            } catch {
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
            record.issue = error as? ImageGenerationIssue ?? ImageGenerationIssue(
                error is CancellationError ? "cancelled" : "generation_failed", error.localizedDescription
            )
            do {
                try save()
            } catch {
                record.state = previousState
                throw error
            }
            lease.release()
        }
    }

    public static func validateOutput(_ output: URL, in directory: URL) throws {
        let root = directory.standardizedFileURL.resolvingSymlinksInPath()
        let target = output.standardizedFileURL.resolvingSymlinksInPath()
        if target.path == root.path || target.path.hasPrefix(root.path + "/") {
            guard target == root.appendingPathComponent("output.png") else {
                throw ImageGenerationIssue("run_output_conflict", "An output inside --run-dir must be named output.png. Choose another output path.")
            }
        }
    }

    private func save() throws {
        record.updatedAt = ImageRunRecord.timestamp()
        try record.write(to: directory.appendingPathComponent(ImageRunRecord.filename))
    }

    /// Replays resolved settings in a new sibling directory. It does not resume
    /// denoising or re-expand a structured prompt. The parent record is immutable.
    public static func retryPlan(at url: URL) throws -> (session: ImageRunSession, plan: ImageGenerationPlan) {
        let parent = try ImageRunRecord.inspect(at: url)
        guard parent.state.isTerminal else { throw ImageGenerationIssue("run_active", "Wait for the image run to stop before retrying.") }
        guard var options = parent.resolvedOptions, let root = parent.modelRoot,
              let manifest = parent.modelManifest, let digest = parent.manifestDigest, let policy = parent.policy else {
            throw ImageGenerationIssue("run_not_prepared", "This run stopped before resolving its settings. Repeat the original image command.")
        }
        let installedDigest = try MereRunModelManifest.loadIfPresent(from: root).map(ImageRunRecord.digest)
        guard installedDigest == parent.installedManifestDigest, try ImageRunRecord.digest(manifest) == digest else {
            throw ImageGenerationIssue("run_model_changed", "The installed model manifest changed. Start a new image command to use it.")
        }
        for input in parent.inputs {
            guard try ImageRunRecord.Artifact.read(input.url) == input else {
                throw ImageGenerationIssue("run_input_changed", "A recorded image input or adapter changed: \(input.url.path).")
            }
        }
        let parentDirectory = ImageRunRecord.recordURL(at: url).deletingLastPathComponent()
        let directory = parentDirectory.deletingLastPathComponent().appendingPathComponent("image-\(UUID().uuidString.lowercased())")
        options.outputURL = directory.appendingPathComponent("output.png")
        let plan = try ImageGenerationPlan.resolve(options, modelRoot: root, manifest: manifest, policy: policy)
        let session = try ImageRunSession(directory: directory, requested: options, modelSelector: parent.modelSelector, parentID: parent.id)
        return (session, plan)
    }
}
