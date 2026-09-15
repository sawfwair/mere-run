import Foundation
import XCTest
@testable import MereRunCore
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

final class ImageRunRecordTests: XCTestCase {
    func testOutputCannotOverwriteRecordOrLock() throws {
        let root = try temporaryDirectory()
        let directory = root.appendingPathComponent("run")
        for name in [ImageRunRecord.filename, ".image-run.lock", "inputs/input.png"] {
            let options = ImageGenerationOptions(prompt: "A camera", outputURL: directory.appendingPathComponent(name))
            XCTAssertThrowsError(try ImageRunSession(directory: directory, requested: options, modelSelector: "fixture")) {
                XCTAssertEqual(($0 as? ImageGenerationIssue)?.code, "run_output_conflict")
            }
            XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
        }
        XCTAssertNoThrow(try ImageRunSession.validateOutput(directory.appendingPathComponent("output.png"), in: directory))
    }

    func testDefaultSeedPolicyPreservesRuntimeBehavior() {
        for backend in ImageGenerationBackend.allCases {
            XCTAssertEqual(ImageGenerationSeed.resolve(0, prompt: "hello", backend: backend), 0)
            XCTAssertEqual(ImageGenerationSeed.resolve(UInt64.max, prompt: "hello", backend: backend), UInt64.max)
        }
        for backend: ImageGenerationBackend in [.flux1, .hiDreamO1, .senseNovaU15, .krea2, .ideogram4] {
            XCTAssertEqual(ImageGenerationSeed.resolve(nil, prompt: "hello", backend: backend), 0xa430_d846_80aa_bd0b)
        }
    }

    func testRecordsRequestedAndEffectiveSettingsBeforeInferenceAndKeepsArtifact() async throws {
        let root = try temporaryDirectory()
        let manifest = MereRunModelManifest(
            id: "fixture", family: .klein, defaults: .init(steps: 28, cfg: 4),
            sources: [.init(role: "weights", repository: "example/model", revision: "abc123")]
        )
        try manifest.write(to: root)
        let options = ImageGenerationOptions(prompt: "Original prompt", outputURL: root.appendingPathComponent("result.png"))
        let directory = root.appendingPathComponent("run")
        let session = try ImageRunSession(directory: directory, requested: options, modelSelector: "fixture")
        var expanded = options
        expanded.prompt = "Expanded prompt"
        let plan = try ImageGenerationPlan.resolve(expanded, modelRoot: root, manifest: manifest)
        let outcome = try await ImageGenerationOperation.execute(plan, recording: session, executor: { _, request, _ in
            let active = try ImageRunRecord.inspect(at: directory)
            XCTAssertEqual(active.state, .running)
            XCTAssertEqual(active.requested.prompt, "Original prompt")
            XCTAssertNil(active.requested.steps)
            XCTAssertNil(active.requested.seed)
            XCTAssertEqual(active.effective, request)
            XCTAssertEqual(request.prompt, "Expanded prompt")
            XCTAssertEqual(request.steps, 28)
            XCTAssertEqual(request.guidanceScale, 4)
            let seed = try XCTUnwrap(request.seed)
            try Data("rendered fixture".utf8).write(to: request.outputURL)
            return GenerationResult(outputURL: request.outputURL, seed: seed)
        })
        let record = try ImageRunRecord.inspect(at: directory)
        XCTAssertEqual(record.state, .succeeded)
        XCTAssertEqual(record.id, outcome.id)
        XCTAssertEqual(record.effective?.seed, outcome.result.seed)
        XCTAssertEqual(record.modelManifest?.sources?.first?.revision, "abc123")
        XCTAssertEqual(record.backend, .flux2Klein)
        XCTAssertEqual(record.artifacts.count, 1)
        let artifact = try XCTUnwrap(record.artifacts.first)
        try FileManager.default.removeItem(at: options.outputURL)
        XCTAssertEqual(try ImageRunRecord.Artifact.read(artifact.url), artifact)
        XCTAssertEqual(try ImageRunRecord.decode(ImageRunRecord.encoder().encode(record)), record)
    }

    func testFailureAndCancellationPersistDifferentTerminalStates() async throws {
        let root = try temporaryDirectory()
        let options = ImageGenerationOptions(prompt: "A camera", outputURL: root.appendingPathComponent("result.png"), seed: 42)
        let plan = try ImageGenerationPlan.resolve(options, modelRoot: root, manifest: .init(id: "fixture", family: .zimage))
        for cancelled in [false, true] {
            let directory = root.appendingPathComponent(cancelled ? "cancelled" : "failed")
            let session = try ImageRunSession(directory: directory, requested: options, modelSelector: "fixture")
            do {
                _ = try await ImageGenerationOperation.execute(plan, recording: session, executor: { _, _, _ in
                    if cancelled { throw CancellationError() }
                    throw ImageGenerationIssue("fixture_failure", "Checkpoint rejected")
                })
                XCTFail("Expected failure")
            } catch {
                XCTAssertEqual(error is CancellationError, cancelled)
            }
            let record = try ImageRunRecord.inspect(at: directory)
            XCTAssertEqual(record.state, cancelled ? .cancelled : .failed)
            XCTAssertEqual(record.issue?.code, cancelled ? "cancelled" : "fixture_failure")
            XCTAssertEqual(record.effective?.seed, 42)
            XCTAssertTrue(record.artifacts.isEmpty)
        }
    }

    func testCancelledExecutorErrorRetainsCancelledRunState() async throws {
        let root = try temporaryDirectory()
        let options = ImageGenerationOptions(prompt: "A camera", outputURL: root.appendingPathComponent("result.png"), seed: 42)
        let plan = try ImageGenerationPlan.resolve(options, modelRoot: root, manifest: .init(id: "fixture", family: .zimage))
        let directory = root.appendingPathComponent("cancelled")
        let session = try ImageRunSession(directory: directory, requested: options, modelSelector: "fixture")
        let task = Task {
            try await ImageGenerationOperation.execute(plan, recording: session, executor: { _, _, _ in
                withUnsafeCurrentTask { $0?.cancel() }
                throw URLError(.cancelled)
            })
        }
        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch {
            XCTAssertTrue(error is CancellationError)
        }
        let record = try ImageRunRecord.inspect(at: directory)
        XCTAssertEqual(record.state, .cancelled)
        XCTAssertEqual(record.issue?.code, "cancelled")
        XCTAssertTrue(record.artifacts.isEmpty)
    }

    func testOutputFailureCannotCreateSuccessfulRecord() async throws {
        let root = try temporaryDirectory()
        let options = ImageGenerationOptions(prompt: "A camera", outputURL: root.appendingPathComponent("missing.png"), seed: 1)
        let directory = root.appendingPathComponent("run")
        let session = try ImageRunSession(directory: directory, requested: options, modelSelector: "fixture")
        let plan = try ImageGenerationPlan.resolve(options, modelRoot: root, manifest: .init(id: "fixture", family: .zimage))
        do {
            _ = try await ImageGenerationOperation.execute(plan, recording: session, executor: { _, request, _ in
                GenerationResult(outputURL: request.outputURL, seed: 1)
            })
            XCTFail("A missing artifact cannot be successful")
        } catch {
            XCTAssertEqual(try ImageRunRecord.inspect(at: directory).state, .failed)
        }
    }

    func testRetrySnapshotsUploadsAndPreservesParentAndResolvedPolicy() async throws {
        let root = try temporaryDirectory()
        let upload = root.appendingPathComponent("upload.png")
        try Data("input fixture".utf8).write(to: upload)
        let manifest = MereRunModelManifest(id: "fixture", family: .klein, defaults: .init(steps: 28, cfg: 4))
        try manifest.write(to: root)
        let options = ImageGenerationOptions(prompt: "A camera", outputURL: root.appendingPathComponent("result.png"), seed: 42, inputImage: upload)
        let directory = root.appendingPathComponent("run")
        let session = try ImageRunSession(directory: directory, requested: options, modelSelector: "fixture")
        let policy = ImageGenerationPolicy(kleinUsesManifestDefaults: false, kleinInputAsReference: false)
        let plan = try ImageGenerationPlan.resolve(options, modelRoot: root, manifest: manifest, policy: policy)
        _ = try await ImageGenerationOperation.execute(plan, recording: session, executor: fixtureExecutor)
        let parentData = try Data(contentsOf: ImageRunRecord.recordURL(at: directory))
        try FileManager.default.removeItem(at: upload)
        let retry = try ImageRunSession.retryPlan(at: directory)
        XCTAssertNotEqual(retry.session.directory, directory)
        XCTAssertEqual(retry.plan.request.seed, 42)
        XCTAssertEqual(retry.plan.request.steps, 4)
        XCTAssertEqual(retry.plan.request.guidanceScale, 1)
        XCTAssertNotNil(retry.plan.request.inputImage)
        XCTAssertEqual(retry.plan.policy, policy)
        _ = try await ImageGenerationOperation.execute(retry.plan, recording: retry.session, executor: fixtureExecutor)
        let retried = try ImageRunRecord.inspect(at: retry.session.directory)
        XCTAssertEqual(retried.parentID, session.id)
        XCTAssertEqual(retried.state, .succeeded)
        XCTAssertEqual(try Data(contentsOf: ImageRunRecord.recordURL(at: directory)), parentData)
    }

    func testRetryRejectsChangedInputAndModelWithoutCreatingDirectory() async throws {
        let root = try temporaryDirectory()
        let input = root.appendingPathComponent("input.png")
        try Data("original".utf8).write(to: input)
        var manifest = MereRunModelManifest(id: "fixture", family: .klein)
        try manifest.write(to: root)
        let options = ImageGenerationOptions(prompt: "A camera", outputURL: root.appendingPathComponent("result.png"), inputImage: input)
        let directory = root.appendingPathComponent("run")
        let session = try ImageRunSession(directory: directory, requested: options, modelSelector: "fixture")
        let plan = try ImageGenerationPlan.resolve(options, modelRoot: root, manifest: manifest)
        _ = try await ImageGenerationOperation.execute(plan, recording: session, executor: fixtureExecutor)
        let contents = try FileManager.default.contentsOfDirectory(atPath: root.path)
        let record = try ImageRunRecord.inspect(at: directory)
        let snapshot = try XCTUnwrap(record.inputs.first?.url)
        try Data("changed".utf8).write(to: snapshot)
        XCTAssertThrowsError(try ImageRunSession.retryPlan(at: directory)) {
            XCTAssertEqual(($0 as? ImageGenerationIssue)?.code, "run_input_changed")
        }
        try Data("original".utf8).write(to: snapshot)
        manifest.defaults = .init(steps: 100, cfg: 4)
        try manifest.write(to: root)
        XCTAssertThrowsError(try ImageRunSession.retryPlan(at: directory)) {
            XCTAssertEqual(($0 as? ImageGenerationIssue)?.code, "run_model_changed")
        }
        try FileManager.default.removeItem(at: root.appendingPathComponent(MereRunModelManifest.filename))
        XCTAssertThrowsError(try ImageRunSession.retryPlan(at: directory)) {
            XCTAssertEqual(($0 as? ImageGenerationIssue)?.code, "run_model_changed")
        }
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath: root.path)), Set(contents).subtracting([MereRunModelManifest.filename]))
    }

    func testActiveAndAbandonedRunsUseLeaseInsteadOfAge() throws {
        let root = try temporaryDirectory()
        let directory = root.appendingPathComponent("run")
        let options = ImageGenerationOptions(prompt: "A camera", outputURL: root.appendingPathComponent("result.png"))
        var session: ImageRunSession? = try ImageRunSession(directory: directory, requested: options, modelSelector: "fixture")
        XCTAssertEqual(try ImageRunRecord.inspect(at: directory).state, .preparing)
        XCTAssertThrowsError(try ImageRunSession.retryPlan(at: directory)) {
            XCTAssertEqual(($0 as? ImageGenerationIssue)?.code, "run_active")
        }
        XCTAssertNotNil(session)
        session = nil
        let recovered = try ImageRunRecord.inspect(at: directory)
        XCTAssertEqual(recovered.state, .interrupted)
        XCTAssertEqual(recovered.issue?.code, "process_interrupted")
        XCTAssertEqual(try ImageRunRecord.inspect(at: directory), recovered)
    }

    func testLeaseIsReleasedWhenOwningProcessIsKilled() throws {
        let root = try temporaryDirectory()
        let directory = root.appendingPathComponent("run")
        let options = ImageGenerationOptions(prompt: "A camera", outputURL: root.appendingPathComponent("result.png"))
        var session: ImageRunSession? = try ImageRunSession(directory: directory, requested: options, modelSelector: "fixture")
        XCTAssertNotNil(session)
        session = nil
        let child = Process()
        child.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        child.arguments = ["-c", "import fcntl,sys,signal; f=open(sys.argv[1],'r+'); fcntl.flock(f,fcntl.LOCK_EX); sys.stdout.write('1'); sys.stdout.flush(); signal.pause()", directory.appendingPathComponent(".image-run.lock").path]
        let output = Pipe()
        child.standardOutput = output
        try child.run()
        defer { if child.isRunning { kill(child.processIdentifier, SIGKILL); child.waitUntilExit() } }
        XCTAssertEqual(try output.fileHandleForReading.read(upToCount: 1), Data("1".utf8))
        XCTAssertEqual(try ImageRunRecord.inspect(at: directory).state, .preparing)
        kill(child.processIdentifier, SIGKILL)
        child.waitUntilExit()
        XCTAssertEqual(try ImageRunRecord.inspect(at: directory).state, .interrupted)
    }

    func testExistingDirectoryAndUnsupportedOrCorruptRecordsArePreserved() throws {
        let root = try temporaryDirectory()
        let options = ImageGenerationOptions(prompt: "A camera", outputURL: root.appendingPathComponent("result.png"))
        XCTAssertThrowsError(try ImageRunSession(directory: root, requested: options, modelSelector: "fixture"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent(ImageRunRecord.filename).path))
        let directory = root.appendingPathComponent("run")
        let session = try ImageRunSession(directory: directory, requested: options, modelSelector: "fixture")
        try session.fail(ImageGenerationIssue("fixture", "failure"))
        let url = ImageRunRecord.recordURL(at: directory)
        let original = try String(contentsOf: url, encoding: .utf8)
        let future = original.replacingOccurrences(of: "\"schemaVersion\" : 1", with: "\"schemaVersion\" : 99")
        XCTAssertNotEqual(future, original)
        try Data(future.utf8).write(to: url)
        XCTAssertThrowsError(try ImageRunRecord.inspect(at: directory)) {
            XCTAssertEqual(($0 as? ImageGenerationIssue)?.code, "run_version_unsupported")
        }
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), future)
        try Data("broken".utf8).write(to: url)
        XCTAssertThrowsError(try ImageRunRecord.inspect(at: directory))
        XCTAssertEqual(try String(contentsOf: url, encoding: .utf8), "broken")
    }

    private var fixtureExecutor: ImageGenerationOperation.Executor {
        { _, request, _ in
            try Data("fixture output".utf8).write(to: request.outputURL)
            return GenerationResult(outputURL: request.outputURL, seed: try XCTUnwrap(request.seed))
        }
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
}
