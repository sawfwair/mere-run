import Foundation
import StudioTestSupport
@testable import StudioKit
import XCTest

@MainActor
final class StudioModelStoreTests: XCTestCase {
    private func controller(_ runner: RecordingProcessRunner) -> MereRunController {
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        controller.cliPath = "/usr/bin/true"
        return controller
    }

    private func inventory(_ id: String) -> String {
        """
        {"inventory":{"rows":[{"id":"\(id)","category":"image","status":"installed"}]},"usageTerms":[]}
        """
    }

    private func capabilities(_ id: String) -> String {
        """
        {"models":[{"id":"\(id)","title":"Fixture","summary":"Test model","minimumUnifiedMemoryGB":1,"recommendedUnifiedMemoryGB":1,"supported":true,"reasons":[]}]}
        """
    }

    private func settle() async { try? await Task.sleep(for: .milliseconds(30)) }

    func testFailedFirstRefreshIsUnknownUntilAnEmptyInventorySucceeds() async {
        let runner = RecordingProcessRunner()
        let host = controller(runner)
        defer { host.terminateAllProcesses() }
        let store = host.modelStore
        XCTAssertFalse(store.hasInventory)
        let failed = Task { await store.refresh() }
        await settle()
        runner.starts[0].stdout("{broken"); runner.starts[0].termination(0)
        await failed.value
        XCTAssertFalse(store.hasInventory)
        XCTAssertNotNil(store.error)
        let recovered = Task { await store.refresh() }
        await settle()
        runner.starts[1].stdout("{\"inventory\":{\"rows\":[]},\"usageTerms\":[]}")
        runner.starts[1].termination(0)
        await settle()
        runner.starts[2].stdout("{\"models\":[]}"); runner.starts[2].termination(0)
        await recovered.value
        XCTAssertTrue(store.hasInventory)
        XCTAssertTrue(store.rows.isEmpty)
        XCTAssertNil(store.error)
    }

    func testRefreshPublishesSharedSnapshotAndRetainsItAfterMalformedOrFailedResponses() async {
        let runner = RecordingProcessRunner()
        let host = controller(runner)
        defer { host.terminateAllProcesses() }
        let store = host.modelStore
        XCTAssertTrue(store === host.modelStore)
        let refresh = Task { await store.refresh() }
        await settle()
        runner.starts[0].stdout(inventory("image-zimage-nano")); runner.starts[0].termination(0)
        await settle()
        XCTAssertTrue(store.rows.isEmpty, "Publish rows and metadata together")
        runner.starts[1].stdout(capabilities("image-zimage-nano")); runner.starts[1].termination(0)
        await refresh.value
        XCTAssertEqual(store.rows.first?.title, "Fixture")
        XCTAssertEqual(store.rows.first?.isInstalled, true)

        for (output, exitCode) in [("{broken", Int32(0)), ("CLI error", Int32(0)), (inventory("wrong"), Int32(1))] {
            let refresh = Task { await store.refresh() }
            await settle()
            runner.starts.last?.stdout(output); runner.starts.last?.termination(exitCode)
            await refresh.value
            XCTAssertEqual(store.rows.map(\.id), ["image-zimage-nano"])
            XCTAssertTrue(store.hasInventory)
            XCTAssertNotNil(store.error)
            XCTAssertFalse(store.isRefreshing)
        }
        let recovered = Task { await store.refresh() }
        await settle()
        runner.starts.last?.stdout(inventory("image-zimage-nano")); runner.starts.last?.termination(0)
        await recovered.value
        XCTAssertNil(store.error)
        XCTAssertEqual(store.rows.first?.title, "Fixture")

    }

    func testLateInventoryCannotReplaceANewerSnapshot() async {
        let runner = RecordingProcessRunner()
        let host = controller(runner)
        defer { host.terminateAllProcesses() }
        let old = Task { await host.modelStore.refresh() }
        await settle()
        let new = Task { await host.modelStore.refresh() }
        await settle()
        runner.starts[1].stdout(inventory("new")); runner.starts[1].termination(0)
        await settle()
        runner.starts[2].stdout(capabilities("new")); runner.starts[2].termination(0)
        await new.value
        runner.starts[0].stdout(inventory("old")); runner.starts[0].termination(0)
        await old.value
        XCTAssertEqual(host.modelStore.rows.map(\.id), ["new"])
        XCTAssertNil(host.modelStore.error)
    }

    func testChangingModelLocationRejectsOldResultsAndRefreshesFromNewLocation() async {
        let runner = RecordingProcessRunner()
        let host = controller(runner)
        let original = host.modelsRoot
        defer { host.modelsRoot = original; host.terminateAllProcesses() }
        let old = Task { await host.modelStore.refresh() }
        await settle()
        host.modelsRoot = "/tmp/studio-new-model-location"
        runner.starts[0].stdout(inventory("old")); runner.starts[0].termination(0)
        await old.value
        XCTAssertTrue(host.modelStore.rows.isEmpty)
        XCTAssertFalse(host.modelStore.hasInventory)
        let new = Task { await host.modelStore.refresh() }
        await settle()
        XCTAssertTrue(runner.starts[1].configuration.arguments.contains(host.modelsRoot))
        runner.starts[1].stdout(inventory("new")); runner.starts[1].termination(0)
        await settle()
        runner.starts[2].stdout(capabilities("new")); runner.starts[2].termination(0)
        await new.value
        XCTAssertEqual(host.modelStore.rows.map(\.id), ["new"])
    }

    private func pull(_ model: String) throws -> StudioRunRequest {
        let template = try XCTUnwrap(CommandCatalog.template(id: .modelPull))
        var draft = template.defaultDraft()
        draft.model = model
        draft.acceptModelLicense = true
        return StudioRunRequest(mode: .createImage, templateID: .modelPull, template: template, draft: draft)
    }

    func testDownloadsShareAdmissionDeduplicationCancellationAndRetryWithoutViews() async throws {
        let runner = RecordingProcessRunner()
        let host = controller(runner)
        defer { host.terminateAllProcesses() }
        let request = try pull("image-zimage-nano")
        XCTAssertTrue(host.modelStore.startPull(request))
        XCTAssertTrue(host.modelStore.startPull(try pull(request.draft.model)))
        XCTAssertEqual(runner.starts.count, 1)
        XCTAssertEqual(host.jobs.all.filter { $0.request.templateID == .modelPull }.count, 1)
        XCTAssertTrue(runner.starts[0].configuration.arguments.contains("--accept-model-license"))
        let job = try XCTUnwrap(host.modelStore.download(modelID: request.draft.model))
        XCTAssertEqual(job.lane, .inference)
        host.modelStore.cancelDownload(modelID: request.draft.model)
        XCTAssertTrue(job.cancelRequested)
        runner.starts[0].stderr("Fixture download stopped.\n")
        runner.starts[0].termination(130)
        await settle()
        XCTAssertTrue(host.modelStore.downloads.isEmpty)
        XCTAssertTrue(host.modelStore.lastCompletedDownload === job)
        XCTAssertTrue(job.log.lines.contains { $0.text == "Fixture download stopped." })
        XCTAssertEqual(job.exitCode, 130)
        XCTAssertTrue(host.modelStore.downloadMessage?.contains("resume") == true)
        XCTAssertTrue(host.modelStore.startPull(try pull(request.draft.model)))
        XCTAssertEqual(host.jobs.all.filter { $0.request.templateID == .modelPull }.count, 2)
        XCTAssertEqual(host.modelStore.downloads.count, 1)
    }

    func testDownloadLogFollowsCompletionOrderWhenTwoDownloadsFail() async throws {
        let runner = RecordingProcessRunner()
        let host = controller(runner)
        defer { host.terminateAllProcesses() }
        let first = try pull("image-zimage-nano")
        let second = try pull("video-ltx25-full-bf16")
        XCTAssertTrue(host.modelStore.startPull(first))
        XCTAssertTrue(host.modelStore.startPull(second))
        runner.starts[1].stderr("Second download failed.\n"); runner.starts[1].termination(1)
        await settle()
        XCTAssertEqual(host.modelStore.lastCompletedDownload?.request.requestID, second.id)
        runner.starts[0].stderr("First download failed later.\n"); runner.starts[0].termination(1)
        await settle()
        XCTAssertEqual(host.modelStore.lastCompletedDownload?.request.requestID, first.id)
        XCTAssertTrue(host.modelStore.lastCompletedDownload?.log.lines.contains { $0.text == "First download failed later." } == true)
    }

    func testDownloadDeduplicationDoesNotReuseAJobFromAnotherModelLocation() throws {
        let runner = RecordingProcessRunner()
        let host = controller(runner)
        let original = host.modelsRoot
        defer { host.modelsRoot = original; host.terminateAllProcesses() }
        let request = try pull("image-zimage-nano")
        XCTAssertTrue(host.modelStore.startPull(request))
        let first = try XCTUnwrap(host.modelStore.download(modelID: request.draft.model))
        host.modelsRoot = "/tmp/studio-second-model-location"
        XCTAssertNil(host.modelStore.download(modelID: request.draft.model))
        XCTAssertTrue(host.modelStore.startPull(try pull(request.draft.model)))
        XCTAssertEqual(runner.starts.count, 2)
        XCTAssertNotEqual(host.modelStore.download(modelID: request.draft.model)?.id, first.id)
        XCTAssertTrue(first.state.isActive)
        host.modelsRoot = original
        XCTAssertEqual(host.modelStore.download(modelID: request.draft.model)?.id, first.id)
    }

    func testQueuedDownloadIsVisibleAndCancellableWithoutLaunchingIt() async throws {
        let runner = RecordingProcessRunner()
        let host = controller(runner)
        defer { host.terminateAllProcesses() }
        for model in ["image-zimage-nano", "text-chat-gemma4-12b-4bit", "video-ltx25-full-bf16"] {
            XCTAssertTrue(host.modelStore.startPull(try pull(model)))
        }
        let queued = try XCTUnwrap(host.modelStore.downloads.last)
        XCTAssertTrue(queued.state.isQueued)
        XCTAssertEqual(runner.starts.count, 2)
        host.modelStore.cancelDownload(modelID: "video-ltx25-full-bf16")
        XCTAssertTrue(queued.state.isTerminal)
        XCTAssertEqual(queued.exitCode, JobResult.cancelledBeforeStartExitCode)
        XCTAssertEqual(runner.starts.count, 2)
    }

    func testDownloadCompletionRechecksCurrentModelInsteadOfTheDownloadedModel() async throws {
        let runner = RecordingProcessRunner()
        let host = controller(runner)
        defer { host.terminateAllProcesses() }
        var draft = StudioDraft(); draft.reset(for: .createImage)
        host.checkReadiness(for: .createImage, draft: draft)
        runner.starts[0].stdout(capabilities("image-zimage-nano")); runner.starts[0].termination(0)
        await settle()
        runner.starts[1].stdout("ID Category Status Size\nimage-zimage-nano image missing 1 GB\n")
        runner.starts[1].termination(0)
        await settle()
        XCTAssertTrue(host.modelStore.startPull(try pull("image-zimage-nano")))
        draft.model = "text-chat-gemma4-12b-4bit"
        host.checkReadiness(for: .createImage, draft: draft)
        runner.starts[3].stdout(capabilities(draft.model)); runner.starts[3].termination(0)
        await settle()
        runner.starts[4].stdout("ID Category Status Size\ntext-chat-gemma4-12b-4bit text installed 1 GB\n")
        runner.starts[4].termination(0)
        await settle()
        XCTAssertEqual(host.readinessByMode[.createImage], .ready)
        runner.starts[2].termination(0)
        await settle()
        let probe = try XCTUnwrap(runner.starts.last { Array($0.configuration.arguments.suffix(2)) == ["model", "list"] })
        probe.stdout("ID Category Status Size\nimage-zimage-nano image missing 1 GB\ntext-chat-gemma4-12b-4bit text installed 1 GB\n")
        probe.termination(0)
        await settle()
        XCTAssertEqual(host.readinessByMode[.createImage], .ready)
        XCTAssertEqual(host.modelStore.downloadMessage, "Download complete.")
    }

    func testFailedReadinessListCannotMarkPartialOutputReady() async {
        let runner = RecordingProcessRunner()
        let host = controller(runner)
        defer { host.terminateAllProcesses() }
        var draft = StudioDraft(); draft.reset(for: .createImage)
        host.checkReadiness(for: .createImage, draft: draft)
        runner.starts[0].stdout(capabilities("image-zimage-nano")); runner.starts[0].termination(0)
        await settle()
        runner.starts[1].stdout("ID Category Status Size\nimage-zimage-nano image installed 1 GB\n")
        runner.starts[1].termination(1)
        await settle()
        XCTAssertEqual(host.readinessByMode[.createImage], .unknown("Could not list models. Check the CLI and model location."))
    }
}
