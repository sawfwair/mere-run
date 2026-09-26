import Foundation
import XCTest
@testable import StudioKit

@MainActor
final class ArtifactRecoveryTests: XCTestCase {
    func testFailedRunClearsUnverifiedLiveOutputWithoutDeletingFile() throws {
        let directory = try temporaryDirectory()
        let output = directory.appendingPathComponent("partial.png")
        try Data("unfinished".utf8).write(to: output)
        let job = try makeJob(output: output)
        let resolver = ArtifactResolver(fileSystem: FileManager.default)
        job.markRunning(status: "Running")
        job.consume(output.path + "\n", stream: .stdout, resolver: resolver)
        XCTAssertEqual(job.primaryArtifactURL, output)
        let result = job.finish(exitCode: 1, resolver: resolver)
        XCTAssertNil(result.outputURL)
        XCTAssertTrue(result.artifactURLs.isEmpty)
        XCTAssertNil(job.primaryArtifactURL)
        XCTAssertEqual(try String(contentsOf: output, encoding: .utf8), "unfinished")
    }

    func testReceiptBackedOutputSurvivesLaterProcessFailure() throws {
        let directory = try temporaryDirectory()
        let output = directory.appendingPathComponent("complete.png")
        try Data("complete".utf8).write(to: output)
        let job = try makeJob(output: output)
        let resolver = ArtifactResolver(fileSystem: FileManager.default)
        job.markRunning(status: "Running")
        job.consume("{\"event\":\"result\",\"exit\":0,\"outputs\":[{\"kind\":\"image\",\"path\":\"\(output.path)\"}]}\n",
                    stream: .stdout, resolver: resolver)
        job.consume(String(repeating: "trailing diagnostic\n", count: 3_000), stream: .stdout, resolver: resolver)
        let result = job.finish(exitCode: 1, resolver: resolver)
        XCTAssertEqual(result.outputURL, output)
        XCTAssertEqual(result.artifactURLs, [output])
    }

    func testFailedLibraryCompletionClearsUnverifiedPreviewAcrossReload() throws {
        let directory = try temporaryDirectory()
        let libraryURL = directory.appendingPathComponent("library.json")
        let output = directory.appendingPathComponent("partial.png")
        let library = StudioLibraryStore(libraryURL: libraryURL)
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        draft.prompt = "fixture"
        let request = try StudioCommandAdapter.makeRequest(mode: .createImage, draft: draft, source: .contract)
        library.start(request: request, commandPreview: "fixture", source: .contract)
        library.updateOutput(id: request.id, outputURL: output)
        library.updateArtifacts(id: request.id, artifactURLs: [output])
        library.complete(id: request.id, exitCode: 1, outputURL: nil, outputText: "failed", commandPreview: "fixture")
        let item = try XCTUnwrap(StudioLibraryStore(libraryURL: libraryURL).items.first)
        XCTAssertEqual(item.status, .failed)
        XCTAssertNil(item.outputURL)
        XCTAssertTrue(item.artifactURLs?.isEmpty ?? true)
    }

    func testCompletedResidentRenderSurvivesCancellationOfNextRender() throws {
        let directory = try temporaryDirectory()
        let output = directory.appendingPathComponent("completed.mp4")
        let template = try XCTUnwrap(CommandCatalog.template(id: .videoSession))
        let draft = template.defaultDraft()
        let job = Job(request: .init(
            lane: .inference, template: template, draft: draft, requestID: UUID(),
            configuration: .init(executableURL: URL(fileURLWithPath: "/usr/bin/true"), arguments: [],
                                 currentDirectoryURL: directory, environment: [:], keepsStandardInputOpen: true),
            displayCommand: "fixture", scopeSource: .contract
        ))
        let resolver = ArtifactResolver(fileSystem: FileManager.default)
        job.markRunning(status: "Ready")
        job.consume("{\"status\":\"result\",\"output\":\"\(output.path)\"}\n", stream: .stdout, resolver: resolver)
        job.setPrimaryArtifact(nil)
        job.cancelRequested = true
        let result = job.finish(exitCode: 15, resolver: resolver)
        XCTAssertEqual(result.outputURL, output)
        XCTAssertEqual(result.artifactURLs, [output])
    }

    func testEmptySuccessReceiptDoesNotConfirmAnExistingPartialFile() throws {
        let directory = try temporaryDirectory()
        let output = directory.appendingPathComponent("partial.png")
        try Data("partial".utf8).write(to: output)
        let resolver = ArtifactResolver(fileSystem: FileManager.default)
        let job = try makeJob(output: output)
        job.markRunning(status: "Running")
        job.consume("{\"event\":\"result\",\"exit\":0,\"outputs\":[]}\n", stream: .stdout, resolver: resolver)
        let result = job.finish(exitCode: 1, resolver: resolver)
        XCTAssertNil(result.outputURL)
        XCTAssertTrue(result.artifactURLs.isEmpty)
    }

    private func makeJob(output: URL) throws -> Job {
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var draft = template.defaultDraft()
        draft.prompt = "fixture"
        draft.outputPath = output.path
        return Job(request: .init(
            lane: .inference, template: template, draft: draft, requestID: UUID(),
            configuration: .init(executableURL: URL(fileURLWithPath: "/usr/bin/true"), arguments: [],
                                 currentDirectoryURL: output.deletingLastPathComponent(), environment: [:], keepsStandardInputOpen: false),
            displayCommand: "fixture", scopeSource: .contract
        ))
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }
}
