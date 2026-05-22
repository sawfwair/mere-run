import Foundation
import XCTest
@testable import MereRunCore

final class ACEStepModelContainerTests: MereRunCoreTestCase {

    func testResourcesValidationAndCachingFromLocalRoots() async throws {
        let root = try TestFileSystem.makeTempDir(prefix: "acestep-container")
        let decoderRoot = root.appending(path: "decoder")
        let vaeRoot = root.appending(path: "vae")
        let lmRoot = root.appending(path: "lm")
        let textRoot = root.appending(path: "text")

        try writeDecoderStub(at: decoderRoot)
        try writeVAEStub(at: vaeRoot)
        try writeLMStub(at: lmRoot)
        try writeLMStub(at: textRoot)

        let container = ACEStepModelContainer(
            decoderRootURL: decoderRoot,
            vaeRootURL: vaeRoot,
            lmRootURL: lmRoot,
            textEncoderRootURL: textRoot
        )

        let first = try await container.resources()
        let second = try await container.resources()

        XCTAssertEqual(first, second)
        XCTAssertEqual(first.decoderResources.modelRootURL, decoderRoot)
        XCTAssertEqual(first.vaeResources.modelRootURL, vaeRoot)
        XCTAssertEqual(first.lmResources?.modelRootURL, lmRoot)
        XCTAssertEqual(first.textEncoderResources?.modelRootURL, textRoot)
    }

    func testResourcesThrowsWhenRequiredDecoderFilesMissing() async throws {
        let root = try TestFileSystem.makeTempDir(prefix: "acestep-container-missing")
        let decoderRoot = root.appending(path: "decoder")
        let vaeRoot = root.appending(path: "vae")
        try writeVAEStub(at: vaeRoot)

        let container = ACEStepModelContainer(
            decoderRootURL: decoderRoot,
            vaeRootURL: vaeRoot
        )

        do {
            _ = try await container.resources()
            XCTFail("Expected missing decoder resources error.")
        } catch let error as ACEStepModelContainer.ContainerError {
            switch error {
            case .missingDecoderFiles(let urls):
                XCTAssertFalse(urls.isEmpty)
            default:
                XCTFail("Expected .missingDecoderFiles, got \(error).")
            }
        }
    }

    func testCheckpointsRootInitializerResolvesSubdirectories() async throws {
        let checkpointsRoot = try TestFileSystem.makeTempDir(prefix: "acestep-checkpoints")
        let turboSubdir = "acestep-v15-turbo"
        let vaeSubdir = "vae"
        let lmSubdir = "music-acestep-5hz-lm-1.7b"
        let textSubdir = "Qwen3-Embedding-0.6B"

        try writeDecoderStub(at: checkpointsRoot.appending(path: turboSubdir))
        try writeVAEStub(at: checkpointsRoot.appending(path: vaeSubdir))
        try writeLMStub(at: checkpointsRoot.appending(path: lmSubdir))
        try writeLMStub(at: checkpointsRoot.appending(path: textSubdir))

        let container = ACEStepModelContainer(
            checkpointsRootURL: checkpointsRoot,
            turboSubdirectory: turboSubdir,
            vaeSubdirectory: vaeSubdir,
            lmSubdirectory: lmSubdir,
            textEncoderSubdirectory: textSubdir
        )
        let resources = try await container.resources()

        XCTAssertEqual(resources.decoderResources.modelRootURL.lastPathComponent, turboSubdir)
        XCTAssertEqual(resources.vaeResources.modelRootURL.lastPathComponent, vaeSubdir)
        XCTAssertEqual(resources.lmResources?.modelRootURL.lastPathComponent, lmSubdir)
        XCTAssertEqual(resources.textEncoderResources?.modelRootURL.lastPathComponent, textSubdir)
    }

    private func writeDecoderStub(at root: URL) throws {
        try TestFileSystem.writeFile(root.appending(path: "config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appending(path: "model.safetensors"))
    }

    private func writeVAEStub(at root: URL) throws {
        try TestFileSystem.writeFile(root.appending(path: "config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appending(path: "diffusion_pytorch_model.safetensors"))
    }

    private func writeLMStub(at root: URL) throws {
        try TestFileSystem.writeFile(root.appending(path: "config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appending(path: "model.safetensors"))
        try TestFileSystem.writeFile(root.appending(path: "tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appending(path: "tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appending(path: "added_tokens.json"), contents: Data("[]".utf8))
    }
}
