import Foundation
import XCTest
@testable import MereRunCore

final class PretrainedModelLoaderTests: XCTestCase {
    private var temporaryRoots: [URL] = []

    override func tearDown() {
        MereRunModelPaths.setProcessModelsDirOverride(nil)
        for root in temporaryRoots {
            try? FileManager.default.removeItem(at: root)
        }
        temporaryRoots.removeAll()
        super.tearDown()
    }

    func testProvidedPathWinsBeforeManagedResolution() async throws {
        let provided = try makeTemporaryDirectory()
        let resolved = try await PretrainedModelLoader.fromPretrainedSnapshot(
            modelPath: provided.path,
            modelId: "not-a-default",
            defaultModelIds: [],
            storageId: "poc-model",
            validate: { _, _ in [provided.appendingPathComponent("missing.json")] }
        )

        XCTAssertEqual(resolved.standardizedFileURL, provided.standardizedFileURL)
    }

    func testCompleteManagedSnapshotRootDoesNotNeedHubFallback() async throws {
        let modelsDir = try makeModelsDirectory()
        let modelRoot = modelsDir.appendingPathComponent("poc-model", isDirectory: true)
        try FileManager.default.createDirectory(at: modelRoot, withIntermediateDirectories: true)
        XCTAssertTrue(
            FileManager.default.createFile(
                atPath: modelRoot.appendingPathComponent("config.json").path,
                contents: Data()
            )
        )

        let resolved = try await PretrainedModelLoader.fromPretrainedSnapshot(
            modelPath: nil,
            modelId: "poc-model",
            defaultModelIds: ["poc-model"],
            storageId: "poc-model",
            hubFallback: nil,
            validate: { root, fileManager in
                let expected = root.appendingPathComponent("config.json")
                return fileManager.fileExists(atPath: expected.path) ? [] : [expected]
            }
        )

        XCTAssertEqual(resolved.standardizedFileURL, modelRoot.standardizedFileURL)
    }

    func testIncompleteManagedSnapshotWithoutHubFallbackThrows() async throws {
        _ = try makeModelsDirectory()

        do {
            _ = try await PretrainedModelLoader.fromPretrainedSnapshot(
                modelPath: nil,
                modelId: "poc-model",
                defaultModelIds: ["poc-model"],
                storageId: "poc-model",
                hubFallback: nil,
                validate: { root, fileManager in
                    let expected = root.appendingPathComponent("config.json")
                    return fileManager.fileExists(atPath: expected.path) ? [] : [expected]
                }
            )
            XCTFail("Expected missing Hugging Face source to throw")
        } catch let error as PretrainedModelLoader.LoadError {
            XCTAssertTrue(error.localizedDescription.contains("Hugging Face Hub source"))
            XCTAssertTrue(error.localizedDescription.contains("poc-model"))
        }
    }

    func testCompleteManagedSingleFileDoesNotNeedHubFallback() async throws {
        let modelsDir = try makeModelsDirectory()
        let managedFile = modelsDir.appendingPathComponent("poc-model.gguf", isDirectory: false)
        XCTAssertTrue(FileManager.default.createFile(atPath: managedFile.path, contents: Data("ok".utf8)))

        let resolved = try await PretrainedModelLoader.fromPretrainedFile(
            modelPath: nil,
            modelId: "poc-model",
            defaultModelIds: ["poc-model"],
            relativePath: "poc-model.gguf",
            hubFallback: nil,
            validate: { url, fileManager in
                fileManager.fileExists(atPath: url.path) ? [] : [url]
            }
        )

        XCTAssertEqual(resolved.standardizedFileURL, managedFile.standardizedFileURL)
    }

    func testIncompleteManagedSingleFileWithoutHubFallbackThrows() async throws {
        _ = try makeModelsDirectory()

        do {
            _ = try await PretrainedModelLoader.fromPretrainedFile(
                modelPath: nil,
                modelId: "poc-model",
                defaultModelIds: ["poc-model"],
                relativePath: "poc-model.gguf",
                hubFallback: nil,
                validate: { url, fileManager in
                    fileManager.fileExists(atPath: url.path) ? [] : [url]
                }
            )
            XCTFail("Expected missing Hugging Face source to throw")
        } catch let error as PretrainedModelLoader.LoadError {
            XCTAssertTrue(error.localizedDescription.contains("Hugging Face Hub source"))
            XCTAssertTrue(error.localizedDescription.contains("poc-model"))
        }
    }

    private func makeModelsDirectory() throws -> URL {
        let root = try makeTemporaryDirectory()
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsDir)
        return modelsDir
    }

    private func makeTemporaryDirectory() throws -> URL {
        let root = try TestFileSystem.makeTempDir(prefix: "mererun-pretrained-loader-tests")
        temporaryRoots.append(root)
        return root
    }
}
