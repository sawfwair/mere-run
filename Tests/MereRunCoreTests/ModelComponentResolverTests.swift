import Foundation
import XCTest
@testable import MereRunCore

final class ModelComponentResolverTests: MereRunCoreTestCase {
    private func writeMinimalManagedImageModel(at root: URL, id: ModelResolver.ModelID) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        try TestFileSystem.writeFile(root.appendingPathComponent("model_index.json"), contents: Data("{}".utf8))

        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        let textEncoder = root.appendingPathComponent("text_encoder", isDirectory: true)
        let transformer = root.appendingPathComponent("transformer", isDirectory: true)
        let vae = root.appendingPathComponent("vae", isDirectory: true)
        let scheduler = root.appendingPathComponent("scheduler", isDirectory: true)

        for directory in [tokenizer, textEncoder, transformer, vae, scheduler] {
            try TestFileSystem.createDirectory(directory)
        }

        try TestFileSystem.writeFile(tokenizer.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(tokenizer.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(tokenizer.appendingPathComponent("merges.txt"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(tokenizer.appendingPathComponent("vocab.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(textEncoder.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(textEncoder.appendingPathComponent("model.safetensors"), contents: Data())
        try TestFileSystem.writeFile(transformer.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(transformer.appendingPathComponent("diffusion_pytorch_model.safetensors"), contents: Data())
        try TestFileSystem.writeFile(vae.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(vae.appendingPathComponent("diffusion_pytorch_model.safetensors"), contents: Data())
        try TestFileSystem.writeFile(scheduler.appendingPathComponent("scheduler_config.json"), contents: Data("{}".utf8))
    }

    func testStaleModelReferenceFallsBackToLocalComponent() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let maxRoot = modelsRoot
            .appendingPathComponent("image-zimage-max", isDirectory: true)
        let localTokenizer = maxRoot.appendingPathComponent("tokenizer", isDirectory: true)
        try writeMinimalManagedImageModel(at: maxRoot, id: .zetaMax)

        var manifest = MereRunModelManifest.template(for: .zetaMax, createdAt: Date(timeIntervalSince1970: 0))
        manifest.components = .init(
            tokenizer: .model(modelID: "image-zimage-nano", path: "tokenizer")
        )

        let resolver = ModelResolver()
        let componentResolver = ModelComponentResolver(
            modelRootURL: maxRoot,
            manifest: manifest,
            modelResolver: resolver
        )

        let resolved = try componentResolver.resolveDirectory(for: .tokenizer, fallbackLocalPath: "tokenizer")
        XCTAssertEqual(resolved.directoryURL.standardizedFileURL, localTokenizer.standardizedFileURL)
        XCTAssertEqual(resolved.sourceModelRootURL.standardizedFileURL, maxRoot.standardizedFileURL)
    }

    func testAnyOfResolvesFirstExistingCandidate() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let sharedRoot = modelsRoot
            .appendingPathComponent("image-zimage-max", isDirectory: true)
        let sharedTokenizer = sharedRoot.appendingPathComponent("tokenizer", isDirectory: true)
        try writeMinimalManagedImageModel(at: sharedRoot, id: .zetaMax)

        let baseRoot = temp.appendingPathComponent("image-zimage-base", isDirectory: true)
        try TestFileSystem.createDirectory(baseRoot)

        var manifest = MereRunModelManifest.template(for: .zetaBase, createdAt: Date(timeIntervalSince1970: 0))
        manifest.components = .init(
            tokenizer: .anyOf([
                .local(path: "tokenizer"),
                .model(modelID: "image-zimage-max", path: "tokenizer"),
            ])
        )

        let resolver = ModelResolver()
        let componentResolver = ModelComponentResolver(
            modelRootURL: baseRoot,
            manifest: manifest,
            modelResolver: resolver
        )

        let resolved = try componentResolver.resolveDirectory(for: .tokenizer, fallbackLocalPath: "tokenizer")
        XCTAssertEqual(resolved.directoryURL.standardizedFileURL, sharedTokenizer.standardizedFileURL)
        XCTAssertEqual(resolved.sourceModelRootURL.standardizedFileURL, sharedRoot.standardizedFileURL)
    }

    func testAnyOfFailsWhenNoCandidatesResolve() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("model", isDirectory: true)
        try TestFileSystem.createDirectory(root)

        let manifest = MereRunModelManifest(
            id: "custom",
            components: .init(tokenizer: .anyOf([.local(path: "missing-tokenizer")]))
        )

        let componentResolver = ModelComponentResolver(modelRootURL: root, manifest: manifest)

        XCTAssertThrowsError(try componentResolver.resolveDirectory(for: .tokenizer, fallbackLocalPath: "tokenizer"))
    }
}
