import Foundation
import XCTest
@testable import MereRunCore

final class ModelComponentResolverTests: MereRunCoreTestCase {

    func testStaleModelReferenceFallsBackToLocalComponent() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let maxRoot = modelsRoot
            .appendingPathComponent("image-zimage-max", isDirectory: true)
        let localTokenizer = maxRoot.appendingPathComponent("tokenizer", isDirectory: true)
        try TestFileSystem.createDirectory(localTokenizer)
        try MereRunModelManifest.template(for: .zetaMax, createdAt: Date(timeIntervalSince1970: 0)).write(to: maxRoot)

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
        try TestFileSystem.createDirectory(sharedTokenizer)
        try MereRunModelManifest.template(for: .zetaMax, createdAt: Date(timeIntervalSince1970: 0)).write(to: sharedRoot)

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
