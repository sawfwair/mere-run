import Foundation
import XCTest
@testable import MereRunCore

final class ModelLocationRegistryTests: XCTestCase {
    func testRegistryRoundTripsNormalizedOrder() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        let registryURL = temp.appendingPathComponent("model_locations.json")
        let firstRoot = temp.appendingPathComponent("first", isDirectory: true)
        let secondRoot = temp.appendingPathComponent("second", isDirectory: true)
        let binding = temp.appendingPathComponent("arbitrary-name", isDirectory: true)

        var registry = ModelLocationRegistry()
        registry.addSearchRoot(firstRoot)
        registry.addSearchRoot(secondRoot)
        registry.addSearchRoot(firstRoot)
        registry.addBinding(modelID: "IMAGE-ZIMAGE-NANO", url: binding, usageTermsAcknowledged: false)
        registry.addBinding(modelID: "image-zimage-nano", url: binding, usageTermsAcknowledged: true)
        try registry.save(to: registryURL)

        let loaded = try ModelLocationRegistry.load(from: registryURL)
        XCTAssertEqual(loaded.searchRoots, [secondRoot.path, firstRoot.path])
        XCTAssertEqual(loaded.bindings, [
            .init(
                modelID: "image-zimage-nano",
                path: binding.path,
                usageTermsAcknowledged: true
            ),
        ])
    }

    func testExplicitBindingResolvesManifestlessDirectory() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        let primary = temp.appendingPathComponent("primary", isDirectory: true)
        let external = temp.appendingPathComponent("LTX-style-arbitrary-name", isDirectory: true)
        try writeMinimalImageModel(at: external, includeManifest: false)

        let locations = ModelLocationSnapshot(
            primaryRoot: primary,
            bindings: [
                .init(modelID: ModelResolver.ModelID.zetaNano.rawValue, path: external.path),
            ]
        )
        let resolved = try ModelResolver(locations: locations).resolve(.zetaNano)

        XCTAssertEqual(resolved.rootURL, external.standardizedFileURL)
        XCTAssertEqual(resolved.source, .registeredBinding)
        XCTAssertTrue(resolved.isExternallyManaged)
    }

    func testExplicitBindingResolvesAcceptedManifestlessLTX25Checkpoint() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        let primary = temp.appendingPathComponent("primary", isDirectory: true)
        let external = temp.appendingPathComponent("LTX-2.5", isDirectory: true)
        try TestFileSystem.createDirectory(external)
        for relativePath in LTX25Resources.requiredRelativePaths {
            let url = external.appendingPathComponent(relativePath)
            try TestFileSystem.createDirectory(url.deletingLastPathComponent())
            try TestFileSystem.writeFile(url, contents: Data())
        }
        try TestFileSystem.createDirectory(
            external.appendingPathComponent(LTX25Resources.textEncoderRelativePath)
                .deletingLastPathComponent()
        )
        try TestFileSystem.writeFile(
            external.appendingPathComponent(LTX25Resources.textEncoderRelativePath),
            contents: Data()
        )

        let unacceptedLocations = ModelLocationSnapshot(
            primaryRoot: primary,
            bindings: [
                .init(
                    modelID: ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue,
                    path: external.path
                ),
            ]
        )
        XCTAssertThrowsError(
            try ModelResolver(locations: unacceptedLocations).resolve(.ltxVideo25DistilledBF16)
        )

        let acceptedLocations = ModelLocationSnapshot(
            primaryRoot: primary,
            bindings: [
                .init(
                    modelID: ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue,
                    path: external.path,
                    usageTermsAcknowledged: true
                ),
            ]
        )
        let resolved = try ModelResolver(locations: acceptedLocations).resolve(.ltxVideo25DistilledBF16)

        XCTAssertEqual(resolved.rootURL, external.standardizedFileURL)
        XCTAssertEqual(resolved.source, .registeredBinding)
        XCTAssertTrue(resolved.isExternallyManaged)
    }

    func testPrimaryStoreWinsOverExternalLocations() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        let primary = temp.appendingPathComponent("primary", isDirectory: true)
        let primaryModel = primary.appendingPathComponent(
            ModelResolver.ModelID.zetaNano.rawValue,
            isDirectory: true
        )
        let external = temp.appendingPathComponent("external", isDirectory: true)
        try writeMinimalImageModel(at: primaryModel, includeManifest: true)
        try writeMinimalImageModel(at: external, includeManifest: false)

        let locations = ModelLocationSnapshot(
            primaryRoot: primary,
            bindings: [
                .init(modelID: ModelResolver.ModelID.zetaNano.rawValue, path: external.path),
            ]
        )
        let resolved = try ModelResolver(locations: locations).resolve(.zetaNano)

        XCTAssertEqual(resolved.rootURL, primaryModel.standardizedFileURL)
        XCTAssertEqual(resolved.source, .localModelStore)
        XCTAssertFalse(resolved.isExternallyManaged)
    }

    func testSearchRootUsesCanonicalDirectoryAndManifest() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        let primary = temp.appendingPathComponent("primary", isDirectory: true)
        let searchRoot = temp.appendingPathComponent("catalog", isDirectory: true)
        let modelRoot = searchRoot.appendingPathComponent(
            ModelResolver.ModelID.zetaNano.rawValue,
            isDirectory: true
        )
        try writeMinimalImageModel(at: modelRoot, includeManifest: true)

        let resolved = try ModelResolver(
            locations: .init(primaryRoot: primary, searchRoots: [searchRoot])
        ).resolve(.zetaNano)

        XCTAssertEqual(resolved.rootURL, modelRoot.standardizedFileURL)
        XCTAssertEqual(resolved.source, .registeredSearchRoot)
        XCTAssertEqual(resolved.catalogRootURL, searchRoot.standardizedFileURL)
    }

    func testExplicitBindingResolvesSingleFileModelDirectory() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        let external = temp.appendingPathComponent("q36-checkpoint", isDirectory: true)
        try TestFileSystem.createDirectory(external)
        try TestFileSystem.writeFile(
            external.appendingPathComponent("checkpoint.gguf"),
            contents: Data()
        )
        let locations = ModelLocationSnapshot(
            primaryRoot: temp.appendingPathComponent("primary", isDirectory: true),
            bindings: [
                .init(modelID: ModelResolver.ModelID.q36NanoGGUF.rawValue, path: external.path),
            ]
        )

        let resolved = try ModelResolver(locations: locations).resolve(.q36NanoGGUF)

        XCTAssertEqual(resolved.rootURL, external.standardizedFileURL)
        XCTAssertEqual(resolved.source, .registeredBinding)
    }

    func testExplicitBindingRejectsMalformedManifest() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        let external = temp.appendingPathComponent("malformed", isDirectory: true)
        try writeMinimalImageModel(at: external, includeManifest: false)
        try TestFileSystem.writeFile(
            external.appendingPathComponent(MereRunModelManifest.filename),
            contents: Data("not-json".utf8)
        )
        let locations = ModelLocationSnapshot(
            primaryRoot: temp.appendingPathComponent("primary", isDirectory: true),
            bindings: [
                .init(modelID: ModelResolver.ModelID.zetaNano.rawValue, path: external.path),
            ]
        )

        XCTAssertThrowsError(try ModelResolver(locations: locations).resolve(.zetaNano))
    }

    private func writeMinimalImageModel(at root: URL, includeManifest: Bool) throws {
        try TestFileSystem.createDirectory(root)
        if includeManifest {
            try MereRunModelManifest.template(
                for: .zetaNano,
                createdAt: Date(timeIntervalSince1970: 0)
            ).write(to: root)
        }
        try TestFileSystem.writeFile(root.appendingPathComponent("model_index.json"), contents: Data("{}".utf8))

        for component in ["text_encoder", "transformer", "vae"] {
            let directory = root.appendingPathComponent(component, isDirectory: true)
            try TestFileSystem.createDirectory(directory)
            try TestFileSystem.writeFile(directory.appendingPathComponent("config.json"), contents: Data("{}".utf8))
            try TestFileSystem.writeFile(directory.appendingPathComponent("model.safetensors"), contents: Data())
        }

        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        try TestFileSystem.createDirectory(tokenizer)
        for filename in ["tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"] {
            try TestFileSystem.writeFile(tokenizer.appendingPathComponent(filename), contents: Data("{}".utf8))
        }

        let scheduler = root.appendingPathComponent("scheduler", isDirectory: true)
        try TestFileSystem.createDirectory(scheduler)
        try TestFileSystem.writeFile(
            scheduler.appendingPathComponent("scheduler_config.json"),
            contents: Data("{}".utf8)
        )
    }
}
