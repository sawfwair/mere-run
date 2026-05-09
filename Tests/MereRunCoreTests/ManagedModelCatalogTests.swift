import XCTest
@testable import MereRunCore

final class ManagedModelCatalogTests: XCTestCase {
    func testEveryCanonicalManagedModelIDHasCatalogSpec() {
        for modelID in ModelResolver.ModelID.allCases {
            let spec = ManagedModelCatalog.spec(for: modelID.rawValue)
            XCTAssertNotNil(spec, "Missing catalog spec for \(modelID.rawValue)")
        }
    }

    func testEveryCatalogSpecMapsToManifestTemplate() {
        for spec in ManagedModelCatalog.allSpecs {
            guard let modelID = ModelResolver.ModelID(rawValue: spec.id) else {
                XCTFail("Catalog spec does not map to a canonical ModelID: \(spec.id)")
                continue
            }

            let manifest = MereRunModelManifest.template(for: modelID, createdAt: Date(timeIntervalSince1970: 0))
            XCTAssertEqual(manifest.id, spec.id)
        }
    }

    func testAllRuntimeAutoDownloadSpecsHaveManagedSource() {
        for spec in ManagedModelCatalog.allSpecs where spec.runtimeAutoDownloadAllowed {
            XCTAssertTrue(
                spec.hubFallback != nil,
                "Runtime auto-download model \(spec.id) has no configured Hugging Face source."
            )
        }
    }

    func testManagedDownloadSourcesAreHuggingFaceOnly() {
        for spec in ManagedModelCatalog.allSpecs where spec.hasAnyManagedDownloadSource() {
            XCTAssertNotNil(spec.hubFallback, "Managed model \(spec.id) should download from Hugging Face.")
        }
    }

    func testNestedASRNormalizationDoesNotRecurseThroughValidation() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        for id in ["speech-asr-qwen3", "speech-asr-parakeet"] {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))

            XCTAssertFalse(
                spec.isManagedRootComplete(root, fileManager: .default),
                "Missing nested ASR roots should validate as incomplete without recursing forever."
            )
            XCTAssertEqual(spec.normalizedRootURL(root, fileManager: .default), root.resolvingSymlinksInPath())
        }
    }

    func testNestedASRNormalizationPrefersCompleteNestedRoot() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let qwenSpec = try XCTUnwrap(ManagedModelCatalog.spec(for: "speech-asr-qwen3"))
        let qwenRoot = root.appendingPathComponent(qwenSpec.id, isDirectory: true)
        try FileManager.default.createDirectory(at: qwenRoot, withIntermediateDirectories: true)
        for file in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"] {
            XCTAssertTrue(FileManager.default.createFile(atPath: qwenRoot.appendingPathComponent(file).path, contents: Data()))
        }

        XCTAssertEqual(qwenSpec.normalizedRootURL(root, fileManager: .default), qwenRoot)

        let parakeetSpec = try XCTUnwrap(ManagedModelCatalog.spec(for: "speech-asr-parakeet"))
        let parakeetRoot = root.appendingPathComponent(parakeetSpec.id, isDirectory: true)
        try FileManager.default.createDirectory(at: parakeetRoot, withIntermediateDirectories: true)
        for file in ["config.json", "model.safetensors", "tokenizer.model"] {
            XCTAssertTrue(FileManager.default.createFile(atPath: parakeetRoot.appendingPathComponent(file).path, contents: Data()))
        }

        XCTAssertEqual(parakeetSpec.normalizedRootURL(root, fileManager: .default), parakeetRoot)
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-managed-model-catalog-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
