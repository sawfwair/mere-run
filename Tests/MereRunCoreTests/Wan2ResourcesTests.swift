import Foundation
import XCTest
@testable import MereRunCore

final class Wan2ResourcesTests: MereRunCoreTestCase {
    func testValidConvertedRootLoadsTypedConfiguration() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeValidRoot(root)

        let resources = Wan2Resources(rootURL: root)
        XCTAssertTrue(resources.validate().isEmpty)
        let config = try resources.loadConfiguration()
        XCTAssertEqual(config.modelType, "ti2v")
        XCTAssertEqual(config.dim, 3_072)
        XCTAssertEqual(config.numLayers, 30)
        XCTAssertEqual(config.vaeZDim, 48)
    }

    func testValidationReportsEveryMissingAsset() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertEqual(
            Set(Wan2Resources(rootURL: root).validate().map(\.lastPathComponent)),
            Set(["config.json", "model.safetensors", "t5_encoder.safetensors", "tokenizer.json", "vae.safetensors"])
        )
    }

    func testConfigurationRejectsNonTI2VArchitecture() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeValidRoot(root, modelType: "t2v")

        XCTAssertThrowsError(try Wan2Resources(rootURL: root).loadConfiguration()) { error in
            XCTAssertTrue(error.localizedDescription.contains("model_type must be ti2v"))
        }
    }

    func testManagedCatalogUsesPinnedConvertedSnapshot() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Wan2Resources.modelID))
        XCTAssertEqual(spec.modelID, .wan22TI2V5BMLX)
        XCTAssertEqual(spec.validationKind, .wan22TI2VMLX)
        XCTAssertEqual(spec.hubFallback?.repoId, Wan2Resources.managedRepoID)
        XCTAssertEqual(spec.hubFallback?.revision, Wan2Resources.managedRevision)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
    }

    func testManifestTemplateDeclaresWanVideoRuntime() {
        let manifest = MereRunModelManifest.template(
            for: .wan22TI2V5BMLX,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(manifest.id, Wan2Resources.modelID)
        XCTAssertEqual(manifest.engine, .wanVideo)
        XCTAssertEqual(manifest.family, .video)
        XCTAssertEqual(manifest.variant, .base)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(manifest.defaults?.steps, 40)
        XCTAssertEqual(manifest.defaults?.cfg, 5.0)
        XCTAssertEqual(manifest.defaults?.sigmaShift, 5.0)
    }

    private func writeValidRoot(_ root: URL, modelType: String = "ti2v") throws {
        let config = """
        {
          "model_type": "\(modelType)",
          "model_version": "2.2",
          "patch_size": [1, 2, 2],
          "text_len": 512,
          "in_dim": 48,
          "dim": 3072,
          "ffn_dim": 14336,
          "text_dim": 4096,
          "out_dim": 48,
          "num_heads": 24,
          "num_layers": 30,
          "vae_stride": [4, 16, 16],
          "vae_z_dim": 48,
          "sample_shift": 5.0,
          "sample_steps": 40,
          "sample_guide_scale": 5.0,
          "sample_fps": 24,
          "frame_num": 81,
          "max_area": 901120
        }
        """
        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data(config.utf8))
        for filename in ["model.safetensors", "t5_encoder.safetensors", "tokenizer.json", "vae.safetensors"] {
            try TestFileSystem.writeFile(root.appendingPathComponent(filename), contents: Data())
        }
    }
}
