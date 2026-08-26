import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class ModelInfoCommandTests: XCTestCase {
    private var temporaryDirectories: [URL] = []

    override func tearDown() {
        for directory in temporaryDirectories {
            try? FileManager.default.removeItem(at: directory)
        }
        temporaryDirectories.removeAll()
        super.tearDown()
    }

    func testLTX23SplitComponentLinesShowActualFiles() throws {
        let root = try makeTempDirectory()
        for file in [
            "split_model.json",
            "config.json",
            "embedded_config.json",
            "connector.safetensors",
            "transformer-distilled.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "vocoder.safetensors",
            "spatial_upscaler_x2_v1_1.safetensors",
            "spatial_upscaler_x2_v1_1_config.json",
            "spatial_upscaler_x1_5_v1_0.safetensors",
            "spatial_upscaler_x1_5_v1_0_config.json",
            "temporal_upscaler_x2_v1_0.safetensors",
            "temporal_upscaler_x2_v1_0_config.json",
        ] {
            try Data("fixture".utf8).write(to: root.appendingPathComponent(file))
        }
        let companion = root.appendingPathComponent("gemma3", isDirectory: true)
        try FileManager.default.createDirectory(at: companion, withIntermediateDirectories: true)

        let lines = ModelInfo.ltx23SplitComponentLines(rootURL: root, companionRootURL: companion)

        XCTAssertTrue(lines.contains("  layout: LTX 2.3 distilled split MLX files"))
        XCTAssertTrue(lines.contains { $0.contains("text_encoder:") && $0.contains("text-encoder-ltx-gemma3-12b-4bit") })
        XCTAssertTrue(lines.contains { $0.contains("transformer-distilled.safetensors") })
        XCTAssertTrue(lines.contains { $0.contains("vocoder.safetensors") })
        XCTAssertFalse(lines.contains { $0.contains("(unresolved)") })
        XCTAssertFalse(lines.contains { $0.contains("(missing)") })
    }

    func testLTX25ComponentLinesShowPackedRuntimeFiles() throws {
        let root = try makeTempDirectory()
        for relativePath in LTX25Resources.requiredRelativePaths {
            let url = root.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Data("fixture".utf8).write(to: url)
        }
        for relativePath in [
            "\(LTX25TextEncoderQuantizedPack.relativeDirectory)/model.safetensors.index.json",
            "\(LTX25TextEncoderQuantizedPack.relativeDirectory)/\(LTX25TextEncoderQuantizedPack.runtimeAssetsFilename)",
        ] {
            let url = root.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Data("fixture".utf8).write(to: url)
        }
        let manifest = MereRunModelManifest.template(
            for: .ltxVideo25DistilledBF16,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        let lines = ModelInfo.ltx25ComponentLines(rootURL: root)

        XCTAssertTrue(ModelInfo.usesLTX25Layout(manifest: manifest, expectedModelID: nil))
        XCTAssertTrue(
            lines.contains("  layout: self-contained LTX 2.5 BF16 + MLX Q4 text files")
        )
        XCTAssertTrue(lines.contains { $0.contains(LTX25Resources.transformerRelativePath) })
        XCTAssertTrue(lines.contains {
            $0.contains(LTX25TextEncoderQuantizedPack.relativeDirectory)
        })
        XCTAssertTrue(lines.contains { $0.contains(LTX25Resources.audioVAERelativePath) })
        XCTAssertFalse(lines.contains { $0.contains("(missing)") })
    }

    func testLTX25FullComponentLinesShowEveryPackedParityAsset() throws {
        let root = try makeTempDirectory()
        for relativePath in LTX25Resources.fullRequiredRelativePaths {
            let url = root.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Data("fixture".utf8).write(to: url)
        }
        let manifest = MereRunModelManifest.template(
            for: .ltxVideo25FullBF16,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        let lines = ModelInfo.ltx25ComponentLines(rootURL: root, full: true)

        XCTAssertTrue(ModelInfo.usesLTX25Layout(manifest: manifest, expectedModelID: nil))
        XCTAssertTrue(lines.contains { $0.contains(LTX25Resources.devTransformerRelativePath) })
        XCTAssertTrue(lines.contains { $0.contains(LTX25Resources.diffusionVideoVAERelativePath) })
        XCTAssertTrue(lines.contains { $0.contains(LTX25Resources.temporalUpsamplerRelativePath) })
        XCTAssertTrue(lines.contains { $0.contains(LTX25Resources.distilledLoRARelativePath) })
        XCTAssertTrue(lines.contains { $0.contains(LTX25Resources.durationHeadRelativePath) })
        XCTAssertFalse(lines.contains { $0.contains("(missing)") })
    }

    func testExternalBindingUsesCatalogManifestWithoutClaimingLocalCreation() {
        let manifest = ModelInfo.catalogManifest(
            modelID: .ltxVideo25FullBF16,
            usageTermsAcknowledged: true
        )

        XCTAssertEqual(manifest.id, ModelResolver.ModelID.ltxVideo25FullBF16.rawValue)
        XCTAssertEqual(manifest.engine, .ltxVideo)
        XCTAssertEqual(manifest.upstreamRepoId, "\(LTX25Resources.sourceRepository)@\(LTX25Resources.sourceRevision)")
        XCTAssertEqual(manifest.usageTermsAcknowledged, true)
        XCTAssertNil(manifest.createdAt)
    }

    func testLTX23SplitLayoutDetectionUsesManifestID() {
        let manifest = MereRunModelManifest.template(
            for: .ltxVideo23AVMLX,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertTrue(ModelInfo.usesLTX23SplitLayout(manifest: manifest, expectedModelID: nil))
        XCTAssertFalse(ModelInfo.usesLTX23SplitLayout(manifest: nil, expectedModelID: "video-ltx-av"))
    }

    func testLTX23FullLayoutReportsDevLoRAAndVocoder() throws {
        let root = try makeTempDirectory()
        for file in [
            "split_model.json",
            "config.json",
            "embedded_config.json",
            "connector.safetensors",
            "transformer-dev.safetensors",
            "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "spatial_upscaler_x2_v1_1.safetensors",
            "spatial_upscaler_x2_v1_1_config.json",
            "vocoder.safetensors",
        ] {
            try Data("fixture".utf8).write(to: root.appendingPathComponent(file))
        }
        let manifest = MereRunModelManifest.template(
            for: .ltxVideo23FullMLX,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        let lines = ModelInfo.ltx23FullComponentLines(rootURL: root, companionRootURL: nil)

        XCTAssertTrue(ModelInfo.usesLTX23FullLayout(manifest: manifest, expectedModelID: nil))
        XCTAssertTrue(lines.contains("  layout: LTX 2.3 full split MLX files (unified AV + A2Vid)"))
        XCTAssertTrue(lines.contains { $0.contains("transformer-dev.safetensors") })
        XCTAssertTrue(lines.contains { $0.contains("distilled-lora-384-1.1.safetensors") })
        XCTAssertTrue(lines.contains { $0.contains("vocoder.safetensors") })
        XCTAssertFalse(lines.contains { $0.contains("(missing)") })
    }

    func testLTX23FullRequestedIDTakesPrecedenceOverCompatibleLegacyManifest() {
        let legacyManifest = MereRunModelManifest.template(
            for: .ltxVideo23A2VMLX,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertTrue(
            ModelInfo.usesLTX23FullLayout(
                manifest: legacyManifest,
                expectedModelID: ModelResolver.ModelID.ltxVideo23FullMLX.rawValue
            )
        )
        XCTAssertFalse(
            ModelInfo.usesLTX23A2VidLayout(
                manifest: legacyManifest,
                expectedModelID: ModelResolver.ModelID.ltxVideo23FullMLX.rawValue
            )
        )
    }

    func testLTXMergedComponentLinesShowMergedModelFiles() throws {
        let root = try makeTempDirectory()
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("tokenizer", isDirectory: true),
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("text_encoder", isDirectory: true),
            withIntermediateDirectories: true
        )
        try Data("fixture".utf8).write(to: root.appendingPathComponent("ltx-2-19b-distilled.safetensors"))
        try Data("fixture".utf8).write(to: root.appendingPathComponent("ltx-2-spatial-upscaler-x2-1.0.safetensors"))

        let lines = ModelInfo.ltxMergedComponentLines(rootURL: root)

        XCTAssertTrue(lines.contains("  layout: LTX distilled merged files"))
        XCTAssertTrue(lines.contains { $0.contains("tokenizer") })
        XCTAssertTrue(lines.contains { $0.contains("text_encoder") })
        XCTAssertTrue(lines.contains { $0.contains("ltx-2-19b-distilled.safetensors") })
        XCTAssertTrue(lines.contains { $0.contains("ltx-2-spatial-upscaler-x2-1.0.safetensors") })
        XCTAssertFalse(lines.contains { $0.contains("(unresolved)") })
        XCTAssertFalse(lines.contains { $0.contains("(missing)") })
    }

    private func makeTempDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelInfoCommandTests.\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        temporaryDirectories.append(directory)
        return directory
    }
}
