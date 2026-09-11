import Foundation
@testable import MereRunCore
import MereRunModelKit
import XCTest

final class MarigoldV2Tests: XCTestCase {

    // MARK: - Adapter mapping

    func testAdapterMapsDiffuserPathsOntoTransformerModules() {
        XCTAssertEqual(MarigoldV2LoRAAdapter.mappedTargetPath("Diffuser.img_in"), "x_embedder")
        XCTAssertEqual(MarigoldV2LoRAAdapter.mappedTargetPath("Diffuser.txt_in"), "context_embedder")
        XCTAssertEqual(
            MarigoldV2LoRAAdapter.mappedTargetPath("Diffuser.norm_out.linear"),
            "norm_out.linear"
        )
    }

    func testAdapterRewritesBlockLevelProjectionNames() {
        let cases = [
            ("Diffuser.transformer_blocks.0.attn.to_out.0", "transformer_blocks.0.attn.to_out"),
            ("Diffuser.transformer_blocks.7.img_mlp.net.0.proj", "transformer_blocks.7.ff.linear1"),
            ("Diffuser.transformer_blocks.7.img_mlp.net.2", "transformer_blocks.7.ff.linear2"),
            ("Diffuser.transformer_blocks.59.txt_mlp.net.0.proj", "transformer_blocks.59.ff_context.linear1"),
            ("Diffuser.transformer_blocks.59.txt_mlp.net.2", "transformer_blocks.59.ff_context.linear2"),
        ]
        for (source, expected) in cases {
            XCTAssertEqual(MarigoldV2LoRAAdapter.mappedTargetPath(source), expected, source)
        }
    }

    func testAdapterLeavesAttentionProjectionsUnchanged() {
        for name in ["to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"] {
            let source = "Diffuser.transformer_blocks.3.attn.\(name)"
            XCTAssertEqual(
                MarigoldV2LoRAAdapter.mappedTargetPath(source),
                "transformer_blocks.3.attn.\(name)"
            )
        }
    }

    func testAdapterPairCountCoversEveryPublishedTarget() {
        // 60 blocks x 12 adapted projections, plus img_in, txt_in, and norm_out.linear.
        XCTAssertEqual(MarigoldV2LoRAAdapter.expectedPairCount, 60 * 12 + 3)
        XCTAssertEqual(MarigoldV2LoRAAdapter.rank, 128)
        // Marigold trains with lora_alpha == rank, so the applied scale is exactly 1.
        XCTAssertEqual(MarigoldV2LoRAAdapter.alpha, Float(MarigoldV2LoRAAdapter.rank))
    }

    // MARK: - Inference sizing

    func testInferenceSizeAlignsToThePatchGrid() {
        let configuration = MarigoldV2InferenceConfiguration(maximumEdge: nil)
        let size = configuration.inferenceSize(width: 1_000, height: 750)
        XCTAssertEqual(size.width % MarigoldV2InferenceConfiguration.alignment, 0)
        XCTAssertEqual(size.height % MarigoldV2InferenceConfiguration.alignment, 0)
        XCTAssertEqual(size.width, 1_008)
        XCTAssertEqual(size.height, 752)
    }

    func testInferenceSizeCapsTheLongestEdgeAndKeepsAspect() {
        let configuration = MarigoldV2InferenceConfiguration(maximumEdge: 512)
        let size = configuration.inferenceSize(width: 4_000, height: 2_000)
        XCTAssertLessThanOrEqual(max(size.width, size.height), 512)
        XCTAssertEqual(size.width, 512)
        XCTAssertEqual(size.height, 256)
    }

    func testInferenceSizeLeavesSmallImagesAlone() {
        let configuration = MarigoldV2InferenceConfiguration(maximumEdge: 1_024)
        let size = configuration.inferenceSize(width: 320, height: 256)
        XCTAssertEqual(size.width, 320)
        XCTAssertEqual(size.height, 256)
    }

    // MARK: - Depth normalization

    func testNormalizationKeepsLogDepthIncreasingAwayFromTheCamera() {
        let raw: [Float] = [-1, -0.5, 0, 0.5, 1]
        let result = MarigoldV2DepthNormalizer.normalize(raw: raw, parameterization: .log)
        XCTAssertEqual(result.values.count, raw.count)
        for index in 1..<result.values.count {
            XCTAssertGreaterThan(result.values[index], result.values[index - 1])
        }
    }

    func testNormalizationFlipsDisparitySoLargerAlwaysMeansFarther() {
        let raw: [Float] = [-1, -0.5, 0, 0.5, 1]
        let result = MarigoldV2DepthNormalizer.normalize(raw: raw, parameterization: .disparity)
        for index in 1..<result.values.count {
            XCTAssertLessThan(result.values[index], result.values[index - 1])
        }
    }

    func testNormalizationStaysStrictlyPositiveSoPreviewsKeepEveryPixel() {
        let raw: [Float] = [-3, -1, 0, 1, 3]
        let result = MarigoldV2DepthNormalizer.normalize(raw: raw, parameterization: .log)
        for value in result.values {
            XCTAssertGreaterThan(value, 0)
            XCTAssertLessThanOrEqual(value, 1)
        }
        XCTAssertEqual(result.values.min(), MarigoldV2DepthNormalizer.normalizedFloor)
    }

    func testNormalizationRecordsTheMappingItApplied() {
        let raw: [Float] = [-2, -1, 0, 1, 2]
        let result = MarigoldV2DepthNormalizer.normalize(raw: raw, parameterization: .log)
        XCTAssertEqual(result.statistics.rawMinimum, -2)
        XCTAssertEqual(result.statistics.rawMaximum, 2)
        XCTAssertLessThan(result.statistics.normalizationNear, result.statistics.normalizationFar)
        XCTAssertEqual(result.statistics.normalizedFloor, MarigoldV2DepthNormalizer.normalizedFloor)
    }

    func testNormalizationHandlesAConstantPrediction() {
        let raw = [Float](repeating: 0.25, count: 16)
        let result = MarigoldV2DepthNormalizer.normalize(raw: raw, parameterization: .log)
        XCTAssertEqual(result.values.count, raw.count)
        for value in result.values {
            XCTAssertGreaterThan(value, 0)
            XCTAssertFalse(value.isNaN)
        }
    }

    // MARK: - Checkpoint metadata

    func testCheckpointMetadataMatchesThePublishedTable() {
        XCTAssertEqual(MarigoldV2Repository.installedCheckpoint, .logStage2)
        XCTAssertEqual(MarigoldV2DepthCheckpoint.logStage2.relativeDirectory, "depth/Log-stage2")
        XCTAssertEqual(MarigoldV2DepthCheckpoint.logStage2.parameterization, .log)
        XCTAssertFalse(MarigoldV2DepthCheckpoint.logStage2.isSeeThrough)
        XCTAssertTrue(MarigoldV2DepthCheckpoint.logStage2.shipsFineTunedVAEDecoder)

        XCTAssertTrue(MarigoldV2DepthCheckpoint.logLayered.isSeeThrough)
        XCTAssertEqual(MarigoldV2DepthCheckpoint.disparityBase.parameterization, .disparity)
        XCTAssertEqual(MarigoldV2DepthCheckpoint.uniformBase.parameterization, .linear)
        // Stage 1 and the uniform variants decode with the frozen base decoder.
        XCTAssertFalse(MarigoldV2DepthCheckpoint.logStage1.shipsFineTunedVAEDecoder)
        XCTAssertFalse(MarigoldV2DepthCheckpoint.uniformBase.shipsFineTunedVAEDecoder)
    }

    func testOnlyDisparityReadsLargerValuesAsCloser() {
        XCTAssertTrue(MarigoldV2DepthParameterization.disparity.increasesTowardCamera)
        XCTAssertFalse(MarigoldV2DepthParameterization.log.increasesTowardCamera)
        XCTAssertFalse(MarigoldV2DepthParameterization.linear.increasesTowardCamera)
    }

    // MARK: - Install layout

    func testResourcesResolveTheMountedAdapterPayload() throws {
        let root = try makeInstallTree(mounted: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let resources = MarigoldV2Resources(rootURL: root)
        XCTAssertEqual(
            resources.trainablesURL.path,
            root.appendingPathComponent("marigold/depth/Log-stage2/trainables.safetensors").path
        )
        XCTAssertTrue(resources.promptEmbedsURL.lastPathComponent.hasSuffix("_prompt_embeds.pt"))
        XCTAssertTrue(resources.promptMaskURL.lastPathComponent.hasSuffix("_prompt_mask.pt"))
        XCTAssertEqual(resources.validate(), [])
    }

    func testResourcesAlsoAcceptABareMarigoldRepositoryRoot() throws {
        let root = try makeInstallTree(mounted: false)
        defer { try? FileManager.default.removeItem(at: root) }

        let resources = MarigoldV2Resources(rootURL: root)
        XCTAssertEqual(
            resources.trainablesURL.path,
            root.appendingPathComponent("depth/Log-stage2/trainables.safetensors").path
        )
        XCTAssertEqual(resources.validate(), [])
    }

    func testValidationReportsAMissingAdapterPayload() throws {
        let root = try makeInstallTree(mounted: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.removeItem(
            at: root.appendingPathComponent("marigold/depth/Log-stage2/trainables.safetensors")
        )

        let missing = MarigoldV2Resources(rootURL: root).validate()
        XCTAssertEqual(missing.count, 1)
        XCTAssertEqual(missing.first?.lastPathComponent, "trainables.safetensors")
    }

    func testValidationDoesNotRequireTheTextEncoder() throws {
        let root = try makeInstallTree(mounted: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // The runtime conditions on precomputed embeddings, so the encoder is
        // neither downloaded nor required to consider the install complete.
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: root.appendingPathComponent("text_encoder").path)
        )
        XCTAssertEqual(MarigoldV2Resources(rootURL: root).validate(), [])
    }

    // MARK: - Catalog registration

    func testCatalogRegistersMarigoldAsAVisionDepthModel() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: MarigoldV2Repository.modelId))
        XCTAssertEqual(spec.category, .visionDepth)
        XCTAssertEqual(spec.validationKind, .marigoldV2)
        XCTAssertEqual(spec.installShape, .structuredRoot)
        XCTAssertEqual(spec.defaultCLICommands, ["vision depth"])
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
    }

    func testCatalogPullsTheFrozenBaseAndMountsTheAdapters() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: MarigoldV2Repository.modelId))
        XCTAssertEqual(spec.hubFallback?.repoId, "Qwen/Qwen-Image-Edit-2509")
        XCTAssertEqual(spec.mountedHubFallbacks.count, 1)
        let mounted = try XCTUnwrap(spec.mountedHubFallbacks.first)
        XCTAssertEqual(mounted.destinationPath, "marigold")
        XCTAssertEqual(mounted.hubFallback.repoId, "huawei-bayerlab/marigold-v2-0")
    }

    func testBaseSnapshotOmitsTheUnusedTextEncoder() {
        let patterns = MarigoldV2Repository.basePatterns
        XCTAssertTrue(patterns.contains("transformer/*"))
        XCTAssertTrue(patterns.contains("vae/*"))
        XCTAssertFalse(patterns.contains { $0.hasPrefix("text_encoder") })
        XCTAssertFalse(patterns.contains { $0.hasPrefix("tokenizer") })
    }

    func testArtifactPinsAddressTheMountedPayload() {
        for pin in MarigoldV2Repository.artifactPins {
            XCTAssertTrue(
                pin.filename.hasPrefix("\(MarigoldV2Repository.adapterMountPath)/"),
                pin.filename
            )
            XCTAssertGreaterThan(pin.byteCount, 0)
            XCTAssertEqual(pin.sha256.count, 64)
        }
    }

    func testManifestTemplateDescribesAffineRelativeDepth() {
        let manifest = MereRunModelManifest.template(
            for: .visionDepthMarigoldV2,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(manifest.id, MarigoldV2Repository.modelId)
        XCTAssertEqual(manifest.engine, .marigoldV2)
        XCTAssertEqual(manifest.family, .depth)
        XCTAssertEqual(manifest.supports, [.relativeDepth])
    }

    // MARK: - Helpers

    private func makeInstallTree(mounted: Bool) throws -> URL {
        let fileManager = FileManager.default
        let root = fileManager.temporaryDirectory
            .appendingPathComponent("marigold-\(UUID().uuidString)", isDirectory: true)

        let adapterRoot = mounted
            ? root.appendingPathComponent(MarigoldV2Repository.adapterMountPath, isDirectory: true)
            : root
        let checkpointDirectory = adapterRoot.appendingPathComponent(
            MarigoldV2Repository.installedCheckpoint.relativeDirectory,
            isDirectory: true
        )
        let embeddingsDirectory = adapterRoot.appendingPathComponent(
            MarigoldV2Repository.promptEmbeddingsDirectory,
            isDirectory: true
        )
        for directory in [
            root.appendingPathComponent("transformer", isDirectory: true),
            root.appendingPathComponent("vae", isDirectory: true),
            checkpointDirectory,
            embeddingsDirectory,
        ] {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        }

        let prefix = MarigoldV2Repository.depthPromptPrefix
        let files = [
            root.appendingPathComponent("model_index.json"),
            root.appendingPathComponent("transformer/config.json"),
            root.appendingPathComponent("transformer/diffusion_pytorch_model.safetensors.index.json"),
            root.appendingPathComponent("vae/config.json"),
            root.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"),
            checkpointDirectory.appendingPathComponent("trainables.safetensors"),
            embeddingsDirectory.appendingPathComponent("\(prefix)_prompt_embeds.pt"),
            embeddingsDirectory.appendingPathComponent("\(prefix)_prompt_mask.pt"),
        ]
        for file in files {
            try Data("fixture".utf8).write(to: file)
        }
        return root
    }
}
