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

    func testMediaOnboardingModelsHaveDiskEstimates() throws {
        for id in ["speech-asr-parakeet", "text-embed-qwen3-0.6b"] {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))
            XCTAssertGreaterThanOrEqual(
                spec.estimatedDownloadBytes ?? 0,
                1_073_741_824,
                "Model \(id) should have enough of a size estimate for disk-space preflight."
            )
        }
    }

    func testImageModelsHaveManagedDownloadSources() {
        let expectedPullableImageIDs = [
            "image-klein-nano",
            "image-klein-base",
            "image-klein-max",
            "image-bonsai-binary",
            "image-bonsai-ternary",
            "image-zimage-nano",
            "image-zimage-base",
            "image-zimage-max",
        ]

        for id in expectedPullableImageIDs {
            let spec = ManagedModelCatalog.spec(for: id)
            XCTAssertNotNil(spec?.hubFallback, "Image model \(id) should be pullable from Hugging Face.")
        }
    }

    func testZImageNanoUsesMFluxHubSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-zimage-nano"))

        XCTAssertEqual(spec.hubFallback?.repoId, "filipstrand/Z-Image-Turbo-mflux-4bit")
        XCTAssertEqual(spec.upstreamRepoId, "filipstrand/Z-Image-Turbo-mflux-4bit")
        XCTAssertEqual(spec.upstreamRevision, "main")
        XCTAssertEqual(spec.hubFallback?.patterns.contains("tokenizer/added_tokens.json"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("transformer/model.safetensors.index.json"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("tokenizer/*"), false)
    }

    func testKleinNanoUsesExplicitHubSourceFiles() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-klein-nano"))

        XCTAssertEqual(spec.hubFallback?.repoId, "stereovoid/flux2-klein-4b-4bit")
        XCTAssertEqual(spec.hubFallback?.patterns.contains("tokenizer/added_tokens.json"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("transformer/diffusion_pytorch_model.safetensors"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("tokenizer/*"), false)
    }

    func testBonsaiTernaryUsesPrismMLHubSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-bonsai-ternary"))

        XCTAssertEqual(spec.hubFallback?.repoId, "prism-ml/bonsai-image-ternary-4B-mlx-2bit")
        XCTAssertEqual(spec.hubFallback?.revision, "main")
        XCTAssertEqual(spec.upstreamRepoId, "prism-ml/bonsai-image-ternary-4B-mlx-2bit")
        XCTAssertEqual(spec.upstreamRevision, "main")
        XCTAssertEqual(spec.validationKind, .bonsaiImage)
        XCTAssertEqual(spec.estimatedDownloadBytes, 3_888_274_558)
    }

    func testBonsaiBinaryUsesPrismMLHubSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-bonsai-binary"))

        XCTAssertEqual(spec.hubFallback?.repoId, "prism-ml/bonsai-image-binary-4B-mlx-1bit")
        XCTAssertEqual(spec.hubFallback?.revision, "main")
        XCTAssertEqual(spec.upstreamRepoId, "prism-ml/bonsai-image-binary-4B-mlx-1bit")
        XCTAssertEqual(spec.upstreamRevision, "main")
        XCTAssertEqual(spec.validationKind, .bonsaiImage)
        XCTAssertEqual(spec.estimatedDownloadBytes, 3_428_210_775)
    }

    func testGemma4TurboUsesNVFP4HubSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Gemma4Resources.turboModelId))

        XCTAssertEqual(spec.hubFallback?.repoId, "mlx-community/gemma-4-26b-a4b-it-nvfp4")
        XCTAssertEqual(spec.upstreamRepoId, "mlx-community/gemma-4-26b-a4b-it-nvfp4")
        XCTAssertEqual(spec.validationKind, .gemma4)
    }

    func testQ36NanoUsesOptiQHubSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.q36NanoModelId))

        XCTAssertEqual(spec.hubFallback?.repoId, "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit")
        XCTAssertEqual(spec.hubFallback?.revision, "63d520640ca7461f31ba66104612135770090340")
        XCTAssertEqual(spec.upstreamRepoId, "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit")
        XCTAssertEqual(spec.validationKind, .q35)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("*.safetensors"), true)
    }

    func testZImageNanoAcceptsMFluxLayoutWithoutDiffusersConfigs() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalMFluxZImageNano(at: root, upstreamRepoId: "filipstrand/Z-Image-Turbo-mflux-4bit@main")

        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-zimage-nano"))
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))
        XCTAssertTrue(ZImageTurboResources(rootURL: root).validate(fileManager: .default).isEmpty)

        let configs = try ZImageTurboModelConfigs.load(from: ZImageTurboResources(rootURL: root))
        XCTAssertEqual(configs.transformer.dim, 3840)
        XCTAssertEqual(configs.transformer.axesLens, [1024, 512, 512])
        XCTAssertEqual(configs.textEncoder.maxPositionEmbeddings, 40960)
        XCTAssertEqual(configs.vae.scalingFactor, 0.3611, accuracy: 0.0001)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "image-zimage-nano")
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertTrue(report.warnings.isEmpty)
    }

    func testBonsaiTernaryAcceptsPrismMLPackedLayout() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalBonsai(at: root, modelID: .bonsaiTernary, bits: 2, modelVersion: "ternary g128")

        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-bonsai-ternary"))
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "image-bonsai-ternary")
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertTrue(report.warnings.isEmpty)
        XCTAssertEqual(report.manifest?.precision, .int2)
        XCTAssertEqual(report.manifest?.quantization?.groupSize, 128)
    }

    func testBonsaiBinaryAcceptsPrismMLPackedLayout() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalBonsai(at: root, modelID: .bonsaiBinary, bits: 1, modelVersion: "binary g128")

        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-bonsai-binary"))
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "image-bonsai-binary")
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertTrue(report.warnings.isEmpty)
        XCTAssertEqual(report.manifest?.precision, .int1)
        XCTAssertEqual(report.manifest?.quantization?.bits, 1)
        XCTAssertEqual(report.manifest?.quantization?.groupSize, 128)
    }

    func testZImageNanoRejectsStaleManagedSource() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalMFluxZImageNano(at: root, upstreamRepoId: "andrevp/Z-Image-Turbo-MLX-4bit@main")

        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-zimage-nano"))
        XCTAssertFalse(spec.isManagedRootComplete(root, fileManager: .default))
        XCTAssertFalse(
            ZImageTurboResources.validateDownloadedRoot(root, modelID: .zetaNano, fileManager: .default).isEmpty
        )
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

    func testACEStepAcceptsUpstreamAndLegacyTurboLayouts() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "music-acestep"))

        let upstreamRoot = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: upstreamRoot) }
        try writeMinimalACEStepRoot(at: upstreamRoot, turboSubdirectory: "acestep-v15-turbo")
        XCTAssertTrue(spec.isManagedRootComplete(upstreamRoot, fileManager: .default))

        let legacyRoot = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: legacyRoot) }
        try writeMinimalACEStepRoot(at: legacyRoot, turboSubdirectory: "music-acestep-v15-turbo")
        XCTAssertTrue(spec.isManagedRootComplete(legacyRoot, fileManager: .default))
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-managed-model-catalog-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func writeMinimalACEStepRoot(at root: URL, turboSubdirectory: String) throws {
        for subdirectory in [turboSubdirectory, "vae", "Qwen3-Embedding-0.6B"] {
            try FileManager.default.createDirectory(
                at: root.appendingPathComponent(subdirectory, isDirectory: true),
                withIntermediateDirectories: true
            )
        }
    }

    private func writeMinimalMFluxZImageNano(at root: URL, upstreamRepoId: String) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var manifest = MereRunModelManifest.template(for: .zetaNano, createdAt: Date(timeIntervalSince1970: 0))
        manifest.upstreamRepoId = upstreamRepoId
        try manifest.write(to: root)

        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        let textEncoder = root.appendingPathComponent("text_encoder", isDirectory: true)
        let transformer = root.appendingPathComponent("transformer", isDirectory: true)
        let vae = root.appendingPathComponent("vae", isDirectory: true)

        for dir in [tokenizer, textEncoder, transformer, vae] {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }

        for file in ["tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"] {
            XCTAssertTrue(FileManager.default.createFile(atPath: tokenizer.appendingPathComponent(file).path, contents: Data()))
        }

        let indexData = Data(#"{"metadata":{"quantization_level":"4","mflux_version":"0.13.0.dev0"},"weight_map":{}}"#.utf8)
        try indexData.write(to: textEncoder.appendingPathComponent("model.safetensors.index.json"))
        try indexData.write(to: transformer.appendingPathComponent("model.safetensors.index.json"))
        try indexData.write(to: vae.appendingPathComponent("model.safetensors.index.json"))
    }

    private func writeMinimalBonsai(
        at root: URL,
        modelID: ModelResolver.ModelID,
        bits: Int,
        modelVersion: String
    ) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try MereRunModelManifest.template(for: modelID, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("manifest.json").path,
            contents: Data(#"{"model_version":"\#(modelVersion)"}"#.utf8)
        ))

        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        let textEncoder = root.appendingPathComponent("text_encoder-mlx-4bit", isDirectory: true)
        let transformer = root.appendingPathComponent("transformer-packed-mflux", isDirectory: true)
        let vae = root.appendingPathComponent("vae", isDirectory: true)
        let scheduler = root.appendingPathComponent("scheduler", isDirectory: true)

        for dir in [tokenizer, textEncoder, transformer, vae, scheduler] {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }

        XCTAssertTrue(FileManager.default.createFile(
            atPath: tokenizer.appendingPathComponent("tokenizer.json").path,
            contents: Data("{}".utf8)
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: tokenizer.appendingPathComponent("tokenizer_config.json").path,
            contents: Data("{}".utf8)
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: textEncoder.appendingPathComponent("config.json").path,
            contents: Data(#"{"quantization_config":{"bits":4,"group_size":64}}"#.utf8)
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: textEncoder.appendingPathComponent("model.safetensors.index.json").path,
            contents: Data(#"{"weight_map":{}}"#.utf8)
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: transformer.appendingPathComponent("config.json").path,
            contents: Data("{}".utf8)
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: transformer.appendingPathComponent("quantization_config.json").path,
            contents: Data(#"{"bits":\#(bits),"group_size":128}"#.utf8)
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: transformer.appendingPathComponent("diffusion_pytorch_model.safetensors").path,
            contents: Data()
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: vae.appendingPathComponent("config.json").path,
            contents: Data("{}".utf8)
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: vae.appendingPathComponent("diffusion_pytorch_model.safetensors").path,
            contents: Data()
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: scheduler.appendingPathComponent("scheduler_config.json").path,
            contents: Data("{}".utf8)
        ))
    }
}
