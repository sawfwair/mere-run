import Foundation
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

    func testModelSourcesCatalogTableMatchesManagedModelCatalog() throws {
        let markdownURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("docs/model-sources.md")
        let markdown = try String(contentsOf: markdownURL, encoding: .utf8)
        let documentedRows = try managedModelCatalogRows(in: markdown)
        let catalogRows = ManagedModelCatalog.allSpecs.map {
            ManagedModelCatalogDocRow(category: $0.category.rawValue, id: $0.id)
        }

        XCTAssertEqual(documentedRows, catalogRows)
    }

    func testRemovedQ35ChatModelIDsAreNotCataloged() {
        XCTAssertNil(ManagedModelCatalog.spec(for: "text-chat-q35"))
        XCTAssertNil(ManagedModelCatalog.spec(for: "text-chat-q35-nano"))
        XCTAssertNil(ManagedModelCatalog.spec(for: "text-chat-q35-nano-gguf"))
        XCTAssertNil(ModelResolver.ModelID(rawValue: "text-chat-q35"))
        XCTAssertNil(ModelResolver.ModelID(rawValue: "text-chat-q35-nano"))
    }

    func testAPIServableModelsOwnCompleteCatalogProfiles() {
        let expectedChatModels = ManagedModelCatalog.allSpecs.filter {
            $0.defaultCLICommands.contains("api serve")
                || $0.id == CodeGenResources.defaultModelId
        }

        XCTAssertFalse(expectedChatModels.isEmpty)
        for spec in expectedChatModels {
            guard let profile = spec.apiProfile else {
                XCTFail("API-served model \(spec.id) is missing a catalog API profile.")
                continue
            }
            XCTAssertEqual(profile.task, .chatCompletions, spec.id)
            XCTAssertNotNil(profile.servingEngine, spec.id)
            XCTAssertGreaterThan(profile.contextWindow ?? 0, 0, spec.id)
            XCTAssertGreaterThan(profile.maximumOutputTokens ?? 0, 0, spec.id)
        }
    }

    func testCatalogProfilesOwnHarnessSpecificCapabilities() throws {
        let ornith = try XCTUnwrap(
            ManagedModelCatalog.apiProfile(for: Q35Resources.ornith9BModelId)
        )
        XCTAssertEqual(ornith.contextWindow, Q35Resources.defaultContextLength)
        XCTAssertEqual(ornith.maximumOutputTokens, 4_096)
        XCTAssertEqual(ornith.thinkingLevels, [.high])
        XCTAssertTrue(ornith.toolCall)
        XCTAssertTrue(ornith.inputModalities.contains(.image))

        let deepseek = try XCTUnwrap(
            ManagedModelCatalog.apiProfile(for: DeepseekV4FlashResources.defaultModelId)
        )
        XCTAssertEqual(deepseek.compatibility.maxTokensField, .maxTokens)
        XCTAssertEqual(deepseek.compatibility.thinkingFormat, .deepseek)
        XCTAssertEqual(deepseek.thinkingLevelMap, [.minimal: .low])
        XCTAssertFalse(deepseek.compatibility.supportsDeveloperRole)
        XCTAssertTrue(deepseek.compatibility.supportsReasoningEffort)
        XCTAssertTrue(deepseek.toolCall)

        let muse = try XCTUnwrap(
            ManagedModelCatalog.apiProfile(for: MuseGlimmerResources.modelId)
        )
        XCTAssertEqual(muse.reasoningEffortStrengths[.minimal], 0.1)
        XCTAssertEqual(muse.reasoningEffortStrengths[.high], 0.8)
        XCTAssertEqual(muse.reasoningEffortStrengths[.max], 1)
    }

    func testQ35OCRModelsAreNotChatRuntimeModels() throws {
        let full = try XCTUnwrap(
            ManagedModelCatalog.spec(for: Q35Resources.infinityParser2ProModelId)
        )
        let quantized = try XCTUnwrap(
            ManagedModelCatalog.spec(for: Q35Resources.infinityParser2ProInt8ModelId)
        )

        XCTAssertNil(full.apiProfile)
        XCTAssertNil(full.defaultRuntimeServingEngine)
        XCTAssertNil(quantized.apiProfile)
        XCTAssertNil(quantized.defaultRuntimeServingEngine)
    }

    func testCompanionAPIProfilesComeFromCatalog() throws {
        let embedding = try XCTUnwrap(
            ManagedModelCatalog.apiProfile(for: ModelResolver.ModelID.qwen3Embedding.rawValue)
        )
        XCTAssertEqual(embedding.task, .embeddings)
        XCTAssertEqual(embedding.inputModalities, [.text])
        XCTAssertEqual(embedding.outputModalities, [.embedding])

        let imageEdit = try XCTUnwrap(
            ManagedModelCatalog.apiProfile(for: QwenImageEditRepository.modelId)
        )
        XCTAssertEqual(imageEdit.task, .imageEdits)
        XCTAssertEqual(imageEdit.inputModalities, [.text, .image])
        XCTAssertEqual(imageEdit.outputModalities, [.image])
    }

    func testAllRuntimeAutoDownloadSpecsHaveManagedSource() {
        for spec in ManagedModelCatalog.allSpecs where spec.runtimeAutoDownloadAllowed {
            XCTAssertTrue(
                spec.hubFallback != nil,
                "Runtime auto-download model \(spec.id) has no configured Hugging Face source."
            )
        }
    }

    func testRestrictedModelInventoryIsCompleteAndCannotAutoDownload() throws {
        let expected = Set([
            "image-klein-9b",
            "image-klein-base-9b",
            "image-krea2-raw",
            "image-krea2-turbo",
            "image-ideogram4-sdnq-uint4",
            "text-chat-lfm25-a1b-8bit",
            "text-chat-lfm25-a1b-bf16",
            "text-chat-lfm25-1.2b-bf16",
            "text-chat-lfm25-1.2b-qad-4bit",
            "text-chat-lfm25-2.6b-4bit",
            "text-chat-lfm25-2.6b-bf16",
            "text-chat-lfm25-2.6b-qad-4bit",
            "text-chat-lfm25-a1b-dspark",
            "text-chat-lfm25-1.2b-dspark",
            "text-chat-lfm25-2.6b-dspark",
            "vision-chat-lfm25-3b-8bit",
            Q35Resources.q38FlashNextMixedModelId,
            Q35Resources.q38FlashNext4BitModelId,
            "vision-segment-sam31",
            "vision-face-buffalo-l",
            "vision-embed-olmoearth-v12-nano",
            "vision-embed-olmoearth-v12-tiny",
            "vision-embed-olmoearth-v12-small",
            "vision-embed-olmoearth-v12-base",
            MuseGlimmerResources.modelId,
            NemotronOmniResources.modelID,
            "image-3d-trellis2-4b",
            MiniMaxMusic3Resources.modelID,
            "music-muscriptor-small",
            "music-muscriptor-medium",
            "music-muscriptor-large",
            "sfx-woosh-dflow",
            "sfx-woosh-flow",
            "sfx-woosh-clap",
            "sfx-woosh-synchformer",
            "sfx-woosh-vflow-8s",
            "sfx-woosh-dvflow-8s",
            "sfx-mmaudio-large-44k-v2",
            "video-ltx-av",
            "video-ltx23-av-mlx",
            "video-ltx23-full-mlx",
            "video-ltx23-a2vid-mlx",
            "video-ltx25-distilled-bf16",
            "video-ltx25-full-bf16",
            "video-minimax-h3-fl2va-mlx",
            "video-minimax-h3-fl2va-bf16-mlx",
            "video-minimax-h3-fl2va-8bit-mlx",
            "video-minimax-h3-ref2va-mlx",
        ])
        let visibleAndCompanionSpecs = ManagedModelCatalog.allSpecs
            + ManagedModelCatalog.allSpecs
                .flatMap(\.companionModelIDs)
                .compactMap { ManagedModelCatalog.spec(for: $0) }
        let restricted = Set(
            visibleAndCompanionSpecs
                .filter { $0.usageRestriction != nil }
                .map(\.id)
        )
        XCTAssertEqual(restricted, expected)
        for spec in visibleAndCompanionSpecs where spec.usageRestriction != nil {
            XCTAssertFalse(spec.runtimeAutoDownloadAllowed, "Restricted model \(spec.id) must never auto-download.")
            let terms = try XCTUnwrap(spec.usageRestriction?.terms)
            XCTAssertFalse(terms.isEmpty)
            for term in terms {
                XCTAssertFalse(term.component.isEmpty)
                XCTAssertFalse(term.license.isEmpty)
                XCTAssertFalse(term.sourceRepoId.isEmpty)
                XCTAssertTrue(
                    term.sourceRevision.range(of: "^[0-9a-f]{40}$", options: .regularExpression) != nil,
                    "Restricted source \(term.sourceRepoId) must use an immutable revision, not \(term.sourceRevision)."
                )
                XCTAssertNotNil(URL(string: term.licenseURL))
            }

            let downloadSources = [spec.hubFallback].compactMap { $0 }
                + spec.mountedHubFallbacks.map(\.hubFallback)
                + spec.companionModelIDs.compactMap { ManagedModelCatalog.spec(for: $0)?.hubFallback }
            for source in downloadSources {
                XCTAssertTrue(
                    source.revision.range(of: "^[0-9a-f]{40}$", options: .regularExpression) != nil,
                    "Restricted download source \(source.repoId) must use an immutable revision, not \(source.revision)."
                )
            }
        }
    }

    func testRestrictedManifestCarriesSourceTermsWithoutClaimingAcceptance() throws {
        let modelID = try XCTUnwrap(ModelResolver.ModelID(rawValue: FaceAnalysisResources.modelID))
        let manifest = MereRunModelManifest.template(for: modelID, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.schemaVersion, 3)
        XCTAssertEqual(manifest.sources?.first?.repository, "deepghs/insightface")
        XCTAssertEqual(
            manifest.sources?.first?.revision,
            "4e1f33d3fe0e50a0945f3a53ab94ae8977ae7ddb"
        )
        XCTAssertEqual(manifest.usageTerms?.first?.license, "InsightFace pretrained model non-commercial research terms")
        XCTAssertNil(manifest.usageTermsAcknowledged)
    }

    func testManagedDownloadSourcesAreHuggingFaceOnly() {
        for spec in ManagedModelCatalog.allSpecs where spec.hasAnyManagedDownloadSource() {
            XCTAssertNotNil(spec.hubFallback, "Managed model \(spec.id) should download from Hugging Face.")
        }
    }

    func testPullableCatalogSpecsHaveDiskEstimates() {
        var specs = ManagedModelCatalog.allSpecs
        let companionSpecs = ManagedModelCatalog.allSpecs
            .flatMap(\.companionModelIDs)
            .compactMap { ManagedModelCatalog.spec(for: $0) }
        specs.append(contentsOf: companionSpecs)

        for spec in specs where spec.hasAnyManagedDownloadSource() {
            XCTAssertGreaterThan(
                spec.estimatedDownloadBytes ?? 0,
                0,
                "Pullable model \(spec.id) should have a byte estimate for disk-space preflight."
            )
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
            "image-klein-base-9b",
            "image-klein-max",
            "image-bonsai-binary",
            "image-bonsai-ternary",
            "image-zimage-nano",
            "image-zimage-base",
            "image-zimage-max",
            "image-krea2-raw",
            "image-krea2-turbo",
            "image-ideogram4-sdnq-uint4",
        ]

        for id in expectedPullableImageIDs {
            let spec = ManagedModelCatalog.spec(for: id)
            XCTAssertNotNil(spec?.hubFallback, "Image model \(id) should be pullable from Hugging Face.")
        }
    }

    func testZImageNanoUsesMFluxHubSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-zimage-nano"))

        XCTAssertEqual(spec.hubFallback?.repoId, "filipstrand/Z-Image-Turbo-mflux-4bit")
        XCTAssertEqual(spec.hubFallback?.revision, "b3a8f31115a11f2f9e2fa0bfbc8d78dcc3e6568b")
        XCTAssertEqual(spec.upstreamRepoId, "filipstrand/Z-Image-Turbo-mflux-4bit")
        XCTAssertEqual(spec.upstreamRevision, "b3a8f31115a11f2f9e2fa0bfbc8d78dcc3e6568b")
        XCTAssertEqual(spec.hubFallback?.patterns.contains("tokenizer/added_tokens.json"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("transformer/model.safetensors.index.json"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("tokenizer/*"), false)
        XCTAssertNil(spec.usageRestriction)
    }

    func testGeometryAnd3DModelsUsePinnedAuthoritativeSources() throws {
        let expected: [String: (repo: String, revision: String, files: Set<String>, bytes: Int64)] = [
            ModelResolver.ModelID.visionGeometryMoGe2Small.rawValue: (
                "Ruicheng/moge-2-vits-normal-onnx",
                "e50ffda41565591092adea54c6ac83d6212e1e23",
                ["model.onnx"],
                140_852_051
            ),
            ModelResolver.ModelID.visionDepthVDASmall.rawValue: (
                "depth-anything/Video-Depth-Anything-Small",
                "256875362cff76724b920335dfb4b29dd611f66e",
                ["video_depth_anything_vits.pth"],
                116_440_756
            ),
            ModelResolver.ModelID.visionDepthVDASmallMetric.rawValue: (
                "depth-anything/Metric-Video-Depth-Anything-Small",
                "273d090f2ce17df50c2872d82c8322c45da5b4dd",
                ["metric_video_depth_anything_vits.pth"],
                116_444_063
            ),
            ModelResolver.ModelID.visionGeometryDA3Small.rawValue: (
                "depth-anything/DA3-SMALL",
                "e08cab65ca0ec38e7826075418411ab90cab4da3",
                ["config.json", "model.safetensors"],
                137_248_940
            ),
            ModelResolver.ModelID.image3DTripoSR.rawValue: (
                "stabilityai/TripoSR",
                "5b521936b01fbe1890f6f9baed0254ab6351c04a",
                ["config.yaml", "model.ckpt"],
                1_677_247_729
            ),
            ModelResolver.ModelID.image3DInstantMeshBase.rawValue: (
                "TencentARC/InstantMesh",
                "b785b4ecfb6636ef34a08c748f96f6a5686244d0",
                ["instant_mesh_base.ckpt"],
                1_253_574_354
            ),
        ]

        for (id, pin) in expected {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))
            XCTAssertEqual(spec.hubFallback?.repoId, pin.repo)
            XCTAssertEqual(spec.hubFallback?.revision, pin.revision)
            let patterns = Set(spec.hubFallback?.patterns ?? [])
            XCTAssertTrue(pin.files.isSubset(of: patterns))
            XCTAssertTrue(patterns.contains("LICENSE*"), "\(id) should retain upstream license files")
            XCTAssertTrue(patterns.contains("NOTICE*"), "\(id) should retain upstream notice files")
            XCTAssertEqual(spec.upstreamRevision, pin.revision)
            XCTAssertEqual(spec.estimatedDownloadBytes, pin.bytes)
        }
    }

    func testGeometryAnd3DManagedValidationRejectsOneByteFalsePositives() throws {
        for pin in GeometryModelPins.all {
            let root = try makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: pin.modelID))
            for artifact in pin.artifacts {
                try Data([0]).write(to: root.appendingPathComponent(artifact.filename))
            }

            XCTAssertFalse(spec.isManagedRootComplete(root), pin.modelID)
            XCTAssertFalse(spec.isManagedRuntimeReady(root), pin.modelID)
            XCTAssertFalse(spec.missingPaths(in: root).isEmpty, pin.modelID)
            XCTAssertTrue(
                spec.validationMessages(in: root).contains { $0.contains("wrong size") },
                pin.modelID
            )
        }
    }

    func testInstantMeshManagedValidationDistinguishesRawDownloadFromNativeRuntime() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.image3DInstantMeshBase.rawValue)
        )
        let sourceRoot = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: sourceRoot) }
        XCTAssertEqual(
            spec.missingPaths(in: sourceRoot).map(\.lastPathComponent),
            ["instant_mesh_base.ckpt"]
        )
        try Data([0]).write(to: sourceRoot.appendingPathComponent("instant_mesh_base.ckpt"))
        XCTAssertFalse(spec.missingPaths(in: sourceRoot).isEmpty)
        XCTAssertFalse(spec.isManagedRootComplete(sourceRoot))
        XCTAssertFalse(spec.isManagedRuntimeReady(sourceRoot))
        XCTAssertTrue(spec.requiresManagedConversion)
        XCTAssertTrue(spec.managedConversionGuidance(at: sourceRoot)?.contains("--output") == true)

        let convertedRoot = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: convertedRoot) }
        let native = convertedRoot.appendingPathComponent(
            InstantMeshResources.managedConvertedDirectoryName,
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: native, withIntermediateDirectories: true)
        for filename in ["model.safetensors", "config.json", "SOURCE.json", "LICENSE"] {
            try Data([0]).write(to: native.appendingPathComponent(filename))
        }
        XCTAssertFalse(spec.missingPaths(in: convertedRoot).isEmpty)
        XCTAssertFalse(spec.isManagedRootComplete(convertedRoot))
        XCTAssertFalse(spec.isManagedRuntimeReady(convertedRoot))
    }

    func testInstantMeshPinnedRawFixtureIsDownloadedButConversionRequiredWhenAvailable() throws {
        let fixturePath = ProcessInfo.processInfo.environment["MERERUN_TEST_INSTANTMESH_SOURCE"] ?? ""
        try XCTSkipIf(fixturePath.isEmpty || !FileManager.default.fileExists(atPath: fixturePath))
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.image3DInstantMeshBase.rawValue)
        )
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("instant_mesh_base.ckpt"),
            withDestinationURL: URL(fileURLWithPath: fixturePath)
        )

        XCTAssertTrue(spec.isManagedRootComplete(root))
        XCTAssertFalse(spec.isManagedRuntimeReady(root))
        XCTAssertTrue(spec.managedConversionGuidance(at: root)?.contains("conversion") == true)
    }

    func testPinnedManagedValidationRejectsCorrectSizeWithWrongChecksum() throws {
        let pin = GeometryModelPins.moge2Small
        let artifact = try XCTUnwrap(pin.artifacts.first)
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: pin.modelID))
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent(artifact.filename)
        XCTAssertTrue(FileManager.default.createFile(atPath: file.path, contents: nil))
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: UInt64(artifact.byteCount))
        try handle.close()

        XCTAssertFalse(spec.isManagedRootComplete(root))
        XCTAssertFalse(spec.isManagedRuntimeReady(root))
        XCTAssertTrue(spec.validationMessages(in: root).contains { $0.contains("checksum mismatch") })
    }

    func testKleinNanoUsesExplicitHubSourceFiles() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-klein-nano"))

        XCTAssertEqual(spec.hubFallback?.repoId, "stereovoid/flux2-klein-4b-4bit")
        XCTAssertEqual(spec.hubFallback?.patterns.contains("tokenizer/added_tokens.json"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("transformer/diffusion_pytorch_model.safetensors"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("tokenizer/*"), false)
    }

    func testKleinMaxRejectsConfigOnlyPartialInstall() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-klein-max"))
        try writeConfigOnlyDiffusersImageRoot(at: root, id: .kleinMax)

        XCTAssertFalse(spec.isManagedRootComplete(root, fileManager: .default))
        let missing = spec.missingPaths(in: root, fileManager: .default).map(\.path)
        XCTAssertTrue(missing.contains { $0.hasSuffix("text_encoder/model.safetensors.index.json") })
        XCTAssertTrue(missing.contains { $0.hasSuffix("transformer/diffusion_pytorch_model.safetensors.index.json") })
        XCTAssertTrue(missing.contains { $0.hasSuffix("vae/diffusion_pytorch_model.safetensors") })
    }

    func testKleinBase9BUsesGatedBaseTransformerAndMountedSharedComponents() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-klein-base-9b"))

        XCTAssertEqual(spec.hubFallback?.repoId, "black-forest-labs/FLUX.2-klein-base-9B")
        XCTAssertEqual(spec.hubFallback?.patterns, ["model_index.json", "transformer/*"])
        XCTAssertEqual(spec.upstreamRepoId, "black-forest-labs/FLUX.2-klein-base-9B")
        XCTAssertEqual(spec.defaultCLICommands, ["image generate", "image train-lora"])
        XCTAssertEqual(spec.usageRestriction?.terms.count, 2)
        XCTAssertTrue(spec.usageRestriction?.terms.contains { $0.component == "Base 9B transformer" } == true)
        XCTAssertTrue(spec.usageRestriction?.terms.contains { $0.component.hasPrefix("shared 9B") } == true)

        let mounted = Dictionary(uniqueKeysWithValues: spec.mountedHubFallbacks.map {
            ($0.destinationPath, $0.hubFallback.repoId)
        })
        XCTAssertEqual(mounted["text_encoder"], "mlx-community/FLUX.2-klein-9B")
        XCTAssertEqual(mounted["tokenizer"], "mlx-community/FLUX.2-klein-9B")
        XCTAssertEqual(mounted["vae"], "mlx-community/FLUX.2-klein-9B")
        XCTAssertEqual(mounted["scheduler"], "mlx-community/FLUX.2-klein-9B")
    }

    func testSAM31ManagedInstallMountsTokenizerForTextPrompts() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "vision-segment-sam31"))

        XCTAssertEqual(spec.hubFallback?.repoId, "mlx-community/sam3.1-bf16")
        XCTAssertEqual(spec.hubFallback?.patterns.contains("model.safetensors"), true)

        let mounted = Dictionary(uniqueKeysWithValues: spec.mountedHubFallbacks.map {
            ($0.destinationPath, $0.hubFallback)
        })
        let tokenizer = try XCTUnwrap(mounted["tokenizer"])
        XCTAssertEqual(tokenizer.repoId, "AEmotionStudio/sam3.1")
        XCTAssertEqual(tokenizer.patterns.contains("tokenizer.json"), true)
        XCTAssertEqual(tokenizer.patterns.contains("tokenizer_config.json"), true)
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

    func testIdeogram4UsesWaveCutSDNQHubSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Ideogram4Resources.modelId))

        XCTAssertEqual(spec.hubFallback?.repoId, "WaveCut/ideogram-4-sdnq-uint4")
        XCTAssertEqual(spec.hubFallback?.revision, "ea2e67436478a97ad6c414c5f947d6d76aa8457d")
        XCTAssertEqual(spec.upstreamRepoId, "WaveCut/ideogram-4-sdnq-uint4")
        XCTAssertEqual(spec.upstreamRevision, "ea2e67436478a97ad6c414c5f947d6d76aa8457d")
        XCTAssertEqual(spec.validationKind, .ideogram4SDNQ)
        XCTAssertEqual(spec.estimatedDownloadBytes, 16 * 1_073_741_824)
        XCTAssertEqual(spec.runtimeAutoDownloadAllowed, false)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("unconditional_transformer/diffusion_pytorch_model.safetensors"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("quantization_manifest.json"), true)
    }

    func testKrea2TurboUsesComponentHubSourceWithoutRootTurboCheckpoint() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Krea2Resources.modelId))
        let patterns = try XCTUnwrap(spec.hubFallback?.patterns)

        XCTAssertEqual(spec.hubFallback?.repoId, Krea2Resources.upstreamRepoId)
        XCTAssertEqual(spec.hubFallback?.revision, Krea2Resources.upstreamRevision)
        XCTAssertEqual(spec.upstreamRepoId, Krea2Resources.upstreamRepoId)
        XCTAssertEqual(spec.upstreamRevision, Krea2Resources.upstreamRevision)
        XCTAssertEqual(spec.validationKind, .krea2)
        XCTAssertEqual(spec.estimatedDownloadBytes, Krea2Resources.estimatedDownloadBytes)
        XCTAssertEqual(spec.runtimeAutoDownloadAllowed, false)
        XCTAssertTrue(patterns.contains("transformer/diffusion_pytorch_model.safetensors.index.json"))
        XCTAssertTrue(patterns.contains("transformer/diffusion_pytorch_model-*.safetensors"))
        XCTAssertTrue(patterns.contains("text_encoder/model.safetensors"))
        XCTAssertFalse(patterns.contains("turbo.safetensors"))
        XCTAssertFalse(patterns.contains("*.safetensors"))
        XCTAssertFalse(patterns.contains("transformer/*"))
    }

    func testKrea2RawUsesComponentHubSourceWithoutRootRawCheckpoint() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Krea2RawResources.modelId))
        let patterns = try XCTUnwrap(spec.hubFallback?.patterns)

        XCTAssertEqual(spec.hubFallback?.repoId, Krea2RawResources.upstreamRepoId)
        XCTAssertEqual(spec.hubFallback?.revision, Krea2RawResources.upstreamRevision)
        XCTAssertEqual(spec.upstreamRepoId, Krea2RawResources.upstreamRepoId)
        XCTAssertEqual(spec.upstreamRevision, Krea2RawResources.upstreamRevision)
        XCTAssertEqual(spec.validationKind, .krea2)
        XCTAssertEqual(spec.estimatedDownloadBytes, Krea2RawResources.estimatedDownloadBytes)
        XCTAssertEqual(spec.runtimeAutoDownloadAllowed, false)
        XCTAssertEqual(spec.defaultCLICommands, ["image train-lora"])
        XCTAssertTrue(patterns.contains("transformer/diffusion_pytorch_model.safetensors.index.json"))
        XCTAssertTrue(patterns.contains("transformer/diffusion_pytorch_model-*.safetensors"))
        XCTAssertTrue(patterns.contains("text_encoder/model.safetensors"))
        XCTAssertFalse(patterns.contains("raw.safetensors"))
        XCTAssertFalse(patterns.contains("*.safetensors"))
        XCTAssertFalse(patterns.contains("transformer/*"))
    }

    func testKrea2TurboAcceptsSymlinkedComponentInstallRoot() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Krea2Resources.modelId))
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let snapshot = temp.appendingPathComponent("snapshot", isDirectory: true)
        try writeMinimalKrea2Root(at: snapshot)

        let root = temp.appendingPathComponent(Krea2Resources.modelId, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try MereRunModelManifest.template(for: .krea2Turbo, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("model_index.json").path,
            contents: Data("{}".utf8)
        ))
        for component in ["tokenizer", "text_encoder", "transformer", "vae", "scheduler"] {
            try FileManager.default.createSymbolicLink(
                at: root.appendingPathComponent(component, isDirectory: true),
                withDestinationURL: snapshot.appendingPathComponent(component, isDirectory: true)
            )
        }

        XCTAssertTrue(spec.missingPaths(in: root, fileManager: .default).isEmpty)
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))
    }

    func testGemma4TurboUsesNVFP4HubSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Gemma4Resources.turboModelId))

        XCTAssertEqual(spec.hubFallback?.repoId, "mlx-community/gemma-4-26b-a4b-it-nvfp4")
        XCTAssertEqual(spec.upstreamRepoId, "mlx-community/gemma-4-26b-a4b-it-nvfp4")
        XCTAssertEqual(spec.validationKind, .gemma4)
    }

    func testGemma4TwelveBUsesGoogleUnifiedHubSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Gemma4Resources.twelveBModelId))

        XCTAssertEqual(spec.category, .textChat)
        XCTAssertEqual(spec.hubFallback?.repoId, Gemma4Resources.twelveBUpstreamModelId)
        XCTAssertEqual(spec.upstreamRepoId, Gemma4Resources.twelveBUpstreamModelId)
        XCTAssertEqual(spec.validationKind, .gemma4)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatGemma4)
        XCTAssertEqual(spec.companionModelIDs, [Gemma4MTPResources.modelId])
    }

    func testGemma4TwelveB4BitUsesMLXQuantizedHubSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Gemma4Resources.twelveB4BitModelId))

        XCTAssertEqual(spec.category, .textChat)
        XCTAssertEqual(spec.hubFallback?.repoId, Gemma4Resources.twelveB4BitUpstreamModelId)
        XCTAssertEqual(spec.hubFallback?.revision, Gemma4Resources.twelveB4BitUpstreamRevision)
        XCTAssertEqual(spec.upstreamRepoId, Gemma4Resources.twelveB4BitUpstreamModelId)
        XCTAssertEqual(spec.upstreamRevision, Gemma4Resources.twelveB4BitUpstreamRevision)
        XCTAssertEqual(spec.estimatedDownloadBytes, 6_773_374_762)
        XCTAssertTrue(spec.hubFallback?.patterns.contains("MERERUN_CONVERSION.json") == true)
        XCTAssertEqual(spec.validationKind, .gemma4)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatGemma4)
        XCTAssertEqual(spec.companionModelIDs, [Gemma4MTPResources.modelId])
    }

    func testGemma4TwelveBVisionRequiresUnifiedProcessorFiles() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Gemma4Resources.visionTwelveBModelId))

        XCTAssertEqual(spec.category, .visionChat)
        XCTAssertEqual(spec.hubFallback?.repoId, Gemma4Resources.twelveBUpstreamModelId)
        XCTAssertEqual(spec.upstreamRepoId, Gemma4Resources.twelveBUpstreamModelId)
        XCTAssertEqual(spec.validationKind, .gemma4Unified)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatGemma4)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("processor_config.json"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("preprocessor_config.json"), true)
        XCTAssertEqual(spec.companionModelIDs, [Gemma4MTPResources.modelId])
    }

    func testGemma4TwelveBMTPAssistantIsCompanionOnly() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Gemma4MTPResources.modelId))

        XCTAssertFalse(ManagedModelCatalog.allSpecs.contains { $0.id == Gemma4MTPResources.modelId })
        XCTAssertEqual(spec.hubFallback?.repoId, Gemma4MTPResources.upstreamModelId)
        XCTAssertEqual(spec.validationKind, .gemma4MTPAssistant)
        XCTAssertEqual(spec.runtimeAutoDownloadAllowed, false)
    }

    func testNativeTextSFTModelsAdvertiseTrainingCommand() throws {
        let gemma = try XCTUnwrap(
            ManagedModelCatalog.spec(for: Gemma4Resources.twelveB4BitModelId)
        )
        let lagunaXS = try XCTUnwrap(
            ManagedModelCatalog.spec(for: LagunaResources.xsModelID)
        )
        let lagunaS = try XCTUnwrap(
            ManagedModelCatalog.spec(for: LagunaResources.modelID)
        )
        let inkling = try XCTUnwrap(
            ManagedModelCatalog.spec(for: InklingResources.modelID)
        )

        XCTAssertTrue(gemma.defaultCLICommands.contains("text train-lora"))
        XCTAssertTrue(lagunaXS.defaultCLICommands.contains("text train-lora"))
        XCTAssertFalse(lagunaS.defaultCLICommands.contains("text train-lora"))
        XCTAssertTrue(inkling.defaultCLICommands.contains("text train-lora"))
    }

    func testLagunaUsesPinnedOfficialTargetAndCompanion() throws {
        let target = try XCTUnwrap(ManagedModelCatalog.spec(for: LagunaResources.modelID))
        let companion = try XCTUnwrap(ManagedModelCatalog.spec(for: LagunaResources.dflashModelID))

        XCTAssertEqual(target.category, .textChat)
        XCTAssertEqual(target.hubFallback?.repoId, LagunaResources.upstreamModelID)
        XCTAssertEqual(target.hubFallback?.revision, LagunaResources.upstreamRevision)
        XCTAssertEqual(target.hubFallback?.patterns, LagunaResources.snapshotPatterns)
        XCTAssertEqual(target.upstreamRepoId, LagunaResources.upstreamModelID)
        XCTAssertEqual(target.upstreamRevision, LagunaResources.upstreamRevision)
        XCTAssertEqual(target.validationKind, .laguna)
        XCTAssertEqual(target.defaultRuntimeServingEngine, .textChatLaguna)
        XCTAssertEqual(target.estimatedDownloadBytes, LagunaResources.estimatedDownloadBytes)
        XCTAssertEqual(target.companionModelIDs, [LagunaResources.dflashModelID])
        XCTAssertFalse(target.runtimeAutoDownloadAllowed)
        XCTAssertNil(target.usageRestriction)
        XCTAssertTrue(LagunaResources.snapshotPatterns.contains("LICENSE*"))

        XCTAssertFalse(ManagedModelCatalog.allSpecs.contains { $0.id == companion.id })
        XCTAssertEqual(companion.hubFallback?.repoId, LagunaResources.dflashUpstreamModelID)
        XCTAssertEqual(companion.hubFallback?.revision, LagunaResources.dflashUpstreamRevision)
        XCTAssertEqual(companion.hubFallback?.patterns, LagunaResources.dflashSnapshotPatterns)
        XCTAssertEqual(companion.validationKind, .lagunaDFlash)
        XCTAssertEqual(companion.estimatedDownloadBytes, LagunaResources.dflashEstimatedDownloadBytes)
        XCTAssertFalse(companion.runtimeAutoDownloadAllowed)
    }

    func testLagunaXSUsesPinnedReleasedTargetWithoutSCompanion() throws {
        let target = try XCTUnwrap(ManagedModelCatalog.spec(for: LagunaResources.xsModelID))

        XCTAssertEqual(target.category, .textChat)
        XCTAssertEqual(target.hubFallback?.repoId, LagunaResources.xsUpstreamModelID)
        XCTAssertEqual(target.hubFallback?.revision, LagunaResources.xsUpstreamRevision)
        XCTAssertEqual(target.hubFallback?.patterns, LagunaResources.snapshotPatterns)
        XCTAssertEqual(target.upstreamRepoId, LagunaResources.xsUpstreamModelID)
        XCTAssertEqual(target.upstreamRevision, LagunaResources.xsUpstreamRevision)
        XCTAssertEqual(target.validationKind, .laguna)
        XCTAssertEqual(target.defaultRuntimeServingEngine, .textChatLaguna)
        XCTAssertEqual(target.estimatedDownloadBytes, LagunaResources.xsEstimatedDownloadBytes)
        XCTAssertEqual(target.companionModelIDs, [])
        XCTAssertFalse(target.runtimeAutoDownloadAllowed)
        XCTAssertNil(target.usageRestriction)
        XCTAssertTrue(LagunaResources.snapshotPatterns.contains("LICENSE*"))
    }

    func testMuseGlimmerUsesPinnedQ4TargetAndDFlash2Companion() throws {
        let target = try XCTUnwrap(ManagedModelCatalog.spec(for: MuseGlimmerResources.modelId))
        let companion = try XCTUnwrap(
            ManagedModelCatalog.spec(for: MuseGlimmerResources.dflash2ModelId)
        )
        let legacy = try XCTUnwrap(
            ManagedModelCatalog.spec(for: MuseGlimmerResources.assistantModelId)
        )

        XCTAssertEqual(target.hubFallback?.repoId, MuseGlimmerResources.artifactRepoId)
        XCTAssertEqual(target.hubFallback?.revision, MuseGlimmerResources.artifactRevision)
        XCTAssertEqual(target.upstreamRepoId, MuseGlimmerResources.artifactRepoId)
        XCTAssertEqual(target.upstreamRevision, MuseGlimmerResources.artifactRevision)
        XCTAssertEqual(target.estimatedDownloadBytes, MuseGlimmerResources.estimatedDownloadBytes)
        XCTAssertEqual(
            target.usageRestriction?.terms.first?.sourceRepoId,
            MuseGlimmerResources.upstreamRepoId
        )
        XCTAssertEqual(
            target.usageRestriction?.terms.first?.sourceRevision,
            MuseGlimmerResources.upstreamRevision
        )
        XCTAssertEqual(target.companionModelIDs, [MuseGlimmerResources.dflash2ModelId])
        XCTAssertEqual(target.validationKind, .museGlimmer)
        XCTAssertFalse(target.runtimeAutoDownloadAllowed)

        XCTAssertFalse(ManagedModelCatalog.allSpecs.contains { $0.id == companion.id })
        XCTAssertEqual(
            companion.hubFallback?.repoId,
            MuseGlimmerResources.dflash2UpstreamRepoId
        )
        XCTAssertEqual(
            companion.hubFallback?.revision,
            MuseGlimmerResources.dflash2UpstreamRevision
        )
        XCTAssertEqual(
            companion.hubFallback?.patterns,
            MuseGlimmerResources.dflash2SnapshotPatterns
        )
        XCTAssertEqual(companion.validationKind, .museGlimmerAssistant)
        XCTAssertEqual(
            companion.estimatedDownloadBytes,
            MuseGlimmerResources.dflash2EstimatedDownloadBytes
        )
        XCTAssertNil(companion.usageRestriction)
        XCTAssertFalse(companion.runtimeAutoDownloadAllowed)
        XCTAssertFalse(ManagedModelCatalog.allSpecs.contains { $0.id == legacy.id })
        XCTAssertEqual(legacy.upstreamRepoId, MuseGlimmerResources.assistantUpstreamRepoId)
    }

    func testQ36NanoUsesOptiQHubSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.q36NanoModelId))

        XCTAssertEqual(spec.hubFallback?.repoId, "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit")
        XCTAssertEqual(spec.hubFallback?.revision, "63d520640ca7461f31ba66104612135770090340")
        XCTAssertEqual(spec.upstreamRepoId, "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit")
        XCTAssertEqual(spec.validationKind, .q35)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("*.safetensors"), true)
    }

    func testQ38TwentySevenBUsesPinnedOfficialBF16Source() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.q38TwentySevenBModelId))

        XCTAssertEqual(ModelResolver.ModelID.q38TwentySevenB.rawValue, spec.id)
        XCTAssertEqual(spec.category, .visionChat)
        XCTAssertEqual(spec.installShape, .directoryRoot)
        XCTAssertEqual(spec.hubFallback?.repoId, Q35Resources.q38TwentySevenBUpstreamRepoId)
        XCTAssertEqual(spec.hubFallback?.revision, Q35Resources.q38TwentySevenBUpstreamRevision)
        XCTAssertEqual(spec.upstreamRepoId, Q35Resources.q38TwentySevenBUpstreamRepoId)
        XCTAssertEqual(spec.upstreamRevision, Q35Resources.q38TwentySevenBUpstreamRevision)
        XCTAssertEqual(spec.validationKind, .q35)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatQ36)
        XCTAssertEqual(spec.estimatedDownloadBytes, 55_586_114_863)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertTrue(spec.defaultCLICommands.contains("model benchmark code"))
        XCTAssertEqual(spec.hubFallback?.patterns.contains("generation_config.json"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("preprocessor_config.json"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("video_preprocessor_config.json"), true)
        XCTAssertEqual(spec.apiProfile?.thinkingLevels, [.low, .medium, .xhigh])
        XCTAssertEqual(spec.apiProfile?.thinkingLevelMap[.high], .xhigh)
        XCTAssertEqual(spec.apiProfile?.compatibility.supportsReasoningEffort, true)
    }

    func testQ38TwentySevenB4BitUsesCrownedTargetAndQuantizedMTPHead() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: Q35Resources.q38TwentySevenB4BitModelId)
        )

        XCTAssertEqual(ModelResolver.ModelID.q38TwentySevenB4Bit.rawValue, spec.id)
        XCTAssertEqual(spec.category, .visionChat)
        XCTAssertEqual(spec.installShape, .directoryRoot)
        XCTAssertEqual(spec.hubFallback?.repoId, Q35Resources.q38TwentySevenB4BitUpstreamRepoId)
        XCTAssertEqual(spec.hubFallback?.revision, Q35Resources.q38TwentySevenB4BitUpstreamRevision)
        XCTAssertEqual(spec.mountedHubFallbacks.count, 2)
        XCTAssertEqual(spec.mountedHubFallbacks.first?.destinationPath, Q35Resources.q38MTPComponentPath)
        XCTAssertEqual(
            spec.mountedHubFallbacks.first?.hubFallback.repoId,
            Q35Resources.q38MTP4BitUpstreamRepoId
        )
        XCTAssertEqual(
            Q35Resources.q38MTP4BitUpstreamRepoId,
            "morgan/qwen38-27b-mtp-r20k-lr3-q4-g64-q2-rerank"
        )
        XCTAssertEqual(
            spec.mountedHubFallbacks.first?.hubFallback.revision,
            Q35Resources.q38MTP4BitUpstreamRevision
        )
        XCTAssertEqual(
            Q35Resources.q38MTP4BitUpstreamRevision,
            "fd4a99c590dd6e468c0e2a28168c235e32151a4b"
        )
        XCTAssertEqual(
            spec.mountedHubFallbacks.first?.hubFallback.patterns,
            Q35Resources.q38MTPComponentSnapshotPatterns
        )
        XCTAssertEqual(
            spec.mountedHubFallbacks.last?.destinationPath,
            Q35Resources.q38LicenseComponentPath
        )
        XCTAssertEqual(
            spec.mountedHubFallbacks.last?.hubFallback.repoId,
            Q35Resources.q38TwentySevenBUpstreamRepoId
        )
        XCTAssertEqual(
            spec.mountedHubFallbacks.last?.hubFallback.patterns,
            Q35Resources.q38LicenseComponentSnapshotPatterns
        )
        XCTAssertEqual(spec.validationKind, .q35)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatQ36)
        XCTAssertEqual(spec.estimatedDownloadBytes, 15_520_000_000)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertTrue(spec.defaultCLICommands.contains("model benchmark code"))
        XCTAssertEqual(spec.apiProfile?.thinkingLevels, [.low, .medium, .xhigh])
        XCTAssertEqual(spec.apiProfile?.thinkingLevelMap[.minimal], .low)
        XCTAssertEqual(spec.apiProfile?.thinkingLevelMap[.max], .xhigh)
    }

    func testQ38FlashNextProfilesUsePinnedSawfwairArtifactsAndLicenseAcceptance() throws {
        let mixed = try XCTUnwrap(
            ManagedModelCatalog.spec(for: Q35Resources.q38FlashNextMixedModelId)
        )
        let q4 = try XCTUnwrap(
            ManagedModelCatalog.spec(for: Q35Resources.q38FlashNext4BitModelId)
        )

        XCTAssertEqual(mixed.hubFallback?.repoId, Q35Resources.q38FlashNextMixedUpstreamRepoId)
        XCTAssertEqual(mixed.hubFallback?.revision, Q35Resources.q38FlashNextMixedUpstreamRevision)
        XCTAssertEqual(mixed.estimatedDownloadBytes, 73_095_335_934)
        XCTAssertEqual(mixed.usageRestriction?.terms.first?.license, "Qwen Community License 1.0")
        XCTAssertFalse(mixed.runtimeAutoDownloadAllowed)
        XCTAssertEqual(q4.hubFallback?.repoId, Q35Resources.q38FlashNext4BitUpstreamRepoId)
        XCTAssertEqual(q4.hubFallback?.revision, Q35Resources.q38FlashNext4BitUpstreamRevision)
        XCTAssertEqual(q4.estimatedDownloadBytes, 104_742_357_706)
        XCTAssertNotNil(q4.usageRestriction)
    }

    func testQ38TwentySevenB4BitValidationRequiresMountedMTPFiles() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: Q35Resources.q38TwentySevenB4BitModelId)
        )
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        for filename in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"] {
            try Data().write(to: root.appendingPathComponent(filename))
        }

        XCTAssertEqual(
            Set(spec.missingPaths(in: root).map(\.lastPathComponent)),
            Set(Q35Resources.q38MTPComponentSnapshotPatterns)
        )

        let mtpRoot = root.appendingPathComponent(Q35Resources.q38MTPComponentPath, isDirectory: true)
        try FileManager.default.createDirectory(at: mtpRoot, withIntermediateDirectories: true)
        for filename in Q35Resources.q38MTPComponentSnapshotPatterns {
            try Data().write(to: mtpRoot.appendingPathComponent(filename))
        }
        let licenseRoot = root.appendingPathComponent(
            Q35Resources.q38LicenseComponentPath,
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: licenseRoot, withIntermediateDirectories: true)
        for filename in Q35Resources.q38LicenseComponentSnapshotPatterns {
            try Data().write(to: licenseRoot.appendingPathComponent(filename))
        }
        XCTAssertTrue(spec.missingPaths(in: root).isEmpty)
    }

    func testBonsai27BUsesPinnedPackedOneBitQ35Source() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.bonsai27B1BitModelId))

        XCTAssertEqual(ModelResolver.ModelID.bonsai27B1Bit.rawValue, spec.id)
        XCTAssertEqual(spec.category, .textChat)
        XCTAssertEqual(spec.installShape, .directoryRoot)
        XCTAssertEqual(spec.hubFallback?.repoId, Q35Resources.bonsai27B1BitUpstreamRepoId)
        XCTAssertEqual(spec.hubFallback?.revision, Q35Resources.bonsai27B1BitUpstreamRevision)
        XCTAssertEqual(spec.upstreamRepoId, Q35Resources.bonsai27B1BitUpstreamRepoId)
        XCTAssertEqual(spec.upstreamRevision, Q35Resources.bonsai27B1BitUpstreamRevision)
        XCTAssertEqual(spec.validationKind, .q35)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatQ36)
        XCTAssertEqual(spec.estimatedDownloadBytes, 5_128_837_600)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("LICENSE*"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("NOTICE*"), true)
    }

    func testTernaryBonsai27BUsesPinnedPackedTwoBitQ35Source() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.bonsai27B2BitModelId))

        XCTAssertEqual(ModelResolver.ModelID.bonsai27B2Bit.rawValue, spec.id)
        XCTAssertEqual(spec.category, .textChat)
        XCTAssertEqual(spec.installShape, .directoryRoot)
        XCTAssertEqual(spec.hubFallback?.repoId, Q35Resources.bonsai27B2BitUpstreamRepoId)
        XCTAssertEqual(spec.hubFallback?.revision, Q35Resources.bonsai27B2BitUpstreamRevision)
        XCTAssertEqual(spec.upstreamRepoId, Q35Resources.bonsai27B2BitUpstreamRepoId)
        XCTAssertEqual(spec.upstreamRevision, Q35Resources.bonsai27B2BitUpstreamRevision)
        XCTAssertEqual(spec.validationKind, .q35)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatQ36)
        XCTAssertEqual(spec.estimatedDownloadBytes, 8_521_085_419)
    }


    func testOrnith9BUsesNativeQ35OptiQHubSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.ornith9BModelId))

        XCTAssertEqual(spec.category, .textCode)
        XCTAssertEqual(spec.hubFallback?.repoId, Q35Resources.ornith9BUpstreamRepoId)
        XCTAssertEqual(spec.hubFallback?.revision, Q35Resources.ornith9BUpstreamRevision)
        XCTAssertEqual(spec.upstreamRepoId, Q35Resources.ornith9BUpstreamRepoId)
        XCTAssertEqual(spec.validationKind, .q35)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatQ36)
        XCTAssertEqual(spec.estimatedDownloadBytes, Q35Resources.ornith9BEstimatedDownloadBytes)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("optiq_metadata.json"), true)
    }

    func testOrnith35BMLXUsesPinnedOfficialNativeQ35Source() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.ornith35BMLXModelId))

        XCTAssertEqual(spec.category, .textCode)
        XCTAssertEqual(spec.installShape, .directoryRoot)
        XCTAssertEqual(spec.hubFallback?.repoId, Q35Resources.ornith35BMLXUpstreamRepoId)
        XCTAssertEqual(spec.hubFallback?.revision, Q35Resources.ornith35BMLXUpstreamRevision)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("generation_config.json"), true)
        XCTAssertEqual(spec.upstreamRepoId, Q35Resources.ornith35BMLXUpstreamRepoId)
        XCTAssertEqual(spec.upstreamRevision, Q35Resources.ornith35BMLXUpstreamRevision)
        XCTAssertEqual(spec.validationKind, .q35)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.estimatedDownloadBytes, Q35Resources.ornith35BMLXEstimatedDownloadBytes)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatQ36)
        XCTAssertEqual(spec.apiProfile?.contextWindow, Q35Resources.ornith35BMLXContextLength)
        XCTAssertEqual(spec.companionModelIDs, [Q35Resources.ornith35BMTPModelId])
    }

    func testOrnith35BMLXQuantizedVariantsUsePinnedOfficialSourcesAndMTPCompanion() throws {
        let expected: [(String, String, String, Int64)] = [
            (
                Q35Resources.ornith35BMLX4BitModelId,
                Q35Resources.ornith35BMLX4BitUpstreamRepoId,
                Q35Resources.ornith35BMLX4BitUpstreamRevision,
                Q35Resources.ornith35BMLX4BitEstimatedDownloadBytes
            ),
            (
                Q35Resources.ornith35BMLX6BitModelId,
                Q35Resources.ornith35BMLX6BitUpstreamRepoId,
                Q35Resources.ornith35BMLX6BitUpstreamRevision,
                Q35Resources.ornith35BMLX6BitEstimatedDownloadBytes
            ),
            (
                Q35Resources.ornith35BMLX8BitModelId,
                Q35Resources.ornith35BMLX8BitUpstreamRepoId,
                Q35Resources.ornith35BMLX8BitUpstreamRevision,
                Q35Resources.ornith35BMLX8BitEstimatedDownloadBytes
            ),
        ]

        for (modelID, repoID, revision, downloadBytes) in expected {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: modelID))
            XCTAssertEqual(spec.hubFallback?.repoId, repoID)
            XCTAssertEqual(spec.hubFallback?.revision, revision)
            XCTAssertEqual(spec.upstreamRepoId, repoID)
            XCTAssertEqual(spec.upstreamRevision, revision)
            XCTAssertEqual(spec.validationKind, .q35)
            XCTAssertEqual(spec.estimatedDownloadBytes, downloadBytes)
            XCTAssertEqual(spec.companionModelIDs, [Q35Resources.ornith35BMTPModelId])
        }

        let companion = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.ornith35BMTPModelId))
        XCTAssertEqual(companion.validationKind, .q35MTPAssistant)
        XCTAssertEqual(companion.hubFallback?.repoId, Q35Resources.ornith35BMTPUpstreamRepoId)
        XCTAssertEqual(companion.hubFallback?.revision, Q35Resources.ornith35BMTPUpstreamRevision)
        XCTAssertEqual(companion.hubFallback?.patterns, Q35Resources.ornith35BMTPSnapshotPatterns)
    }

    func testOrnith35BMTPCompanionResolvesFromManagedModelStore() throws {
        let modelsRoot = FileManager.default.temporaryDirectory.appendingPathComponent(
            "ornith-mtp-store-\(UUID().uuidString)",
            isDirectory: true
        )
        defer {
            MereRunModelPaths.setProcessModelsDirOverride(nil)
            try? FileManager.default.removeItem(at: modelsRoot)
        }
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)
        let companionRoot = modelsRoot.appendingPathComponent(
            Q35Resources.ornith35BMTPModelId,
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: companionRoot, withIntermediateDirectories: true)
        try Data("{}".utf8).write(
            to: companionRoot.appendingPathComponent("model.safetensors.index.json")
        )
        try Data([0]).write(
            to: companionRoot.appendingPathComponent(Q35Resources.ornith35BMTPShardFilename)
        )
        try MereRunModelManifest.template(for: .ornith35BMTP).write(to: companionRoot)

        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.ornith35BMTPModelId))
        XCTAssertEqual(spec.missingPaths(in: companionRoot), [])
        XCTAssertEqual(spec.managedRuntimeURL()?.standardizedFileURL, companionRoot.standardizedFileURL)
    }

    func testInfinityParser2ProRequiresExplicitPull() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.infinityParser2ProModelId))

        XCTAssertEqual(spec.category, .visionOCR)
        XCTAssertEqual(spec.hubFallback?.repoId, Q35Resources.infinityParser2ProUpstreamRepoId)
        XCTAssertEqual(spec.hubFallback?.revision, Q35Resources.infinityParser2ProUpstreamRevision)
        XCTAssertEqual(spec.validationKind, .q35)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
    }

    func testInfinityParser2ProInt8RequiresExplicitPull() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Q35Resources.infinityParser2ProInt8ModelId))

        XCTAssertEqual(spec.category, .visionOCR)
        XCTAssertEqual(spec.hubFallback?.repoId, Q35Resources.infinityParser2ProInt8UpstreamRepoId)
        XCTAssertEqual(spec.hubFallback?.revision, Q35Resources.infinityParser2ProInt8UpstreamRevision)
        XCTAssertEqual(spec.validationKind, .q35)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.estimatedDownloadBytes, 38 * 1_073_741_824)
        XCTAssertEqual(spec.defaultCLICommands, ["vision ocr"])
    }

    func testQ36NanoGGUFUsesUpstreamFilePath() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "text-chat-q36-nano-gguf"))

        XCTAssertEqual(spec.installShape, .singleFile(relativePath: "text-chat-q36-nano-gguf.gguf"))
        XCTAssertEqual(spec.hubFallback?.repoId, "unsloth/Qwen3.6-35B-A3B-GGUF")
        XCTAssertEqual(spec.hubFallback?.patterns, ["Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"])
        XCTAssertEqual(spec.hubFallback?.filePath, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
        XCTAssertEqual(spec.upstreamRepoId, "unsloth/Qwen3.6-35B-A3B-GGUF")
        XCTAssertEqual(spec.validationKind, .codegenGGUF)
    }

    func testInklingSmallUsesPinnedNativeMLXSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: InklingResources.modelID))

        XCTAssertEqual(spec.category, .textChat)
        XCTAssertEqual(spec.installShape, .directoryRoot)
        XCTAssertEqual(spec.hubFallback?.repoId, InklingResources.artifactRepoID)
        XCTAssertEqual(spec.hubFallback?.revision, InklingResources.artifactRevision)
        XCTAssertEqual(spec.hubFallback?.patterns, InklingResources.snapshotPatterns)
        XCTAssertNil(spec.hubFallback?.filePath)
        XCTAssertEqual(spec.upstreamRepoId, InklingResources.artifactRepoID)
        XCTAssertEqual(spec.upstreamRevision, InklingResources.artifactRevision)
        XCTAssertEqual(spec.validationKind, .inkling)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.estimatedDownloadBytes, InklingResources.estimatedDownloadBytes)
        XCTAssertNil(spec.defaultRuntimeServingEngine)
        XCTAssertTrue(spec.defaultCLICommands.contains("text train-lora"))
    }

    func testInklingSmallRequiresNativeMLXRootFiles() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: InklingResources.modelID))
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let missing = spec.missingPaths(in: root, fileManager: .default)
        XCTAssertEqual(Set(missing.map(\.lastPathComponent)), [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        ])

        for filename in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: root.appendingPathComponent(filename).path,
                contents: Data()
            ))
        }
        let index = #"{"weight_map":{"language_model.model.embed_tokens.weight":"model-00001-of-00001.safetensors"}}"#
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("model.safetensors.index.json").path,
            contents: Data(index.utf8)
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("model-00001-of-00001.safetensors").path,
            contents: Data()
        ))

        XCTAssertTrue(spec.missingPaths(in: root, fileManager: .default).isEmpty)
        XCTAssertTrue(spec.validateRuntimeURL(root, fileManager: .default).isEmpty)
    }

    func testNorthMiniCodeUsesPinnedHubSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: NorthMiniCodeResources.modelId))

        XCTAssertEqual(spec.category, .textCode)
        XCTAssertEqual(spec.installShape, .singleFile(relativePath: NorthMiniCodeResources.managedRelativePath))
        XCTAssertEqual(spec.hubFallback?.repoId, NorthMiniCodeResources.upstreamRepoId)
        XCTAssertEqual(spec.hubFallback?.revision, NorthMiniCodeResources.upstreamRevision)
        XCTAssertEqual(spec.hubFallback?.patterns, [NorthMiniCodeResources.ggufFile])
        XCTAssertEqual(spec.hubFallback?.filePath, NorthMiniCodeResources.ggufFile)
        XCTAssertEqual(spec.upstreamRepoId, NorthMiniCodeResources.upstreamRepoId)
        XCTAssertEqual(spec.upstreamRevision, NorthMiniCodeResources.upstreamRevision)
        XCTAssertEqual(spec.validationKind, .codegenGGUF)
        XCTAssertTrue(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.estimatedDownloadBytes, NorthMiniCodeResources.estimatedDownloadBytes)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textCode)
    }

    func testOrnith35BUsesPinnedGGUFHubSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Ornith35BCodeResources.modelId))

        XCTAssertEqual(spec.category, .textCode)
        XCTAssertEqual(spec.installShape, .singleFile(relativePath: Ornith35BCodeResources.managedRelativePath))
        XCTAssertEqual(spec.hubFallback?.repoId, Ornith35BCodeResources.upstreamRepoId)
        XCTAssertEqual(spec.hubFallback?.revision, Ornith35BCodeResources.upstreamRevision)
        XCTAssertEqual(spec.hubFallback?.patterns, [Ornith35BCodeResources.ggufFile])
        XCTAssertEqual(spec.hubFallback?.filePath, Ornith35BCodeResources.ggufFile)
        XCTAssertEqual(spec.upstreamRepoId, Ornith35BCodeResources.upstreamRepoId)
        XCTAssertEqual(spec.upstreamRevision, Ornith35BCodeResources.upstreamRevision)
        XCTAssertEqual(spec.validationKind, .codegenGGUF)
        XCTAssertTrue(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.estimatedDownloadBytes, Ornith35BCodeResources.estimatedDownloadBytes)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textCode)
    }

    func testQwen3CodeAcceptsSymlinkedNestedHubGGUFInstallRoot() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: CodeGenResources.defaultModelId))
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        XCTAssertEqual(spec.hubFallback?.filePath, CodeGenResources.hubGGUFPath)

        let snapshot = temp.appendingPathComponent("snapshot", isDirectory: true)
        let snapshotFile = snapshot.appendingPathComponent(CodeGenResources.hubGGUFPath, isDirectory: false)
        try FileManager.default.createDirectory(
            at: snapshotFile.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        XCTAssertTrue(FileManager.default.createFile(atPath: snapshotFile.path, contents: Data()))

        let root = temp.appendingPathComponent(CodeGenResources.defaultModelId, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try MereRunModelManifest.template(for: .qwen3Code, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("Qwen3-Coder-Next-Q4_K_M", isDirectory: true),
            withDestinationURL: snapshot.appendingPathComponent("Qwen3-Coder-Next-Q4_K_M", isDirectory: true)
        )

        XCTAssertTrue(spec.missingPaths(in: root, fileManager: .default).isEmpty)
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))
    }

    func testLFM2UsesLiquidAIHubSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: LFM2Resources.defaultModelId))

        XCTAssertEqual(spec.category, .textChat)
        XCTAssertEqual(spec.installShape, .directoryRoot)
        XCTAssertEqual(spec.hubFallback?.repoId, LFM2Resources.upstreamRepoId)
        XCTAssertEqual(spec.hubFallback?.revision, LFM2Resources.upstreamRevision)
        XCTAssertEqual(spec.upstreamRepoId, LFM2Resources.upstreamRepoId)
        XCTAssertEqual(spec.upstreamRevision, LFM2Resources.upstreamRevision)
        XCTAssertEqual(spec.validationKind, .lfm2)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatLFM2)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("model.safetensors.index.json"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("*.safetensors"), true)
    }

    func testDenseLFM2UsesPinnedFourBitLiquidAIHubPartition() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: LFM2Resources.denseModelId))

        XCTAssertEqual(spec.category, .textChat)
        XCTAssertEqual(spec.installShape, .directoryRoot)
        XCTAssertEqual(spec.hubFallback?.repoId, LFM2Resources.denseUpstreamRepoId)
        XCTAssertEqual(spec.hubFallback?.revision, LFM2Resources.denseUpstreamRevision)
        XCTAssertEqual(spec.hubFallback?.patterns, LFM2Resources.denseSnapshotPatterns)
        XCTAssertEqual(spec.validationKind, .lfm2)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatLFM2)
        XCTAssertEqual(spec.estimatedDownloadBytes, 1_601_108_788)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)

        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let variant = root.appendingPathComponent(LFM2Resources.denseVariantSubdirectory, isDirectory: true)
        try FileManager.default.createDirectory(at: variant, withIntermediateDirectories: true)
        for filename in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        ] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: variant.appendingPathComponent(filename).path,
                contents: Data()
            ))
        }

        XCTAssertEqual(spec.normalizedRootURL(root), variant.standardizedFileURL)
        XCTAssertTrue(spec.missingPaths(in: root).isEmpty)
    }

    func testLFM25QADModelsUsePinnedSawfwairMLXSnapshots() throws {
        let expected = [
            (
                LFM2Resources.smallQADModelId,
                LFM2Resources.smallQADRepoId,
                LFM2Resources.smallQADRevision,
                LFM2Resources.smallQADEstimatedDownloadBytes
            ),
            (
                LFM2Resources.denseQADModelId,
                LFM2Resources.denseQADRepoId,
                LFM2Resources.denseQADRevision,
                LFM2Resources.denseQADEstimatedDownloadBytes
            ),
        ]

        for (modelID, repoID, revision, downloadBytes) in expected {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: modelID))
            XCTAssertEqual(spec.category, .textChat)
            XCTAssertEqual(spec.installShape, .directoryRoot)
            XCTAssertEqual(spec.hubFallback?.repoId, repoID)
            XCTAssertEqual(spec.hubFallback?.revision, revision)
            XCTAssertEqual(spec.hubFallback?.patterns, LFM2Resources.qadSnapshotPatterns)
            XCTAssertEqual(spec.validationKind, .lfm2)
            XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatLFM2)
            XCTAssertEqual(spec.estimatedDownloadBytes, downloadBytes)
            XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
            XCTAssertNotNil(spec.usageRestriction)
        }
    }

    func testLFM25TargetsUseTheirPinnedDSparkCompanions() throws {
        let pairs = [
            (
                LFM2Resources.smallModelId,
                LFM2Resources.smallDSparkModelId,
                LFM2Resources.smallDSparkRepoId,
                LFM2Resources.smallDSparkRevision
            ),
            (
                LFM2Resources.denseBF16ModelId,
                LFM2Resources.denseDSparkModelId,
                LFM2Resources.denseDSparkRepoId,
                LFM2Resources.denseDSparkRevision
            ),
            (
                LFM2Resources.a1bBF16ModelId,
                LFM2Resources.defaultDSparkModelId,
                LFM2Resources.defaultDSparkRepoId,
                LFM2Resources.defaultDSparkRevision
            ),
        ]

        for (targetID, dsparkID, repoID, revision) in pairs {
            let target = try XCTUnwrap(ManagedModelCatalog.spec(for: targetID))
            let dspark = try XCTUnwrap(ManagedModelCatalog.spec(for: dsparkID))
            XCTAssertEqual(target.companionModelIDs, [dsparkID])
            XCTAssertEqual(dspark.hubFallback?.repoId, repoID)
            XCTAssertEqual(dspark.hubFallback?.revision, revision)
            XCTAssertEqual(dspark.validationKind, .lfm2DSpark)
            XCTAssertFalse(dspark.runtimeAutoDownloadAllowed)
            XCTAssertNotNil(dspark.usageRestriction)
        }

        XCTAssertTrue(
            try XCTUnwrap(ManagedModelCatalog.spec(for: LFM2Resources.denseModelId))
                .companionModelIDs.isEmpty
        )
        XCTAssertTrue(
            try XCTUnwrap(ManagedModelCatalog.spec(for: LFM2Resources.defaultModelId))
                .companionModelIDs.isEmpty
        )
    }

    func testLFM25DSparkCompanionValidatesWithoutStandaloneChatComponents() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        for filename in ["config.json", "model.safetensors"] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: root.appendingPathComponent(filename).path,
                contents: Data()
            ))
        }
        try MereRunModelManifest.template(
            for: .lfm25Dense2_6BDSpark,
            createdAt: Date(timeIntervalSince1970: 0)
        ).write(to: root)

        let report = MereRunModelValidator.validate(
            modelRoot: root,
            expectedModelID: LFM2Resources.denseDSparkModelId
        )
        XCTAssertTrue(report.isValid, report.errors.joined(separator: "\n"))
    }

    func testLFM25VisionUsesPinnedLiquidAIMLXCheckpoint() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: LFM2Resources.visionModelId))

        XCTAssertEqual(spec.category, .visionChat)
        XCTAssertEqual(spec.installShape, .directoryRoot)
        XCTAssertEqual(spec.hubFallback?.repoId, LFM2Resources.visionUpstreamRepoId)
        XCTAssertEqual(spec.hubFallback?.revision, LFM2Resources.visionUpstreamRevision)
        XCTAssertEqual(spec.hubFallback?.patterns, LFM2Resources.visionSnapshotPatterns)
        XCTAssertEqual(spec.validationKind, .lfm2)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatLFM2)
        XCTAssertEqual(spec.estimatedDownloadBytes, 3_736_739_700)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertTrue(spec.hubFallback?.patterns.contains("processor_config.json") == true)

        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        for filename in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        ] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: root.appendingPathComponent(filename).path,
                contents: Data()
            ))
        }
        XCTAssertEqual(spec.missingPaths(in: root).map(\.lastPathComponent), ["processor_config.json"])
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("processor_config.json").path,
            contents: Data()
        ))
        XCTAssertTrue(spec.missingPaths(in: root).isEmpty)
    }

    func testQwen3TTSSpecsDownloadSpeechTokenizer() throws {
        for id in ["speech-tts-qwen3-nano", "speech-tts-qwen3-customvoice"] {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))

            XCTAssertEqual(spec.category, .speechTTS)
            XCTAssertEqual(spec.validationKind, .qwen3TTS)
            XCTAssertEqual(spec.hubFallback?.patterns.contains("speech_tokenizer/*"), true)
        }
    }

    func testQwen3TTSRejectsRootMissingSpeechTokenizer() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalQwen3TTSRootWithoutSpeechTokenizer(at: root, id: .qwen3TTSNano)

        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "speech-tts-qwen3-nano"))
        XCTAssertFalse(spec.isManagedRootComplete(root, fileManager: .default))

        let missing = spec.missingPaths(in: root, fileManager: .default).map(\.path)
        XCTAssertTrue(missing.contains { $0.hasSuffix("speech_tokenizer/config.json") })
        XCTAssertTrue(missing.contains { $0.hasSuffix("speech_tokenizer") })
    }

    func testQwen3TTSAcceptsSymlinkedSpeechTokenizerDirectory() throws {
        let root = try makeTemporaryDirectory()
        let snapshot = try makeTemporaryDirectory()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: snapshot)
        }
        try writeMinimalQwen3TTSRootWithoutSpeechTokenizer(at: root, id: .qwen3TTSCustomVoice)

        let tokenizer = snapshot.appendingPathComponent("speech_tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        for file in ["config.json", "model.safetensors"] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: tokenizer.appendingPathComponent(file).path,
                contents: Data("{}".utf8)
            ))
        }
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("speech_tokenizer", isDirectory: true),
            withDestinationURL: tokenizer
        )

        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "speech-tts-qwen3-customvoice"))
        XCTAssertTrue(spec.missingPaths(in: root, fileManager: .default).isEmpty)
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))
    }

    func testZImageNanoAcceptsMFluxLayoutWithoutDiffusersConfigs() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalMFluxZImageNano(
            at: root,
            upstreamRepoId: "filipstrand/Z-Image-Turbo-mflux-4bit@b3a8f31115a11f2f9e2fa0bfbc8d78dcc3e6568b"
        )

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

    func testZImageNanoResolvesLegacyMainManifestThroughManagedSymlink() throws {
        let modelsRoot = try makeTemporaryDirectory()
        let snapshotRoot = try makeTemporaryDirectory()
        defer {
            MereRunModelPaths.setProcessModelsDirOverride(nil)
            try? FileManager.default.removeItem(at: modelsRoot)
            try? FileManager.default.removeItem(at: snapshotRoot)
        }
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)
        try writeMinimalMFluxZImageNano(
            at: snapshotRoot,
            upstreamRepoId: "filipstrand/Z-Image-Turbo-mflux-4bit@main"
        )
        try FileManager.default.createSymbolicLink(
            at: modelsRoot.appendingPathComponent("image-zimage-nano", isDirectory: true),
            withDestinationURL: snapshotRoot
        )

        let resolved = try XCTUnwrap(ModelResolver().resolveIfPresent(.zetaNano))
        XCTAssertEqual(resolved.modelID, .zetaNano)
        XCTAssertEqual(resolved.source, .localModelStore)
        XCTAssertEqual(resolved.rootURL.resolvingSymlinksInPath(), snapshotRoot.resolvingSymlinksInPath())
    }

    func testManagedRef2VARequiresTheCurrentPinnedArtifactRevision() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        for filename in MiniMaxH3Resources.requiredFiles {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: root.appendingPathComponent(filename).path,
                contents: Data()
            ))
        }

        let modelID = ModelResolver.ModelID.miniMaxH3Ref2VAMLX
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: modelID.rawValue))
        var manifest = MereRunModelManifest.template(
            for: modelID,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        manifest.upstreamRepoId = "Sawfwair/MiniMax-H3-Ref2VA-MLX-8bit"
            + "@abb9114fe9d6e3cccc6376eee1abaf09d3f2a9fe"
        try manifest.write(to: root)
        XCTAssertFalse(spec.isManagedRootComplete(root, fileManager: .default))

        manifest.upstreamRepoId = "\(MiniMaxH3Resources.ref2vaArtifactRepository)"
            + "@\(MiniMaxH3Resources.ref2vaArtifactRevision)"
        try manifest.write(to: root)
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))
    }

    func testManagedCompactH3RequiresCurrentPinnedArtifactRevision() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let cases: [(ModelResolver.ModelID, String, String)] = [
            (
                .miniMaxH3FL2VABF16MLX,
                MiniMaxH3Resources.compactBF16ArtifactRepository,
                MiniMaxH3Resources.compactBF16ArtifactRevision
            ),
            (
                .miniMaxH3FL2VAQ8MLX,
                MiniMaxH3Resources.q8ArtifactRepository,
                MiniMaxH3Resources.q8ArtifactRevision
            ),
        ]
        for (modelID, repository, revision) in cases {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: modelID.rawValue))
            var manifest = MereRunModelManifest.template(
                for: modelID,
                createdAt: Date(timeIntervalSince1970: 0)
            )
            manifest.upstreamRepoId = "\(repository)@stale-revision"
            try manifest.write(to: root)
            XCTAssertFalse(spec.managedSourceMatches(root, fileManager: .default), modelID.rawValue)

            manifest.upstreamRepoId = "\(repository)@\(revision)"
            try manifest.write(to: root)
            XCTAssertTrue(spec.managedSourceMatches(root, fileManager: .default), modelID.rawValue)
        }
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

    func testIdeogram4AcceptsSDNQDiffusersLayout() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalIdeogram4(at: root)

        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Ideogram4Resources.modelId))
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: Ideogram4Resources.modelId)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertTrue(report.warnings.isEmpty)
        XCTAssertEqual(report.manifest?.engine, .ideogram4)
        XCTAssertEqual(report.manifest?.family, .ideogram)
        XCTAssertEqual(report.manifest?.components?.unconditionalTransformer, .local(path: "unconditional_transformer"))
    }

    func testIdeogram4RejectsMissingUnconditionalTransformer() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalIdeogram4(at: root)
        try FileManager.default.removeItem(at: root.appendingPathComponent("unconditional_transformer"))

        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: Ideogram4Resources.modelId))
        XCTAssertFalse(spec.isManagedRootComplete(root, fileManager: .default))

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: Ideogram4Resources.modelId)
        XCTAssertFalse(report.errors.isEmpty)
        XCTAssertTrue(report.errors.contains { $0.contains("unconditional_transformer/config.json") })
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

    func testACEStepXLTurboUsesMountedXLLayout() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: ModelResolver.ModelID.aceStepXLTurbo.rawValue))
        XCTAssertEqual(spec.hubFallback?.repoId, "ACE-Step/Ace-Step1.5")
        XCTAssertEqual(spec.hubFallback?.revision, "19671f406d603126926c1b7e2adc169acbcade22")
        XCTAssertEqual(spec.mountedHubFallbacks.map(\.destinationPath), ["acestep-v15-xl-turbo"])
        XCTAssertEqual(spec.mountedHubFallbacks.first?.hubFallback.repoId, "ACE-Step/acestep-v15-xl-turbo")
        XCTAssertEqual(
            spec.mountedHubFallbacks.first?.hubFallback.revision,
            "d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee"
        )

        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalACEStepRoot(at: root, turboSubdirectory: "acestep-v15-xl-turbo")
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))

        let standardRoot = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: standardRoot) }
        try writeMinimalACEStepRoot(at: standardRoot, turboSubdirectory: "acestep-v15-turbo")
        XCTAssertFalse(spec.isManagedRootComplete(standardRoot, fileManager: .default))
    }

    func testACEStepXLBaseAndSFTUsePinnedMountedLayouts() throws {
        let fixtures: [
            (
                modelID: ModelResolver.ModelID,
                destination: String,
                repoID: String,
                revision: String
            )
        ] = [
            (
                .aceStepXLBase,
                "acestep-v15-xl-base",
                "ACE-Step/acestep-v15-xl-base",
                "220c1166efbdd9583eafcb12eb160594bbfcb241"
            ),
            (
                .aceStepXLSFT,
                "acestep-v15-xl-sft",
                "ACE-Step/acestep-v15-xl-sft",
                "d06de46b4622f781cf07f4a013a67d591ca52819"
            ),
        ]

        for fixture in fixtures {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: fixture.modelID.rawValue))
            XCTAssertEqual(spec.hubFallback?.repoId, "ACE-Step/Ace-Step1.5")
            XCTAssertEqual(spec.hubFallback?.revision, "19671f406d603126926c1b7e2adc169acbcade22")
            XCTAssertEqual(spec.mountedHubFallbacks.map(\.destinationPath), [fixture.destination])
            XCTAssertEqual(spec.mountedHubFallbacks.first?.hubFallback.repoId, fixture.repoID)
            XCTAssertEqual(spec.mountedHubFallbacks.first?.hubFallback.revision, fixture.revision)
            XCTAssertEqual(
                spec.mountedHubFallbacks.first?.hubFallback.patterns.contains("apg_guidance.py"),
                true
            )

            let root = try makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            try writeMinimalACEStepRoot(at: root, turboSubdirectory: fixture.destination)
            XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))
        }
    }

    func testACEStepXLTurboLM4BRequiresMountedLM() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: ModelResolver.ModelID.aceStepXLTurboLM4B.rawValue))
        XCTAssertEqual(
            spec.mountedHubFallbacks.map(\.destinationPath),
            ["acestep-v15-xl-turbo", "acestep-5Hz-lm-4B"]
        )
        XCTAssertEqual(spec.mountedHubFallbacks.last?.hubFallback.repoId, "ACE-Step/acestep-5Hz-lm-4B")
        XCTAssertEqual(
            spec.mountedHubFallbacks.last?.hubFallback.revision,
            "0a3ec94b557aea7d508da38b31cfe7341f6ff737"
        )

        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalACEStepRoot(at: root, turboSubdirectory: "acestep-v15-xl-turbo")
        XCTAssertFalse(spec.isManagedRootComplete(root, fileManager: .default))

        try writeMinimalACEStepLMRoot(at: root.appendingPathComponent("acestep-5Hz-lm-4B", isDirectory: true))
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))
    }

    func testACEStepStandalonePlannerModelsUsePinnedSources() throws {
        let planner17 = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.aceStepLM17B.rawValue)
        )
        XCTAssertEqual(planner17.validationKind, .aceStepLM)
        XCTAssertEqual(planner17.hubFallback?.repoId, "ACE-Step/Ace-Step1.5")
        XCTAssertEqual(
            planner17.hubFallback?.revision,
            "19671f406d603126926c1b7e2adc169acbcade22"
        )
        XCTAssertEqual(
            planner17.hubFallback?.patterns,
            ["acestep-5Hz-lm-1.7B/*"]
        )

        let nestedRoot = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: nestedRoot) }
        let nestedLM = nestedRoot.appendingPathComponent(
            "acestep-5Hz-lm-1.7B",
            isDirectory: true
        )
        try writeMinimalACEStepLMRoot(at: nestedLM)
        XCTAssertEqual(
            planner17.normalizedRootURL(nestedRoot, fileManager: .default),
            nestedLM.resolvingSymlinksInPath()
        )
        XCTAssertTrue(planner17.isManagedRootComplete(nestedRoot, fileManager: .default))

        let planner4 = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.aceStepLM4B.rawValue)
        )
        XCTAssertEqual(planner4.validationKind, .aceStepLM)
        XCTAssertEqual(planner4.hubFallback?.repoId, "ACE-Step/acestep-5Hz-lm-4B")
        XCTAssertEqual(
            planner4.hubFallback?.revision,
            "0a3ec94b557aea7d508da38b31cfe7341f6ff737"
        )
    }

    func testMagentaRT2SpecsUsePinnedExportedRuntimeAssets() throws {
        let small = try XCTUnwrap(ManagedModelCatalog.spec(for: ModelResolver.ModelID.magentaRT2Small.rawValue))
        XCTAssertEqual(small.category, .music)
        XCTAssertEqual(small.hubFallback?.repoId, "google/magenta-realtime-2")
        XCTAssertEqual(small.hubFallback?.revision, "010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc")
        XCTAssertEqual(small.validationKind, .magentaRT2)
        XCTAssertEqual(small.hubFallback?.patterns.contains("models/mrt2_small/mrt2_small.mlxfn"), true)
        XCTAssertEqual(small.hubFallback?.patterns.contains("checkpoints/mrt2_small.safetensors"), false)

        let base = try XCTUnwrap(ManagedModelCatalog.spec(for: ModelResolver.ModelID.magentaRT2Base.rawValue))
        XCTAssertEqual(base.category, .music)
        XCTAssertEqual(base.hubFallback?.repoId, "google/magenta-realtime-2")
        XCTAssertEqual(base.hubFallback?.revision, "010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc")
        XCTAssertEqual(base.validationKind, .magentaRT2)
        XCTAssertEqual(base.hubFallback?.patterns.contains("models/mrt2_base/mrt2_base.mlxfn"), true)
        XCTAssertEqual(base.hubFallback?.patterns.contains("checkpoints/mrt2_base.safetensors"), false)
    }

    func testMagentaRT2RootValidationRequiresModelAndSharedResources() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: ModelResolver.ModelID.magentaRT2Small.rawValue))

        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalMagentaRT2Root(at: root, modelName: "mrt2_small")
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))

        try FileManager.default.removeItem(at: root.appendingPathComponent("resources/musiccoca/text_encoder.tflite"))
        XCTAssertFalse(spec.isManagedRootComplete(root, fileManager: .default))
        XCTAssertEqual(
            spec.missingPaths(in: root, fileManager: .default).map(\.lastPathComponent),
            ["text_encoder.tflite"]
        )
    }

    func testMuScriptorSpecsUsePinnedGatedCheckpointLayouts() throws {
        let expected: [(ModelResolver.ModelID, String, String)] = [
            (.muScriptorSmall, "MuScriptor/muscriptor-small", "8c127f603b807520fa465c838e9bfee8a91ada4e"),
            (.muScriptorMedium, "MuScriptor/muscriptor-medium", "f32236969308476e01fd3aae67357de5feb05a2d"),
            (.muScriptorLarge, "MuScriptor/muscriptor-large", "8809fdfbed2affa7ade94a7059e746e3880720e7"),
        ]
        for (modelID, repoID, revision) in expected {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: modelID.rawValue))
            XCTAssertEqual(spec.category, .music)
            XCTAssertEqual(spec.hubFallback?.repoId, repoID)
            XCTAssertEqual(spec.hubFallback?.revision, revision)
            XCTAssertEqual(spec.hubFallback?.patterns.contains("config.json"), true)
            XCTAssertEqual(spec.hubFallback?.patterns.contains("model.safetensors"), true)
            XCTAssertEqual(spec.hubFallback?.patterns.contains("LICENSE*"), true)
            XCTAssertEqual(spec.hubFallback?.patterns.contains("README.md"), true)
            XCTAssertEqual(spec.validationKind, .muScriptor)
            XCTAssertEqual(spec.defaultCLICommands, ["music transcribe"])
        }
    }

    func testMuScriptorRootRequiresConfigAndWeights() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: ModelResolver.ModelID.muScriptorSmall.rawValue))
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("{}".utf8).write(to: root.appendingPathComponent("config.json"))
        XCTAssertFalse(spec.isManagedRootComplete(root, fileManager: .default))
        try Data().write(to: root.appendingPathComponent("model.safetensors"))
        XCTAssertTrue(spec.missingPaths(in: root, fileManager: .default).isEmpty)
    }

    func testWooshDFlowSpecUsesFocusedHuggingFaceMirrorLayout() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: ModelResolver.ModelID.wooshDFlow.rawValue))

        XCTAssertEqual(spec.category, .sfx)
        XCTAssertEqual(spec.hubFallback?.repoId, WooshResources.huggingFaceMirrorRepoId)
        XCTAssertEqual(spec.validationKind, .woosh)
        XCTAssertEqual(spec.defaultCLICommands, ["sfx generate"])
        XCTAssertTrue(spec.hubFallback?.patterns.contains("checkpoints/Woosh-DFlow/*") == true)
        XCTAssertTrue(spec.hubFallback?.patterns.contains("checkpoints/Woosh-VFlow-8s/*") == false)
        XCTAssertEqual(spec.mountedHubFallbacks.first?.destinationPath, "checkpoints/TextConditionerA/tokenizer")
        XCTAssertEqual(spec.mountedHubFallbacks.first?.hubFallback.repoId, WooshResources.robertaTokenizerRepoId)
        XCTAssertEqual(spec.mountedHubFallbacks.first?.hubFallback.revision, WooshResources.robertaTokenizerRevision)
    }

    func testWooshFlowSpecUsesOriginalFlowCheckpointLayout() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: ModelResolver.ModelID.wooshFlow.rawValue))

        XCTAssertEqual(spec.category, .sfx)
        XCTAssertEqual(spec.hubFallback?.repoId, WooshResources.huggingFaceMirrorRepoId)
        XCTAssertEqual(spec.validationKind, .woosh)
        XCTAssertTrue(spec.hubFallback?.patterns.contains("checkpoints/Woosh-Flow/*") == true)
        XCTAssertTrue(spec.hubFallback?.patterns.contains("checkpoints/Woosh-DFlow/*") == false)
        XCTAssertEqual(spec.mountedHubFallbacks.first?.destinationPath, "checkpoints/TextConditionerA/tokenizer")
    }

    func testWooshDFlowRootValidationRequiresT2AComponentsAndTokenizer() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: ModelResolver.ModelID.wooshDFlow.rawValue))

        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalWooshDFlowRoot(at: root)
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))

        try FileManager.default.removeItem(
            at: root.appendingPathComponent("checkpoints/TextConditionerA/tokenizer/tokenizer.json")
        )
        try FileManager.default.removeItem(
            at: root.appendingPathComponent("checkpoints/TextConditionerA/tokenizer/vocab.json")
        )
        XCTAssertFalse(spec.isManagedRootComplete(root, fileManager: .default))
        XCTAssertEqual(
            spec.missingPaths(in: root, fileManager: .default).map(\.lastPathComponent),
            ["tokenizer.json"]
        )
    }

    func testWooshFlowRootValidationRequiresOriginalFlowComponents() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: ModelResolver.ModelID.wooshFlow.rawValue))

        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalWooshRoot(at: root, generatorComponent: "Woosh-Flow")
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))

        try FileManager.default.removeItem(
            at: root.appendingPathComponent("checkpoints/Woosh-Flow/weights.safetensors")
        )
        XCTAssertFalse(spec.isManagedRootComplete(root, fileManager: .default))
        XCTAssertEqual(
            spec.missingPaths(in: root, fileManager: .default).map(\.lastPathComponent),
            ["weights.safetensors"]
        )
    }

    func testWooshSynchformerSpecUsesCompanionSafetensorsSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: ModelResolver.ModelID.wooshSynchformer.rawValue))

        XCTAssertEqual(spec.category, .sfx)
        XCTAssertEqual(spec.hubFallback?.repoId, WooshResources.synchformerRepoId)
        XCTAssertEqual(spec.validationKind, .wooshSynchformer)
        XCTAssertEqual(spec.defaultCLICommands, ["sfx video generate"])
        XCTAssertEqual(spec.hubFallback?.patterns, [WooshResources.synchformerFilename])

        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertFalse(spec.isManagedRootComplete(root, fileManager: .default))
        XCTAssertEqual(
            spec.missingPaths(in: root, fileManager: .default).map(\.lastPathComponent),
            [WooshResources.synchformerFilename]
        )
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent(WooshResources.synchformerFilename).path,
            contents: Data()
        ))
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))
    }

    func testMMAudioSpecUsesPinnedNativeAssetsAndNonCommercialCheckpointSource() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.mmaudioLarge44kV2.rawValue)
        )

        XCTAssertEqual(spec.category, .sfx)
        XCTAssertEqual(spec.validationKind, .mmaudio)
        XCTAssertEqual(spec.hubFallback?.repoId, MMAudioResources.convertedWeightsRepoID)
        XCTAssertEqual(spec.hubFallback?.revision, MMAudioResources.convertedWeightsRevision)
        XCTAssertEqual(spec.defaultCLICommands, ["sfx generate", "sfx video generate"])
        XCTAssertEqual(spec.mountedHubFallbacks.map(\.destinationPath), ["clip", "bigvgan"])
        XCTAssertEqual(spec.upstreamRevision, MMAudioResources.upstreamRevision)
        XCTAssertEqual(MMAudioResources.architectureLicense, "MIT")
        XCTAssertEqual(MMAudioResources.checkpointLicense, "CC-BY-NC-4.0")
        XCTAssertEqual(
            MMAudioResources.clipModelLicense,
            "Apple Machine Learning Research Model License Agreement"
        )
        XCTAssertEqual(MMAudioResources.bigVGANLicense, "MIT")
    }

    func testMMAudioRootRequiresGeneratorConditionersCodecAndVocoder() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.mmaudioLarge44kV2.rawValue)
        )
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let clip = root.appendingPathComponent("clip", isDirectory: true)
        let bigvgan = root.appendingPathComponent("bigvgan", isDirectory: true)
        try FileManager.default.createDirectory(at: clip, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: bigvgan, withIntermediateDirectories: true)
        for file in [
            MMAudioResources.networkFilename,
            MMAudioResources.clipFilename,
            MMAudioResources.synchformerFilename,
            MMAudioResources.vaeFilename,
        ] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: root.appendingPathComponent(file).path,
                contents: Data()
            ))
        }
        XCTAssertTrue(FileManager.default.createFile(
            atPath: clip.appendingPathComponent("tokenizer.json").path,
            contents: Data("{}".utf8)
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: bigvgan.appendingPathComponent("config.json").path,
            contents: Data("{}".utf8)
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: bigvgan.appendingPathComponent(MMAudioResources.bigVGANPyTorchFilename).path,
            contents: Data()
        ))
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))

        try FileManager.default.removeItem(at: root.appendingPathComponent(MMAudioResources.vaeFilename))
        XCTAssertEqual(
            spec.missingPaths(in: root, fileManager: .default).map(\.lastPathComponent),
            [MMAudioResources.vaeFilename]
        )
    }

    func testLTX23MLXSpecUsesDgrauetSplitSource() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: ModelResolver.ModelID.ltxVideo23AVMLX.rawValue))

        XCTAssertEqual(spec.category, .video)
        XCTAssertEqual(spec.hubFallback?.repoId, "dgrauet/ltx-2.3-mlx")
        XCTAssertEqual(spec.hubFallback?.revision, "baa5f235ea04fd9c95899d751295c4fd825ee4e2")
        XCTAssertEqual(spec.upstreamRepoId, "dgrauet/ltx-2.3-mlx")
        XCTAssertEqual(spec.upstreamRevision, "baa5f235ea04fd9c95899d751295c4fd825ee4e2")
        XCTAssertEqual(spec.validationKind, .ltxVideo23MLX)
        XCTAssertEqual(spec.runtimeAutoDownloadAllowed, false)
        XCTAssertEqual(spec.companionModelIDs, [ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue])
        XCTAssertEqual(spec.usageRestriction?.terms.count, 2)
        XCTAssertTrue(spec.usageRestriction?.terms.contains { $0.license == "Gemma Terms of Use" } == true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("embedded_config.json"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("vocoder.safetensors"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("transformer-dev.safetensors"), false)
    }

    func testLTX23A2VidSpecUsesOnlyRequiredDevAndLoRAAssets() throws {
        let id = ModelResolver.ModelID.ltxVideo23A2VMLX.rawValue
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))

        XCTAssertEqual(spec.category, .video)
        XCTAssertEqual(spec.hubFallback?.repoId, "dgrauet/ltx-2.3-mlx")
        XCTAssertEqual(spec.hubFallback?.revision, "baa5f235ea04fd9c95899d751295c4fd825ee4e2")
        XCTAssertEqual(spec.validationKind, .ltxVideo23A2VMLX)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.companionModelIDs, [ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue])
        XCTAssertEqual(spec.hubFallback?.patterns.contains("transformer-dev.safetensors"), true)
        XCTAssertEqual(
            spec.hubFallback?.patterns.contains("ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
            true
        )
        XCTAssertEqual(spec.hubFallback?.patterns.contains("transformer-distilled.safetensors"), false)

        let manifest = MereRunModelManifest.template(
            for: .ltxVideo23A2VMLX,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertTrue(manifest.supports?.contains(.audioToVideoGeneration) == true)
        XCTAssertEqual(manifest.defaults?.steps, 30)
    }

    func testLTX23FullSpecUnifiesJointAVAndA2VidAssets() throws {
        let id = ModelResolver.ModelID.ltxVideo23FullMLX.rawValue
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))

        XCTAssertEqual(spec.validationKind, .ltxVideo23FullMLX)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("transformer-dev.safetensors"), true)
        XCTAssertEqual(
            spec.hubFallback?.patterns.contains("ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
            true
        )
        XCTAssertEqual(spec.hubFallback?.patterns.contains("vocoder.safetensors"), true)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("transformer-distilled.safetensors"), false)
        XCTAssertEqual(spec.resolutionFallbackIDs, [ModelResolver.ModelID.ltxVideo23A2VMLX.rawValue])

        let legacySpec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.ltxVideo23A2VMLX.rawValue)
        )
        XCTAssertEqual(legacySpec.resolutionFallbackIDs, [id])

        let manifest = MereRunModelManifest.template(
            for: .ltxVideo23FullMLX,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(manifest.variant, .base)
        XCTAssertTrue(manifest.supports?.contains(.videoGeneration) == true)
        XCTAssertTrue(manifest.supports?.contains(.audioToVideoGeneration) == true)
    }

    func testLTX25SpecPinsSelfContainedSawfwairRelease() throws {
        let id = ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))

        XCTAssertEqual(id, LTX25Resources.modelID)
        XCTAssertEqual(spec.category, .video)
        XCTAssertEqual(spec.installShape, .structuredRoot)
        XCTAssertEqual(spec.validationKind, .ltxVideo25)
        XCTAssertEqual(spec.hubFallback?.repoId, LTX25Resources.managedRepository)
        XCTAssertEqual(spec.hubFallback?.revision, LTX25Resources.managedRevision)
        XCTAssertEqual(spec.hubFallback?.patterns, LTX25Resources.snapshotPatterns)
        XCTAssertEqual(spec.upstreamRepoId, LTX25Resources.managedRepository)
        XCTAssertEqual(spec.upstreamRevision, LTX25Resources.managedRevision)
        XCTAssertEqual(spec.estimatedDownloadBytes, LTX25Resources.estimatedDownloadBytes)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertTrue(spec.companionModelIDs.isEmpty)
        XCTAssertEqual(spec.usageRestriction?.terms.count, 2)
        XCTAssertTrue(spec.usageRestriction?.terms.contains { $0.component == "Gemma 4 text encoder" } == true)

        let manifest = MereRunModelManifest.template(
            for: .ltxVideo25DistilledBF16,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(manifest.engine, .ltxVideo)
        XCTAssertEqual(manifest.family, .video)
        XCTAssertEqual(manifest.variant, .distilled)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(LTX25Resources.managedRepository)@\(LTX25Resources.managedRevision)"
        )
    }

    func testLTX25FullSpecPinsEveryOfficialParityComponent() throws {
        let id = ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))

        XCTAssertEqual(id, LTX25Resources.fullModelID)
        XCTAssertEqual(spec.category, .video)
        XCTAssertEqual(spec.installShape, .structuredRoot)
        XCTAssertEqual(spec.validationKind, .ltxVideo25)
        XCTAssertEqual(spec.hubFallback?.repoId, LTX25Resources.sourceRepository)
        XCTAssertEqual(spec.hubFallback?.revision, LTX25Resources.sourceRevision)
        XCTAssertEqual(spec.hubFallback?.patterns, LTX25Resources.fullSnapshotPatterns)
        XCTAssertEqual(spec.estimatedDownloadBytes, LTX25Resources.fullEstimatedDownloadBytes)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertTrue(spec.companionModelIDs.isEmpty)
        XCTAssertEqual(spec.usageRestriction?.terms.count, 2)

        let manifest = MereRunModelManifest.template(
            for: .ltxVideo25FullBF16,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(manifest.engine, .ltxVideo)
        XCTAssertEqual(manifest.family, .video)
        XCTAssertEqual(manifest.variant, .base)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(manifest.defaults?.steps, 30)
        XCTAssertTrue(manifest.supports?.contains(.videoGeneration) == true)
        XCTAssertTrue(manifest.supports?.contains(.audioToVideoGeneration) == true)
    }

    func testSCAIL2SpecPinsImmutableSawfwairRelease() throws {
        let id = ModelResolver.ModelID.scail2Video14BMLX.rawValue
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))

        XCTAssertEqual(spec.category, .video)
        XCTAssertEqual(spec.validationKind, .scail2MLX)
        XCTAssertEqual(spec.upstreamRepoId, SCAIL2Resources.upstreamRepoID)
        XCTAssertEqual(spec.upstreamRevision, SCAIL2Resources.upstreamRevision)
        XCTAssertEqual(spec.hubFallback?.repoId, SCAIL2Resources.managedRepoID)
        XCTAssertEqual(spec.hubFallback?.revision, SCAIL2Resources.managedRevision)
        XCTAssertEqual(spec.hubFallback?.patterns, SCAIL2Resources.snapshotPatterns)
        XCTAssertTrue(spec.canBePulledWithoutConfiguration)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.estimatedDownloadBytes, 46_648_000_000)
        XCTAssertEqual(spec.defaultCLICommands, ["video animate"])

        let manifest = MereRunModelManifest.template(
            for: .scail2Video14BMLX,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(manifest.engine, .wanVideo)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(manifest.defaults?.steps, 40)
        XCTAssertEqual(manifest.defaults?.cfg, 5)
        XCTAssertEqual(manifest.defaults?.sigmaShift, 3)
        XCTAssertTrue(manifest.supports?.contains(.videoGeneration) == true)
    }

    func testCosmos3EdgeSpecPinsCompleteOfficialOmnimodalSnapshot() throws {
        let id = ModelResolver.ModelID.cosmos3EdgeMLX.rawValue
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))

        XCTAssertEqual(spec.category, .video)
        XCTAssertEqual(spec.installShape, .structuredRoot)
        XCTAssertEqual(spec.validationKind, .cosmos3EdgeMLX)
        XCTAssertEqual(spec.upstreamRepoId, Cosmos3Resources.officialRepoID)
        XCTAssertEqual(spec.upstreamRevision, Cosmos3Resources.officialRevision)
        XCTAssertEqual(spec.hubFallback?.repoId, Cosmos3Resources.officialRepoID)
        XCTAssertEqual(spec.hubFallback?.revision, Cosmos3Resources.officialRevision)
        XCTAssertEqual(spec.hubFallback?.patterns, Cosmos3Resources.snapshotPatterns)
        XCTAssertTrue(spec.canBePulledWithoutConfiguration)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.estimatedDownloadBytes, 9_200_000_000)
        XCTAssertNil(spec.usageRestriction)
        XCTAssertEqual(spec.defaultCLICommands, [
            "video cosmos3",
            "video cosmos3 --mode reasoner",
            "world serve --backend cosmos3",
        ])
        XCTAssertTrue(Cosmos3Resources.snapshotPatterns.contains("transformer/*"))
        XCTAssertTrue(Cosmos3Resources.snapshotPatterns.contains("vae/*"))
        XCTAssertTrue(Cosmos3Resources.snapshotPatterns.contains("vision_encoder/*"))
        XCTAssertTrue(Cosmos3Resources.snapshotPatterns.contains("model-*.safetensors"))
        XCTAssertTrue(Cosmos3Resources.snapshotPatterns.contains("LICENSE*"))

        let manifest = MereRunModelManifest.template(
            for: .cosmos3EdgeMLX,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(manifest.engine, .cosmos3Edge)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(manifest.defaults?.steps, 35)
        XCTAssertEqual(manifest.defaults?.cfg, 6)
        XCTAssertEqual(manifest.defaults?.sigmaShift, 10)
        XCTAssertTrue(manifest.supports?.contains(.videoGeneration) == true)
        XCTAssertTrue(manifest.supports?.contains(.actionGeneration) == true)
        XCTAssertTrue(manifest.supports?.contains(.worldSimulation) == true)
        XCTAssertTrue(manifest.supports?.contains(.visionReasoning) == true)
    }

    func testLTX23CompanionGemma3TextEncoderSpecIsKnownButHidden() throws {
        let companionID = ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: companionID))

        XCTAssertFalse(ManagedModelCatalog.allModelIDs.contains(companionID))
        XCTAssertEqual(spec.category, .textChat)
        XCTAssertEqual(spec.hubFallback?.repoId, "mlx-community/gemma-3-12b-it-4bit")
        XCTAssertEqual(spec.hubFallback?.revision, "14d891e009084901c434304fe93a86fd9013e84c")
        XCTAssertEqual(spec.upstreamRevision, "14d891e009084901c434304fe93a86fd9013e84c")
        XCTAssertEqual(spec.validationKind, .hfTextChat)
        XCTAssertEqual(spec.runtimeAutoDownloadAllowed, false)
        XCTAssertNil(spec.usageRestriction)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("README.md"), true)
    }

    func testHFTextRootValidationRequiresShardsNamedByIndex() throws {
        let companionID = ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: companionID))
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for file in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: root.appendingPathComponent(file).path,
                contents: Data("{}".utf8)
            ))
        }
        let index = """
        {
          "metadata": {"total_size": 1},
          "weight_map": {
            "language_model.model.embed_tokens.weight": "model-00001-of-00002.safetensors"
          }
        }
        """
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("model.safetensors.index.json").path,
            contents: Data(index.utf8)
        ))

        XCTAssertEqual(
            spec.missingPaths(in: root, fileManager: .default).map(\.lastPathComponent),
            ["model-00001-of-00002.safetensors"]
        )
    }

    func testLTX23MLXRootValidationRequiresSplitComponents() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: ModelResolver.ModelID.ltxVideo23AVMLX.rawValue))

        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalLTX23MLXRoot(at: root)
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))

        try FileManager.default.removeItem(at: root.appendingPathComponent("vocoder.safetensors"))
        XCTAssertFalse(spec.isManagedRootComplete(root, fileManager: .default))
        XCTAssertEqual(
            spec.missingPaths(in: root, fileManager: .default).map(\.lastPathComponent),
            ["vocoder.safetensors"]
        )
    }

    func testLTX23A2VidRootValidationRequiresDevTransformerAndLoRA() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.ltxVideo23A2VMLX.rawValue)
        )
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalLTX23A2VidRoot(at: root)
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))

        let lora = root.appendingPathComponent("ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
        try FileManager.default.removeItem(at: lora)
        XCTAssertEqual(
            spec.missingPaths(in: root, fileManager: .default).map(\.lastPathComponent),
            ["ltx-2.3-22b-distilled-lora-384-1.1.safetensors"]
        )
    }

    func testLTX23FullRootValidationAlsoRequiresVocoder() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.ltxVideo23FullMLX.rawValue)
        )
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalLTX23FullRoot(at: root)
        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))

        try FileManager.default.removeItem(at: root.appendingPathComponent("vocoder.safetensors"))
        XCTAssertEqual(
            spec.missingPaths(in: root, fileManager: .default).map(\.lastPathComponent),
            ["vocoder.safetensors"]
        )
    }

    func testLTX25RootValidationRequiresEveryNativeRuntimeArtifact() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue)
        )
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalLTX25Root(at: root)

        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))
        let missing = root.appendingPathComponent(LTX25Resources.videoVAERelativePath)
        try FileManager.default.removeItem(at: missing)
        XCTAssertEqual(spec.missingPaths(in: root, fileManager: .default), [missing])
        XCTAssertTrue(spec.validationMessages(in: root).contains { $0.contains("Missing required LTX 2.5 file") })
    }

    func testLTX25FullRootValidationRequiresDevLoRAAndParityAssets() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: ModelResolver.ModelID.ltxVideo25FullBF16.rawValue)
        )
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeMinimalLTX25FullRoot(at: root)

        XCTAssertTrue(spec.isManagedRootComplete(root, fileManager: .default))
        let missing = root.appendingPathComponent(LTX25Resources.distilledLoRARelativePath)
        try FileManager.default.removeItem(at: missing)
        XCTAssertEqual(spec.missingPaths(in: root, fileManager: .default), [missing])
        XCTAssertTrue(spec.validationMessages(in: root).contains { $0.contains("Missing required LTX 2.5 file") })
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-managed-model-catalog-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func managedModelCatalogRows(in markdown: String) throws -> [ManagedModelCatalogDocRow] {
        let startMarker = "<!-- managed-model-catalog:start -->"
        let endMarker = "<!-- managed-model-catalog:end -->"
        let startRange = try XCTUnwrap(markdown.range(of: startMarker))
        let tableStart = startRange.upperBound..<markdown.endIndex
        let endRange = try XCTUnwrap(markdown.range(of: endMarker, range: tableStart))
        let tableMarkdown = markdown[startRange.upperBound..<endRange.lowerBound]

        return tableMarkdown.split(separator: "\n").compactMap { line in
            managedModelCatalogRow(from: String(line))
        }
    }

    private func managedModelCatalogRow(from line: String) -> ManagedModelCatalogDocRow? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("| `") else {
            return nil
        }

        let columns = trimmed.split(separator: "|", omittingEmptySubsequences: false)
        guard columns.count >= 4 else {
            return nil
        }

        return ManagedModelCatalogDocRow(
            category: markdownCodeValue(String(columns[1])),
            id: markdownCodeValue(String(columns[2]))
        )
    }

    private func markdownCodeValue(_ value: String) -> String {
        value.trimmingCharacters(in: .whitespaces)
            .trimmingCharacters(in: CharacterSet(charactersIn: "`"))
    }

    private func writeMinimalLTX23MLXRoot(at root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try MereRunModelManifest.template(
            for: .ltxVideo23AVMLX,
            createdAt: Date(timeIntervalSince1970: 0)
        ).write(to: root)

        for file in [
            "config.json",
            "embedded_config.json",
            "split_model.json",
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
            let contents = file.hasSuffix(".json")
                ? Data(#"{"model_version":"2.3.0"}"#.utf8)
                : Data()
            XCTAssertTrue(FileManager.default.createFile(
                atPath: root.appendingPathComponent(file).path,
                contents: contents
            ))
        }
    }

    private func writeMinimalLTX25Root(at root: URL) throws {
        try MereRunModelManifest.template(
            for: .ltxVideo25DistilledBF16,
            createdAt: Date(timeIntervalSince1970: 0)
        ).write(to: root)
        for relativePath in LTX25Resources.requiredRelativePaths {
            let url = root.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            XCTAssertTrue(FileManager.default.createFile(atPath: url.path, contents: Data()))
        }
        let textEncoder = root.appendingPathComponent(LTX25Resources.textEncoderRelativePath)
        try FileManager.default.createDirectory(
            at: textEncoder.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        XCTAssertTrue(FileManager.default.createFile(atPath: textEncoder.path, contents: Data()))
    }

    private func writeMinimalLTX25FullRoot(at root: URL) throws {
        try MereRunModelManifest.template(
            for: .ltxVideo25FullBF16,
            createdAt: Date(timeIntervalSince1970: 0)
        ).write(to: root)
        for relativePath in LTX25Resources.fullRequiredRelativePaths {
            let url = root.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            XCTAssertTrue(FileManager.default.createFile(atPath: url.path, contents: Data()))
        }
    }

    private func writeMinimalLTX23A2VidRoot(at root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try MereRunModelManifest.template(
            for: .ltxVideo23A2VMLX,
            createdAt: Date(timeIntervalSince1970: 0)
        ).write(to: root)
        for file in [
            "config.json",
            "embedded_config.json",
            "split_model.json",
            "connector.safetensors",
            "transformer-dev.safetensors",
            "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "spatial_upscaler_x2_v1_1.safetensors",
            "spatial_upscaler_x2_v1_1_config.json",
        ] {
            let contents = file.hasSuffix(".json")
                ? Data(#"{"model_version":"2.3.0"}"#.utf8)
                : Data()
            XCTAssertTrue(FileManager.default.createFile(
                atPath: root.appendingPathComponent(file).path,
                contents: contents
            ))
        }
    }

    private func writeMinimalLTX23FullRoot(at root: URL) throws {
        try writeMinimalLTX23A2VidRoot(at: root)
        try MereRunModelManifest.template(
            for: .ltxVideo23FullMLX,
            createdAt: Date(timeIntervalSince1970: 0)
        ).write(to: root)
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("vocoder.safetensors").path,
            contents: Data()
        ))
    }

    private func writeConfigOnlyDiffusersImageRoot(at root: URL, id: ModelResolver.ModelID) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("model_index.json").path,
            contents: Data("{}".utf8)
        ))

        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        let textEncoder = root.appendingPathComponent("text_encoder", isDirectory: true)
        let transformer = root.appendingPathComponent("transformer", isDirectory: true)
        let vae = root.appendingPathComponent("vae", isDirectory: true)
        let scheduler = root.appendingPathComponent("scheduler", isDirectory: true)
        for directory in [tokenizer, textEncoder, transformer, vae, scheduler] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }

        XCTAssertTrue(FileManager.default.createFile(
            atPath: tokenizer.appendingPathComponent("tokenizer_config.json").path,
            contents: Data("{}".utf8)
        ))
        for directory in [textEncoder, transformer, vae] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: directory.appendingPathComponent("config.json").path,
                contents: Data("{}".utf8)
            ))
        }
        XCTAssertTrue(FileManager.default.createFile(
            atPath: scheduler.appendingPathComponent("scheduler_config.json").path,
            contents: Data("{}".utf8)
        ))
    }

    private func writeMinimalKrea2Root(at root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try MereRunModelManifest.template(for: .krea2Turbo, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("model_index.json").path,
            contents: Data("{}".utf8)
        ))

        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        let textEncoder = root.appendingPathComponent("text_encoder", isDirectory: true)
        let transformer = root.appendingPathComponent("transformer", isDirectory: true)
        let vae = root.appendingPathComponent("vae", isDirectory: true)
        let scheduler = root.appendingPathComponent("scheduler", isDirectory: true)
        for directory in [tokenizer, textEncoder, transformer, vae, scheduler] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }

        for file in [
            tokenizer.appendingPathComponent("tokenizer.json"),
            tokenizer.appendingPathComponent("tokenizer_config.json"),
            textEncoder.appendingPathComponent("config.json"),
            textEncoder.appendingPathComponent("model.safetensors"),
            transformer.appendingPathComponent("config.json"),
            transformer.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json"),
            transformer.appendingPathComponent("diffusion_pytorch_model-00001-of-00003.safetensors"),
            vae.appendingPathComponent("config.json"),
            vae.appendingPathComponent("diffusion_pytorch_model.safetensors"),
            scheduler.appendingPathComponent("scheduler_config.json"),
        ] {
            let contents = file.pathExtension == "json" ? Data("{}".utf8) : Data()
            XCTAssertTrue(FileManager.default.createFile(atPath: file.path, contents: contents))
        }
    }

    private func writeMinimalACEStepRoot(at root: URL, turboSubdirectory: String) throws {
        for subdirectory in [turboSubdirectory, "vae", "Qwen3-Embedding-0.6B"] {
            try FileManager.default.createDirectory(
                at: root.appendingPathComponent(subdirectory, isDirectory: true),
                withIntermediateDirectories: true
            )
        }
    }

    private func writeMinimalACEStepLMRoot(at root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for file in [
            "config.json",
            "tokenizer_config.json",
            "added_tokens.json",
            "tokenizer.json",
            "model.safetensors.index.json",
        ] {
            XCTAssertTrue(FileManager.default.createFile(atPath: root.appendingPathComponent(file).path, contents: Data()))
        }
    }

    private func writeMinimalMagentaRT2Root(at root: URL, modelName: String) throws {
        let modelDir = root.appendingPathComponent("models/\(modelName)", isDirectory: true)
        let musicCoCa = root.appendingPathComponent("resources/musiccoca", isDirectory: true)
        let spectrostream = root.appendingPathComponent("resources/spectrostream", isDirectory: true)
        for directory in [modelDir, musicCoCa, spectrostream] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        for file in ["\(modelName).mlxfn", "\(modelName)_state.safetensors"] {
            XCTAssertTrue(FileManager.default.createFile(atPath: modelDir.appendingPathComponent(file).path, contents: Data()))
        }
        for file in [
            "audio_preprocessor.tflite",
            "mapper.tflite",
            "music_encoder.tflite",
            "pretrained_vector_quantizer.tflite",
            "spm.model",
            "text_encoder.tflite",
        ] {
            XCTAssertTrue(FileManager.default.createFile(atPath: musicCoCa.appendingPathComponent(file).path, contents: Data()))
        }
        for file in [
            "decoder.safetensors",
            "encoder.safetensors",
            "quantizer.safetensors",
            "spectrostream_encoder.mlxfn",
        ] {
            XCTAssertTrue(FileManager.default.createFile(atPath: spectrostream.appendingPathComponent(file).path, contents: Data()))
        }
    }

    private func writeMinimalWooshDFlowRoot(at root: URL) throws {
        try writeMinimalWooshRoot(at: root, generatorComponent: "Woosh-DFlow")
    }

    private func writeMinimalWooshRoot(at root: URL, generatorComponent: String) throws {
        for component in [generatorComponent, "Woosh-AE", "TextConditionerA"] {
            let dir = root.appendingPathComponent("checkpoints/\(component)", isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            XCTAssertTrue(FileManager.default.createFile(atPath: dir.appendingPathComponent("config.yaml").path, contents: Data()))
            XCTAssertTrue(FileManager.default.createFile(atPath: dir.appendingPathComponent("weights.safetensors").path, contents: Data()))
        }
        let tokenizer = root.appendingPathComponent("checkpoints/TextConditionerA/tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        for file in ["tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"] {
            XCTAssertTrue(FileManager.default.createFile(atPath: tokenizer.appendingPathComponent(file).path, contents: Data()))
        }
    }

    private func writeMinimalQwen3TTSRootWithoutSpeechTokenizer(
        at root: URL,
        id: ModelResolver.ModelID
    ) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        let files = [
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "tokenizer_config.json",
            "vocab.json",
        ]

        for file in files {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: root.appendingPathComponent(file).path,
                contents: Data("{}".utf8)
            ))
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

    private func writeMinimalIdeogram4(at root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try MereRunModelManifest.template(
            for: .ideogram4SDNQUInt4,
            createdAt: Date(timeIntervalSince1970: 0)
        ).write(to: root)

        for file in ["model_index.json", "quantization_manifest.json"] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: root.appendingPathComponent(file).path,
                contents: Data("{}".utf8)
            ))
        }

        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        let textEncoder = root.appendingPathComponent("text_encoder", isDirectory: true)
        let transformer = root.appendingPathComponent("transformer", isDirectory: true)
        let unconditional = root.appendingPathComponent("unconditional_transformer", isDirectory: true)
        let vae = root.appendingPathComponent("vae", isDirectory: true)
        let scheduler = root.appendingPathComponent("scheduler", isDirectory: true)

        for dir in [tokenizer, textEncoder, transformer, unconditional, vae, scheduler] {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }

        XCTAssertTrue(FileManager.default.createFile(
            atPath: tokenizer.appendingPathComponent("tokenizer_config.json").path,
            contents: Data("{}".utf8)
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: textEncoder.appendingPathComponent("config.json").path,
            contents: Data(#"{"model_type":"qwen3_vl_text"}"#.utf8)
        ))
        for component in [transformer, unconditional, vae] {
            XCTAssertTrue(FileManager.default.createFile(
                atPath: component.appendingPathComponent("config.json").path,
                contents: Data("{}".utf8)
            ))
            XCTAssertTrue(FileManager.default.createFile(
                atPath: component.appendingPathComponent("diffusion_pytorch_model.safetensors").path,
                contents: Data()
            ))
        }
        XCTAssertTrue(FileManager.default.createFile(
            atPath: scheduler.appendingPathComponent("scheduler_config.json").path,
            contents: Data("{}".utf8)
        ))
    }
}

private struct ManagedModelCatalogDocRow: Equatable {
    let category: String
    let id: String
}
