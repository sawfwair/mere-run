import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class NemotronOmniTests: MereRunCoreTestCase {
    func testExpertPackCapacityProbeUsesPortableFileSystemAttributes() throws {
        let capacity = try NemotronOmniExpertPack.availableCapacity(
            at: FileManager.default.temporaryDirectory
        )

        XCTAssertGreaterThan(capacity, 0)
    }

    func testPublishedExpertPackTakesPrecedenceOverLegacyCache() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "nemotron-published-pack-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let publishedURL = root.appendingPathComponent(
            NemotronOmniExpertPack.publishedFilename
        )
        let tensors = Dictionary(uniqueKeysWithValues: (0..<46).map {
            ("tensor.\($0)", MLXArray([UInt8($0)]))
        })
        try MLX.save(
            arrays: tensors,
            metadata: [
                "format": NemotronOmniExpertPack.format,
                "source_revision": NemotronOmniResources.upstreamRevision,
                "payload_bytes": String(NemotronOmniResources.packedExpertWeightBytes),
            ],
            url: publishedURL
        )

        XCTAssertEqual(
            NemotronOmniExpertPack.optimizedURLIfValid(rootURL: root),
            publishedURL
        )
    }

    func testNativeRepositoryAndLocalRootAreRecognized() {
        XCTAssertTrue(NemotronOmniResources.handles(
            modelSpec: NemotronOmniResources.nativeRepoID
        ))
        XCTAssertTrue(NemotronOmniResources.handles(
            modelSpec: "/models/\(NemotronOmniResources.nativeRepoID.split(separator: "/").last!)"
        ))
    }

    func testPinnedOmniConfigDecodesAllThreeMediaTowers() throws {
        let config = try JSONDecoder().decode(
            NemotronOmniConfig.self,
            from: Data(Self.configJSON.utf8)
        )

        XCTAssertEqual(config.maximumSequenceLength, 131_072)
        XCTAssertEqual(config.language.hiddenSize, 2_688)
        XCTAssertEqual(config.language.layerBlockTypes.count, 52)
        XCTAssertEqual(config.language.layerBlockTypes.filter { $0 == "mamba" }.count, 23)
        XCTAssertEqual(config.language.layerBlockTypes.filter { $0 == "moe" }.count, 23)
        XCTAssertEqual(config.language.layerBlockTypes.filter { $0 == "attention" }.count, 6)
        XCTAssertEqual(config.vision.version, "c-radio_v4-h")
        XCTAssertEqual(config.vision.videoTemporalPatchSize, 2)
        XCTAssertEqual(config.sound.modelType, "parakeet")
        XCTAssertEqual(config.sound.numHiddenLayers, 24)
        XCTAssertEqual(config.sound.subsamplingConvChannels, 256)
        XCTAssertEqual(config.sound.numMelBins, 128)
    }

    func testPlaceholderCountsMatchPinnedProcessorGeometry() {
        XCTAssertEqual(
            NemotronOmniPlaceholderPlanner.imageTokenCount(sourcePatchCount: 1_024),
            256
        )
        XCTAssertEqual(
            NemotronOmniPlaceholderPlanner.audioTokenCount(sampleCount: 16_000),
            13
        )
    }

    func testVideoPromptUsesOfficialTubeletTimestampLabels() {
        let video = NemotronOmniPreparedVideo(
            pixelValues: MLX.zeros([4, 16, 16, 3]),
            frameCount: 4,
            tokensPerTubelet: 2,
            frameLabels: [
                " sampled at 0.00 seconds",
                " sampled at 1.25 seconds",
                " sampled at 2.50 seconds",
                " sampled at 3.75 seconds",
            ]
        )

        XCTAssertEqual(
            video.promptPrefix,
            "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 1.25 seconds: <image>\n"
                + "Frame 3 sampled at 2.50 seconds and frame 4 sampled at 3.75 seconds: <image>\n"
        )
    }

    func testParakeetFrontendMatchesReferenceFeatureSamples() throws {
        let sampleRate = 16_000
        var state: UInt32 = 0x1234_5678
        let samples = (0..<sampleRate).map { _ -> Float in
            state = 1_664_525 &* state &+ 1_013_904_223
            return (Float(state >> 8) / 16_777_216 - 0.5) * 0.4
        }
        let features = try NemotronOmniAudioProcessor.logMelSpectrogram(samples: samples)
            .asType(.float32)
            .asArray(Float.self)
        let expected: [(row: Int, column: Int, value: Float)] = [
            (0, 0, -2.0018),
            (0, 20, -0.1479),
            (0, 60, 0.2351),
            (1, 0, 0.2397),
            (50, 20, -0.2435),
            (99, 127, 0.9031),
            (100, 0, 0),
        ]
        for sample in expected {
            XCTAssertEqual(
                features[sample.row * 128 + sample.column],
                sample.value,
                accuracy: 0.08,
                "feature[\(sample.row),\(sample.column)]"
            )
        }
    }

    func testDynamicImagePatchGeometryAndMediaExpansion() throws {
        let imageGrid = NemotronOmniImageProcessor.imagePatchGrid(
            width: 1_536,
            height: 864
        )
        XCTAssertEqual(imageGrid.width, 96)
        XCTAssertEqual(imageGrid.height, 54)
        let videoGrid = NemotronOmniImageProcessor.videoPatchGrid(
            width: 1_920,
            height: 1_080
        )
        XCTAssertEqual(videoGrid.width, 42)
        XCTAssertEqual(videoGrid.height, 24)
        XCTAssertEqual(
            try NemotronOmniGenerator.expandImagePlaceholders(
                [7, 18, 8],
                tokenCounts: [3],
                imageTokenID: 18
            ),
            [7, 19, 18, 18, 18, 20, 8]
        )
        XCTAssertEqual(
            try NemotronOmniGenerator.expandAudioPlaceholders(
                [7, 27, 8],
                tokenCounts: [2],
                soundTokenID: 27
            ),
            [7, 28, 27, 27, 29, 8]
        )
    }

    func testMoEPrefillPlanBoundsWatchdogSafeTokenBatches() {
        XCTAssertEqual(
            NemotronOmniMoE.prefillRanges(tokenCount: 10),
            [0..<4, 4..<8, 8..<10]
        )
        XCTAssertEqual(NemotronOmniMoE.prefillRanges(tokenCount: 1), [0..<1])
    }

    func testBF16ExpertWeightKeyDecodesExactCheckpointNamespace() throws {
        let key = try XCTUnwrap(NemotronOmniExpertWeightKey(
            checkpointKey:
                "language_model.backbone.layers.51.mixer.experts.127.down_proj.weight"
        ))
        XCTAssertEqual(key.layer, 51)
        XCTAssertEqual(key.expert, 127)
        XCTAssertEqual(key.projection, .down)
        XCTAssertNil(NemotronOmniExpertWeightKey(
            checkpointKey: "language_model.backbone.layers.51.mixer.experts.weight"
        ))
    }

    func testPinnedConfigRejectsArchitectureDrift() {
        let drifted = Self.configJSON.replacingOccurrences(
            of: "\"max_sequence_length\":131072",
            with: "\"max_sequence_length\":65536"
        )
        XCTAssertThrowsError(try JSONDecoder().decode(
            NemotronOmniConfig.self,
            from: Data(drifted.utf8)
        ))
    }

    func testPinnedPreprocessorContractDecodesAndRejectsDrift() throws {
        let config = try JSONDecoder().decode(
            NemotronOmniPreprocessorConfig.self,
            from: Data(Self.preprocessorJSON.utf8)
        )
        XCTAssertEqual(config.normalizationMean.count, 3)
        XCTAssertEqual(config.maxNumPatches, 13_312)

        let drifted = Self.preprocessorJSON.replacingOccurrences(
            of: "\"patch_size\":16",
            with: "\"patch_size\":14"
        )
        XCTAssertThrowsError(try JSONDecoder().decode(
            NemotronOmniPreprocessorConfig.self,
            from: Data(drifted.utf8)
        ))
    }

    func testCatalogDeclaresNativeOmniRuntimeAndInputs() throws {
        let spec = try XCTUnwrap(
            ManagedModelCatalog.spec(for: NemotronOmniResources.modelID)
        )
        let profile = try XCTUnwrap(spec.apiProfile)

        XCTAssertEqual(spec.category, .omniChat)
        XCTAssertEqual(spec.validationKind, .nemotronOmni)
        XCTAssertEqual(spec.hubFallback?.repoId, NemotronOmniResources.nativeRepoID)
        XCTAssertEqual(spec.hubFallback?.revision, NemotronOmniResources.nativeRevision)
        XCTAssertEqual(spec.upstreamRepoId, NemotronOmniResources.upstreamRepoID)
        XCTAssertEqual(spec.upstreamRevision, NemotronOmniResources.upstreamRevision)
        XCTAssertEqual(
            spec.estimatedDownloadBytes,
            NemotronOmniResources.estimatedDownloadBytes
        )
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.defaultRuntimeServingEngine, .textChatNemotronOmni)
        XCTAssertEqual(profile.inputModalities, [.text, .image, .audio, .video])
        XCTAssertEqual(profile.outputModalities, [.text])
        XCTAssertEqual(profile.contextWindow, 131_072)
        XCTAssertEqual(profile.maximumOutputTokens, 20_480)
        XCTAssertTrue(profile.reasoning)
        XCTAssertTrue(profile.toolCall)
        XCTAssertFalse(profile.structuredOutput)
        XCTAssertTrue(
            ManagedModelCapabilityCatalog.support(
                for: spec,
                on: MereRunMachineProfile(
                    physicalMemoryBytes: 128 * 1_073_741_824,
                    processorName: "Test Mac",
                    isAppleSiliconMac: true,
                    isLinux: false
                )
            ).isSupported
        )
        XCTAssertEqual(
            spec.usageRestriction?.terms.first?.license,
            "NVIDIA Open Model Agreement"
        )
    }

    func testManifestRetainsOmniCapabilitiesAndPinnedProvenance() {
        let manifest = MereRunModelManifest.template(
            for: .nemotron3NanoOmni30BA3BBF16,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(manifest.engine, .nemotronOmni)
        XCTAssertEqual(manifest.family, .nemotron)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(
            Set(manifest.supports ?? []),
            Set([
                .chat,
                .codeGeneration,
                .visionChat,
                .visionOCR,
                .audioUnderstanding,
                .videoUnderstanding,
                .documentUnderstanding,
            ])
        )
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(NemotronOmniResources.upstreamRepoID)@\(NemotronOmniResources.upstreamRevision)"
        )
    }

    func testValidationRequiresEveryIndexedShard() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        for filename in [
            "README.md",
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ] {
            try Data().write(to: root.appendingPathComponent(filename))
        }
        try Data("""
        {"metadata":{"total_size":2},"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}
        """.utf8).write(to: root.appendingPathComponent("model.safetensors.index.json"))
        try Data([0]).write(to: root.appendingPathComponent("model-00001-of-00002.safetensors"))

        XCTAssertEqual(
            NemotronOmniResources.missingTargetFiles(rootURL: root).map(\.lastPathComponent),
            ["model-00002-of-00002.safetensors"]
        )
    }

    func testStructuralValidationPinsConfigProcessorAndShardIndex() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        for filename in [
            "README.md",
            "chat_template.jinja",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ] {
            try Data().write(to: root.appendingPathComponent(filename))
        }
        try Data(Self.configJSON.utf8).write(to: root.appendingPathComponent("config.json"))
        try Data(Self.preprocessorJSON.utf8).write(
            to: root.appendingPathComponent("preprocessor_config.json")
        )
        let shardEntries = try (1...NemotronOmniResources.checkpointShardCount).map { index in
            let filename = String(
                format: "model-%05d-of-%05d.safetensors",
                index,
                NemotronOmniResources.checkpointShardCount
            )
            try Data([0]).write(to: root.appendingPathComponent(filename))
            return "\"weight\(index)\":\"\(filename)\""
        }.joined(separator: ",")
        let pinnedIndex = """
        {"metadata":{"total_size":\(NemotronOmniResources.checkpointWeightBytes)},"weight_map":{\(shardEntries)}}
        """
        let indexURL = root.appendingPathComponent("model.safetensors.index.json")
        try Data(pinnedIndex.utf8).write(to: indexURL)

        XCTAssertEqual(NemotronOmniResources.validationMessages(rootURL: root), [])

        let driftedIndex = pinnedIndex.replacingOccurrences(
            of: String(NemotronOmniResources.checkpointWeightBytes),
            with: "1"
        )
        try Data(driftedIndex.utf8).write(to: indexURL)
        XCTAssertEqual(
            NemotronOmniResources.validationMessages(rootURL: root),
            [
                "Safetensors index does not match the pinned 17-shard, "
                    + "66031270520-byte checkpoint contract.",
            ]
        )
    }

    func testStructuralValidationAcceptsStandaloneNativeCheckpoint() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        for filename in NemotronOmniNativeCheckpoint.supportFilenames {
            try Data().write(to: root.appendingPathComponent(filename))
        }
        let manifest = NemotronOmniNativeCheckpointManifest(
            schemaVersion: 1,
            format: NemotronOmniNativeCheckpoint.format,
            sourceRepository: NemotronOmniResources.upstreamRepoID,
            sourceRevision: NemotronOmniResources.upstreamRevision,
            totalWeightBytes: NemotronOmniResources.checkpointWeightBytes,
            expertWeightBytes: NemotronOmniResources.packedExpertWeightBytes,
            nonExpertWeightBytes: NemotronOmniResources.nativeNonExpertWeightBytes,
            nonExpertShardCount: NemotronOmniResources.checkpointShardCount
        )
        try JSONEncoder().encode(manifest).write(
            to: NemotronOmniNativeCheckpoint.manifestURL(rootURL: root)
        )
        let expertArrays = Dictionary(uniqueKeysWithValues: (0..<46).map {
            ("tensor.\($0)", MLXArray([UInt8($0)]))
        })
        try MLX.save(
            arrays: expertArrays,
            metadata: [
                "format": NemotronOmniExpertPack.format,
                "source_revision": NemotronOmniResources.upstreamRevision,
                "payload_bytes": String(NemotronOmniResources.packedExpertWeightBytes),
            ],
            url: root.appendingPathComponent(NemotronOmniExpertPack.publishedFilename)
        )
        var weightMap: [String: String] = [:]
        for index in 1...NemotronOmniResources.checkpointShardCount {
            let filename = String(
                format: "model-%05d-of-%05d.safetensors",
                index,
                NemotronOmniResources.checkpointShardCount
            )
            weightMap["tensor.\(index)"] = filename
            try Data([0]).write(to: root.appendingPathComponent(filename))
        }
        let nativeIndex = HFSafetensorsIndex(
            metadata: .init(totalSize: NemotronOmniResources.nativeNonExpertWeightBytes),
            weightMap: weightMap
        )
        try JSONEncoder().encode(nativeIndex).write(
            to: root.appendingPathComponent("model.safetensors.index.json")
        )

        XCTAssertEqual(NemotronOmniResources.validationMessages(rootURL: root), [])
    }

    func testNativeExporterRejectsAnAlreadyNativeSource() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data().write(to: NemotronOmniNativeCheckpoint.manifestURL(rootURL: root))

        XCTAssertThrowsError(try NemotronOmniNativeCheckpoint.export(
            sourceRootURL: root,
            destinationRootURL: root.appendingPathComponent("output")
        )) { error in
            guard case NemotronOmniNativeCheckpointError.invalidSource(let messages) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(messages, ["Source is already a standalone native checkpoint."])
        }
    }

    func testOpenAIContentRoundTripsImageAudioAndVideoURLs() throws {
        let data = Data("""
        {
          "role":"user",
          "content":[
            {"type":"text","text":"Summarize the meeting."},
            {"type":"image_url","image_url":{"url":"file:///tmp/slide.png"}},
            {"type":"audio_url","audio_url":{"url":"file:///tmp/meeting.wav"}},
            {"type":"video_url","video_url":{"url":"file:///tmp/meeting.mp4"}}
          ]
        }
        """.utf8)
        let message = try JSONDecoder().decode(OpenAIChatMessage.self, from: data)

        XCTAssertEqual(message.content, "Summarize the meeting.")
        XCTAssertEqual(message.imageURLs, ["file:///tmp/slide.png"])
        XCTAssertEqual(message.audioURLs, ["file:///tmp/meeting.wav"])
        XCTAssertEqual(message.videoURLs, ["file:///tmp/meeting.mp4"])

        let roundTripped = try JSONDecoder().decode(
            OpenAIChatMessage.self,
            from: JSONEncoder().encode(message)
        )
        XCTAssertEqual(roundTripped.imageURLs, message.imageURLs)
        XCTAssertEqual(roundTripped.audioURLs, message.audioURLs)
        XCTAssertEqual(roundTripped.videoURLs, message.videoURLs)
    }

    private static let configJSON = """
    {
      "architectures":["NemotronH_Nano_Omni_Reasoning_V3"],
      "model_type":"NemotronH_Nano_Omni_Reasoning_V3",
      "max_sequence_length":131072,
      "img_context_token_id":18,
      "video_context_token_id":131081,
      "sound_context_token_id":27,
      "downsample_ratio":0.5,
      "projector_hidden_size":20480,
      "vit_hidden_size":1280,
      "video_pruning_rate":0.7,
      "vision_config":{
        "version":"c-radio_v4-h",
        "patch_size":16,
        "min_num_patches":1024,
        "max_num_patches":13312,
        "video_target_num_patches":1024,
        "video_temporal_patch_size":2,
        "separate_video_embedder":true
      },
      "sound_config":{
        "model_type":"parakeet",
        "hidden_size":1024,
        "num_attention_heads":8,
        "num_hidden_layers":24,
        "intermediate_size":4096,
        "conv_kernel_size":9,
        "subsampling_factor":8,
        "subsampling_conv_channels":256,
        "subsampling_conv_kernel_size":3,
        "subsampling_conv_stride":2,
        "num_mel_bins":128,
        "projection_hidden_size":4096,
        "sampling_rate":16000
      },
      "llm_config":{
        "model_type":"nemotron_h",
        "hidden_size":2688,
        "vocab_size":131072,
        "num_hidden_layers":52,
        "hybrid_override_pattern":"MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
        "max_position_embeddings":262144,
        "num_attention_heads":32,
        "num_key_value_heads":2,
        "head_dim":128,
        "norm_eps":0.00001,
        "mamba_head_dim":64,
        "mamba_num_heads":64,
        "ssm_state_size":128,
        "n_groups":8,
        "conv_kernel":4,
        "time_step_min":0.001,
        "time_step_max":0.1,
        "n_routed_experts":128,
        "n_shared_experts":1,
        "num_experts_per_tok":6,
        "moe_intermediate_size":1856,
        "moe_shared_expert_intermediate_size":3712,
        "routed_scaling_factor":2.5,
        "norm_topk_prob":true,
        "n_group":1,
        "topk_group":1,
        "eos_token_id":11
      }
    }
    """

    private static let preprocessorJSON = """
    {
      "patch_size":16,
      "downsample_ratio":0.5,
      "norm_mean":[0.48145466,0.4578275,0.40821073],
      "norm_std":[0.26862954,0.26130258,0.27577711],
      "min_num_patches":1024,
      "max_num_patches":13312
    }
    """
}
