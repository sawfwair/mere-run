import MLX
import XCTest
@testable import MereRunCore

final class MiniMaxMusic3Tests: MereRunCoreTestCase {
    func testInstalledConditionEncoderAndVocoderMatchUpstream() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_MINIMAX_MUSIC3_MODEL_ROOT"],
              let fixturePath = environment["MERERUN_MINIMAX_MUSIC3_PARITY_FIXTURE"]
        else {
            throw XCTSkip("set MiniMax Music 3 model and parity fixture paths to run component parity")
        }
        let resources = MiniMaxMusic3Resources(rootURL: URL(fileURLWithPath: root))
        let fixture = try MLX.loadArrays(url: URL(fileURLWithPath: fixturePath))

        let conditionEncoder = MiniMaxMusic3ConditionEncoder(
            configuration: try resources.loadConditionConfiguration()
        )
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.conditionEncoderURL
                .appendingPathComponent("diffusion_pytorch_model.safetensors.index.json"),
            singleURL: resources.conditionEncoderURL
                .appendingPathComponent("diffusion_pytorch_model.safetensors"),
            to: conditionEncoder,
            dtype: .bfloat16,
            verify: .noUnusedKeys,
            mapper: MiniMaxMusic3ConditionEncoder.mapWeight
        )
        let conditionOutput = conditionEncoder(try XCTUnwrap(fixture["condition_input"]).asType(.bfloat16))
        try assertUpstreamParity(
            conditionOutput,
            try XCTUnwrap(fixture["condition_output"]),
            minimumSNR: 30,
            component: "condition encoder"
        )

        let vocoder = MiniMaxMusic3Vocoder(configuration: try resources.loadVocoderConfiguration())
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.vocoderURL
                .appendingPathComponent("diffusion_pytorch_model.safetensors.index.json"),
            singleURL: resources.vocoderURL
                .appendingPathComponent("diffusion_pytorch_model.safetensors"),
            to: vocoder,
            dtype: .bfloat16,
            verify: .noUnusedKeys,
            mapper: MiniMaxMusic3Vocoder.mapWeight
        )
        let vocoderInput = try XCTUnwrap(fixture["vocoder_input"]).asType(.bfloat16)
        var hidden = vocoderInput.reshaped(2, 64, 2).transposed(0, 2, 1)
        hidden = vocoder.decoderInputProjection(hidden)
        try assertUpstreamParity(
            hidden,
            try XCTUnwrap(fixture["vocoder_dec"]).transposed(0, 2, 1),
            minimumSNR: 25,
            component: "vocoder input projection"
        )
        hidden = vocoder.inputConvolution(hidden)
        try assertUpstreamParity(
            hidden,
            try XCTUnwrap(fixture["vocoder_cin"]).transposed(0, 2, 1),
            minimumSNR: 25,
            component: "vocoder input convolution"
        )
        let block = vocoder.blocks[0]
        hidden = block.snake1(hidden)
        try assertUpstreamParity(
            hidden,
            try XCTUnwrap(fixture["vocoder_snake0"]).transposed(0, 2, 1),
            minimumSNR: 25,
            component: "vocoder block 0 snake"
        )
        hidden = block.transposedConvolution(hidden)
        try assertUpstreamParity(
            hidden,
            try XCTUnwrap(fixture["vocoder_conv_t0"]).transposed(0, 2, 1),
            minimumSNR: 25,
            component: "vocoder block 0 transposed convolution"
        )
        hidden = block.residualUnit1(hidden)
        try assertUpstreamParity(
            hidden,
            try XCTUnwrap(fixture["vocoder_res1_0"]).transposed(0, 2, 1),
            minimumSNR: 25,
            component: "vocoder block 0 residual 1"
        )
        hidden = block.residualUnit2(hidden)
        try assertUpstreamParity(
            hidden,
            try XCTUnwrap(fixture["vocoder_res2_0"]).transposed(0, 2, 1),
            minimumSNR: 25,
            component: "vocoder block 0 residual 2"
        )
        hidden = block.residualUnit3(hidden)
        try assertUpstreamParity(
            hidden,
            try XCTUnwrap(fixture["vocoder_res3_0"]).transposed(0, 2, 1),
            minimumSNR: 25,
            component: "vocoder block 0 residual 3"
        )

        let vocoderOutput = vocoder(vocoderInput)
        try assertUpstreamParity(
            vocoderOutput,
            try XCTUnwrap(fixture["vocoder_output"]),
            minimumSNR: 25,
            component: "vocoder"
        )
    }

    func testInstalledDepthDecoderMatchesUpstream() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_MINIMAX_MUSIC3_MODEL_ROOT"],
              let fixturePath = environment["MERERUN_MINIMAX_MUSIC3_PARITY_FIXTURE"]
        else {
            throw XCTSkip("set MiniMax Music 3 model and parity fixture paths to run component parity")
        }
        let resources = MiniMaxMusic3Resources(rootURL: URL(fileURLWithPath: root))
        let fixture = try MLX.loadArrays(url: URL(fileURLWithPath: fixturePath))
        let depthDecoder = MiniMaxMusic3DepthDecoder(
            configuration: try resources.loadDepthConfiguration()
        )
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.depthDecoderURL
                .appendingPathComponent("diffusion_pytorch_model.safetensors.index.json"),
            singleURL: resources.depthDecoderURL
                .appendingPathComponent("diffusion_pytorch_model.safetensors"),
            to: depthDecoder,
            dtype: .float32,
            verify: .noUnusedKeys
        )

        let output = depthDecoder(try XCTUnwrap(fixture["depth_f32_input"]))
        try assertUpstreamParity(
            output,
            try XCTUnwrap(fixture["depth_f32_output"]),
            minimumSNR: 40,
            component: "RVQ depth decoder"
        )
        try assertUpstreamParity(
            depthDecoder.logits(output[0..., -1, 0...], codebookIndex: 0),
            try XCTUnwrap(fixture["depth_f32_logits"]),
            minimumSNR: 40,
            component: "RVQ depth head"
        )
        try assertUpstreamParity(
            depthDecoder.embedResidualCodes(MLXArray([Int32(0), 1_023]), codebookIndex: 0),
            try XCTUnwrap(fixture["depth_f32_embed"]),
            minimumSNR: 60,
            component: "RVQ residual embedding"
        )
    }

    func testInstalledFlowTransformerMatchesUpstream() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_MINIMAX_MUSIC3_MODEL_ROOT"],
              let fixturePath = environment["MERERUN_MINIMAX_MUSIC3_PARITY_FIXTURE"]
        else {
            throw XCTSkip("set MiniMax Music 3 model and parity fixture paths to run component parity")
        }
        let resources = MiniMaxMusic3Resources(rootURL: URL(fileURLWithPath: root))
        let fixture = try MLX.loadArrays(url: URL(fileURLWithPath: fixturePath))
        let transformer = MiniMaxMusic3Transformer(
            configuration: try resources.loadTransformerConfiguration()
        )
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.transformerURL
                .appendingPathComponent("diffusion_pytorch_model.safetensors.index.json"),
            singleURL: resources.transformerURL
                .appendingPathComponent("diffusion_pytorch_model.safetensors"),
            to: transformer,
            dtype: .float32,
            verify: .noUnusedKeys,
            mapper: MiniMaxMusic3Transformer.mapWeight
        )

        let output = transformer(
            latents: try XCTUnwrap(fixture["transformer_f32_latents"]),
            timestep: try XCTUnwrap(fixture["transformer_f32_timestep"]),
            condition: try XCTUnwrap(fixture["transformer_f32_condition"])
        )
        try assertUpstreamParity(
            output,
            try XCTUnwrap(fixture["transformer_f32_output"]),
            minimumSNR: 40,
            component: "flow transformer"
        )
    }

    func testInstalledLanguageModelMatchesUpstream() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_MINIMAX_MUSIC3_MODEL_ROOT"],
              let fixturePath = environment["MERERUN_MINIMAX_MUSIC3_PARITY_FIXTURE"]
        else {
            throw XCTSkip("set MiniMax Music 3 model and parity fixture paths to run component parity")
        }
        let resources = MiniMaxMusic3Resources(rootURL: URL(fileURLWithPath: root))
        let fixture = try MLX.loadArrays(url: URL(fileURLWithPath: fixturePath))
        let languageModel = MiniMaxMusic3LanguageModel(
            configuration: try resources.loadLanguageConfiguration()
        )
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.languageModelURL.appendingPathComponent("model.safetensors.index.json"),
            singleURL: resources.languageModelURL.appendingPathComponent("model.safetensors"),
            to: languageModel,
            dtype: .float32,
            verify: .noUnusedKeys
        )

        let embeddings = languageModel.embed(tokenIDs: try XCTUnwrap(fixture["lm_ids"]))
        let hidden = languageModel.hidden(
            embeddings: embeddings,
            cache: languageModel.makeCache(),
            lastPositionOnly: false
        )
        try assertUpstreamParity(
            hidden,
            try XCTUnwrap(fixture["lm_hidden"]),
            minimumSNR: 40,
            component: "language model"
        )
        try assertUpstreamParity(
            languageModel.logits(hidden[0..., -1, 0...]),
            try XCTUnwrap(fixture["lm_logits"]),
            minimumSNR: 40,
            component: "language-model head"
        )

        let cache = languageModel.makeCache()
        let cachedPrefill = languageModel.hidden(
            embeddings: languageModel.embed(tokenIDs: try XCTUnwrap(fixture["lm_cached_ids"])),
            cache: cache,
            lastPositionOnly: false
        )
        try assertUpstreamParity(
            cachedPrefill,
            try XCTUnwrap(fixture["lm_cached_prefill_hidden"]),
            minimumSNR: 40,
            component: "cached language-model prefill"
        )
        let cachedDecode = languageModel.hidden(
            embeddings: try XCTUnwrap(fixture["lm_next_embeds"]),
            cache: cache,
            lastPositionOnly: false
        )
        try assertUpstreamParity(
            cachedDecode,
            try XCTUnwrap(fixture["lm_cached_decode_hidden"]),
            minimumSNR: 40,
            component: "cached language-model decode"
        )
        try assertUpstreamParity(
            languageModel.logits(cachedDecode[0..., -1, 0...]),
            try XCTUnwrap(fixture["lm_cached_decode_logits"]),
            minimumSNR: 40,
            component: "cached language-model head"
        )
    }

    func testInstalledTokenizerMatchesUpstreamOfficialPrompt() throws {
        guard let root = ProcessInfo.processInfo.environment["MERERUN_MINIMAX_MUSIC3_MODEL_ROOT"] else {
            throw XCTSkip("set MERERUN_MINIMAX_MUSIC3_MODEL_ROOT to run checkpoint parity")
        }
        let tokenizer = try ACEStep5HzLMTokenizer.load(
            from: URL(fileURLWithPath: root).appendingPathComponent("tokenizer"),
            requireAudioCodeTokens: false
        )
        let prompt = MiniMaxMusic3Prompt.assemble(
            caption: "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
                + "Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. Arrangement: "
                + "fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus.",
            lyrics: "[verse]\nMorning light filtering through the pine\nEvery quiet street is yours and mine\n"
                + "[chorus]\nSoftly the world begins to breathe"
        )

        XCTAssertEqual(
            tokenizer.encode(prompt, addSpecialTokens: false),
            [
                151_644, 151_671, 37_525, 25, 44_066, 2_420, 13, 88_219, 25, 220, 24, 21, 13, 5_309,
                25, 356, 3_598, 13, 45_763, 323, 31_387, 11, 4_752, 29_273, 1_119, 279, 55_810, 13,
                86_045, 1_127, 25, 8_413, 8_778, 2_990, 11, 3_265, 323, 11_486, 88, 11, 3_100, 41_315,
                17_774, 550, 304, 279, 55_810, 13, 18_418, 56_633, 25, 14_317, 93_499, 16_986, 323,
                8_413, 26_278, 26, 61_539, 46_289, 323, 48_585, 21_529, 3_725, 304, 279, 55_810, 13,
                151_672, 151_673, 28_463, 921, 58, 4_450, 921, 84_344, 3_100, 29_670, 1_526, 279,
                33_597, 198, 11_510, 11_340, 8_592, 374, 18_316, 323, 10_485, 198, 58, 6_150, 355,
                921, 30_531, 398, 279, 1_879, 12_033, 311, 36_297, 151_674, 151_645, 151_669,
            ]
        )
    }

    func testPromptAssemblyMatchesCheckpointContract() {
        let prompt = MiniMaxMusic3Prompt.assemble(
            caption: "# Dream pop\n- <|tempo 118 bpm|>\n**wide guitars**",
            lyrics: "[Verse] dropped words\nHello [Whisper]\n[CHORUS]\nWe glow"
        )

        XCTAssertEqual(
            prompt,
            "<|im_start|><|caption_start|>Dream pop\ntempo is 118 bpm\nwide guitars<|caption_end|>"
                + "<|lyrics_start|>[start]\n[verse]\nHello\n[whisper]\n[chorus]\nWe glow"
                + "<|lyrics_end|><|im_end|><|audio_start|>"
        )
    }

    func testChunkAndLatentTimelineMathMatchesReference() {
        XCTAssertEqual(MiniMaxMusic3Prompt.chunkStarts(frameCount: 200), [0])
        XCTAssertEqual(MiniMaxMusic3Prompt.chunkStarts(frameCount: 201), [0, 100])
        XCTAssertEqual(MiniMaxMusic3Prompt.chunkStarts(frameCount: 300), [0, 100])
        XCTAssertEqual(MiniMaxMusic3Prompt.latentLength(frameCount: 200), 689)
        XCTAssertEqual(MiniMaxMusic3Prompt.latentLength(frameCount: 100), 344)
    }

    func testConditionEncoderProducesLatentAlignedShape() {
        let configuration = MiniMaxMusic3ConditionConfiguration(
            conditionHiddenDim: 4,
            numConditionLayers: 2,
            outDim: 3,
            inputSamplingRate: 24_000,
            inputHopLength: 960,
            outputSamplingRate: 44_100,
            outputHopLength: 512
        )
        let model = MiniMaxMusic3ConditionEncoder(configuration: configuration)
        let output = model(MLXArray.zeros([1, 10, 8]))
        MLX.eval(output)
        XCTAssertEqual(output.shape, [1, 34, 3])
    }

    func testDepthDecoderTinyConfigurationPreservesSequenceShape() {
        let configuration = MiniMaxMusic3DepthConfiguration(
            hiddenSize: 8,
            numLayers: 1,
            numAttentionHeads: 2,
            intermediateSize: 16,
            audioVocabSize: 16,
            numCodebooks: 3,
            maxPositionEmbeddings: 4
        )
        let model = MiniMaxMusic3DepthDecoder(configuration: configuration)
        let output = model(MLXArray.zeros([2, 3, 8]))
        MLX.eval(output)
        XCTAssertEqual(output.shape, [2, 3, 8])
        XCTAssertEqual(model.logits(output[0..., -1, 0...], codebookIndex: 0).shape, [2, 16])
    }

    func testFlowTransformerTinyConfigurationPreservesLatentShape() {
        let configuration = MiniMaxMusic3TransformerConfiguration(
            inChannels: 2,
            conditionDim: 4,
            numLayers: 1,
            numAttentionHeads: 2,
            attentionHeadDim: 4,
            ffInnerDim: 16,
            rotaryDim: 2,
            fourierEmbeddingDim: 4
        )
        let model = MiniMaxMusic3Transformer(configuration: configuration)
        let output = model(
            latents: MLXArray.zeros([1, 2, 7]),
            timestep: MLXArray([Float(0.5)]),
            condition: MLXArray.zeros([1, 7, 4])
        )
        MLX.eval(output)
        XCTAssertEqual(output.shape, [1, 2, 7])
    }

    func testSemanticTopKIgnoresTextVocabularyLogits() {
        let vocabulary = MiniMaxMusic3Prompt.audioCodeOffset
            + MiniMaxMusic3Prompt.semanticVocabularySize
        var values = [Float](repeating: -100, count: 2 * vocabulary)
        for batch in 0..<2 {
            let base = batch * vocabulary
            for token in 0..<50 {
                values[base + token] = 100
            }
            for token in 0..<50 {
                values[base + MiniMaxMusic3Prompt.audioCodeOffset + token] = Float(50 - token)
            }
        }

        let guided = MiniMaxMusic3Pipeline.guidedSemanticLogits(
            MLXArray(values).reshaped(2, vocabulary)
        )
        MLX.eval(guided)

        XCTAssertTrue(guided[0, MiniMaxMusic3Prompt.audioCodeOffset].item(Float.self).isFinite)
        XCTAssertFalse(guided[0, 0].item(Float.self).isFinite)
    }

    func testManagedCatalogPinsSelectiveRestrictedSnapshot() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: MiniMaxMusic3Resources.modelID))
        XCTAssertEqual(spec.upstreamRepoId, MiniMaxMusic3Resources.repository)
        XCTAssertEqual(spec.upstreamRevision, MiniMaxMusic3Resources.revision)
        XCTAssertEqual(spec.estimatedDownloadBytes, 28_517_620_807)
        XCTAssertEqual(spec.validationKind, .miniMaxMusic3)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertNotNil(spec.usageRestriction)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("qwen_7B/*"), false)
        XCTAssertEqual(spec.hubFallback?.patterns.contains("transformer/*"), true)
    }

    func testCapabilityDescriptorRequiresLargeUnifiedMemory() throws {
        let descriptor = try XCTUnwrap(
            ManagedModelCapabilityCatalog.descriptor(for: MiniMaxMusic3Resources.modelID)
        )
        XCTAssertEqual(descriptor.minimumUnifiedMemoryGB, 64)
        XCTAssertEqual(descriptor.recommendedUnifiedMemoryGB, 96)
    }

    func testManagedInstallValidatorUsesMiniMaxComponentContract() throws {
        let root = try TestFileSystem.makeTempDir(prefix: "minimax-music3-validator")
        defer { try? FileManager.default.removeItem(at: root) }

        try MereRunModelManifest.template(
            for: .miniMaxMusic3,
            createdAt: Date(timeIntervalSince1970: 0)
        ).write(to: root)

        let files = [
            "LICENSE",
            "config.json",
            "modular_model_index.json",
            "condition_encoder/config.json",
            "condition_encoder/diffusion_pytorch_model.safetensors",
            "language_model/config.json",
            "language_model/model.safetensors",
            "rvq_depth_decoder/config.json",
            "rvq_depth_decoder/diffusion_pytorch_model.safetensors",
            "scheduler/scheduler_config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vocoder/config.json",
            "vocoder/diffusion_pytorch_model.safetensors",
        ]
        for path in files {
            try TestFileSystem.writeFile(root.appendingPathComponent(path))
        }

        let report = MereRunModelValidator.validate(
            modelRoot: root,
            expectedModelID: MiniMaxMusic3Resources.modelID
        )
        XCTAssertTrue(report.isValid, report.errors.joined(separator: "\n"))
        XCTAssertFalse(report.warnings.contains { $0.contains("engine mismatch") })
    }

    private func assertUpstreamParity(
        _ actual: MLXArray,
        _ expected: MLXArray,
        minimumSNR: Float,
        component: String
    ) throws {
        XCTAssertEqual(actual.shape, expected.shape, component)
        let reference = expected.asType(.float32)
        let error = actual.asType(.float32) - reference
        MLX.eval(error)
        let signalPower = MLX.mean(reference * reference).item(Float.self)
        let errorPower = MLX.mean(error * error).item(Float.self)
        let snr = 10 * log10(signalPower / max(errorPower, Float.leastNonzeroMagnitude))
        let maximumError = MLX.max(MLX.abs(error)).item(Float.self)
        XCTAssertGreaterThanOrEqual(snr, minimumSNR, "\(component): SNR \(snr) dB, max error \(maximumError)")
    }
}
