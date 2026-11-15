import Foundation
import XCTest
import MLX
import MLXNN
@testable import MereRunGemmaModel
import MereRunMLXTestSupport
@testable import AudioSOVTranscribeModel

final class SOVTranscribeModelTests: MLXTestCase {
    struct Tensor: Decodable {
        let shape: [Int]
        let values: [Float]
        var array: MLXArray { MLXArray(values, shape) }
    }
    struct Fixture: Decodable {
        let config: SOVTranscribeConfig
        let draftConfig: SOVDFlashConfig
        let audioWeights: [String: Tensor]
        let projectorWeights: [String: Tensor]
        let draftWeights: [String: Tensor]
        let mel: Tensor
        let encoded: Tensor
        let projected: Tensor
        let context: Tensor
        let noise: Tensor
        let denoised: Tensor
    }
    private func fixture() throws -> Fixture {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "reference", withExtension: "json", subdirectory: "Fixtures/SOV"))
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }
    private func assertClose(_ actual: MLXArray, _ expected: MLXArray, accuracy: Float = 0.00005,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertLessThanOrEqual(abs(actual - expected).max().item(Float.self), accuracy, file: file, line: line)
    }
    private func load(_ weights: [String: Tensor], into model: Module) throws {
        try model.update(parameters: ModuleParameters.unflattened(weights.map { ($0.key, $0.value.array) }), verify: .all)
    }

    func testWhisperEncoderAndStackedProjectionMatchIndependentNumpy() throws {
        let f = try fixture()
        try f.config.validate()
        let encoder = SOVAudioEncoder(config: f.config.audioConfig)
        let projector = SOVAudioProjector(config: f.config)
        try load(f.audioWeights, into: encoder)
        try load(f.projectorWeights, into: projector)
        let encoded = try encoder(f.mel.array)
        assertClose(encoded, f.encoded.array)
        assertClose(projector(encoded), f.projected.array)
        XCTAssertThrowsError(try encoder(MLXArray.zeros([1, 128, 7])))
    }

    func testDFlashMixedAttentionAndIncrementalContextMatchIndependentNumpy() throws {
        let f = try fixture()
        try f.draftConfig.validate(target: f.config.textConfig)
        let model = SOVDFlashModel(config: f.draftConfig)
        try load(f.draftWeights, into: model)
        let states = [0: f.context.array[0..., 0..., ..<8], 1: f.context.array[0..., 0..., 8...]]
        let full = model.makeCache()
        model.appendContext(states, cache: full)
        assertClose(model.denoise(f.noise.array, cache: full), f.denoised.array)
        XCTAssertEqual(full.map(\.offset), [5, 5])
        XCTAssertEqual(full[1].currentState()!.0.dim(2), 5)
        let incremental = model.makeCache()
        for range in [0..<2, 2..<5] {
            model.appendContext(states.mapValues { $0[0..., range, 0...] }, cache: incremental)
        }
        assertClose(model.denoise(f.noise.array, cache: incremental), f.denoised.array)
        // Denoising never adds the diffusion block to committed context.
        XCTAssertEqual(incremental.map(\.offset), [5, 5])
        assertClose(model.denoise(f.noise.array, cache: incremental), f.denoised.array)
    }

    func testSlidingDraftMaskIsCausalAndExcludesExpiredContext() {
        let mask = SOVDFlashAttention.slidingMask(offset: 5, contextLength: 3, length: 3, window: 3, dtype: .float32)
        let visible = mask.asArray(Float.self).map { $0 == 0 }
        XCTAssertEqual(visible, [false, true, true, true, false, false,
                                 false, false, true, true, true, false,
                                 false, false, false, true, true, true])
    }

    func testReleasedConfigsIdentifySixTargetLayersAndFifteenDraftTokens() throws {
        let mainURL = try XCTUnwrap(Bundle.module.url(forResource: "released-config", withExtension: "json", subdirectory: "Fixtures/SOV"))
        let draftURL = try XCTUnwrap(Bundle.module.url(forResource: "released-draft-config", withExtension: "json", subdirectory: "Fixtures/SOV"))
        let main = try JSONDecoder().decode(SOVTranscribeConfig.self, from: Data(contentsOf: mainURL))
        let draft = try JSONDecoder().decode(SOVDFlashConfig.self, from: Data(contentsOf: draftURL))
        try main.validate()
        try draft.validate(target: main.textConfig)
        XCTAssertEqual(main.audioConfig.maxSourcePositions / main.downsampleFactor, 375)
        XCTAssertEqual(draft.blockSize - 1, 15)
        XCTAssertEqual(draft.dflash.targetLayerIDs, [1, 6, 11, 17, 22, 27])
        XCTAssertEqual(draft.layerTypes, Array(repeating: "sliding_attention", count: 4) + ["full_attention"])
    }

    private func target(_ config: Gemma4TextConfig) throws -> Gemma4TextCausalLM {
        let model = Gemma4TextCausalLM(config: config)
        var updates = model.parameters().flattened().map { name, array in
            (name, name.contains(".rope.") ? array : name.hasSuffix("norm.weight") || name.hasSuffix("layer_scalar")
                ? MLXArray.ones(array.shape) : MLXArray.zeros(array.shape))
        }
        var table = [Float](repeating: 0, count: config.vocabSize * config.hiddenSize)
        for dim in 0..<config.hiddenSize {
            table[7 * config.hiddenSize + dim] = Float(dim + 1) / 8
            table[4 * config.hiddenSize + dim] = Float(dim + 1) / 16
        }
        updates.removeAll { $0.0 == "language_model.embed_tokens.weight" }
        updates.append(("language_model.embed_tokens.weight", MLXArray(table, [config.vocabSize, config.hiddenSize])))
        try model.update(parameters: ModuleParameters.unflattened(updates), verify: .all)
        XCTAssertGreaterThan(model.rawInputEmbeddings(MLXArray([Int32(7)]).reshaped(1, 1)).max().item(Float.self), 0)
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(parameters["language_model.norm.weight"]!.asArray(Float.self), Array(repeating: Float(1), count: 8))
        XCTAssertEqual(parameters["language_model.layers.0.layer_scalar"]!.item(Float.self), 1)
        let ids = MLXArray([Int32(7)]).reshaped(1, 1)
        XCTAssertEqual(argMax(model.outputLogits(model.inputEmbeddings(for: ids)), axis: -1).item(Int.self), 7)
        let forward = model.forwardEmbeddings(model.inputEmbeddings(for: ids), inputIDs: ids, cache: model.makeAttentionCache(),
                                              captureLayerIndices: [0, 1], lastLogitsOnly: true)
        XCTAssertGreaterThan(forward.states[0]!.max().item(Float.self), 0)
        XCTAssertGreaterThan(forward.states[1]!.max().item(Float.self), 0)
        XCTAssertEqual(argMax(forward.logits, axis: -1).item(Int.self), 7)
        return model
    }

    func testGreedyVerificationMatchesTargetAcrossAcceptanceRejectionAndWindowRollover() throws {
        let f = try fixture()
        let target = try target(f.config.textConfig)
        let ids = MLXArray([Int32(2), 7]).reshaped(1, 2)
        let plain = try SOVTranscribeDecoder.decode(target: target, embeddings: target.inputEmbeddings(for: ids), inputIDs: ids,
                                                   maxTokens: 20, eosTokens: [], prefillChunkSize: 1)
        XCTAssertEqual(plain.tokens, Array(repeating: 7, count: 20))
        for accepted in [false, true] {
            let draft = SOVDFlashModel(config: f.draftConfig)
            let updates = draft.parameters().flattened().map { name, array in
                (name, accepted && name.hasSuffix("norm.weight") ? MLXArray.ones(array.shape) : MLXArray.zeros(array.shape))
            }
            try draft.update(parameters: ModuleParameters.unflattened(updates), verify: .all)
            let result = try SOVTranscribeDecoder.decode(target: target, embeddings: target.inputEmbeddings(for: ids), inputIDs: ids,
                                                        draft: draft, maxTokens: 20, eosTokens: [], prefillChunkSize: 1)
            XCTAssertEqual(result.tokens, plain.tokens)
            XCTAssertGreaterThan(result.verificationRounds, 0)
            if accepted { XCTAssertGreaterThan(result.acceptedDraftTokens, 0); XCTAssertEqual(result.rejectedRounds, 0) }
            else {
                XCTAssertGreaterThan(result.rejectedRounds, 0)
                XCTAssertEqual(result.acceptedDraftTokens, 0)
                XCTAssertEqual(result.reusedVerificationRounds, result.rejectedRounds)
            }
            let pipelined = try SOVTranscribeDecoder.decode(target: target, embeddings: target.inputEmbeddings(for: ids), inputIDs: ids,
                draft: draft, maxTokens: 20, eosTokens: [], prefillChunkSize: 1, pipelined: true)
            XCTAssertEqual(pipelined.tokens, result.tokens)
            XCTAssertEqual(pipelined.acceptedLengths, result.acceptedLengths)
            XCTAssertEqual(pipelined.reusedVerificationRounds, result.reusedVerificationRounds)
            XCTAssertEqual(pipelined.emittedDraftTokens, result.emittedDraftTokens)
            let rollback = try SOVTranscribeDecoder.decode(target: target, embeddings: target.inputEmbeddings(for: ids), inputIDs: ids,
                draft: draft, maxTokens: 20, eosTokens: [], prefillChunkSize: 1, pipelined: true, rollbackCache: true)
            XCTAssertEqual(rollback.tokens, result.tokens)
            var timings: [SOVDecodeRoundTiming] = []
            let instrumented = try SOVTranscribeDecoder.decode(target: target, embeddings: target.inputEmbeddings(for: ids), inputIDs: ids,
                draft: draft, maxTokens: 20, eosTokens: [], prefillChunkSize: 1, pipelined: true, rollbackCache: true,
                roundTiming: { timings.append($0) }, diagnostics: SOVDecodeDiagnostics())
            XCTAssertEqual(instrumented.tokens, rollback.tokens)
            XCTAssertEqual(instrumented.acceptedLengths, rollback.acceptedLengths)
            XCTAssertEqual(timings.count, instrumented.verificationRounds)
            XCTAssertEqual(timings.map(\.round), Array(1...timings.count))
            XCTAssertTrue(timings.allSatisfy { $0.endUptime >= $0.startUptime && $0.wallSeconds > 0 })
        }
    }


    func testVaryingTargetSequenceAgreesWithDFlashAndStopsAtLaterEOS() throws {
        let f = try fixture()
        let target = Gemma4TextCausalLM(config: f.config.textConfig)
        // Deterministic nonzero weights exercise real context dependence rather than a constant head.
        let weights = target.parameters().flattened().enumerated().map { index, item -> (String, MLXArray) in
            let (name, value) = item
            if name.contains(".rope.") { return (name, value) }
            let phase = MLXArray(Float(index) * 0.37)
            let sequence = MLXArray((0..<value.size).map { Float($0) * 0.71 })
            let data = sin(sequence + phase).reshaped(value.shape)
            if name.hasSuffix("norm.weight") || name.hasSuffix("layer_scalar") { return (name, data * 0.1 + 1) }
            return (name, data * 0.7)
        }
        try target.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let draft = SOVDFlashModel(config: f.draftConfig)
        try load(f.draftWeights, into: draft)
        let ids = MLXArray([Int32(3), 11, 6, 9]).reshaped(1, 4)
        func decode(_ draft: SOVDFlashModel?, stop: Set<Int> = [], pipelined: Bool = false) throws -> SOVDecodeResult {
            try SOVTranscribeDecoder.decode(target: target, embeddings: target.inputEmbeddings(for: ids), inputIDs: ids,
                                            draft: draft, maxTokens: 23, eosTokens: stop, prefillChunkSize: 2, pipelined: pipelined)
        }
        let plain = try decode(nil)
        let speculated = try decode(draft)
        XCTAssertEqual(speculated.tokens, plain.tokens)
        XCTAssertGreaterThan(speculated.rejectedRounds, 0)
        XCTAssertEqual(speculated.reusedVerificationRounds, speculated.rejectedRounds)
        XCTAssertGreaterThan(Set(plain.tokens).count, 1)
        let pipelined = try decode(draft, pipelined: true)
        XCTAssertEqual(pipelined.tokens, plain.tokens)
        XCTAssertEqual(pipelined.acceptedLengths, speculated.acceptedLengths)
        if let later = plain.tokens.dropFirst().first(where: { $0 != plain.tokens.first }) {
            let expected = Array(plain.tokens.prefix { $0 != later })
            XCTAssertEqual(try decode(draft, stop: [later]).tokens, expected)
            XCTAssertEqual(try decode(draft, stop: [later], pipelined: true).tokens, expected)
        }
    }

    private func moeConfig() throws -> Gemma4TextConfig {
        let data = Data(#"{"model_type":"gemma4_text","hidden_size":8,"num_hidden_layers":2,"intermediate_size":12,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"max_position_embeddings":128,"rms_norm_eps":0.000001,"vocab_size":32,"enable_moe_block":true,"num_experts":4,"top_k_experts":2,"moe_intermediate_size":6}"#.utf8)
        return try JSONDecoder().decode(Gemma4TextConfig.self, from: data)
    }

    func testFP32RouterUsesFP32Input() throws {
        let config = try moeConfig()
        let router = Gemma4Router(config: config)
        let projection = (0..<32).map { sin(Float($0) * 0.23) * 0.2 }
        let scale = (0..<8).map { Float($0 + 1) / 5 }
        let expertScale: [Float] = [1, 0.7, 1.1, 0.8]
        try router.update(parameters: ModuleParameters.unflattened([
            ("proj.weight", MLXArray(projection, [4, 8])), ("scale", MLXArray(scale)), ("per_expert_scale", MLXArray(expertScale))
        ]), verify: .all)
        let input = MLXArray([Float(0.23), -0.91, 0.12, 0.77, 0.45, -0.3, 1.7, -0.63], [1, 1, 8]).asType(.bfloat16)
        let x = input.asType(.float32).asArray(Float.self).map(Double.init)
        let rms = sqrt(x.reduce(0) { $0 + $1 * $1 } / 8 + 0.000001)
        let scores = (0..<4).map { row in
            (0..<8).reduce(Double(0)) { $0 + x[$1] / rms * Double(scale[$1]) / sqrt(8) * Double(projection[row * 8 + $1]) }
        }
        let expected = Array(scores.indices.sorted { scores[$0] > scores[$1] }.prefix(2))
        let route = router(input)
        let indices = route.indices.asArray(Int.self)
        let weights = route.weights.asArray(Float.self)
        XCTAssertEqual(Set(indices), Set(expected))
        XCTAssertEqual(route.weights.dtype, .float32)
        let denominator = expected.reduce(Double(0)) { $0 + exp(scores[$1]) }
        for (index, weight) in zip(indices, weights) {
            XCTAssertEqual(weight, Float(exp(scores[index]) / denominator) * expertScale[index], accuracy: 0.00001)
        }
    }

    func testCompiledMoEPostSegmentReadsUpdatedParameters() throws {
        let model = Gemma4TextCausalLM(config: try moeConfig())
        model.prepareForFullPrecisionExpertWeights()
        let ids = MLXArray([Int32(7), 4, 11, 2]).reshaped(1, 4)
        func forward(_ compiled: Bool) -> MLXArray {
            model.setExperimentalCompiledPost(compiled)
            let result = model.forwardEmbeddings(model.inputEmbeddings(for: ids), inputIDs: ids, cache: model.makeAttentionCache()).logits
            eval(result)
            return result
        }
        let (weightName, weight) = try XCTUnwrap(model.parameters().flattened().first {
            $0.0 == "language_model.layers.0.mlp.gate_proj.weight"
        })
        var previous: MLXArray?
        for scale in [Float(1), 0.7] {
            try model.update(parameters: ModuleParameters.unflattened([
                ("language_model.layers.0.layer_scalar", MLXArray([scale])),
                (weightName, weight * scale)
            ]), verify: .none)
            let actual = forward(true)
            assertClose(actual, forward(false), accuracy: 0.0001)
            if let previous { XCTAssertGreaterThan(abs(actual - previous).max().item(Float.self), 0.00001) }
            previous = actual
        }
    }

    func testRestrictedDraftHeadMapsLocalRowsToGlobalTokens() throws {
        let f = try fixture()
        let target = try target(f.config.textConfig)
        let draft = SOVDFlashModel(config: f.draftConfig)
        try draft.update(parameters: ModuleParameters.unflattened(draft.parameters().flattened().map { name, value in
            (name, name.hasSuffix("norm.weight") ? MLXArray.ones(value.shape) : MLXArray.zeros(value.shape))
        }), verify: .all)
        let head = Linear(weight: concatenated([MLXArray.zeros([1, 8]), MLXArray.ones([1, 8])]), bias: nil)
        let proposal = draft.proposal(anchor: 7, count: 3, target: target, cache: draft.makeCache(), outputHead: head,
            tokenMapping: MLXArray([Int32(3), 7]), confidence: true)
        XCTAssertEqual(proposal.ids.asArray(Int.self), [7, 7, 7])
        XCTAssertTrue(try XCTUnwrap(proposal.confidence).asArray(Float.self).allSatisfy { $0 > 0.5 })
    }

    func testAdaptiveAfterFirstPreservesInitialBlockAndBudget() throws {
        let f = try fixture()
        let target = try target(f.config.textConfig)
        let draft = SOVDFlashModel(config: f.draftConfig)
        try draft.update(parameters: ModuleParameters.unflattened(draft.parameters().flattened().map { name, value in
            (name, name.hasSuffix("norm.weight") ? MLXArray.ones(value.shape) : MLXArray.zeros(value.shape))
        }), verify: .all)
        let ids = MLXArray([Int32(7)]).reshaped(1, 1)
        let result = try SOVTranscribeDecoder.decode(target: target, embeddings: target.inputEmbeddings(for: ids), inputIDs: ids,
            draft: draft, maxTokens: 20, eosTokens: [], pipelined: true, adaptiveConfidence: 1,
            adaptiveAfterFirst: true)
        XCTAssertEqual(result.tokens, Array(repeating: 7, count: 20))
        XCTAssertEqual(result.proposedLengths.first, f.draftConfig.blockSize - 1)
        XCTAssertTrue(result.proposedLengths.dropFirst().allSatisfy { $0 == 1 })
        XCTAssertEqual(result.proposedLengths.count, result.acceptedLengths.count)
    }

    func testBF16ExpertOutputPreservesInputPrecisionOnGPU() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "BF16 gathered expert arithmetic requires the GPU.")
        let config = try moeConfig()
        let input = MLXArray.ones([1, 1, 8]).asType(.bfloat16)
        let indices = MLXArray([Int32(0), 3]).reshaped(1, 1, 2)
        let weights = MLXArray([Float(0.35), 0.65]).reshaped(1, 1, 2)
        let experts = Gemma4Experts(config: config)
        let full = Gemma4SwitchGLU(config: config, quantized: false)
        experts.update(modules: ModuleChildren.unflattened([("switch_glu", full)]))
        try experts.update(parameters: ModuleParameters.unflattened(experts.parameters().flattened().map {
            ($0.0, MLXArray.ones($0.1.shape).asType(.bfloat16) * 0.1)
        }), verify: .all)
        let output = experts(input, indices: indices, weights: weights)
        XCTAssertEqual(output.dtype, .bfloat16)
        XCTAssertTrue(output.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testEOSAndBudgetAreRespected() throws {
        let f = try fixture()
        let target = try target(f.config.textConfig)
        let ids = MLXArray([Int32(7)]).reshaped(1, 1)
        let empty = try SOVTranscribeDecoder.decode(target: target, embeddings: target.inputEmbeddings(for: ids), inputIDs: ids,
                                                   maxTokens: 5, eosTokens: [7])
        XCTAssertEqual(empty.tokens, [])
        let one = try SOVTranscribeDecoder.decode(target: target, embeddings: target.inputEmbeddings(for: ids), inputIDs: ids,
                                                 maxTokens: 1, eosTokens: [])
        XCTAssertEqual(one.tokens, [7])
        XCTAssertThrowsError(try SOVTranscribeDecoder.decode(target: target, embeddings: target.inputEmbeddings(for: ids), inputIDs: ids,
                                                           maxTokens: 0, eosTokens: []))
        XCTAssertThrowsError(try SOVTranscribeDecoder.decode(target: target, embeddings: target.inputEmbeddings(for: ids), inputIDs: ids,
                                                           maxTokens: 129, eosTokens: []))
    }
}
