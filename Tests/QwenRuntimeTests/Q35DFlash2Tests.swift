import Foundation
import MereRunKVCache
import MereRunTensor
import MLX
import MLXNN
import MLXRandom
import XCTest
import MereRunMLXTestSupport
@testable import MereRunQwenModel

final class Q35DFlash2Tests: MLXTestCase {
    func testPublishedConfigurationAndParameterPaths() throws {
        let config = try JSONDecoder().decode(Q35DFlash2Config.self, from: Data(Self.published.utf8))
        XCTAssertEqual(config.dflash.targetLayerIDs, [5, 19, 33, 47, 61])
        XCTAssertEqual(config.dflash.blockSize, 8)
        XCTAssertEqual(config.slidingWindow, 2048)
        let small = try draftConfiguration()
        try small.validate(target: targetConfiguration())
        let paths = Set(Q35DFlash2Model(config: small).parameters().flattened().map(\.0))
        for key in ["fc.weight", "hidden_norm.weight", "layers.0.input_layernorm.weight",
                    "layers.0.self_attn.q_norm.weight", "layers.0.attention_conv.base_kernel",
                    "layers.0.mlp_conv.kernel_projection.weight", "candidate_selector.predecessor_codebook.weight"] {
            XCTAssertTrue(paths.contains(key), key)
        }
        XCTAssertEqual(Q35DFlash2Model.weightKey("candidate_selector.successor_codebook"),
                       "candidate_selector.successor_codebook.weight")
    }

    func testConfigurationRejectsIncompatibleContracts() throws {
        let config = try targetConfiguration()
        for (old, replacement) in [("\"is_causal\":false", "\"is_causal\":true"),
                                    ("\"target_layer_ids\":[0,1]", "\"target_layer_ids\":[1,0]"),
                                    ("\"target_layer_ids\":[0,1]", "\"target_layer_ids\":[0,2]"),
                                    ("\"conv_group_size\":16", "\"conv_group_size\":3"),
                                    ("\"selector_top_k\":2", "\"selector_top_k\":65"),
                                    ("\"rope_type\":\"default\"", "\"rope_type\":\"linear\"") ] {
            let modified = Self.smallDraft.replacingOccurrences(of: old, with: replacement)
            XCTAssertNotEqual(modified, Self.smallDraft)
            let draft = try JSONDecoder().decode(Q35DFlash2Config.self, from: Data(modified.utf8))
            XCTAssertThrowsError(try draft.validate(target: config), old)
        }
    }

    func testCapturesAreZeroBasedLayerOutputsBeforeFinalNorm() throws {
        let config = try targetConfiguration()
        let model = Q35Model(config: config)
        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let before = model.embeddings(for: tokens)
        let expected = model.model.layers[0](before, fullMask: .causal, cache: .linear(Q35LinearCache()))
        let output = model.forwardPrefill(tokens, cache: caches(), captureLayerIndices: [0, 1])
        assertClose(try XCTUnwrap(output.capturedHiddenStates[0]), expected)
        XCTAssertEqual(output.capturedHiddenStates[1]?.shape, [1, 3, 32])
        XCTAssertEqual(output.hidden.shape, [1, 1, 32])
        XCTAssertTrue(model.forwardPrefill(tokens, cache: caches()).capturedHiddenStates.isEmpty)
    }

    func testDraftWindowAndForkMatchFullContextAfterChunking() throws {
        MLXRandom.seed(501)
        let draft = Q35DFlash2Model(config: try draftConfiguration())
        let states = [0: MLXRandom.normal([1, 11, 32]), 1: MLXRandom.normal([1, 11, 32])]
        let full = draft.makeCache()
        draft.appendTargetContext(states, cache: full)
        let chunked = draft.makeCache()
        for range in [0..<3, 3..<8, 8..<11] {
            draft.appendTargetContext(states.mapValues { $0[0..., range, 0...] }, cache: chunked)
        }
        for (lhs, rhs) in zip(full, chunked) {
            XCTAssertEqual(lhs.offset, 11)
            XCTAssertEqual(lhs.keys?.dim(2), 4)
            assertClose(try XCTUnwrap(lhs.keys), try XCTUnwrap(rhs.keys))
            let original = try XCTUnwrap(lhs.keys).asArray(Float.self)
            let fork = lhs.fork()
            fork.append(keys: MLXArray.zeros([1, 2, 2, 8]), values: MLXArray.zeros([1, 2, 2, 8]))
            XCTAssertEqual(lhs.keys?.asArray(Float.self), original)
            XCTAssertEqual(lhs.offset, 11)
            XCTAssertEqual(fork.offset, 13)
        }
        let target = Q35Model(config: try targetConfiguration())
        let offsets = full.map(\.offset)
        let expected = draft.draft(anchor: 2, count: 3, cache: full, target: target).asArray(Int32.self)
        let actual = draft.draft(anchor: 2, count: 3, cache: chunked, target: target).asArray(Int32.self)
        XCTAssertEqual(actual, expected)
        XCTAssertEqual(full.map(\.offset), offsets, "Proposals must not advance confirmed context")
    }

    func testDraftMaskKeepsWholeBlockAndSlidesOnlyContext() {
        let mask = Q35DFlash2Attention.blockMask(offset: 5, length: 3, contextLength: 4, window: 4, dtype: .float32)
        XCTAssertEqual(mask.asArray(Float.self).map { $0 == 0 }, [
            false, true, true, true, true, true, true,
            false, false, true, true, true, true, true,
            false, false, false, true, true, true, true,
        ])
    }

    func testDeferredCacheForksStayIndependentWithoutMaterialization() throws {
        let cache = Q35DFlash2Cache(window: 4)
        let original = MLXArray((0..<48).map(Float.init)).reshaped(1, 2, 3, 8)
        cache.append(keys: original, values: original * 2)
        let fork = cache.fork()
        let next = MLXArray.ones([1, 2, 2, 8]) * 100
        cache.append(keys: next, values: next * 2)
        XCTAssertEqual(try XCTUnwrap(fork.keys).asArray(Float.self), original.asArray(Float.self))
        XCTAssertEqual(try XCTUnwrap(fork.values).asArray(Float.self), (original * 2).asArray(Float.self))
        fork.append(keys: next * 3, values: next * 4)
        let expected = concatenated([original[0..., 0..., 1..., 0...], next], axis: 2)
        XCTAssertEqual(try XCTUnwrap(cache.keys).asArray(Float.self), expected.asArray(Float.self))
        XCTAssertEqual(cache.offset, 5)
        XCTAssertEqual(fork.offset, 5)
    }

    func testFusedDraftAttentionMatchesSeparateProjectionsAndInvalidates() throws {
        MLXRandom.seed(509)
        let config = try draftConfiguration()
        let attention = Q35DFlash2Attention(config: config)
        attention.update(parameters: attention.parameters().mapValues { $0.asType(.bfloat16) })
        quantize(model: attention, groupSize: 32, bits: 4, filter: { _, module in module is Linear })
        let context = MLXRandom.normal([1, 7, config.hiddenSize]).asType(.bfloat16)
        let input = MLXRandom.normal([1, 4, config.hiddenSize]).asType(.bfloat16)
        let referenceCache = Q35DFlash2Cache(window: 4)
        attention.appendContext(context, cache: referenceCache)
        let expected = attention(input, cache: referenceCache)
        MLX.eval(expected)
        attention.prepareProjections()
        let fusedCache = Q35DFlash2Cache(window: 4)
        attention.appendContext(context, cache: fusedCache)
        let actual = attention(input, cache: fusedCache)
        XCTAssertEqual((actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self), 0)
        XCTAssertEqual(
            try XCTUnwrap(fusedCache.keys).asArray(Float.self),
            try XCTUnwrap(referenceCache.keys).asArray(Float.self)
        )
        XCTAssertEqual(
            try XCTUnwrap(fusedCache.values).asArray(Float.self),
            try XCTUnwrap(referenceCache.values).asArray(Float.self)
        )
        let replacement = Linear(
            weight: MLXArray.zeros([config.numAttentionHeads * config.headDim, config.hiddenSize], dtype: .bfloat16),
            bias: nil
        )
        attention.update(modules: ModuleChildren.unflattened([("q_proj", replacement)]))
        let invalidated = attention(input, cache: fusedCache)
        MLX.eval(invalidated)
        // A mixed dense/quantized group cannot fuse. Rebuilding explicitly
        // provides the independent separate-projection result after replacement.
        attention.prepareProjections()
        let replacedReference = attention(input, cache: referenceCache)
        XCTAssertEqual(
            (invalidated.asType(.float32) - replacedReference.asType(.float32)).abs().max().item(Float.self), 0
        )
    }

    func testProjectionQuantizationUpdatesEveryDraftLayerTogether() throws {
        let json = Self.smallDraft
            .replacingOccurrences(of: "\"num_hidden_layers\":1", with: "\"num_hidden_layers\":2")
            .replacingOccurrences(of: "[\"sliding_attention\"]", with: "[\"sliding_attention\",\"sliding_attention\"]")
        let config = try JSONDecoder().decode(Q35DFlash2Config.self, from: Data(json.utf8))
        let draft = Q35DFlash2Model(config: config)
        draft.update(parameters: draft.parameters().mapValues { $0.asType(.bfloat16) })
        draft.quantizeProjections(groupSize: 32)
        let parameters = Dictionary(uniqueKeysWithValues: draft.parameters().flattened())
        for layer in 0..<2 {
            for name in ["self_attn.q_proj", "mlp.gate_proj", "attention_conv.kernel_projection"] {
                XCTAssertNotNil(parameters["layers.\(layer).\(name).scales"])
            }
        }
        XCTAssertEqual(parameters["candidate_selector.predecessor_codebook.weight"]?.dtype, .bfloat16)
        XCTAssertNil(parameters["candidate_selector.predecessor_codebook.scales"])
        draft.prepareDrafting()
        let cache = draft.makeCache()
        draft.appendTargetContext([0: MLXArray.ones([1, 3, 32]), 1: MLXArray.ones([1, 3, 32])], cache: cache)
        let tokens = draft.draft(anchor: 2, count: 3, cache: cache, target: Q35Model(config: try targetConfiguration()))
        XCTAssertEqual(tokens.shape, [1, 3])
        XCTAssertTrue(tokens.asArray(Int32.self).allSatisfy { (0..<64).contains($0) })
    }

    func testGreedyDecodeAcceptsVerifiedBlocksAndRejectsWrongProposals() throws {
        for reject in [false, true] {
            for affine in [false, true] {
                let (target, draft) = try constantModels(reject: reject)
                let cache = caches(affine: affine)
                let prompt = MLXArray([Int32(2), 3, 4]).reshaped(1, 3)
                let output = target.forwardPrefill(prompt, cache: cache, captureLayerIndices: [0, 1])
                let history = draft.makeCache()
                draft.appendTargetContext(output.capturedHiddenStates, cache: history)
                var generated: [Int] = []
                let stats = try Q35DFlash2Decoder.decode(
                    model: target, draft: draft, draftCache: history, initialLogits: output.logits,
                    layerCaches: cache, eosSet: [], tokenBudget: 17, emit: { generated.append($0) }
                )
                XCTAssertEqual(generated, Array(repeating: 0, count: 17))
                XCTAssertGreaterThan(stats.rounds, 0)
                if reject {
                    XCTAssertEqual(stats.acceptedTokens, 0)
                    XCTAssertEqual(history.map(\.offset), [19])
                    XCTAssertEqual(stats.recoveryForwards > 0, affine)
                } else {
                    XCTAssertGreaterThan(stats.acceptedTokens, 0)
                }
            }
        }
    }

    func testSampledDecodeCommitsOnlyAcceptedHistoryAndRecoversQuantizedCaches() throws {
        for affine in [false, true] {
            MLXRandom.seed(71)
            let (target, draft) = try constantModels(reject: true)
            let cache = caches(affine: affine)
            let output = target.forwardPrefill(
                MLXArray([Int32(2), 3, 4]).reshaped(1, 3), cache: cache, captureLayerIndices: [0, 1]
            )
            let history = draft.makeCache()
            draft.appendTargetContext(output.capturedHiddenStates, cache: history)
            let sampling = Q35DFlash2Sampling(temperature: 0.7, history: [2, 3, 4]) { _, prefix in
                // A deterministic target distribution exercises sampled
                // rejection while making the correct emitted history exact.
                let next = (prefix![-1] + 1) % 64
                let probabilities = MLXArray.zeros([64])
                probabilities[next] = MLXArray(Float(1))
                return probabilities
            }
            var generated: [Int] = []
            let stats = try Q35DFlash2Decoder.decode(
                model: target, draft: draft, draftCache: history, initialLogits: output.logits,
                layerCaches: cache, eosSet: [], tokenBudget: 12, sampling: sampling,
                emit: { generated.append($0) }
            )
            XCTAssertEqual(generated, Array(5...16))
            XCTAssertEqual(stats.acceptedTokens, 0)
            XCTAssertEqual(stats.recoveryForwards > 0, affine)
            XCTAssertEqual(history.map(\.offset), [14])
        }
    }

    func testSampledDecodeAcceptsBlocksAndStopsAtSampledEOS() throws {
        let (target, draft) = try constantModels(reject: false)
        for eos in [Set<Int>(), [0]] {
            let cache = caches()
            let output = target.forwardPrefill(
                MLXArray([Int32(3)]).reshaped(1, 1), cache: cache, captureLayerIndices: [0, 1]
            )
            let history = draft.makeCache()
            draft.appendTargetContext(output.capturedHiddenStates, cache: history)
            let sampling = Q35DFlash2Sampling(temperature: 1, history: nil) { _, _ in
                let probabilities = MLXArray.zeros([64])
                probabilities[0] = MLXArray(Float(1))
                return probabilities
            }
            MLXRandom.seed(113)
            var generated: [Int] = []
            let stats = try Q35DFlash2Decoder.decode(
                model: target, draft: draft, draftCache: history, initialLogits: output.logits,
                layerCaches: cache, eosSet: eos, tokenBudget: 17, sampling: sampling,
                emit: { generated.append($0) }
            )
            XCTAssertEqual(generated, eos.isEmpty ? Array(repeating: 0, count: 17) : [])
            if eos.isEmpty { XCTAssertGreaterThan(stats.acceptedTokens, 0) }
        }
    }

    func testSampledHybridVerificationMatchesIndependentSerialConditionalLogits() throws {
        MLXRandom.seed(210)
        let target = Q35Model(config: try targetConfiguration())
        target.update(parameters: target.parameters().mapValues { MLXRandom.normal($0.shape) * 0.1 })
        let oracle = Q35Model(config: try targetConfiguration())
        try oracle.update(parameters: target.parameters(), verify: .all)
        let draft = Q35DFlash2Model(config: try draftConfiguration())
        let cache = caches()
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let output = target.forwardPrefill(prompt, cache: cache, captureLayerIndices: [0, 1])
        let history = draft.makeCache()
        draft.appendTargetContext(output.capturedHiddenStates, cache: history)
        var comparisons = 0
        let sampling = Q35DFlash2Sampling(temperature: 1, history: [1, 2, 3]) { logits, prefix in
            let tokens = prefix!.asArray(Int32.self)
            // Recreate the conditional state with independent one-token forwards.
            let serialCache = self.caches()
            var expected = oracle.forwardPrefill(prompt, cache: serialCache)
            for token in tokens.dropFirst(3) {
                expected = oracle.forward(MLXArray([token]).reshaped(1, 1), cache: serialCache)
            }
            self.assertClose(logits, expected.logits[0, -1, 0...])
            comparisons += 1
            return softmax(logits.asType(.float32))
        }
        var generated: [Int] = []
        let stats = try Q35DFlash2Decoder.decode(
            model: target, draft: draft, draftCache: history, initialLogits: output.logits,
            layerCaches: cache, eosSet: [], tokenBudget: 9, sampling: sampling, emit: { generated.append($0) }
        )
        XCTAssertEqual(generated.count, 9)
        XCTAssertGreaterThan(comparisons, stats.rounds)
    }

    func testSampledSeededRequestsRepeatWithoutHistoryLeakage() throws {
        let (target, draft) = try constantModels(reject: false)
        let sampling = Q35DFlash2Sampling(temperature: 1, history: nil) { _, _ in
            MLXArray([Float(0.2), 0.3, 0.5] + Array(repeating: Float(0), count: 61))
        }
        var runs: [[Int]] = []
        for _ in 0..<2 {
            let cache = caches()
            let output = target.forwardPrefill(
                MLXArray([Int32(3)]).reshaped(1, 1), cache: cache, captureLayerIndices: [0, 1]
            )
            let history = draft.makeCache()
            draft.appendTargetContext(output.capturedHiddenStates, cache: history)
            MLXRandom.seed(577)
            var tokens: [Int] = []
            _ = try Q35DFlash2Decoder.decode(
                model: target, draft: draft, draftCache: history, initialLogits: output.logits,
                layerCaches: cache, eosSet: [], tokenBudget: 32, sampling: sampling, emit: { tokens.append($0) }
            )
            runs.append(tokens)
        }
        XCTAssertEqual(runs[0], runs[1])
        XCTAssertGreaterThan(Set(runs[0]).count, 1)
        XCTAssertTrue(Set(runs[0]).isSubset(of: [0, 1, 2]))
    }

    func testEOSBudgetAndCancellation() throws {
        let (target, draft) = try constantModels(reject: false)
        let output = target.forwardPrefill(MLXArray([Int32(3)]).reshaped(1, 1), cache: caches())
        for (budget, eos, expected) in [(0, Set<Int>(), []), (1, [], [0]), (8, [0], [])] {
            var tokens: [Int] = []
            let stats = try Q35DFlash2Decoder.decode(
                model: target, draft: draft, draftCache: draft.makeCache(), initialLogits: output.logits,
                layerCaches: caches(), eosSet: eos, tokenBudget: budget, emit: { tokens.append($0) }
            )
            XCTAssertEqual(tokens, expected)
            XCTAssertEqual(stats.rounds, 0)
        }
        XCTAssertThrowsError(try Q35DFlash2Decoder.decode(
            model: target, draft: draft, draftCache: draft.makeCache(), initialLogits: output.logits,
            layerCaches: caches(), eosSet: [], tokenBudget: 8,
            checkCancellation: { throw CancellationError() }, emit: { _ in XCTFail("Cancelled request emitted a token") }
        ))
    }

    func testPartialAcceptanceMatchesSerialWithNonzeroHybridState() throws {
        MLXRandom.seed(908)
        let target = Q35Model(config: try targetConfiguration())
        target.update(parameters: target.parameters().mapValues { MLXRandom.normal($0.shape) * 0.01 })
        var embedding = [Float](repeating: 0, count: 64 * 32)
        var head = embedding
        for token in 0..<64 { embedding[token * 32 + token % 32] = 1 }
        for token in 0..<32 { head[token * 32 + (token + 31) % 32] = 10 }
        try target.update(parameters: ModuleParameters.unflattened([
            ("model.embed_tokens.weight", MLXArray(embedding, [64, 32])),
            ("lm_head.weight", MLXArray(head, [64, 32])),
        ]), verify: .noUnusedKeys)
        let configData = Self.smallDraft
            .replacingOccurrences(of: "\"selector_rank\":1", with: "\"selector_rank\":32")
            .replacingOccurrences(of: "\"selector_top_k\":2", with: "\"selector_top_k\":64")
        let draft = Q35DFlash2Model(config: try JSONDecoder().decode(
            Q35DFlash2Config.self, from: Data(configData.utf8)
        ))
        draft.update(parameters: draft.parameters().mapValues { MLXArray.zeros(like: $0) })
        var predecessor = embedding
        var successor = embedding
        for index in (32 * 32)..<(64 * 32) { successor[index] = 0 }
        for token in 0..<64 {
            for column in 0..<32 { predecessor[token * 32 + column] = 0 }
            predecessor[token * 32 + (token == 6 ? 0 : (token + 1) % 32)] = 1000
        }
        try draft.update(parameters: ModuleParameters.unflattened([
            ("norm.weight", MLXArray.ones([32])),
            ("candidate_selector.hidden_projection.weight", MLXArray.ones([32, 32])),
            ("candidate_selector.predecessor_codebook.weight", MLXArray(predecessor, [64, 32])),
            ("candidate_selector.successor_codebook.weight", MLXArray(successor, [64, 32])),
        ]), verify: .noUnusedKeys)
        let prompt = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let serialCache = caches()
        let serialHistory = draft.makeCache()
        var serial = target.forwardPrefill(prompt, cache: serialCache, captureLayerIndices: [0, 1])
        draft.appendTargetContext(serial.capturedHiddenStates, cache: serialHistory)
        var snapshots = [3: serialHistory[0].fork()]
        var expected: [Int] = []
        for index in 0..<17 {
            let token = argMax(serial.logits[0, -1, 0...]).item(Int.self)
            expected.append(token)
            if index < 16 {
                serial = target.forward(
                    MLXArray([Int32(token)]).reshaped(1, 1), cache: serialCache, captureLayerIndices: [0, 1]
                )
                draft.appendTargetContext(serial.capturedHiddenStates, cache: serialHistory)
                snapshots[serialHistory[0].offset] = serialHistory[0].fork()
            }
        }
        let candidateCache = caches()
        let initial = target.forwardPrefill(prompt, cache: candidateCache, captureLayerIndices: [0, 1])
        let history = draft.makeCache()
        draft.appendTargetContext(initial.capturedHiddenStates, cache: history)
        var actual: [Int] = []
        var events: [String] = []
        var observedAccepted = 0
        let stats = try Q35DFlash2Decoder.decode(
            model: target, draft: draft, draftCache: history, initialLogits: initial.logits,
            layerCaches: candidateCache, eosSet: [], tokenBudget: 17,
            observer: { event in
                switch event {
                case .beginRound: events.append("begin")
                case let .drafted(tokens):
                    events.append("draft")
                    MLX.eval(tokens)
                case .verificationSubmitted: events.append("submit")
                case .verificationCompleted: events.append("verified")
                case let .accepted(count):
                    events.append("accept")
                    observedAccepted += count
                case let .finished(state):
                    events.append("finish")
                    MLX.eval(state)
                }
            }, emit: { actual.append($0) }
        )
        XCTAssertEqual(actual, expected)
        XCTAssertEqual(observedAccepted, stats.acceptedTokens)
        XCTAssertEqual(events, (0..<stats.rounds).flatMap { _ in
            ["begin", "draft", "submit", "verified", "accept", "finish"]
        })
        XCTAssertEqual(expected, Array(4...20))
        XCTAssertGreaterThan(stats.acceptedTokens, 0)
        XCTAssertLessThan(stats.acceptedTokens, stats.draftedTokens)
        XCTAssertEqual(stats.recoveryForwards, 0)
        // Finishing the budget need not commit the final block. Compare the
        // last confirmed snapshot with independent one-token target execution.
        let reference = try XCTUnwrap(snapshots[history[0].offset])
        assertClose(try XCTUnwrap(history[0].keys), try XCTUnwrap(reference.keys))
        assertClose(try XCTUnwrap(history[0].values), try XCTUnwrap(reference.values))
    }

    private func constantModels(reject: Bool) throws -> (Q35Model, Q35DFlash2Model) {
        let target = Q35Model(config: try targetConfiguration())
        target.update(parameters: target.parameters().mapValues { MLXArray.zeros(like: $0) })
        var head = [Float](repeating: 0, count: 64 * 32)
        for index in 0..<32 {
            head[index] = 1
            head[32 + index] = 0.5
        }
        try target.update(parameters: ModuleParameters.unflattened([
            ("model.embed_tokens.weight", MLXArray.ones([64, 32])),
            ("lm_head.weight", MLXArray(head, [64, 32])),
        ]), verify: .noUnusedKeys)
        let draft = Q35DFlash2Model(config: try draftConfiguration())
        draft.update(parameters: draft.parameters().mapValues { MLXArray.zeros(like: $0) })
        var successors = [Float](repeating: 0, count: 64)
        if reject { successors[1] = 100 }
        try draft.update(parameters: ModuleParameters.unflattened([
            ("norm.weight", MLXArray.ones([32])),
            ("candidate_selector.hidden_projection.weight", MLXArray.ones([1, 32])),
            ("candidate_selector.predecessor_codebook.weight", MLXArray.ones([64, 1])),
            ("candidate_selector.successor_codebook.weight", MLXArray(successors, [64, 1])),
        ]), verify: .noUnusedKeys)
        return (target, draft)
    }

    private func caches(affine: Bool = false) -> [Q35LayerCache?] {
        let full: KVCache = affine ? AffineQuantizedKVCache(groupSize: 32, bits: 8) : KVCacheSimple()
        return [.linear(Q35LinearCache()), .full(full)]
    }

    private func assertClose(_ lhs: MLXArray, _ rhs: MLXArray) {
        XCTAssertLessThan((lhs - rhs).abs().max().item(Float.self), 1e-4)
    }

    private func targetConfiguration() throws -> Q35Config {
        try JSONDecoder().decode(Q35Config.self, from: Data(Self.smallTarget.utf8))
    }

    private func draftConfiguration() throws -> Q35DFlash2Config {
        try JSONDecoder().decode(Q35DFlash2Config.self, from: Data(Self.smallDraft.utf8))
    }

    private static let smallTarget = #"""
    {"model_type":"qwen3_5","text_config":{
      "model_type":"qwen3_5_text","hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,
      "num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,
      "layer_types":["linear_attention","full_attention"],
      "linear_num_key_heads":1,"linear_num_value_heads":2,
      "linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,
      "attention_bias":false,"attention_dropout":0,"attn_output_gate":true,"eos_token_id":63,"vocab_size":64,
      "max_position_embeddings":4096,"rms_norm_eps":0.000001,
      "rope_parameters":{"rope_theta":10000,"partial_rotary_factor":1}}}
    """#

    private static let smallDraft = #"""
    {"architectures":["DFlash2DraftModel"],"model_type":"qwen3","is_causal":false,
     "attention_bias":false,"hidden_act":"silu","tie_word_embeddings":false,
     "hidden_size":32,"intermediate_size":64,"num_hidden_layers":1,"num_target_layers":2,
     "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":64,
     "rms_norm_eps":0.000001,"sliding_window":4,"layer_types":["sliding_attention"],
     "rope_parameters":{"rope_theta":10000,"rope_type":"default"},
     "dflash_config":{"block_size":4,"conv_group_size":16,"conv_kernel_size":2,
       "mask_token_id":63,"selector_rank":1,"selector_top_k":2,"target_layer_ids":[0,1]}}
    """#

    // incoai/Qwen3.8-27B-DFlash2@015e795645c74b1a0eeef3b570031fb62e769bc5
    private static let published = #"""
    {
      "architectures": [
        "DFlash2DraftModel"
      ],
      "attention_bias": false,
      "attention_dropout": 0.0,
      "bos_token_id": null,
      "is_causal": false,
      "dflash_config": {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 248070,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [
          5,
          19,
          33,
          47,
          61
        ]
      },
      "dtype": "bfloat16",
      "eos_token_id": 248044,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 5120,
      "initializer_range": 0.02,
      "intermediate_size": 17408,
      "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention"
      ],
      "max_position_embeddings": 262144,
      "max_window_layers": 5,
      "model_type": "qwen3",
      "num_attention_heads": 32,
      "num_hidden_layers": 5,
      "num_key_value_heads": 8,
      "num_target_layers": 64,
      "pad_token_id": 248044,
      "rms_norm_eps": 1e-06,
      "rope_parameters": {
        "rope_theta": 10000000,
        "rope_type": "default"
      },
      "sliding_window": 2048,
      "tie_word_embeddings": false,
      "transformers_version": "5.15.0",
      "use_cache": true,
      "use_sliding_window": true,
      "vocab_size": 248320
    }
    """#
}
