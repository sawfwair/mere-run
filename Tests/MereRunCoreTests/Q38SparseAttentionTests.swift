import Foundation
import MLX
import MLXFast
import MLXRandom
import XCTest
@testable import MereRunCore

final class Q38SparseAttentionTests: MereRunCoreTestCase {
    func testSelectorIncludesTopCompleteBlocksAndCurrentPartialBlock() {
        let selection = Q38SparseAttention.select(
            scores: MLXArray([Float(1), 9, 2, 8]).reshaped(1, 1, 4),
            offsets: [16], budget: 8, ratio: 4
        )
        XCTAssertEqual(selectedTokens(selection), Set(Array(4..<8) + Array(12..<17)))
    }

    func testSelectorMasksFutureBlocksBeforeTopKAndSeparatesBatchRows() {
        let selection = Q38SparseAttention.select(
            scores: MLXArray([Float(1), 2, 3, 100, 1, 2, 3, 100]).reshaped(2, 1, 4),
            offsets: [8, 16], budget: 8, ratio: 4
        )
        XCTAssertEqual(selectedTokens(selection, row: 0), Set(0..<9))
        XCTAssertEqual(selectedTokens(selection, row: 1), Set(8..<17))
    }

    func testSelectorCoversFirstTokensAndExact2048Boundary() {
        let first = Q38SparseAttention.select(
            scores: MLXArray.ones([1, 5, 4]), offsets: [0], budget: 8, ratio: 4
        )
        for query in 0..<5 {
            XCTAssertEqual(selectedTokens(first, query: query), Set(0...query))
        }
        let boundary = Q38SparseAttention.select(
            scores: MLXArray.ones([1, 2, 512]), offsets: [2_047], budget: 2_048, ratio: 4
        )
        XCTAssertEqual(selectedTokens(boundary), Set(0..<2_048))
        XCTAssertEqual(selectedTokens(boundary, query: 1), Set(0..<2_049))
    }

    func testGatheredAttentionMatchesDenseMaskedGQAForEachBatchRow() {
        MLXRandom.seed(81)
        let queries = MLXRandom.normal([2, 4, 3, 4])
        let keys = MLXRandom.normal([2, 2, 12, 4])
        let values = MLXRandom.normal([2, 2, 12, 4])
        let selection = Q38SparseAttention.select(
            scores: MLXRandom.normal([2, 3, 3]), offsets: [5, 9], budget: 4, ratio: 4
        )
        let actual = Q38SparseAttention.attend(
            queries: queries, keys: keys, values: values,
            indices: selection.indices, valid: selection.valid, scale: 0.5
        )
        for row in 0..<2 {
            for query in 0..<3 {
                let selected = selectedTokens(selection, row: row, query: query)
                let mask = MLXArray((0..<12).map { selected.contains($0) ? Float(0) : -Float.infinity })
                    .reshaped(1, 1, 1, 12)
                let expected = MLXFast.scaledDotProductAttention(
                    queries: queries[row..<(row + 1), 0..., query..<(query + 1), 0...],
                    keys: keys[row..<(row + 1), 0..., 0..., 0...],
                    values: values[row..<(row + 1), 0..., 0..., 0...], scale: 0.5, mask: .array(mask)
                )
                assertClose(actual[row..<(row + 1), 0..., query..<(query + 1), 0...], expected)
            }
        }
    }

    func testCompressionPoolsRawKeysBeforeNormalizationAndUsesFirstPosition() throws {
        let indexer = Q38QSAIndexer(config: try config())
        let raw = MLXArray((0..<32).map { Float($0 + 1) }).reshaped(1, 1, 8, 4)
        let positions = Q38QSAIndexer.positionRows(
            batch: 1, count: 8, offsets: [0],
            positionIds: MLXArray([Int32(5), 6, 9, 10, 17, 18, 21, 22]).reshaped(1, 8)
        )
        let actual = indexer.compressedKeys(raw, positions: positions)
        var expected: [Float] = []
        for group in 0..<2 {
            let mean = (0..<4).map { Float(group * 16 + $0 + 7) }
            let norm = sqrt(mean.reduce(Float(0)) { $0 + $1 * $1 } / 4 + 0.000001)
            let unit = mean.map { $0 / norm }
            let angle = Float(group == 0 ? 5 : 17)
            expected.append(contentsOf: [
                unit[0] * cos(angle) - unit[1] * sin(angle),
                unit[1] * cos(angle) + unit[0] * sin(angle), unit[2], unit[3],
            ])
        }
        assertClose(actual, MLXArray(expected).reshaped(1, 1, 2, 4))
    }

    func testLongPrefillMatchesChunkedAndSerialAttentionAcrossBudget() throws {
        MLXRandom.seed(82)
        let attention = Q35FullAttention(config: try config())
        let input = MLXRandom.normal([1, 25, 8])
        let expected = forward(attention, input, Q38QSACache())
        let chunkCache = Q38QSACache()
        var chunks: [MLXArray] = []
        for bounds in [0..<4, 4..<11, 11..<25] {
            chunks.append(forward(attention, input[0..., bounds, 0...], chunkCache))
        }
        assertClose(MLX.concatenated(chunks, axis: 1), expected)
        let serialCache = Q38QSACache()
        let serial = (0..<25).map {
            forward(attention, input[0..., $0..<($0 + 1), 0...], serialCache)
        }
        assertClose(MLX.concatenated(serial, axis: 1), expected)
    }

    func testPublishedBudgetAt32KUsesSparseHistoryAndMatchesFreshPrefillTail() throws {
        MLXRandom.seed(87)
        let attention = Q35FullAttention(config: try config(budget: 2_048))
        // Populate a long history without running every preceding query. Then
        // compare the final query against a real prefill across the 2K boundary.
        let input = MLXRandom.normal([1, 2_053, 8])
        let expected = forward(attention, input, Q38QSACache())
        let cache = Q38QSACache()
        _ = forward(attention, input[0..., 0..<2_047, 0...], cache)
        let actual = forward(attention, input[0..., 2_047..., 0...], cache)
        assertClose(actual, expected[0..., 2_047..., 0...])

        let longCache = Q38QSACache()
        let historyLength = 32_768
        let keys = MLXRandom.normal([1, 2, historyLength, 4])
        let values = MLXRandom.normal([1, 2, historyLength, 4])
        let indexKeys = MLXRandom.normal([1, 1, historyLength, 4])
        let positions = Q38QSAIndexer.positionRows(
            batch: 1, count: historyLength, offsets: [0], positionIds: nil
        )
        _ = longCache.update(keys: keys, values: values)
        _ = longCache.updateIndexer(keys: indexKeys, positions: positions)
        let output = forward(attention, MLXRandom.normal([1, 1, 8]), longCache)
        XCTAssertTrue(MLX.abs(output).max().item(Float.self).isFinite)
        XCTAssertEqual(longCache.offset, 32_769)
    }

    func testQuantizedMainCacheForkRetainsSeparateUnquantizedIndexerState() throws {
        MLXRandom.seed(88)
        let cache = Q38QSACache(attention: AffineQuantizedKVCache(groupSize: 64, bits: 4, step: 256))
        let keys = MLXRandom.normal([1, 1, 9, 64])
        let indexKeys = MLXRandom.normal([1, 1, 9, 4])
        let positions = Q38QSAIndexer.positionRows(batch: 1, count: 9, offsets: [0], positionIds: nil)
        _ = cache.update(keys: keys, values: -keys)
        _ = cache.updateIndexer(keys: indexKeys, positions: positions)
        XCTAssertFalse(Q35LayerCache.full(cache).canRestoreVerificationPrefix(totalTokens: 3, tokenCount: 1))
        let fork = try XCTUnwrap(cache.fork() as? Q38QSACache)
        let extra = MLXArray.ones([1, 1, 1, 64])
        _ = fork.update(keys: extra, values: extra)
        let history = fork.updateIndexer(
            keys: MLXArray.ones([1, 1, 1, 4]),
            positions: Q38QSAIndexer.positionRows(batch: 1, count: 1, offsets: [9], positionIds: nil)
        )
        assertClose(history.0[0..., 0..., 0..<9, 0...], indexKeys)
        XCTAssertEqual(history.1[0, 0, 9, 0].item(Int32.self), 9)
        XCTAssertEqual(cache.offset, 9)
        XCTAssertEqual(fork.offset, 10)
    }

    func testShortPrefixForkKeepsIndexerHistoryAndIsolatesBranches() throws {
        MLXRandom.seed(83)
        let attention = Q35FullAttention(config: try config())
        let prefix = MLXRandom.normal([1, 7, 8])
        let suffix = MLXRandom.normal([1, 11, 8])
        let original = Q38QSACache()
        _ = forward(attention, prefix, original)
        let fork = try XCTUnwrap(original.fork() as? Q38QSACache)
        let actual = forward(attention, suffix, fork)
        let fresh = forward(attention, MLX.concatenated([prefix, suffix], axis: 1), Q38QSACache())
        assertClose(actual, fresh[0..., 7..., 0...])
        XCTAssertEqual(original.offset, 7)
        let other = -suffix
        let otherActual = forward(attention, other, original)
        let otherFresh = forward(attention, MLX.concatenated([prefix, other], axis: 1), Q38QSACache())
        assertClose(otherActual, otherFresh[0..., 7..., 0...])
    }

    func testRejectedDraftRollbackRestoresBothHistoriesAcrossBlockBoundary() throws {
        MLXRandom.seed(84)
        let attention = Q35FullAttention(config: try config())
        let prefix = MLXRandom.normal([1, 11, 8])
        let verified = MLXRandom.normal([1, 6, 8])
        let next = MLXRandom.normal([1, 3, 8])
        let cache = Q38QSACache()
        _ = forward(attention, prefix, cache)
        _ = forward(attention, verified, cache)
        let layerCache = Q35LayerCache.full(cache)
        XCTAssertTrue(layerCache.restoreVerificationPrefix(totalTokens: 6, tokenCount: 2))
        XCTAssertEqual(cache.offset, 13)
        let actual = forward(attention, next, cache)
        let fresh = forward(attention, MLX.concatenated([
            prefix, verified[0..., 0..<2, 0...], next,
        ], axis: 1), Q38QSACache())
        assertClose(actual, fresh[0..., 13..., 0...])
    }

    func testRaggedBatchAndUnbatchMatchIndependentCachedRequests() throws {
        MLXRandom.seed(85)
        let attention = Q35FullAttention(config: try config())
        let caches = [Q38QSACache(), Q38QSACache()]
        _ = forward(attention, MLXRandom.normal([1, 5, 8]), caches[0])
        _ = forward(attention, MLXRandom.normal([1, 19, 8]), caches[1])
        let batched = try XCTUnwrap(caches[0].batched(with: caches) as? Q38QSACache)
        XCTAssertEqual(batched.rowOffsets, [5, 19])
        let input = MLXRandom.normal([2, 3, 8])
        let expected = caches.enumerated().map { row, cache in
            forward(attention, input[row..<(row + 1), 0..., 0...], cache)
        }
        assertClose(forward(attention, input, batched), MLX.concatenated(expected, axis: 0))
        let split = try XCTUnwrap(batched.unbatchedRows(count: 2) as? [Q38QSACache])
        XCTAssertEqual(split.map(\.offset), [8, 22])
        for row in 0..<2 {
            let next = MLXRandom.normal([1, 2, 8])
            assertClose(forward(attention, next, split[row]), forward(attention, next, caches[row]))
        }
    }

    func testMTPHistoryPrimingIsChunkedAndKeepsOnlyCommittedTransitions() throws {
        MLXRandom.seed(86)
        let config = try config()
        let model = Q35Model(config: config)
        let mtp = Q38MTPModel(config: config)
        let tokens = (0..<300).map { $0 % 30 }
        let hidden = MLXRandom.normal([1, 300, 32])
        let cache = Q38QSACache()
        let session = Q35MTPDraftSession(promptTokens: tokens, promptHidden: hidden, historyCache: cache)
        let drafts = mtp.draftBlock(
            lastToken: 4, hidden: hidden[0..., 299..., 0...], blockSize: 4,
            session: session, baseModel: model
        )
        XCTAssertEqual(drafts.count, 3)
        XCTAssertTrue(drafts.tokens.allSatisfy { $0 >= 0 && $0 < 32 })
        XCTAssertEqual(session.committedHistoryCount, 300)
        XCTAssertEqual(cache.offset, 300)
        let referenceCache = Q38QSACache()
        let nextTokens = MLXArray((Array(tokens.dropFirst()) + [4]).map(Int32.init)).reshaped(1, 300)
        let reference = mtp.forwardDraft(
            inputEmbeddings: model.embeddings(for: nextTokens), hiddenStates: hidden, cache: referenceCache
        )
        let expectedToken = model.greedyDraftToken(from: reference.logitsHidden[0..., 299..., 0...])
        XCTAssertEqual(drafts.tokens[0], Int(expectedToken.item(Int32.self)))
        let continuation = MLXRandom.normal([1, 1, 8])
        let hiddenContinuation = MLXRandom.normal([1, 1, 32])
        let actual = mtp.forwardDraft(inputEmbeddings: continuation, hiddenStates: hiddenContinuation, cache: cache)
        let expected = mtp.forwardDraft(inputEmbeddings: continuation, hiddenStates: hiddenContinuation, cache: referenceCache)
        assertClose(actual.logitsHidden, expected.logitsHidden)
    }

    func testRejectedMTPDraftRestoresMultiStreamHiddenBeforeNextProposal() throws {
        MLXRandom.seed(89)
        let config = try config()
        let model = Q35Model(config: config)
        let mtp = Q38MTPModel(config: config)
        let prompt = Array(1...11)
        for accepted in 0..<3 {
            let cache = Q38QSACache()
            let output = model.forward(
                MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count), cache: [.full(cache)]
            )
            let promptHidden = try XCTUnwrap(output.mtpHidden)
            let session = Q35MTPDraftSession(
                promptTokens: prompt, promptHidden: promptHidden, historyCache: Q38QSACache()
            )
            let proposals = mtp.draftBlock(
                lastToken: 12, hidden: promptHidden[0..., 10..., 0...], blockSize: 4,
                session: session, baseModel: model
            ).tokens
            let candidate = model.forward(
                MLXArray(([12] + proposals).map(Int32.init)).reshaped(1, 4),
                cache: [.full(cache)], targetVerify: true
            )
            XCTAssertTrue(Q35LayerCache.full(cache).restoreVerificationPrefix(
                totalTokens: 4, tokenCount: accepted + 1
            ))
            let restored = session.restoredVerificationState(
                from: candidate, acceptedTokens: Array(proposals.prefix(accepted))
            )
            XCTAssertEqual(restored.hidden.shape, [1, 1, 32])
            assertClose(restored.hidden, try XCTUnwrap(candidate.mtpHidden)[0..., accepted..<(accepted + 1), 0...])
            assertClose(restored.logits, candidate.logits[0..., accepted..<(accepted + 1), 0...])
            let next = mtp.draftBlock(
                lastToken: 13, hidden: restored.hidden, blockSize: 4,
                session: session, baseModel: model
            )
            XCTAssertEqual(next.tokens.count, 3)
            XCTAssertEqual(session.committedHistoryCount, prompt.count + accepted + 1)
        }
    }

    private func forward(_ attention: Q35FullAttention, _ input: MLXArray, _ cache: Q38QSACache) -> MLXArray {
        let output = attention(input, mask: cache.makeMask(n: input.dim(1)), cache: cache)
        MLX.eval(output)
        return output
    }

    private func assertClose(_ actual: MLXArray, _ expected: MLXArray, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertLessThan(MLX.abs(actual - expected).max().item(Float.self), 0.0001, file: file, line: line)
    }

    private func selectedTokens(
        _ selection: (indices: MLXArray, valid: MLXArray), row: Int = 0, query: Int = 0
    ) -> Set<Int> {
        let indices = selection.indices[row, query, 0...].asArray(Int32.self)
        let valid = selection.valid[row, query, 0...].asArray(Bool.self)
        return Set(zip(indices, valid).compactMap { $1 ? Int($0) : nil })
    }

    private func config(budget: Int = 8) throws -> Q35Config {
        try JSONDecoder().decode(Q35Config.self, from: Data(#"""
        {
          "model_type": "qwen4_exp", "architectures": ["Qwen4ExpForConditionalGeneration"],
          "tie_word_embeddings": false,
          "text_config": {
            "model_type": "qwen4_exp_text", "hidden_size": 8, "num_hidden_layers": 1,
            "intermediate_size": 8, "num_experts": 2, "num_experts_per_tok": 1,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
            "layer_types": ["full_attention"], "linear_num_value_heads": 1,
            "linear_num_key_heads": 1, "linear_key_head_dim": 4, "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 2, "max_position_embeddings": 4096,
            "rms_norm_eps": 0.000001, "attention_bias": false, "attention_dropout": 0,
            "attn_output_gate": true, "output_gate_type": "sigmoid", "hc_count": 4, "hc_lowrank": 4,
            "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 4,
            "indexer_budget": \#(budget), "indexer_compress_ratio": 4, "ple_layer_ids": [],
            "vocab_size": 32, "eos_token_id": 31,
            "mtp": {"hybrid": true, "layer_types": ["full_attention"], "num_hidden_layers": 1},
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.5}
          }
        }
        """#.utf8))
    }
}
