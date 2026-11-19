import Foundation
import MLX
import MLXRandom
import MereRunKVCache
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel
@testable import MereRunTensor

final class Q35TreeVerificationTests: MLXTestCase {
    func testCompactVocabularyMapsCandidatesBeforeLearnedEdges() {
        MLXRandom.seed(1942)
        let selector = DFlash2CandidateSelector(vocabularySize: 64, hiddenSize: 32, rank: 32, topK: 4)
        let ids: [Int32] = [2, 5, 8, 11, 24, 31, 37, 41, 48, 53, 59, 63]
        let vocabulary = MLXArray(ids)
        let mask = MLXArray((0..<64).map { ids.contains(Int32($0)) })
        for depth in [1, 7, 15] {
            let hidden = MLXRandom.normal([1, depth, 32])
            let full = MLXRandom.normal([1, depth, 64])
            let compact = take(full, vocabulary, axis: -1)
            let expected = selector.tree(hidden: hidden,
                logits: which(mask, full, MLXArray(-Float.infinity)), anchor: 1, nodeBudget: 15)
            let actual = selector.tree(hidden: hidden, logits: compact, anchor: 1,
                nodeBudget: 15, vocabulary: vocabulary)
            XCTAssertEqual(actual.tokens, expected.tokens)
            XCTAssertEqual(actual.parents, expected.parents)
            for (a, b) in zip(actual.logProbabilities, expected.logProbabilities) {
                XCTAssertEqual(a, b, accuracy: 1e-5)
            }
        }
    }

    func testCompactDraftRowsIncludeWholeControlTilesAndExcludePadding() {
        let full = MLXArray(0..<Q35DFlash2DraftHead.vocabularySize).reshaped(-1, 1)
        let selected = Q35DFlash2DraftHead.rows(full).asArray(Int32.self)
        let real = Array(0..<98_304) + Array(248_032..<248_320)
        XCTAssertEqual(selected.count, 98_624)
        XCTAssertEqual(Array(selected.prefix(98_592)), real.map(Int32.init))
        XCTAssertEqual(Array(selected.suffix(32)), (0..<32).map(Int32.init))
    }

    func testEmptyCommittedPrefixAfterRollbackDoesNotExportZeroLengthBuffers() {
        let cache = KVCacheSimple(step: 4)
        let values = MLXArray.ones([1, 2, 3, 4])
        _ = cache.update(keys: values, values: values)
        XCTAssertNotNil(cache.committedVerificationState)
        cache.rollback(toOffset: 0)
        XCTAssertNil(cache.committedVerificationState)
        _ = cache.update(keys: values, values: -values)
        XCTAssertNotNil(cache.committedVerificationState)
    }

    func testBatchedLatticePreservesDepthAndPredecessorScores() {
        MLXRandom.seed(719)
        let selector = DFlash2CandidateSelector(vocabularySize: 64, hiddenSize: 32, rank: 32, topK: 4)
        for depth in [1, 7, 15] {
            let hidden = MLXRandom.normal([1, depth, 32])
            let logits = MLXRandom.normal([1, depth, 64])
            let candidates = argPartition(-logits, kth: 3, axis: -1)[0..., 0..., 0..<4]
            let unary = takeAlong(logits, candidates, axis: -1)
            let projected = selector.hiddenProjection(hidden)
            let scores = stacked((0..<depth).map { position in
                let parents = position == 0 ? MLXArray([Int32(1), 1, 1, 1])
                    : candidates[0, position - 1, 0...].asType(.int32)
                let edge = (selector.predecessorCodebook(parents).expandedDimensions(axis: 1)
                    * projected[0, position, 0...]
                    * selector.successorCodebook(candidates[0, position, 0...])
                        .expandedDimensions(axis: 0)).sum(axis: -1)
                let conditional = (edge + unary[0, position, 0...]).asType(.float32)
                return conditional - logSumExp(conditional, axis: -1, keepDims: true)
            })
            let expected = DFlash2CandidateSelector.buildTree(anchor: 1,
                candidates: candidates.asArray(Int32.self).map(Int.init), scores: scores.asArray(Float.self),
                depthCount: depth, candidateCount: 4, nodeBudget: 15)
            let actual = selector.tree(hidden: hidden, logits: logits, anchor: 1, nodeBudget: 15)
            XCTAssertEqual(actual.tokens, expected.tokens)
            XCTAssertEqual(actual.parents, expected.parents)
            for (a, b) in zip(actual.logProbabilities, expected.logProbabilities) {
                XCTAssertEqual(a, b, accuracy: 1e-5)
            }
        }
    }

    func testParentsAndNonPrefixTargetPath() throws {
        for parents in [[], [0], [-1, 1], [-1, -1], [-1, 0, 3]] {
            XCTAssertThrowsError(try Q35VerificationTree(parents: parents))
        }
        let proposal = DFlash2CandidateSelector.buildTree(anchor: 1, candidates: [10, 11, 20, 21, 30, 31],
            scores: [-0.1, -1.5, -0.1, -1.5, -1.9, -0.2, -0.05, -3, -0.1, -2, -3, -0.05],
            depthCount: 3, candidateCount: 2, nodeBudget: 6)
        XCTAssertEqual(proposal.tokens, [1, 10, 21, 31, 11, 20, 30])
        XCTAssertEqual(proposal.parents, [-1, 0, 1, 2, 0, 4, 5])
        let tree = try Q35VerificationTree(parents: proposal.parents)
        let picks = [11, 63, 63, 63, 20, 30, 63]
        XCTAssertEqual(tree.matchingPath(tokens: proposal.tokens, targetPicks: picks, budget: 16, eos: [63]), [0, 4, 5, 6])
        XCTAssertEqual(tree.matchingPath(tokens: proposal.tokens, targetPicks: picks, budget: 2, eos: []), [0, 4])
        XCTAssertEqual(tree.matchingPath(tokens: proposal.tokens, targetPicks: picks, budget: 16, eos: [11]), [0])
    }

    func testPrefixCommitIsolatesForksAndOverwritesRejectedCapacity() throws {
        for prefix in [0, 3, 4] {
            for accepted in [1, 2, 8] {
                let live = KVCacheSimple(step: 4)
                let prefixKeys = MLXArray((0..<(prefix * 6)).map(Float.init)).reshaped(1, 2, prefix, 3)
                if prefix > 0 { _ = live.update(keys: prefixKeys, values: -prefixKeys) }
                let snapshot = try XCTUnwrap(live.fork() as? KVCacheSimple)
                let candidate = try XCTUnwrap(live.fork() as? KVCacheSimple)
                let proposed = (MLXArray(0..<48).asType(.float32) + 100).reshaped(1, 2, 8, 3)
                _ = candidate.update(keys: proposed, values: -proposed)
                Q35LayerCache.full(live).commitVerificationTree(candidate: .full(candidate),
                                                               path: Array(0..<accepted))
                let expected = concatenated([prefixKeys, proposed[0..., 0..., ..<accepted, 0...]], axis: 2)
                XCTAssertEqual(live.offset, prefix + accepted)
                close(live.verificationRows(Array(0..<live.offset)).0, expected, tolerance: 0)
                let committedSnapshot = try XCTUnwrap(live.fork() as? KVCacheSimple)
                // Force growth from an aligned write head that still has rejected
                // allocation rows. Those rows must not enter the published prefix.
                let next = MLXArray.full([1, 2, 17, 3], values: MLXArray(Float(900)))
                let advanced = live.update(keys: next, values: -next)
                close(advanced.0, concatenated([expected, next], axis: 2), tolerance: 0)
                close(advanced.1, -concatenated([expected, next], axis: 2), tolerance: 0)
                close(committedSnapshot.verificationRows(Array(0..<committedSnapshot.offset)).0,
                      expected, tolerance: 0)
                // Mutate the candidate wrapper, then read both live snapshots.
                candidate.rollback(toOffset: prefix)
                let poison = MLXArray.full([1, 2, 8, 3], values: MLXArray(Float(-700)))
                _ = candidate.update(keys: poison, values: poison)
                close(live.verificationRows(Array(0..<live.offset)).0,
                      concatenated([expected, next], axis: 2), tolerance: 0)
                close(committedSnapshot.verificationRows(Array(0..<committedSnapshot.offset)).0,
                      expected, tolerance: 0)
                if prefix > 0 {
                    close(snapshot.verificationRows(Array(0..<prefix)).0, prefixKeys, tolerance: 0)
                }
            }
        }
    }

    func testEveryBranchAndCommitMatchesIndependentSerialState() throws {
        MLXRandom.seed(716)
        let config = try JSONDecoder().decode(Q35Config.self, from: Data(Q35DFlash2Tests.smallTarget.utf8))
        let model = Q35Model(config: config)
        let tree = try Q35VerificationTree(parents: [-1, 0, 0, 1, 2, 3, 2, 5])
        let tokens = [1, 4, 8, 9, 12, 15, 18, 21]
        let base = caches()
        _ = model.forward(MLXArray([Int32(2), 3, 5, 7]).reshaped(1, 4), cache: base)
        let candidate = base.map { $0?.fork() }
        let output = model.forwardTree(tokens, tree: tree, cache: candidate, captureLayerIndices: [0, 1])
        MLX.eval(output.logits)
        for row in tokens.indices {
            let path = tree.paths[row]
            let reference = base.map { $0?.fork() }
            var step: Q35ForwardOutput?
            for index in path {
                step = model.forward(MLXArray([Int32(tokens[index])]).reshaped(1, 1), cache: reference,
                                     captureLayerIndices: [0, 1])
            }
            close(output.logits[0..., row..<(row + 1), 0...], step!.logits)
            for layer in [0, 1] {
                close(output.capturedHiddenStates[layer]![0..., row..<(row + 1), 0...],
                      step!.capturedHiddenStates[layer]!)
            }
            let committed = base.map { $0?.fork() }
            let disposable = base.map { $0?.fork() }
            _ = model.forwardTree(tokens, tree: tree, cache: disposable)
            for (live, verified) in zip(committed, disposable) {
                live!.commitVerificationTree(candidate: verified!, path: path)
            }
            compareCaches(committed, reference)
            let next = MLXArray([Int32(22)]).reshaped(1, 1)
            close(model.forward(next, cache: committed).logits, model.forward(next, cache: reference).logits)
            compareCaches(committed, reference)
            if case .full(let cache) = base[1]! { XCTAssertEqual(cache.offset, 4) }
            // Poison every rejected sibling. Accepted logits and committed state
            // must remain independent of their values and flattened row indices.
            let poisoned = tokens.indices.map { path.contains($0) ? tokens[$0] : (tokens[$0] + 29) % 63 }
            let poisonCache = base.map { $0?.fork() }
            let poisonOutput = model.forwardTree(poisoned, tree: tree, cache: poisonCache)
            close(Q35VerificationTree.take(poisonOutput.logits, rows: path),
                  Q35VerificationTree.take(output.logits, rows: path), tolerance: 0)
            XCTAssertNil(Q35TreeVerificationContext.current)
        }
    }

    func testTreeDecodeMatchesSerialAndCommitsCallerOwnedCaches() throws {
        MLXRandom.seed(718)
        let config = try JSONDecoder().decode(Q35Config.self, from: Data(Q35DFlash2Tests.smallTarget.utf8))
        let draftConfig = try JSONDecoder().decode(Q35DFlash2Config.self, from: Data(Q35DFlash2Tests.smallDraft.utf8))
        let model = Q35Model(config: config), draft = Q35DFlash2Model(config: draftConfig)
        let prompt = MLXArray([Int32(2), 3, 5]).reshaped(1, 3)
        for (maximumNodes, budget, adaptive) in [
            (3, 24, false), (7, 24, false), (15, 1, false), (15, 2, false),
            (15, 8, false), (15, 24, false), (15, 48, true)
        ] {
            let base = caches(), reference = caches(), context = draft.makeCache()
            let initial = model.forward(prompt, cache: base, captureLayerIndices: [0, 1])
            draft.appendTargetContext(initial.capturedHiddenStates, cache: context)
            var logits = model.forward(prompt, cache: reference).logits
            var expected: [Int] = []
            for _ in 0..<budget {
                let token = argMax(logits[0, -1, 0...]).item(Int.self)
                expected.append(token)
                logits = model.forward(MLXArray([Int32(token)]).reshaped(1, 1), cache: reference).logits
            }
            var actual: [Int] = []
            let stats = try Q35DFlash2TreePolicy.$maximumNodes.withValue(maximumNodes) {
                try Q35DFlash2TreePolicy.$adaptiveWidthEnabledOverride.withValue(adaptive) {
                    try Q35DFlash2Decoder.decodeTree(model: model, draft: draft, draftCache: context,
                        initialLogits: initial.logits, layerCaches: base, eosSet: [], tokenBudget: budget,
                        emit: { actual.append($0) })
                }
            }
            XCTAssertEqual(actual, expected)
            XCTAssertEqual(stats.treeRounds, stats.rounds)
            XCTAssertTrue(stats.treeRoundsByNodeCount.dropFirst(maximumNodes + 1).allSatisfy { $0 == 0 })
            if adaptive {
                XCTAssertGreaterThan(stats.treeRoundsByNodeCount[7], 0)
                XCTAssertGreaterThan(stats.treeRoundsByNodeCount[15], 0)
                XCTAssertNotNil(stats.treeRoundSecondsByNodeCount[7])
                XCTAssertNotNil(stats.treeRoundSecondsByNodeCount[15])
            }
            if budget > 1 {
                XCTAssertGreaterThan(stats.treeRounds, 0)
                if case .full(let cache) = base[1]! { XCTAssertGreaterThan(cache.offset, 3) }
            }
        }
        XCTAssertFalse(Q35DFlash2TreePolicy.enabled)
    }

    private func caches() -> [Q35LayerCache?] { [.linear(Q35LinearCache()), .full(KVCacheSimple(step: 4))] }

    func testEOSAndCancellationPreserveThePublishedPath() throws {
        enum Stop: Error { case cancelled }
        MLXRandom.seed(723)
        let config = try JSONDecoder().decode(Q35Config.self, from: Data(Q35DFlash2Tests.smallTarget.utf8))
        let draftConfig = try JSONDecoder().decode(Q35DFlash2Config.self, from: Data(Q35DFlash2Tests.smallDraft.utf8))
        let model = Q35Model(config: config), draft = Q35DFlash2Model(config: draftConfig)
        let base = caches(), reference = caches(), context = draft.makeCache()
        let prompt = MLXArray([Int32(2), 3, 5]).reshaped(1, 3)
        let initial = model.forward(prompt, cache: base, captureLayerIndices: [0, 1])
        _ = model.forward(prompt, cache: reference)
        draft.appendTargetContext(initial.capturedHiddenStates, cache: context)
        let eos = argMax(initial.logits[0, -1, 0...]).item(Int.self)
        var published: [Int] = []
        let empty = try Q35DFlash2Decoder.decodeTree(model: model, draft: draft, draftCache: context,
            initialLogits: initial.logits, layerCaches: base, eosSet: [eos], tokenBudget: 24, emit: { published.append($0) })
        XCTAssertTrue(published.isEmpty)
        XCTAssertEqual(empty.treeRounds, 0)
        compareCaches(base, reference)
        var checks = 0
        XCTAssertThrowsError(try Q35DFlash2Decoder.decodeTree(model: model, draft: draft, draftCache: context,
            initialLogits: initial.logits, layerCaches: base, eosSet: [], tokenBudget: 24,
            checkCancellation: { checks += 1; if checks == 2 { throw Stop.cancelled } }, emit: { published.append($0) }))
        XCTAssertFalse(published.isEmpty)
        for token in published { _ = model.forward(MLXArray([Int32(token)]).reshaped(1, 1), cache: reference) }
        compareCaches(base, reference)
        XCTAssertEqual(context[0].offset, 3 + published.count)
    }

    private func close(_ lhs: MLXArray, _ rhs: MLXArray, tolerance: Float = 1e-4) {
        let delta = (lhs.asType(.float32) - rhs.asType(.float32)).abs().max().item(Float.self)
        XCTAssertLessThanOrEqual(delta, tolerance)
    }
    private func compareCaches(_ lhs: [Q35LayerCache?], _ rhs: [Q35LayerCache?]) {
        if case .linear(let a) = lhs[0]!, case .linear(let b) = rhs[0]! {
            close(a.convState!, b.convState!)
            close(a.recurrentState!, b.recurrentState!)
        }
        if case .full(let a as KVCacheSimple) = lhs[1]!, case .full(let b as KVCacheSimple) = rhs[1]! {
            XCTAssertEqual(a.offset, b.offset)
            let indices = Array(0..<a.offset)
            let aState = a.verificationRows(indices)
            let bState = b.verificationRows(indices)
            close(aState.0, bState.0)
            close(aState.1, bState.1)
        }
    }
}
