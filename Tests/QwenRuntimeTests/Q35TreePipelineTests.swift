import Foundation
import MereRunKVCache
import MereRunMLXTestSupport
import MLX
import MLXRandom
import XCTest
@testable import MereRunQwenModel
@testable import MereRunTensor

final class Q35TreePipelineTests: MLXTestCase {
    func testCombinedPricedCopiedPipelinedDecodeMatchesSerialAcrossBudgets() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Pipeline decoding requires Metal.")
        MLXRandom.seed(789)
        let json = Q35DFlash2Tests.smallTarget
            .replacingOccurrences(of: "\"linear_key_head_dim\":8", with: "\"linear_key_head_dim\":32")
            .replacingOccurrences(of: "\"linear_value_head_dim\":8", with: "\"linear_value_head_dim\":32")
        let model = Q35Model(config: try JSONDecoder().decode(Q35Config.self, from: Data(json.utf8)))
        let draft = Q35DFlash2Model(config: try JSONDecoder().decode(Q35DFlash2Config.self,
            from: Data(Q35DFlash2Tests.smallDraft.utf8)))
        for budget in [1, 2, 16, 48] {
            let live: [Q35LayerCache?] = [.linear(Q35LinearCache()), .full(KVCacheSimple(step: 4))]
            let reference = live.map { $0?.fork() }, context = draft.makeCache()
            let prompt = MLXArray([Int32(2), 3, 5, 7]).reshaped(1, 4)
            let initial = model.forward(prompt, cache: live, captureLayerIndices: [0, 1])
            draft.appendTargetContext(initial.capturedHiddenStates, cache: context)
            var logits = model.forward(prompt, cache: reference).logits, expected: [Int] = []
            for _ in 0..<budget {
                let token = argMax(logits[0, -1, 0...]).item(Int.self)
                expected.append(token)
                logits = model.forward(MLXArray([Int32(token)]).reshaped(1, 1), cache: reference).logits
            }
            var actual: [Int] = []
            try Q35TreePipeline.$enabled.withValue(true) {
                try Q35DFlash2TreePolicy.$pricedNodesEnabled.withValue(true) {
                    try Q35DFlash2TreePolicy.$contextCopiesEnabled.withValue(true) {
                        try Q35DFlash2TreePolicy.$calibratedScoresEnabledOverride.withValue(true) {
                            _ = try Q35DFlash2Decoder.decodeTree(model: model, draft: draft, draftCache: context,
                                initialLogits: initial.logits, layerCaches: live, eosSet: [], tokenBudget: budget,
                                initialHistory: [2, 3, 5, 7], emit: { actual.append($0) })
                        }
                    }
                }
            }
            XCTAssertEqual(actual, expected, "budget=\(budget)")
        }
    }

    func testDeviceMatcherHonorsNonPrefixPathsEOSAndBudget() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Device matching requires Metal.")
        let proposal = DFlash2TreeProposal(tokens: [1, 4, 8, 9, 12, 15, 18, 21],
            parents: [-1, 0, 0, 1, 2, 3, 2, 5], logProbabilities: Array(repeating: 0, count: 8))
        let tree = try Q35VerificationTree(parents: proposal.parents)
        for terminal in proposal.tokens.indices {
            var picks = Array(repeating: 63, count: tree.count)
            let targetPath = tree.paths[terminal]
            for (parent, child) in zip(targetPath, targetPath.dropFirst()) { picks[parent] = proposal.tokens[child] }
            for budget in [1, 2, 16] {
                for eos in [Set<Int>(), Set([63]), Set([proposal.tokens[terminal]])] {
                    let expected = tree.matchingPath(tokens: proposal.tokens, targetPicks: picks, budget: budget, eos: eos)
                    let actual = Q35TreePipeline.match(picks: MLXArray(picks.map(Int32.init)),
                        proposal: proposal, budget: budget, eos: eos, prefix: 5).host.asArray(Int32.self)
                    XCTAssertEqual(Int(actual[0]), expected.count)
                    XCTAssertEqual(actual.dropFirst(2).prefix(expected.count).map(Int.init), expected)
                    XCTAssertEqual(Int(actual[1]), picks[expected.last!])
                }
            }
        }
    }

    func testPreparedGPUStateAndContinuationMatchLegacyCommitForEveryPath() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Prepared state requires Metal.")
        MLXRandom.seed(785)
        let json = Q35DFlash2Tests.smallTarget
            .replacingOccurrences(of: "\"linear_key_head_dim\":8", with: "\"linear_key_head_dim\":32")
            .replacingOccurrences(of: "\"linear_value_head_dim\":8", with: "\"linear_value_head_dim\":32")
        let model = Q35Model(config: try JSONDecoder().decode(Q35Config.self, from: Data(json.utf8)))
        let base: [Q35LayerCache?] = [.linear(Q35LinearCache()), .full(KVCacheSimple(step: 4))]
        _ = model.forward(MLXArray([Int32(2), 3, 5, 7]).reshaped(1, 4), cache: base)
        let candidate = base.map { $0?.fork() }
        let tokens = [1, 4, 8, 9, 12, 15, 18, 21]
        let tree = try Q35TreePipeline.$enabled.withValue(true) {
            try Q35VerificationTree(parents: [-1, 0, 0, 1, 2, 3, 2, 5])
        }
        _ = Q35TreePipeline.$enabled.withValue(true) { model.forwardTree(tokens, tree: tree, cache: candidate) }
        let frame = Q35TreePipeline.Frame()
        defer { frame.finish() }
        for terminal in tokens.indices {
            let path = tree.paths[terminal]
            var picks = Array(repeating: 63, count: tokens.count)
            for (parent, child) in zip(path, path.dropFirst()) { picks[parent] = tokens[child] }
            let proposal = DFlash2TreeProposal(tokens: tokens, parents: tree.parents,
                logProbabilities: Array(repeating: 0, count: tokens.count))
            let match = Q35TreePipeline.match(picks: MLXArray(picks.map(Int32.init)),
                proposal: proposal, budget: 16, eos: [63], prefix: 4)
            let prepared = frame.prepare(caches: candidate, tree: tree, match: match, prefix: 4)
            MLX.asyncEval(prepared.flatMap(\.state))
            let count = Int(match.host.asArray(Int32.self)[0])
            XCTAssertEqual(count, path.count)
            let live = base.map { $0?.fork() }, expected = base.map { $0?.fork() }
            for ((cache, reference), (state, source)) in zip(zip(live, expected), zip(prepared, candidate)) {
                state.publish(to: cache!, path: path)
                let copy: Q35LayerCache
                if case .linear(let original) = source! {
                    let c = original.fork()
                    c.verificationReplay = original.verificationReplay; c.verificationTree = original.verificationTree
                    c.verificationGates = original.verificationGates; copy = .linear(c)
                } else { copy = source!.fork() }
                reference!.commitVerificationTree(candidate: copy, path: path)
            }
            compare(live, expected)
            let next = MLXArray([Int32(22)]).reshaped(1, 1)
            exact(model.forward(next, cache: live).logits, model.forward(next, cache: expected).logits)
            compare(live, expected)
            if case .full(let kv) = base[1]! { XCTAssertEqual(kv.offset, 4) }
        }
    }

    private func exact(_ a: MLXArray, _ b: MLXArray) {
        XCTAssertEqual((a.asType(.float32) - b.asType(.float32)).abs().max().item(Float.self), 0)
    }
    private func compare(_ a: [Q35LayerCache?], _ b: [Q35LayerCache?]) {
        if case .linear(let x) = a[0]!, case .linear(let y) = b[0]! {
            exact(x.convState!, y.convState!); exact(x.recurrentState!, y.recurrentState!)
        }
        if case .full(let x as KVCacheSimple) = a[1]!, case .full(let y as KVCacheSimple) = b[1]! {
            XCTAssertEqual(x.offset, y.offset)
            let lhs = x.verificationRows(Array(0..<x.offset)), rhs = y.verificationRows(Array(0..<y.offset))
            exact(lhs.0, rhs.0); exact(lhs.1, rhs.1)
        }
    }
}
