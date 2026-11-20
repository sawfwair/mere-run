#if os(macOS)
import Metal
import MLX
import MLXNN
import MereRunTensor
import XCTest
@testable import MereRunQwenModel

final class Q35NativeSampledHeadTests: XCTestCase {
    func testDevicePicksPathAndOverflow() throws {
        guard #available(macOS 26.0, *), let device = MTLCreateSystemDefaultDevice(),
              device.supportsFamily(.apple10) else { throw XCTSkip("Requires an M5 Metal device") }
        let sampler = Q35PositionSampler(seed: 73, temperature: 1, topP: 0.95, topK: 20)
        let program = try Q35NativeSampledHead(device: device)
        let tree = try Q35VerificationTree(parents: [-1, 0, 0, 1])
        let rows = tree.count, vocabulary = 248_320
        let logits = try XCTUnwrap(device.makeBuffer(length: rows * vocabulary * 2, options: .storageModeShared))
        let metadata = try XCTUnwrap(device.makeBuffer(length: 34 * 4, options: .storageModeShared))
        let words = logits.contents().assumingMemoryBound(to: UInt16.self)
        for row in 0..<rows {
            for id in 0..<vocabulary {
                let score: Float = id.isMultiple(of: 2048) ? Float((id / 2048 + row * 5) % 61) / 4 : -20
                words[row * vocabulary + id] = UInt16(score.bitPattern >> 16)
            }
        }
        let expected = (0..<rows).map { row in
            sampler.choose(.init(start: words + row * vocabulary, count: vocabulary), position: 91 + tree.depths[row])
        }
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let request = Q35GreedyTreeRequest(head: Linear(1, 1), tokens: [2, expected[0], 1, expected[1]], budget: rows, eos: [])
        func run(_ plan: Q35VerificationTree, _ input: Q35GreedyTreeRequest, _ draws: Q35RejectionDraws? = nil,
                 rule: Q35PositionSampler? = nil, position: Int = 91) throws -> [Int] {
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder(dispatchType: .concurrent))
            program.encode(into: encoder, logits: logits, sampler: rule ?? sampler, position: position, tree: plan, request: input,
                           metadata: metadata, hazards: M5NativeHazards(), rejection: draws)
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            XCTAssertEqual(command.status, .completed)
            return UnsafeBufferPointer(start: metadata.contents().assumingMemoryBound(to: Int32.self), count: 34).map(Int.init)
        }
        let selected = try run(tree, request)
        XCTAssertEqual(Array(selected[18..<(18 + rows)]), expected)
        XCTAssertEqual(Array(selected[2..<(2 + selected[0])]), [0, 1, 3])
        XCTAssertEqual(selected[1], expected[3])
        // At a thinking boundary the root may be the final permitted row even when its child matches.
        for budget in 1...2 {
            let capped = Q35GreedyTreeRequest(head: request.head, tokens: request.tokens, budget: budget, eos: [])
            let result = try run(tree, capped)
            XCTAssertEqual(Array(result[2..<(2 + result[0])]), Array([0, 1].prefix(budget)))
            XCTAssertEqual(result[1], expected[budget - 1])
        }
        var serial = sampler
        for order: Q35SerialTreeKeys.Order in [.sorted, .vocabulary] {
            serial.serialKeys = Q35SerialTreeKeys(seed: sampler.seed, position: 91, order: order)
            serial.serialKeys?.prepare(position: 91, count: rows)
            let serialPicks = (0..<rows).map { row in
                serial.choose(.init(start: words + row * vocabulary, count: vocabulary), position: 91 + tree.depths[row])
            }
            let serialMetadata = try run(tree, request, rule: serial)
            XCTAssertEqual(Array(serialMetadata[18..<(18 + rows)]), serialPicks)
        }
        let chain = try Q35VerificationTree(parents: [-1, 0, 1, 2])
        var candidates = (0..<(3 * 16)).map { Int32($0 + 1) }
        candidates[0] = Int32(expected[0])
        let proposal = Q35SampledProposal(candidates: candidates, scores: Array(repeating: 0, count: 3 * 256),
            depth: 3, noise: (0..<48).map { $0.isMultiple(of: 16) ? 1 : 0 })
        let input = Q35GreedyTreeRequest(head: request.head, tokens: [2] + proposal.tokens, budget: rows, eos: [])
        let mass = sampler.support(.init(start: words + vocabulary, count: vocabulary))
        let ids = MLXArray(mass.map { Int32($0.id) })
        for seed in [42, 73, 109, 1009, 2429] {
            let keys = (0..<rows).map { MLXRandom.key(UInt64(seed + $0)) }
            let draws = Q35RejectionDraws(proposal: proposal, uniforms: [0, 0.9, 0.9],
                                          keys: stacked(keys).asArray(UInt32.self))
            let pick = argMax(MLX.log(MLXArray(mass.map(\.probability))) + MLXRandom.gumbel([vocabulary], key: keys[1])[ids])
                .item(Int.self)
            let result = try run(chain, input, draws)
            XCTAssertEqual(result[0], 2)
            XCTAssertEqual(Array(result[2..<4]), [0, 1])
            XCTAssertEqual(result[1], mass[pick].id)
            let stop = Q35GreedyTreeRequest(head: input.head, tokens: input.tokens,
                budget: rows, eos: [proposal.tokens[0]])
            let stopped = try run(chain, stop, draws)
            XCTAssertEqual(Array(stopped.prefix(3)), [1, proposal.tokens[0], 0])
        }
        // Top 20 BF16 logits at the first changed LRU draw. IDs 1238 and 2394 tie at the nucleus boundary.
        let lru = Q35PositionSampler(seed: 109, temperature: 1, topP: 0.95, topK: 20)
        let ranked: [(Int, Float)] = [
            (913, 22), (413, 21.5), (696, 21.5), (4346, 20.625), (2357, 20.5), (16782, 20.375),
            (440, 20.25), (638, 20.125), (948, 20.125), (9495, 20.125), (3921, 19.5), (303, 19.125),
            (364, 19.125), (1238, 19), (2394, 19), (615, 18.875), (1521, 18.625), (4016, 18.375),
            (1144, 18), (18904, 18)
        ]
        for id in 0..<vocabulary { words[id] = UInt16(Float(-20).bitPattern >> 16) }
        for (id, value) in ranked { words[id] = UInt16(value.bitPattern >> 16) }
        // A lower score in each tile keeps the compactor on-device for this sparse fixture.
        for tile in 0..<122 { words[tile * 2048] = UInt16(Float(15).bitPattern >> 16) }
        let absolutePosition = 94 + 942
        XCTAssertEqual(lru.choose(.init(start: words, count: vocabulary), position: absolutePosition), 1238)
        XCTAssertEqual(try run(tree, request, rule: lru, position: absolutePosition)[18], 1238)
        for id in 0..<vocabulary { words[id] = 0 }
        XCTAssertEqual(try run(tree, request)[0], 0, "Full-vocabulary cutoff ties require reference selection before commit")
    }
}
#endif
