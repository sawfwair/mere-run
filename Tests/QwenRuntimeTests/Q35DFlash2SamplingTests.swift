@testable import MereRunTensor
import MLX
import MLXRandom
import XCTest
import MereRunMLXTestSupport
@testable import MereRunQwenModel

final class Q35DFlash2SamplingTests: MLXTestCase {
    /// Enumerate all proposal/rejection outcomes independently. Their combined
    /// output mass must equal p, including tokens absent from the draft shortlist.
    func testAcceptanceAndCorrectionPreserveTargetDistribution() {
        let cases: [([Float], [Int32], [Float])] = [
            ([0.1, 0.2, 0.3, 0.4], [0, 2], [0.8, 0.2]),
            ([0, 0, 0.3, 0.7], [0, 1], [0.25, 0.75]),
            ([0.6, 0.4, 0, 0], [1, 0], [0.4, 0.6]),
            ([0.999, 0.001, 0, 0], [1], [1]),
        ]
        for (target, ids, draft) in cases {
            let p = MLXArray(target)
            let acceptance = Q35DFlash2Sampling.acceptanceProbability(target: p[MLXArray(ids)], draft: MLXArray(draft))
                .asArray(Float.self)
            let residual = Q35DFlash2Sampling.correction(
                target: p, candidateIDs: MLXArray(ids), candidateProbabilities: MLXArray(draft)
            ).asArray(Float.self)
            let rejectionMass = zip(draft, acceptance).reduce(Float(0)) { $0 + $1.0 * (1 - $1.1) }
            var output = residual.map { rejectionMass * $0 }
            for index in ids.indices { output[Int(ids[index])] += draft[index] * acceptance[index] }
            for index in target.indices { XCTAssertEqual(output[index], target[index], accuracy: 1e-6) }
            XCTAssertEqual(residual.reduce(0, +), 1, accuracy: 1e-6)
            XCTAssertTrue(residual.allSatisfy(\.isFinite))
        }
    }

    func testSelectedDrawAndFollowingKeyMatchDrawingEveryRow() {
        let ids = MLXArray([Int32(1), 2])
        let masses = MLXArray([Float(0.75), 0.25])
        let selection = DFlash2Selection(
            tokens: MLXArray([Int32(1), 2]).reshaped(1, 2),
            candidateTokenIds: stacked([ids, ids]).reshaped(1, 2, 2),
            candidateProbabilities: stacked([masses, masses]).reshaped(1, 2, 2)
        )
        let sampling = Q35DFlash2Sampling(temperature: 1, history: nil) { logits, _ in logits }
        let logits = MLXArray([Float(0.4), 0.35, 0.25, 0.4, 0.5, 0.1, 0.2, 0.4, 0.4]).reshaped(1, 3, 3)
        var reached = Set<Int>()
        for seed in 0..<48 {
            let reference = withRandomState(MLXRandom.RandomState(seed: UInt64(seed))) {
                var flags: [Int32] = []
                var draws: [Int32] = []
                for index in 0..<2 {
                    let target = logits[0, index, 0...]
                    let p = Q35DFlash2Sampling.acceptanceProbability(target: target[index + 1], draft: masses[index])
                    flags.append((MLXRandom.uniform() .< p).asType(.int32).item(Int32.self))
                    draws.append(Q35DFlash2Sampling.sample(Q35DFlash2Sampling.correction(
                        target: target, candidateIDs: ids, candidateProbabilities: masses
                    )).item(Int32.self))
                }
                draws.append(Q35DFlash2Sampling.sample(logits[0, 2, 0...]).item(Int32.self))
                let row = flags.firstIndex(of: 0) ?? flags.count
                reached.insert(row)
                return (flags, draws[row], resolve().asArray(UInt32.self))
            }
            withRandomState(MLXRandom.RandomState(seed: UInt64(seed))) {
                let result = sampling.verify(logits: logits, selection: selection, history: nil)
                XCTAssertEqual(result.accepted.asArray(Int32.self), reference.0)
                XCTAssertEqual(result.nextToken.item(Int32.self), reference.1)
                XCTAssertEqual(resolve().asArray(UInt32.self), reference.2)
            }
        }
        XCTAssertEqual(reached, [0, 1, 2])
    }

    func testVerifierConditionsEachPositionOnItsOwnPrefixAndKeepsEOSMass() {
        let selection = DFlash2Selection(
            tokens: MLXArray([Int32(1), 2]).reshaped(1, 2),
            candidateTokenIds: MLXArray([Int32(1), 2]).reshaped(1, 2, 1),
            candidateProbabilities: MLXArray.ones([1, 2, 1])
        )
        var histories: [[Int32]] = []
        let sampling = Q35DFlash2Sampling(temperature: 1, history: [0]) { _, history in
            histories.append(history!.asArray(Int32.self))
            let probabilities = MLXArray.zeros([4])
            // First proposal is accepted; second is rejected to token 3,
            // which is outside its shortlist and may represent EOS.
            probabilities[histories.count == 1 ? 1 : 3] = MLXArray(Float(1))
            return probabilities
        }
        let result = sampling.verify(
            logits: MLXArray.zeros([1, 3, 4]), selection: selection, history: MLXArray([Int32(0)])
        )
        XCTAssertEqual(result.accepted.asArray(Int32.self), [1, 0])
        XCTAssertEqual(result.nextToken.item(Int32.self), 3)
        XCTAssertEqual(histories, [[0], [0, 1], [0, 1, 2]])
    }
}
