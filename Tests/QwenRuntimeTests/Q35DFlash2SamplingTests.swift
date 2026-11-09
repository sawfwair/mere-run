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
        XCTAssertEqual(result.replacements.asArray(Int32.self)[1...], [3, 3])
        XCTAssertEqual(histories, [[0], [0, 1], [0, 1, 2]])
    }
}
