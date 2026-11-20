import XCTest
@testable import MereRunQwenModel

final class Q35PositionSamplerTests: XCTestCase {
    func testKthScoreTiesAndNucleusSupportUseVocabularyOrder() {
        let logits: [Float] = [2, 2, 2, 2, 2, 3, 1, -0.0, 0.0]
        let words = logits.map { UInt16($0.bitPattern >> 16) }
        for (topP, expected) in [(0.7, Set([5])), (0.95, Set([0, 5]))] {
            var observed = Set<Int>()
            for seed in 0..<128 {
                let rule = Q35PositionSampler(seed: UInt64(seed), temperature: 1, topP: topP, topK: 2)
                observed.insert(words.withUnsafeBufferPointer { rule.choose($0, position: 9) })
            }
            XCTAssertEqual(observed, expected)
        }
    }

    func testNucleusCrossingWithEqualMassUsesLowerIDs() {
        let words = [UInt16](repeating: 0, count: 8)
        var chosen = Set<Int>()
        for seed in 0..<64 {
            let rule = Q35PositionSampler(seed: UInt64(seed), temperature: 1, topP: 0.5, topK: 4)
            chosen.insert(words.withUnsafeBufferPointer { rule.choose($0, position: 3) })
        }
        XCTAssertEqual(chosen, [0, 1])
    }

    func testSparseTargetMassRetainsEverySurvivingCutoffTie() {
        let words = [Float(2), 2, 2, 2, 2, 3, 1, 0, 0].map { UInt16($0.bitPattern >> 16) }
        let rule = Q35PositionSampler(seed: 42, temperature: 1, topP: 0.7, topK: 2)
        let p = words.withUnsafeBufferPointer { rule.support($0) }
        XCTAssertEqual(p.map(\.id), [2, 3, 4, 5])
        XCTAssertEqual(p.reduce(0) { $0 + $1.probability }, 1, accuracy: 1e-6)
        XCTAssertEqual(p.last!.probability / p.first!.probability, exp(Float(1)), accuracy: 1e-5)
    }

    func testPositionsAndSeedsAreIndependentlyKeyed() {
        let first = Q35PositionSampler(seed: 42, temperature: 1, topP: 0.95, topK: 20)
        let second = Q35PositionSampler(seed: 73, temperature: 1, topP: 0.95, topK: 20)
        XCTAssertEqual(first.noise(position: 1, id: 1024), first.noise(position: 1, id: 1024))
        XCTAssertNotEqual(first.noise(position: 1, id: 1024), first.noise(position: 2, id: 1024))
        XCTAssertNotEqual(first.noise(position: 1, id: 1024), second.noise(position: 1, id: 1024))
    }
}
