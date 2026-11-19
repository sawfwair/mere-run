import MereRunDecode
import MereRunMLXTestSupport
@testable import MereRunQwenModel
import MLX
import XCTest

final class Q35SerialTreeSamplingTests: MLXTestCase {
    func testWindowedTreeDrawsMatchSerialSamplerIncludingCutoffTies() {
        let size = 248_320
        var values = Array(repeating: Float(-20), count: size)
        for index in 0..<45 { values[index * 3877] = Float(index / 9) / 2 }
        let words = values.map { UInt16(truncatingIfNeeded: $0.bitPattern >> 16) }
        let logits = MLXArray(values).asType(.bfloat16)
        let config = GenerationConfig(temperature: 1, topK: 20, topP: 0.95)
        for seed: UInt64 in [42, 109, .max] {
            var rule = Q35PositionSampler(seed: seed, temperature: 1, topP: 0.95, topK: 20)
            rule.serialKeys = Q35SerialTreeKeys(seed: seed, position: 73)
            let stream = MLXRandom.RandomState(seed: seed)
            for position in 73..<91 {
                rule.serialKeys?.prepare(position: position, count: 8)
                let key = rule.serialKeys![position]
                let expectedKey = stream.next().asArray(UInt32.self)
                XCTAssertEqual([key.x, key.y], expectedKey)
            }
            for order: Q35SerialTreeKeys.Order in [.sorted, .vocabulary] {
                let expected = withRandomState(MLXRandom.RandomState(seed: seed)) {
                    (0..<18).map { _ -> Int in
                        if order == .sorted {
                            return sampledTokenArray(logits: logits, config: config,
                                previousTokenIndices: nil, banMask: nil).item(Int.self)
                        }
                        let probabilities = samplingDistribution(logits: logits, config: config,
                            previousTokenIndices: nil, banMask: nil)
                        return Q35DFlash2Sampling.sample(probabilities).item(Int.self)
                    }
                }
                rule.serialKeys = Q35SerialTreeKeys(seed: seed, position: 73, order: order)
                let actual = words.withUnsafeBufferPointer { buffer in
                    (73..<91).map { position in
                        rule.serialKeys?.prepare(position: position, count: 8)
                        return rule.choose(buffer, position: position)
                    }
                }
                XCTAssertEqual(actual, expected)
            }
        }
    }
}
