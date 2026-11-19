import MLX
import MLXNN
import MLXRandom
import MereRunMLXTestSupport
import XCTest

final class Q35TreeRoPETests: MLXTestCase {
    func testPerNodeOffsetsMatchStaticSerialPositionsExactly() {
        MLXRandom.seed(724)
        let depths = [Array(0..<16), [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4]]
        for dtype in [DType.float32, .bfloat16] {
            for (width, rotary) in [(32, 32), (256, 64), (256, 256)] {
                let layer = RoPE(dimensions: rotary, traditional: false, base: 10_000_000)
                let input = MLXRandom.normal([1, 24, 16, width]).asType(dtype)
                for depth in depths {
                    for prefix in [0, 1016, 4610, 8190, 65_530] {
                        let expected = concatenated(depth.enumerated().map { row, position in
                            layer(input[0..., 0..., row..<(row + 1), 0...], offset: prefix + position)
                        }, axis: 2)
                        let offsets = MLXArray(depth.map { Int32(prefix + $0) })
                        let actual = layer(input.transposed(2, 1, 0, 3), offset: offsets).transposed(2, 1, 0, 3)
                        XCTAssertEqual((actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self), 0,
                                       "dtype=\(dtype), width=\(width), rotary=\(rotary), prefix=\(prefix)")
                    }
                }
            }
        }
    }
}
