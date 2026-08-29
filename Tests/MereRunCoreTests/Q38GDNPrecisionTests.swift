import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class Q38GDNPrecisionTests: MereRunCoreTestCase {
    func testFlashNextPreworkRetainsReferenceFloat32Normalization() {
        let heads = 2
        let dimension = 32
        let channels = heads * dimension * 3
        let qkv = MLXArray((0..<(3 * channels)).map { Float($0 % 43 - 21) / 31 }, [1, 3, channels])
            .asType(.bfloat16)
        let state = MLXArray.zeros([1, 3, channels], dtype: .bfloat16)
        let weight = MLXArray((0..<(channels * 4)).map { Float($0 % 7 - 3) / 13 }, [channels, 4, 1])
            .asType(.bfloat16)
        let actual = q35GDNPreworkOps(
            qkv: qkv, convState: state, convWeight: weight,
            numKeyHeads: heads, numValueHeads: heads,
            keyHeadDim: dimension, valueHeadDim: dimension,
            normalizeInFloat32: true
        )
        let convolved = MLXNN.silu(MLX.conv1d(
            MLX.concatenated([state, qkv], axis: 1), weight, groups: channels
        )).asType(.float32)
        let rawQ = convolved[.ellipsis, 0..<(heads * dimension)].reshaped(1, 3, heads, dimension)
        let rawK = convolved[.ellipsis, (heads * dimension)..<(heads * dimension * 2)]
            .reshaped(1, 3, heads, dimension)
        // The reference normalizes the FP32 convolution outputs using the
        // squared SUM plus epsilon, then scales Q. Neither intermediate is BF16.
        let expectedQ = rawQ * MLX.rsqrt(MLX.square(rawQ).sum(axis: -1, keepDims: true) + 1e-6)
            / Float(dimension).squareRoot()
        let expectedK = rawK * MLX.rsqrt(MLX.square(rawK).sum(axis: -1, keepDims: true) + 1e-6)
        XCTAssertEqual(actual.q.dtype, .float32)
        XCTAssertEqual(actual.k.dtype, .float32)
        XCTAssertEqual(actual.v.dtype, .float32)
        XCTAssertEqual(actual.convState.dtype, .bfloat16)
        XCTAssertLessThan((actual.q - expectedQ).abs().max().item(Float.self), 1e-7)
        XCTAssertLessThan((actual.k - expectedK).abs().max().item(Float.self), 1e-7)

        let legacy = q35GDNPreworkOps(
            qkv: qkv, convState: state, convWeight: weight,
            numKeyHeads: heads, numValueHeads: heads,
            keyHeadDim: dimension, valueHeadDim: dimension
        )
        XCTAssertEqual(legacy.q.dtype, .bfloat16, "Other Qwen architectures keep their existing path")
        XCTAssertGreaterThan((legacy.k.asType(.float32) - expectedK).abs().max().item(Float.self), 1e-4)
    }
}
