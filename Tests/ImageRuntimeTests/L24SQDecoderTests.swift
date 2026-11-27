import MLX
import MLXNN
import MereRunMLXTestSupport
import XCTest
@testable import MereRunImageModels

final class L24SQDecoderTests: MLXTestCase {
    func testRectangularGridMatchesReferencePixels() throws {
        let decoder = L24SQDecoder(width: 8, heads: 1, layers: 1, codeCount: 5,
                                   embeddingWidth: 3, outputWidth: 5, patchSize: 2)
        let weights = decoder.parameters().flattened().map { key, parameter in
            let offset = key.utf8.reduce(0) { $0 + Int($1) }
            let values = (0..<parameter.size).map { Float(($0 * 13 + offset) % 37 - 18) / 32 }
            return (key, MLXArray(values).reshaped(parameter.shape))
        }
        try decoder.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let pixels = decoder.decode(MLXArray([1, 4, 3, 0, 2, 1]), rows: 2, columns: 3)
        // Independent FP32 forward pass with planar RGB patches, axial RoPE and full attention.
        let expected: [UInt8] = [
            114, 174, 234, 102, 163, 76, 142, 205, 255, 95, 158, 73, 115, 175, 236, 99, 160, 73,
            193, 61, 122, 85, 146, 59, 201, 37, 100, 52, 115, 30, 196, 58, 119, 86, 146, 59,
            113, 174, 234, 104, 164, 77, 114, 175, 235, 101, 161, 74, 114, 174, 234, 102, 163, 76,
            191, 63, 123, 84, 145, 58, 195, 60, 120, 85, 146, 59, 193, 61, 122, 85, 146, 58
        ]
        XCTAssertEqual(pixels.shape, [4, 6, 3])
        XCTAssertEqual(pixels.asArray(UInt8.self), expected)
    }
}
