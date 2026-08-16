import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

/// Weight-only ConvRot oracle matching PocketAiHub's direct MLX evaluation.
///
/// Run the production-width timing explicitly; set
/// `MERERUN_MINIMAX_MUSIC3_CONVROT_ROWS` to change the default 100-row input:
///
///     MERERUN_MINIMAX_MUSIC3_CONVROT_WEIGHT_ONLY_E2E=1 \
///       swift test --filter MiniMaxMusic3ConvRotWeightOnlyResearchTests/testProductionWidth
final class MiniMaxMusic3ConvRotWeightOnlyResearchTests: MereRunCoreTestCase {
    func testConvRotWeightOnlyProjectionMatchesDenseReference() throws {
        let inputDimensions = 256
        let outputDimensions = 8
        let codes = MLXArray((0..<(outputDimensions * inputDimensions)).map {
            Int8($0 % 31 - 15)
        }).reshaped(outputDimensions, inputDimensions)
        let rowScales = MLXArray((0..<outputDimensions).map {
            Float($0 + 1) / 1_000
        })
        let input = MLXArray((0..<(3 * inputDimensions)).map {
            Float($0 % 29 - 14) / 17
        }).reshaped(3, inputDimensions).asType(.bfloat16)
        let layer = try XCTUnwrap(MiniMaxMusic3ConvRotWeightOnlyLinear(
            codes: codes,
            rowScales: rowScales,
            bias: nil
        ))

        let actual = layer(input).asType(.float32)
        let denseWeight = layer.denseReferenceWeight(dtype: .bfloat16)
        let expected = MLX.matmul(input, denseWeight.transposed()).asType(.float32)
        MLX.eval(actual, expected)

        XCTAssertTrue(MLX.allClose(actual, expected, rtol: 2e-2, atol: 2e-2).item(Bool.self))
    }

    func testProductionWidthConvRotWeightOnlyTiming() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment[
            "MERERUN_MINIMAX_MUSIC3_CONVROT_WEIGHT_ONLY_E2E"
        ] == "1" else {
            throw XCTSkip("set the ConvRot weight-only E2E flag")
        }
        let rows = max(1, Int(environment["MERERUN_MINIMAX_MUSIC3_CONVROT_ROWS"] ?? "") ?? 100)
        let inputDimensions = 2_048
        let outputDimensions = 2_048
        let codes = (MLX.sin(
            MLX.arange(outputDimensions * inputDimensions, dtype: .float32) / 17
        ) * 31).asType(.int8).reshaped(outputDimensions, inputDimensions)
        let scales = (MLX.sin(
            MLX.arange(outputDimensions, dtype: .float32) / 11
        ) * 0.0008 + 0.001).asType(.float32)
        let input = MLX.sin(
            MLX.arange(rows * inputDimensions, dtype: .float32) / 23
        ).reshaped(rows, inputDimensions).asType(.bfloat16)
        Self.trace("fixtures declared")
        let layer = try XCTUnwrap(MiniMaxMusic3ConvRotWeightOnlyLinear(
            codes: codes,
            rowScales: scales,
            bias: nil
        ))
        Self.trace("ConvRot layer initialized")
        let denseWeight = layer.denseReferenceWeight(dtype: .bfloat16)
        Self.trace("dense oracle declared")
        MLX.eval(layer(input), MLX.matmul(input, denseWeight.transposed()))
        Self.trace("warmup complete")

        let convRotMilliseconds = Self.bestMilliseconds { layer(input) }
        let denseMilliseconds = Self.bestMilliseconds {
            MLX.matmul(input, denseWeight.transposed())
        }
        let actual = layer(input).asType(.float32)
        let expected = MLX.matmul(input, denseWeight.transposed()).asType(.float32)
        MLX.eval(actual, expected)
        let cosine = MLX.sum(actual * expected)
            / MLX.sqrt(MLX.sum(MLX.square(actual)) * MLX.sum(MLX.square(expected)))

        print(
            "MiniMax Music 3 ConvRot weight-only: "
                + String(format: "rows=%d convrot=%.3fms dense=%.3fms ratio=%.3fx cosine=%.7f",
                    rows,
                    convRotMilliseconds,
                    denseMilliseconds,
                    convRotMilliseconds / denseMilliseconds,
                    cosine.item(Float.self))
        )
    }

    private static func trace(_ message: String) {
        FileHandle.standardError.write(Data("[convrot-benchmark] \(message)\n".utf8))
    }

    private static func bestMilliseconds(
        rounds: Int = 5,
        _ operation: () -> MLXArray
    ) -> Double {
        MLX.eval(operation())
        var best = Double.greatestFiniteMagnitude
        for _ in 0..<rounds {
            let start = DispatchTime.now().uptimeNanoseconds
            MLX.eval(operation())
            let elapsed = DispatchTime.now().uptimeNanoseconds - start
            best = min(best, Double(elapsed) / 1_000_000)
        }
        return best
    }
}

private final class MiniMaxMusic3ConvRotWeightOnlyLinear: Linear {
    let inputDimensions: Int
    let outputDimensions: Int
    let quantizationScales: MLXArray
    let quantizationBiases: MLXArray
    let rotation: MLXArray

    override var shape: (Int, Int) {
        (outputDimensions, inputDimensions)
    }

    init?(codes: MLXArray, rowScales: MLXArray, bias: MLXArray?) {
        guard codes.ndim == 2,
              codes.dtype == .int8,
              codes.dim(1).isMultiple(of: 256),
              rowScales.size == codes.dim(0)
        else {
            return nil
        }
        self.outputDimensions = codes.dim(0)
        self.inputDimensions = codes.dim(1)
        let scales = MLX.repeated(
            rowScales.reshaped(outputDimensions, 1),
            count: inputDimensions / 128,
            axis: 1
        )
        self.quantizationScales = scales
        self.quantizationBiases = scales * -128
        self.rotation = miniMaxMusic3ConvRotH256()
        let unsigned = (codes.asType(.int16) + 128).asType(.uint8).contiguous()
        let packed = unsigned.view(dtype: .uint32).reshaped(outputDimensions, inputDimensions / 4)
        super.init(weight: packed, bias: bias)
        MLX.eval(weight, quantizationScales, quantizationBiases)
        freeze()
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let rotated = rotate(input)
        var output = MLX.quantizedMM(
            rotated,
            weight,
            scales: quantizationScales,
            biases: quantizationBiases,
            transpose: true,
            groupSize: 128,
            bits: 8,
            mode: .affine
        ).asType(input.dtype)
        if let bias {
            output = output + bias.asType(output.dtype)
        }
        return output
    }

    func denseReferenceWeight(dtype: DType) -> MLXArray {
        let rowScales = quantizationScales[0..., 0..<1]
        let signedCodes = weight.view(dtype: .uint8).asType(.int16) - 128
        let rotatedWeight = signedCodes.asType(.float32) * rowScales.asType(.float32)
        return MLX.matmul(
            rotatedWeight.reshaped(-1, 256),
            rotation.transposed()
        ).reshaped(outputDimensions, inputDimensions).asType(dtype)
    }

    private func rotate(_ input: MLXArray) -> MLXArray {
        let shape = input.shape
        return MLX.matmul(
            input.reshaped(-1, shape.last! / 256, 256),
            rotation.asType(input.dtype)
        ).reshaped(shape)
    }
}

private func miniMaxMusic3ConvRotH256() -> MLXArray {
    let h4: [[Float]] = [
        [1, 1, 1, -1],
        [1, 1, -1, 1],
        [1, -1, 1, 1],
        [-1, 1, 1, 1],
    ]
    let scale = Float(1 / Double(256).squareRoot())
    var values: [Float] = []
    values.reserveCapacity(256 * 256)
    for row in 0..<256 {
        for column in 0..<256 {
            var value: Float = 1
            var rowDigits = row
            var columnDigits = column
            for _ in 0..<4 {
                value *= h4[rowDigits % 4][columnDigits % 4]
                rowDigits /= 4
                columnDigits /= 4
            }
            values.append(value * scale)
        }
    }
    return MLXArray(values).reshaped(256, 256)
}
