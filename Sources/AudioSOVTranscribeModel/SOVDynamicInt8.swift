import Foundation
import MLX
import MLXFast
#if os(macOS)
import Metal
#endif

/// Experimental M5 GPU FFN arithmetic. Original checkpoint tensors remain intact.
package final class SOVDynamicInt8 {
    package static var isSupported: Bool {
        #if os(macOS)
        if #available(macOS 26.2, *) {
            return Device.defaultDevice().deviceType == .gpu && MTLCreateSystemDefaultDevice()?.supportsFamily(.apple10) == true
        }
        #endif
        return false
    }

    private let packKernel: MLXFast.MLXFastKernel
    private let matmulKernel: MLXFast.MLXFastKernel
    private let packed: [MLXArray]
    private let exponent: Int
    private let rotate: Bool
    private let inputSize: Int
    private let outputSize: Int

    package init(weight: MLXArray, rotate: Bool, exponent: Int = 0) throws {
        guard (0...2).contains(exponent) else {
            throw SOVTranscribeError.invalid("Dynamic int8 block exponents must be in 0...2.")
        }
        self.exponent = exponent
        guard Self.isSupported else {
            throw SOVTranscribeError.invalid("Dynamic int8 SOV kernels require an M5 GPU and macOS 26.2 or later.")
        }
        guard weight.ndim == 2, weight.dtype == .bfloat16,
              weight.dim(0).isMultiple(of: 64), weight.dim(1).isMultiple(of: 256),
              (256...8192).contains(weight.dim(1)) else {
            throw SOVTranscribeError.invalid("Dynamic int8 weights require BF16, N divisible by 64, and K divisible by 256 in 256...8192.")
        }
        inputSize = weight.dim(1); outputSize = weight.dim(0); self.rotate = rotate
        let url = Bundle.module.url(forResource: "SOVDynamicInt8", withExtension: "metal", subdirectory: "Kernels")!
        let source = try String(contentsOf: url, encoding: .utf8)
        let header = source.components(separatedBy: "// ROTATE_BODY\n")[0]
        let fused = source.components(separatedBy: "// FUSED_PACK_BODY\n")[1].components(separatedBy: "// PACK_BODY\n")[0]
        let matmul = source.components(separatedBy: "// MATMUL_BODY\n")[1]
        packKernel = MLXFast.metalKernel(name: "sov_dynamic_fused_pack", inputNames: ["input"],
            outputNames: ["codes", "row_scale", "exponents"], source: fused, header: header)
        matmulKernel = MLXFast.metalKernel(name: "sov_dynamic_int8_mpp",
            inputNames: ["aq", "a_scale", "a_exp", "bq", "b_scale", "b_exp", "bias"],
            outputNames: ["output"], source: matmul, header: header)
        packed = Self.pack(weight, rotate: rotate, exponent: exponent, kernel: packKernel)
        eval(packed)
    }

    private static func pack(_ x: MLXArray, rotate: Bool, exponent: Int, kernel: MLXFast.MLXFastKernel) -> [MLXArray] {
        let rows = x.dim(0), width = x.dim(1), threads = width / 256 * 32
        return kernel([x], template: [("K", width), ("BK", 256), ("Rotate", rotate), ("TileMajor", true), ("MaxExponent", exponent)],
            grid: (rows * threads, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [x.shape, [rows], [rows, width / 256]], outputDTypes: [.int8, .float32, .uint8])
    }

    package func callAsFunction(_ x: MLXArray, bias: MLXArray) -> MLXArray {
        precondition(x.ndim >= 2 && x.dim(-1) == inputSize && x.dtype == .bfloat16 && bias.shape == [outputSize])
        let rows = x.size / inputSize, paddedRows = (rows + 63) / 64 * 64
        let matrix = x.reshaped(rows, inputSize)
        let padded = rows == paddedRows ? matrix : concatenated([
            matrix, MLXArray.zeros([paddedRows - rows, inputSize], dtype: x.dtype)
        ], axis: 0)
        let activation = Self.pack(padded, rotate: rotate, exponent: exponent, kernel: packKernel)
        let result = matmulKernel(activation + packed + [bias],
            template: [("M", paddedRows), ("N", outputSize), ("K", inputSize), ("BK", 256),
                       ("IntegerAccumulation", true), ("MaxExponent", exponent), ("TileMajor", true), ("Out", x.dtype)],
            grid: (paddedRows / 64 * (outputSize / 64) * 128, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[paddedRows, outputSize]], outputDTypes: [x.dtype])[0]
        var shape = x.shape
        shape[shape.count - 1] = outputSize
        return result[0..<rows, 0...].reshaped(shape)
    }
}
