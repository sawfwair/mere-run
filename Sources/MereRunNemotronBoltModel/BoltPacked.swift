import Foundation
import MLX
import MLXFast

public enum BoltPackedFormat: Sendable {
    case iq2XXS
    case q8_0

    var blockSize: Int { self == .iq2XXS ? 256 : 32 }
    var blockBytes: Int { self == .iq2XXS ? 66 : 34 }
}

/// Retains original GGML blocks. The Metal path decodes only the selected
/// expert rows inside the dot product; it never expands the full model.
public struct BoltPackedMatrix {
    let weight: MLXArray
    let format: BoltPackedFormat
    let inputSize: Int
    let outputSize: Int
    let expertCount: Int

    public init(weight: MLXArray, format: BoltPackedFormat, inputSize: Int, outputSize: Int) throws {
        let experts = weight.ndim == 4 ? weight.dim(0) : 1
        guard inputSize > 0, outputSize > 0, inputSize.isMultiple(of: format.blockSize),
              weight.dtype == .uint8,
              weight.shape == [outputSize, inputSize / format.blockSize, format.blockBytes]
                || weight.shape == [experts, outputSize, inputSize / format.blockSize, format.blockBytes] else {
            throw NemotronBoltError.invalidCheckpoint("Invalid packed matrix dimensions")
        }
        self.weight = weight
        self.format = format
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.expertCount = experts
    }

    public func callAsFunction(_ input: MLXArray, indices: MLXArray? = nil) -> MLXArray {
        precondition(input.dim(-1) == inputSize)
        let routes = indices?.size ?? (input.size / inputSize)
        let topK = indices?.dim(-1) ?? 1
        let expanded = input.size / inputSize == routes
        let flat = input.reshaped(-1, inputSize)
        let selection = indices?.reshaped(-1).asType(.uint32) ?? MLXArray.zeros([routes], dtype: .uint32)
        #if os(macOS)
        if Device.defaultDevice().deviceType == .gpu {
            let output = BoltPackedMetal.kernel(
                [flat, weight, selection, BoltIQ2Grid.array],
                template: [
                    ("T", input.dtype), ("K", inputSize), ("O", outputSize),
                    ("BS", format.blockSize), ("BB", format.blockBytes),
                    ("IQ2", format == .iq2XXS), ("TOPK", expanded ? 1 : topK),
                ],
                grid: (32, outputSize, routes), threadGroup: (32, 4, 1),
                outputShapes: [[routes, outputSize]], outputDTypes: [input.dtype]
            )[0]
            let shape = indices.map { $0.shape + [outputSize] } ?? Array(input.shape.dropLast()) + [outputSize]
            return output.reshaped(shape)
        }
        #endif
        // Portable reference path for structural/parity tests and CPU hosts.
        let bytes = weight.asArray(UInt8.self)
        let ids = selection.asArray(UInt32.self)
        let x = flat.asType(.float32).asArray(Float.self)
        var values = [Float](repeating: 0, count: routes * outputSize)
        for route in 0..<routes {
            let inputRow = expanded ? route : route / topK
            for row in 0..<outputSize {
                let base = (Int(ids[route]) * outputSize + row) * (inputSize / format.blockSize) * format.blockBytes
                var sum: Float = 0
                for block in 0..<(inputSize / format.blockSize) {
                    let decoded = Self.decodeBlock(Array(bytes[(base + block * format.blockBytes)..<(base + (block + 1) * format.blockBytes)]), format: format)
                    for column in 0..<format.blockSize {
                        sum += x[inputRow * inputSize + block * format.blockSize + column] * decoded[column]
                    }
                }
                values[route * outputSize + row] = sum
            }
        }
        let shape = indices.map { $0.shape + [outputSize] } ?? Array(input.shape.dropLast()) + [outputSize]
        return MLXArray(values, shape).asType(input.dtype)
    }

    /// Independent scalar decoder, also used as the portable reference path.
    public static func decodeBlock(_ bytes: [UInt8], format: BoltPackedFormat) -> [Float] {
        precondition(bytes.count == format.blockBytes)
        let scale = Float(Float16(bitPattern: UInt16(bytes[0]) | UInt16(bytes[1]) << 8))
        if format == .q8_0 {
            return bytes.dropFirst(2).map { scale * Float(Int8(bitPattern: $0)) }
        }
        var result = [Float](repeating: 0, count: 256)
        for group in 0..<8 {
            let base = 2 + group * 8
            let signsAndScale = (0..<4).reduce(UInt32(0)) { $0 | UInt32(bytes[base + 4 + $1]) << (8 * $1) }
            let groupScale = scale * (0.5 + Float(signsAndScale >> 28)) * 0.25
            for lane in 0..<4 {
                let grid = BoltIQ2Grid.values[Int(bytes[base + lane])]
                let sevenSigns = (signsAndScale >> (7 * lane)) & 127
                let signs = sevenSigns | UInt32(sevenSigns.nonzeroBitCount & 1) << 7
                for column in 0..<8 {
                    let value = Float((grid >> (8 * column)) & 255)
                    result[group * 32 + lane * 8 + column] = groupScale * value * (signs & (1 << column) == 0 ? 1 : -1)
                }
            }
        }
        return result
    }
}

#if os(macOS)
private enum BoltPackedMetal {
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_bolt_packed_mm", inputNames: ["x", "w", "ids", "grid_values"],
        outputNames: ["y"], source: """
        uint row = thread_position_in_grid.y;
        uint route = thread_position_in_grid.z;
        uint lane = thread_index_in_simdgroup;
        uint expert = ids[route];
        auto xp = x + (route / TOPK) * K;
        auto wp = w + (expert * O + row) * (K / BS) * BB;
        float total = 0.0f;
        for (uint block = 0; block < K / BS; ++block) {
            auto p = wp + block * BB;
            float d = float(as_type<half>(ushort(uint(p[0]) | (uint(p[1]) << 8))));
            if (IQ2) {
                for (uint group = 0; group < 8; ++group) {
                    auto q = p + 2 + group * 8;
                    uint aux = uint(q[4]) | (uint(q[5]) << 8) | (uint(q[6]) << 16) | (uint(q[7]) << 24);
                    uint sub = lane / 8;
                    uint j = lane % 8;
                    uint signs = (aux >> (7 * sub)) & 127;
                    signs |= (popcount(signs) & 1) << 7;
                    ulong grid = grid_values[q[sub]];
                    float v = float((grid >> (8 * j)) & 255);
                    float ds = d * (0.5f + float(aux >> 28)) * 0.25f;
                    float sign = (signs & (1 << j)) ? -1.0f : 1.0f;
                    total += float(xp[block * 256 + group * 32 + lane]) * ds * v * sign;
                }
            } else {
                total += float(xp[block * 32 + lane]) * d * float(as_type<char>(p[2 + lane]));
            }
        }
        total = simd_sum(total);
        if (lane == 0) y[route * O + row] = T(total);
        """
    )
}
#endif
