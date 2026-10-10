import Foundation
import MLX
import MLXFast

/// A row-major CQ2/CQ4 matrix. Only requested embedding rows are expanded.
struct WhistleCactusQuant {
    let packed: MLXArray
    let norms: MLXArray
    let codebook: MLXArray
    let columns: Int
    let rows: Int
    let bits: Int
    var rowOffset = 0

    func slice(_ range: Range<Int>) -> Self {
        var result = self
        result.rowOffset += range.lowerBound
        return Self(packed: packed, norms: norms, codebook: codebook, columns: columns,
                    rows: range.count, bits: bits, rowOffset: result.rowOffset)
    }

    static func rotate(_ input: MLXArray) -> MLXArray {
        var x = input.reshaped(-1, 128)
        for width in [1, 2, 4, 8, 16, 32, 64] {
            let pairs = x.reshaped(-1, 2, width)
            let a = pairs[0..., 0, 0...]
            let b = pairs[0..., 1, 0...]
            x = stacked([a + b, a - b], axis: 1).reshaped(-1, 128)
        }
        return x.reshaped(input.shape) / sqrt(Float(128))
    }

    func project(_ input: MLXArray) -> MLXArray {
        let shape = Array(input.shape.dropLast()) + [rows]
        let rotated = Self.rotate(input).reshaped(-1, columns)
        let parameters = MLXArray([Int32(columns), Int32(rows), Int32(bits), Int32(rowOffset)])
        return Self.multiply([packed, norms, codebook, rotated, parameters],
                             grid: (32, rows, rotated.dim(0)), threadGroup: (32, 1, 1),
                             outputShapes: [shape], outputDTypes: [.float32])[0]
    }

    func gather(_ indices: MLXArray) -> MLXArray {
        let parameters = MLXArray([Int32(columns), Int32(bits), Int32(rowOffset)])
        let rotated = Self.readRows([packed, norms, codebook, indices.asType(.int32), parameters],
                                    grid: (indices.size * columns, 1, 1), threadGroup: (256, 1, 1),
                                    outputShapes: [[indices.size, columns]], outputDTypes: [.float32])[0]
        return Self.rotate(rotated)
    }

    private static let multiply = MLXFast.metalKernel(
        name: "whistle_cq_matmul", inputNames: ["w", "norm", "cb", "x", "p"], outputNames: ["y"], source: """
        const int lane = thread_position_in_threadgroup.x;
        const int row = threadgroup_position_in_grid.y;
        const int batch = threadgroup_position_in_grid.z;
        const int cols = p[0], rows = p[1], bits = p[2], offset = p[3];
        const int stored = row + offset;
        float acc = 0;
        for (int k = lane; k < cols; k += 32) {
            int bp = k * bits;
            uint index = (uint(w[stored * cols * bits / 8 + bp / 8]) >> (bp % 8)) & ((1u << bits) - 1);
            acc += x[batch * cols + k] * cb[index] * norm[stored * (cols / 128) + k / 128];
        }
        float total = simd_sum(acc);
        if (lane == 0) y[batch * rows + row] = total;
        """, ensureRowContiguous: true
    )

    private static let readRows = MLXFast.metalKernel(
        name: "whistle_cq_rows", inputNames: ["w", "norm", "cb", "indices", "p"], outputNames: ["y"], source: """
        const uint i = thread_position_in_grid.x;
        const int cols = p[0], bits = p[1], offset = p[2];
        if (i >= uint(indices_shape[0] * cols)) return;
        const int col = i % cols, row = indices[i / cols] + offset;
        const int bp = col * bits;
        uint index = (uint(w[row * cols * bits / 8 + bp / 8]) >> (bp % 8)) & ((1u << bits) - 1);
        y[i] = cb[index] * norm[row * (cols / 128) + col / 128];
        """, ensureRowContiguous: true
    )
}
