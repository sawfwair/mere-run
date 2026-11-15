import MLX
import MLXFast

/// Re-encodes the same unsigned six-bit codes in eight-bit lanes. Scales and
/// biases are shared; no floating-point weights are reconstructed or requantized.
final class Gemma4ExpandedExpertWeights {
    /// A non-Module holder keeps derived arrays out of parameter reflection and
    /// lets invalidation release the previous encoding without an item-cache alias.
    final class Cache {
        private let expander: Gemma4ExpandedExpertWeights
        private var expanded: MLXArray?

        init(_ packed: MLXArray, inputWidth: Int, expander: Gemma4ExpandedExpertWeights) {
            self.expander = expander
            expanded = expander(packed, inputWidth: inputWidth)
        }

        var value: MLXArray? { expanded }
        func invalidate() { expanded = nil }
        func resolve(_ packed: MLXArray, inputWidth: Int) -> MLXArray {
            if expanded == nil { expanded = expander(packed, inputWidth: inputWidth) }
            return expanded!
        }
    }

    private let kernel = MLXFast.metalKernel(name: "gemma4_expand_q6_codes",
        inputNames: ["packed"], outputNames: ["expanded"], source: """
        const uint i = thread_position_in_grid.x;
        if (i >= Count) return;
        const device uchar* bytes = reinterpret_cast<const device uchar*>(packed);
        const uint a = bytes[3*i], b = bytes[3*i+1], c = bytes[3*i+2];
        const uint q0 = a & 63;
        const uint q1 = (a >> 6) | ((b & 15) << 2);
        const uint q2 = (b >> 4) | ((c & 3) << 4);
        const uint q3 = c >> 2;
        expanded[i] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        """)

    func callAsFunction(_ packed: MLXArray, inputWidth: Int) -> MLXArray {
        let shape = Array(packed.shape.dropLast()) + [inputWidth / 4]
        let count = shape.reduce(1, *)
        return kernel([packed], template: [("Count", count)], grid: (count, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [shape], outputDTypes: [.uint32])[0]
    }
}
