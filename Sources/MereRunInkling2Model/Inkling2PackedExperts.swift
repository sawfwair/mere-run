import MLX

/// The checkpoint stores gate and up rows interleaved. Gather both in one
/// packed matmul, then use the same routed slots for the down projection.
struct Inkling2PackedExperts {
    let gateUp: Projection
    let down: Projection

    init(weights: Inkling2Weights, prefix: String) throws {
        gateUp = try Projection(weights: weights, key: prefix + "w13_weight")
        down = try Projection(weights: weights, key: prefix + "w2_weight")
    }

    func callAsFunction(
        _ input: MLXArray, indices: MLXArray, routing: MLXArray, gateCap: Float, upCap: Float, verifying: Bool
    ) -> MLXArray {
        #if os(macOS)
        let hidden: MLXArray
        if verifying, let outer = gateUp.expertScales,
           input.dtype == .bfloat16, input.dim(-1).isMultiple(of: 16), gateUp.packed.dim(-2).isMultiple(of: 8) {
            hidden = Inkling2ExpertEpilogues.fusedGate(
                input, packed: gateUp.packed, scales: gateUp.scales, indices: indices, outer: outer,
                gateCap: gateCap, upCap: upCap
            )
        } else {
            let expanded = expandedDimensions(input, axes: [-2, -3])
            let gates = gateUp.rawMatmul(expanded, indices: indices)
            hidden = Inkling2ExpertEpilogues.activate(gates, indices: indices, scales: gateUp.expertScales,
                                                        dtype: input.dtype, gateCap: gateCap, upCap: upCap)
        }
        if verifying, let outer = down.expertScales,
           hidden.dtype == .bfloat16, hidden.dim(-1).isMultiple(of: 16), down.packed.dim(-2).isMultiple(of: 8),
           indices.dim(-1) <= 8 {
            return Inkling2ExpertEpilogues.fusedDown(
                hidden, packed: down.packed, scales: down.scales, indices: indices, outer: outer, routing: routing
            )
        }
        let output = down.rawMatmul(hidden, indices: indices)
        return Inkling2ExpertEpilogues.reduce(output, indices: indices, scales: down.expertScales,
                                             routing: routing, dtype: input.dtype)
        #else
        let expanded = expandedDimensions(input, axes: [-2, -3])
        let projected = gateUp(expanded, indices: indices)
            .reshaped(input.dim(0), indices.dim(1), -1, 2)
        let activated = Inkling2Math.activation(gate: projected[.ellipsis, 0], up: projected[.ellipsis, 1],
                                                gateCap: gateCap, upCap: upCap)
        let output = down(activated.expandedDimensions(axis: -2), indices: indices).squeezed(axis: -2)
        return (output.asType(.float32) * routing.expandedDimensions(axis: -1))
            .sum(axis: -2).asType(input.dtype)
        #endif
    }

    struct Projection {
        let packed: MLXArray
        let scales: MLXArray
        let expertScales: MLXArray?

        func rawMatmul(_ input: MLXArray, indices: MLXArray) -> MLXArray {
            if expertScales != nil {
                return gatherQuantizedMM(input.asType(.float32), packed, scales: scales, biases: nil,
                                         rhsIndices: indices, groupSize: 16, bits: 4, mode: .nvfp4)
            }
            return gatherQuantizedMM(input, packed, scales: scales, biases: nil,
                                     rhsIndices: indices, groupSize: 32, bits: 8, mode: .mxfp8)
        }

        func callAsFunction(_ input: MLXArray, indices: MLXArray) -> MLXArray {
            let result = rawMatmul(input, indices: indices)
            if let expertScales {
                // MLX consumes the raw per-block E4M3 scales. Apply each
                // selected expert's outer factor in FP32 before BF16 rounding.
                let factor = take(expertScales, indices, axis: 0)
                return (result * expandedDimensions(factor, axes: [-1, -2])).asType(input.dtype)
            }
            return result
        }
    }
}

extension Inkling2PackedExperts.Projection {
    init(weights: Inkling2Weights, key: String) throws {
        let packed = try weights.read(key).view(dtype: .uint32)
        let scales = try weights.read(key + "_scale")
        let outerScale: MLXArray?
        if weights.dtype(key) == "U8" {
            outerScale = try weights.read(key + "_amax") / weights.read(key + "_outer_scale_denominator")
        } else {
            outerScale = nil
        }
        eval([packed, scales] + [outerScale].compactMap { $0 })
        self.init(packed: packed, scales: scales, expertScales: outerScale)
    }
}
