import MLX

struct Inkling2ReferenceAffineProjection {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let spec: Inkling2Weights.ReferenceAffineSpec

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        quantizedMM(input.asType(scales.dtype), weight, scales: scales, biases: biases,
                    groupSize: spec.groupSize, bits: spec.bits, mode: .affine)
    }

    func slice(_ rows: Range<Int>) -> Self {
        Self(weight: weight[rows], scales: scales[rows], biases: biases[rows], spec: spec)
    }

    func embeddings(_ tokens: MLXArray) -> MLXArray {
        dequantized(weight[tokens], scales: scales[tokens], biases: biases[tokens],
                    groupSize: spec.groupSize, bits: spec.bits, mode: .affine)
    }

    func unpacked() -> MLXArray {
        dequantized(weight, scales: scales, biases: biases, groupSize: spec.groupSize, bits: spec.bits, mode: .affine)
    }
}
