import Foundation
import MLX
import MLXFast

final class Gemma4PolarTensorState {
    let packed: MLXArray
    let norms: MLXArray
    let bits: Int
    let dtype: DType
    let headDim: Int
    let rotation: MLXArray
    let rotationTransposed: MLXArray
    let centroids: MLXArray
    let innerBoundaries: MLXArray
    let tokenCount: Int

    var tokenCapacity: Int {
        packed.dim(2)
    }

    var packedWidth: Int {
        packed.dim(3)
    }

    init(source: MLXArray, bits: Int) {
        let headDim = source.dim(3)
        let rotation = Gemma4PolarRotation.matrix(dim: headDim)
        let rotationTransposed = rotation.transposed()
        let centroids = Gemma4PolarCodebook.centroids(bits: bits, dim: headDim)
        let innerBoundaries = Gemma4PolarCodebook.innerBoundaries(bits: bits, dim: headDim)
        let source32 = source.asType(.float32)
        let norms = MLX.sqrt(MLX.sum(source32 * source32, axis: -1, keepDims: true))
        let normalized = source32 / MLX.maximum(norms, MLXArray(Float32(1e-8)))
        let rotated = MLX.matmul(normalized, rotationTransposed)
        let packedWidth = (headDim * bits + 31) / 32
        let packKernel = Gemma4PolarFastKernels.packKernel(bits: bits, dim: headDim, packedWidth: packedWidth)
        let packed = packKernel(
            [rotated, innerBoundaries],
            template: [
                ("Bits", bits),
                ("Dim", headDim),
                ("PackedWidth", packedWidth),
            ],
            grid: (packedWidth, source.dim(1), source.dim(0) * source.dim(2)),
            threadGroup: (max(1, min(32, packedWidth)), 1, 1),
            outputShapes: [[source.dim(0), source.dim(1), source.dim(2), packedWidth]],
            outputDTypes: [.uint32]
        )[0]

        self.packed = packed
        self.norms = norms
        self.bits = bits
        self.dtype = source.dtype
        self.headDim = headDim
        self.rotation = rotation
        self.rotationTransposed = rotationTransposed
        self.centroids = centroids
        self.innerBoundaries = innerBoundaries
        self.tokenCount = source.dim(2)
    }

    init(
        packed: MLXArray,
        norms: MLXArray,
        bits: Int,
        dtype: DType,
        headDim: Int,
        rotation: MLXArray,
        rotationTransposed: MLXArray,
        centroids: MLXArray,
        innerBoundaries: MLXArray,
        tokenCount: Int
    ) {
        self.packed = packed
        self.norms = norms
        self.bits = bits
        self.dtype = dtype
        self.headDim = headDim
        self.rotation = rotation
        self.rotationTransposed = rotationTransposed
        self.centroids = centroids
        self.innerBoundaries = innerBoundaries
        self.tokenCount = tokenCount
    }

    func appending(_ source: MLXArray) -> Gemma4PolarTensorState {
        let next = Gemma4PolarTensorState(source: source, bits: bits)
        return Gemma4PolarTensorState(
            packed: Gemma4KVTokenStorage.appended(packed, rows: next.packed, validCount: tokenCount),
            norms: Gemma4KVTokenStorage.appended(norms, rows: next.norms, validCount: tokenCount),
            bits: bits,
            dtype: dtype,
            headDim: headDim,
            rotation: rotation,
            rotationTransposed: rotationTransposed,
            centroids: centroids,
            innerBoundaries: innerBoundaries,
            tokenCount: tokenCount + next.tokenCount
        )
    }

    func dequantized() -> MLXArray {
        dequantized(tokenRange: 0..<tokenCount)
    }

    func dequantized(tokenRange: Range<Int>) -> MLXArray {
        let unpackKernel = Gemma4PolarFastKernels.unpackKernel(bits: bits, dim: headDim, packedWidth: packedWidth)
        let roundedDim = ((headDim + 31) / 32) * 32
        let tokenCounts = MLXArray([UInt32(tokenRange.count), UInt32(tokenRange.lowerBound)])
        let rotated = unpackKernel(
            [packed, norms, centroids, tokenCounts],
            template: [
                ("Bits", bits),
                ("Dim", headDim),
                ("PackedWidth", packedWidth),
            ],
            grid: (roundedDim, packed.dim(1), packed.dim(0) * tokenRange.count),
            threadGroup: (32, 1, 1),
            outputShapes: [[packed.dim(0), packed.dim(1), tokenRange.count, headDim]],
            outputDTypes: [.float32]
        )[0]
        return MLX.matmul(rotated, rotation).asType(dtype)
    }
}
