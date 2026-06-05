import MLX
import MLXNN

/// SDNQ asymmetric uint4 embedding weights.
///
/// The forward path gathers packed rows first and dequantizes only the requested
/// token rows, mirroring upstream SDNQ's embedding implementation.
public final class SDNQUInt4Embedding: Embedding {
    private let embeddingCount: Int
    private let dimensions: Int
    private let scales: MLXArray
    private let zeroPoints: MLXArray
    private let packedWeight: MLXArray
    private let groupSize: Int
    private let outputDType: DType
    private var cachedDequantizedWeight: MLXArray?

    public init(
        weight: MLXArray,
        scales: MLXArray,
        zeroPoints: MLXArray,
        embeddingCount: Int,
        dimensions: Int,
        groupSize: Int,
        outputDType: DType = .bfloat16
    ) {
        self.embeddingCount = embeddingCount
        self.dimensions = dimensions
        self.scales = scales
        self.zeroPoints = zeroPoints
        self.packedWeight = weight
        self.groupSize = groupSize
        self.outputDType = outputDType
        super.init(weight: MLX.zeros([embeddingCount, dimensions], dtype: outputDType))
    }

    public override var shape: (Int, Int) {
        (embeddingCount, dimensions)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(dimensions % 2 == 0, "SDNQ uint4 embedding dimension must be even.")
        precondition(dimensions % groupSize == 0, "SDNQ uint4 embedding dimension must be divisible by group size.")

        var tokenIds = x
        if tokenIds.dtype != .int32 {
            tokenIds = tokenIds.asType(.int32)
        }

        let groupCount = dimensions / groupSize
        let packedRows = packedWeight.reshaped([embeddingCount, dimensions / 2])
        let selectedPacked = packedRows[tokenIds]
        let selectedScales = scales[tokenIds]
        let selectedZeroPoints = zeroPoints[tokenIds]
        let quantizedShape = tokenIds.shape + [groupCount, groupSize]
        let outputShape = tokenIds.shape + [dimensions]
        return SDNQUInt4Dequantizer.dequantize(
            weight: selectedPacked,
            scales: selectedScales,
            zeroPoints: selectedZeroPoints,
            quantizedShape: quantizedShape,
            resultShape: outputShape,
            dtype: outputDType
        )
    }

    public override func asLinear(_ x: MLXArray) -> MLXArray {
        MLX.matmul(x, dequantizedWeight(dtype: x.dtype).T)
    }

    private func dequantizedWeight(dtype: DType) -> MLXArray {
        if let cachedDequantizedWeight, cachedDequantizedWeight.dtype == dtype {
            return cachedDequantizedWeight
        }
        let groupCount = dimensions / groupSize
        let dequantized = SDNQUInt4Dequantizer.dequantize(
            weight: packedWeight,
            scales: scales,
            zeroPoints: zeroPoints,
            quantizedShape: [embeddingCount, groupCount, groupSize],
            resultShape: [embeddingCount, dimensions],
            dtype: dtype
        )
        eval(dequantized)
        cachedDequantizedWeight = dequantized
        return dequantized
    }
}
