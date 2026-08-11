import MLX
import MLXNN

/// ModelOpt NVFP4 is the same E2M1 packing and E4M3 block-scale format that
/// MLX calls `nvfp4`. NVIDIA checkpoints additionally carry one FP32 scale per
/// matrix, which is deliberately retained instead of folding/requantizing it.
final class NemotronHNVFP4Linear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "global_scale") var globalScale: MLXArray

    init(inputDimensions: Int, outputDimensions: Int) {
        self._weight.wrappedValue = MLXArray.zeros(
            [outputDimensions, inputDimensions / 8],
            dtype: .uint32
        )
        self._scales.wrappedValue = MLXArray.zeros(
            [outputDimensions, inputDimensions / 16],
            dtype: .uint8
        )
        self._globalScale.wrappedValue = MLXArray(1, dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // MLX's Metal NVFP4 kernels have distinct fast paths here. Keeping the
        // one-token input in BF16 makes autoregressive decode substantially
        // faster, while applying the linear scalar to a multi-token input is
        // faster for prefill and speculative verification blocks.
        let isSingleToken = x.ndim < 2 || x.dim(x.ndim - 2) == 1
        let input = isSingleToken ? x : x.asType(.float32) * globalScale
        let output = MLX.quantizedMM(
            input,
            weight,
            scales: scales,
            biases: nil,
            transpose: true,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4
        )
        if isSingleToken {
            return (output.asType(.float32) * globalScale).asType(x.dtype)
        }
        return output.asType(x.dtype)
    }
}

final class NemotronHNVFP4SwitchLinear: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "global_scale") var globalScale: MLXArray

    init(inputDimensions: Int, outputDimensions: Int, expertCount: Int) {
        self._weight.wrappedValue = MLXArray.zeros(
            [expertCount, outputDimensions, inputDimensions / 8],
            dtype: .uint32
        )
        self._scales.wrappedValue = MLXArray.zeros(
            [expertCount, outputDimensions, inputDimensions / 16],
            dtype: .uint8
        )
        self._globalScale.wrappedValue = MLXArray.ones([expertCount], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let topK = indices.dim(2)
        let inputDimensions = x.dim(x.ndim - 1)
        let routeCount = batch * length * topK
        let flatInput: MLXArray
        if x.ndim == 4 && x.dim(2) == topK {
            flatInput = x.reshaped(routeCount, 1, inputDimensions)
        } else {
            flatInput = MLX.repeated(
                x.expandedDimensions(axis: -2),
                count: topK,
                axis: -2
            ).reshaped(routeCount, 1, inputDimensions)
        }
        let flatIndices = indices.reshaped(routeCount)
        let selectedScales = globalScale.take(flatIndices, axis: 0)
            .reshaped(routeCount, 1, 1)
        let input = length == 1
            ? flatInput
            : flatInput.asType(.float32) * selectedScales
        let output = portableGatherQuantizedMM(
            input,
            weight,
            scales: scales,
            biases: nil,
            rhsIndices: flatIndices,
            transpose: true,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4,
            sortedIndices: false
        )
        let scaledOutput = length == 1
            ? output.asType(.float32) * selectedScales
            : output
        return scaledOutput.asType(x.dtype)
            .reshaped(batch, length, topK, output.dim(-1))
    }
}
