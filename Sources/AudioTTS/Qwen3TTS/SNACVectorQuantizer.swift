import Foundation
import MLX
import MLXNN

// MARK: - Vector Quantizer

/// Single codebook for vector quantization
final class VectorQuantize: Module {
    @ModuleInfo(key: "codebook") var codebook: MLXArray

    let codebookSize: Int
    let codebookDim: Int

    init(codebookSize: Int, codebookDim: Int) {
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim

        // L2-normalized codebook
        self._codebook.wrappedValue = MLXRandom.normal([codebookSize, codebookDim]) * 0.02
    }

    /// Lookup codes from codebook
    /// - Parameter codes: [B, T] tensor of code indices
    /// - Returns: [B, T, D] tensor of quantized vectors
    func fromCodes(_ codes: MLXArray) -> MLXArray {
        // Normalize codebook for L2 lookup
        let norm = MLX.sqrt(MLX.sum(codebook * codebook, axis: -1, keepDims: true) + 1e-12)
        let normalizedCodebook = codebook / norm

        // Gather from codebook
        let flatCodes = codes.reshaped(-1)
        let gathered = normalizedCodebook[flatCodes]

        // Reshape back to [B, T, D]
        let shape = codes.shape + [codebookDim]
        return gathered.reshaped(shape)
    }
}

// MARK: - Residual Vector Quantizer

/// Residual vector quantization with multiple codebooks
final class ResidualVectorQuantize: Module {
    @ModuleInfo(key: "layers") var layers: [VectorQuantize]
    @ModuleInfo(key: "project_in") var projectIn: Linear?
    @ModuleInfo(key: "project_out") var projectOut: Linear?

    let numQuantizers: Int
    let codebookSize: Int
    let codebookDim: Int
    let inputDim: Int

    init(
        inputDim: Int,
        codebookSize: Int = 4096,
        codebookDim: Int = 8,
        numQuantizers: Int = 3
    ) {
        self.numQuantizers = numQuantizers
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self.inputDim = inputDim

        // Create quantizer layers
        var quantizers: [VectorQuantize] = []
        for _ in 0..<numQuantizers {
            quantizers.append(VectorQuantize(codebookSize: codebookSize, codebookDim: codebookDim))
        }
        self._layers.wrappedValue = quantizers

        // Projection layers if dimensions differ
        if inputDim != codebookDim {
            self._projectIn.wrappedValue = Linear(inputDim, codebookDim)
            self._projectOut.wrappedValue = Linear(codebookDim, inputDim)
        } else {
            self._projectIn.wrappedValue = nil
            self._projectOut.wrappedValue = nil
        }
    }

    /// Reconstruct audio features from hierarchical codes
    /// - Parameter codes: Array of code tensors, one per quantizer level
    ///   - codes[0]: [B, T] base layer codes
    ///   - codes[1]: [B, T*2] second layer codes
    ///   - codes[2]: [B, T*4] third layer codes
    /// - Returns: [B, T*4, inputDim] reconstructed features
    func fromCodes(_ codes: [[Int]]) -> MLXArray {
        // Convert to MLXArrays
        let codeArrays = codes.map { MLXArray($0.map { Int32($0) }).expandedDimensions(axis: 0) }
        return fromCodeArrays(codeArrays)
    }

    /// Reconstruct from MLXArray codes
    func fromCodeArrays(_ codes: [MLXArray]) -> MLXArray {
        // The SNAC model has 3 quantizer levels with different temporal resolutions:
        // Layer 0: stride 8 (slowest, base layer)
        // Layer 1: stride 4 (2x faster)
        // Layer 2: stride 2 (4x faster than base)

        // Get quantized vectors from each level
        var quantizedLayers: [MLXArray] = []
        for (i, layer) in layers.enumerated() {
            if i < codes.count {
                let quantized = layer.fromCodes(codes[i])
                quantizedLayers.append(quantized)
            }
        }

        // Upsample and sum residuals
        // Start with the coarsest (base) layer
        guard !quantizedLayers.isEmpty else {
            fatalError("No codes provided")
        }

        var result = quantizedLayers[0]

        // Upsample base layer 2x and add layer 1
        if quantizedLayers.count > 1 {
            result = upsample(result, factor: 2)
            result = result + quantizedLayers[1]
        }

        // Upsample 2x more and add layer 2
        if quantizedLayers.count > 2 {
            result = upsample(result, factor: 2)
            result = result + quantizedLayers[2]
        }

        // Project back to input dimension if needed
        if let projectOut {
            result = projectOut(result)
        }

        return result
    }

    /// Simple repeat-based upsampling
    private func upsample(_ x: MLXArray, factor: Int) -> MLXArray {
        // x: [B, T, D]
        let (batch, time, dim) = (x.dim(0), x.dim(1), x.dim(2))
        // Repeat each timestep
        let expanded = x.expandedDimensions(axis: 2)  // [B, T, 1, D]
        let repeated = MLX.tiled(expanded, repetitions: [1, 1, factor, 1])  // [B, T, factor, D]
        return repeated.reshaped(batch, time * factor, dim)
    }
}
