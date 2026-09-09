import Foundation
import MLX
import MLXNN

// MARK: - DenseLayer Protocol

/// Protocol for linear/dense layers that transform inputs via matrix multiplication.
/// Both `Linear` and `QuantizedLinear` conform to this protocol.
public protocol DenseLayer: Module {
    /// Forward pass: applies the linear transformation to the input.
    func callAsFunction(_ x: MLXArray) -> MLXArray
}

// Extend MLX's Linear to conform (QuantizedLinear inherits from Linear so gets it automatically)
extension Linear: DenseLayer {}

// MARK: - DenseLayer Factory

/// Factory for creating Linear or QuantizedLinear layers based on available weights.
/// Note: Using @unchecked Sendable since MLXArray is used within actor isolation.
public struct DenseLayerFactory: @unchecked Sendable {
    /// Pre-loaded weight arrays, keyed by full path (e.g., "transformer_blocks.0.attn.to_q.weight")
    public let arrays: [String: MLXArray]

    /// Quantization configuration (nil for full-precision models)
    public let quantConfig: QuantizationConfig?

    public init(arrays: [String: MLXArray] = [:], quantConfig: QuantizationConfig? = nil) {
        self.arrays = arrays
        self.quantConfig = quantConfig
    }

    /// Check if weights for a given path indicate quantization
    public func isQuantized(path: String) -> Bool {
        arrays["\(path).scales"] != nil && arrays["\(path).biases"] != nil
    }

    /// Create a dense layer for the given path.
    /// If quantized weights are available, creates QuantizedLinear; otherwise creates Linear.
    /// - Parameters:
    ///   - path: The module path (e.g., "x_embedder" or "transformer_blocks.0.attn.to_q")
    ///   - inputDim: Input dimension
    ///   - outputDim: Output dimension
    ///   - bias: Whether to include a bias term
    /// - Returns: A DenseLayer (either Linear or QuantizedLinear)
    public func makeDenseLayer(
        path: String,
        inputDim: Int,
        outputDim: Int,
        bias: Bool = true
    ) -> any DenseLayer {
        let weightKey = "\(path).weight"
        let scalesKey = "\(path).scales"
        let biasesKey = "\(path).biases"
        let biasKey = "\(path).bias"

        // Check if this is a quantized layer
        if let weight = arrays[weightKey],
           let scales = arrays[scalesKey],
           let biases = arrays[biasesKey],
           let quantConfig {
            // Create QuantizedLinear with pre-computed scales/biases
            // Note: QuantizedLinear init order is (weight, bias, scales, biases, groupSize, bits)
            let layerBias = bias ? arrays[biasKey] : nil
            return QuantizedLinear(
                weight: weight,
                bias: layerBias,
                scales: scales,
                biases: biases,
                groupSize: quantConfig.groupSize,
                bits: quantConfig.bits
            )
        }

        // Create standard Linear layer
        let linear = Linear(inputDim, outputDim, bias: bias)

        // If weights are available, apply them via update
        if let weight = arrays[weightKey] {
            var params: [String: MLXArray] = ["weight": weight]
            if bias, let b = arrays[biasKey] {
                params["bias"] = b
            }
            linear.update(parameters: ModuleParameters.unflattened(params))
        }

        return linear
    }

    /// Create a tuple of dense layers (used for MLP patterns)
    public func makeDenseLayerPair(
        path: String,
        inputDim: Int,
        hiddenDim: Int,
        outputDim: Int,
        bias: Bool = true
    ) -> (any DenseLayer, any DenseLayer) {
        let first = makeDenseLayer(path: "\(path).0", inputDim: inputDim, outputDim: hiddenDim, bias: bias)
        let second = makeDenseLayer(path: "\(path).1", inputDim: hiddenDim, outputDim: outputDim, bias: bias)
        return (first, second)
    }

    /// A factory that creates standard Linear layers (no pre-loaded weights)
    public static let standard = DenseLayerFactory()
}

// MARK: - Quantization Configuration

/// Configuration for quantized model loading
public struct QuantizationConfig: Codable, Sendable {
    public let bits: Int
    public let groupSize: Int

    public init(bits: Int, groupSize: Int) {
        self.bits = bits
        self.groupSize = groupSize
    }

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
    }
}

// MARK: - Weight Array Loader

/// Utility for loading weight arrays from pre-quantized model files
public enum QuantizedWeightLoader {
    /// Load all weight arrays from a model directory
    /// - Parameters:
    ///   - url: URL to the model.safetensors file
    /// - Returns: Dictionary of weight arrays keyed by path
    public static func loadArrays(from url: URL) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: url)
    }

    /// Load quantization config from a model directory
    public static func loadConfig(from url: URL) throws -> QuantizationConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(QuantizationConfig.self, from: data)
    }
}

// MARK: - Pre-Quantized Modules

/// An embedding layer backed by already-quantized weights (plus scales/biases).
///
/// This avoids materializing full-precision embedding matrices when loading pre-quantized models.
public final class PreQuantizedEmbedding: Embedding, Quantized {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public let scales: MLXArray
    public let biases: MLXArray?
    private var cachedLinearWeight: MLXArray?
    private var cachedLinearWeightDType: DType?

    public init(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.scales = scales
        self.biases = biases
        super.init(weight: weight)
        self.freeze()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let originalShape = x.shape
        let flat = x.flattened()
        let out = dequantized(
            weight[flat],
            scales: scales[flat],
            biases: biases == nil ? nil : biases![flat],
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )

        return out.reshaped(originalShape + [-1])
    }

    public override func asLinear(_ x: MLXArray) -> MLXArray {
        #if os(Linux)
        if Device.defaultDevice().deviceType == .gpu, mode == .affine {
            switch MLXCUDAQuant.nativeDecision(
                for: .quantizedMM,
                bits: bits,
                groupSize: groupSize,
                quantizationMode: mode
            ) {
            case .native:
                return nativeLinear(x)
            case .dense:
                return denseLinear(x)
            case .probe:
                let output = nativeLinear(x)
                do {
                    try MLX.checkedEval(output)
                    MLXCUDAQuant.completeProbe(
                        operation: .quantizedMM,
                        bits: bits,
                        groupSize: groupSize,
                        quantizationMode: mode,
                        supported: true
                    )
                    return output
                } catch {
                    MLXCUDAQuant.completeProbe(
                        operation: .quantizedMM,
                        bits: bits,
                        groupSize: groupSize,
                        quantizationMode: mode,
                        supported: false
                    )
                    return denseLinear(x)
                }
            }
        }
        #endif

        return nativeLinear(x)
    }

    private func nativeLinear(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }

    func denseLinear(_ x: MLXArray) -> MLXArray {
        let fullWeight: MLXArray
        if let cachedLinearWeight, cachedLinearWeightDType == x.dtype {
            fullWeight = cachedLinearWeight
        } else {
            fullWeight = dequantized(
                weight,
                scales: scales,
                biases: biases,
                groupSize: groupSize,
                bits: bits,
                mode: mode,
                dtype: x.dtype
            )
            eval(fullWeight)
            cachedLinearWeight = fullWeight
            cachedLinearWeightDType = x.dtype
        }
        return matmul(x, fullWeight.T)
    }
}

// MARK: - Pre-Quantized Weight Application

/// Utilities for swapping `Linear`/`Embedding` layers to pre-quantized variants and applying the
/// full parameter dictionary.
public enum PreQuantizedModelLoader {
    /// Replace `Linear`/`Embedding` leaf modules that have quantized tensors present in `arrays`.
    /// This enables loading weights saved by `saveQuantizedModule(...)` without materializing
    /// full-precision weights first.
    public static func applyQuantizedLeafModules(
        arrays: [String: MLXArray],
        quantConfig: QuantizationConfig,
        to model: Module
    ) throws {
        let updates: [(String, Module)] = model.leafModules().flattened().compactMap { path, module in
            guard let weight = arrays["\(path).weight"],
                  let scales = arrays["\(path).scales"] else {
                return nil
            }

            let biases = arrays["\(path).biases"]

            if module is Embedding {
                return (
                    path,
                    PreQuantizedEmbedding(
                        weight: weight,
                        scales: scales,
                        biases: biases,
                        groupSize: quantConfig.groupSize,
                        bits: quantConfig.bits
                    )
                )
            }

            if module is Linear {
                let bias = arrays["\(path).bias"]
                let q = QuantizedLinear(
                    weight: weight,
                    bias: bias,
                    scales: scales,
                    biases: biases,
                    groupSize: quantConfig.groupSize,
                    bits: quantConfig.bits
                )
                q.freeze()
                return (path, q)
            }

            return nil
        }

        guard !updates.isEmpty else { return }
        try model.update(modules: ModuleChildren.unflattened(updates), verify: .none)
    }
}
