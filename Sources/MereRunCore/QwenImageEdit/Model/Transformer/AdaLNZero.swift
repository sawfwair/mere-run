import Foundation
import MLX
import MLXNN

/// Adaptive Layer Normalization with Zero initialization (AdaLN-Zero)
/// Used in DiT/MMDiT architectures for timestep/condition-dependent modulation.
///
/// Produces 6 modulation parameters: (shift1, scale1, gate1, shift2, scale2, gate2)
/// where 1 = attention block, 2 = MLP block
public final class AdaLNZero: Module {
    public let hiddenSize: Int
    public let conditioningDim: Int

    @ModuleInfo(key: "linear") private var linear: any DenseLayer

    /// Initialize AdaLN-Zero module
    /// - Parameters:
    ///   - hiddenSize: Hidden dimension of the transformer
    ///   - conditioningDim: Dimension of the conditioning input (timestep embedding)
    ///   - factory: Factory for creating dense layers
    ///   - basePath: Base path for weight keys
    public init(
        hiddenSize: Int,
        conditioningDim: Int? = nil,
        factory: DenseLayerFactory = .standard,
        basePath: String = ""
    ) {
        self.hiddenSize = hiddenSize
        self.conditioningDim = conditioningDim ?? hiddenSize
        // Output 6 * hiddenSize for (shift1, scale1, gate1, shift2, scale2, gate2)
        let path = basePath.isEmpty ? "linear" : "\(basePath).linear"
        self._linear.wrappedValue = factory.makeDenseLayer(
            path: path,
            inputDim: self.conditioningDim,
            outputDim: 6 * hiddenSize,
            bias: true
        )
        super.init()
    }

    /// Compute modulation parameters from conditioning input
    /// - Parameter conditioning: Conditioning tensor [batch, conditioningDim]
    /// - Returns: Tuple of 6 tensors each [batch, 1, hiddenSize]:
    ///           (attnShift, attnScale, attnGate, mlpShift, mlpScale, mlpGate)
    public func callAsFunction(_ conditioning: MLXArray) -> AdaLNModulation {
        let mod = linear(MLXNN.silu(conditioning))

        // Split into 6 chunks
        let chunk = hiddenSize
        let attnShift = mod[0..., 0..<chunk].expandedDimensions(axis: 1)
        let attnScale = mod[0..., chunk..<(2*chunk)].expandedDimensions(axis: 1)
        let attnGate = mod[0..., (2*chunk)..<(3*chunk)].expandedDimensions(axis: 1)
        let mlpShift = mod[0..., (3*chunk)..<(4*chunk)].expandedDimensions(axis: 1)
        let mlpScale = mod[0..., (4*chunk)..<(5*chunk)].expandedDimensions(axis: 1)
        let mlpGate = mod[0..., (5*chunk)...].expandedDimensions(axis: 1)

        return AdaLNModulation(
            attnShift: attnShift,
            attnScale: attnScale,
            attnGate: attnGate,
            mlpShift: mlpShift,
            mlpScale: mlpScale,
            mlpGate: mlpGate
        )
    }
}

/// Container for AdaLN modulation parameters
public struct AdaLNModulation {
    public let attnShift: MLXArray   // [batch, 1, hidden]
    public let attnScale: MLXArray   // [batch, 1, hidden]
    public let attnGate: MLXArray    // [batch, 1, hidden]
    public let mlpShift: MLXArray    // [batch, 1, hidden]
    public let mlpScale: MLXArray    // [batch, 1, hidden]
    public let mlpGate: MLXArray     // [batch, 1, hidden]

    /// Apply pre-attention modulation: (1 + scale) * norm(x) + shift
    public func modulatePreAttn(_ x: MLXArray) -> MLXArray {
        (1 + attnScale) * x + attnShift
    }

    /// Apply attention gate
    public func gateAttn(_ x: MLXArray) -> MLXArray {
        attnGate * x
    }

    /// Apply pre-MLP modulation: (1 + scale) * norm(x) + shift
    public func modulatePreMLP(_ x: MLXArray) -> MLXArray {
        (1 + mlpScale) * x + mlpShift
    }

    /// Apply MLP gate
    public func gateMLP(_ x: MLXArray) -> MLXArray {
        mlpGate * x
    }
}

/// Simplified AdaLN for single-stream blocks (3 params instead of 6)
public final class AdaLNSingle: Module {
    public let hiddenSize: Int
    public let conditioningDim: Int

    @ModuleInfo(key: "linear") private var linear: any DenseLayer

    public init(
        hiddenSize: Int,
        conditioningDim: Int? = nil,
        factory: DenseLayerFactory = .standard,
        basePath: String = ""
    ) {
        self.hiddenSize = hiddenSize
        self.conditioningDim = conditioningDim ?? hiddenSize
        // Output 3 * hiddenSize for (shift, scale, gate) - combined attn+mlp
        let path = basePath.isEmpty ? "linear" : "\(basePath).linear"
        self._linear.wrappedValue = factory.makeDenseLayer(
            path: path,
            inputDim: self.conditioningDim,
            outputDim: 3 * hiddenSize,
            bias: true
        )
        super.init()
    }

    public func callAsFunction(_ conditioning: MLXArray) -> AdaLNSingleModulation {
        let mod = linear(MLXNN.silu(conditioning))

        let chunk = hiddenSize
        let shift = mod[0..., 0..<chunk].expandedDimensions(axis: 1)
        let scale = mod[0..., chunk..<(2*chunk)].expandedDimensions(axis: 1)
        let gate = mod[0..., (2*chunk)...].expandedDimensions(axis: 1)

        return AdaLNSingleModulation(shift: shift, scale: scale, gate: gate)
    }
}

public struct AdaLNSingleModulation {
    public let shift: MLXArray
    public let scale: MLXArray
    public let gate: MLXArray

    public func modulate(_ x: MLXArray) -> MLXArray {
        (1 + scale) * x + shift
    }

    public func apply(_ x: MLXArray) -> MLXArray {
        gate * x
    }
}
