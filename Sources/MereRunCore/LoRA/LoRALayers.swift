import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

/// Role of a LoRA adapter in a fused/stacked configuration.
public enum LoRARole: String, Sendable {
    /// The LoRA being actively trained.
    case train
    /// A pre-trained "assistant" LoRA that provides a foundation for the training LoRA.
    case assistant
}

public protocol TrainableLoRALayer: AnyObject {
    var loraRank: Int { get }
    var loraAlpha: Float { get }
    var loraDown: MLXArray { get set }
    var loraUp: MLXArray { get set }
    var isActive: Bool { get set }

    /// Role of this LoRA adapter (train or assistant). Default is train.
    var role: LoRARole { get set }

    // Optimizer state (Adam m and v) for resumable training
    var loraDownM: MLXArray? { get set }
    var loraDownV: MLXArray? { get set }
    var loraUpM: MLXArray? { get set }
    var loraUpV: MLXArray? { get set }
}

/// LoRA wrapper for a standard (non-quantized) Linear.
///
/// Forward: y = xWᵀ + scale * (xAᵀ)Bᵀ
public final class LoRALinear: Linear, TrainableLoRALayer {
    public let loraRank: Int
    public let loraAlpha: Float

    public var loraDown: MLXArray
    public var loraUp: MLXArray
    public var isActive: Bool = true
    public var role: LoRARole = .train

    // Optimizer state for resumable training
    public var loraDownM: MLXArray?
    public var loraDownV: MLXArray?
    public var loraUpM: MLXArray?
    public var loraUpV: MLXArray?

    private let scale: Float

    public init(base: Linear, rank: Int, alpha: Float? = nil, zeroInitUp: Bool = false) {
        let (outDim, inDim) = base.shape
        self.loraRank = max(1, rank)
        self.loraAlpha = alpha ?? Float(self.loraRank)
        self.scale = self.loraAlpha / Float(self.loraRank)

        // Init loraDown with Kaiming-uniform (standard LoRA)
        let bound = 1.0 / sqrt(Float(inDim))
        self.loraDown = MLXRandom.uniform(low: -bound, high: bound, [self.loraRank, inDim]).asType(.float32)

        if zeroInitUp {
            // True zero init (same as ai-toolkit/kohya)
            self.loraUp = MLXArray.zeros([outDim, self.loraRank], dtype: .float32)
        } else {
            let initScale = 1.0 / sqrt(Float(self.loraRank))
            self.loraUp = MLXRandom.normal([outDim, self.loraRank]).asType(.float32) * (0.01 * initScale)
        }

        super.init(weight: base.weight, bias: base.bias)
    }

    nonisolated(unsafe) private static var hasLoggedOnce = false

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let baseOut = super.callAsFunction(x)
        guard isActive else { return baseOut }

        // Compute LoRA in fp32 for stability, then cast back.
        let xF = x.asType(.float32)
        let down = loraDown
        let up = loraUp
        let loraOut = MLX.matmul(MLX.matmul(xF, down.T), up.T) * MLXArray(scale)

        // Debug: log layers with non-zero weights (those that were actually trained)
        if ProcessInfo.processInfo.environment["MERERUN_LORA_DEBUG"] != nil {
            let downNorm = MLX.sqrt(MLX.sum(down * down)).item(Float.self)
            let upNorm = MLX.sqrt(MLX.sum(up * up)).item(Float.self)
            if !Self.hasLoggedOnce && upNorm > 0.1 {
                Self.hasLoggedOnce = true
                let baseNorm = MLX.sqrt(MLX.sum(baseOut * baseOut)).item(Float.self)
                let loraNorm = MLX.sqrt(MLX.sum(loraOut * loraOut)).item(Float.self)
                FileHandle.standardError.write(Data("[LoRA Debug] Forward (trained layer): base_norm=\(baseNorm), lora_norm=\(loraNorm), down_norm=\(downNorm), up_norm=\(upNorm), scale=\(scale)\n".utf8))
            }
        }

        return baseOut + loraOut.asType(baseOut.dtype)
    }
}

/// LoRA wrapper for a QuantizedLinear-like layer, but implemented as a Linear subclass so
/// we can keep LoRA trainable (QuantizedLinear keeps itself frozen).
///
/// Forward: y = quantizedMM(x, Wq) + bias + scale * (xAᵀ)Bᵀ
public final class LoRAQuantizedLinear: Linear, TrainableLoRALayer {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    public let scales: MLXArray
    public let biases: MLXArray?
    public let residualDown: MLXArray?
    public let residualUp: MLXArray?
    private let denseBaseFallback: Bool

    public let loraRank: Int
    public let loraAlpha: Float
    public var loraDown: MLXArray
    public var loraUp: MLXArray
    public var isActive: Bool = true
    public var role: LoRARole = .train

    // Optimizer state for resumable training
    public var loraDownM: MLXArray?
    public var loraDownV: MLXArray?
    public var loraUpM: MLXArray?
    public var loraUpV: MLXArray?

    private let scale: Float

    public override var shape: (Int, Int) {
        let shape = weight.shape2
        return (shape.0, shape.1 * 32 / bits)
    }

    public init(base: QuantizedLinear, rank: Int, alpha: Float? = nil, zeroInitUp: Bool = false) {
        self.groupSize = base.groupSize
        self.bits = base.bits
        self.mode = base.mode
        self.scales = base.scales
        self.biases = base.biases
        if let residual = base as? ResidualQuantizedLinear {
            self.residualDown = residual.residualDown
            self.residualUp = residual.residualUp
        } else {
            self.residualDown = nil
            self.residualUp = nil
        }
        self.denseBaseFallback = base is PortableQuantizedLinear

        let (outDim, inDim) = base.shape
        self.loraRank = max(1, rank)
        self.loraAlpha = alpha ?? Float(self.loraRank)
        self.scale = self.loraAlpha / Float(self.loraRank)

        // Init loraDown with Kaiming-uniform (standard LoRA)
        let bound = 1.0 / sqrt(Float(inDim))
        self.loraDown = MLXRandom.uniform(low: -bound, high: bound, [self.loraRank, inDim]).asType(.float32)

        if zeroInitUp {
            // True zero init (same as ai-toolkit/kohya)
            self.loraUp = MLXArray.zeros([outDim, self.loraRank], dtype: .float32)
        } else {
            let initScale = 1.0 / sqrt(Float(self.loraRank))
            self.loraUp = MLXRandom.normal([outDim, self.loraRank]).asType(.float32) * (0.01 * initScale)
        }

        super.init(weight: base.weight, bias: base.bias)
    }

    public init(base: LoRAQuantizedLinear, rank: Int, alpha: Float? = nil, zeroInitUp: Bool = false) {
        self.groupSize = base.groupSize
        self.bits = base.bits
        self.mode = base.mode
        self.scales = base.scales
        self.biases = base.biases
        self.residualDown = base.residualDown
        self.residualUp = base.residualUp
        self.denseBaseFallback = base.denseBaseFallback

        let (outDim, inDim) = base.shape
        self.loraRank = max(1, rank)
        self.loraAlpha = alpha ?? Float(self.loraRank)
        self.scale = self.loraAlpha / Float(self.loraRank)

        // Init loraDown with Kaiming-uniform (standard LoRA)
        let bound = 1.0 / sqrt(Float(inDim))
        self.loraDown = MLXRandom.uniform(low: -bound, high: bound, [self.loraRank, inDim]).asType(.float32)

        if zeroInitUp {
            self.loraUp = MLXArray.zeros([outDim, self.loraRank], dtype: .float32)
        } else {
            let initScale = 1.0 / sqrt(Float(self.loraRank))
            self.loraUp = MLXRandom.normal([outDim, self.loraRank]).asType(.float32) * (0.01 * initScale)
        }

        super.init(weight: base.weight, bias: base.bias)
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out: MLXArray
        if denseBaseFallback {
            let denseWeight = MLX.dequantized(
                weight,
                scales: scales,
                biases: biases,
                groupSize: groupSize,
                bits: bits,
                mode: mode,
                dtype: x.dtype
            )
            out = MLX.matmul(x, denseWeight.T)
        } else {
            out = quantizedMM(
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
        if let bias {
            out = out + bias
        }

        if let residualDown, let residualUp {
            let xF = x.asType(.float32)
            let residual = MLX.matmul(MLX.matmul(xF, residualDown.T), residualUp.T)
            out = out + residual.asType(out.dtype)
        }

        guard isActive else { return out }

        let xF = x.asType(.float32)
        let loraOut = MLX.matmul(MLX.matmul(xF, loraDown.T), loraUp.T) * MLXArray(scale)
        return out + loraOut.asType(out.dtype)
    }
}

// MARK: - FusedLoRALinear

/// A Linear layer that supports multiple stacked LoRA adapters.
/// Used for training a new LoRA on top of an existing "assistant" LoRA.
///
/// Forward: y = xWᵀ + Σ(scale_i * (xA_iᵀ)B_iᵀ) for each active LoRA
public final class FusedLoRALinear: Linear, TrainableLoRALayer {
    /// The base Linear layer (may be quantized internally via QuantizedLinear reference).
    private let baseLinear: Linear

    /// All LoRA adapters stacked on this layer.
    public var loras: [TrainableLoRALayer]

    /// The primary (trainable) LoRA - convenience accessor for the first `train` role LoRA.
    public var trainLoRA: TrainableLoRALayer? {
        loras.first { $0.role == .train }
    }

    /// Assistant LoRAs - all LoRAs with `assistant` role.
    public var assistantLoRAs: [TrainableLoRALayer] {
        loras.filter { $0.role == .assistant }
    }

    // TrainableLoRALayer conformance - delegates to the primary trainable LoRA
    public var loraRank: Int {
        trainLoRA?.loraRank ?? loras.first?.loraRank ?? 0
    }

    public var loraAlpha: Float {
        trainLoRA?.loraAlpha ?? loras.first?.loraAlpha ?? 0
    }

    public var loraDown: MLXArray {
        get { trainLoRA?.loraDown ?? MLXArray.zeros([1, 1]) }
        set { trainLoRA?.loraDown = newValue }
    }

    public var loraUp: MLXArray {
        get { trainLoRA?.loraUp ?? MLXArray.zeros([1, 1]) }
        set { trainLoRA?.loraUp = newValue }
    }

    public var isActive: Bool {
        get { trainLoRA?.isActive ?? true }
        set { trainLoRA?.isActive = newValue }
    }

    public var role: LoRARole {
        get { trainLoRA?.role ?? .train }
        set { trainLoRA?.role = newValue }
    }

    public var loraDownM: MLXArray? {
        get { trainLoRA?.loraDownM }
        set { trainLoRA?.loraDownM = newValue }
    }

    public var loraDownV: MLXArray? {
        get { trainLoRA?.loraDownV }
        set { trainLoRA?.loraDownV = newValue }
    }

    public var loraUpM: MLXArray? {
        get { trainLoRA?.loraUpM }
        set { trainLoRA?.loraUpM = newValue }
    }

    public var loraUpV: MLXArray? {
        get { trainLoRA?.loraUpV }
        set { trainLoRA?.loraUpV = newValue }
    }

    /// Whether assistant LoRAs are currently disabled (for preview without assistant influence).
    public var assistantsDisabled: Bool = false

    public init(base: Linear, loras: [TrainableLoRALayer]) {
        self.baseLinear = base
        self.loras = loras
        super.init(weight: base.weight, bias: base.bias)
    }

    /// Create a FusedLoRALinear from an existing LoRALinear (converts it to assistant and adds a new train LoRA).
    public convenience init(
        existingLoRA: LoRALinear,
        newRank: Int,
        newAlpha: Float? = nil,
        zeroInitUp: Bool = true
    ) {
        // Mark existing as assistant
        existingLoRA.role = .assistant

        // Create new trainable LoRA
        let newLoRA = LoRALinear(
            base: Linear(weight: existingLoRA.weight, bias: existingLoRA.bias),
            rank: newRank,
            alpha: newAlpha,
            zeroInitUp: zeroInitUp
        )
        newLoRA.role = .train

        self.init(base: existingLoRA, loras: [existingLoRA, newLoRA])
    }

    /// Create a FusedLoRALinear from an existing LoRAQuantizedLinear.
    public convenience init(
        existingLoRA: LoRAQuantizedLinear,
        newRank: Int,
        newAlpha: Float? = nil,
        zeroInitUp: Bool = true
    ) {
        // Mark existing as assistant
        existingLoRA.role = .assistant

        // Create new trainable LoRA on top
        let newLoRA = LoRAQuantizedLinear(
            base: existingLoRA,
            rank: newRank,
            alpha: newAlpha,
            zeroInitUp: zeroInitUp
        )
        newLoRA.role = .train

        self.init(base: existingLoRA, loras: [existingLoRA, newLoRA])
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Get base output
        var out = baseLinear(x)

        // Add contribution from each active LoRA
        let xF = x.asType(.float32)
        for lora in loras {
            guard lora.isActive else { continue }
            if assistantsDisabled && lora.role == .assistant { continue }

            let scale = lora.loraAlpha / Float(lora.loraRank)
            let loraOut = MLX.matmul(MLX.matmul(xF, lora.loraDown.T), lora.loraUp.T) * MLXArray(scale)
            out = out + loraOut.asType(out.dtype)
        }

        return out
    }

    /// Temporarily disable assistant LoRAs for preview/evaluation.
    public func withAssistantsDisabled<T>(_ body: () throws -> T) rethrows -> T {
        let wasDisabled = assistantsDisabled
        assistantsDisabled = true
        defer { assistantsDisabled = wasDisabled }
        return try body()
    }
}
